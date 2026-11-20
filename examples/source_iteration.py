"""
Example: shielding of CubeSat electronics, solved using source iteration
and LD finite element discretization.

Solves monoenergetic transport equation in the case of isotropic scattering in a
source-free medium with incoming flux boundary condiitons.

    Omega . grad(psi) + sigma_t psi = sigma_s phi + (incident flux on the boundary)

on a 2D slice through the CubeSat (cubesat.py): an aluminum shell around two circuit
boards, silicon dies (the sensitive volumes), a battery and a reaction wheel.

    python examples/source_iteration.py          # quadrilateral mesh (non-conforming)
    python examples/source_iteration.py --tet    # triangle mesh

"""

import argparse
import time
from pathlib import Path

import numpy as np

from medea import Material, MaterialGroup, TransportSweep, mfem

DATA = Path(__file__).resolve().parent / "data"

# Cross sections [1/cm]. Attributes default to 1, 2, ... in this order, matching the
# material ids that cubesat.py writes to the mesh.
MATERIALS = MaterialGroup(
    Material("vacuum", xs_total = 0.0, xs_scatter = 0.0),
    Material("aluminum", xs_total = 1.0, xs_scatter = 0.9),
    Material("silicon", xs_total = 11.0, xs_scatter = 10.0),
    Material("fr4", xs_total = 11.0, xs_scatter = 10.0),
    Material("tantalum", xs_total = 11.0, xs_scatter = 10.0),
    Material("battery", xs_total = 11.0, xs_scatter = 10.0),
    Material("steel", xs_total =11.0,  xs_scatter = 10.0),
)

INCIDENT_FLUX = 100.0  # psi(x, \Omega) for \Omega_x > 0
TOLERANCE = 1e-4  # scalar flux relative convergence tolerance
MAX_ITERATIONS = 500


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tet", action="store_true", help="use the triangle mesh")
    parser.add_argument("--order", type=int, default=1, help="DFEM order")

    mesh_file = "cubesat_2d.msh" if parser.parse_args().tet else "cubesat_2d_hex.mesh"
    DG_ORDER = parser.parse_args().order
    if DG_ORDER < 0:
        raise ValueError("--order must be a non-negative integer")

    if Path(DATA / mesh_file).exists() == False:
        raise ValueError(
            f"Mesh file does not exist at {Path(DATA / mesh_file)}. Generate using `python cubesat.py`"
        )

    start = time.perf_counter()

    mesh = mfem.Mesh(str(DATA / mesh_file))
    MATERIALS.check_mesh(mesh)  # every mesh attribute must have a material

    angles, weights = s6_quadrature()

    fec = mfem.DG_FECollection(DG_ORDER, 2, mfem.BasisType.GaussLobatto)
    space = mfem.FiniteElementSpace(mesh, fec)  # vector of unknowns

    # Set up constant values: LHS, isotropic scattering operator, BC
    transport = streaming_collision_operators(
        space, angles, MATERIALS
    )  # Omega.nabla + sigma_t
    incident = incident_flux_sources(space, angles)  # boundary inflow, per direction
    scattering = scattering_operator(space, MATERIALS)  # phi -> sigma_s * phi

    ####### Create sweep cell ordering #######
    sweeper = TransportSweep(mesh, space, angles)  # cell order per direction
    ##########################################

    phi = np.zeros(space.GetNDofs())  # scalar flux
    psi = [mfem.Vector(space.GetNDofs()) for _ in angles]  # angular flux, per direction

    for iteration in range(1, MAX_ITERATIONS + 1):
        q = scattering_source(scattering, phi)
        rhs = [mfem.Vector(inflow + q) for inflow in incident]

        ################## Sweep #################
        sweeper.sweep(transport, rhs, out=psi)
        ##########################################

        new_phi = integrate_over_angle(psi, weights)

        error = np.abs(new_phi - phi).max() / np.abs(new_phi).max()
        phi = new_phi
        print(f"iteration {iteration}, error {error:.3e}")
        if error < TOLERANCE:
            break

    print(
        f"\n{mesh.GetNE()} cells\n{len(angles)} directions\n"
        f"{iteration} iterations\n{time.perf_counter() - start:.2f} s\n"
    )
    write_paraview(mesh, space, phi, psi)
    return phi


# ------------------------------------------------------------------ algorithm steps --


def s6_quadrature():
    """Level-symmetric S6 in 2D: 24 directions (unit cosines mu, eta), weights sum to 1."""
    mu_eta = [  # first quadrant
        (0.2666355, 0.2666355),
        (0.2666355, 0.6815076),
        (0.2666355, 0.9261808),
        (0.6815076, 0.2666355),
        (0.6815076, 0.6815076),
        (0.9261808, 0.2666355),
    ]
    w = [0.1761263, 0.1572071, 0.1761263, 0.1572071, 0.1572071, 0.1761263]
    angles, weights = [], []
    for sx, sy in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:  # reflect into all quadrants
        for (mu, eta), wk in zip(mu_eta, w):
            angles.append(mfem.Vector([sx * mu, sy * eta]))
            weights.append(wk)
    return angles, np.array(weights) / sum(weights)


def streaming_collision_operators(space, angles, materials):
    """For each direction, the matrix of  Omega . grad + sigma_t  (upwind, per cell).

    Streaming inside a cell, upwind coupling across cell faces (and outflow through
    the boundary), plus collisions: sigma_t times the cell mass matrix.
    """
    sigma_t = materials.xs_total[0].to_PWConst()
    operators = []
    for omega in angles:
        L = mfem.BilinearForm(space)
        L.AddDomainIntegrator(
            mfem.ConvectionIntegrator(mfem.VectorConstantCoefficient(omega), 1.0)
        )
        L.AddInteriorFaceIntegrator(
            mfem.NonconservativeDGTraceIntegrator(
                mfem.VectorConstantCoefficient(omega), 1.0
            )
        )
        L.AddBdrFaceIntegrator(
            mfem.NonconservativeDGTraceIntegrator(
                mfem.VectorConstantCoefficient(omega), 1.0
            )
        )
        L.AddDomainIntegrator(mfem.MassIntegrator(sigma_t))
        L.Assemble()
        operators.append(L)
    return operators


def incident_flux_sources(space, angles):
    """For each direction, the source vector from flux entering through the boundary.

    Directions with Omega_x > 0 carry INCIDENT_FLUX wherever they enter the domain;
    all other directions enter with zero flux.
    """
    sources = []
    for omega in angles:
        flux = mfem.ConstantCoefficient(INCIDENT_FLUX if omega[0] > 0 else 0.0)
        b = mfem.LinearForm(space)
        b.AddBdrFaceIntegrator(
            mfem.BoundaryFlowIntegrator(
                flux, mfem.VectorConstantCoefficient(omega), -1.0
            )
        )
        b.Assemble()
        sources.append(b.GetDataArray().copy())
    return sources


def scattering_operator(space, materials):
    """Matrix that turns the scalar flux into the isotropic scattering source,
    q = sigma_s * phi / 4pi, weighted by the finite-element basis."""
    sigma_s = mfem.ProductCoefficient(
        1 / (4 * np.pi), materials.xs_scatter[0,0].to_PWConst()
    )
    S = mfem.BilinearForm(space)
    S.AddDomainIntegrator(mfem.MassIntegrator(sigma_s))
    S.Assemble()
    S.Finalize()
    return S


def scattering_source(scattering, phi):
    """Isotropic scattering source q = sigma_s * phi (same for every direction)."""
    q = mfem.Vector(len(phi))
    scattering.Mult(mfem.Vector(phi), q)
    return q.GetDataArray().copy()


def integrate_over_angle(psi, weights):
    """Scalar flux phi = sum_k w_k psi_k from the angular flux in each direction."""
    return sum(w * psi_k.GetDataArray() for w, psi_k in zip(weights, psi))


def write_paraview(mesh, space, phi, psi):
    """Save the scalar flux and each directional angular flux for ParaView."""
    out = mfem.ParaViewDataCollection("source_iteration", mesh)
    out.SetLevelsOfDetail(2)
    out.SetHighOrderOutput(True)
    phi_field = mfem.GridFunction(space)
    phi_field.Assign(mfem.Vector(phi))
    out.RegisterField("phi", phi_field)
    psi_fields = []
    for k, psi_k in enumerate(psi):
        field = mfem.GridFunction()
        field.MakeRef(space, psi_k, 0)
        out.RegisterField(f"psi_{k}", field)
        psi_fields.append(field)  # keep alive until Save
    out.SetCycle(0)
    out.SetTime(0.0)
    out.Save()


if __name__ == "__main__":
    main()
