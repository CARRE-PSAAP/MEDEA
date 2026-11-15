"""
Example problem:

Cubesat with sensitive volumes (silicon dies) inside an aluminum shell.
Run in 1D (vertical line), 2D (slice) or 3D with --dim; the meshes come from
cubesat.py (examples/data/cubesat_{1,2,3}d.msh). Incoming flux on the
x = 0 side of the domain (the bottom of the 1D line).

    python examples/source_iteration.py --dim 2

Source iteration algorithm with 1e-6 relative global convergence tolerance

"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np

import medea
from medea._backend import mfem

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dim", type=int, choices=[1, 2, 3], default=2)
parser.add_argument("--mesh", help="mesh file (default: examples/data/cubesat_<dim>d.msh)")
args = parser.parse_args()
dim = args.dim

start = time.perf_counter()


current_dir = Path(__file__).resolve().parent

sigma_t_vac = 0.0
sigma_s_vac = 0.0

sigma_t_al = 1.0
sigma_s_al = 0.9

sigma_t_si = 11.0
sigma_s_si = 10.0

sigma_t_fr = 11.0
sigma_s_fr = 10.0

sigma_t_ta = 11.0
sigma_s_ta = 10.0

sigma_t_bat = 11.0
sigma_s_bat = 10.0

sigma_t_steel = 11.0
sigma_s_steel = 10.0


sigma_svec = mfem.Vector([sigma_s_vac, sigma_s_al, sigma_s_si, sigma_s_fr, sigma_s_ta, sigma_s_bat, sigma_s_steel])

sigma_total = mfem.PWConstCoefficient(
    mfem.Vector([sigma_t_vac, sigma_t_al, sigma_t_si, sigma_t_fr, sigma_t_ta, sigma_t_bat, sigma_t_steel])
)


mesh = mfem.Mesh(args.mesh or f"{current_dir}/data/cubesat_{dim}d.msh")


# ---- angular quadrature: quad[dim - 1] = (angles, weights), weights sum to 1
def reflect(first_orthant, w):
    """Reflect first-orthant direction cosines (n, d) into all 2**d orthants."""
    first_orthant = np.asarray(first_orthant)
    signs = np.array(list(itertools.product((1, -1), repeat=first_orthant.shape[1])))
    ang = np.concatenate([first_orthant * s for s in signs])
    weights = np.tile(w, len(signs))
    return [mfem.Vector([float(v) for v in a]) for a in ang], weights / weights.sum()


# 1D: S4 double Gauss-Legendre (4 Gauss points on each of mu in [-1,0], [0,1])
x, w = np.polynomial.legendre.leggauss(4)
quad_1d = reflect(((x + 1) / 2)[:, None], w)

# 2D/3D: S6 level-symmetric, one octant (mu, eta); xi is fixed by mu^2+eta^2+xi^2 = 1
mu_eta = np.array(
    [
        [0.2666355, 0.2666355],
        [0.2666355, 0.6815076],
        [0.2666355, 0.9261808],
        [0.6815076, 0.2666355],
        [0.6815076, 0.6815076],
        [0.9261808, 0.2666355],
    ]
)
xi = np.sqrt(1.0 - (mu_eta**2).sum(axis=1))
weights = [0.1761263, 0.1572071, 0.1761263, 0.1572071, 0.1572071, 0.1761263]
quad_2d = reflect(mu_eta, weights)  # 24 angles
quad_3d = reflect(np.column_stack([mu_eta, xi]), weights)  # 48 angles

quad = [quad_1d, quad_2d, quad_3d]
angles, angle_weights = quad[dim - 1]


def scalar_flux(angular_flux, quad_weights=angle_weights) -> mfem.Vector:
    """Scalar flux phi = sum_k w_k * psi_k from a list of per-angle angular flux vectors."""
    if len(angular_flux) != len(quad_weights):
        raise ValueError("Number of angular flux vectors must match number of weights")
    phi = np.zeros(angular_flux[0].Size())
    for w, psi in zip(quad_weights, angular_flux):
        phi += w * psi.GetDataArray()
    return mfem.Vector(phi)


class TransportSolver:
    def __init__(
        self, sigma_t: mfem.PWConstCoefficient, sigma_s: mfem.PWConstCoefficient
    ):
        self.sigma_s = sigma_s
        # create bilinear forms for k in angles

        fec = mfem.DG_FECollection(1, mesh.Dimension(), mfem.BasisType.GaussLobatto)
        self.fespace = mfem.FiniteElementSpace(mesh, fec)
        self.A = [mfem.BilinearForm(self.fespace) for _ in angles]

        self.transport_operator = medea.sweep.TransportSweep(mesh, self.fespace, angles)

        # Preallocated angular flux, one vector per angle; overwritten in place by every sweep
        self.psi = [mfem.Vector(self.fespace.GetNDofs()) for _ in range(len(angles))]

        # Scattering source Q = sigma_s * phi is built once. phi_gf is updated in place
        # each iteration, so the coefficient (and the linear forms) always see the latest phi.
        self.phi_gf = mfem.GridFunction(self.fespace)
        self.phi_gf.Assign(0.0)
        self.source_coeff = mfem.ProductCoefficient(
            self.sigma_s, mfem.GridFunctionCoefficient(self.phi_gf)
        )
        self.b = self._build_linear_forms()

        alpha = 1.0

        for k in range(len(angles)):

            velocity = angles[k]
            a = self.A[k]

            a.AddDomainIntegrator(
                mfem.ConvectionIntegrator(
                    mfem.VectorConstantCoefficient(velocity), alpha
                )
            )
            a.AddInteriorFaceIntegrator(
                mfem.NonconservativeDGTraceIntegrator(
                    mfem.VectorConstantCoefficient(velocity), alpha
                )
            )
            a.AddBdrFaceIntegrator(
                mfem.NonconservativeDGTraceIntegrator(
                    mfem.VectorConstantCoefficient(velocity), alpha
                )
            )
            a.AddDomainIntegrator(mfem.MassIntegrator(sigma_t))
            a.Assemble()

    def _build_linear_forms(self) -> list[mfem.LinearForm]:
        """One linear form per angle (inflow boundary term + scattering source)."""
        b_list = []
        neg_alpha = -1.0
        for k in range(len(angles)):
            inflow = mfem.ConstantCoefficient(100.0 if angles[k][0] > 0 else 0.0)
            b = mfem.LinearForm(self.fespace)
            b.AddBdrFaceIntegrator(
                mfem.BoundaryFlowIntegrator(
                    inflow, mfem.VectorConstantCoefficient(angles[k]), neg_alpha
                )
            )
            b.AddDomainIntegrator(mfem.DomainLFIntegrator(self.source_coeff))
            b_list.append(b)
        return b_list

    def sweep(self, phi: mfem.Vector) -> list[mfem.Vector]:
        """Sweep all angles given scalar flux phi (same DG dof layout as self.fespace).

        Scattering source is isotropic: Q = sigma_s * phi. The angular flux is written
        in place into self.psi (the same vectors are returned every call).
        """
        self.phi_gf.Assign(phi)  # in place: source_coeff sees the new phi
        for b in self.b:
            b.Assemble()  # re-assembles (zeroes, then fills) the existing linear form
        return self.transport_operator.sweep(self.A, self.b, out=self.psi)


sigma_scatter = mfem.PWConstCoefficient(sigma_svec)
solver = TransportSolver(sigma_total, sigma_scatter)

phi = mfem.Vector(solver.fespace.GetNDofs())
diff = mfem.Vector(solver.fespace.GetNDofs())
phi.Assign(0.0)

error = 1

iter = 0
while error > 1e-4:
    iter += 1
    phi_old = mfem.Vector(phi.GetDataArray())
    psi = solver.sweep(phi)
    phi = scalar_flux(psi)

    mfem.subtract_vector(phi, phi_old, diff)
    error = diff.Normlinf() / phi.Normlinf()
    print(f"iteration {iter}, error {error}")


end = time.perf_counter()
elapsed = end - start

print("\n---------\n")
print(f"{mesh.GetNE()} cells")
print(
    f"Elapsed time: {elapsed:.2f} seconds\n{(elapsed/(iter * len(psi))):.2e}s per sweep"
)

gf = mfem.GridFunction(solver.fespace)
gf.Assign(phi)

dc = mfem.ParaViewDataCollection("source_iteration", mesh)
dc.SetLevelsOfDetail(2)
dc.SetHighOrderOutput(True)
dc.RegisterField("phi", gf)
gfs = []
for k, psi_k in enumerate(psi):
    g = mfem.GridFunction()
    g.MakeRef(solver.fespace, psi_k, 0)
    dc.RegisterField(f"psi_{k}", g)
    gfs.append(g)
dc.Save()
dc.SetCycle(0)
dc.SetTime(0.0)
