"""
Example problem:

Cubesat with sensitive volume (simple)
2D square region; rectangular sensitive volumes in center, surrounded by thin shield
in vacuum. Isotropic incoming flux on one side of domain.

Source iteration algorithm with 1e-6 relative global convergence tolerance

"""

import time
from pathlib import Path

import numpy as np

import medea
from medea._backend import mfem

start = time.perf_counter()


current_dir = Path(__file__).resolve().parent

sigma_t_al = 1.0
sigma_s_al = 0.9

sigma_t_si = 11.0
sigma_s_si = 10.0

sigma_t_vac = 0.0
sigma_s_vac = 0.0

sigma_svec = mfem.Vector([sigma_s_vac, sigma_s_al, sigma_s_si])

sigma_total = mfem.PWConstCoefficient(
    mfem.Vector([sigma_t_vac, sigma_t_al, sigma_t_si])
)


mesh = mfem.Mesh(f"{current_dir}/data/cubesat.msh")



# level-symmetric quadrature
K = 6

ne_angles = [
    mfem.Vector([0.2666355, 0.2666355]),
    mfem.Vector([0.2666355, 0.6815076]),
    mfem.Vector([0.2666355, 0.9261808]),
    mfem.Vector([0.6815076, 0.2666355]),
    mfem.Vector([0.6815076, 0.6815076]),
    mfem.Vector([0.9261808, 0.2666355]),
]

weights = [0.1761263, 0.1572071, 0.1761263, 0.1572071, 0.1572071, 0.1761263]
angles = [mfem.Vector(2) for _ in range(4 * K)]

for k in range(K):
    angles[k] = mfem.Vector([ne_angles[k][0], ne_angles[k][1]])
    angles[k + (1 * K)] = mfem.Vector([-ne_angles[k][0], ne_angles[k][1]])
    angles[k + (2 * K)] = mfem.Vector([ne_angles[k][0], -ne_angles[k][1]])
    angles[k + (3 * K)] = mfem.Vector([-ne_angles[k][0], -ne_angles[k][1]])

angle_weights = np.tile(weights, 4)
angle_weights = angle_weights / angle_weights.sum()


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
        self.schedules = [medea.sweep.Schedule(mesh, angle) for angle in angles]

        fec = mfem.DG_FECollection(1, mesh.Dimension(), mfem.BasisType.GaussLobatto)
        self.fespace = mfem.FiniteElementSpace(mesh, fec)
        self.A = [mfem.BilinearForm(self.fespace) for k in range(4 * K)]

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

    def sweep(self, phi: mfem.Vector):
        """Sweep all angles given scalar flux phi (same DG dof layout as self.fespace).

        Scattering source is isotropic: Q = sigma_s * phi, added to the
        linear form. Returns a list of angular flux vectors.
        """
        solution = []

        phi_gf = mfem.GridFunction(self.fespace)
        phi_gf.Assign(phi)
        phi_coeff = mfem.GridFunctionCoefficient(phi_gf)
        source_coeff = mfem.ProductCoefficient(self.sigma_s, phi_coeff)

        neg_alpha = -1.0
        for k in range(4 * K):
            a = self.A[k]
            if angles[k][0] > 0:

                inflow = mfem.ConstantCoefficient(100.0)
            else:
                inflow = mfem.ConstantCoefficient(0.0)
            velocity = angles[k]
            b = mfem.LinearForm(self.fespace)
            b.AddBdrFaceIntegrator(
                mfem.BoundaryFlowIntegrator(
                    inflow, mfem.VectorConstantCoefficient(velocity), neg_alpha
                )
            )
            b.AddDomainIntegrator(mfem.DomainLFIntegrator(source_coeff))
            b.Assemble()
            solution.append(self.schedules[k].sweep(a, b))
        return solution


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
