"""Mode-agnostic tests: the same assertions run in serial and parallel.

The ``num_*`` and ``integral`` fields are totals over all ranks, so they do not
depend on how many ranks the suite is launched with.
"""

import pytest

import medea
from medea._backend import mfem

from pathlib import Path
current_dir = Path(__file__).resolve().parent


def test_sweep_equivalent_to_global_solve():
    mesh = mfem.Mesh(f"{current_dir}/data/small_tri_disk.msh")
    velocity = mfem.Vector([0.25, 0.866025404])

    ordinate = medea.sweep.Schedule(mesh, velocity)

    sigma = mfem.ConstantCoefficient(0.5)
    inflow = mfem.ConstantCoefficient(1.0)

    fec = mfem.DG_FECollection(1, mesh.Dimension(), mfem.BasisType.GaussLobatto)
    fespace = mfem.FiniteElementSpace(mesh, fec)

    alpha = 1.0

    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(
        mfem.ConvectionIntegrator(mfem.VectorConstantCoefficient(velocity), alpha)
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
    a.AddDomainIntegrator(mfem.MassIntegrator(sigma))

    neg_alpha = -1.0

    b = mfem.LinearForm(fespace)
    b.AddBdrFaceIntegrator(
        mfem.BoundaryFlowIntegrator(
            inflow, mfem.VectorConstantCoefficient(velocity), neg_alpha
        )
    )

    b.Assemble()
    a.Assemble()
    a.Finalize()

    sweep_soln = ordinate.sweep(a, b)

    x = mfem.GridFunction(fespace)
    x.Assign(0.0)

    A = mfem.OperatorPtr()
    B = mfem.Vector()
    X = mfem.Vector()

    ess_tdof_list = mfem.intArray()
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)

    AA = mfem.OperatorHandle2SparseMatrix(A)
    M = mfem.GSSmoother(AA)
    mfem.GMRES(AA, M, B, X, 0, 500, 200, 0, 1e-16)
    a.RecoverFEMSolution(X, b, x)

    diff = mfem.Vector(X.Size())
    mfem.subtract_vector(X, sweep_soln, diff)
    
    assert(diff.Normlinf() < 1e-8)



def _make_curved_mesh() -> mfem.Mesh:
    # 1. Create a minimal 1-element 2D triangle mesh (vertices at (0,0), (1,0), (0,1))
    mesh = mfem.Mesh(2, 3, 1)  # dim=2, num_vertices=3, num_elements=1
    mesh.AddVertex([0.0, 0.0])
    mesh.AddVertex([1.0, 0.0])
    mesh.AddVertex([0.0, 1.0])
    mesh.AddTriangle([0, 1, 2])
    mesh.FinalizeTriMesh(1, 1, True)

    # 2. Promote the mesh geometry to quadratic (order = 2)
    mesh.SetCurvature(2)

    # 3. Perturb a midpoint node so the face is genuinely non-straight
    nodes = mesh.GetNodes()
    # Grab a NumPy view of the node coordinates and shift the first midpoint slightly
    coords = nodes.GetDataArray()
    coords[3] += 0.1  # Shifts an edge midpoint along x

    return mesh

def test_reject_curved_mesh():
    mesh = _make_curved_mesh()
    velocity = mfem.Vector([0.25, 0.866025404])
    with pytest.warns(UserWarning, match="Curved meshes not verified"):
        ordinate = medea.sweep.Schedule(mesh, velocity)

def test_angle_wrong_dim():
    mesh = mfem.Mesh(4, 4, "TRIANGLE")
    vec1d = mfem.Vector([1.0])
    vec3d = mfem.Vector([1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Invalid dimensions in 'angle' for mesh of dimension 2"):
        medea.sweep.Schedule(mesh, vec1d)
    with pytest.raises(ValueError, match="Invalid dimensions in 'angle' for mesh of dimension 2"):
        medea.sweep.Schedule(mesh, vec3d)