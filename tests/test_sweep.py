"""Mode-agnostic tests: the same assertions run in serial and parallel.

The ``num_*`` and ``integral`` fields are totals over all ranks, so they do not
depend on how many ranks the suite is launched with.
"""

import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import medea
from medea._backend import mfem


def _build_forms(fespace, velocity):
    sigma = mfem.ConstantCoefficient(0.5)
    inflow = mfem.ConstantCoefficient(1.0)
    alpha = 1.0
    neg_alpha = -1.0

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

    b = mfem.LinearForm(fespace)
    b.AddBdrFaceIntegrator(
        mfem.BoundaryFlowIntegrator(
            inflow, mfem.VectorConstantCoefficient(velocity), neg_alpha
        )
    )

    b.Assemble()
    a.Assemble()
    a.Finalize()

    return a, b


def _global_solve(fespace, a, b):
    """Reference solution of ``a x = b`` using a sparse direct solve."""
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)

    A = mfem.OperatorPtr()
    B = mfem.Vector()
    X = mfem.Vector()

    ess_tdof_list = mfem.intArray()
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)

    # PyMFEM is not built with UMFPACK, so factor the CSR matrix with SciPy
    AA = mfem.OperatorHandle2SparseMatrix(A)
    csr = sp.csr_matrix(
        (AA.GetDataArray(), AA.GetJArray(), AA.GetIArray()),
        shape=(AA.Height(), AA.Width()),
    )
    X = mfem.Vector(spla.spsolve(csr.tocsc(), B.GetDataArray()))
    a.RecoverFEMSolution(X, b, x)

    return X


# Per-dimension meshes and angles. Each has >= 3 angles (some tests index angle 2),
# and angle 2 is axis-aligned where that is meaningful.
CASES = {
    "1d": (
        lambda: mfem.Mesh(8, 1.0),
        [[1.0], [-1.0], [0.5]],
    ),
    "2d": (
        lambda: mfem.Mesh(4, 4, "TRIANGLE"),
        [
            [0.25, 0.866025404],  # generic direction
            [-0.866025404, 0.25],  # opposite x half-plane
            [1.0, 0.0],  # axis-aligned: some faces are parallel to the sweep
        ],
    ),
    "3d": (
        lambda: mfem.Mesh(3, 3, 3, "TETRAHEDRON"),
        [
            [0.3, 0.5, 0.812403840],  # generic direction
            [-0.6, 0.48, -0.64],  # mixed signs
            [1.0, 0.0, 0.0],  # axis-aligned
        ],
    ),
}

all_dims = pytest.mark.parametrize("dim", list(CASES))
order012 = pytest.mark.parametrize("dg_order", [0, 1, 2])


def _setup(dim="2d", dg_order=1):
    make_mesh, angles = CASES[dim]
    mesh = make_mesh()
    velocities = [mfem.Vector(v) for v in angles]

    fec = mfem.DG_FECollection(dg_order, mesh.Dimension(), mfem.BasisType.GaussLobatto)
    fespace = mfem.FiniteElementSpace(mesh, fec)

    forms = [_build_forms(fespace, v) for v in velocities]
    a_list = [a for a, _ in forms]
    b_list = [b for _, b in forms]
    return mesh, velocities, fespace, a_list, b_list


def _assert_matches_global_solve(fespace, a_list, b_list, indices, solutions):
    assert len(solutions) == len(indices)
    for k, sweep_soln in zip(indices, solutions):
        X = _global_solve(fespace, a_list[k], b_list[k])
        diff = mfem.Vector(X.Size())
        mfem.subtract_vector(X, sweep_soln, diff)
        assert diff.Normlinf() < 1e-8, f"angle {k} differs from global solve"


@all_dims
@order012
def test_sweep_equivalent_to_global_solve(dim, dg_order):
    """Default sweep covers every angle, in the order they were given."""
    mesh, velocities, fespace, a_list, b_list = _setup(dim, dg_order)

    ordinate = medea.sweep.TransportSweep(mesh, fespace, velocities)
    solutions = ordinate.sweep(a_list, b_list)

    _assert_matches_global_solve(
        fespace, a_list, b_list, range(len(velocities)), solutions
    )


@all_dims
def test_sweep_angle_subset(dim):
    """``sweep(..., range)`` solves only the angles indexed by the range."""
    mesh, velocities, fespace, a_list, b_list = _setup(dim)

    ordinate = medea.sweep.TransportSweep(mesh, fespace, velocities)

    subset = range(1, len(velocities))
    solutions = ordinate.sweep(
        a_list[slice(subset.start, subset.stop, subset.step)],
        b_list[slice(subset.start, subset.stop, subset.step)],
        angle_range=subset,
    )
    _assert_matches_global_solve(fespace, a_list, b_list, subset, solutions)

    m = 2
    single = range(m, m + 1)
    solutions = ordinate.sweep(
        a_list[slice(single.start, single.stop, single.step)],
        b_list[slice(single.start, single.stop, single.step)],
        angle_range=m,
    )
    _assert_matches_global_solve(fespace, a_list, b_list, single, solutions)


def test_sweep_range_out_of_bounds():
    mesh, velocities, fespace, a_list, b_list = _setup()
    ordinate = medea.sweep.TransportSweep(mesh, fespace, velocities)

    with pytest.raises(IndexError):
        ordinate.sweep(a_list, b_list, angle_range=range(len(velocities) + 1))


@all_dims
def test_sweep_out_written_in_place(dim):
    """Passing ``out`` fills those exact vectors and returns the same list."""
    mesh, velocities, fespace, a_list, b_list = _setup(dim)
    ordinate = medea.sweep.TransportSweep(mesh, fespace, velocities)

    ndofs = fespace.GetNDofs()
    out = [mfem.Vector(ndofs) for _ in velocities]
    for v in out:
        v.Assign(123.0)  # stale data must be overwritten
    ids = [id(v) for v in out]

    result = ordinate.sweep(a_list, b_list, out)

    assert result is out
    assert [id(v) for v in result] == ids
    _assert_matches_global_solve(fespace, a_list, b_list, range(len(velocities)), out)

    # reusing the same buffers a second time still gives the right answer
    ordinate.sweep(a_list, b_list, out)
    _assert_matches_global_solve(fespace, a_list, b_list, range(len(velocities)), out)


@all_dims
def test_sweep_out_with_angle_subset(dim):
    mesh, velocities, fespace, a_list, b_list = _setup(dim)
    ordinate = medea.sweep.TransportSweep(mesh, fespace, velocities)

    out = [mfem.Vector(fespace.GetNDofs())]
    ordinate.sweep([a_list[2]], [b_list[2]], out, angle_range=2)
    _assert_matches_global_solve(fespace, a_list, b_list, [2], out)


def test_sweep_out_wrong_length():
    mesh, velocities, fespace, a_list, b_list = _setup()
    ordinate = medea.sweep.TransportSweep(mesh, fespace, velocities)

    out = [mfem.Vector(fespace.GetNDofs()) for _ in range(len(velocities) - 1)]
    with pytest.raises(ValueError):
        ordinate.sweep(a_list, b_list, out)


def test_sweep_out_wrong_size():
    mesh, velocities, fespace, a_list, b_list = _setup()
    ordinate = medea.sweep.TransportSweep(mesh, fespace, velocities)

    out = [mfem.Vector(fespace.GetNDofs()) for _ in velocities]
    out[1] = mfem.Vector(fespace.GetNDofs() + 1)
    with pytest.raises(ValueError, match="Output vector 1"):
        ordinate.sweep(a_list, b_list, out)


def test_sweep_out_is_keyword_or_third_positional_only():
    """``angle_range`` is keyword-only."""
    mesh, velocities, fespace, a_list, b_list = _setup()
    ordinate = medea.sweep.TransportSweep(mesh, fespace, velocities)

    with pytest.raises(TypeError):
        ordinate.sweep(a_list, b_list, None, range(len(velocities)))


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
    fec = mfem.DG_FECollection(1, mesh.Dimension(), mfem.BasisType.GaussLobatto)
    fespace = mfem.FiniteElementSpace(mesh, fec)

    velocity = mfem.Vector([0.25, 0.866025404])
    with pytest.warns(UserWarning, match="Curved meshes not verified"):
        medea.sweep.TransportSweep(mesh, fespace, [velocity])


def test_angle_wrong_dim():
    mesh = mfem.Mesh(4, 4, "TRIANGLE")
    vec1d = mfem.Vector([1.0])
    vec3d = mfem.Vector([1.0, 0.0, 0.0])

    fec = mfem.DG_FECollection(1, mesh.Dimension(), mfem.BasisType.GaussLobatto)
    fespace = mfem.FiniteElementSpace(mesh, fec)

    with pytest.raises(
        ValueError, match="Invalid dimensions in 'angle' for mesh of dimension 2"
    ):
        medea.sweep.TransportSweep(mesh, fespace, [vec1d])
    with pytest.raises(
        ValueError, match="Invalid dimensions in 'angle' for mesh of dimension 2"
    ):
        medea.sweep.TransportSweep(mesh, fespace, [vec3d])
