"""Each compiled kernel is checked against its pure-Python reference."""

import numpy as np
import pytest

from medea import kernels, k_reference


def make_system(sizes, seed=0, density=0.5):
    """Random block system in DG form.

    Element ``e`` owns ``sizes[e]`` dofs (global numbering scrambled, and unsorted
    inside the element). Its rows hold a dense, diagonally dominant, nonsymmetric
    diagonal block plus random couplings to dofs of elements that come *earlier* in
    ``order``, so the sweep is exact: the result must satisfy ``A @ u = f``.
    """
    rng = np.random.default_rng(seed)
    n_el, n = len(sizes), int(sum(sizes))
    start = np.concatenate([[0], np.cumsum(sizes)])
    perm = rng.permutation(n)
    dofs = [perm[start[e] : start[e + 1]].astype(np.int32) for e in range(n_el)]
    order = rng.permutation(n_el).astype(np.int32)

    A = np.zeros((n, n))
    for k, e in enumerate(order):
        d = dofs[e]
        A[np.ix_(d, d)] = rng.uniform(-1, 1, (d.size, d.size)) + 2 * d.size * np.eye(
            d.size
        )
        for up in order[:k]:
            u = dofs[up]
            keep = rng.random((d.size, u.size)) < density
            A[np.ix_(d, u)] = keep * rng.uniform(-1, 1, (d.size, u.size))

    return {
        "A": A,
        "f": rng.uniform(-1, 1, n),
        "order": order,
        "dofs": dofs,
        "offsets": start.astype(np.int32),
        "indices": np.concatenate(dofs).astype(np.int32),
    }


def to_csr(A):
    rows, cols = np.nonzero(A)
    i = np.zeros(A.shape[0] + 1, dtype=np.int32)
    i[1:] = np.cumsum(np.bincount(rows, minlength=A.shape[0]))
    return i, cols.astype(np.int32), A[rows, cols].astype(np.float64)


def run_kernel(sysd, out, order=None):
    i, j, d = to_csr(sysd["A"])
    order = sysd["order"] if order is None else np.asarray(order, dtype=np.int32)
    kernels.k_sweep(order, i, j, d, sysd["f"], sysd["offsets"], sysd["indices"], out)
    return out


def run_reference(sysd, out, order=None):
    i, j, d = to_csr(sysd["A"])
    order = sysd["order"] if order is None else np.asarray(order, dtype=np.int32)
    k_reference.sweep(
        order, len(sysd["f"]), i, j, d, sysd["f"], sysd["offsets"], sysd["indices"], out
    )
    return out


@pytest.mark.parametrize(
    "sizes,seed",
    [
        ([1], 0),
        ([3], 1),
        ([3] * 2, 2),
        ([3] * 40, 3),  # P1 triangles
        ([4] * 40, 4),
        ([1, 2, 3, 5, 8, 2, 1, 4], 5),  # mixed local sizes
        ([20] * 6, 6),  # larger local systems
    ],
)
def test_sweep_kernel_matches_reference(sizes, seed):
    s = make_system(sizes, seed)
    n = len(s["f"])
    got = run_kernel(s, np.zeros(n))
    ref = run_reference(s, np.zeros(n))
    np.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("seed", range(5))
def test_sweep_kernel_solves_global_system(seed):
    # Independent of the reference: upstream couplings move to the RHS, so the
    # assembled sweep solution must satisfy the full system.
    s = make_system([3] * 25, seed)
    u = run_kernel(s, np.zeros(len(s["f"])))
    np.testing.assert_allclose(s["A"] @ u, s["f"], rtol=1e-10, atol=1e-12)


def test_sweep_kernel_hand_checked_chain():
    # u0 = 1; row 1: -u0 + 2*u1 = 3 -> u1 = 2; row 2: -u1 + 4*u2 = 6 -> u2 = 2
    A = np.array([[2.0, 0, 0], [-1, 2, 0], [0, -1, 4]])
    s = {
        "A": A,
        "f": np.array([2.0, 3.0, 6.0]),
        "order": np.array([0, 1, 2], dtype=np.int32),
        "offsets": np.array([0, 1, 2, 3], dtype=np.int32),
        "indices": np.array([0, 1, 2], dtype=np.int32),
    }
    np.testing.assert_allclose(run_kernel(s, np.zeros(3)), [1.0, 2.0, 2.0])


def test_sweep_kernel_needs_pivoting():
    # Zero on the diagonal: fails without row pivoting.
    s = {
        "A": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "f": np.array([1.0, 2.0]),
        "order": np.array([0], dtype=np.int32),
        "offsets": np.array([0, 2], dtype=np.int32),
        "indices": np.array([0, 1], dtype=np.int32),
    }
    np.testing.assert_allclose(run_kernel(s, np.zeros(2)), [2.0, 1.0])


def test_sweep_kernel_unsorted_element_dofs():
    # Local ordering is [2, 0, 1], not ascending: the local<->global map must hold.
    A = np.array([[3.0, 1, 0], [0, 4, 1], [1, 0, 5]])
    s = {
        "A": A,
        "f": np.array([1.0, 2.0, 3.0]),
        "order": np.array([0], dtype=np.int32),
        "offsets": np.array([0, 3], dtype=np.int32),
        "indices": np.array([2, 0, 1], dtype=np.int32),
    }
    np.testing.assert_allclose(run_kernel(s, np.zeros(3)), np.linalg.solve(A, s["f"]))


def test_sweep_kernel_writes_result_into_out():
    s = make_system([3] * 10, seed=7)
    out = np.zeros(len(s["f"]))
    run_kernel(s, out)
    assert np.all(out != 0.0)  # every dof belongs to a swept element


def test_sweep_kernel_ignores_initial_out_for_swept_dofs():
    # Every off-element column is upstream, so garbage in `out` must not leak.
    s = make_system([3] * 15, seed=8)
    n = len(s["f"])
    clean = run_kernel(s, np.zeros(n))
    dirty = run_kernel(s, np.full(n, np.nan))
    np.testing.assert_array_equal(clean, dirty)


def test_sweep_kernel_does_not_modify_inputs():
    s = make_system([3] * 10, seed=9)
    i, j, d = to_csr(s["A"])
    before = [
        x.copy() for x in (i, j, d, s["f"], s["order"], s["offsets"], s["indices"])
    ]
    kernels.k_sweep(
        s["order"], i, j, d, s["f"], s["offsets"], s["indices"], np.zeros(len(s["f"]))
    )
    for b, a in zip(before, (i, j, d, s["f"], s["order"], s["offsets"], s["indices"])):
        np.testing.assert_array_equal(a, b)


def test_sweep_kernel_partial_order_leaves_other_dofs_untouched():
    s = make_system([3] * 8, seed=10)
    n = len(s["f"])
    sweep = s["order"][:3]
    out = np.full(n, -7.0)
    run_kernel(s, out, order=sweep)
    touched = np.concatenate([s["dofs"][e] for e in sweep])
    untouched = np.setdiff1d(np.arange(n), touched)
    np.testing.assert_array_equal(out[untouched], -7.0)
    ref = np.full(n, -7.0)
    run_reference(s, ref, order=sweep)
    np.testing.assert_allclose(out, ref, rtol=1e-11)


def test_sweep_kernel_empty_order():
    s = make_system([3] * 4, seed=11)
    out = np.full(len(s["f"]), 1.5)
    run_kernel(s, out, order=np.array([], dtype=np.int32))
    np.testing.assert_array_equal(out, 1.5)


def test_sweep_kernel_order_independent_for_decoupled_elements():
    # With no inter-element coupling, any element order gives the same answer.
    s = make_system([3] * 12, seed=12, density=0.0)
    n = len(s["f"])
    a = run_kernel(s, np.zeros(n))
    b = run_kernel(s, np.zeros(n), order=s["order"][::-1])
    np.testing.assert_allclose(a, b, rtol=1e-13)


# --------------------------------------------------------------------------- #
# k_sweep: O(1) argument checks in the binding                           #
# --------------------------------------------------------------------------- #

ARG_NAMES = [
    "order",
    "blf_csr_i",
    "blf_csr_j",
    "blf_csr_data",
    "source_array",
    "elem_dof_offset",
    "elem_dof_indices",
    "out",
]


def valid_args(n_el=4, size=3, seed=20):
    s = make_system([size] * n_el, seed)
    i, j, d = to_csr(s["A"])
    out = np.zeros(len(s["f"]))
    return [s["order"], i, j, d, s["f"], s["offsets"], s["indices"], out]


def call(args):
    return kernels.k_sweep(*args)


def with_arg(args, name, value):
    new = list(args)
    new[ARG_NAMES.index(name)] = value
    return new


def test_sweep_kernel_valid_args_pass():
    args = valid_args()
    call(args)
    assert np.all(args[-1] != 0.0)


@pytest.mark.parametrize("name", ARG_NAMES)
def test_sweep_kernel_rejects_non_1d(name):
    args = valid_args()
    arr = args[ARG_NAMES.index(name)]
    with pytest.raises(ValueError, match="1-D"):
        call(with_arg(args, name, np.ascontiguousarray(arr.reshape(-1, 1))))


def test_sweep_kernel_rejects_wrong_csr_i_length():
    args = valid_args()
    for bad in (args[1][:-1], np.append(args[1], args[1][-1])):
        with pytest.raises(ValueError, match="blf_csr_i must be 1-D of length"):
            call(with_arg(args, "blf_csr_i", bad))


def test_sweep_kernel_rejects_csr_j_data_mismatch():
    args = valid_args()
    with pytest.raises(ValueError, match="blf_csr_j must be 1-D of length"):
        call(with_arg(args, "blf_csr_j", args[2][:-1]))
    with pytest.raises(ValueError, match="blf_csr_j must be 1-D of length"):
        call(with_arg(args, "blf_csr_data", args[3][:-1]))


def test_sweep_kernel_rejects_csr_i_bad_endpoints():
    args = valid_args()
    bad = args[1].copy()
    bad[0] = 1
    with pytest.raises(ValueError, match="start at 0"):
        call(with_arg(args, "blf_csr_i", bad))
    bad = args[1].copy()
    bad[-1] += 1
    with pytest.raises(ValueError, match="end at the number of nonzeros"):
        call(with_arg(args, "blf_csr_i", bad))


def test_sweep_kernel_rejects_bad_elem_dof_offset():
    args = valid_args()
    with pytest.raises(ValueError, match="must not be empty"):
        call(with_arg(args, "elem_dof_offset", np.array([], dtype=np.int32)))
    bad = args[5].copy()
    bad[0] = 1
    with pytest.raises(ValueError, match="elem_dof_offset must start at 0"):
        call(with_arg(args, "elem_dof_offset", bad))
    bad = args[5].copy()
    bad[-1] += 1
    with pytest.raises(ValueError, match="end at len"):
        call(with_arg(args, "elem_dof_offset", bad))


def test_sweep_kernel_rejects_order_longer_than_element_count():
    args = valid_args(n_el=4)
    with pytest.raises(ValueError, match="more entries than there are elements"):
        call(with_arg(args, "order", np.arange(5, dtype=np.int32)))


def test_sweep_kernel_allows_order_shorter_than_element_count():
    args = valid_args(n_el=4)
    call(with_arg(args, "order", args[0][:2]))


def test_sweep_kernel_rejects_wrong_out_size():
    args = valid_args()
    n = len(args[-1])
    for bad in (np.zeros(n - 1), np.zeros(n + 1)):
        with pytest.raises(ValueError, match="out must be 1-D of length"):
            call(with_arg(args, "out", bad))


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.complex128])
def test_sweep_kernel_rejects_non_float64_out(dtype):
    # Must not silently write into a converted copy.
    args = valid_args()
    bad = np.zeros(len(args[-1]), dtype=dtype)
    with pytest.raises(ValueError, match="float64"):
        call(with_arg(args, "out", bad))


def test_sweep_kernel_rejects_non_array_out():
    args = valid_args()
    for bad in (list(args[-1]), None, 3.0):
        with pytest.raises(TypeError, match="numpy array"):
            call(with_arg(args, "out", bad))


def test_sweep_kernel_rejects_non_contiguous_out():
    args = valid_args()
    bad = np.zeros(2 * len(args[-1]))[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        call(with_arg(args, "out", bad))


def test_sweep_kernel_rejects_readonly_out():
    args = valid_args()
    bad = np.zeros(len(args[-1]))
    bad.setflags(write=False)
    with pytest.raises(ValueError, match="writeable"):
        call(with_arg(args, "out", bad))


def test_sweep_kernel_writes_into_callers_array_through_a_view():
    # A writeable contiguous view must be written in place, not copied.
    args = valid_args()
    n = len(args[-1])
    base = np.zeros(n + 5)
    call(with_arg(args, "out", base[2 : 2 + n]))
    assert np.all(base[2 : 2 + n] != 0.0)
    assert np.all(base[:2] == 0.0) and np.all(base[2 + n :] == 0.0)


def test_sweep_kernel_failed_check_leaves_out_untouched():
    args = valid_args()
    bad = args[1].copy()
    bad[-1] += 1
    out = args[-1]
    with pytest.raises(ValueError):
        call(with_arg(args, "blf_csr_i", bad))
    np.testing.assert_array_equal(out, 0.0)
