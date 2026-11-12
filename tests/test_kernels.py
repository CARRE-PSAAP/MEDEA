"""Each compiled kernel is checked against its pure-Python reference."""

import numpy as np
import pytest

from medea import _kernels, _reference


def random_dag(n, max_up=3, seed=0):
    """Cells 0..n-1 in sweep order; each has up to max_up upstream (lower) cells."""
    rng = np.random.default_rng(seed)
    offsets, cells = [0], []
    for c in range(n):
        k = int(rng.integers(0, min(c, max_up) + 1))
        cells.extend(rng.choice(c, size=k, replace=False) if k else [])
        offsets.append(len(cells))
    return (
        np.arange(n, dtype=np.int64),
        np.array(offsets, dtype=np.int64),
        np.array(cells, dtype=np.int64),
        rng.uniform(0.1, 1.0, len(cells)),
        rng.uniform(0.5, 2.0, n),
        rng.uniform(0.0, 1.0, n),
    )


@pytest.mark.parametrize("n,seed", [(1, 0), (2, 1), (50, 2), (1000, 3)])
def test_matches_reference(n, seed):
    args = random_dag(n, seed=seed)
    np.testing.assert_allclose(
        _kernels.sweep_toy(*args), _reference.sweep_toy(*args), rtol=1e-13
    )


def test_hand_checked_chain():
    # 0 -> 1 -> 2, weight 1, sigma 2, source 1 everywhere
    psi = _kernels.sweep_toy(
        order=[0, 1, 2],
        up_offsets=[0, 0, 1, 2],
        up_cells=[0, 1],
        up_weights=[1.0, 1.0],
        sigma=[2.0, 2.0, 2.0],
        source=[1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(psi, [0.5, 0.75, 0.875])


def test_other_valid_order_gives_same_result():
    # Two independent chains, 0->1->2 and 3->4->5, swept in different valid orders.
    cells = [0, 1, 3, 4]
    rest = dict(
        up_offsets=[0, 0, 1, 2, 2, 3, 4],
        up_cells=cells,
        up_weights=[0.5, 0.25, 0.5, 0.25],
        sigma=[1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        source=[1.0, 0.0, 2.0, 1.0, 0.5, 0.0],
    )
    a = _kernels.sweep_toy(order=[0, 1, 2, 3, 4, 5], **rest)
    b = _kernels.sweep_toy(order=[3, 0, 4, 1, 5, 2], **rest)
    np.testing.assert_array_equal(a, b)


def test_accepts_plain_lists_and_other_dtypes():
    psi = _kernels.sweep_toy([0], [0, 0], [], [], [2], [1])
    np.testing.assert_allclose(psi, [0.5])


def test_empty():
    assert _kernels.sweep_toy([], [0], [], [], [], []).shape == (0,)


def test_rejects_upstream_after_downstream():
    with pytest.raises(ValueError, match="not before"):
        _kernels.sweep_toy([1, 0], [0, 0, 1], [1], [1.0], [1.0, 1.0], [0.0, 0.0])


def test_rejects_bad_input():
    good = random_dag(5)
    with pytest.raises(ValueError, match="permutation"):
        _kernels.sweep_toy([0, 0, 1, 2, 3], *good[1:])
    with pytest.raises(ValueError, match="nonzero"):
        _kernels.sweep_toy(good[0], good[1], good[2], good[3], np.zeros(5), good[5])
    with pytest.raises(ValueError, match="out of range"):
        _kernels.sweep_toy([0], [0, 1], [5], [1.0], [1.0], [1.0])
    with pytest.raises(ValueError, match="length"):
        _kernels.sweep_toy(good[0], good[1], good[2], good[3], good[4], [1.0])
    with pytest.raises(ValueError, match="1-D"):
        _kernels.sweep_toy([0], [0, 1], [[0]], [1.0], [1.0], [1.0])
