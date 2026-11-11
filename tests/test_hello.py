"""Mode-agnostic tests: the same assertions run in serial and parallel.

The ``num_*`` and ``integral`` fields are totals over all ranks, so they do not
depend on how many ranks the suite is launched with.
"""

import pytest

import medea


def test_mode_is_valid():
    assert medea.MFEM_MODE in ("ser", "par")


def test_hello_mfem_counts():
    r = medea.hello_mfem(n=4, order=2)
    assert r.num_elements == 16
    assert r.num_dofs == 81  # (n*order + 1)^2


def test_hello_mfem_integral():
    assert medea.hello_mfem(n=3, order=1).integral == pytest.approx(1.0)


def test_ranks_are_consistent():
    r = medea.hello_mfem()
    assert 0 <= r.rank < r.num_ranks
    assert 0 < r.local_elements <= r.num_elements
    assert 0 < r.local_dofs <= r.num_dofs


def test_mfem_version_reported():
    import mfem

    assert medea.hello_mfem().mfem_version == mfem.__version__


def test_bad_args():
    with pytest.raises(ValueError):
        medea.hello_mfem(n=0)
    with pytest.raises(ValueError):
        medea.hello_mfem(order=0)
