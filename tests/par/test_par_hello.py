"""Parallel-only checks. Everything mode-agnostic lives in tests/test_hello.py.

Run on one rank with plain ``pytest``, and on several with
``mpiexec -n 2 pytest --with-mpi``.
"""

import pytest
from mpi4py import MPI

import medea

comm = MPI.COMM_WORLD


def test_mode_is_par():
    assert medea.MFEM_MODE == "par"


def test_ranks_match_mpi4py():
    r = medea.hello_mfem()
    assert r.rank == comm.Get_rank()
    assert r.num_ranks == comm.Get_size()


def test_results_do_not_depend_on_rank_count():
    # Same expected values whether this runs on 1, 2 or 4 ranks.
    n, order = 5, 2
    r = medea.hello_mfem(n=n, order=order)
    assert r.num_elements == n * n
    assert r.num_dofs == (n * order + 1) ** 2
    assert r.integral == pytest.approx(1.0)
    assert comm.allreduce(r.local_elements) == n * n


# These need a real partition, so they run only under mpiexec -n 2 --with-mpi.


@pytest.mark.mpi(min_size=2)
def test_mesh_is_partitioned():
    r = medea.hello_mfem(n=4)
    assert r.num_ranks >= 2
    assert 0 < r.local_elements < r.num_elements


@pytest.mark.mpi(min_size=2)
def test_shared_dofs_not_double_counted():
    # Dofs on partition boundaries are visible to several ranks but owned by
    # exactly one, so the owned ("true") counts sum to the global count.
    r = medea.hello_mfem(n=4, order=2)
    assert comm.allreduce(r.local_dofs) == r.num_dofs
