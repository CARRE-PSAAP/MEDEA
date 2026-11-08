"""Tests in this directory need a MEDEA built against parallel (MPI) MFEM.

On a serial install they are not collected at all (so their ``mfem.par`` and
``mpi4py`` imports never run). Set MEDEA_REQUIRE_PARALLEL=1, as CI does, to
make a serial install an error instead, so a broken parallel build cannot
pass by silently skipping everything.

Tests that need more than one rank carry ``@pytest.mark.mpi(min_size=2)``
(from pytest-mpi) and run only under ``mpiexec -n 2 pytest --with-mpi``.
"""

import os

import medea

if medea.MFEM_MODE != "par":
    if os.environ.get("MEDEA_REQUIRE_PARALLEL"):
        raise RuntimeError(
            "MEDEA_REQUIRE_PARALLEL is set but medea.MFEM_MODE is "
            f"{medea.MFEM_MODE!r}; install with scripts/install_parallel.sh"
        )
    collect_ignore_glob = ["*.py"]
