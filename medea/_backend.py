"""
Loads parallel or serial mfem, depending on which is installed, and provides
wrappers for MPI.

To load PyMFEM in MEDEA modules, use ``from medea._backend import mfem``.

To access MPI, use ``from medea._backend import comm``. Use helper function
``is_parallel()`` to determine if code is being executed using MPI. Use
``rank()`` and ``size()`` to get MPI world parameters.

Notes:

1. If something already imported one of them, that one is used (and a
   conflicting ``MEDEA_BACKEND`` is an error).
2. ``MEDEA_BACKEND=ser`` or ``par`` forces a mode. Forcing ``par`` when no
   parallel PyMFEM is installed is an error.
3. Otherwise ("auto"): parallel if a parallel PyMFEM is installed, else serial.
"""

import importlib
import os
from pathlib import Path

import mfem as _mfem_pkg  # the base package; importing it loads no mode

_VALID = ("auto", "ser", "par")


def _parallel_installed():
    """True if this PyMFEM was built with parallel support.

    A parallel build ships its libraries in ``mfem/external/par``.
    """
    return (Path(_mfem_pkg.__file__).parent / "external" / "par").is_dir()


def _resolve():
    requested = os.environ.get("MEDEA_BACKEND", "auto").strip().lower() or "auto"
    if requested not in _VALID:
        raise ValueError(
            f"MEDEA_BACKEND={requested!r} is not valid; use one of {_VALID}"
        )

    loaded = getattr(_mfem_pkg, "mfem_mode", None)  # "serial", "parallel" or None
    if loaded is not None:
        mode = "par" if loaded == "parallel" else "ser"
        if requested not in ("auto", mode):
            raise RuntimeError(
                f"MEDEA_BACKEND={requested!r} but mfem.{mode} is already "
                "imported in this process; PyMFEM cannot load both. Import "
                "medea before mfem, or make the choices agree."
            )
        return mode

    if requested == "par" and not _parallel_installed():
        raise ImportError(
            "MEDEA_BACKEND=par but this PyMFEM has no parallel build. "
            "See scripts/install_parallel.sh."
        )
    if requested != "auto":
        return requested
    return "par" if _parallel_installed() else "ser"


MFEM_MODE = _resolve()  # "ser" or "par"
mfem = importlib.import_module(f"mfem.{MFEM_MODE}")
is_parallel = MFEM_MODE == "par"

if is_parallel:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
else:
    MPI = None
    comm = None


def rank(c=None):
    """Rank of ``c`` (default: the world); always 0 in serial mode."""
    c = comm if c is None else c
    return 0 if c is None else c.Get_rank()


def size(c=None):
    """Number of ranks in ``c`` (default: the world); always 1 in serial mode."""
    c = comm if c is None else c
    return 1 if c is None else c.Get_size()
