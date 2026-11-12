"""MEDEA: deterministic transport tools built on MFEM."""

from ._backend import MFEM_MODE  # "ser" or "par": which PyMFEM module is in use
from .hello import HelloResult, hello_mfem

try:
    from . import _kernels
    from ._kernels import sweep_toy
except ImportError as err:  # compiled module missing
    raise ImportError(
        "medea._kernels (the compiled extension) is missing. Install with "
        "'pip install -e .'. If you are running from the repository root with a "
        "non-editable install, Python is importing the source tree instead of "
        "the installed package: run from another directory."
    ) from err

__all__ = ["HelloResult", "hello_mfem", "MFEM_MODE", "sweep_toy"]
