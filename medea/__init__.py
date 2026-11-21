"""MEDEA: deterministic transport tools built on MFEM."""

from ._backend import MFEM_MODE  # "ser" or "par": which PyMFEM module is in use
import medea.sweep
from medea._backend import mfem

try:
    from . import kernels
except ImportError as err:  # compiled module missing
    raise ImportError(
        "medea.kernels (the compiled extension) is missing. Install with "
        "'pip install -e .'. If you are running from the repository root with a "
        "non-editable install, Python is importing the source tree instead of "
        "the installed package: run from another directory."
    ) from err

from ._backend import mfem
from .materials import Material, MaterialGroup
from .sweep import TransportSweep

__all__ = ["MFEM_MODE", "Material", "MaterialGroup", "TransportSweep", "mfem"]
