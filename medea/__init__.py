"""MEDEA: deterministic transport tools built on MFEM."""

from . import _core
from ._core import HelloResult, hello_mfem

MFEM_MODE = _core.MFEM_MODE  # "ser" or "par": which PyMFEM build was linked

__all__ = ["HelloResult", "hello_mfem", "MFEM_MODE"]
