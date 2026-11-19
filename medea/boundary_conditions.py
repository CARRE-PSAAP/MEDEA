import numpy as np

from medea._backend import mfem

from .quadrature import QuadratureSet


class BoundaryCondition:
    """
    This class stores incoming flux boundary conditions, indexed by quadrature angle
    number and :class:`mfem.Mesh` boundary attribute number. Since the boundary
    condition is strongly coupled to mesh and quadrature, it must be validated before
    being converted to an MFEM coefficient.

    Usage:

    .. code-block:: python

        import numpy as np
        from medea import BoundaryCondition
        from medea.mfem import Mesh

        ...
        # build mesh with 3 boundary attribute types
        # build quadrature set with 16 angles
        ...

        # isotropic incoming flux, psi=1.0
        bc = BoundaryCondition(np.ones((3, 16)))

        # validate with mesh and quadrature
        bc.validate(mesh, quad)

        # get MFEM coefficients
        incoming_mu_0 = bc[0] # returns mfem.PWConstCoefficient
    """
    def __init__(self, values: np.ndarray, attributes: list[int] | None = None):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"BoundaryCondition values must be 2D, got {values.ndim}D")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("BoundaryCondition values must be finite and non-negative")

        if attributes is None:
            attributes = range(1, values.shape[0] + 1)
        if len(attributes) != values.shape[0]:
            raise ValueError(
                f"Length of attributes ({len(attributes)}) must match length of values ({values.shape[0]})"
            )
        self._attr = list(attributes)
        self._data = values
        self._coefficients = None

    def validate(self, mesh: mfem.Mesh, quad: QuadratureSet):
        sh = self._data.shape
        mesh_attr = set(mesh.bdr_attributes.GetDataArray().tolist())
        if set(self._attr) != mesh_attr:
            raise ValueError(
                f"BoundaryCondition attributes {sorted(self._attr)} do not match Mesh boundary attributes {sorted(mesh_attr)}"
            )
        if sh[1] != len(quad.angles):
            raise ValueError(
                f"Number of BoundaryCondition angles ({sh[1]}) does not match number of quadrature angles ({len(quad.angles)})"
            )

        # build Coefficients here so validation is forced
        self._coefficients = []
        for i in range(sh[1]):
            vec = np.zeros(max(self._attr))
            vec[np.array(self._attr) - 1] = self._data[:, i]
            self._coefficients.append(mfem.PWConstCoefficient(mfem.Vector(vec)))

    def __getitem__(self, index):
        # index by quadrature angle
        if self._coefficients is None:
            raise RuntimeError("BoundaryCondition must be validated before indexing")
        return self._coefficients[index]
