import copy
import warnings

import numpy as np

from medea._backend import mfem


class Material:
    """Data and name for one material.

    :param name: Non-empty string identifying the material.

    :param xs_total: Total cross section, shape ``(n_groups,)``. A scalar
        is converted to shape ``(1,)``.

    :param xs_scatter: Scattering cross section, shape ``(n_groups, n_groups)``.
        A scalar is converted to shape ``(1, 1)``.

    :param n_groups: Number of energy groups (default 1). ``xs_total`` and
        ``xs_scatter`` are checked against it.

    :param optional: Any further properties as keyword arguments,
        ``{property: array_like}``, e.g. ``stopping_power=[1.0, 2.0]``.

    All data values are converted to ``np.ndarray[np.float64]`` (including
    scalars) so that they have a ``shape``. This allows multiple materials to
    be checked against each other, ensuring all data members have the same shape.

    Every property is available by name, ``mat["xs_total"]``; total and scattering
    cross sections are also available as attributes, ``mat.xs_total``.

    Note: ``xs_scatter`` is the cross section only, i.e. the first Legendre moment
    of the differential scattering cross section. Scattering angular distributions
    are specified using optional parameters (to be added).
    """

    __slots__ = ("_name", "_n_groups", "_data")

    def __init__(self, name, xs_total, xs_scatter, n_groups: int = 1, **optional):
        if not isinstance(name, str) or not name:
            raise ValueError(f"material name must be a non-empty string; got {name!r}")
        if (
            isinstance(n_groups, bool)
            or not isinstance(n_groups, (int, np.integer))
            or n_groups < 1
        ):
            raise ValueError(f"n_groups must be a positive integer; got {n_groups!r}")
        self._name = name
        self._n_groups = int(n_groups)
        self._data = {}
        for prop, value in {
            "xs_total": xs_total,
            "xs_scatter": xs_scatter,
            **optional,
        }.items():
            arr = np.array(value, dtype=np.float64)  # copies
            if arr.ndim == 0 and prop == "xs_total":
                arr = arr.reshape(1)  # scalar -> (1,)
            elif arr.ndim == 0 and prop == "xs_scatter":
                arr = arr.reshape(1, 1)  # scalar -> (1, 1)
            arr.flags.writeable = False
            self._data[prop] = arr

        G = self._n_groups
        for prop, expected in (("xs_total", (G,)), ("xs_scatter", (G, G))):
            if self._data[prop].shape != expected:
                raise ValueError(
                    f"material {name!r}: {prop} has shape {self._data[prop].shape}, "
                    f"expected {expected} (n_groups={G})"
                )

    @property
    def name(self):
        return self._name

    @property
    def n_groups(self):
        """Number of energy groups."""
        return self._n_groups

    @property
    def xs_total(self):
        """Total cross section, shape ``(n_groups,)``."""
        return self._data["xs_total"]

    @property
    def xs_scatter(self):
        """Scattering cross section, shape ``(n_groups, n_groups)``."""
        return self._data["xs_scatter"]

    @property
    def shapes(self):
        """:return: ``{property: shape}`` for this material, e.g. ``{"xs_total": (2,)}``"""
        return {p: a.shape for p, a in self._data.items()}

    def replace(self, **properties):
        # Return a new ``Material`` (same name) with some properties changed or added.
        return Material(
            self._name, n_groups=self._n_groups, **{**self._data, **properties}
        )

    def __getitem__(self, prop):
        try:
            return self._data[prop]
        except KeyError:
            raise KeyError(
                f"material {self._name!r} has no property {prop!r}; "
                f"properties are {list(self._data)}"
            ) from None

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, prop):
        return prop in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def __repr__(self):
        return f"Material({self._name!r}, properties={list(self._data)})"


class MaterialProperty:
    """Dict-like view (material name -> data) of one property across materials.

    Created by indexing :class:`MaterialGroup`, not directly. ``array`` is a
    read-only numpy view of shape ``(M, *shape)`` (material axis first); no data
    is copied when indexing.

    Keys:

    * ``str``: the data for that material.
    * ``int``, ``slice``, ``Ellipsis``, or a tuple of them: index the non-material
      axes and return a new view, so ``v[i][j] == v[i, j]``.

    Iteration, ``in``, ``keys()``, ``items()`` and ``len()`` run over material
    names. (Not a ``Mapping``: non-string keys index axes, not materials.)
    """

    def __init__(self, array, coefficients, names, index):
        self._array = array
        self._coefficients = coefficients  # object array, shape == array.shape[1:]
        self._names = names
        self._index = index

    @property
    def array(self):
        """The underlying ``(M, *shape)`` array (read-only)."""
        return self._array

    @property
    def shape(self):
        """Shape of the data for a single material (material axis excluded)."""
        return self._array.shape[1:]

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)

    def keys(self):
        return self._names

    def items(self):
        return ((n, self[n]) for n in self._names)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self._array[self._index[key]]
            except KeyError:
                raise KeyError(
                    f"unknown material {key!r}; materials are {list(self._names)}"
                ) from None

        items = key if isinstance(key, tuple) else (key,)
        array = self._array[(slice(None),) + items]  # raises IndexError if too deep
        coefficients = self._coefficients[items]
        if not isinstance(coefficients, np.ndarray):  # fully indexed: bare element
            wrapped = np.empty((), dtype=object)
            wrapped[()] = coefficients
            coefficients = wrapped
        return MaterialProperty(array, coefficients, self._names, self._index)

    def to_PWConst(self):
        """Return the ``mfem.PWConstCoefficient`` object(s) for this property.

        Returns an object array of shape ``self.shape`` whose elements are
        coefficients taking, on each mesh attribute, that element's value for the
        material mapped to it (0 for unmapped attributes). A scalar view
        (``shape == ()``) returns the single coefficient instead.

        The coefficients are built once, when the :class:`MaterialGroup` is created
        (or remapped); this only hands them out, so repeated calls return the same
        coefficient objects.
        """
        if self._coefficients.ndim == 0:
            return self._coefficients[()]
        return self._coefficients.copy()  # new container, same coefficients

    def __repr__(self):
        return f"MaterialProperty(shape={self.shape}, materials={list(self._names)})"


class MaterialGroup:
    r"""A set of :class:`Material` objects assigned to mesh attributes.

    :param *materials: :class:``Material`` objects; Names must be unique, and all materials must
        have the same properties (cross sections) and shape.
    :param attributes: ``{mesh_attribute: material}`` where each material is a
        ``Material`` from this group or its name. Keys are positive
        integers. A material may be mapped to several attributes, and materials may be
        skipped (will trigger a warning). Defaults to
        ``{1: first_material, 2: second_material, ...}``.

    Index by property name (``mats["xs_total"]``, or ``mats.xs_total`` for the required
    cross sections) and use :func:`to_PWConst()` to convert to MFEM coefficients.

    .. code-block:: python

        from medea.materials import MaterialGroup

        mats = MaterialGroup(
            Material("Al", [2.0, 1.0], [[0.1, 1.0], [0.0, 0.5]], n_groups=2),
            Material("SiO2", [3.0, 1.5], [[0.2, 1.9], [0.9, 0.0]], n_groups=2),
            attributes = {1: "Al", 2: "SiO2"},
        )

        mats.xs_total       # same as mats["xs_total"]
        mats["xs_total"]    # {"Al": np.ndarray([1.0, 2.0]),
                            #  "SiO2": np.ndarray([1.5, 3.0])}

        mats["xs_total"][0] #  {"Al": np.ndarray([1.0]),
                            #   "SiO2": np.ndarray([1.5])}

        mats["xs_total"][0].to_PWConst() # mfem.PWConstCoeff(mfem.Vector(1.0, 1.5))

        mats = mats.remap({1: "SiO2", 2: "Al"}) # switches mapping of the two materials

    Use :meth:`.check_mesh` and :meth:`.require` to validate a material group for use
    in a method.
    """

    def __init__(self, *materials, attributes=None):
        self._build(materials, attributes)

    @classmethod
    def from_dict(cls, materials, attributes=None, n_groups=1):
        """Build from a Python dictionary, ``{material_name: {property: array_like}}``.

        Each inner dictionary must contain ``xs_total`` and ``xs_scatter``; all
        materials get ``n_groups`` energy groups. ``attributes`` follows the
        constructor's rules.
        """
        if not materials:
            raise ValueError("'materials' must be non-empty")
        self = object.__new__(cls)
        self._build(
            [Material(n, n_groups=n_groups, **d) for n, d in materials.items()],
            attributes,
        )
        return self

    def _build(self, materials, attributes):
        if not materials:
            raise ValueError("a MaterialGroup needs at least one material")
        for m in materials:
            if not isinstance(m, Material):
                raise TypeError(
                    f"MaterialGroup takes Material objects; got {type(m).__name__} "
                    "(use MaterialGroup.from_dict for plain dictionaries)"
                )

        self._materials = tuple(materials)
        self._names = tuple(m.name for m in materials)
        if len(set(self._names)) != len(self._names):
            dupes = sorted({n for n in self._names if self._names.count(n) > 1})
            raise ValueError(f"duplicate material names {dupes}")
        self._index = {n: i for i, n in enumerate(self._names)}

        self._check_consistency()
        self._stack_data()

        if attributes is None:
            attributes = dict(enumerate(self._names, start=1))
        self._set_attributes(attributes)

        unused = set(self._names) - set(self._attributes.values())
        if unused:
            warnings.warn(
                f"materials {sorted(unused)} are not assigned to any mesh attribute"
            )

    def _check_consistency(self):
        first = self._materials[0]
        props = list(first)
        self._n_groups = first.n_groups
        for m in self._materials:
            if m.n_groups != first.n_groups:
                raise ValueError(
                    f"material {m.name!r} has n_groups={m.n_groups}, "
                    f"but {first.name!r} has n_groups={first.n_groups}"
                )
            if set(m) != set(props):
                raise ValueError(
                    f"material {m.name!r} has properties {sorted(m)}, "
                    f"but {first.name!r} has {sorted(props)}; all materials must "
                    "have the same properties"
                )
        for prop in props:
            shape = first[prop].shape
            for m in self._materials:
                if m[prop].shape != shape:
                    raise ValueError(
                        f"property {prop!r}: material {m.name!r} has shape "
                        f"{m[prop].shape}, but {first.name!r} has shape {shape}"
                    )

    def _stack_data(self):
        """Stack each property over materials into a read-only ``(M, *shape)`` array."""
        self._data = {}
        for prop in self._materials[0]:
            stacked = np.stack([m[prop] for m in self._materials])
            stacked.flags.writeable = False
            self._data[prop] = stacked

    def _set_attributes(self, attributes):
        keys = sorted(attributes)
        if not keys:
            raise ValueError("'attributes' must be non-empty")
        if not all(isinstance(a, (int, np.integer)) and a >= 1 for a in keys):
            raise ValueError(f"mesh attributes must be positive integers; got {keys}")

        resolved = {}
        for a in keys:
            resolved[int(a)] = self._resolve(attributes[a])
        unknown = {n for n in resolved.values() if n not in self._index}
        if unknown:
            raise ValueError(
                f"'attributes' refers to unknown materials {sorted(unknown)}; "
                f"materials are {list(self._names)}"
            )

        self._attributes = resolved
        # rows[a - 1] = material row for mesh attribute a, or -1 if unmapped
        rows = np.full(keys[-1], -1, dtype=np.intp)
        for a, name in resolved.items():
            rows[a - 1] = self._index[name]
        self._build_coefficients(rows)

    def _resolve(self, material):
        """Name of ``material`` (a ``Material`` of this group, or a name)."""
        if isinstance(material, Material):
            i = self._index.get(material.name)
            if i is None or self._materials[i] is not material:
                raise ValueError(
                    f"material {material.name!r} is not part of this group"
                )
            return material.name
        return material

    def _build_coefficients(self, rows):
        """Build every property's ``PWConstCoefficient`` array for the current attributes."""
        self._coefficients = {}
        for prop, data in self._data.items():
            out = np.empty(data.shape[1:], dtype=object)
            for idx in np.ndindex(*out.shape):
                per_mat = data[(slice(None),) + idx]
                # Appending 0 makes the "unmapped" row index -1 select it.
                vals = np.append(per_mat, 0.0)[rows]
                out[idx] = mfem.PWConstCoefficient(mfem.Vector(vals))
            self._coefficients[prop] = out

    def __getitem__(self, prop):
        try:
            data = self._data[prop]
        except KeyError:
            raise KeyError(
                f"unknown property {prop!r}; properties are {list(self._data)}"
            ) from None
        return MaterialProperty(
            data, self._coefficients[prop], self._names, self._index
        )

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @property
    def materials(self):
        """The :class:`Material` objects, in input order."""
        return self._materials

    @property
    def names(self):
        """Material names, in input order."""
        return self._names

    @property
    def properties(self):
        """
        Property names shared by every material.

        :return: ``tuple[str]``
        """
        return tuple(self._data)

    @property
    def n_groups(self):
        """Number of energy groups (shared by every material)."""
        return self._n_groups

    @property
    def xs_total(self):
        """:class:`MaterialProperty` for ``xs_total``; same as ``self["xs_total"]``."""
        return self["xs_total"]

    @property
    def xs_scatter(self):
        """:class:`MaterialProperty` for ``xs_scatter``; same as ``self["xs_scatter"]``."""
        return self["xs_scatter"]

    @property
    def attributes(self):
        """``{mesh_attribute: material_name}``, sorted by attribute."""
        return dict(self._attributes)

    def material(self, name):
        """The :class:`Material` called ``name``."""
        try:
            return self._materials[self._index[name]]
        except KeyError:
            raise KeyError(
                f"unknown material {name!r}; materials are {list(self._names)}"
            ) from None

    def require(self, *properties, **shapes):
        """Raise ``ValueError`` unless the data suits a method.

        Positional arguments name properties that must exist. Keyword arguments name a
        property *and* the exact shape each material's data must have (an ``int`` or a
        tuple), so a method states what it needs in terms of :attr:`n_groups`. The
        required ``xs_total`` and ``xs_scatter`` are always present and already
        checked, so this is for optional properties:

        .. code-block:: python

            G = mats.n_groups
            mats.require("temp", stopping_power=(G,))

        Shapes are exact. A scalar does not stand in for ``(1,)``, because indexing it
        as ``mats["stopping_power"][0]`` would fail; give one-group data as ``[1.0]``.
        """
        names = list(properties) + [p for p in shapes if p not in properties]
        missing = [p for p in names if p not in self._data]
        if missing:
            raise ValueError(
                f"materials are missing properties {missing}; "
                f"available properties are {list(self._data)}"
            )
        for prop, expected in shapes.items():
            if isinstance(expected, (int, np.integer)):
                expected = (expected,)
            expected = tuple(expected)
            actual = self._data[prop].shape[1:]
            if actual != expected:
                raise ValueError(
                    f"property {prop!r} has shape {actual} for each material, "
                    f"expected {expected} (n_groups={self._n_groups})"
                )

    def remap(self, attributes):
        """Return a new :class:`MaterialGroup` with a different attribute map.

        ``attributes`` follows the constructor's rules. Data arrays are shared,
        not copied, and this object is unchanged.
        """
        new = copy.copy(self)
        new._set_attributes(attributes)
        return new

    def check_mesh(self, mesh):
        """Raise ``ValueError`` if ``mesh`` has attributes with no assigned material."""
        missing = set(mesh.attributes.ToList()) - set(self._attributes)
        if missing:
            raise ValueError(
                f"mesh has attributes {sorted(missing)} with no material "
                f"assigned; mapped attributes are {sorted(self._attributes)}"
            )

    def __repr__(self):
        return (
            f"MaterialGroup(materials={list(self._names)}, "
            f"properties={list(self._data)}, n_groups={self._n_groups})"
        )
