import copy
import warnings

import numpy as np

from medea._backend import mfem


class MaterialView:
    """Dict-like view (material name -> data) onto part of a reaction.

    Created by indexing :class:`MaterialData`, not directly. ``array`` is a
    read-only numpy view of shape ``(M, *shape)`` (material axis first); no data
    is copied when indexing.

    Keys:

    * ``str``: the data for that material.
    * ``int``, ``slice``, ``Ellipsis``, or a tuple of them: index the non-material
      axes and return a new view, so ``v[i][j] == v[i, j]``.

    Iteration, ``in``, ``keys()``, ``items()`` and ``len()`` run over material
    names. (Not a ``Mapping``: non-string keys index axes, not materials.)
    """

    def __init__(self, array, names, index, rows):
        self._array = array
        self._names = names
        self._index = index
        self._rows = rows

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
        return MaterialView(
            self._array[(slice(None),) + items], self._names, self._index, self._rows
        )

    def to_PWConst(self):
        """Convert to ``mfem.PWConstCoefficient`` object(s).

        Returns an object array of shape ``self.shape`` whose elements are
        coefficients taking, on each mesh attribute, that element's value for the
        material mapped to it (0 for unmapped attributes). A scalar view
        (``shape == ()``) returns the single coefficient instead.

        Builds one coefficient per element, so index first where possible: a full
        ``(G, G, L)`` scatter array creates ``G*G*L`` coefficients.
        """
        out = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(*self.shape):
            per_mat = self._array[(slice(None),) + idx]
            # Appending 0 makes the "unmapped" row index -1 select it.
            vals = np.append(per_mat, 0.0)[self._rows]
            out[idx] = mfem.PWConstCoefficient(mfem.Vector(vals))
        return out[()] if out.ndim == 0 else out

    def __repr__(self):
        return f"MaterialView(shape={self.shape}, materials={list(self._names)})"


class MaterialData:
    """Cross-section data for a set of materials.

    Args:
        materials: ``{material_name: {reaction: array_like}}``. All materials
            must have the same reactions, each with the same shape. Values are
            converted to float64; a scalar has shape ``()``.
        attributes: ``{mesh_attribute: material_name}``. Keys must be positive
            integers (gaps allowed; unmapped attributes evaluate to 0) and values
            must be names in ``materials``. A material may appear on several
            attributes; one on none triggers a warning. Defaults to
            ``{1: first_material, 2: second_material, ...}``.

    Index with a reaction name to get a :class:`MaterialView`.
    """

    def __init__(self, materials, attributes=None):
        if not materials:
            raise ValueError("'materials' must be non-empty")

        self._names = tuple(materials)
        self._index = {n: i for i, n in enumerate(self._names)}
        if attributes is None:
            attributes = dict(enumerate(self._names, start=1))
        self._set_attributes(attributes)

        unused = set(self._names) - set(attributes.values())
        if unused:
            warnings.warn(
                f"materials {sorted(unused)} are not assigned to any mesh attribute",
                stacklevel=2,
            )

        first = self._names[0]
        reactions = list(materials[first])
        for name in self._names:
            if set(materials[name]) != set(reactions):
                raise ValueError(
                    f"material {name!r} has reactions {sorted(materials[name])}, "
                    f"but {first!r} has {sorted(reactions)}; all materials must "
                    "have the same reactions"
                )

        self._data = {}
        for rxn in reactions:
            arrs = [np.asarray(materials[n][rxn], dtype=np.float64) for n in self._names]
            shape = arrs[0].shape
            for n, a in zip(self._names, arrs):
                if a.shape != shape:
                    raise ValueError(
                        f"reaction {rxn!r}: material {n!r} has shape {a.shape}, "
                        f"but {first!r} has shape {shape}"
                    )
            stacked = np.stack(arrs)  # copies, so the caller's arrays are untouched
            stacked.flags.writeable = False
            self._data[rxn] = stacked

    def _set_attributes(self, attributes):
        keys = sorted(attributes)
        if not keys:
            raise ValueError("'attributes' must be non-empty")
        if not all(isinstance(a, (int, np.integer)) and a >= 1 for a in keys):
            raise ValueError(f"mesh attributes must be positive integers; got {keys}")
        unknown = set(attributes.values()) - set(self._names)
        if unknown:
            raise ValueError(
                f"'attributes' refers to unknown materials {sorted(unknown)}; "
                f"materials are {list(self._names)}"
            )
        self._attributes = {int(a): attributes[a] for a in keys}
        # rows[a - 1] = material row for mesh attribute a, or -1 if unmapped
        self._rows = np.full(keys[-1], -1, dtype=np.intp)
        for a in keys:
            self._rows[a - 1] = self._index[attributes[a]]

    def __getitem__(self, reaction):
        try:
            data = self._data[reaction]
        except KeyError:
            raise KeyError(
                f"unknown reaction {reaction!r}; reactions are {list(self._data)}"
            ) from None
        return MaterialView(data, self._names, self._index, self._rows)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @property
    def names(self):
        """Material names, in input order."""
        return self._names

    @property
    def attributes(self):
        """``{mesh_attribute: material_name}``, sorted by attribute."""
        return dict(self._attributes)

    def remap(self, attributes):
        """Return a new ``MaterialData`` with a different attribute map.

        ``attributes`` follows the constructor's rules. Data arrays are shared,
        not copied; this object is unchanged.
        """
        new = copy.copy(self)
        new._set_attributes(attributes)
        return new

    def check_mesh(self, mesh):
        """Raise ``ValueError`` if ``mesh`` has attributes with no material.

        Mapped attributes that the mesh does not use are fine.
        """
        missing = set(mesh.attributes.ToList()) - set(self._attributes)
        if missing:
            raise ValueError(
                f"mesh has attributes {sorted(missing)} with no material "
                f"assigned; mapped attributes are {sorted(self._attributes)}"
            )

    def __repr__(self):
        return (
            f"MaterialData(materials={list(self._names)}, "
            f"reactions={list(self._data)})"
        )
