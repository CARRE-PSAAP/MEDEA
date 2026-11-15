"""
Transport sweep scheduling and kernel.
"""

import warnings
from collections import deque

import numpy as np

from medea._backend import mfem
from medea._kernels import sweep_kernel


class TransportSweep:
    """
    Object for creating mesh sweep orderings and performing the sweep.
    The constructor uses the sign of (n dot angle), where n is the face normal vector,
    to determine element coupling for a transport solve in the direction of ``angle``.
    Kahn's Algorithm is then used to determine an ordering for those elements, where
    each dependency is visited before the element that needs it.

    The ordering is based only on the unit normal vector of the face (not actual ``dof``
    coupling), so for now the sweep ordering is only guaranteed for flat faces.
    """

    def __init__(
        self,
        mesh: mfem.Mesh,
        fespace: mfem.FiniteElementSpace,
        angles: mfem.Vector | list[mfem.Vector],
    ):
        # a single angle is treated as a list of one

        if isinstance(angles, mfem.Vector):
            angles = [angles]

        sdim = mesh.SpaceDimension()

        # ensure every angle and the mesh dimension match
        for angle in angles:
            if angle.Size() != sdim:
                raise ValueError(
                    f"Invalid dimensions in 'angle' for mesh of dimension {sdim}"
                )

        self.angles = angles

        is_curved = mesh.NURBSext is not None
        nodal_fes = mesh.GetNodalFESpace()
        if nodal_fes is not None and nodal_fes.FEColl().GetOrder() > 1:
            is_curved = True

        if is_curved == True:
            warnings.warn("Curved meshes not verified", UserWarning)

        TOL = 1e-14
        num_elements = mesh.GetNE()

        # prepare element -> dof table
        dof_lists = [
            np.asarray(fespace.GetElementDofs(e), dtype=np.int32)
            for e in range(num_elements)
        ]
        self._elem_dof_offsets = np.zeros(num_elements + 1, dtype=np.int32)
        self._elem_dof_offsets[1:] = np.cumsum([len(d) for d in dof_lists])
        self._elem_dof_indices = np.concatenate(dof_lists)

        elem1 = []
        elem2 = []
        normals = []
        nor = mfem.Vector(sdim)
        for face_id in range(mesh.GetNumFaces()):
            e1, e2 = mesh.GetFaceElements(face_id)
            if e2 < 0:
                continue

            face = mesh.GetFaceElementTransformations(face_id)
            ip = mfem.Geometries.GetCenter(face.GetGeometryType())
            face.SetIntPoint(ip)
            mfem.CalcOrtho(face.Jacobian(), nor)

            elem1.append(e1)
            elem2.append(e2)
            normals.append(nor.GetDataArray().copy())

        elem1 = np.array(elem1, dtype=int)
        elem2 = np.array(elem2, dtype=int)
        normals = np.array(normals).reshape(len(elem1), sdim)

        self.order = []
        for angle in angles:
            direction = normals @ angle.GetDataArray()

            # faces parallel to the sweep direction carry no upwind dependency
            keep = np.abs(direction) >= TOL
            forward = direction > 0
            parents = np.where(forward, elem1, elem2)[keep]
            kids = np.where(forward, elem2, elem1)[keep]

            # children[element] lists all elements that depend on this one
            children = [[] for _ in range(num_elements)]
            in_degree = np.bincount(kids, minlength=num_elements).tolist()
            for parent, child in zip(parents.tolist(), kids.tolist()):
                children[parent].append(child)

            queue = deque(e for e, deg in enumerate(in_degree) if deg == 0)
            result = []

            while queue:
                ind = queue.popleft()
                for c in children[ind]:
                    in_degree[c] -= 1
                    if in_degree[c] == 0:
                        queue.append(c)
                result.append(ind)
            if len(result) != num_elements:
                raise ValueError("Cycle detected in mesh")
            self.order.append(np.array(result))

    def sweep(
        self,
        bilinear_forms: list[mfem.BilinearForm],
        linear_forms: list[mfem.LinearForm],
        out: list[mfem.Vector] | None = None,
        *,
        angle_range: range | int | None = None,
    ) -> list[mfem.Vector]:
        """
        Sweep the mesh for the angles indexed by ``angle_range`` (default: all angles),
        using the orderings determined in ``__init__``. ``bilinear_forms[k]`` and
        ``linear_forms[k]`` are the pre-assembled forms for angle ``k``, so that their
        data can be extracted.

        If ``out`` is given, it must hold one vector of size ``ndofs`` per index in
        ``angle_range``; each is overwritten in place with the solution for that angle.
        Otherwise new vectors are allocated.

        Returns ``out`` (or the newly allocated list), one solution per index in
        ``angle_range``, in the order of the range.
        """
        num_angles = len(self.order)
        if angle_range is None:
            angle_range = range(num_angles)

        if isinstance(angle_range, int):
            m = angle_range
            angle_range = range(m, m + 1)

        for k in angle_range:
            if not 0 <= k < num_angles:
                raise IndexError(
                    f"Angle index {k} out of range for {num_angles} angles"
                )
        if len(bilinear_forms) != len(angle_range) or len(linear_forms) != len(
            angle_range
        ):
            raise ValueError(
                f"Expected one bilinear and linear form per angle ({len(angle_range)})"
            )

        ndofs = bilinear_forms[0].FESpace().GetNDofs() if len(angle_range) else 0

        if out is None:
            out = [mfem.Vector(ndofs) for _ in angle_range]
        else:
            if len(out) != len(angle_range):
                raise ValueError(
                    f"Expected one output vector per angle ({len(angle_range)})"
                )
            for i, v in enumerate(out):
                if v.Size() != ndofs:
                    raise ValueError(
                        f"Output vector {i} has size {v.Size()}, expected {ndofs}"
                    )

        for m in bilinear_forms:
            m.Finalize()
        for i, k in enumerate(angle_range):
            self._sweep_angle(bilinear_forms[i], linear_forms[i], self.order[k], out[i])

        return out

    def _sweep_angle(
        self,
        bilinear_form: mfem.BilinearForm,
        linear_form: mfem.LinearForm,
        order: np.array,
        solution: mfem.Vector,
    ) -> mfem.Vector:
        """
        Sweep the mesh in the element ``order`` for a single angle. In each element,
        the dense local system is solved directly.
        """

        # prepare data to pass to kernel
        M = bilinear_form.SpMat()
        I = M.GetIArray()
        J = M.GetJArray()
        D = M.GetDataArray()
        b = linear_form.GetDataArray()

        # end data preparation

        sweep_kernel(
            order,
            I,
            J,
            D,
            b,
            self._elem_dof_offsets,
            self._elem_dof_indices,
            solution.GetDataArray()[:],
        )

        return solution
