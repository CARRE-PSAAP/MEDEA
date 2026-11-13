"""
Transport sweep scheduling and kernel.
"""

import numpy as np
import warnings

from medea._backend import mfem


class Schedule:
    """
    mesh sweep object.
    Schedule.__init__ calls Kahn kernel to create sweep ordering. create one per angle
    Schedule.sweep() traverses mesh in the order determined at __init__
    """

    def __init__(self, mesh: mfem.Mesh, angle: mfem.vector):
        # ensure angle and mesh dimension match
        sdim = mesh.SpaceDimension()

        is_curved = mesh.NURBSext is not None
        nodal_fes = mesh.GetNodalFESpace()
        if nodal_fes is not None and  nodal_fes.FEColl().GetOrder() > 1:
            is_curved = True

        if is_curved == True:
            warnings.warn("Curved meshes not verified", UserWarning)
    
        if angle.Size() != sdim:
            raise ValueError(
                f"Invalid dimensions in 'angle' for mesh of dimension {sdim}"
            )

        face_table = mesh.GetFaceToElementTable()
        num_faces = mesh.GetNumFaces()

        children = [[] for _ in range(mesh.GetNE())]
        in_degree = [0 for _ in range(mesh.GetNE())]

        # children[element] lists all elements that depend on this one
        # parents[element] lists all elements which this one depends on

        for face_id in range(num_faces):
            num_elements = face_table.RowSize(face_id)
            if num_elements > 1:
                # assume elems is in ascending order
                elems = face_table.GetRowList(face_id)

                face = mesh.GetFaceElementTransformations(face_id)
                ip = mfem.Geometries.GetCenter(face.GetGeometryType())
                face.SetIntPoint(ip)
                nor = mfem.Vector(sdim)
                mfem.CalcOrtho(face.Jacobian(), nor)

                direction = nor * angle
                parent = elems[0] if direction > 0 else elems[1]
                child = elems[1] if direction > 0 else elems[0]

                children[parent].append(child)
                in_degree[child] += 1

        queue = [i for i, x in enumerate(in_degree) if x == 0]
        result = []

        while len(queue) != 0:
            ind = queue.pop(0)
            for c in children[ind]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
            result.append(ind)
        if len(result) != mesh.GetNE():
            raise ValueError("Cycle detected in mesh")
        self.order = result

    def sweep(self, bilinear_form: mfem.BilinearForm, linear_form: mfem.LinearForm):
        # expects pre-built bilinear form matrix and source matrix
        # writes to 'solution'

        fespace = bilinear_form.FESpace()
        element_dofs = []

        # # For each element in schedule order:
        for elem in self.order:
            dofs = fespace.GetElementDofs(elem)
            element_dofs.append(dofs)
            # Get dofs; add to list of dofs

        solution = np.zeros(fespace.GetNDofs())

        M = bilinear_form.SpMat()
        for dofs in element_dofs:
            cols = mfem.intArray()
            vals = mfem.Vector()

            local_matrix = [[0.0 for _ in range(len(dofs))] for _ in range(len(dofs))]
            incoming_source = [0.0 for _ in range(len(dofs))]

            for index, d in enumerate(dofs):
                # get rows of bilinear form
                M.GetRow(d, cols, vals)
                listcols = cols.GetDataArray()
                listvals = vals.GetDataArray()

                for i in range(len(listcols)):
                    if listcols[i] in dofs:
                        local_matrix[index][dofs.index(listcols[i])] = listvals[i]
                    else:
                        incoming_source[index] -= listvals[i] * solution[listcols[i]]

            a = np.array(linear_form.GetDataArray()[dofs]) + np.array(incoming_source)
            u = np.linalg.solve(np.array(local_matrix), a)
            solution[dofs] = u
        return mfem.Vector(solution)
