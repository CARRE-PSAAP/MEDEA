"""Pure-Python outlines of compiled kernels.

Slow and simple on purpose: the tests check each compiled kernel against its
reference here, so a faster rewrite can be trusted.
"""

import numpy as np
from numba import njit


def sweep(
    order: np.ndarray[np.int32],
    n_dofs: int,
    blf_csr_i: np.ndarray[np.int32],
    blf_csr_j: np.ndarray[np.int32],
    blf_csr_data: np.ndarray[np.float64],
    linear_form: np.ndarray[np.float64],
    elem_dof_offset: np.ndarray[np.int32],
    elem_dof_indices: np.ndarray[np.int32],
    out: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:

    # for each element
    for e in order:
        start = elem_dof_offset[e]
        end = elem_dof_offset[e + 1]
        dofs = elem_dof_indices[start:end]

        local_matrix = np.zeros((dofs.size, dofs.size), dtype=np.float64)
        local_rhs = linear_form[dofs]
        
        # for each local dof (row in bilinear form)
        for i, d in enumerate(dofs):
            start = blf_csr_i[d]
            end = blf_csr_i[d + 1]
            cols = blf_csr_j[start:end]
            vals = blf_csr_data[start:end]

            # for each value in this row
            for c, v in zip(cols, vals):
                found = False
                
                # check if local
                for j, dp in enumerate(dofs):
                    if c == dp:
                        local_matrix[i, j] = v
                        found = True
                        break
                if found is False:
                    local_rhs[i] -= v * out[c]
        out[dofs] = np.linalg.solve(local_matrix, local_rhs)

