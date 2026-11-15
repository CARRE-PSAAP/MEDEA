"""Pure-Python outlines of compiled kernels.

Slow and simple on purpose: the tests check each compiled kernel against its
reference here, so a faster rewrite can be trusted.
"""

import numpy as np

from numba import njit

@njit
def sweep(
    order: np.ndarray[np.int32],
    n_dofs: int,
    blf_csr_i: np.ndarray[np.int32],
    blf_csr_j: np.ndarray[np.int32],
    blf_csr_data: np.ndarray[np.float64],
    source_array: np.ndarray[np.float64],
    elem_dof_offset: np.ndarray[np.int32],
    elem_dof_indices: np.ndarray[np.int32],
) -> np.ndarray[np.float64]:

    solution = np.zeros(n_dofs, dtype=np.float64)

    for e in order:
        start = elem_dof_offset[e]
        end = elem_dof_offset[e + 1]
        dofs = elem_dof_indices[start:end]

        local_matrix = np.zeros((dofs.size, dofs.size), dtype=np.float64)
        local_rhs = source_array[dofs]

        for i, d in enumerate(dofs):
            start = blf_csr_i[d]
            end = blf_csr_i[d + 1]
            cols = blf_csr_j[start:end]
            vals = blf_csr_data[start:end]

            for c, v in zip(cols, vals):
                found = False
                for j, dp in enumerate(dofs):
                    if c == dp:
                        local_matrix[i, j] = v
                        found = True
                        break
                if found is False:
                    local_rhs[i] -= v * solution[c]
        solution[dofs] = np.linalg.solve(local_matrix, local_rhs)

    return(solution)


def sweep_toy(order, up_offsets, up_cells, up_weights, sigma, source):
    """Reference for :func:`medea._kernels.sweep_toy`."""
    order = np.asarray(order)
    n = len(order)
    psi = np.zeros(n)
    done = np.zeros(n, dtype=bool)
    for c in order:
        acc = source[c]
        for k in range(up_offsets[c], up_offsets[c + 1]):
            u = up_cells[k]
            if not done[u]:
                raise ValueError(f"upstream cell {u} is not before cell {c} in order")
            acc += up_weights[k] * psi[u]
        psi[c] = acc / sigma[c]
        done[c] = True
    return psi
