"""Pure-Python outlines of compiled kernels.

Slow and simple on purpose: the tests check each compiled kernel against its
reference here, so a faster rewrite can be trusted.
"""

import numpy as np


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
