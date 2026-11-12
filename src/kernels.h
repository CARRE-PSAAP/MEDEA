#pragma once
#include <cstdint>

// Compiled kernels operate on plain arrays only. They must NOT include MFEM or
// MPI headers: that keeps this extension independent of PyMFEM and of the
// serial/parallel choice, so one build serves both. Extract mesh data into
// arrays in Python (see CONTRIBUTING.md) and pass those in.

namespace medea {

/// @brief Placeholder sweep-shaped kernel: one upwind pass over a cell graph.
///
/// Visits the cells in the given order and sets
///   psi[c] = (source[c] + sum_k weight[k] * psi[up[k]]) / sigma[c]
/// where k runs over the upstream neighbours of c (CSR layout). Upstream cells
/// must appear earlier in `order`. Replace with the real sweep.
///
/// @param n_cells    number of cells
/// @param order      sweep order, length n_cells (a permutation of 0..n-1)
/// @param up_offsets CSR row offsets into up_cells/up_weights, length n_cells+1
/// @param up_cells   upstream neighbour cell indices, length up_offsets[n]
/// @param up_weights coupling weight per upstream neighbour, same length
/// @param sigma      per-cell divisor (e.g. total cross section), nonzero
/// @param source     per-cell source
/// @param psi        output, length n_cells
/// @throws std::invalid_argument if an index is out of range or sigma is zero
void sweep_toy(std::int64_t n_cells, const std::int64_t* order,
               const std::int64_t* up_offsets, const std::int64_t* up_cells,
               const double* up_weights, const double* sigma,
               const double* source, double* psi);

}  // namespace medea
