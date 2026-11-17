#pragma once
#include <Eigen/Sparse>
#include <cstdint>

// Compiled kernels operate on plain arrays only. They must NOT include MFEM or
// MPI headers: that keeps this extension independent of PyMFEM and of the
// serial/parallel choice.

namespace medea {

void k_sweep(
    std::int64_t n_elements, std::int64_t n_dofs, const std::int32_t* order,
    Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, int>> mat,
    const double* linear_form, const std::int32_t* elem_dof_offset,
    const std::int32_t* elem_dof_indices, double* solution);

}  // namespace medea
