#include "kernels.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace medea {

using SpMatCSR = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
void sweep_kernel(std::int64_t n_elements, std::int64_t n_dofs,
                  const std::int32_t* order, Eigen::Map<const SpMatCSR> mat,
                  const double* linear_form,
                  const std::int32_t* elem_dof_offset,
                  const std::int32_t* elem_dof_indices, double* solution) {
  using Eigen::MatrixXd;
  using Eigen::Ref;
  using Eigen::VectorXd;
  using SpMatMap = Eigen::Map<const SpMatCSR>;

  Eigen::Map<const VectorXd> source(linear_form, n_dofs);
  Eigen::Map<VectorXd> soln(solution, n_dofs);

  // Allocate local matrix and source vector
  int max_local_ndof = 0;
  for (std::int64_t e = 0; e < n_elements; ++e)
    max_local_ndof =
        std::max(max_local_ndof, elem_dof_offset[e + 1] - elem_dof_offset[e]);

  MatrixXd A_buf(max_local_ndof, max_local_ndof);
  VectorXd b_buf(max_local_ndof), x_buf(max_local_ndof);

  // translate local dof index to local index
  // dof_to_local[index] = -1 means not in this element
  // dof_to_local[index] = 0,1,... means local index 0,1....
  std::vector<int> dof_to_local(n_dofs, -1);

  for (std::int64_t k = 0; k < n_elements; ++k) {
    const int el = order[k];
    const int* dofs = elem_dof_indices + elem_dof_offset[el];
    const int nlocal = elem_dof_offset[el + 1] - elem_dof_offset[el];

    for (int i = 0; i < nlocal; ++i) dof_to_local[dofs[i]] = i;

    auto A = A_buf.topLeftCorner(nlocal, nlocal);
    auto b = b_buf.head(nlocal);
    auto x = x_buf.head(nlocal);
    A.setZero();

    // for each local row
    for (int i = 0; i < nlocal; ++i) {
      const int d = dofs[i];
      b(i) = source[d];

      // for each column
      for (SpMatMap::InnerIterator it(mat, d); it; ++it) {
        const int j = dof_to_local[it.col()];
        if (j >= 0)
          // dof is in this element, put it in the local matrix
          A(i, j) = it.value();
        else
          // dof is neighboring, add to source vector
          b(i) -= it.value() * soln[it.col()];
      }
    }

    Eigen::PartialPivLU<Ref<MatrixXd>> lu(A);
    x = lu.solve(b);

    for (int i = 0; i < nlocal; ++i) {
      soln[dofs[i]] = x(i);
      dof_to_local[dofs[i]] = -1;
    }
  }
}

void sweep_toy(std::int64_t n_cells, const std::int64_t* order,
               const std::int64_t* up_offsets, const std::int64_t* up_cells,
               const double* up_weights, const double* sigma,
               const double* source, double* psi) {
  // Validate everything first, so the sweep loop below can index freely.
  std::vector<char> seen(n_cells, 0);
  for (std::int64_t i = 0; i < n_cells; ++i) {
    const std::int64_t c = order[i];
    if (c < 0 || c >= n_cells || seen[c])
      throw std::invalid_argument("order must be a permutation of 0..n-1");
    seen[c] = 1;
    if (sigma[c] == 0.0)
      throw std::invalid_argument("sigma must be nonzero (cell " +
                                  std::to_string(c) + ")");
  }
  if (up_offsets[0] != 0)
    throw std::invalid_argument("up_offsets[0] must be 0");
  for (std::int64_t c = 0; c < n_cells; ++c) {
    if (up_offsets[c + 1] < up_offsets[c])
      throw std::invalid_argument("up_offsets must be non-decreasing");
    for (std::int64_t k = up_offsets[c]; k < up_offsets[c + 1]; ++k)
      if (up_cells[k] < 0 || up_cells[k] >= n_cells)
        throw std::invalid_argument("up_cells index out of range");
  }

  std::vector<char> done(n_cells, 0);
  for (std::int64_t i = 0; i < n_cells; ++i) {
    const std::int64_t c = order[i];
    double acc = source[c];
    for (std::int64_t k = up_offsets[c]; k < up_offsets[c + 1]; ++k) {
      if (!done[up_cells[k]])
        throw std::invalid_argument(
            "upstream cell " + std::to_string(up_cells[k]) +
            " is not before cell " + std::to_string(c) + " in order");
      acc += up_weights[k] * psi[up_cells[k]];
    }
    psi[c] = acc / sigma[c];
    done[c] = 1;
  }
}

}  // namespace medea
