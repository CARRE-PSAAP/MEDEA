#include "kernels.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace medea {

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
