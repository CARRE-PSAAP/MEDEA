#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels.h"

namespace py = pybind11;

namespace {

using IntArray =
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using RealArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

void require_1d(const py::buffer_info& info, const char* name,
                py::ssize_t length) {
  if (info.ndim != 1 || info.shape[0] != length)
    throw std::invalid_argument(std::string(name) + " must be 1-D of length " +
                                std::to_string(length));
}

RealArray sweep_toy(IntArray order, IntArray up_offsets, IntArray up_cells,
                    RealArray up_weights, RealArray sigma, RealArray source) {
  const py::ssize_t n = order.size();
  const py::buffer_info order_i = order.request();
  const py::buffer_info off_i = up_offsets.request();
  const py::buffer_info cells_i = up_cells.request();
  const py::buffer_info w_i = up_weights.request();
  const py::buffer_info sigma_i = sigma.request();
  const py::buffer_info src_i = source.request();

  require_1d(order_i, "order", n);
  require_1d(off_i, "up_offsets", n + 1);
  require_1d(sigma_i, "sigma", n);
  require_1d(src_i, "source", n);
  if (cells_i.ndim != 1) throw std::invalid_argument("up_cells must be 1-D");
  if (static_cast<const std::int64_t*>(off_i.ptr)[n] != cells_i.shape[0])
    throw std::invalid_argument(
        "up_cells length must equal the last entry of up_offsets");
  require_1d(w_i, "up_weights", cells_i.shape[0]);

  RealArray psi(n);
  double* psi_ptr = psi.mutable_data();
  const auto* order_p = static_cast<const std::int64_t*>(order_i.ptr);
  const auto* off_p = static_cast<const std::int64_t*>(off_i.ptr);
  const auto* cells_p = static_cast<const std::int64_t*>(cells_i.ptr);
  const auto* w_p = static_cast<const double*>(w_i.ptr);
  const auto* sigma_p = static_cast<const double*>(sigma_i.ptr);
  const auto* src_p = static_cast<const double*>(src_i.ptr);
  {
    py::gil_scoped_release release;
    medea::sweep_toy(n, order_p, off_p, cells_p, w_p, sigma_p, src_p, psi_ptr);
  }
  return psi;
}

}  // namespace

PYBIND11_MODULE(_kernels, m) {
  m.doc() =
      "MEDEA compiled kernels. Plain NumPy arrays in and out; no MFEM or MPI.";
  m.def("sweep_toy", &sweep_toy, py::arg("order"), py::arg("up_offsets"),
        py::arg("up_cells"), py::arg("up_weights"), py::arg("sigma"),
        py::arg("source"),
        "Placeholder upwind sweep over a CSR cell graph. Returns psi.");
}
