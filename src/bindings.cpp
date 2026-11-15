#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Sparse>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels.h"

namespace py = pybind11;

namespace {

using IntArray2 =
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using RealArray2 =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

void require_1d(const py::buffer_info& info, const char* name,
                py::ssize_t length) {
  if (info.ndim != 1 || info.shape[0] != length)
    throw std::invalid_argument(std::string(name) + " must be 1-D of length " +
                                std::to_string(length));
}

RealArray2 sweep_toy(IntArray2 order, IntArray2 up_offsets, IntArray2 up_cells,
                     RealArray2 up_weights, RealArray2 sigma,
                     RealArray2 source) {
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

  RealArray2 psi(n);
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

using IntArray =
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using RealArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

using vDouble = std::vector<double>;
using vInt = std::vector<int>;

RealArray sweep_kernel(IntArray order, IntArray blf_csr_i, IntArray blf_csr_j,
                       RealArray blf_csr_data, RealArray linear_form,
                       IntArray elem_dof_offset, IntArray elem_dof_indices,
                       py::array_t<double, py::array::c_style> out) {
  py::ssize_t n_elements = order.size();
  const py::buffer_info order_i = order.request();
  const py::buffer_info blf_csr_i_i = blf_csr_i.request();
  const py::buffer_info blf_csr_j_i = blf_csr_j.request();
  const py::buffer_info blf_csr_data_i = blf_csr_data.request();
  py::ssize_t n_nonzero = blf_csr_data.size();
  const py::buffer_info linear_form_i = linear_form.request();
  py::ssize_t n_dofs = linear_form.size();
  const py::buffer_info elem_dof_offset_i = elem_dof_offset.request();
  const py::buffer_info elem_dof_indices_i = elem_dof_indices.request();
  const py::buffer_info out_i = out.request();

  using SpMatCSR = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
  Eigen::Map<const SpMatCSR> mat(n_dofs, n_dofs, n_nonzero, blf_csr_i.data(),
                                 blf_csr_j.data(), blf_csr_data.data());

  const auto* order_p =
      static_cast<const std::int32_t*>(order_i.ptr);  // n_elements big
  const auto* blf_csr_i_p =
      static_cast<const std::int32_t*>(blf_csr_i_i.ptr);  // n_dofs + 1 big
  const auto* blf_csr_j_p = static_cast<const std::int32_t*>(
      blf_csr_j_i.ptr);  // last element of I + 1 big
  const auto* blf_csr_data_p =
      static_cast<const double*>(blf_csr_data_i.ptr);  // " "
  const auto* linear_form_p =
      static_cast<const double*>(linear_form_i.ptr);  // n_dofs big
  const auto* elem_dof_offset_p = static_cast<const std::int32_t*>(
      elem_dof_offset_i.ptr);  // n_elements big
  const auto* elem_dof_indices_p = static_cast<const std::int32_t*>(
      elem_dof_indices_i.ptr);  // n_elements big
  auto* out_p = static_cast<double*>(out_i.ptr);

  medea::sweep_kernel(n_elements, n_dofs, order_p, mat, linear_form_p,
                      elem_dof_offset_p, elem_dof_indices_p, out_p);
  return RealArray();
}

}  // namespace

PYBIND11_MODULE(_kernels, m) {
  m.doc() =
      "MEDEA compiled kernels. Plain NumPy arrays in and out; no MFEM or MPI.";
  m.def("sweep_toy", &sweep_toy, py::arg("order"), py::arg("up_offsets"),
        py::arg("up_cells"), py::arg("up_weights"), py::arg("sigma"),
        py::arg("source"),
        "Placeholder upwind sweep over a CSR cell graph. Returns psi.");

  m.def("sweep_kernel", &sweep_kernel, py::arg("order"), py::arg("blf_csr_i"),
        py::arg("blf_csr_j"), py::arg("blf_csr_data"), py::arg("source_array"),
        py::arg("elem_dof_offset"), py::arg("elem_dof_indices"), py::arg("out"),
        "Solves transport equation by sweeping elements in ``order``, using "
        "the CSR bilinear form matrix ``blf_csr_*`` and linear form ``lf``.");
}
