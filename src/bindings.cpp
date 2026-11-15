#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Sparse>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "kernels.h"

namespace py = pybind11;

namespace {


void require_1d(const py::buffer_info& info, const char* name,
                py::ssize_t length) {
  if (info.ndim != 1 || info.shape[0] != length)
    throw std::invalid_argument(std::string(name) + " must be 1-D of length " +
                                std::to_string(length));
}

using IntArray =
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using RealArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

using vDouble = std::vector<double>;
using vInt = std::vector<int>;

void require_1d_any(const py::buffer_info& info, const char* name) {
  if (info.ndim != 1)
    throw std::invalid_argument(std::string(name) + " must be 1-D");
}

// ``out`` is written in place, so it must be the caller's own array: a
// converting array_t argument would silently hand the kernel a copy.
double* checked_out_ptr(const py::object& out_obj, py::ssize_t n_dofs) {
  if (!py::isinstance<py::array>(out_obj))
    throw py::type_error("out must be a numpy array");
  py::array out = py::reinterpret_borrow<py::array>(out_obj);
  if (!py::array_t<double>::check_(out))
    throw std::invalid_argument("out must have dtype float64");
  if (out.ndim() != 1 || out.shape(0) != n_dofs)
    throw std::invalid_argument("out must be 1-D of length " +
                                std::to_string(n_dofs));
  if (!(out.flags() & py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_))
    throw std::invalid_argument("out must be C-contiguous");
  if (!out.writeable()) throw std::invalid_argument("out must be writeable");
  return static_cast<double*>(out.mutable_data());
}

RealArray k_sweep(IntArray order, IntArray blf_csr_i, IntArray blf_csr_j,
                       RealArray blf_csr_data, RealArray linear_form,
                       IntArray elem_dof_offset, IntArray elem_dof_indices,
                       py::object out) {
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

  // ---- O(1) argument checks (contents of the arrays are not validated) ----
  require_1d_any(order_i, "order");
  require_1d_any(blf_csr_i_i, "blf_csr_i");
  require_1d_any(blf_csr_j_i, "blf_csr_j");
  require_1d_any(blf_csr_data_i, "blf_csr_data");
  require_1d_any(linear_form_i, "source_array");
  require_1d_any(elem_dof_offset_i, "elem_dof_offset");
  require_1d_any(elem_dof_indices_i, "elem_dof_indices");

  // Sizes that Eigen's int-indexed sparse Map and the kernel store as int32.
  constexpr py::ssize_t kInt32Max = std::numeric_limits<std::int32_t>::max();
  if (n_dofs >= kInt32Max || n_nonzero > kInt32Max ||
      elem_dof_indices.size() > kInt32Max)
    throw std::invalid_argument(
        "problem size exceeds int32 range (n_dofs, nnz and the number of "
        "element dof entries must fit in int32)");

  // The system is square, so the number of dofs is taken from the linear form.
  require_1d(blf_csr_i_i, "blf_csr_i", n_dofs + 1);
  require_1d(blf_csr_j_i, "blf_csr_j", n_nonzero);

  const auto* csr_i = static_cast<const std::int32_t*>(blf_csr_i_i.ptr);
  if (csr_i[0] != 0) throw std::invalid_argument("blf_csr_i must start at 0");
  if (csr_i[n_dofs] != n_nonzero)
    throw std::invalid_argument(
        "blf_csr_i must end at the number of nonzeros (len(blf_csr_data))");

  if (elem_dof_offset.size() < 1)
    throw std::invalid_argument("elem_dof_offset must not be empty");
  const auto* offs = static_cast<const std::int32_t*>(elem_dof_offset_i.ptr);
  const py::ssize_t total_elements = elem_dof_offset.size() - 1;
  if (offs[0] != 0)
    throw std::invalid_argument("elem_dof_offset must start at 0");
  if (offs[total_elements] != elem_dof_indices.size())
    throw std::invalid_argument(
        "elem_dof_offset must end at len(elem_dof_indices)");
  if (n_elements > total_elements)
    throw std::invalid_argument(
        "order has more entries than there are elements "
        "(len(elem_dof_offset) - 1)");

  double* out_p = checked_out_ptr(out, n_dofs);

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

  medea::k_sweep(n_elements, n_dofs, order_p, mat, linear_form_p,
                      elem_dof_offset_p, elem_dof_indices_p, out_p);
  return RealArray();
}

}  // namespace

PYBIND11_MODULE(kernels, m) {
  m.doc() =
      "MEDEA compiled kernels. These contain plain Numpy arrays and do not "
      "interface with MFEM or MPI";

  m.def("k_sweep", &k_sweep, py::arg("order"), py::arg("blf_csr_i"),
        py::arg("blf_csr_j"), py::arg("blf_csr_data"), py::arg("source_array"),
        py::arg("elem_dof_offset"), py::arg("elem_dof_indices"), py::arg("out"),
        "Solves transport equation by sweeping elements in ``order``, using "
        "the CSR bilinear form matrix ``blf_csr_*`` and linear form ``lf``.");
}
