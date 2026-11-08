#include <pybind11/pybind11.h>

#include "hello_mfem.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "MEDEA compiled core (links against the MFEM shipped with PyMFEM)";
  m.attr("MFEM_MODE") = MEDEA_MFEM_MODE;

  py::class_<medea::HelloResult>(m, "HelloResult")
      .def_readonly("mfem_version", &medea::HelloResult::mfem_version)
      .def_readonly("num_elements", &medea::HelloResult::num_elements)
      .def_readonly("num_vertices", &medea::HelloResult::num_vertices)
      .def_readonly("num_dofs", &medea::HelloResult::num_dofs)
      .def_readonly("integral", &medea::HelloResult::integral);

  m.def("hello_mfem", &medea::hello_mfem, py::arg("n") = 4,
        py::arg("order") = 2,
        "Build an n x n unit-square mesh with an H1 space and integrate f=1.");
}
