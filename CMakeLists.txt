cmake_minimum_required(VERSION 3.18)
project(medea LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# ---------------------------------------------------------------------------
# medea._kernels: compiled kernels that work on plain arrays.
#
# This extension deliberately does NOT link MFEM or MPI. Mesh data is pulled
# out into NumPy arrays in Python, so one build serves both the serial and
# parallel PyMFEM and does not depend on the PyMFEM version.
# ---------------------------------------------------------------------------
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_kernels src/bindings.cpp src/kernels.cpp)

install(TARGETS _kernels DESTINATION medea)
