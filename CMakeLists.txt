cmake_minimum_required(VERSION 3.18)
project(medea LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- MFEM flavour: "ser" or "par" (matches PyMFEM's mfem.ser / mfem.par) ----
set(MEDEA_MFEM_MODE "ser" CACHE STRING "PyMFEM build to link against: ser or par")
if(DEFINED ENV{MEDEA_MFEM_MODE} AND NOT "$ENV{MEDEA_MFEM_MODE}" STREQUAL "")
  set(MEDEA_MFEM_MODE "$ENV{MEDEA_MFEM_MODE}")
endif()
set_property(CACHE MEDEA_MFEM_MODE PROPERTY STRINGS ser par)
if(NOT MEDEA_MFEM_MODE MATCHES "^(ser|par)$")
  message(FATAL_ERROR "MEDEA_MFEM_MODE must be 'ser' or 'par', got '${MEDEA_MFEM_MODE}'")
endif()

# --- Locate the MFEM shipped inside the installed PyMFEM ---------------------
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
set(MEDEA_MFEM_ROOT "" CACHE PATH "Override: directory containing include/ and lib/ of MFEM")

if(NOT MEDEA_MFEM_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import importlib.util, os, sys; s = importlib.util.find_spec('mfem'); print(os.path.dirname(s.origin)) if s else sys.exit(1)"
    OUTPUT_VARIABLE _mfem_pkg OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR
      "PyMFEM ('mfem') is not importable by ${Python_EXECUTABLE}. Install it first "
      "and build with --no-build-isolation, or set -DMEDEA_MFEM_ROOT=<path>.")
  endif()
  set(MEDEA_MFEM_ROOT "${_mfem_pkg}/external/${MEDEA_MFEM_MODE}")
endif()

if(NOT EXISTS "${MEDEA_MFEM_ROOT}/include/mfem.hpp")
  message(FATAL_ERROR
    "No MFEM '${MEDEA_MFEM_MODE}' build found at ${MEDEA_MFEM_ROOT}.\n"
    "The installed PyMFEM may be the other flavour. For 'par', PyMFEM must be built "
    "from source with --with-parallel (see README).")
endif()
message(STATUS "MEDEA: using MFEM (${MEDEA_MFEM_MODE}) at ${MEDEA_MFEM_ROOT}")

find_library(MEDEA_MFEM_LIB NAMES mfem PATHS "${MEDEA_MFEM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(mfem_imported SHARED IMPORTED)
set_target_properties(mfem_imported PROPERTIES
  IMPORTED_LOCATION "${MEDEA_MFEM_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${MEDEA_MFEM_ROOT}/include")
if(MEDEA_MFEM_MODE STREQUAL "par")
  find_package(MPI REQUIRED COMPONENTS CXX)
  set_property(TARGET mfem_imported APPEND PROPERTY INTERFACE_LINK_LIBRARIES MPI::MPI_CXX)
endif()

# --- Extension module --------------------------------------------------------
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core src/bindings.cpp src/hello_mfem.cpp)
target_link_libraries(_core PRIVATE mfem_imported)
target_compile_definitions(_core PRIVATE MEDEA_MFEM_MODE="${MEDEA_MFEM_MODE}")

# Find PyMFEM's libmfem at runtime instead of baking in a build-machine path
# (the wheel's libmfem lives at <site-packages>/mfem/external/<mode>/lib).
set_target_properties(_core PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "${MEDEA_MFEM_ROOT}/lib")

install(TARGETS _core DESTINATION medea)
