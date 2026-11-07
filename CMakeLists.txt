cmake_minimum_required(VERSION 3.18)
project(medea LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# ---------------------------------------------------------------------------
# 1. Which PyMFEM build to use: "ser" (serial) or "par" (MPI).
#    Set with the environment variable MEDEA_MFEM_MODE, or with
#    pip's -Ccmake.define.MEDEA_MFEM_MODE=par
# ---------------------------------------------------------------------------
set(MEDEA_MFEM_MODE "ser" CACHE STRING "PyMFEM build to link: ser or par")
if(DEFINED ENV{MEDEA_MFEM_MODE})
  set(MEDEA_MFEM_MODE $ENV{MEDEA_MFEM_MODE})
endif()
if(NOT MEDEA_MFEM_MODE STREQUAL "ser" AND NOT MEDEA_MFEM_MODE STREQUAL "par")
  message(FATAL_ERROR "MEDEA_MFEM_MODE must be 'ser' or 'par'")
endif()

# ---------------------------------------------------------------------------
# 2. Find the MFEM that ships inside the installed PyMFEM package, i.e.
#    <site-packages>/mfem/external/<mode>/{include,lib}.
#    Set MEDEA_MFEM_ROOT to use a different MFEM instead.
# ---------------------------------------------------------------------------
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

set(MEDEA_MFEM_ROOT "" CACHE PATH "Optional: MFEM directory with include/ and lib/")
if(NOT MEDEA_MFEM_ROOT)
  # Ask Python where the 'mfem' package lives (without importing it).
  set(find_mfem "import importlib.util as u; print(u.find_spec('mfem').submodule_search_locations[0])")
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "${find_mfem}"
    OUTPUT_VARIABLE mfem_package
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE python_status)
  if(NOT python_status EQUAL 0)
    message(FATAL_ERROR "PyMFEM is not installed. Install it first and use --no-build-isolation.")
  endif()
  set(MEDEA_MFEM_ROOT ${mfem_package}/external/${MEDEA_MFEM_MODE})
endif()

if(NOT EXISTS ${MEDEA_MFEM_ROOT}/include/mfem.hpp)
  message(FATAL_ERROR "No '${MEDEA_MFEM_MODE}' MFEM found in ${MEDEA_MFEM_ROOT}. "
                      "Parallel needs a PyMFEM built from source (see README).")
endif()
message(STATUS "MEDEA: using ${MEDEA_MFEM_MODE} MFEM from ${MEDEA_MFEM_ROOT}")

set(mfem_include_dirs ${MEDEA_MFEM_ROOT}/include)
set(mfem_lib_dirs     ${MEDEA_MFEM_ROOT}/lib)
set(mfem_libs         mfem)

# The parallel build also needs MPI, plus hypre and METIS, which PyMFEM
# installs one level up, in <mfem>/external/{include,lib}.
if(MEDEA_MFEM_MODE STREQUAL "par")
  find_package(MPI REQUIRED COMPONENTS CXX)
  list(APPEND mfem_include_dirs ${MEDEA_MFEM_ROOT}/../include)
  list(APPEND mfem_lib_dirs     ${MEDEA_MFEM_ROOT}/../lib)
  list(APPEND mfem_libs         HYPRE metis MPI::MPI_CXX)
endif()

# ---------------------------------------------------------------------------
# 3. The Python extension module: medea._core
# ---------------------------------------------------------------------------
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core src/bindings.cpp src/hello_mfem.cpp)

target_include_directories(_core PRIVATE ${mfem_include_dirs})
target_link_directories(_core PRIVATE ${mfem_lib_dirs})
target_link_libraries(_core PRIVATE ${mfem_libs})
target_compile_definitions(_core PRIVATE MEDEA_MFEM_MODE="${MEDEA_MFEM_MODE}")

# Tell the module where to find MFEM's .so files when Python imports it.
set_target_properties(_core PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "${mfem_lib_dirs}")

install(TARGETS _core DESTINATION medea)
