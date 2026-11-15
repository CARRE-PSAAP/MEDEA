cmake_minimum_required(VERSION 3.18)
project(medea LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG 5.0.1
    GIT_SHALLOW TRUE
)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(kernels src/bindings.cpp src/kernels.cpp)

target_link_libraries(kernels PRIVATE Eigen3::Eigen)
install(TARGETS kernels DESTINATION medea)
