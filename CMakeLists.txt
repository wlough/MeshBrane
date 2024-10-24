# extern: pybind11, Eigen3, tinyply

cmake_minimum_required(VERSION 3.10)
project(MeshBrane VERSION 0.1.0 LANGUAGES CXX)

## Require c++20 and standard libraries (mostly for coroutines)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Directory containing CMake modules 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/.cmake_modules")

# Output directory locations
## Shared libraries (.so files on Unix/Linux, .dylib files on macOS, .dll files on Windows)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
## Static libraries (.a files on Unix/Linux, .lib files on Windows)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
## Executables (.exe files on Windows, no extension on Unix/Linux)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

include(FetchContent)


option(MESHBRANE_GET_OWN_EIGEN "Download own Eigen" ON)
option(MESHBRANE_GET_OWN_TINYPLY "Download own tinyply" ON)

option(BUILD_PYMESHBRANE "Build Python bindings for MeshBrane" ON)
option(MESHBRANE_GET_OWN_MAYAVI "Download own Mayavi" ON)
option(MESHBRANE_GET_OWN_PYBIND11 "Download own pybind11" ON)

set(MESHBRANE_VENV_PATH "" CACHE PATH "Path to Python virtual environment")

  # ############################################################################
  # GET EIGEN
  # ############################################################################
# External dependencies
if(MESHBRANE_GET_OWN_EIGEN)
  # ############################################################################
  # GET EIGEN
  # ############################################################################
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
  )
  FetchContent_GetProperties(eigen)
  if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
    include(EigenChecker)
    eigen3checker(${eigen_SOURCE_DIR} 3.3)
  endif()
  add_library(eigen INTERFACE)
  add_library(Eigen3::Eigen ALIAS eigen)
  target_include_directories(
    eigen SYSTEM INTERFACE $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
                           $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
  )
else()
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()



