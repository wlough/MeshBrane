include(FetchContent)




if(MESHBRANE_GET_OWN_EIGEN)
  # ############################################################################
  # Eigen
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



# ##############################################################################
# Pybind11
# ##############################################################################
if(M3DG_GET_OWN_PYBIND11)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.4
    GIT_SHALLOW TRUE
    SYSTEM SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pybind11-src" BINARY_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/pybind11-build"
  )
  FetchContent_MakeAvailable(pybind11)
else()
  find_package(pybind11 REQUIRED CONFIG)
endif()