cmake_minimum_required(VERSION 3.16)
if(APPLE)
  project(meshbrane VERSION 0.0.0 LANGUAGES C CXX OBJC)
else()
  project(meshbrane VERSION 0.0.0 LANGUAGES C CXX)
endif()

# Require C++20 and standard libraries for coroutines
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Directory containing CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Initialize output directory locations to MeshBrane/build/stuff
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib) # .so, .dylib, .dll
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib) # .a, .lib
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin) # .exe, .out

# Create a version number for the software from the git sha1 of the current commit
# Can be referenced as "@GIT_SHA1@" in a header file from a compiler definition
# include(GetGitRevisionDescription)
# get_git_head_revision(GIT_REFSPEC GIT_SHA1)
#add_definitions(-DSC_VERSION="${GIT_SHA1}")

if(DEBUG OR TRACE)
  set(CMAKE_BUILD_TYPE "Debug")
else()
  set(CMAKE_BUILD_TYPE "Release")
endif()


# These options can be overridden by the user with -D<option>=<value>
option(BUILD_RIGID_SPINDLE_SIM "Build rigid spindle simulation executable" ON)
option(BUILD_PARAM_SWEEP "Build param_sweep executable" ON)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

####################################################################
# External dependencies
####################################################################
find_package(yaml-cpp REQUIRED)
if(TARGET yaml-cpp::yaml-cpp)
    message(STATUS "  yaml-cpp found")
    set(yaml-cpp_FOUND TRUE)
    set(yaml_cpp_TARGETS yaml-cpp::yaml-cpp)
    message(STATUS "  Appending linked libs yaml_cpp_TARGETS: ${yaml_cpp_TARGETS}")
    list(APPEND MESHBRANE_LINKED_EXTERN
    ${yaml_cpp_TARGETS}
)
    message(STATUS "  Setting yaml_cpp_TARGETS: ${yaml_cpp_TARGETS}")
else()
    message(FATAL_ERROR "yaml-cpp not found")
endif()
find_package(Eigen3 REQUIRED NO_MODULE)
if(TARGET Eigen3::Eigen)
    message(STATUS "  Eigen3 found")
    set(Eigen3_FOUND TRUE)
    set(Eigen3_TARGETS Eigen3::Eigen)
    message(STATUS "  Appending linked libs Eigen3_TARGETS: ${Eigen3_TARGETS}")
    list(APPEND MESHBRANE_LINKED_EXTERN ${Eigen3_TARGETS})
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()

add_subdirectory(extern)

if (tinyply_FOUND)
  message(STATUS "  Appending linked libs tinyply_TARGETS: ${tinyply_TARGETS}")
  list(APPEND MESHBRANE_LINKED_EXTERN ${tinyply_TARGETS})
else()
  message(FATAL_ERROR "tinyply not found")
endif()

if (libigl_FOUND)
  message(STATUS "  Appending linked libs libigl_TARGETS: ${libigl_TARGETS}")
  list(APPEND MESHBRANE_LINKED_EXTERN ${libigl_TARGETS})
else()
  message(FATAL_ERROR "libigl not found")
endif()

message(STATUS "  MESHBRANE_LINKED_EXTERN: ${MESHBRANE_LINKED_EXTERN}")

####################################################################
####################################################################


find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(MESHBRANE_MAKE_MOVIE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/make_a_movie.py"
)

find_program(FFMPEG_EXECUTABLE ffmpeg)

if(FFMPEG_EXECUTABLE)
    message(STATUS "Found ffmpeg: ${FFMPEG_EXECUTABLE}")
    set(MESHBRANE_HAS_FFMPEG 1)
    # set(MESHBRANE_FFMPEG_EXECUTABLE "${FFMPEG_EXECUTABLE}")
else()
    message(WARNING "ffmpeg not found. Movie generation will be unavailable.")
    set(MESHBRANE_HAS_FFMPEG 0)
    # set(MESHBRANE_FFMPEG_EXECUTABLE "")
endif()


configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/meshbrane/meshbrane_config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/meshbrane/meshbrane_config.hpp"
    @ONLY
)

####################################################################
####################################################################
add_subdirectory(include)
add_subdirectory(src)

##############################################################################
##############################################################################
add_library(meshbrane_objlib OBJECT ${MESHBRANE_SOURCES} ${MESHBRANE_HEADERS})

target_include_directories(
  meshbrane_objlib
  PUBLIC
  $<BUILD_INTERFACE:${MESHBRANE_PROJECT_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_link_libraries(meshbrane_objlib PUBLIC ${MESHBRANE_LINKED_EXTERN})

set_target_properties(meshbrane_objlib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(meshbrane SHARED $<TARGET_OBJECTS:meshbrane_objlib>)
target_link_libraries(meshbrane PUBLIC meshbrane_objlib)

install(TARGETS meshbrane_objlib
  EXPORT MeshBraneTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS meshbrane
  EXPORT MeshBraneTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/meshbrane)

# Export additional install files (for clients using CMake)
# install(EXPORT MeshBraneTargets
#   NAMESPACE meshbrane::
#   FILE MeshBraneConfig.cmake
#   DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshbrane)


# Allow testing for main and subprojects if -DTESTS=TRUE
if(TESTS)
  add_subdirectory(tests)
endif()
