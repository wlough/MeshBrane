# Find external dependencies
# For each external library, set list of targets to be linked to meshbrane library
message(STATUS "Fetching external dependencies")
include(FetchContent)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tinyply EXCLUDE_FROM_ALL SYSTEM)
if(TARGET tinyply)
    message(STATUS "  tinyply found")
    target_include_directories(tinyply INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tinyply/source>)
    set(tinyply_FOUND TRUE PARENT_SCOPE)
    set(tinyply_TARGETS tinyply PARENT_SCOPE)
    # message(STATUS "  Setting tinyply_TARGETS: ${tinyply_TARGETS}")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libigl EXCLUDE_FROM_ALL SYSTEM)
if (TARGET igl::core)
    message(STATUS "  libigl found")
    set(libigl_FOUND TRUE PARENT_SCOPE)
    set(libigl_TARGETS igl_core PARENT_SCOPE)

    igl_include(glfw)
    igl_include(opengl)
    igl_include(stb)
    set(libigl_TARGETS igl_core igl_opengl igl_glfw glad igl_stb PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

endif()
