# Build meshbrane library and executables
file(GLOB MESHBRANE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
set(MESHBRANE_SOURCES
    "${MESHBRANE_SOURCES}"
    PARENT_SCOPE
)

if (BUILD_RIGID_SPINDLE_SIM)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/rigid_spindle_sim")
endif()
if (BUILD_PARAM_SWEEP)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/param_sweep")
endif()
