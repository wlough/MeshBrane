add_executable(rigid_spindle_sim "main.cpp")
target_link_libraries(rigid_spindle_sim meshbrane)
set_target_properties(rigid_spindle_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)