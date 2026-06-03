add_executable(param_sweep "main.cpp")
target_link_libraries(param_sweep meshbrane)
set_target_properties(param_sweep PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
