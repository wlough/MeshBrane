add_executable(envelope_test_sim "main.cpp")
target_link_libraries(envelope_test_sim meshbrane)
set_target_properties(envelope_test_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
