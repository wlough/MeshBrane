add_executable(chromatin_test_sim "main.cpp")
target_link_libraries(chromatin_test_sim meshbrane)
set_target_properties(chromatin_test_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
