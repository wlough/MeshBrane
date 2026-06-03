# Set lists of headers for meshbrane and meshbrane_graphics
# Set include directories for meshbrane and meshbrane_graphics

file(GLOB MESHBRANE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/meshbrane/*.hpp")

set(MESHBRANE_HEADERS
    "${MESHBRANE_HEADERS}"
    PARENT_SCOPE
)

set(MESHBRANE_PROJECT_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
    PARENT_SCOPE
)



# set(MESHBRANE_HEADERS
#     "${CMAKE_CURRENT_SOURCE_DIR}/meshbrane/simple_generator.hpp"
#     "${CMAKE_CURRENT_SOURCE_DIR}/meshbrane/meshbrane_data_types.hpp"
#     "${CMAKE_CURRENT_SOURCE_DIR}/meshbrane/half_edge_mesh.hpp"
#     "${CMAKE_CURRENT_SOURCE_DIR}/meshbrane/mesh_builder.hpp"
#     PARENT_SCOPE
# )


# set(MESHBRANE_INCLUDE_DIR
#     "${CMAKE_CURRENT_SOURCE_DIR}/meshbrane"
#     PARENT_SCOPE
# )


# file(GLOB MESHBRANE_GRAPHICS_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/meshbrane_graphics/*.hpp")

# set(MESHBRANE_GRAPHICS_HEADERS
#     "${MESHBRANE_GRAPHICS_HEADERS}"
#     PARENT_SCOPE
# )

# set(MESHBRANE_GRAPHICS_INCLUDE_DIR
#     "${CMAKE_CURRENT_SOURCE_DIR}/meshbrane_graphics"
#     PARENT_SCOPE
# )

