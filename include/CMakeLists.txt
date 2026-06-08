# Set lists of headers for meshbrane
# Set include directories for meshbrane

file(GLOB MESHBRANE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/meshbrane/*.hpp")

set(MESHBRANE_HEADERS
    "${MESHBRANE_HEADERS}"
    PARENT_SCOPE
)

set(MESHBRANE_PROJECT_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
    PARENT_SCOPE
)
