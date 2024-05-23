cmake_minimum_required(VERSION 3.10) # or whatever minimum version you require
project(MeshBrane) # replace with your project's name

# Add the executable
add_executable(MeshBrane main.cpp brane_utils.cpp)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# If you have include directories, you can specify them like this:
include_directories(/usr/include/eigen3)

# If you have libraries to link, you can specify them like this:
# target_link_libraries(MyProject PRIVATE your_library)