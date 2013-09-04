# Create the vector executable

# Add include directories
include_directories(${TiledArray_INCLUDE_DIRS})

# Add the vector executable
add_executable(vector EXCLUDE_FROM_ALL vector.cpp)
set_target_properties(vector PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${TiledArray_COMPILE_FLAGS}"
    LINK_FLAGS "${TiledArray_LINK_FLAGS}")
target_link_libraries(vector ${TiledArray_LIBRARIES})
add_dependencies(vector External)
add_dependencies(example vector)
