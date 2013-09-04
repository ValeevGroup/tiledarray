# Create the pmap executable

# Add include directories
include_directories(${TiledArray_INCLUDE_DIRS})

# Add the pmap executable
add_executable(pmap EXCLUDE_FROM_ALL pmap.cpp)
set_target_properties(pmap PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${TiledArray_COMPILE_FLAGS}"
    LINK_FLAGS "${TiledArray_LINK_FLAGS}")
target_link_libraries(pmap ${TiledArray_LIBRARIES})
add_dependencies(pmap External)
add_dependencies(example pmap)
