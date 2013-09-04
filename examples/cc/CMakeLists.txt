# Create the ccd and ccsd executable

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TiledArray_INCLUDE_DIRS})

add_library(inputlib OBJECT input_data.cpp)
add_dependencies(inputlib External)

# Add the ccd executable
add_executable(ccd EXCLUDE_FROM_ALL ccd.cpp $<TARGET_OBJECTS:inputlib>)
set_target_properties(ccd PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${TiledArray_COMPILE_FLAGS}"
    LINK_FLAGS "${TiledArray_LINK_FLAGS}")
target_link_libraries(ccd ${TiledArray_LIBRARIES})
add_dependencies(ccd External)
add_dependencies(example ccd)

# Add the ccd executable
add_executable(ccsd EXCLUDE_FROM_ALL ccsd.cpp $<TARGET_OBJECTS:inputlib>)
set_target_properties(ccsd PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${TiledArray_COMPILE_FLAGS}"
    LINK_FLAGS "${TiledArray_LINK_FLAGS}")
target_link_libraries(ccsd ${TiledArray_LIBRARIES})
add_dependencies(ccsd External)
add_dependencies(example ccsd)
