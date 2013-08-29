
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_config_h.in
  ${PROJECT_BINARY_DIR}/tests/unit_test_config.h
)

# Create the ta_test executable
set(executable ta_test)
file(GLOB ta_test_src_files *.cpp)
add_executable(${executable} EXCLUDE_FROM_ALL ${ta_test_src_files})

# Add include directories and compiler flags for ta_test
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/tests
    ${TiledArray_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})
set_target_properties(${executable} PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${TiledArray_COMPILE_FLAGS}"
    COMPILE_DEFINITIONS "TILEDARRAY_NO_USER_ERROR_MESSAGES=1"
    LINK_FLAGS "${TiledArray_LINK_FLAGS}")
target_link_libraries(${executable} ${TiledArray_LIBRARIES} ${Boost_LIBRARIES})

# Add targets
add_dependencies(${executable} External)
add_dependencies(check ${executable})

# Add a test
add_test(${executable} ${executable})
