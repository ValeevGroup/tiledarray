# Create the blas, eigen, ta_band, ta_dense, and ta_sparse executable

# Add include directories
include_directories(${TiledArray_INCLUDE_DIRS})

# Add the blas executable
add_executable(blas EXCLUDE_FROM_ALL blas.cpp)
set_target_properties(blas PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${TiledArray_COMPILE_FLAGS}"
    LINK_FLAGS "${TiledArray_LINK_FLAGS}")
target_link_libraries(blas ${TiledArray_LIBRARIES})
add_dependencies(blas External)
add_dependencies(example blas)

# Add the eigen executable
add_executable(eigen EXCLUDE_FROM_ALL eigen.cpp)
set_target_properties(eigen PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${TiledArray_COMPILE_FLAGS}"
    LINK_FLAGS "${TiledArray_LINK_FLAGS}")
target_link_libraries(eigen ${TiledArray_LIBRARIES})
add_dependencies(eigen External)
add_dependencies(example eigen)

# Add the blas executable
add_executable(ta_band EXCLUDE_FROM_ALL ta_band.cpp)
set_target_properties(ta_band PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${TiledArray_COMPILE_FLAGS}"
    LINK_FLAGS "${TiledArray_LINK_FLAGS}")
target_link_libraries(ta_band ${TiledArray_LIBRARIES})
add_dependencies(ta_band External)
add_dependencies(example ta_band)

# Add the ta_dense executable
add_executable(ta_dense EXCLUDE_FROM_ALL ta_dense.cpp)
set_target_properties(ta_dense PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${TiledArray_COMPILE_FLAGS}"
    LINK_FLAGS "${TiledArray_LINK_FLAGS}")
target_link_libraries(ta_dense ${TiledArray_LIBRARIES})
add_dependencies(ta_dense External)
add_dependencies(example ta_dense)

# Add the blas executable
add_executable(ta_sparse EXCLUDE_FROM_ALL ta_sparse.cpp)
set_target_properties(ta_sparse PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${TiledArray_COMPILE_FLAGS}"
    LINK_FLAGS "${TiledArray_LINK_FLAGS}")
target_link_libraries(ta_sparse ${TiledArray_LIBRARIES})
add_dependencies(ta_sparse External)
add_dependencies(example ta_sparse)

