
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h
)

set(executable ta_test)
add_executable(   ${executable}
                  ta_test.cpp
                  permutation.cpp
                  coordinates.cpp
                  range_iterator.cpp
                  range.cpp
                  transform_iterator.cpp
                  dense_storage.cpp
                  tensor.cpp
                  bitset.cpp
                  shape.cpp
                  tiled_range1.cpp
                  tiled_range.cpp
                  indexed_iterator.cpp
                  blocked_pmap.cpp
                  cyclic_pmap.cpp
                  hash_pmap.cpp
                  distributed_storage.cpp
                  tensor_impl.cpp
                  array.cpp
                  eigen.cpp
                  variable_list.cpp
                  tile_op_permute.cpp
                  tile_op_add.cpp
                  tile_op_scal_add.cpp
                  tile_op_subt.cpp
                  tile_op_scal_subt.cpp
                  tile_op_mult.cpp
                  tile_op_scal_mult.cpp
                  tile_op_neg.cpp
                  tile_op_scal.cpp
                  tile_op_noop.cpp
                  tensor_expression_impl.cpp
                  annotated_tensor.cpp
                  unary_tensor.cpp
                  binary_tensor.cpp
                  reduce_task.cpp
                  contraction_tensor_impl.cpp
                  vspgemm.cpp
                  summa.cpp
                  contraction_tensor.cpp
                  expressions.cpp
)

include_directories(${MADNESS_INCLUDE_DIR})

target_link_libraries(
  ${executable}
  ${LINK_LIBRARIES})
