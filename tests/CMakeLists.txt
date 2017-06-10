#
#  This file is a part of TiledArray.
#  Copyright (C) 2013  Virginia Tech
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  Justus Calvin
#  Department of Chemistry, Virginia Tech
#
#  CMakeLists.txt
#  Jul 19, 2013
#

CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_config_h.in
  ${PROJECT_BINARY_DIR}/tests/unit_test_config.h
)

# Create the ta_test executable
set(executable ta_test)
# NOTE: The order of files here represent the order that tests are run and
# dependencies.
set(ta_test_src_files  ta_test.cpp
    meta.cpp
    type_traits.cpp
    utility.cpp
    permutation.cpp
    symm_permutation_group.cpp
    symm_irrep.cpp
    symm_representation.cpp
    range.cpp
    block_range.cpp
    perm_index.cpp
    transform_iterator.cpp
    bitset.cpp
    math_outer.cpp
    math_partial_reduce.cpp
    math_transpose.cpp
    math_blas.cpp
    tensor.cpp
    tensor_of_tensor.cpp
    tensor_tensor_view.cpp
    tensor_shift_wrapper.cpp
    tiled_range1.cpp
    tiled_range.cpp
    blocked_pmap.cpp
    hash_pmap.cpp
    cyclic_pmap.cpp
    replicated_pmap.cpp
    dense_shape.cpp
    sparse_shape.cpp
    distributed_storage.cpp
    tensor_impl.cpp
    array_impl.cpp
    variable_list.cpp
    dist_array.cpp
    eigen.cpp
    dist_op_dist_cache.cpp
    dist_op_group.cpp
    dist_op_communicator.cpp
    tile_op_noop.cpp
    tile_op_scal.cpp
    dist_eval_array_eval.cpp
    dist_eval_unary_eval.cpp
    tile_op_add.cpp
    tile_op_scal_add.cpp
    tile_op_subt.cpp
    tile_op_scal_subt.cpp
    dist_eval_binary_eval.cpp
    tile_op_mult.cpp
    tile_op_scal_mult.cpp
    tile_op_contract_reduce.cpp
    reduce_task.cpp
    proc_grid.cpp
    dist_eval_contraction_eval.cpp
    expressions.cpp
    expressions_mixed.cpp
    foreach.cpp
)
        
if(ENABLE_ELEMENTAL)
    list(APPEND ta_test_src_files elemental.cpp)
endif()
add_executable(${executable} EXCLUDE_FROM_ALL ${ta_test_src_files})

# Add include directories and compiler flags for ta_test
target_include_directories(${executable} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Boost_INCLUDE_DIRS}
    )
# unit tests
# 1. should disable error messages
# 2. must emit exceptions in asserts since some tests check for exceptions
target_compile_definitions(${executable} PRIVATE TILEDARRAY_NO_USER_ERROR_MESSAGES=1
    TA_EXCEPTION_ERROR=1)
# if tiledarray library was compiled without exceptions, use TA header-only
if (NOT TA_DEFAULT_ERROR EQUAL 1)
  target_compile_definitions(${executable} PRIVATE TILEDARRAY_HEADER_ONLY=1)
  target_link_libraries(${executable} PUBLIC MADworld "${LAPACK_LIBRARIES}")
else()
  target_link_libraries(${executable} PUBLIC tiledarray)
endif()
target_link_libraries(${executable} PUBLIC ${MADNESS_DISABLEPIE_LINKER_FLAG})

# Add targets
add_dependencies(${executable} External)
add_test(build_${executable} "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${executable})

# Add a test(s)
if(ENABLE_MPI)
  add_test(NAME ${executable}-np-1
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${executable}> ${MPIEXEC_POSTFLAGS})
  add_test(NAME ${executable}-np-2
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${executable}> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(${executable}-np-1 PROPERTIES DEPENDS build_${executable})
  set_tests_properties(${executable}-np-2 PROPERTIES DEPENDS build_${executable})
else()
  add_test(NAME ${executable}
      COMMAND ${executable})
  set_tests_properties(${executable} PROPERTIES DEPENDS build_${executable})
endif()
