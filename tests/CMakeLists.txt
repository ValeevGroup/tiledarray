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
set(ta_test_src_files  ta_test.cpp                                                  
    array.cpp                  bitset.cpp                      blocked_pmap.cpp
    cyclic_pmap.cpp            dense_shape.cpp                 dist_eval_array_eval.cpp
    dist_eval_binary_eval.cpp  dist_eval_contraction_eval.cpp  dist_eval_unary_eval.cpp
    dist_op_communicator.cpp   dist_op_dist_cache.cpp          dist_op_group.cpp         
    distributed_storage.cpp    eigen.cpp                       elemental.cpp             
    hash_pmap.cpp              math_blas.cpp                   math_outer.cpp            
    math_partial_reduce.cpp    math_transpose.cpp              perm_index.cpp            
    permutation.cpp            proc_grid.cpp                   range.cpp
    range_iterator.cpp         reduce_task.cpp                 replicated_pmap.cpp
    sparse_shape.cpp           tensor.cpp                      tensor_impl.cpp
    tile_op_add.cpp            tile_op_contract_reduce.cpp     tile_op_mult.cpp
    tile_op_neg.cpp            tile_op_noop.cpp                tile_op_scal.cpp
    tile_op_scal_add.cpp       tile_op_scal_mult.cpp           tile_op_scal_subt.cpp
    tile_op_subt.cpp           tiled_range.cpp                 tiled_range1.cpp
    transform_iterator.cpp     utility.cpp                     variable_list.cpp)
        
if(NOT ENABLE_ELEMENTAL)
    list(APPEND ta_test_src_files elemental.cpp)
endif()
add_executable(${executable} EXCLUDE_FROM_ALL ${ta_test_src_files})

# Add include directories and compiler flags for ta_test
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Boost_INCLUDE_DIRS})
set_target_properties(${executable} PROPERTIES
    COMPILE_DEFINITIONS "TILEDARRAY_NO_USER_ERROR_MESSAGES=1")
target_link_libraries(${executable} ${TiledArray_LIBRARIES})

# Add targets
add_dependencies(${executable} External)
add_dependencies(check ${executable})

# Add a test
add_test(${executable} ${executable})
