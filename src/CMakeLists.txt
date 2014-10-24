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

# library headers
include_directories(${TiledArray_INCLUDE_DIRS})
set(tiledarray_header_files 
    TiledArray/algebra/conjgrad.h             TiledArray/algebra/diis.h
    TiledArray/algebra/utils.h                TiledArray/dist_eval/array_eval.h
    TiledArray/dist_eval/binary_eval.h        TiledArray/dist_eval/contraction_eval.h
    TiledArray/dist_eval/dist_eval.h          TiledArray/dist_eval/unary_eval.h
    TiledArray/distributed_storage.h          TiledArray/expressions/add_engine.h
    TiledArray/expressions/add_expr.h         TiledArray/expressions/binary_engine.h
    TiledArray/expressions/binary_expr.h      TiledArray/expressions/cont_engine.h
    TiledArray/expressions/expr.h             TiledArray/expressions/expr_engine.h
    TiledArray/expressions/expr_trace.h       TiledArray/expressions/leaf_engine.h
    TiledArray/expressions/mult_engine.h      TiledArray/expressions/mult_expr.h
    TiledArray/expressions/scal_engine.h      TiledArray/expressions/scal_expr.h
    TiledArray/expressions/scal_tsr_engine.h  TiledArray/expressions/scal_tsr_expr.h
    TiledArray/expressions/subt_engine.h      TiledArray/expressions/subt_expr.h
    TiledArray/expressions/tsr_engine.h       TiledArray/expressions/tsr_expr.h
    TiledArray/expressions/unary_engine.h     TiledArray/expressions/unary_expr.h
    TiledArray/expressions/variable_list.h    TiledArray/math/blas.h
    TiledArray/math/eigen.h                   TiledArray/math/functional.h
    TiledArray/math/gemm_helper.h             TiledArray/math/math.h
    TiledArray/math/outer.h                   TiledArray/math/partial_reduce.h
    TiledArray/math/transpose.h               TiledArray/math/vector_op.h
    TiledArray/pmap/blocked_pmap.h            TiledArray/pmap/cyclic_pmap.h
    TiledArray/pmap/hash_pmap.h               TiledArray/pmap/pmap.h
    TiledArray/pmap/replicated_pmap.h         TiledArray/policies/dense_policy.h
    TiledArray/policies/sparse_policy.h       TiledArray/tile_op/add.h
    TiledArray/tile_op/binary_interface.h     TiledArray/tile_op/binary_reduction.h
    TiledArray/tile_op/contract_reduce.h      TiledArray/tile_op/mult.h
    TiledArray/tile_op/neg.h                  TiledArray/tile_op/noop.h
    TiledArray/tile_op/reduce_wrapper.h       TiledArray/tile_op/scal.h
    TiledArray/tile_op/scal_add.h             TiledArray/tile_op/scal_mult.h
    TiledArray/tile_op/scal_subt.h            TiledArray/tile_op/subt.h
    TiledArray/tile_op/tile_interface.h       TiledArray/tile_op/type_traits.h
    TiledArray/tile_op/unary_interface.h      TiledArray/tile_op/unary_reduction.h
    TiledArray/array.h         TiledArray/array.h         TiledArray/bitset.h
    TiledArray/counter_probe.h TiledArray/dense_shape.h   TiledArray/eigen.h
    TiledArray/elemental.h     TiledArray/error.h         TiledArray/expressions.h
    TiledArray/madness.h       TiledArray/perm_index.h    TiledArray/permutation.h
    TiledArray/proc_grid.h     TiledArray/range.h         TiledArray/range_iterator.h
    TiledArray/reduce_task.h   TiledArray/replicator.h    TiledArray/shape.h
    TiledArray/size_array.h    TiledArray/sparse_shape.h  TiledArray/tensor.h
    TiledArray/tensor_impl.h   TiledArray/tiled_range.h   TiledArray/tiled_range1.h
    TiledArray/transform_iterator.h TiledArray/type_traits.h  TiledArray/utility.h
    TiledArray/val_array.h     TiledArray/zero_tensor.h   tiledarray.h
    tiledarray_fwd.h)

# Install header files
install(
  DIRECTORY
      ${PROJECT_SOURCE_DIR}/src/
      ${PROJECT_BINARY_DIR}/src/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
  PATTERN "CMakeFiles" EXCLUDE
)

# Create the TiledArray library
#file(GLOB_RECURSE tiledarray_src_files RELATIVE_PATH *.cpp)
#add_library(tiledarray ${tiledarray_src_files} ${tiledarray_header_files})

# Install library
#install(TARGETS tiledarray DESTINATION lib)
