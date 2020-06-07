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
set(TILEDARRAY_HEADER_FILES
tiledarray.h
tiledarray_fwd.h
TiledArray/config.h
TiledArray/array_impl.h
TiledArray/bitset.h
TiledArray/block_range.h
TiledArray/dense_shape.h
TiledArray/dist_array.h
TiledArray/distributed_storage.h
TiledArray/error.h
TiledArray/external/madness.h
TiledArray/initialize.h
TiledArray/perm_index.h
TiledArray/permutation.h
TiledArray/proc_grid.h
TiledArray/range.h
TiledArray/range_iterator.h
TiledArray/reduce_task.h
TiledArray/replicator.h
TiledArray/shape.h
TiledArray/size_array.h
TiledArray/sparse_shape.h
TiledArray/tensor.h
TiledArray/tensor_impl.h
TiledArray/tile.h
TiledArray/tiled_range.h
TiledArray/tiled_range1.h
TiledArray/transform_iterator.h
TiledArray/type_traits.h
TiledArray/utility.h
TiledArray/val_array.h
TiledArray/version.h
TiledArray/zero_tensor.h
TiledArray/algebra/conjgrad.h
TiledArray/algebra/diis.h
TiledArray/algebra/utils.h
TiledArray/conversions/btas.h
TiledArray/conversions/clone.h
TiledArray/conversions/dense_to_sparse.h
TiledArray/conversions/eigen.h
TiledArray/conversions/foreach.h
TiledArray/conversions/vector_of_arrays.h
TiledArray/conversions/make_array.h
TiledArray/conversions/sparse_to_dense.h
TiledArray/conversions/scalapack.h
TiledArray/conversions/to_new_tile_type.h
TiledArray/conversions/truncate.h
TiledArray/dist_eval/array_eval.h
TiledArray/dist_eval/binary_eval.h
TiledArray/dist_eval/contraction_eval.h
TiledArray/dist_eval/dist_eval.h
TiledArray/dist_eval/unary_eval.h
TiledArray/expressions/add_engine.h
TiledArray/expressions/add_expr.h
TiledArray/expressions/binary_engine.h
TiledArray/expressions/binary_expr.h
TiledArray/expressions/blk_tsr_engine.h
TiledArray/expressions/blk_tsr_expr.h
TiledArray/expressions/cont_engine.h
TiledArray/expressions/contraction_helpers.h
TiledArray/expressions/expr.h
TiledArray/expressions/expr_engine.h
TiledArray/expressions/expr_trace.h
TiledArray/expressions/leaf_engine.h
TiledArray/expressions/mult_engine.h
TiledArray/expressions/mult_expr.h
TiledArray/expressions/scal_engine.h
TiledArray/expressions/scal_expr.h
TiledArray/expressions/scal_tsr_engine.h
TiledArray/expressions/scal_tsr_expr.h
TiledArray/expressions/subt_engine.h
TiledArray/expressions/subt_expr.h
TiledArray/expressions/tsr_engine.h
TiledArray/expressions/tsr_expr.h
TiledArray/expressions/unary_engine.h
TiledArray/expressions/unary_expr.h
TiledArray/expressions/variable_list.h
TiledArray/external/btas.h
TiledArray/math/blas.h
TiledArray/math/eigen.h
TiledArray/math/gemm_helper.h
TiledArray/math/outer.h
TiledArray/math/parallel_gemm.h
TiledArray/math/partial_reduce.h
TiledArray/math/transpose.h
TiledArray/math/vector_op.h
TiledArray/pmap/blocked_pmap.h
TiledArray/pmap/cyclic_pmap.h
TiledArray/pmap/hash_pmap.h
TiledArray/pmap/pmap.h
TiledArray/pmap/replicated_pmap.h
TiledArray/policies/dense_policy.h
TiledArray/policies/sparse_policy.h
TiledArray/special/diagonal_array.h
TiledArray/symm/irrep.h
TiledArray/symm/permutation.h
TiledArray/symm/permutation_group.h
TiledArray/symm/representation.h
TiledArray/tensor/complex.h
TiledArray/tensor/kernels.h
TiledArray/tensor/operators.h
TiledArray/tensor/permute.h
TiledArray/tensor/shift_wrapper.h
TiledArray/tensor/tensor.h
TiledArray/tensor/tensor_interface.h
TiledArray/tensor/tensor_map.h
TiledArray/tensor/trace.h
TiledArray/tensor/type_traits.h
TiledArray/tensor/utility.h
TiledArray/tile_interface/add.h
TiledArray/tile_interface/cast.h
TiledArray/tile_interface/clone.h
TiledArray/tile_interface/permute.h
TiledArray/tile_interface/scale.h
TiledArray/tile_interface/shift.h
TiledArray/tile_op/add.h
TiledArray/tile_op/binary_reduction.h
TiledArray/tile_op/binary_wrapper.h
TiledArray/tile_op/contract_reduce.h
TiledArray/tile_op/mult.h
TiledArray/tile_op/noop.h
TiledArray/tile_op/reduce_wrapper.h
TiledArray/tile_op/scal.h
TiledArray/tile_op/shift.h
TiledArray/tile_op/subt.h
TiledArray/tile_op/tile_interface.h
TiledArray/tile_op/unary_reduction.h
TiledArray/tile_op/unary_wrapper.h
TiledArray/util/annotation.h
TiledArray/util/backtrace.h
TiledArray/util/bug.h
TiledArray/util/annotation.h
TiledArray/util/initializer_list.h
TiledArray/util/logger.h
TiledArray/util/random.h
TiledArray/util/singleton.h
TiledArray/util/time.h
TiledArray/util/vector.h
)

if(CUDA_FOUND)

  list(APPEND TILEDARRAY_HEADER_FILES
     TiledArray/external/cuda.h
     TiledArray/external/cutt.h
     TiledArray/math/cublas.h
          TiledArray/cuda/btas_cublas.h
          TiledArray/cuda/btas_um_tensor.h
          TiledArray/cuda/cpu_cuda_vector.h
          TiledArray/cuda/cuda_task_fn.h
          TiledArray/cuda/kernel/mult_kernel.h
          TiledArray/cuda/kernel/mult_kernel_impl.h
          TiledArray/cuda/kernel/reduce_kernel.h
          TiledArray/cuda/kernel/reduce_kernel_impl.h
          TiledArray/cuda/platform.h
          TiledArray/cuda/thrust.h
          TiledArray/cuda/um_allocator.h
          TiledArray/cuda/um_storage.h)

endif(CUDA_FOUND)

set(TILEDARRAY_SOURCE_FILES
TiledArray/tensor/tensor.cpp
TiledArray/sparse_shape.cpp
TiledArray/tensor_impl.cpp
TiledArray/array_impl.cpp
TiledArray/dist_array.cpp
TiledArray/util/backtrace.cpp
TiledArray/util/bug.cpp
)

# the list of libraries on which TiledArray depends on, will be cached later
set(_TILEDARRAY_DEPENDENCIES MADworld TiledArray_Eigen BTAS::BTAS)

# TODO better ways to handle tiledarray cuda dependency
if(CUDA_FOUND)

  list(APPEND TILEDARRAY_SOURCE_FILES
          TiledArray/cuda/btas_um_tensor.cpp
          TiledArray/cuda/cpu_cuda_vector.cu
          TiledArray/cuda/kernel/mult_kernel.cu
          TiledArray/cuda/kernel/reduce_kernel.cu
          TiledArray/cuda/um_storage.cu)

  set_source_files_properties(TiledArray/cuda/btas_um_tensor.cpp
          PROPERTIES
          LANGUAGE CUDA)

  # the list of libraries on which TiledArray depends on
  list(APPEND _TILEDARRAY_DEPENDENCIES CUDA::cublas CUDA::nvToolsExt TiledArray_UMPIRE TiledArray_CUTT)

endif(CUDA_FOUND)

if( TARGET TiledArray_SCALAPACK )
  list(APPEND _TILEDARRAY_DEPENDENCIES TiledArray_SCALAPACK)
endif()
list(APPEND _TILEDARRAY_DEPENDENCIES "${LAPACK_LIBRARIES}")

# cache deps as TILEDARRAY_PRIVATE_LINK_LIBRARIES
set(TILEDARRAY_PRIVATE_LINK_LIBRARIES ${_TILEDARRAY_DEPENDENCIES} CACHE STRING "List of libraries on which TiledArray depends on")

# Create the TiledArray librar{y,ies}
add_library(tiledarray ${TILEDARRAY_SOURCE_FILES} ${TILEDARRAY_HEADER_FILES})
if (BUILD_SHARED_LIBS)
  add_library(tiledarray-static STATIC ${TILEDARRAY_SOURCE_FILES} ${TILEDARRAY_HEADER_FILES})
  set_property(TARGET tiledarray-static PROPERTY POSITION_INDEPENDENT_CODE ON)
  set(TILEDARRAY_LIBRARY_TARGET_SUFFIXES "none;-static")
  # shared library interface does not need to include MADworld
  set(private_link_libraries "${TILEDARRAY_PRIVATE_LINK_LIBRARIES}")
  list(REMOVE_ITEM private_link_libraries MADworld)
  set(TILEDARRAY_PRIVATE_LINK_LIBRARIES "${private_link_libraries}" CACHE STRING "List of libraries on which TiledArray depends on" FORCE)
  add_dependencies(tiledarray External-tiledarray)
else()
  set(TILEDARRAY_LIBRARY_TARGET_SUFFIXES "none")
endif()

set(targetnames)
foreach (suffix ${TILEDARRAY_LIBRARY_TARGET_SUFFIXES})
  if (suffix STREQUAL none)
    set(suffix)
  endif ()
  set(targetname tiledarray${suffix})
  list(APPEND targetnames ${targetname})

  target_include_directories(${targetname} INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
      $<INSTALL_INTERFACE:${TILEDARRAY_INSTALL_INCLUDEDIR}>
      )

  target_link_libraries(${targetname} PUBLIC ${TILEDARRAY_PRIVATE_LINK_LIBRARIES})

  if (targetname STREQUAL tiledarray AND BUILD_SHARED_LIBS)
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
      target_link_libraries(tiledarray PRIVATE -Wl,--whole-archive MADworld-static -Wl,--no-whole-archive)
    elseif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
      target_link_libraries(tiledarray PRIVATE -Wl,-all_load MADworld-static -Wl,-noall_load)
    else ()
      target_link_libraries(tiledarray PRIVATE MADworld-static)
    endif ()
    # propagate dependencies of MADworld-static to the interface
    include(CloneTarget)
    clone_imported_target(MADworld-interface MADworld-static)
    target_link_libraries(tiledarray INTERFACE MADworld-interface)
  elseif (targetname STREQUAL tiledarray-static)
    target_link_libraries(${targetname} PUBLIC MADworld-static)
  endif ()

  if (TARGET build-madness)
    add_dependencies(${targetname} build-madness)
  endif ()
  if (TARGET eigen3)
    add_dependencies(${targetname} eigen3)
  endif ()
  if (TARGET btas)
    add_dependencies(${targetname} btas)
  endif ()
  if (TARGET cutt)
    add_dependencies(${targetname} cutt)
  endif ()
  if (TARGET Umpire)
    add_dependencies(${targetname} Umpire)
  endif ()
  # append current CMAKE_CXX_FLAGS
  string(REPLACE " " ";" CMAKE_CXX_FLAG_LIST "${CMAKE_CXX_FLAGS}")
  target_compile_options(${targetname} PUBLIC ${CMAKE_CXX_FLAG_LIST})
  target_compile_features(${targetname} PUBLIC "cxx_std_${CMAKE_CXX_STANDARD}")
  target_include_directories(${targetname} PUBLIC ${Boost_INCLUDE_DIRS})

  if (CUDA_FOUND)
    target_include_directories(${targetname} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_features(tiledarray PUBLIC "cuda_std_${CMAKE_CUDA_STANDARD}")
  endif (CUDA_FOUND)

  if (LAPACK_INCLUDE_DIRS)
    target_include_directories(${targetname} PUBLIC ${LAPACK_INCLUDE_DIRS})
  endif (LAPACK_INCLUDE_DIRS)
  if (LAPACK_COMPILE_OPTIONS)
    target_compile_options(${targetname} PUBLIC ${LAPACK_COMPILE_OPTIONS})
  endif (LAPACK_COMPILE_OPTIONS)
  if (LAPACK_COMPILE_DEFINITIONS)
    target_compile_definitions(${targetname} PUBLIC ${LAPACK_COMPILE_DEFINITIONS})
  endif (LAPACK_COMPILE_DEFINITIONS)

endforeach (suffix ${TILEDARRAY_LIBRARY_TARGET_SUFFIXES})

# Add library to the list of installed components
install(TARGETS ${targetnames} EXPORT tiledarray COMPONENT tiledarray
    LIBRARY DESTINATION "${TILEDARRAY_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${TILEDARRAY_INSTALL_LIBDIR}")
# Install header files
install(
  DIRECTORY
      ${PROJECT_SOURCE_DIR}/src/
      ${PROJECT_BINARY_DIR}/src/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
  PATTERN "CMakeFiles" EXCLUDE
)

    
