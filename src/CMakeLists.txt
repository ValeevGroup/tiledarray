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
TiledArray/fwd.h
${PROJECT_BINARY_DIR}/src/TiledArray/config.h
TiledArray/array_impl.h
TiledArray/bitset.h
TiledArray/block_range.h
TiledArray/dense_shape.h
TiledArray/dist_array.h
TiledArray/distributed_storage.h
TiledArray/error.h
TiledArray/initialize.h
TiledArray/perm_index.h
TiledArray/permutation.h
TiledArray/proc_grid.h
TiledArray/range.h
TiledArray/range1.h
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
${PROJECT_BINARY_DIR}/src/TiledArray/version.h
TiledArray/zero_tensor.h
TiledArray/math/linalg/forward.h
TiledArray/math/linalg/basic.h
TiledArray/math/linalg/util.h
TiledArray/math/linalg/cholesky.h
TiledArray/math/linalg/heig.h
TiledArray/math/linalg/lu.h
TiledArray/math/linalg/qr.h
TiledArray/math/linalg/svd.h
TiledArray/math/linalg/scalapack/util.h
TiledArray/math/linalg/scalapack/block_cyclic.h
TiledArray/math/linalg/scalapack/cholesky.h
TiledArray/math/linalg/scalapack/heig.h
TiledArray/math/linalg/scalapack/lu.h
TiledArray/math/linalg/scalapack/qr.h
TiledArray/math/linalg/scalapack/svd.h
TiledArray/math/linalg/ttg/cholesky.h
TiledArray/math/linalg/ttg/util.h
TiledArray/math/solvers/conjgrad.h
TiledArray/math/solvers/diis.h
TiledArray/math/solvers/cp.h
TiledArray/math/solvers/cp/btas_cp.h
TiledArray/math/solvers/cp/cp.h
TiledArray/math/solvers/cp/cp_als.h
TiledArray/math/solvers/cp/cp_reconstruct.h
TiledArray/conversions/btas.h
TiledArray/conversions/clone.h
TiledArray/conversions/concat.h
TiledArray/conversions/dense_to_sparse.h
TiledArray/conversions/eigen.h
TiledArray/conversions/foreach.h
TiledArray/conversions/vector_of_arrays.h
TiledArray/conversions/make_array.h
TiledArray/conversions/sparse_to_dense.h
TiledArray/conversions/to_new_tile_type.h
TiledArray/conversions/truncate.h
TiledArray/conversions/retile.h
TiledArray/dist_eval/array_eval.h
TiledArray/dist_eval/binary_eval.h
TiledArray/dist_eval/contraction_eval.h
TiledArray/dist_eval/dist_eval.h
TiledArray/dist_eval/unary_eval.h
TiledArray/einsum/index.h
TiledArray/einsum/range.h
TiledArray/einsum/string.h
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
TiledArray/expressions/fwd.h
TiledArray/expressions/leaf_engine.h
TiledArray/expressions/mult_engine.h
TiledArray/expressions/mult_expr.h
TiledArray/expressions/permopt.h
TiledArray/expressions/product.h
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
TiledArray/expressions/index_list.h
TiledArray/external/btas.h
TiledArray/external/madness.h
TiledArray/external/umpire.h
TiledArray/host/env.h
TiledArray/host/allocator.h
TiledArray/math/blas.h
TiledArray/math/gemm_helper.h
TiledArray/math/outer.h
TiledArray/math/parallel_gemm.h
TiledArray/math/partial_reduce.h
TiledArray/math/transpose.h
TiledArray/math/vector_op.h
TiledArray/math/scalapack.h
TiledArray/math/linalg/rank-local.h
TiledArray/pmap/blocked_pmap.h
TiledArray/pmap/cyclic_pmap.h
TiledArray/pmap/hash_pmap.h
TiledArray/pmap/pmap.h
TiledArray/pmap/replicated_pmap.h
TiledArray/pmap/round_robin_pmap.h
TiledArray/pmap/user_pmap.h
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
TiledArray/tensor/type_traits.h
TiledArray/tensor/utility.h
TiledArray/tile_interface/add.h
TiledArray/tile_interface/cast.h
TiledArray/tile_interface/clone.h
TiledArray/tile_interface/permute.h
TiledArray/tile_interface/scale.h
TiledArray/tile_interface/shift.h
TiledArray/tile_interface/trace.h
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
TiledArray/util/function.h
TiledArray/util/initializer_list.h
TiledArray/util/logger.h
TiledArray/util/ptr_registry.h
TiledArray/util/random.h
TiledArray/util/singleton.h
TiledArray/util/threads.h
TiledArray/util/thread_specific.h
TiledArray/util/time.h
TiledArray/util/vector.h
)

if(HIP_FOUND OR CUDA_FOUND)
  list(APPEND TILEDARRAY_HEADER_FILES
          TiledArray/external/device.h
          TiledArray/external/librett.h)
endif()

if(CUDA_FOUND OR HIP_FOUND)
  list(APPEND TILEDARRAY_HEADER_FILES
          TiledArray/device/blas.cpp
          TiledArray/device/blas.h
          TiledArray/device/btas.h
          TiledArray/device/btas_um_tensor.h
          TiledArray/device/device_task_fn.h
          TiledArray/device/kernel/mult_kernel.h
          TiledArray/device/kernel/reduce_kernel.h
          TiledArray/device/kernel/thrust/mult_kernel.h
          TiledArray/device/kernel/thrust/reduce_kernel.h
          TiledArray/device/platform.h
          TiledArray/device/thrust.h
          TiledArray/device/allocators.h
          TiledArray/device/um_storage.h)
  if(CUDA_FOUND)
    list(APPEND TILEDARRAY_HEADER_FILES
       TiledArray/external/cuda.h
       TiledArray/device/cpu_cuda_vector.h)
  endif(CUDA_FOUND)
endif(CUDA_FOUND OR HIP_FOUND)

set(TILEDARRAY_SOURCE_FILES
TiledArray/tiledarray.cpp
TiledArray/tensor/tensor.cpp
TiledArray/sparse_shape.cpp
TiledArray/tensor_impl.cpp
TiledArray/array_impl.cpp
TiledArray/dist_array.cpp
TiledArray/version.cpp
TiledArray/einsum/index.cpp
TiledArray/expressions/permopt.cpp
TiledArray/math/linalg/basic.cpp
TiledArray/math/linalg/rank-local.cpp
TiledArray/util/backtrace.cpp
TiledArray/util/bug.cpp
TiledArray/util/ptr_registry.cpp
TiledArray/util/random.cpp
TiledArray/util/threads.cpp
)
# feed TILEDARRAY_GIT_REVISION and TILEDARRAY_GIT_DESCRIPTION to TiledArray/version.cpp only to avoid recompiling everything
set_source_files_properties(
        TiledArray/version.cpp
        PROPERTIES COMPILE_DEFINITIONS
        "TILEDARRAY_GIT_REVISION=\"${TILEDARRAY_GIT_REVISION}\";TILEDARRAY_GIT_DESCRIPTION=\"${TILEDARRAY_GIT_DESCRIPTION}\""
)

# the list of libraries on which TiledArray depends on, will be cached later
# when FetchContent umpire: set(_TILEDARRAY_DEPENDENCIES MADworld TiledArray_Eigen BTAS::BTAS blaspp_headers umpire)
set(_TILEDARRAY_DEPENDENCIES MADworld TiledArray_Eigen BTAS::BTAS blaspp_headers TiledArray_UMPIRE)

if(CUDA_FOUND OR HIP_FOUND)

  set(TILEDARRAY_DEVICE_SOURCE_FILES
          TiledArray/device/btas_um_tensor.cpp
  )

  if(CUDA_FOUND)

    list(APPEND TILEDARRAY_DEVICE_SOURCE_FILES
          TiledArray/device/cpu_cuda_vector.cu
          TiledArray/device/kernel/thrust/mult_kernel.cu
          TiledArray/device/kernel/thrust/reduce_kernel.cu
          TiledArray/device/um_storage.cu)

    foreach( f IN LISTS TILEDARRAY_DEVICE_SOURCE_FILES )
      set_source_files_properties( "${f}"
          PROPERTIES
          INCLUDE_DIRECTORIES "${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}")
    endforeach()

    # the list of libraries on which TiledArray depends on
    list(APPEND _TILEDARRAY_DEPENDENCIES CUDA::cudart CUDA::cublas CUDA::nvToolsExt)

  endif(CUDA_FOUND)

  if (HIP_FOUND)
    list(APPEND TILEDARRAY_DEVICE_SOURCE_FILES
            TiledArray/device/kernel/thrust/mult_kernel.hip
            TiledArray/device/kernel/thrust/reduce_kernel.hip
    )

    list(APPEND _TILEDARRAY_DEPENDENCIES hip::host
            # N.B. linking to rocthrust makes all files in tiledarray target compiled as HIP ...
            #      seemingly (like with CUDA thrust) linking to this target is not needed
            # roc::rocthrust
    )
  endif()

  # LibreTT needed for either CUDA or HIP
  list(APPEND _TILEDARRAY_DEPENDENCIES TiledArray_LIBRETT)

  list(APPEND TILEDARRAY_SOURCE_FILES "${TILEDARRAY_DEVICE_SOURCE_FILES}")
endif(CUDA_FOUND OR HIP_FOUND)

if( TARGET TiledArray_SCALAPACK )
  list(APPEND _TILEDARRAY_DEPENDENCIES TiledArray_SCALAPACK)
endif()
list(APPEND _TILEDARRAY_DEPENDENCIES "${LAPACK_LIBRARIES}")

if( TARGET ttg-parsec )
  list(APPEND _TILEDARRAY_DEPENDENCIES ttg-parsec)
endif()

if (IntelMKL_FAIR_DISPATCH AND BLAS_IS_MKL)
    message(WARNING "created tiledarray_mkl_dispatch")
    add_library(tiledarray_mkl_dispatch OBJECT
      TiledArray/external/agnerfog/intel_mkl_cpuid_patch.c
      TiledArray/external/agnerfog/intel_mkl_feature_patch.c
    )
    # N.B. --allow-multiple-definition is a GNU linker extension
    list(APPEND _TILEDARRAY_DEPENDENCIES $<TARGET_OBJECTS:tiledarray_mkl_dispatch> -Wl,--allow-multiple-definition)
endif()

# cache deps as TILEDARRAY_PRIVATE_LINK_LIBRARIES
set(TILEDARRAY_PRIVATE_LINK_LIBRARIES ${_TILEDARRAY_DEPENDENCIES} CACHE STRING "List of libraries on which TiledArray depends on")

# Create the TiledArray librar{y,ies}
add_library(tiledarray ${TILEDARRAY_SOURCE_FILES} ${TILEDARRAY_HEADER_FILES})

  set(targetname tiledarray)

  target_include_directories(${targetname} INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
      $<INSTALL_INTERFACE:${TILEDARRAY_INSTALL_INCLUDEDIR}>
      )

  target_link_libraries(${targetname} PUBLIC ${TILEDARRAY_PRIVATE_LINK_LIBRARIES})
  target_link_libraries(${targetname} PUBLIC MADworld)
  target_link_libraries(${targetname} PUBLIC Boost::headers)

  # build all external deps before building tiledarray
  add_dependencies(${targetname} External-tiledarray)

  # append current CMAKE_CXX_FLAGS
  string(REPLACE " " ";" CMAKE_CXX_FLAG_LIST "${CMAKE_CXX_FLAGS}")
  target_compile_options(${targetname} PUBLIC ${CMAKE_CXX_FLAG_LIST})
  target_compile_features(${targetname} PUBLIC "cxx_std_${CMAKE_CXX_STANDARD}")

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

if( TARGET ttg-parsec )
  target_compile_definitions(${targetname} PUBLIC TTG_USE_PARSEC=1)
endif()

# Add library to the list of installed components
install(TARGETS tiledarray EXPORT tiledarray COMPONENT tiledarray
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
