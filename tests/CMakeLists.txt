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
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_config.h.in
  ${PROJECT_BINARY_DIR}/tests/unit_test_config.h
)

# Create the ta_test executable
set(executable ta_test)
# N.B.: The order of files here represents the order in which the tests are run.
# N.B. 2: if you want to trim this down you may need to resolve linker errors due to missing fixture deps manually
set(ta_test_src_files  ta_test.cpp
    range1.cpp
    range.cpp
    block_range.cpp
    type_traits.cpp
    tensor.cpp
    tensor_of_tensor.cpp
    tensor_tensor_view.cpp
    tensor_shift_wrapper.cpp
    tiled_range1.cpp
    tiled_range.cpp
    blocked_pmap.cpp
    round_robin_pmap.cpp
    hash_pmap.cpp
    cyclic_pmap.cpp
    replicated_pmap.cpp
    dense_shape.cpp
    sparse_shape.cpp
    distributed_storage.cpp
    tensor_impl.cpp
    array_impl.cpp
    index_list.cpp
    bipartite_index_list.cpp
    dist_array.cpp
    conversions.cpp
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
    expressions_sparse.cpp
    expressions_complex.cpp
    expressions_btas.cpp
    expressions_mixed.cpp
    foreach.cpp
    solvers.cpp
    initializer_list.cpp
    diagonal_array.cpp
    retile.cpp
    tot_dist_array_part1.cpp
    tot_dist_array_part2.cpp
    random.cpp
    trace.cpp
    tot_expressions.cpp
    annotation.cpp
    diagonal_array.cpp
    contraction_helpers.cpp
    s_t_t_contract_.cpp
    t_t_t_contract_.cpp
    t_s_t_contract_.cpp
  # t_tot_tot_contract_.cpp
  # tot_tot_tot_contract_.cpp
    einsum.cpp
    linalg.cpp
    cp.cpp
    btas.cpp
)

if(CUDA_FOUND OR HIP_FOUND)
  list(APPEND ta_test_src_files librett.cpp expressions_device_um.cpp tensor_um.cpp)
endif()

# if using C++20 must use Boost 1.74 or later:
# - https://en.cppreference.com/w/cpp/io/basic_ostream/operator_ltlt2
# - https://github.com/boostorg/test/commit/db6b98c72783351e0acd3c558691323a7a103ba9
if (CMAKE_CXX_STANDARD GREATER_EQUAL 20 AND DEFINED Boost_VERSION)
  if (Boost_VERSION VERSION_LESS 1.74.0)
     message(FATAL_ERROR "Boost 1.74 or later required to build TA unit tests when using C++20 or higher")
  endif()
endif()

# if tiledarray library was compiled without exceptions, use TA header-only (see below)
if (NOT (TA_ASSERT_POLICY STREQUAL TA_ASSERT_THROW) AND NOT CUDA_FOUND AND FALSE)
  add_ta_executable(${executable} "${ta_test_src_files}" "MADworld;${TILEDARRAY_PRIVATE_LINK_LIBRARIES}")
  target_compile_definitions(${executable} PRIVATE TILEDARRAY_HEADER_ONLY=1)
  if (LAPACK_INCLUDE_DIRS)
    target_include_directories(${executable} PRIVATE ${LAPACK_INCLUDE_DIRS})
  endif(LAPACK_INCLUDE_DIRS)
  if (LAPACK_COMPILE_OPTIONS)
    target_compile_options(${executable} PRIVATE ${LAPACK_COMPILE_OPTIONS})
  endif(LAPACK_COMPILE_OPTIONS)
  if (LAPACK_COMPILE_DEFINITIONS)
    target_compile_definitions(${executable} PRIVATE ${LAPACK_COMPILE_DEFINITIONS})
  endif(LAPACK_COMPILE_DEFINITIONS)
else()
  add_ta_executable(${executable} "${ta_test_src_files}" "tiledarray")
endif()
# if Boost is modularized, need to explicitly state that we need Boost::test module
if (Boost_IS_MODULARIZED)
  target_link_libraries(${executable} PRIVATE Boost::unit_test_framework)
endif()
# Add include directories and compiler flags for ta_test
target_include_directories(${executable} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/tests
    )
# unit tests
# 1. should disable error messages
# N.B. used to manually define TA_EXCEPTION_ERROR here; since some code is now compiled (rather than is header-only) it
#      is too late to do this here; must set TA_ERROR=throw if want to run unit tests
target_compile_definitions(${executable} PRIVATE TILEDARRAY_NO_USER_ERROR_MESSAGES=1
    MADNESS_DISPLAY_EXCEPTION_BREAK_MESSAGE=0)

# Add targets
add_test(tiledarray/unit/build "${CMAKE_COMMAND}" --build ${PROJECT_BINARY_DIR} --target ${executable})
set_tests_properties(tiledarray/unit/build PROPERTIES FIXTURES_SETUP TA_UNIT_TESTS_EXEC)

# N.B. some CUDA unit tests require TA_CUDA_NUM_STREAMS=1 for now
set(TA_UNIT_TESTS_ENVIRONMENT "MAD_NUM_THREADS=2;TA_CUDA_NUM_STREAMS=1")

# Add a test(s)
if(ENABLE_MPI)
  set (${executable}_np_1_args --run_test=!@distributed)
  set (${executable}_np_2_args --run_test=!@serial)
  foreach(p RANGE 1 2)
    add_test(
      NAME tiledarray/unit/run-np-${p}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${p}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${executable}> --log_level=unit_scope ${${executable}_np_${p}_args}
        ${MPIEXEC_POSTFLAGS}
      )
    if (p GREATER 1)
      set(TA_UNIT_TESTS_ENVIRONMENT "${TA_UNIT_TESTS_ENVIRONMENT};TA_UT_DISTRIBUTED=1")
    endif()
    set_tests_properties(tiledarray/unit/run-np-${p}
            PROPERTIES FIXTURES_REQUIRED TA_UNIT_TESTS_EXEC
            ENVIRONMENT "${TA_UNIT_TESTS_ENVIRONMENT}"
            )
  endforeach(p)
else()
  add_test(NAME tiledarray/unit/run-np-1
           COMMAND ${executable})
  set_tests_properties(tiledarray/unit/run-np-1
          PROPERTIES FIXTURES_REQUIRED TA_UNIT_TESTS_EXEC
          ENVIRONMENT "${TA_UNIT_TESTS_ENVIRONMENT}"
          )
endif()
