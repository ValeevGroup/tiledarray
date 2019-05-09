#
#  This file is a part of TiledArray.
#  Copyright (C) 2018  Virginia Tech
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
#  Chong Peng
#  Department of Chemistry, Virginia Tech
#
#  CMakeLists.txt
#  Aug 01, 2018
#


if(CUDA_FOUND)

  cuda_add_executable(cuda_cutt EXCLUDE_FROM_ALL cuda_cutt.cpp)
  target_link_libraries(cuda_cutt PUBLIC tiledarray)
  add_dependencies(examples cuda_cutt)

  cuda_add_executable(cuda_task EXCLUDE_FROM_ALL cuda_task.cpp)
  target_link_libraries(cuda_task PUBLIC tiledarray)
  add_dependencies(examples cuda_task)

  # Add the ta_dense_cuda executable
  cuda_add_executable(ta_dense_cuda ta_dense_cuda.cpp)
  target_link_libraries(ta_dense_cuda PRIVATE tiledarray)
  add_dependencies(examples ta_dense_cuda)

  cuda_add_executable(ta_cc_abcd_cuda ta_cc_abcd_cuda.cpp)
  target_link_libraries(ta_cc_abcd_cuda PRIVATE tiledarray)
  add_dependencies(examples ta_cc_abcd_cuda)

  cuda_add_executable(ta_vector_cuda ta_vector_cuda.cpp)
  target_link_libraries(ta_vector_cuda PRIVATE tiledarray)
  add_dependencies(examples ta_vector_cuda)

  cuda_add_executable(ta_reduce_cuda ta_reduce_cuda.cpp)
  target_link_libraries(ta_reduce_cuda PRIVATE tiledarray)
  add_dependencies(examples ta_reduce_cuda)
endif(CUDA_FOUND)
