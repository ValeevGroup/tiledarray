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
#  Sep 4, 2013
#

find_package(CUDA)

# Create example executable

foreach(_exec blas eigen ta_band ta_dense ta_sparse ta_dense_nonuniform
              ta_dense_asymm ta_sparse_grow ta_dense_new_tile
              ta_cc_abcd)

  # Add executable
  add_executable(${_exec} EXCLUDE_FROM_ALL ${_exec}.cpp)
  target_link_libraries(${_exec} PRIVATE tiledarray ${MADNESS_DISABLEPIE_LINKER_FLAG})
  add_dependencies(${_exec} External)
  add_dependencies(examples ${_exec})

endforeach()

# NB ta_dense_elem has not been revised for El
if(ENABLE_ELEMENTAL)
    # Add the ta_dense_elem executable
#    add_executable(ta_dense_elem EXCLUDE_FROM_ALL ta_dense_elem.cpp)
#    target_link_libraries(ta_dense_elem PRIVATE tiledarray)
#    add_dependencies(ta_dense_elem External)
#    add_dependencies(examples ta_dense_elem)
endif(ENABLE_ELEMENTAL)

if(CUDA_FOUND)
    # Add the ta_dense_cuda executable
    cuda_add_executable(ta_dense_cuda EXCLUDE_FROM_ALL ta_dense_cuda.cpp cpu_cuda_vector.h cpu_cuda_vector.cu)
    cuda_add_cublas_to_target(ta_dense_cuda)
    target_link_libraries(ta_dense_cuda general tiledarray ${MADNESS_DISABLEPIE_LINKER_FLAG})
    add_dependencies(ta_dense_cuda External)
    add_dependencies(examples ta_dense_cuda)
endif(CUDA_FOUND)
