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

  foreach(_exec gpu_librett gpu_task ta_dense_gpu ta_cc_abcd_gpu ta_vector_gpu ta_reduce_gpu)

    # Add executable
    add_ta_executable(${_exec} "${_exec}.cpp" "tiledarray")
    add_dependencies(examples-tiledarray ${_exec})

  endforeach()

endif(CUDA_FOUND)
