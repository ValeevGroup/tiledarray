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

  foreach(_exec ta_cc_abcd_cuda ta_vector_cuda ta_reduce_cuda)

    # Add executable
    add_ta_executable(${_exec} "${_exec}.cpp" "tiledarray")
    add_dependencies(examples-tiledarray ${_exec})

  endforeach()

endif()

if(CUDA_FOUND OR HIP_FOUND)

    foreach(_exec device_task ta_dense_device)

        # Add executable
        add_ta_executable(${_exec} "${_exec}.cpp" "tiledarray")
        add_dependencies(examples-tiledarray ${_exec})

    endforeach()

endif()
