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

endif()
