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
#  CMakeLists.txt
#

# Strided-DGEMM ToT throughput benches (strided-vs-current arena DGEMM).

foreach(_exec opa_strided_arena_dgemm opb_strided_arena_dgemm regime_a_hce_e_strided_bench)
  add_ta_executable(${_exec} "${_exec}.cpp" "tiledarray")
  add_dependencies(examples-tiledarray ${_exec})
endforeach()
