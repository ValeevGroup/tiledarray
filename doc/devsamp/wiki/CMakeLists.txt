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

# List of all complete examples
set(wikiexamples user-guide-1;user-guide-1a;user-guide-2)

foreach(example ${wikiexamples})
  add_executable(${example} EXCLUDE_FROM_ALL ${example}.cpp)
  target_link_libraries(${example} PRIVATE tiledarray ${MADNESS_DISABLEPIE_LINKER_FLAG})
  add_dependencies(examples-tiledarray ${example})
endforeach()
