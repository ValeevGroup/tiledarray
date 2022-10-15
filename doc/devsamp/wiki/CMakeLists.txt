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
set(wikiexamples user-guide-1;user-guide-1a;user-guide-1b;user-guide-2)

# if target examples-tiledarray does not exist yet, create it
if (NOT TARGET examples-tiledarray)
  add_custom_target_subproject(tiledarray examples)
endif()

foreach(example ${wikiexamples})
  add_ta_executable(${example} ${example}.cpp tiledarray)
  add_dependencies(examples-tiledarray ${example})
endforeach()
