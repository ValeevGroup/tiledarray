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
 
# if target examples-tiledarray does not exist yet, create it
if (NOT TARGET examples-tiledarray)
  add_custom_target_subproject(tiledarray examples)
endif()

# Add Subdirectories
add_subdirectory (cc)
add_subdirectory (cuda)
add_subdirectory (dgemm)
add_subdirectory (demo)
add_subdirectory (elemental)
add_subdirectory (scalapack)
add_subdirectory (fock)
add_subdirectory (mpi_tests)
add_subdirectory (pmap_test)
add_subdirectory (vector_tests)
