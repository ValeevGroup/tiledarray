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

#-- Add an Option to toggle the generation of the API documentation

find_package(Doxygen 1.8.12)
if(DOXYGEN_FOUND)
  configure_file(Doxyfile.in Doxyfile @ONLY IMMEDIATE)
  # uses standard GNU target name for building html documentation, "html"
  add_custom_target (html USES_TERMINAL
    COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/dox/Doxyfile
    SOURCES ${PROJECT_BINARY_DIR}/doc/dox/Doxyfile)
  add_dependencies(doc html)
endif()
