#
# This file is a part of TiledArray.
# Copyright (C) 2013  Virginia Tech
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

AC_DEFUN([ACX_CHECK_MADNESS], [
  AC_CHECK_HEADER([world/world.h], [], [
    AC_MSG_ERROR([Unable to find the required M-A-D-N-E-S-S header file.])
  ])
  
  ACX_CXX_CHECK_LIB([MADworld], [madness::initialize(int argc = 0, char** argv = 0)],
    [LIBS="$LIBS -lMADworld"], [AC_MSG_ERROR([The required library MADworld could not be found.])])
])