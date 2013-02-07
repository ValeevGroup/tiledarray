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

AC_DEFUN([ACX_CHECK_STATIC_ASSERT], [
  acx_static_assert=no
  AC_MSG_CHECKING([for static_assert support])
  
  AC_COMPILE_IFELSE(
    [
      AC_LANG_PROGRAM(
        [[]],
        [[static_assert( true );]]
      )
    ],
    [
      AC_DEFINE([TILEDARRAY_HAVE_STATIC_ASSERT],[1],[define if compiler supports static_assert.])
      acx_static_assert=yes
    ]
  )
  
  AC_MSG_RESULT([$acx_static_assert])

  if test "$acx_static_assert" = no; then
    AC_CHECK_HEADER([boost/static_assert.hpp],
      [], 
      [AC_MSG_ERROR([Unable to find Boost Static Assert header file.])]
    )
  fi

])