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

AC_DEFUN([ACX_CHECK_RVALUE_REF], [
  acx_rvalue_ref=no
  AC_MSG_CHECKING([for compiler rvalue reference support])
  
  AC_COMPILE_IFELSE(
    [
      AC_LANG_PROGRAM(
        [[int func(int&& i) { return i; }]],
        [[func(int(1));]]
      )
    ],
    [
      AC_DEFINE([TILEDARRAY_RVALUE_REF],[1],[define if compiler supports rvalue references.])
      acx_rvalue_ref=yes
    ]
  )

  AC_MSG_RESULT([$acx_rvalue_ref])
  
  if test "$acx_rvalue_ref" = yes; then
  
    # Check for std::identity support
    acx_has_std_identity=no
    AC_MSG_CHECKING([for std::identity])
    
    AC_COMPILE_IFELSE(
      [
        AC_LANG_PROGRAM(
          [[#include <utility>]],
          [[using std::identity;]]
        )
      ],
      [
        AC_DEFINE([TILEDARRAY_HAS_STD_IDENTITY],[1],[define if compiler has std::identity.])
        acx_has_std_identity=yes
      ]
    )
  
    AC_MSG_RESULT([$acx_has_std_identity])

    # Check for std::move support
    acx_has_std_move=no
    AC_MSG_CHECKING([for std::move])
    
    AC_COMPILE_IFELSE(
      [
        AC_LANG_PROGRAM(
          [[#include <utility>]],
          [[using std::move;]]
        )
      ],
      [
        AC_DEFINE([TILEDARRAY_HAS_STD_MOVE],[1],[define if compiler has std::move.])
        acx_has_std_move=yes
      ]
    )
  
    AC_MSG_RESULT([$acx_has_std_move])
    
    # Check for std::forward support
    acx_has_std_forward=no
    AC_MSG_CHECKING([for std::forward])
    
    AC_COMPILE_IFELSE(
      [
        AC_LANG_PROGRAM(
          [[#include <utility>]],
          [[using std::forward;]]
        )
      ],
      [
        AC_DEFINE([TILEDARRAY_HAS_STD_FORWARD],[1],[define if compiler has std::forward.])
        acx_has_std_forward=yes
      ]
    )
  
    AC_MSG_RESULT([$acx_has_std_forward])
  fi
])