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

# ACX_ENABLE_CXXFLAGS_ARG( enable-feature , flag , default arg , default=yes|no )
AC_DEFUN([ACX_ENABLE_CXXFLAGS_ARG], [
  AC_ARG_ENABLE([$1],
    [AC_HELP_STRING([--enable-[$1]@<:@=yes|no|ARG@:>@],
      [Enable $1 compiler flag Ex: $2=$3 @<:@default=$4@:>@.]) ],
    [
      case $enableval in
        yes)
          CXXFLAGS="$CXXFLAGS $2=$3"
        ;;
        no)
        ;;
        *)
          CXXFLAGS="$CXXFLAGS $2=$enableval"
        ;;
      esac
    ], [
      if test "$4" = yes; then
        CXXFLAGS="$CXXFLAGS $2=$3"
      fi 
    ]
  )
])