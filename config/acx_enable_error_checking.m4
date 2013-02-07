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

AC_DEFUN([ACX_ENABLE_ERROR_CHECKING], [
  TA_DEFAULT_ERROR=3
  AC_ARG_ENABLE([error-checking],
    [AC_HELP_STRING([--enable-error-checking@<:@=throw|assert|no@:>@],
      [Enable default error checking@<:@default=throw@:>@.])],
    [
      case $enableval in
        yes)
          AC_DEFINE([TA_DEFAULT_ERROR], [1], 
            [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
        ;;
        throw)
          AC_DEFINE([TA_DEFAULT_ERROR], [1],
            [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
        ;;
        assert)
          AC_DEFINE([TA_DEFAULT_ERROR], [2],
            [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
        ;;
        no)
          AC_DEFINE([TA_DEFAULT_ERROR], [0],
            [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
        ;;
        *)
          AC_MSG_ERROR([Invalid input for error checking.])
        ;;
      esac   
    ],
    [AC_DEFINE([TA_DEFAULT_ERROR], [0],
      [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])]
  )
])