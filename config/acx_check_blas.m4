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

AC_DEFUN([ACX_CHECK_BLAS], [
  
  acx_have_blas=no
  
  AC_CHECK_FUNC([dgemm_], [acx_have_blas=yes])

  if test $acx_have_blas = no; then
    AC_CHECK_FUNC([dgemm], [acx_have_blas=yes])
  fi
  if test $acx_have_blas = no; then
    AC_CHECK_FUNC([dgemm__], [acx_have_blas=yes])
  fi
  if test $acx_have_blas = no; then
    AC_CHECK_FUNC([DGEMM], [acx_have_blas=yes])
  fi
  if test $acx_have_blas = no; then
    AC_CHECK_FUNC([DGEMM_], [acx_have_blas=yes])
  fi
    
  if test acx_have_blas = "no"; then
    AC_MSG_ERROR([Unable to find BLAS.])
  fi 

])