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

AC_DEFUN([ACX_WITH_BLAS], [

  AC_ARG_WITH([blas], [AS_HELP_STRING([--with-blas@<:@=yes|no|check@:>@],
      [use BLAS for matrix and vector operations@<:@Default=check@:>@.])],
    [
      case $withval in
      yes)
        acx_with_blas=yes
      ;;
      no)
        acx_with_blas=no
      ;;
      *)
        acx_with_blas=$withval
      ;;
      esac
    ],
    [acx_with_blas=check]
  )
  
  if test $acx_with_blas != "no"; then
    acx_have_blas=no
        
    if test $acx_have_blas = no; then
      AC_CHECK_FUNC([jag_dgemm],
        [
          acx_have_blas=yes
          AC_DEFINE([TILEDARRAY_HAS_BLAS], [1], [Defines when BLAS functions are available.])
          AC_DEFINE([F77_SGEMM], [jag_sgemm], [Alias BLAS SGEMM function name])
          AC_DEFINE([F77_DGEMM], [jag_dgemm], [Alias BLAS DGEMM function name])
          AC_DEFINE([F77_CGEMM], [jag_cgemm], [Alias BLAS CGEMM function name])
          AC_DEFINE([F77_ZGEMM], [jag_zgemm], [Alias BLAS ZGEMM function name])
          AC_DEFINE([F77_SGEMV], [jag_sgemv], [Alias BLAS SGEMV function name])
          AC_DEFINE([F77_DGEMV], [jag_dgemv], [Alias BLAS DGEMV function name])
          AC_DEFINE([F77_CGEMV], [jag_cgemv], [Alias BLAS CGEMV function name])
          AC_DEFINE([F77_ZGEMV], [jag_zgemv], [Alias BLAS ZGEMV function name])
          AC_DEFINE([F77_SGER], [jag_sger], [Alias BLAS SGER function name])
          AC_DEFINE([F77_DGER], [jag_dger], [Alias BLAS DGER function name])
          AC_DEFINE([F77_CGER], [jag_cger], [Alias BLAS CGER function name])
          AC_DEFINE([F77_ZGER], [jag_zger], [Alias BLAS ZGER function name])
          AC_DEFINE([F77_SSCAL], [jag_sscal], [Alias BLAS SSCAL function name])
          AC_DEFINE([F77_DSCAL], [jag_dscal], [Alias BLAS DSCAL function name])
          AC_DEFINE([F77_CSCAL], [jag_cscal], [Alias BLAS CSCAL function name])
          AC_DEFINE([F77_ZSCAL], [jag_zscal], [Alias BLAS ZSCAL function name])
          AC_DEFINE([F77_CSSCAL], [jag_csscal], [Alias BLAS CSSCAL function name])
          AC_DEFINE([F77_ZDSCAL], [jag_zdscal], [Alias BLAS ZDSCAL function name])
          AC_DEFINE([F77_SDOT], [jag_sdot], [Alias BLAS SDOT function name])
          AC_DEFINE([F77_DDOT], [jag_ddot], [Alias BLAS DDOT function name])
          AC_DEFINE([F77_CDOTU], [jag_cdotu], [Alias BLAS CDOTU function name])
          AC_DEFINE([F77_ZDOTU], [jag_zdotu], [Alias BLAS ZDOTU function name])
        ]
      )
    fi
    if test $acx_have_blas = no; then
      AC_CHECK_FUNC([dgemm_],
        [
          acx_have_blas=yes
          AC_DEFINE([TILEDARRAY_HAS_BLAS], [1], [Defines when BLAS functions are available.])
          AC_DEFINE([F77_SGEMM], [sgemm_], [Alias BLAS SGEMM function name])
          AC_DEFINE([F77_DGEMM], [dgemm_], [Alias BLAS DGEMM function name])
          AC_DEFINE([F77_CGEMM], [cgemm_], [Alias BLAS CGEMM function name])
          AC_DEFINE([F77_ZGEMM], [zgemm_], [Alias BLAS ZGEMM function name])
          AC_DEFINE([F77_SGEMV], [sgemv_], [Alias BLAS SGEMV function name])
          AC_DEFINE([F77_DGEMV], [dgemv_], [Alias BLAS DGEMV function name])
          AC_DEFINE([F77_CGEMV], [cgemv_], [Alias BLAS CGEMV function name])
          AC_DEFINE([F77_ZGEMV], [zgemv_], [Alias BLAS ZGEMV function name])
          AC_DEFINE([F77_SGER], [sger_], [Alias BLAS SGER function name])
          AC_DEFINE([F77_DGER], [dger_], [Alias BLAS DGER function name])
          AC_DEFINE([F77_CGER], [cger_], [Alias BLAS CGER function name])
          AC_DEFINE([F77_ZGER], [zger_], [Alias BLAS ZGER function name])
          AC_DEFINE([F77_SSCAL], [sscal_], [Alias BLAS SSCAL function name])
          AC_DEFINE([F77_DSCAL], [dscal_], [Alias BLAS DSCAL function name])
          AC_DEFINE([F77_CSCAL], [cscal_], [Alias BLAS CSCAL function name])
          AC_DEFINE([F77_ZSCAL], [zscal_], [Alias BLAS ZSCAL function name])
          AC_DEFINE([F77_CSSCAL], [csscal_], [Alias BLAS CSSCAL function name])
          AC_DEFINE([F77_ZDSCAL], [zdscal_], [Alias BLAS ZDSCAL function name])
          AC_DEFINE([F77_SDOT], [sdot_], [Alias BLAS SDOT function name])
          AC_DEFINE([F77_DDOT], [ddot_], [Alias BLAS DDOT function name])
          AC_DEFINE([F77_CDOTU], [cdotu_], [Alias BLAS CDOTU function name])
          AC_DEFINE([F77_ZDOTU], [zdotu_], [Alias BLAS ZDOTU function name])
        ]
      )
    fi
    if test $acx_have_blas = no; then
      AC_CHECK_FUNC([dgemm],
        [
          acx_have_blas=yes
          AC_DEFINE([TILEDARRAY_HAS_BLAS], [1], [Defines when BLAS functions are available.])
        ]
      )
    fi
    if test $acx_have_blas = no; then
      AC_CHECK_FUNC([dgemm__],
        [
          acx_have_blas=yes
          AC_DEFINE([TILEDARRAY_HAS_BLAS], [1], [Defines when BLAS functions are available.])
          AC_DEFINE([F77_SGEMM], [sgemm__], [Alias BLAS SGEMM function name])
          AC_DEFINE([F77_DGEMM], [dgemm__], [Alias BLAS DGEMM function name])
          AC_DEFINE([F77_CGEMM], [cgemm__], [Alias BLAS CGEMM function name])
          AC_DEFINE([F77_ZGEMM], [zgemm__], [Alias BLAS ZGEMM function name])
          AC_DEFINE([F77_SGEMV], [sgemv__], [Alias BLAS SGEMV function name])
          AC_DEFINE([F77_DGEMV], [dgemv__], [Alias BLAS DGEMV function name])
          AC_DEFINE([F77_CGEMV], [cgemv__], [Alias BLAS CGEMV function name])
          AC_DEFINE([F77_ZGEMV], [zgemv__], [Alias BLAS ZGEMV function name])
          AC_DEFINE([F77_SGER], [sger__], [Alias BLAS SGER function name])
          AC_DEFINE([F77_DGER], [dger__], [Alias BLAS DGER function name])
          AC_DEFINE([F77_CGER], [cger__], [Alias BLAS CGER function name])
          AC_DEFINE([F77_ZGER], [zger__], [Alias BLAS ZGER function name])
          AC_DEFINE([F77_SSCAL], [sscal__], [Alias BLAS SSCAL function name])
          AC_DEFINE([F77_DSCAL], [dscal__], [Alias BLAS DSCAL function name])
          AC_DEFINE([F77_CSCAL], [cscal__], [Alias BLAS CSCAL function name])
          AC_DEFINE([F77_ZSCAL], [zscal__], [Alias BLAS ZSCAL function name])
          AC_DEFINE([F77_CSSCAL], [csscal__], [Alias BLAS CSSCAL function name])
          AC_DEFINE([F77_ZDSCAL], [zdscal__], [Alias BLAS ZDSCAL function name])
          AC_DEFINE([F77_SDOT], [sdot__], [Alias BLAS SDOT function name])
          AC_DEFINE([F77_DDOT], [ddot__], [Alias BLAS DDOT function name])
          AC_DEFINE([F77_CDOTU], [cdotu__], [Alias BLAS CDOTU function name])
          AC_DEFINE([F77_ZDOTU], [zdotu__], [Alias BLAS ZDOTU function name])
        ]
      )
    fi
    if test $acx_have_blas = no; then
      AC_CHECK_FUNC([DGEMM],
        [
          acx_have_blas=yes
          AC_DEFINE([TILEDARRAY_HAS_BLAS], [1], [Defines when BLAS functions are available.])
          AC_DEFINE([F77_SGEMM], [SGEMM], [Alias BLAS SGEMM function name])
          AC_DEFINE([F77_DGEMM], [DGEMM], [Alias BLAS DGEMM function name])
          AC_DEFINE([F77_CGEMM], [CGEMM], [Alias BLAS CGEMM function name])
          AC_DEFINE([F77_ZGEMM], [ZGEMM], [Alias BLAS ZGEMM function name])
          AC_DEFINE([F77_SGEMV], [SGEMV], [Alias BLAS SGEMV function name])
          AC_DEFINE([F77_DGEMV], [DGEMV], [Alias BLAS DGEMV function name])
          AC_DEFINE([F77_CGEMV], [CGEMV], [Alias BLAS CGEMV function name])
          AC_DEFINE([F77_ZGEMV], [ZGEMV], [Alias BLAS ZGEMV function name])
          AC_DEFINE([F77_SGER], [SGER], [Alias BLAS SGER function name])
          AC_DEFINE([F77_DGER], [DGER], [Alias BLAS DGER function name])
          AC_DEFINE([F77_CGER], [CGER], [Alias BLAS CGER function name])
          AC_DEFINE([F77_ZGER], [ZGER], [Alias BLAS ZGER function name])
          AC_DEFINE([F77_SSCAL], [SSCAL], [Alias BLAS SSCAL function name])
          AC_DEFINE([F77_DSCAL], [DSCAL], [Alias BLAS DSCAL function name])
          AC_DEFINE([F77_CSCAL], [CSCAL], [Alias BLAS CSCAL function name])
          AC_DEFINE([F77_ZSCAL], [ZSCAL], [Alias BLAS ZSCAL function name])
          AC_DEFINE([F77_CSSCAL], [CSSCAL], [Alias BLAS CSSCAL function name])
          AC_DEFINE([F77_ZDSCAL], [ZDSCAL], [Alias BLAS ZDSCAL function name])
          AC_DEFINE([F77_SDOT], [SDOTU], [Alias BLAS SDOT function name])
          AC_DEFINE([F77_DDOT], [DDOTU], [Alias BLAS DDOT function name])
          AC_DEFINE([F77_CDOTU], [CDOTU], [Alias BLAS CDOTU function name])
          AC_DEFINE([F77_ZDOTU], [ZDOTU], [Alias BLAS ZDOTU function name])
        ]
      )
    fi
    if test $acx_have_blas = no; then
      AC_CHECK_FUNC([DGEMM_],
        [
          acx_have_blas=yes
          AC_DEFINE([TILEDARRAY_HAS_CBLAS], [1], [Defines when BLAS functions are available.])
          AC_DEFINE([F77_SGEMM], [SGEMM_], [Alias BLAS SGEMM function name])
          AC_DEFINE([F77_DGEMM], [DGEMM_], [Alias BLAS DGEMM function name])
          AC_DEFINE([F77_CGEMM], [CGEMM_], [Alias BLAS CGEMM function name])
          AC_DEFINE([F77_ZGEMM], [ZGEMM_], [Alias BLAS ZGEMM function name])
          AC_DEFINE([F77_SGEMV], [SGEMV_], [Alias BLAS SGEMV function name])
          AC_DEFINE([F77_DGEMV], [DGEMV_], [Alias BLAS DGEMV function name])
          AC_DEFINE([F77_CGEMV], [CGEMV_], [Alias BLAS CGEMV function name])
          AC_DEFINE([F77_ZGEMV], [ZGEMV_], [Alias BLAS ZGEMV function name])
          AC_DEFINE([F77_SGER], [SGER_], [Alias BLAS SGER function name])
          AC_DEFINE([F77_DGER], [DGER_], [Alias BLAS DGER function name])
          AC_DEFINE([F77_CGER], [CGER_], [Alias BLAS CGER function name])
          AC_DEFINE([F77_ZGER], [ZGER_], [Alias BLAS ZGER function name])
          AC_DEFINE([F77_SSCAL], [SSCAL_], [Alias BLAS SSCAL function name])
          AC_DEFINE([F77_DSCAL], [DSCAL_], [Alias BLAS DSCAL function name])
          AC_DEFINE([F77_CSCAL], [CSCAL_], [Alias BLAS CSCAL function name])
          AC_DEFINE([F77_ZSCAL], [ZSCAL_], [Alias BLAS ZSCAL function name])
          AC_DEFINE([F77_CSSCAL], [CSSCAL_], [Alias BLAS CSSCAL function name])
          AC_DEFINE([F77_ZDSCAL], [ZDSCAL_], [Alias BLAS ZDSCAL function name])
          AC_DEFINE([F77_SDOT], [SDOT_], [Alias BLAS SDOT function name])
          AC_DEFINE([F77_DDOT], [DDOTSUB_], [Alias BLAS DDOT function name])
          AC_DEFINE([F77_CDOTU], [CDOTU_], [Alias BLAS CDOTU function name])
          AC_DEFINE([F77_ZDOTU], [ZDOTU_], [Alias BLAS ZDOTU function name])
        ]
      )
    fi
    
    if test $acx_with_blas$acx_have_blas = "yesno"; then
      AC_MSG_ERROR([Unable to find BLAS.])
    fi 
  fi
])