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
          AC_DEFINE([sgemm], [jag_sgemm], [Alias BLAS sgemm function name])
          AC_DEFINE([dgemm], [jag_dgemm], [Alias BLAS dgemm function name])
          AC_DEFINE([cgemm], [jag_cgemm], [Alias BLAS cgemm function name])
          AC_DEFINE([zgemm], [jag_zgemm], [Alias BLAS zgemm function name])
          AC_DEFINE([sscal], [jag_sscal], [Alias BLAS sscal function name])
          AC_DEFINE([dscal], [jag_dscal], [Alias BLAS dscal function name])
          AC_DEFINE([cscal], [jag_cscal], [Alias BLAS cscal function name])
          AC_DEFINE([zscal], [jag_zscal], [Alias BLAS zscal function name])
          AC_DEFINE([csscal], [jag_csscal], [Alias BLAS csscal function name])
          AC_DEFINE([zdscal], [jag_zdscal], [Alias BLAS zdscal function name])
          AC_DEFINE([sdot], [jag_sdot], [Alias BLAS sdot function name])
          AC_DEFINE([ddot], [jag_ddot], [Alias BLAS ddot function name])
          AC_DEFINE([cdotu], [jag_cdotu], [Alias BLAS cdotu function name])
          AC_DEFINE([zdotu], [jag_zdotu], [Alias BLAS zdotu function name])
        ]
      )
    fi
    if test $acx_have_blas = no; then
      AC_CHECK_FUNC([dgemm_],
        [
          acx_have_blas=yes
          AC_DEFINE([TILEDARRAY_HAS_BLAS], [1], [Defines when BLAS functions are available.])
          AC_DEFINE([sgemm], [sgemm_], [Alias BLAS sgemm function name])
          AC_DEFINE([dgemm], [dgemm_], [Alias BLAS dgemm function name])
          AC_DEFINE([cgemm], [cgemm_], [Alias BLAS cgemm function name])
          AC_DEFINE([zgemm], [zgemm_], [Alias BLAS zgemm function name])
          AC_DEFINE([sscal], [sscal_], [Alias BLAS sscal function name])
          AC_DEFINE([dscal], [dscal_], [Alias BLAS dscal function name])
          AC_DEFINE([cscal], [cscal_], [Alias BLAS cscal function name])
          AC_DEFINE([zscal], [zscal_], [Alias BLAS zscal function name])
          AC_DEFINE([csscal], [csscal_], [Alias BLAS csscal function name])
          AC_DEFINE([zdscal], [zdscal_], [Alias BLAS zdscal function name])
          AC_DEFINE([sdot], [sdot_], [Alias BLAS sdot function name])
          AC_DEFINE([ddot], [ddot_], [Alias BLAS ddot function name])
          AC_DEFINE([cdotu], [cdotu_], [Alias BLAS cdotu function name])
          AC_DEFINE([zdotu], [zdotu_], [Alias BLAS zdotu function name])
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
          AC_DEFINE([sgemm], [sgemm__], [Alias BLAS sgemm function name])
          AC_DEFINE([dgemm], [dgemm__], [Alias BLAS dgemm function name])
          AC_DEFINE([cgemm], [cgemm__], [Alias BLAS cgemm function name])
          AC_DEFINE([zgemm], [zgemm__], [Alias BLAS zgemm function name])
          AC_DEFINE([sscal], [sscal__], [Alias BLAS sscal function name])
          AC_DEFINE([dscal], [dscal__], [Alias BLAS dscal function name])
          AC_DEFINE([cscal], [cscal__], [Alias BLAS cscal function name])
          AC_DEFINE([zscal], [zscal__], [Alias BLAS zscal function name])
          AC_DEFINE([csscal], [csscal__], [Alias BLAS csscal function name])
          AC_DEFINE([zdscal], [zdscal__], [Alias BLAS zdscal function name])
          AC_DEFINE([sdot], [sdot__], [Alias BLAS sdot function name])
          AC_DEFINE([ddot], [ddot__], [Alias BLAS ddot function name])
          AC_DEFINE([cdotu], [cdotu__], [Alias BLAS cdotu function name])
          AC_DEFINE([zdotu], [zdotu__], [Alias BLAS zdotu function name])
        ]
      )
    fi
    if test $acx_have_blas = no; then
      AC_CHECK_FUNC([DGEMM],
        [
          acx_have_blas=yes
          AC_DEFINE([TILEDARRAY_HAS_BLAS], [1], [Defines when BLAS functions are available.])
          AC_DEFINE([sgemm], [SGEMM], [Alias BLAS sgemm function name])
          AC_DEFINE([dgemm], [DGEMM], [Alias BLAS dgemm function name])
          AC_DEFINE([cgemm], [CGEMM], [Alias BLAS cgemm function name])
          AC_DEFINE([zgemm], [ZGEMM], [Alias BLAS zgemm function name])
          AC_DEFINE([sscal], [SSCAL], [Alias BLAS sscal function name])
          AC_DEFINE([dscal], [DSCAL], [Alias BLAS dscal function name])
          AC_DEFINE([cscal], [CSCAL], [Alias BLAS cscal function name])
          AC_DEFINE([zscal], [ZSCAL], [Alias BLAS zscal function name])
          AC_DEFINE([csscal], [CSSCAL], [Alias BLAS csscal function name])
          AC_DEFINE([zdscal], [ZDSCAL], [Alias BLAS zdscal function name])
          AC_DEFINE([sdot], [SDOTU], [Alias BLAS sdot function name])
          AC_DEFINE([ddot], [DDOTU], [Alias BLAS ddot function name])
          AC_DEFINE([cdotu], [CDOTU], [Alias BLAS cdotu function name])
          AC_DEFINE([zdotu], [ZDOTU], [Alias BLAS zdotu function name])
        ]
      )
    fi
    if test $acx_have_blas = no; then
      AC_CHECK_FUNC([DGEMM_],
        [
          acx_have_blas=yes
          AC_DEFINE([TILEDARRAY_HAS_CBLAS], [1], [Defines when BLAS functions are available.])
          AC_DEFINE([sgemm], [SGEMM_], [Alias BLAS sgemm function name])
          AC_DEFINE([dgemm], [DGEMM_], [Alias BLAS dgemm function name])
          AC_DEFINE([cgemm], [CGEMM_], [Alias BLAS cgemm function name])
          AC_DEFINE([zgemm], [ZGEMM_], [Alias BLAS zgemm function name])
          AC_DEFINE([sscal], [SSCAL_], [Alias BLAS sscal function name])
          AC_DEFINE([dscal], [DSCAL_], [Alias BLAS dscal function name])
          AC_DEFINE([cscal], [CSCAL_], [Alias BLAS cscal function name])
          AC_DEFINE([zscal], [ZSCAL_], [Alias BLAS zscal function name])
          AC_DEFINE([csscal], [CSSCAL_], [Alias BLAS csscal function name])
          AC_DEFINE([zdscal], [ZDSCAL_], [Alias BLAS zdscal function name])
          AC_DEFINE([sdot], [SDOT_], [Alias BLAS sdot function name])
          AC_DEFINE([ddot], [DDOTSUB_], [Alias BLAS ddot function name])
          AC_DEFINE([cdotu], [CDOTU_], [Alias BLAS cdotu function name])
          AC_DEFINE([zdotu], [ZDOTU_], [Alias BLAS zdotu function name])
        ]
      )
    fi
    
    if test $acx_with_blas$acx_have_blas = "yesno"; then
      AC_MSG_ERROR([Unable to find BLAS.])
    fi 
  fi
])