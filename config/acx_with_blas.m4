AC_DEFUN([ACX_CHECK_CBLAS],[
  acx_have_cblas=no
  AC_CHECK_HEADERS([mkl.h cblas.h],
    [
      AC_CHECK_FUNC([cblas_dgemm],
        [
          acx_have_cblas=yes
          AC_DEFINE([TILEDARRAY_HAS_CBLAS], [1], [Defines when cblas functions are available.])
        ]
      )
    ]
  )
])

AC_DEFUN([ACX_CHECK_BLAS],[
  acx_have_blas=no
  AC_CHECK_FUNC([dgemm], [acx_have_blas=yes])
    
  if test $acx_have_blas = no; then
    AC_CHECK_FUNC([dgemm_],
      [
        acx_have_blas=yes
        AC_DEFINE([dgemm], [dgemm_], [Set to the Fortran dgemm function name.])
        AC_DEFINE([sgemm], [sgemm_], [Set to the Fortran sgemm function name.])
        AC_DEFINE([zgemm], [zgemm_], [Set to the Fortran zgemm function name.])
        AC_DEFINE([cgemm], [cgemm_], [Set to the Fortran cgemm function name.])
      ])
  fi
  if test $acx_have_blas = no; then
    AC_CHECK_FUNC([dgemm__],
      [
        acx_have_blas=yes
        AC_DEFINE([dgemm], [dgemm__], [Set to the Fortran dgemm function name.])
        AC_DEFINE([sgemm], [sgemm__], [Set to the Fortran sgemm function name.])
        AC_DEFINE([zgemm], [zgemm__], [Set to the Fortran zgemm function name.])
        AC_DEFINE([cgemm], [cgemm__], [Set to the Fortran cgemm function name.])
      ])
  fi
  if test $acx_have_blas = no; then
    AC_CHECK_FUNC([DGEMM],
      [
        acx_have_blas=yes
        AC_DEFINE([dgemm], [DGEMM], [Set to the Fortran dgemm function name.])
        AC_DEFINE([sgemm], [SGEMM], [Set to the Fortran sgemm function name.])
        AC_DEFINE([zgemm], [ZGEMM], [Set to the Fortran zgemm function name.])
        AC_DEFINE([cgemm], [CGEMM], [Set to the Fortran cgemm function name.])
      ])
  fi
  if test $acx_have_blas = no; then
    AC_CHECK_FUNC([dgemm_],
      [
        acx_have_blas=yes
        AC_DEFINE([dgemm], [DGEMM_], [Set to the Fortran dgemm function name.])
        AC_DEFINE([sgemm], [SGEMM_], [Set to the Fortran sgemm function name.])
        AC_DEFINE([zgemm], [ZGEMM_], [Set to the Fortran zgemm function name.])
        AC_DEFINE([cgemm], [CGEMM_], [Set to the Fortran cgemm function name.])
      ])
  fi
  
  if test $acx_have_blas = yes; then
    AC_CHECK_FUNC([dgemm_],
      [
        AC_DEFINE([TILEDARRAY_HAS_BLAS], [1], [Defines when Fortran BLAS functions are available.])
      ])
  fi
])

AC_DEFUN([ACX_WITH_BLAS], [

  AC_ARG_WITH([blas], [AS_HELP_STRING([--with-blas@<:@=yes|no|check@:>@],
      [use BLAS for gemm operations@<:@Default=check@:>@.])],
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
    ACX_CHECK_CBLAS
    if test $acx_have_cblas = "no"; then
      ACX_CHECK_BLAS
    fi
    
    if test $acx_with_blas$acx_have_cblas$acx_have_blas = "yesnono"; then
      AC_MSG_ERROR([Unable to find BLAS.])
    fi 
  fi


])