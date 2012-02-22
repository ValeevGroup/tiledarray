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
    
    if test $acx_with_blas$acx_have_cblas = "yesno"; then
      AC_MSG_ERROR([Unable to find CBLAS.])
    fi 
  fi
])