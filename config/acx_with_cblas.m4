AC_DEFUN([ACX_WITH_CBLAS], [

  AC_ARG_WITH([cblas], [AS_HELP_STRING([--with-cblas@<:@=yes|no|check@:>@],
      [use CBLAS for gemm operations@<:@Default=check@:>@.])],
    [
      case $withval in
      yes)
        acx_with_cblas=yes
      ;;
      no)
        acx_with_cblas=no
      ;;
      *)
        acx_with_cblas=$withval
      ;;
      esac
    ],
    [acx_with_cblas=check]
  )
  
  if test $acx_with_cblas != "no"; then
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
    
    if test $acx_with_cblas$acx_have_cblas = "yesno"; then
      AC_MSG_ERROR([Unable to find CBLAS.])
    fi 
  fi
])