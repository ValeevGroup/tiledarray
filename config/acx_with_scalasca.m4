AC_DEFUN([ACX_WITH_SCALASCA],[
  AC_ARG_WITH([scalasca], [AS_HELP_STRING([--with-scalasca@<:@=dir@:>@],
      [Build with Scalasca instrumentation])],
    [acx_with_scalasca=$withval],
    [acx_with_scalasca=no]
  )
  
  case "$acx_with_scalasca" in
  yes)
    CPPFLAGS="-DEPIK=1 $CPPFLAGS"
    AC_CHECK_HEADERS([epik_user.h], 
      [], [AC_MSG_ERROR([Unable to include epic_user.h.])])
  ;;
  no)
    # Do nothing
  ;;
  *)
    CPPFLAGS="-DEPIK=1 -I$acx_with_scalasca/include $CPPFLAGS"
    AC_CHECK_HEADERS([epik_user.h], 
      [], [AC_MSG_ERROR([Unable to include epic_user.h.])])
  ;;
  esac

])