AC_DEFUN([ACX_CHECK_STATIC_ASSERT], [
  acx_static_assert=no
  AC_MSG_CHECKING([for static_assert support])
  
  AC_COMPILE_IFELSE(
    [
      AC_LANG_PROGRAM(
        [[]],
        [[static_assert( true );]]
      )
    ],
    [
      AC_DEFINE([TILEDARRAY_HAVE_STATIC_ASSERT],[1],[define if compiler supports static_assert.])
      acx_static_assert=yes
    ]
  )
  
  AC_MSG_RESULT([$acx_static_assert])

  if test "$acx_static_assert" = no; then
    AC_CHECK_HEADER([boost/static_assert.hpp],
      [], 
      [AC_MSG_ERROR([Unable to find Boost Static Assert header file.])]
    )
  fi

])