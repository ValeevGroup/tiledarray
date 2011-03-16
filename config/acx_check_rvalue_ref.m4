AC_DEFUN([ACX_CHECK_RVALUE_REF], [
  acx_has_rvalue_ref=no
  AC_MSG_CHECKING([for compiler rvalue reference support])
  
    AC_COMPILE_IFELSE(
    [
      AC_LANG_PROGRAM(
        [[int func(int&& i) { return i; }]],
        [[func(int(1));]]
      )
    ],
    [
      AC_DEFINE([TILEDARRAY_RVALUE_REF],[1],[define if compiler supports rvalue references.])
      acx_has_rvalue_ref=yes
    ]
  )

  AC_MSG_RESULT([$acx_has_rvalue_ref])
])