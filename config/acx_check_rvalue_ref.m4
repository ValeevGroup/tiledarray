AC_DEFUN([ACX_CHECK_RVALUE_REF], [
  acx_rvalue_ref=no
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
      acx_rvalue_ref=yes
    ]
  )

  AC_MSG_RESULT([$acx_rvalue_ref])
  
  if test "$acx_rvalue_ref" = yes; then
  
    # Check for std::identity support
    acx_has_std_identity=no
    AC_MSG_CHECKING([for std::identity])
    
    AC_COMPILE_IFELSE(
      [
        AC_LANG_PROGRAM(
          [[#include <utility>]],
          [[using std::identity;]]
        )
      ],
      [
        AC_DEFINE([TILEDARRAY_HAS_STD_IDENTITY],[1],[define if compiler has std::identity.])
        acx_has_std_identity=yes
      ]
    )
  
    AC_MSG_RESULT([$acx_has_std_identity])

    # Check for std::move support
    acx_has_std_move=no
    AC_MSG_CHECKING([for std::move])
    
    AC_COMPILE_IFELSE(
      [
        AC_LANG_PROGRAM(
          [[#include <utility>]],
          [[using std::move;]]
        )
      ],
      [
        AC_DEFINE([TILEDARRAY_HAS_STD_MOVE],[1],[define if compiler has std::move.])
        acx_has_std_move=yes
      ]
    )
  
    AC_MSG_RESULT([$acx_has_std_move])
    
    # Check for std::forward support
    acx_has_std_forward=no
    AC_MSG_CHECKING([for std::forward])
    
    AC_COMPILE_IFELSE(
      [
        AC_LANG_PROGRAM(
          [[#include <utility>]],
          [[using std::forward;]]
        )
      ],
      [
        AC_DEFINE([TILEDARRAY_HAS_STD_FORWARD],[1],[define if compiler has std::forward.])
        acx_has_std_forward=yes
      ]
    )
  
    AC_MSG_RESULT([$acx_has_std_forward])
  fi
])