AC_DEFUN([ACX_CHECK_VARIADIC_TEMPLATE_CLASS], [
  acx_variadic_template_class=no
  AC_MSG_CHECKING([for compiler variadic template class support])
  
    AC_COMPILE_IFELSE(
    [
      AC_LANG_PROGRAM(
        [[template <typename ... T> class C {};]],
        [[C<int, double, char> c;]]
      )
    ],
    [
      AC_DEFINE([TILEDARRAY_VARIADIC_TEMPLATE_CLASS],[1],[define if compiler supports variadic template classes.])
      acx_variadic_template_class=yes
    ]
  )

  AC_MSG_RESULT([$acx_variadic_template_class])
])

AC_DEFUN([ACX_CHECK_VARIADIC_TEMPLATE_FUNCTION], [
  acx_variadic_template_function=no
  AC_MSG_CHECKING([for compiler variadic template function support])
  
    AC_COMPILE_IFELSE(
    [
      AC_LANG_PROGRAM(
        [[template<class ... Types> void f(Types ... args) { }]],
        [[int i = 0; f(i, i);]]
      )
    ],
    [
      AC_DEFINE([TILEDARRAY_VARIADIC_TEMPLATE_FUNCTION],[1],[define if compiler supports variadic template functions.])
      acx_variadic_template_function=yes
    ]
  )

  AC_MSG_RESULT([$acx_variadic_template_function])
])

AC_DEFUN([ACX_CHECK_VARIADIC_SIZEOF], [
  acx_variadic_sizeof=no
  AC_MSG_CHECKING([for compiler sizeof... support])
  
    AC_COMPILE_IFELSE(
    [
      AC_LANG_PROGRAM(
        [[
#include <cstddef>
template<class... Types> struct count { static const std::size_t value = sizeof...(Types); };
        ]],
        [[std::size_t n = count<int, double, char>::value;]]
      )
    ],
    [
      AC_DEFINE([TILEDARRAY_VARIADIC_SIZEOF],[1],[define if compiler supports variadic template functions.])
      acx_variadic_sizeof=yes
    ]
  )

  AC_MSG_RESULT([$acx_variadic_sizeof])
])

AC_DEFUN([ACX_CHECK_VARIADIC_TEMPLATE], [
  ACX_CHECK_VARIADIC_TEMPLATE_CLASS
  ACX_CHECK_VARIADIC_TEMPLATE_FUNCTION
  ACX_CHECK_VARIADIC_SIZEOF
  if test "$acx_variadic_template_class$acx_variadic_template_function$acx_variadic_sizeof" = yesyesyes; then
    AC_DEFINE([TILEDARRAY_VARIADIC_TEMPLATE],[1],[define if compiler supports variadic templates.])
  fi
])