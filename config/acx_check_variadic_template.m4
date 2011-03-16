AC_DEFUN([ACX_CHECK_VARIADIC_TEMPLATE], [
  acx_variadic_template=no
  AC_MSG_CHECKING([for compiler variadic template support])
  
    AC_COMPILE_IFELSE(
    [
      AC_LANG_PROGRAM(
        [[
#include <cstddef>

// Basic variadic template support
template <typename ... T> class C;
template <typename T, typename ... TT> class C<T, TT...> { };
template <typename T> class C<T> { };
template <> class C<> { };

// sizeof... support
template<class... Types> 
struct count {
  static const std::size_t value = sizeof...(Types);
};

// Variadic template template support
template<template<typename...> class T, typename... U>
struct eval { };

// Variadic function support
template<class ... Types> void f(Types ... args) { }
        ]],
        [[
C<int, double, char> c;
std::size_t n = count<int,int>::value;
f(n);
        ]]
      )
    ],
    [
      AC_DEFINE([TILEDARRAY_VARIADIC_TEMPLATE],[1],[define if compiler supports variadic templates.])
      acx_variadic_template=yes
    ]
  )

  AC_MSG_RESULT([$acx_variadic_template])
])
