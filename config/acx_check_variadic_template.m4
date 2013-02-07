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
