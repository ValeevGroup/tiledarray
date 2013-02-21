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

# Initialize autoconf
AC_INIT([TiledArray], [0.0], 
  [http://code.google.com/p/tiledarray/issues/list],
  [tiledarray], [http://code.google.com/p/tiledarray/])
AC_CONFIG_SRCDIR([src/tiled_array.h])
AC_CONFIG_MACRO_DIR([config])
AC_CONFIG_AUX_DIR([build-aux])
AC_LANG([C++])

# Initialize automake
AM_INIT_AUTOMAKE([ -Wall foreign no-define])

# Check for BSD style install programs
AC_PROG_INSTALL

# Check for mpi c++ compiler.
initial_CFLAGS="$CFLAGS"
initial_CXXFLAGS="$CXXFLAGS"

ACX_PROG_MPICXX
AC_PROG_CXX([$MPICXX])

CFLAGS="$initial_CFLAGS"
CXXFLAGS="$initial_CXXFLAGS"

AC_CHECK_TYPE([long double], AC_DEFINE([TILEDARRAY_HAS_LONG_DOUBLE],[1],[define if compiler supports long double.]), [x = 0] )
AC_CHECK_TYPE([long long int], AC_DEFINE([TILEDARRAY_HAS_LONG_LONG],[1],[define if compiler supports long long.]), [x = 0] )

# Set compiler standard
ACX_ENABLE_CXXFLAGS_ARG([cplusplus0x], [-std], [c++0x], [no])

# Check for C++11 language features
#ACX_CHECK_VARIADIC_TEMPLATE
#ACX_CHECK_RVALUE_REF
ACX_CHECK_STATIC_ASSERT

# Set debugging
ACX_ENABLE_CXXFLAGS_OPT([debugging], [-g], [], [no])

# Set optimization
ACX_ENABLE_CXXFLAGS_OPT([optimization], [-O], [2], [yes])

AC_CHECK_HEADER([mpi.h], [], [AC_MSG_ERROR([Unable to find mpi.h.])])

# Check for mpi execution command
AC_ARG_VAR([MPIEXEC], [MPI command used to run parallel/distributed programs.])
AC_CHECK_PROGS([MPIEXEC], [mpiexec mpirun], [:])
if test "$DOXYGEN" = :; then
  AC_MSG_WARN([MPI execution program was not found.])
fi
AC_SUBST([MPIEXEC])

# Set default error checking
ACX_ENABLE_ERROR_CHECKING

# Check for the presence of doxygen and dot
AC_CHECK_PROGS([DOXYGEN], [doxygen], [:])
AC_CHECK_PROGS([DOT], [dot], [:])
if test "$DOXYGEN" = :; then
  AC_MSG_WARN([Doxygen was not found. You will not be able to build the documentation.])
fi
AC_SUBST([DOXYGEN])
AC_SUBST([DOT])
if test "$DOT" != :; then
  AC_SUBST([HAVE_DOT], [YES])
else
  AC_SUBST([HAVE_DOT], [NO])
fi
AM_SUBST_NOTMAKE([DOXYGEN])
AM_SUBST_NOTMAKE([DOT])
AM_SUBST_NOTMAKE([HAVE_DOT])

# Check for the presence of Eigen header.
AC_CHECK_HEADER([Eigen/Core], [], [
  AC_MSG_ERROR([Unable to find the required Eigen header file.])
])

# Check for the presence of M-A-D-N-E-S-S header and library.
ACX_CHECK_MADNESS

# Check for the presence of boost header files and add the --with-boost-test-lib
# option. 
ACX_CHECK_BOOST
ACX_WITH_BOOST_TEST_LIB
ACX_WITH_BLAS

# Display options
AC_MSG_NOTICE([CPPFLAGS = $CPPFLAGS])
AC_MSG_NOTICE([CXXFLAGS = $CXXFLAGS])
AC_MSG_NOTICE([LDFLAGS  = $LDFLAGS])
AC_MSG_NOTICE([LIBS     = $LIBS])
# Output files
AC_CONFIG_HEADERS([src/config.h:src/config_h.in])
AC_CONFIG_HEADERS([src/TiledArray/config.h:src/TiledArray/config_h.in])
AC_CONFIG_HEADERS([tests/unit_test_config.h:tests/unit_test_config_h.in])
AC_CONFIG_FILES([doc/Doxyfile])
AC_CONFIG_FILES([
  Makefile
    src/Makefile
      src/TiledArray/Makefile
    tests/Makefile
    examples/Makefile
      examples/cc/Makefile
      examples/mpi_tests/Makefile
      examples/dgemm/Makefile
      examples/pmap_test/Makefile
    doc/Makefile])
# There is bug in M4 1.4.6 which is currently the default on OSX which requires
# A new line after AC_OUTPUT
AC_OUTPUT

