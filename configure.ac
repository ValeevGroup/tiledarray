# Initialize autoconf
AC_INIT([TiledArray], [0.0], [tiledarray@vt.edu])
AC_CONFIG_SRCDIR([src/tiled_array.h])
AC_CONFIG_MACRO_DIR([config])
AC_CONFIG_AUX_DIR([build-aux])
AC_LANG([C++])

# Initialize automake
AM_INIT_AUTOMAKE([ -Wall foreign ])

# Check for BSD style install programs
AC_PROG_INSTALL

# Set compiler standard
ACX_ENABLE_CXXFLAGS_ARG([cplusplus0x], [-std], [c++0x], [no])

# Check for C++0x language features
ACX_CHECK_VARIADIC_TEMPLATE
ACX_CHECK_RVALUE_REF
ACX_CHECK_STATIC_ASSERT

# Set debugging
ACX_ENABLE_CXXFLAGS_OPT([debugging], [-g], [], [no])

# Set optimization
ACX_ENABLE_CXXFLAGS_OPT([optimization], [-O], [2], [yes])

# Check for mpi c++ compiler.
initial_CFLAGS="$CFLAGS"
initial_CXXFLAGS="$CXXFLAGS"

ACX_PROG_MPICXX
AC_PROG_CXX([$MPICXX])

CFLAGS="$initial_CFLAGS"
CXXFLAGS="$initial_CXXFLAGS"

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

# Check for the presence of doxygen
AC_CHECK_PROGS([DOXYGEN], [doxygen], [:])
if test "$DOXYGEN" = :; then
  AC_MSG_WARN([Doxygen was not found. You will not be able to build the documentation.])
fi
AC_SUBST([DOXYGEN])

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

ACX_WITH_CBLAS

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
      examples/ta_dgemm/Makefile
    doc/Makefile])
AC_OUTPUT