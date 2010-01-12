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
AC_ARG_ENABLE([cpp0x],
  [AC_HELP_STRING([--enable-cpp0x@<:@=standard@:>@], [Enable experimental c++0x standard (e.g. ARG=c++0x) @<:@default=no@:>@.])],
  [
    case $enableval in
      yes)
        CXXFLAGS="$CXXFLAGS -std=c++0x"
      ;;
      no)
      ;;
      *)
        CXXFLAGS="$CXXFLAGS -std=$enableval"
      ;;
    esac
  ]
)

# Set debugging
AC_ARG_ENABLE([debugging],
  [AC_HELP_STRING([--enable-debugging@<:@=ARG@:>@], [Enable debugging information @<:@default=no@:>@.])],
  [
    case $enableval in
      yes)
        CXXFLAGS="$CXXFLAGS -g"
      ;;
      no)
      ;;
      *)
        CXXFLAGS="$CXXFLAGS -g$enableval"
      ;;
    esac
  ]
)

# Set optimization
AC_ARG_ENABLE([optimization],
  [AC_HELP_STRING([--enable-optimization@<:@=ARG@:>@], [Enable code optimization @<:@default=2@:>@.])],
  [
    case $enableval in
      yes)
        CXXFLAGS="$CXXFLAGS -O2"
      ;;
      no)
      ;;
      *)
        CXXFLAGS="$CXXFLAGS -O$enableval"
      ;;
    esac
  ]
)

# Check for mpi c++ compiler.
ACX_PROG_MPICXX
AC_PROG_CXX([$MPICXX])

# Check for mpi execution command
AC_ARG_VAR([MPIEXEC], [MPI command used to run parallel/distributed programs.])
AC_CHECK_PROGS([MPIEXEC], [mpiexec mpirun], [:])
if test "$DOXYGEN" = :; then
  AC_MSG_WARN([MPI execution program was not found.])
fi
AC_SUBST([MPIEXEC])

# Set default error checking
TA_DEFAULT_ERROR=3
AC_ARG_ENABLE([error-checking],
  [AC_HELP_STRING([--enable-error-checking@<:@=throw|assert|no@:>@], [Enable default error checking@<:@default=throw@:>@.])],
  [
    case $enableval in
      yes)
        AC_DEFINE([TA_DEFAULT_ERROR], [1], [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
      ;;
      throw)
        AC_DEFINE([TA_DEFAULT_ERROR], [1], [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
      ;;
      assert)
        AC_DEFINE([TA_DEFAULT_ERROR], [2], [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
      ;;
      no)
        AC_DEFINE([TA_DEFAULT_ERROR], [0], [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
      ;;
      *)
        AC_MSG_ERROR([Invalid input for error checking.])
      ;;
    esac   
  ],
  [AC_DEFINE([TA_DEFAULT_ERROR], [0], [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])]
)

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
AC_CHECK_HEADER([world/world.h], [], [
  AC_MSG_ERROR([Unable to find the required M-A-D-N-E-S-S header file.])
])

ACX_CXX_CHECK_LIB([MADworld], [madness::initialize(int argc = 0, char** argv = 0)],
  [LIBS="$LIBS -lMADworld"], [AC_MSG_ERROR([The required library MADworld could not be found.])])


# Check for the presence of Boost headers.
AC_CHECK_HEADERS([boost/shared_ptr.hpp \
                  boost/make_shared.hpp \
                  boost/type_traits.hpp \
                  boost/iterator/transform_iterator.hpp \
                  boost/iterator/zip_iterator.hpp \
                  boost/functional.hpp \
                  boost/array.hpp \
                  boost/operators.hpp \
                  boost/scoped_array.hpp \
                  boost/iterator/iterator_facade.hpp \
                  boost/tuple/tuple.hpp],
                  [], [
                    AC_MSG_ERROR([Unable to find one or more of the required Boost header files.])
                  ])

AC_CHECK_HEADER([boost/test/unit_test.hpp], [], [AC_MSG_ERROR([Unable to find Boost Test header file.])])

AC_CHECK_LIB([boost_unit_test_framework], [main],
  [
    AC_DEFINE([TA_UNIT_TEST_DYLINK], [1], [Defines the boost unit test framework linkage.])
    LIBS="-lboost_unit_test_framework $LIBS"
  ], [
    AC_CHECK_HEADER([boost/test/included/unit_test.hpp], [], [AC_MSG_ERROR([Unable to find Boost Test header file.])])
    AC_MSG_WARN([The Boost unit test framework library was not found. You will still be able to compile and run unit tests, but it will take more time to compile.]) 
  ]
)

AC_ARG_WITH([blas],
            [AS_HELP_STRING([--with-blas@<:@=DIR@:>@], [DIR where the blas library is installed.])],
            [
              case $with_blas in
                yes)
                ;;
                no)
                ;;
                *)
                ;;
              esac
              if test "$with_blas" != yes; then
                CPPFLAGS="$CPPFLAGS -I$with_blas/include"
                LDFLAGS="$LDFLAGS -L$with_blas/lib"
              fi
              AC_CHECK_HEADER([cblas.h], [], [AC_MSG_ERROR([Unable to find the cblas header file.])])
              AC_DEFINE([TA_USE_CBLAS], [1], [TiledArray will use blas for some math operations.])
            ])


#define config.h top and bottom
AH_TOP([#ifndef TILEDARRAY_CONFIGURE_H__INCULDED
#define TILEDARRAY_CONFIGURE_H__INCULDED

#include <TiledArray/package.h>
])
AH_BOTTOM([#endif // TILEDARRAY_CONFIGURE_H__INCULDED])

# Display options
AC_MSG_NOTICE([CPPFLAGS = $CPPFLAGS])
AC_MSG_NOTICE([CXXFLAGS = $CXXFLAGS])
AC_MSG_NOTICE([LDFLAGS = $LDFLAGS])
AC_MSG_NOTICE([LIBS = $LIBS])
# Output files
AC_CONFIG_HEADERS([src/TiledArray/config.h:src/TiledArray/config_h.in])
AC_CONFIG_FILES([doc/Doxyfile])
AC_CONFIG_FILES([Makefile src/Makefile src/TiledArray/Makefile Tests/Makefile doc/Makefile])
AC_OUTPUT