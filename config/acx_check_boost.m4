AC_DEFUN([ACX_CHECK_BOOST_TEST_LIB], [
  AC_CHECK_LIB([boost_unit_test_framework], [main], [
      AC_DEFINE([BOOST_TEST_DYN_LINK], [1], [Defines the boost unit test framework linkage.])
      LIBS="-lboost_unit_test_framework $LIBS"
    ], [AC_CHECK_HEADER([boost/test/included/unit_test.hpp], [], [AC_MSG_ERROR([Unable to find Boost Test header file.])])]
  )
])

AC_DEFUN([ACX_CHECK_BOOST], [
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
    [], [AC_MSG_ERROR([Unable to find one or more of the required Boost header files.]) ])
  
  AC_CHECK_HEADERS([boost/test/unit_test.hpp \
                   boost/test/output_test_stream.hpp], 
    [], [AC_MSG_ERROR([Unable to find Boost Test header file.])])
  
  AC_ARG_WITH([boost-test-lib],
              [AS_HELP_STRING([--with-boost-test-lib@<:@=yes|no|check@:>@],
                [Link with Boost Unit Test Framework library @<:@default=check@:>@.])
              ], [
                case $with_boost_test_lib in
                  yes)
                    AC_CHECK_HEADER([boost/test/included/unit_test.hpp], [],
                      [AC_MSG_ERROR([Unable to find Boost Test header file.])])
                    AC_CHECK_LIB([boost_unit_test_framework], [main], [
                        AC_DEFINE([BOOST_TEST_DYN_LINK], [1],
                          [Defines the boost unit test framework linkage.])
                        LIBS="-lboost_unit_test_framework $LIBS"
                      ],
                      [AC_MSG_ERROR([Unable to find Boost Unit Test library.])])
                  ;;
                  no)
                    AC_CHECK_HEADER([boost/test/included/unit_test.hpp], [],
                      [AC_MSG_ERROR([Unable to find Boost Test header file.])])
                  ;;
                  check)
                    ACX_CHECK_BOOST_TEST_LIB
                  ;;
                  *)
                    AC_MSG_ERROR([Invalid argument for --with-boost-test-lib, valid arguments are yes, no, or check.])
                  ;;
                esac
              ], [ACX_CHECK_BOOST_TEST_LIB])
])
