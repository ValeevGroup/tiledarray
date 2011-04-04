AC_DEFUN([ACX_WITH_BOOST_TEST_LIB],[
  AC_ARG_WITH([boost-test-lib], [
    AS_HELP_STRING([--with-boost-test-lib@<:@=lib-name@:>@],
      [use Boost.Test library. You may specify a certain library e.g. --with-boost-test-lib=boost_unit_test_framework-gcc-mt ])],
    [
      case $withval in
      yes)
        acx_with_boost_test_lib=yes
      ;;
      no)
        acx_with_boost_test_lib=no
      ;;
      *)
        acx_with_boost_test_lib=$withval
      ;;
      esac
    ],
    [acx_with_boost_test_lib=yes]
  )
  
  AC_CHECK_HEADERS([boost/test/unit_test.hpp boost/test/output_test_stream.hpp], 
    [], [AC_MSG_ERROR([Unable to find Boost Test header file.])])
  
  case "$acx_with_boost_test_lib" in
  yes)
    AC_MSG_CHECKING([for Boost.Test lib])
    acx_with_boost_test_lib_LIBS_save=$LIBS
    LIBS="-lboost_unit_test_framework $LIBS"
    AC_LINK_IFELSE(
      [
        AC_LANG_PROGRAM([[
            #define BOOST_TEST_DYN_LINK
            #include <boost/test/unit_test.hpp>
            using boost::unit_test::test_suite;
            test_suite* init_unit_test_suite( int argc, char * argv[] ) {
            test_suite* test= BOOST_TEST_SUITE( "Unit test example 1" );
            return test;
            }
          ]], [[ return 0;]])
      ],
      [
        AC_DEFINE([BOOST_TEST_DYN_LINK], [1], [Defines the boost unit test framework linkage.])
        AC_MSG_RESULT([yes])
      ],
      [
        LIBS=$acx_with_boost_test_lib_LIBS_save
        AC_MSG_RESULT([no])
        AC_CHECK_HEADER([boost/test/included/unit_test.hpp],
          [AC_MSG_NOTICE([Unable to find a suitable Boost.Test lib file. Using header-only version of Boost.Test.])], 
          [AC_MSG_ERROR([Unable to find Boost.Test header file.])])
      ]
    )
  ;;
  no)
    # We are not using the library so check for the header.
    AC_CHECK_HEADER([boost/test/included/unit_test.hpp], [], 
      [AC_MSG_ERROR([Unable to find Boost.Test header file.])])
  
  ;;
  *)
    AC_MSG_CHECKING([for Boost.Test lib])
    AC_DEFINE([BOOST_TEST_DYN_LINK], [1], [Defines the boost unit test framework linkage.])
    LIBS="-l$acx_with_boost_test_lib $LIBS"
    AC_LINK_IFELSE(
      [
        AC_LANG_PROGRAM([[
            #define BOOST_TEST_DYN_LINK
            #include <boost/test/unit_test.hpp>
            using boost::unit_test::test_suite;
            test_suite* init_unit_test_suite( int argc, char * argv[] ) {
            test_suite* test= BOOST_TEST_SUITE( "Unit test example 1" );
            return test;
            }
          ]], [[ return 0;]])
      ], [AC_MSG_RESULT([yes])],
      [
        AC_MSG_RESULT([yes])
        AC_MSG_ERROR([Unable to find a suitable Boost.Test lib file.])
      ]
    )
  ;;
  esac
])