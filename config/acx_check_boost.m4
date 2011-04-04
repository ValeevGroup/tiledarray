AC_DEFUN([ACX_CHECK_BOOST_TEST_LIB], [
  AC_SEARCH_LIBS([main], [boost_unit_test_framework boost_unit_test_framework-mt],
    [
      AC_DEFINE([BOOST_TEST_DYN_LINK], [1], [Defines the boost unit test framework linkage.])
      LIBS="-l$ac_cv_search_main $LIBS"
    ]
  )
  
  if test "$ac_cv_search_main" = no; then
    AC_CHECK_HEADER([boost/test/included/unit_test.hpp], [], 
      [AC_MSG_ERROR([Unable to find Boost Test header file.])])
  fi
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
])
