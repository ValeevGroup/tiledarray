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
  
  AC_CHECK_HEADERS([boost/test/unit_test.hpp \
                   boost/test/output_test_stream.hpp], 
    [], [AC_MSG_ERROR([Unable to find Boost Test header file.])])
])
