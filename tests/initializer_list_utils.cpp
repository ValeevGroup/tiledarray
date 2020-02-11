/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "TiledArray/initializer_list_utils.h"
#include "unit_test_config.h"
#include <boost/mpl/find.hpp>
#include <complex>

//------------------------------------------------------------------------------
// Declare some typedefs which will make the unit tests more readable
//------------------------------------------------------------------------------

// Typedef to make declaring initializer_lists tolerable
template<typename T> using il = std::initializer_list<T>;

// Typedef of an initializer list for a vector
template<typename T> using vector_il = il<T>;

// Typedef of an initializer list for a matrix
template<typename T> using matrix_il = il<il<T>>;

// Typedef of an il for a rank 3 tensor
template<typename T> using tensor3_il = il<il<il<T>>>;

// Typedef of single-precision complex number
using complexf = std::complex<float>;

// Typedef of double-precision complex number
using complexd = std::complex<double>;

// List of initializer_list types that map to rank 0 tensors
using scalar_type_list = boost::mpl::list<float, double, complexf, complexd>;

// List of initializer_list types that map to vectors
using vector_type_list = boost::mpl::list<vector_il<float>,
                                         vector_il<double>,
                                         vector_il<complexf>,
                                         vector_il<complexd>>;

// List of initializer_list types that map to matrices
using matrix_type_list = boost::mpl::list<matrix_il<float>,
                                          matrix_il<double>,
                                          matrix_il<complexf>,
                                          matrix_il<complexd>>;

// List of initializer_list types that map to rank 3 tensors
using tensor3_type_list = boost::mpl::list<tensor3_il<float>,
                                         tensor3_il<double>,
                                         tensor3_il<complexf>,
                                         tensor3_il<complexd>>;

//------------------------------------------------------------------------------
// Unit tests for struct: IsInitializerList
//------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE(is_initializer_list_class)

  // This test makes sure TA::IsInitializerList correctly recognizes that common
  // tensor element types, such as float and double, are not initializer lists
  BOOST_AUTO_TEST_CASE_TEMPLATE(scalar, T, scalar_type_list){
    BOOST_CHECK(!TiledArray::IsInitializerList<T>::value);
  }

  // Test is_initializer_list makes sure TA::IsInitializerList correctly
  // recognizes that initializer lists of common tensor element types, such as
  // float and double, are indeed initializer lists. Nestings of up to 3
  // initializer lists are tested.

  BOOST_AUTO_TEST_CASE_TEMPLATE(vector, T, vector_type_list){
    BOOST_CHECK(TiledArray::IsInitializerList<T>::value);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(matrix, T, matrix_type_list){
    BOOST_CHECK(TiledArray::IsInitializerList<T>::value);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(tensor3, T, tensor3_type_list){
    BOOST_CHECK(TiledArray::IsInitializerList<T>::value);
  }

BOOST_AUTO_TEST_SUITE_END()

//------------------------------------------------------------------------------
// is_initializer_list_v helper variable
//------------------------------------------------------------------------------

// Test is_initializer_list_helper_variable tests that the helper variable
// is_initializer_list_v<T> correctly aliases IsInitializerList<T>::value for
// std::initializer_list types consistent with a scalar, vector, matrix, and a
// rank 3 tensor.

BOOST_AUTO_TEST_SUITE(is_initializer_list_helper)

  BOOST_AUTO_TEST_CASE_TEMPLATE(scalar, T, scalar_type_list){
    BOOST_CHECK(!TiledArray::is_initializer_list_v<T>);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(vector, T, vector_type_list){
    BOOST_CHECK(TiledArray::is_initializer_list_v<T>);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(matrix, T, matrix_type_list){
    BOOST_CHECK(TiledArray::is_initializer_list_v<T>);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(tensor3, T, tensor3_type_list){
    BOOST_CHECK(TiledArray::is_initializer_list_v<T>);
  }

BOOST_AUTO_TEST_SUITE_END()

//------------------------------------------------------------------------------
// Unit tests for struct: InitializerListRank
//------------------------------------------------------------------------------

// The following four tests respectively test that InitializerListRank correctly
// determines that an initializer lists consistent with a scalar, vector,
// matrix, and rank 3 tensor are of ranks 0, 1, 2, and 3
BOOST_AUTO_TEST_SUITE(initializer_list_rank_class)

  BOOST_AUTO_TEST_CASE_TEMPLATE(scalar, T, scalar_type_list){
    BOOST_CHECK_EQUAL(TiledArray::InitializerListRank<T>::value, 0);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(vector, T, vector_type_list){
    BOOST_CHECK_EQUAL(TiledArray::InitializerListRank<T>::value, 1);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(matrix, T, matrix_type_list){
    BOOST_CHECK_EQUAL(TiledArray::InitializerListRank<T>::value, 2);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(tensor3, T, tensor3_type_list){
    BOOST_CHECK_EQUAL(TiledArray::InitializerListRank<T>::value, 3);
  }

BOOST_AUTO_TEST_SUITE_END()

//------------------------------------------------------------------------------
// initializer_list_rank_v helper variable
//------------------------------------------------------------------------------

// Test initializer_list_rank_helper ensures that the helper variable
// initializer_list_rank_v correctly aliases InitializerListRank<T>::value
// for a scalar, vector, matrix, and rank 3 tensor.
BOOST_AUTO_TEST_SUITE(initializer_list_rank_helper)

  BOOST_AUTO_TEST_CASE_TEMPLATE(scalar, T, scalar_type_list){
    BOOST_CHECK_EQUAL(TiledArray::initializer_list_rank_v<T>, 0);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(vector, T, vector_type_list){
    BOOST_CHECK_EQUAL(TiledArray::initializer_list_rank_v<T>, 1);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(matrix, T, matrix_type_list){
    BOOST_CHECK_EQUAL(TiledArray::initializer_list_rank_v<T>, 2);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(tensor3, T, tensor3_type_list){
    BOOST_CHECK_EQUAL(TiledArray::initializer_list_rank_v<T>, 3);
  }

BOOST_AUTO_TEST_SUITE_END()