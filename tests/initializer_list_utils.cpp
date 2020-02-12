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
#include "array_fixture.h"
#include "TiledArray/initializer_list_utils.h"
#include "unit_test_config.h"
#include <boost/mpl/find.hpp>
#include <complex>

using namespace TiledArray;

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
    BOOST_CHECK(!IsInitializerList<T>::value);
  }

  // Test is_initializer_list makes sure TA::IsInitializerList correctly
  // recognizes that initializer lists of common tensor element types, such as
  // float and double, are indeed initializer lists. Nestings of up to 3
  // initializer lists are tested.

  BOOST_AUTO_TEST_CASE_TEMPLATE(vector, T, vector_type_list){
    BOOST_CHECK(IsInitializerList<T>::value);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(matrix, T, matrix_type_list){
    BOOST_CHECK(IsInitializerList<T>::value);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(tensor3, T, tensor3_type_list){
    BOOST_CHECK(IsInitializerList<T>::value);
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
    BOOST_CHECK(!is_initializer_list_v<T>);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(vector, T, vector_type_list){
    BOOST_CHECK(is_initializer_list_v<T>);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(matrix, T, matrix_type_list){
    BOOST_CHECK(is_initializer_list_v<T>);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(tensor3, T, tensor3_type_list){
    BOOST_CHECK(is_initializer_list_v<T>);
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
    BOOST_CHECK_EQUAL(InitializerListRank<T>::value, 0);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(vector, T, vector_type_list){
    BOOST_CHECK_EQUAL(InitializerListRank<T>::value, 1);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(matrix, T, matrix_type_list){
    BOOST_CHECK_EQUAL(InitializerListRank<T>::value, 2);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(tensor3, T, tensor3_type_list){
    BOOST_CHECK_EQUAL(InitializerListRank<T>::value, 3);
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
    BOOST_CHECK_EQUAL(initializer_list_rank_v<T>, 0);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(vector, T, vector_type_list){
    BOOST_CHECK_EQUAL(initializer_list_rank_v<T>, 1);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(matrix, T, matrix_type_list){
    BOOST_CHECK_EQUAL(initializer_list_rank_v<T>, 2);
  }

  BOOST_AUTO_TEST_CASE_TEMPLATE(tensor3, T, tensor3_type_list){
    BOOST_CHECK_EQUAL(initializer_list_rank_v<T>, 3);
  }

BOOST_AUTO_TEST_SUITE_END()

//------------------------------------------------------------------------------
// tiled_range_from_il free function
//------------------------------------------------------------------------------

/* These unit tests focus on testing the tiled_range_from_il free function. We
 * ensure that the correct TiledRange is created for:
 *
 * - a scalar
 * - an empty vector
 * - a vector
 * - a square matrix (nrows == ncols)
 * - a tall matrix   (nrows > ncols)
 * - a short matrix  (nrows < ncols)
 * - a square rank3 tensor   (dim[0] == dim[1] == dim[2])
 * - oblate rank3 tensor     (dim[0] < dim[1] == dim[2])
 * - prolate rank3 tensor    (dim[0] > dim[1] == dim[2])
 * - asymmetric rank3 tensor (dim[0] < dim[1] < dim[2])
 */

BOOST_AUTO_TEST_SUITE(tiled_range_from_il_fxn)

  BOOST_AUTO_TEST_CASE(scalar){
    TiledRange corr{};
    double il{};
    BOOST_CHECK_EQUAL(tiled_range_from_il(il), corr);
  }

  BOOST_AUTO_TEST_CASE(empty_vector){
    vector_il<double> il{};
    BOOST_CHECK_THROW(tiled_range_from_il(il), Exception);
  }

  BOOST_AUTO_TEST_CASE(vector){
    TiledRange corr{{0, 3}};
    vector_il<double> il{1, 2, 3};
    BOOST_CHECK_EQUAL(tiled_range_from_il(il), corr);
  }

  BOOST_AUTO_TEST_CASE(square_matrix){
    TiledRange corr{{0, 3}, {0, 3}};
    matrix_il<double> il{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    BOOST_CHECK_EQUAL(tiled_range_from_il(il), corr);
  }

  BOOST_AUTO_TEST_CASE(tall_matrix){
    TiledRange corr{{0, 4}, {0, 3}};
    matrix_il<double> il{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    BOOST_CHECK_EQUAL(tiled_range_from_il(il), corr);
  }

  BOOST_AUTO_TEST_CASE(short_matrix){
    TiledRange corr{{0, 2}, {0, 3}};
    matrix_il<double> il{{1, 2, 3}, {4, 5, 6}};
    BOOST_CHECK_EQUAL(tiled_range_from_il(il), corr);
  }

  BOOST_AUTO_TEST_CASE(square_rank3){
    TiledRange corr{{0, 2}, {0, 2}, {0, 2}};
    tensor3_il<double> il{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    BOOST_CHECK_EQUAL(tiled_range_from_il(il), corr);
  }

  BOOST_AUTO_TEST_CASE(oblate_rank3){
    TiledRange corr{{0, 1}, {0, 2}, {0, 2}};
    tensor3_il<double> il{{{1, 2}, {3, 4}}};
    BOOST_CHECK_EQUAL(tiled_range_from_il(il), corr);
  }

  BOOST_AUTO_TEST_CASE(prolate_rank3){
    TiledRange corr{{0, 2}, {0, 1}, {0, 1}};
    tensor3_il<double> il{{{1}}, {{2}}};
    BOOST_CHECK_EQUAL(tiled_range_from_il(il), corr);
  }

  BOOST_AUTO_TEST_CASE(asymmetric_rank3){
    TiledRange corr{{0, 1}, {0, 2}, {0, 3}};
    tensor3_il<double> il{{{1, 2, 3}, {4, 5, 6}}};
    BOOST_CHECK_EQUAL(tiled_range_from_il(il), corr);
  }

BOOST_AUTO_TEST_SUITE_END()

//------------------------------------------------------------------------------
// flatten_il free function
//------------------------------------------------------------------------------

/* These unit tests focus on testing the flatten_il free function. We ensure
 * that the function correctly returns an iterator pointing to just past the
 * last inserted element and that the container contains the tensor in row-major
 * format. These tests are run for:
 *
 * - a scalar
 * - an empty vector
 * - a vector
 * - a bad matrix (nested initializer lists are not the same length)
 * - an empty matrix
 * - a square matrix (nrows == ncols)
 * - a tall matrix   (nrows > ncols)
 * - a short matrix  (nrows < ncols)
 * - an empty rank3 tensor
 * - a square rank3 tensor   (dim[0] == dim[1] == dim[2])
 * - oblate rank3 tensor     (dim[0] < dim[1] == dim[2])
 * - prolate rank3 tensor    (dim[0] > dim[1] == dim[2])
 * - asymmetric rank3 tensor (dim[0] < dim[1] < dim[2])
 */

BOOST_AUTO_TEST_SUITE(flatten_il_fxn)

  BOOST_AUTO_TEST_CASE(scalar){
    std::array<double, 1> buffer{}, corr{};
    double il = 3.14;
    BOOST_CHECK(flatten_il(il, buffer.begin()) == buffer.begin());
    BOOST_CHECK(buffer == corr);
  }

  BOOST_AUTO_TEST_CASE(empty_vector){
    std::array<double, 0> corr{};
    std::array<double, 0> buffer{};
    vector_il<double> il{};
    BOOST_CHECK(flatten_il(il, buffer.begin()) == buffer.end());
    BOOST_CHECK(buffer == corr);
  }

  BOOST_AUTO_TEST_CASE(vector){
     std::array<double, 3> corr{1, 2, 3};
     std::array<double, 3> buffer{};
     vector_il<double> il{1, 2, 3};
     BOOST_CHECK(flatten_il(il, buffer.begin()) == buffer.end());
     BOOST_CHECK(buffer == corr);
  }

  BOOST_AUTO_TEST_CASE(bad_matrix){
    std::array<double, 5> buffer{};
    matrix_il<double> il{{1, 2}, {3, 4, 5}};
    BOOST_CHECK_THROW(flatten_il(il, buffer.begin()), Exception);
  }

  BOOST_AUTO_TEST_CASE(empty_matrix){
    std::array<double, 0> corr{};
    std::array<double, 0> buffer{};
    matrix_il<double> il{{}};
    BOOST_CHECK(flatten_il(il, buffer.begin()) == buffer.end());
    BOOST_CHECK(buffer == corr);
  }

  BOOST_AUTO_TEST_CASE(square_matrix){
    std::array<double, 9> corr{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<double, 9> buffer{};
    matrix_il<double> il{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    BOOST_CHECK(flatten_il(il, buffer.begin())== buffer.end());
    BOOST_CHECK(buffer == corr);
  }

  BOOST_AUTO_TEST_CASE(tall_matrix){
    std::array<double, 12> corr{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::array<double, 12> buffer{};
    matrix_il<double> il{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    BOOST_CHECK(flatten_il(il, buffer.begin())== buffer.end());
    BOOST_CHECK(buffer == corr);
  }

  BOOST_AUTO_TEST_CASE(short_matrix){
    std::array<double, 6> corr{1, 2, 3, 4, 5, 6};
    std::array<double, 6> buffer{};
    matrix_il<double> il{{1, 2, 3}, {4, 5, 6}};
    BOOST_CHECK(flatten_il(il, buffer.begin())== buffer.end());
    BOOST_CHECK(buffer == corr);
  }

  BOOST_AUTO_TEST_CASE(empty_rank3){
    std::array<double, 0> corr{};
    std::array<double, 0> buffer{};
    tensor3_il<double> il{{{}}};
    BOOST_CHECK(flatten_il(il, buffer.begin()) == buffer.end());
    BOOST_CHECK(buffer == corr);
  }

  BOOST_AUTO_TEST_CASE(square_rank3){
    std::array<double, 8> corr{1, 2, 3, 4, 5, 6, 7, 8};
    std::array<double, 8> buffer{};
    tensor3_il<double> il{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    BOOST_CHECK(flatten_il(il, buffer.begin())== buffer.end());
    BOOST_CHECK(buffer == corr);
  }

  BOOST_AUTO_TEST_CASE(oblate_rank3){
    std::array<double, 4> corr{1, 2, 3, 4};
    std::array<double, 4> buffer{};
    tensor3_il<double> il{{{1, 2}, {3, 4}}};
    BOOST_CHECK(flatten_il(il, buffer.begin())== buffer.end());
    BOOST_CHECK(buffer == corr);
  }

  BOOST_AUTO_TEST_CASE(prolate_rank3){
    std::array<double, 2> corr{1, 2};
    std::array<double, 2> buffer{};
    tensor3_il<double> il{{{1}}, {{2}}};
    BOOST_CHECK(flatten_il(il, buffer.begin())== buffer.end());
    BOOST_CHECK(buffer == corr);
  }

  BOOST_AUTO_TEST_CASE(asymmetric_rank3){
    std::array<double, 6> corr{1, 2, 3, 4, 5, 6};
    std::array<double, 6> buffer{};
    tensor3_il<double> il{{{1, 2, 3}, {4, 5, 6}}};
    BOOST_CHECK(flatten_il(il, buffer.begin())== buffer.end());
    BOOST_CHECK(buffer == corr);
  }

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(array_from_il_fxn, ArrayFixture)

  BOOST_AUTO_TEST_CASE(scalar){
    double il = 3.14;
    // Uncommenting this line should cause a static assert failure
    // array_from_il<DistArray<>>(world, il);
  }

  BOOST_AUTO_TEST_CASE(vector){
     vector_il<double> il{1, 2, 3};
     array_from_il<DistArray<>>(world, il);
  }

BOOST_AUTO_TEST_SUITE_END()