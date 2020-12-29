/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2017  Virginia Tech
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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  type_traits.cpp
 *  Apr 7, 2017
 *
 */

#include "unit_test_config.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "TiledArray/dist_eval/array_eval.h"
#include "TiledArray/tensor.h"
#include "TiledArray/tile_op/noop.h"
#include "TiledArray/type_traits.h"

struct TypeTraitsFixture {
  TypeTraitsFixture() {}

  ~TypeTraitsFixture() {}

};  // TypeTraitsFixture

BOOST_FIXTURE_TEST_SUITE(type_traits_suite, TypeTraitsFixture,
                         TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(sanity) {
  constexpr bool double_has_value_type =
      TiledArray::detail::has_member_type_value_type<double>::value;
  BOOST_CHECK(!double_has_value_type);
}

BOOST_AUTO_TEST_CASE(vector) {
  using container = std::vector<int>;

  constexpr bool has_value_type =
      TiledArray::detail::has_member_type_value_type<container>::value;
  BOOST_CHECK(has_value_type);
  constexpr bool has_allocator_type =
      TiledArray::detail::has_member_type_allocator_type<container>::value;
  BOOST_CHECK(has_allocator_type);
  constexpr bool has_size_type =
      TiledArray::detail::has_member_type_size_type<container>::value;
  BOOST_CHECK(has_size_type);
  constexpr bool has_difference_type =
      TiledArray::detail::has_member_type_difference_type<container>::value;
  BOOST_CHECK(has_difference_type);
  constexpr bool has_reference =
      TiledArray::detail::has_member_type_reference<container>::value;
  BOOST_CHECK(has_reference);
  constexpr bool has_const_reference =
      TiledArray::detail::has_member_type_const_reference<container>::value;
  BOOST_CHECK(has_const_reference);
  constexpr bool has_pointer =
      TiledArray::detail::has_member_type_pointer<container>::value;
  BOOST_CHECK(has_pointer);
  constexpr bool has_const_pointer =
      TiledArray::detail::has_member_type_const_pointer<container>::value;
  BOOST_CHECK(has_const_pointer);
  constexpr bool has_iterator =
      TiledArray::detail::has_member_type_iterator<container>::value;
  BOOST_CHECK(has_iterator);
  constexpr bool has_const_iterator =
      TiledArray::detail::has_member_type_const_iterator<container>::value;
  BOOST_CHECK(has_const_iterator);
  constexpr bool has_reverse_iterator =
      TiledArray::detail::has_member_type_reverse_iterator<container>::value;
  BOOST_CHECK(has_reverse_iterator);
  constexpr bool has_const_reverse_iterator =
      TiledArray::detail::has_member_type_const_reverse_iterator<
          container>::value;
  BOOST_CHECK(has_const_reverse_iterator);

  constexpr bool has_size =
      TiledArray::detail::has_member_function_size_anyreturn<container>::value;
  constexpr bool has_size_const =
      TiledArray::detail::has_member_function_size_anyreturn<
          const container>::value;
  constexpr bool has_size_size_type =
      TiledArray::detail::has_member_function_size<container,
                                                   container::size_type>::value;
  constexpr bool has_size_size_type_const =
      TiledArray::detail::has_member_function_size<const container,
                                                   container::size_type>::value;
  constexpr bool has_size_bool =
      TiledArray::detail::has_member_function_size<const container,
                                                   bool>::value;
  BOOST_CHECK(has_size && has_size_const);
  BOOST_CHECK(has_size_size_type);
  BOOST_CHECK(has_size_size_type_const);
  BOOST_CHECK(!has_size_bool);

  constexpr bool has_empty =
      TiledArray::detail::has_member_function_empty_anyreturn<container>::value;
  constexpr bool has_empty_bool =
      TiledArray::detail::has_member_function_empty<container, bool>::value;
  BOOST_CHECK(has_empty);
  BOOST_CHECK(has_empty_bool);

  constexpr bool has_clear =
      TiledArray::detail::has_member_function_clear_anyreturn<container>::value;
  constexpr bool has_clear_const =
      TiledArray::detail::has_member_function_clear_anyreturn<
          const container>::value;
  constexpr bool has_clear_void =
      TiledArray::detail::has_member_function_clear<container, void>::value;
  constexpr bool has_clear_void_const =
      TiledArray::detail::has_member_function_clear<const container,
                                                    void>::value;
  constexpr bool has_clear_bool =
      TiledArray::detail::has_member_function_clear<container, bool>::value;
  constexpr bool has_clear_bool_const =
      TiledArray::detail::has_member_function_clear<const container,
                                                    bool>::value;
  BOOST_CHECK(has_clear);
  BOOST_CHECK(!has_clear_const);
  BOOST_CHECK(has_clear_void);
  BOOST_CHECK(!has_clear_void_const);
  BOOST_CHECK(!has_clear_bool);
  BOOST_CHECK(!has_clear_bool_const);

  constexpr bool has_begin =
      TiledArray::detail::has_member_function_begin_anyreturn<container>::value;
  constexpr bool has_cbegin =
      TiledArray::detail::has_member_function_cbegin_anyreturn<
          container>::value;
  constexpr bool has_rbegin =
      TiledArray::detail::has_member_function_rbegin_anyreturn<
          container>::value;
  constexpr bool has_crbegin =
      TiledArray::detail::has_member_function_crbegin_anyreturn<
          container>::value;
  constexpr bool has_end =
      TiledArray::detail::has_member_function_end_anyreturn<container>::value;
  constexpr bool has_cend =
      TiledArray::detail::has_member_function_cend_anyreturn<container>::value;
  constexpr bool has_rend =
      TiledArray::detail::has_member_function_rend_anyreturn<container>::value;
  constexpr bool has_crend =
      TiledArray::detail::has_member_function_crend_anyreturn<container>::value;
  BOOST_CHECK(has_begin);
  BOOST_CHECK(has_cbegin);
  BOOST_CHECK(has_rbegin);
  BOOST_CHECK(has_crbegin);
  BOOST_CHECK(has_end);
  BOOST_CHECK(has_cend);
  BOOST_CHECK(has_rend);
  BOOST_CHECK(has_crend);
}

BOOST_AUTO_TEST_CASE(unordered_map) {
  using container = std::unordered_map<int, int>;
  constexpr bool has_begin =
      TiledArray::detail::has_member_function_begin_anyreturn<container>::value;
  constexpr bool has_cbegin =
      TiledArray::detail::has_member_function_cbegin_anyreturn<
          container>::value;
  constexpr bool has_rbegin =
      TiledArray::detail::has_member_function_rbegin_anyreturn<
          container>::value;
  constexpr bool has_crbegin =
      TiledArray::detail::has_member_function_crbegin_anyreturn<
          container>::value;
  constexpr bool has_end =
      TiledArray::detail::has_member_function_end_anyreturn<container>::value;
  constexpr bool has_cend =
      TiledArray::detail::has_member_function_cend_anyreturn<container>::value;
  constexpr bool has_rend =
      TiledArray::detail::has_member_function_rend_anyreturn<container>::value;
  constexpr bool has_crend =
      TiledArray::detail::has_member_function_crend_anyreturn<container>::value;
  BOOST_CHECK(has_begin);
  BOOST_CHECK(has_cbegin);
  BOOST_CHECK(!has_rbegin);
  BOOST_CHECK(!has_crbegin);
  BOOST_CHECK(has_end);
  BOOST_CHECK(has_cend);
  BOOST_CHECK(!has_rend);
  BOOST_CHECK(!has_crend);
}

using std::max;
GENERATE_IS_FREE_FUNCTION_ANYRETURN(max)

BOOST_AUTO_TEST_CASE(_max_) {
  constexpr bool there_is_max_int_int =
      is_free_function_max_anyreturn_v<int, int>;
  constexpr bool there_is_max_int_double =
      is_free_function_max_anyreturn_v<int, double>;
  BOOST_CHECK(there_is_max_int_int);
  BOOST_CHECK(!there_is_max_int_double);
}

using std::min;
GENERATE_IS_FREE_FUNCTION_ANYRETURN(min)

BOOST_AUTO_TEST_CASE(_min_) {
  constexpr bool there_is_min_int_int =
      is_free_function_min_anyreturn_v<int, int>;
  constexpr bool there_is_min_int_double =
      is_free_function_min_anyreturn_v<int, double>;
  BOOST_CHECK(there_is_min_int_int);
  BOOST_CHECK(!there_is_min_int_double);
}

struct IncompleteType;
struct CompleteType {};

BOOST_AUTO_TEST_CASE(is_complete_type) {
  constexpr bool void_is_complete_type =
      TiledArray::detail::is_complete_type_v<void>;
  BOOST_CHECK(!void_is_complete_type);
  constexpr bool IncompleteType_is_complete_type =
      TiledArray::detail::is_complete_type_v<IncompleteType>;
  BOOST_CHECK(!IncompleteType_is_complete_type);
  constexpr bool CompleteType_is_complete_type =
      TiledArray::detail::is_complete_type_v<CompleteType>;
  BOOST_CHECK(CompleteType_is_complete_type);
}

BOOST_AUTO_TEST_CASE(convertibility) {
  using TileD = TiledArray::Tensor<double>;
  using LazyTileD = TiledArray::detail::LazyArrayTile<
      TileD, TiledArray::detail::Noop<TileD, TileD, true>>;
  {
    using T = LazyTileD;
    constexpr bool lazy_tile_is_explconv_to_tile =
        TiledArray::detail::is_explicitly_convertible<
            T, typename T::eval_type>::value;
    constexpr bool lazy_tile_is_implconv_to_tile =
        TiledArray::detail::is_implicitly_convertible<
            T, typename T::eval_type>::value;
    BOOST_CHECK(lazy_tile_is_explconv_to_tile);
    BOOST_CHECK(!lazy_tile_is_implconv_to_tile);
    constexpr bool lazy_tile_is_explconv_to_tilefut =
        TiledArray::detail::is_explicitly_convertible<
            T, madness::Future<typename T::eval_type>>::value;
    constexpr bool lazy_tile_is_implconv_to_tilefut =
        TiledArray::detail::is_implicitly_convertible<
            T, madness::Future<typename T::eval_type>>::value;
    constexpr bool lazy_tile_has_explconv_to_tilefut =
        TiledArray::detail::has_conversion_operator_v<
            T, madness::Future<typename T::eval_type>>;
    BOOST_CHECK(!lazy_tile_is_explconv_to_tilefut);
    BOOST_CHECK(!lazy_tile_is_implconv_to_tilefut);
    BOOST_CHECK(!lazy_tile_has_explconv_to_tilefut);

    constexpr bool lazy_tile_has_conversion_operator_to_tile =
        TiledArray::detail::has_conversion_operator_v<T, TileD>;
    constexpr bool lazy_tile_has_conversion_operator_to_tilefut =
        TiledArray::detail::has_conversion_operator_v<T,
                                                      madness::Future<TileD>>;
    BOOST_CHECK(lazy_tile_has_conversion_operator_to_tile);
    BOOST_CHECK(!lazy_tile_has_conversion_operator_to_tilefut);
  }
}

BOOST_AUTO_TEST_SUITE_END()
