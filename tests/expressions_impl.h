/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2019  Virginia Tech
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
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *
 *  expressions_fixture.h
 *  Jan 19, 2019
 *
 */

#ifndef TILEDARRAY_TEST_EXPRESSIONS_IMPL_H
#define TILEDARRAY_TEST_EXPRESSIONS_IMPL_H

constexpr int nrepeats = 5;

BOOST_FIXTURE_TEST_CASE_TEMPLATE(tensor_factories, F, Fixtures, F) {
  auto& a = F::a;
  auto& c = F::c;
  auto& aC = F::aC;

  const auto& ca = a;
  const std::array<int, 3> lobound{{3, 3, 3}};
  const std::array<int, 3> upbound{{5, 5, 5}};

  using TiledArray::eigen::iv;

  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") += a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = c("a,c,b") + a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") -= a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = c("a,c,b") - a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") *= a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = c("a,c,b") * a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a").conj());
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block({3, 3, 3}, {5, 5, 5}));
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block({{3, 5}, {3, 5}, {3, 5}}));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           a("a,b,c").block(boost::combine(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block(iv(3, 3, 3), iv(5, 5, 5)));
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block(
                           iv(3, 3, 3), iv(iv(3, 3, 3) + iv(2, 2, 2))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = ca("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = ca("c,b,a").conj());
  BOOST_CHECK_NO_THROW(c("a,b,c") = ca("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") = ca("a,b,c").block({3, 3, 3}, {5, 5, 5}));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           ca("a,b,c").block({{3, 5}, {3, 5}, {3, 5}}));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           ca("a,b,c").block(boost::combine(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           ca("a,b,c").block(iv(3, 3, 3), iv(5, 5, 5)));

  // make sure that c("abc") = a("abc") does a deep copy
  {
    BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,   b, c"));
    for (auto&& idx : c.tiles_range()) {
      if (c.is_local(idx) && !c.is_local(idx) && a.is_local(idx) &&
          !a.is_zero(idx)) {
        BOOST_CHECK(c.find_local(idx).get().data() !=
                    a.find_local(idx).get().data());
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(block_tensor_factories, F, Fixtures, F) {
  auto& a = F::a;
  auto& c = F::c;

  const auto& ca = a;
  const std::array<int, 3> lobound{{3, 3, 3}};
  const std::array<int, 3> upbound{{5, 5, 5}};

  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           a("a,b,c").block({3, 3, 3}, {5, 5, 5}).conj());
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") += a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           c("b,a,c") + a("b,a,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") -= a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           c("b,a,c") - a("b,a,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") *= a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           c("b,a,c") * a("b,a,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block(lobound, upbound).conj());
  BOOST_CHECK_NO_THROW(c("a,b,c") = ca("a,b,c").block(lobound, upbound).conj());

  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block(lobound, upbound) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           2 * (2 * a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           (2 * a("a,b,c").block(lobound, upbound)) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = -a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(2 * a("a,b,c").block(lobound, upbound)));

  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           conj(conj(a("a,b,c").block(lobound, upbound))));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           conj(2 * a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           conj(conj(2 * a("a,b,c").block(lobound, upbound))));

  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           2 * conj(a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           conj(a("a,b,c").block(lobound, upbound)) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           2 * conj(2 * a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           conj(2 * a("a,b,c").block(lobound, upbound)) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           -conj(2 * a("a,b,c").block(lobound, upbound)));
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scaled_tensor_factories, F, Fixtures, F) {
  auto& a = F::a;
  auto& c = F::c;
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a") * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (2 * a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(2 * a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(a("c,b,a"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(2 * a("c,b,a"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * a("c,b,a")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(2 * a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(2 * a("c,b,a")));
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(add_factories, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a") + b("a,b,c"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (a("c,b,a") + b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (a("c,b,a") + b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (2 * (a("c,b,a") + b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (2 * (a("c,b,a") + b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(a("c,b,a") + b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(2 * (a("c,b,a") + b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") + b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(a("c,b,a") + b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") + b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(2 * (a("c,b,a") + b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (conj(a("c,b,a") + b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") + b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(a("c,b,a") + b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") + b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(2 * (a("c,b,a") + b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(a("c,b,a") + b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(2 * (a("c,b,a") + b("a,b,c"))) * 2);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(subt_factories, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a") - b("a,b,c"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (a("c,b,a") - b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (a("c,b,a") - b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (2 * (a("c,b,a") - b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (2 * (a("c,b,a") - b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(a("c,b,a") - b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(2 * (a("c,b,a") - b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") - b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(a("c,b,a") - b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") - b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(2 * (a("c,b,a") - b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (conj(a("c,b,a") - b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") - b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(a("c,b,a") - b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") - b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(2 * (a("c,b,a") - b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(a("c,b,a") - b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(2 * (a("c,b,a") - b("a,b,c"))) * 2);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(mult_factories, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a") * b("a,b,c"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (a("c,b,a") * b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (a("c,b,a") * b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (2 * (a("c,b,a") * b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (2 * (a("c,b,a") * b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(a("c,b,a") * b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(2 * (a("c,b,a") * b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") * b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(a("c,b,a") * b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") * b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(2 * (a("c,b,a") * b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (conj(a("c,b,a") * b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") * b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(a("c,b,a") * b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") * b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(2 * (a("c,b,a") * b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(a("c,b,a") * b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(2 * (a("c,b,a") * b("a,b,c"))) * 2);
}

// TODO need to check if result is correct
BOOST_FIXTURE_TEST_CASE_TEMPLATE(reduce_factories, F, Fixtures, F) {
  auto& a = F::a;

  BOOST_CHECK_NO_THROW(a("a,b,c").sum().get());
  BOOST_CHECK_NO_THROW(a("a,b,c").product().get());
  BOOST_CHECK_NO_THROW(a("a,b,c").squared_norm().get());
  BOOST_CHECK_NO_THROW(a("a,b,c").norm().get());
  //  BOOST_CHECK_NO_THROW(a("a,b,c").min().get());
  //  BOOST_CHECK_NO_THROW(a("a,b,c").max().get());
  BOOST_CHECK_NO_THROW(a("a,b,c").abs_min().get());
  BOOST_CHECK_NO_THROW(a("a,b,c").abs_max().get());
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;

  Permutation perm({2, 1, 0});
  BOOST_REQUIRE_NO_THROW(a("a,b,c") = b("c,b,a"));

  for (std::size_t i = 0ul; i < b.size(); ++i) {
    const std::size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    if (a.is_local(perm_index) && !a.is_zero(perm_index)) {
      auto a_tile = a.find(perm_index).get();
      auto perm_b_tile = perm * b.find(i).get();

      BOOST_CHECK_EQUAL(a_tile.range(), perm_b_tile.range());
      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        BOOST_CHECK_EQUAL(a_tile[j], perm_b_tile[j]);
    } else if (a.is_local(perm_index) && a.is_zero(perm_index)) {
      BOOST_CHECK(b.is_zero(i));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scale_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  Permutation perm({2, 1, 0});
  BOOST_REQUIRE_NO_THROW(a("α,β,γ") = 2 * b("γ,β,α"));

  for (std::size_t i = 0ul; i < b.size(); ++i) {
    const std::size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    if (a.is_local(perm_index) && !a.is_zero(perm_index)) {
      auto a_tile = a.find(perm_index).get();
      auto perm_b_tile = perm * b.find(i).get();

      BOOST_CHECK_EQUAL(a_tile.range(), perm_b_tile.range());
      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        BOOST_CHECK_EQUAL(a_tile[j], 2 * perm_b_tile[j]);
    } else if (a.is_local(perm_index) && a.is_zero(perm_index)) {
      BOOST_CHECK(b.is_zero(i));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(block, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index))) {
      auto arg_tile = a.find(block_range.ordinal(index)).get();
      auto result_tile = c.find(index).get();

      for (unsigned int r = 0u; r < arg_tile.range().rank(); ++r) {
        BOOST_CHECK_EQUAL(
            result_tile.range().lobound(r),
            arg_tile.range().lobound(r) - a.trange().data()[r].tile(3).first);

        BOOST_CHECK_EQUAL(
            result_tile.range().upbound(r),
            arg_tile.range().upbound(r) - a.trange().data()[r].tile(3).first);

        BOOST_CHECK_EQUAL(result_tile.range().extent(r),
                          arg_tile.range().extent(r));

        BOOST_CHECK_EQUAL(result_tile.range().stride(r),
                          arg_tile.range().stride(r));
      }
      BOOST_CHECK_EQUAL(result_tile.range().volume(),
                        arg_tile.range().volume());

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], arg_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c").block({3, 3, 3}, {5, 5, 5}) +
                                      b("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index)) ||
        !b.is_zero(block_range.ordinal(index))) {
      auto result_tile = c.find(index).get();

      auto a_tile = a.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : a.find(block_range.ordinal(index)).get();
      auto b_tile = b.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : b.find(block_range.ordinal(index)).get();

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], a_tile[j] + b_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(const_block, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;
  const auto& ca = a;
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = ca("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index))) {
      auto arg_tile = a.find(block_range.ordinal(index)).get();
      auto result_tile = c.find(index).get();

      for (unsigned int r = 0u; r < arg_tile.range().rank(); ++r) {
        BOOST_CHECK_EQUAL(
            result_tile.range().lobound(r),
            arg_tile.range().lobound(r) - a.trange().data()[r].tile(3).first);

        BOOST_CHECK_EQUAL(
            result_tile.range().upbound(r),
            arg_tile.range().upbound(r) - a.trange().data()[r].tile(3).first);

        BOOST_CHECK_EQUAL(result_tile.range().extent(r),
                          arg_tile.range().extent(r));

        BOOST_CHECK_EQUAL(result_tile.range().stride(r),
                          arg_tile.range().stride(r));
      }
      BOOST_CHECK_EQUAL(result_tile.range().volume(),
                        arg_tile.range().volume());

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], arg_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c").block({3, 3, 3}, {5, 5, 5}) +
                                      b("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index)) ||
        !b.is_zero(block_range.ordinal(index))) {
      auto result_tile = c.find(index).get();

      auto a_tile = a.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : a.find(block_range.ordinal(index)).get();
      auto b_tile = b.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : b.find(block_range.ordinal(index)).get();

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], a_tile[j] + b_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scal_block, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;
  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             2 * a("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index))) {
      auto arg_tile = a.find(block_range.ordinal(index)).get();
      auto result_tile = c.find(index).get();

      for (unsigned int r = 0u; r < arg_tile.range().rank(); ++r) {
        BOOST_CHECK_EQUAL(
            result_tile.range().lobound(r),
            arg_tile.range().lobound(r) - a.trange().data()[r].tile(3).first);

        BOOST_CHECK_EQUAL(
            result_tile.range().upbound(r),
            arg_tile.range().upbound(r) - a.trange().data()[r].tile(3).first);

        BOOST_CHECK_EQUAL(result_tile.range().extent(r),
                          arg_tile.range().extent(r));

        BOOST_CHECK_EQUAL(result_tile.range().stride(r),
                          arg_tile.range().stride(r));
      }
      BOOST_CHECK_EQUAL(result_tile.range().volume(),
                        arg_tile.range().volume());

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * arg_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             2 * (a("a,b,c").block({3, 3, 3}, {5, 5, 5}) +
                                  b("a,b,c").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index)) ||
        !b.is_zero(block_range.ordinal(index))) {
      auto result_tile = c.find(index).get();
      auto a_tile = a.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : a.find(block_range.ordinal(index)).get();
      auto b_tile = b.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : b.find(block_range.ordinal(index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (a_tile[j] + b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             2 * (3 * a("a,b,c").block({3, 3, 3}, {5, 5, 5}) +
                                  4 * b("a,b,c").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index)) ||
        !b.is_zero(block_range.ordinal(index))) {
      auto result_tile = c.find(index).get();
      auto a_tile = a.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : a.find(block_range.ordinal(index)).get();
      auto b_tile = b.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : b.find(block_range.ordinal(index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (3 * a_tile[j] + 4 * b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(permute_block, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;
  Permutation perm({2, 1, 0});
  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("c,b,a").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * c.tiles_range().idx(index));

    if (!a.is_zero(block_range.ordinal(perm_index))) {
      auto arg_tile = perm * a.find(block_range.ordinal(perm_index)).get();
      auto result_tile = c.find(index).get();

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], arg_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             2 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * c.tiles_range().idx(index));

    if (!a.is_zero(block_range.ordinal(perm_index))) {
      auto arg_tile = perm * a.find(block_range.ordinal(perm_index)).get();
      auto result_tile = c.find(index).get();

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * arg_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             2 * (3 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}) +
                                  4 * b("a,b,c").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * c.tiles_range().idx(index));

    if (!a.is_zero(block_range.ordinal(perm_index)) ||
        !b.is_zero(block_range.ordinal(index))) {
      auto result_tile = c.find(index).get();
      auto a_tile = a.is_zero(block_range.ordinal(perm_index))
                        ? F::make_zero_tile(result_tile.range())
                        : perm * a.find(block_range.ordinal(perm_index)).get();
      auto b_tile = b.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : b.find(block_range.ordinal(index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (3 * a_tile[j] + 4 * b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             2 * (3 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}) +
                                  4 * b("c,b,a").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * c.tiles_range().idx(index));

    if (!a.is_zero(block_range.ordinal(perm_index)) ||
        !b.is_zero(block_range.ordinal(perm_index))) {
      auto result_tile = c.find(index).get();
      auto a_tile = a.is_zero(block_range.ordinal(perm_index))
                        ? F::make_zero_tile(result_tile.range())
                        : perm * a.find(block_range.ordinal(perm_index)).get();
      auto b_tile = b.is_zero(block_range.ordinal(perm_index))
                        ? F::make_zero_tile(result_tile.range())
                        : perm * b.find(block_range.ordinal(perm_index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (3 * a_tile[j] + 4 * b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(assign_subblock_block, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  c.fill_local(0.0);

  for (int repeat = 0; repeat != nrepeats; ++repeat)
    BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                               2 * a("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!c.is_zero(block_range.ordinal(index))) {
      auto arg_tile = a.find(block_range.ordinal(index)).get();
      auto result_tile = c.find(block_range.ordinal(index)).get();

      BOOST_CHECK_EQUAL(result_tile.range(), arg_tile.range());

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * arg_tile[j]);
      }
    } else {
      BOOST_CHECK(a.is_zero(block_range.ordinal(index)));
    }
  }

  for (int repeat = 0; repeat != nrepeats; ++repeat)
    BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                               2 * (a("a,b,c").block({3, 3, 3}, {5, 5, 5}) +
                                    b("a,b,c").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index)) ||
        !b.is_zero(block_range.ordinal(index))) {
      auto result_tile = c.find(block_range.ordinal(index)).get();
      auto a_tile = a.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : a.find(block_range.ordinal(index)).get();
      auto b_tile = b.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : b.find(block_range.ordinal(index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (a_tile[j] + b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(block_range.ordinal(index)));
    }
  }

  for (int repeat = 0; repeat != nrepeats; ++repeat)
    BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                               2 *
                               (3 * a("a,b,c").block({3, 3, 3}, {5, 5, 5}) +
                                4 * b("a,b,c").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index)) ||
        !b.is_zero(block_range.ordinal(index))) {
      auto result_tile = c.find(block_range.ordinal(index)).get();
      auto a_tile = a.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : a.find(block_range.ordinal(index)).get();
      auto b_tile = b.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : b.find(block_range.ordinal(index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (3 * a_tile[j] + 4 * b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(block_range.ordinal(index)));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(assign_subblock_permute_block, F, Fixtures,
                                 F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  c.fill_local(0.0);

  Permutation perm({2, 1, 0});
  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  for (int repeat = 0; repeat != nrepeats; ++repeat)
    BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                               a("c,b,a").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    //    const size_t perm_index = block_range.ordinal(perm *
    //    c.tiles_range().idx(index));
    auto perm_index = perm * block_range.idx(index);

    if (!a.is_zero(block_range.ordinal(perm_index))) {
      auto arg_tile = perm * a.find(block_range.ordinal(perm_index)).get();
      auto result_tile = c.find(block_range.ordinal(index)).get();

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], arg_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(block_range.ordinal(index)));
    }
  }

  for (int repeat = 0; repeat != nrepeats; ++repeat)
    BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                               2 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    auto perm_index = perm * block_range.idx(index);

    if (!a.is_zero(block_range.ordinal(perm_index))) {
      auto arg_tile = perm * a.find(block_range.ordinal(perm_index)).get();
      auto result_tile = c.find(block_range.ordinal(index)).get();

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * arg_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(block_range.ordinal(index)));
    }
  }

  for (int repeat = 0; repeat != nrepeats; ++repeat)
    BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                               2 *
                               (3 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}) +
                                4 * b("a,b,c").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    auto perm_index = perm * block_range.idx(index);

    if (!a.is_zero(block_range.ordinal(perm_index)) ||
        !b.is_zero(block_range.ordinal(index))) {
      auto result_tile = c.find(block_range.ordinal(index)).get();
      auto a_tile = a.is_zero(block_range.ordinal(perm_index))
                        ? F::make_zero_tile(result_tile.range())
                        : perm * a.find(block_range.ordinal(perm_index)).get();
      auto b_tile = b.is_zero(block_range.ordinal(index))
                        ? F::make_zero_tile(result_tile.range())
                        : b.find(block_range.ordinal(index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (3 * a_tile[j] + 4 * b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(block_range.ordinal(index)));
    }
  }

  for (int repeat = 0; repeat != nrepeats; ++repeat)
    BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                               2 *
                               (3 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}) +
                                4 * b("c,b,a").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    auto perm_index = perm * block_range.idx(index);

    if (!a.is_zero(block_range.ordinal(perm_index)) ||
        !b.is_zero(block_range.ordinal(perm_index))) {
      auto result_tile = c.find(block_range.ordinal(index)).get();
      auto a_tile = a.is_zero(block_range.ordinal(perm_index))
                        ? F::make_zero_tile(result_tile.range())
                        : perm * a.find(block_range.ordinal(perm_index)).get();
      auto b_tile = b.is_zero(block_range.ordinal(perm_index))
                        ? F::make_zero_tile(result_tile.range())
                        : perm * b.find(block_range.ordinal(perm_index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (3 * a_tile[j] + 4 * b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(block_range.ordinal(index)));
    }
  }
}

// TODO need to test the correctness here
BOOST_FIXTURE_TEST_CASE_TEMPLATE(assign_subblock_block_contract, F, Fixtures,
                                 F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& w = F::w;

  for (int repeat = 0; repeat != nrepeats; ++repeat)
    BOOST_REQUIRE_NO_THROW(w("a,b").block({3, 3}, {5, 5}) =
                               a("a,c,d").block({3, 2, 3}, {5, 5, 5}) *
                               b("c,d,b").block({2, 3, 3}, {5, 5, 5}));
}
// TODO need to test the correctness here
BOOST_FIXTURE_TEST_CASE_TEMPLATE(assign_subblock_block_permute_contract, F,
                                 Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& w = F::w;

  for (int repeat = 0; repeat != nrepeats; ++repeat)
    BOOST_REQUIRE_NO_THROW(w("a,b").block({3, 3}, {5, 5}) =
                               a("a,c,d").block({3, 2, 3}, {5, 5, 5}) *
                               b("d,c,b").block({3, 2, 3}, {5, 5, 5}));
}
// TODO need to test the correctness here
BOOST_FIXTURE_TEST_CASE_TEMPLATE(block_contract, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& w = F::w;

  for (int repeat = 0; repeat != nrepeats; ++repeat)
    BOOST_REQUIRE_NO_THROW(w("a,b") = a("a,c,d").block({3, 2, 3}, {5, 5, 5}) *
                                      b("c,d,b").block({2, 3, 3}, {5, 5, 5}));
}
// TODO need to test the correctness here
BOOST_FIXTURE_TEST_CASE_TEMPLATE(block_permute_contract, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& w = F::w;

  for (int repeat = 0; repeat != nrepeats; ++repeat)
    BOOST_REQUIRE_NO_THROW(w("a,b") = a("a,c,d").block({3, 2, 3}, {5, 5, 5}) *
                                      b("d,c,b").block({3, 2, 3}, {5, 5, 5}));
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(add, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") + b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] + b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) + b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) + b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") + (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] + (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) + (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) + (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c") + 3 * b("a,b,c")) +
                                      2 * (a("a,b,c") - b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (4 * a_tile[j]) + (b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(add_to, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(c("a,b,c") += b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] + b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = c("a,b,c") + b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] + b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(add_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) + (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) + (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) + (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile = b.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) + (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) && b.is_zero(perm_index));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scale_add, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") + b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] + b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) + b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) + b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") + (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] + (3 * b_tile[j])));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             5 * ((2 * a("a,b,c")) + (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) + (3 * b_tile[j])));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c") + 3 * b("a,b,c")) +
                                           2 * (a("a,b,c") - b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (4 * a_tile[j] + b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scale_add_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) + (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) + (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) + (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile = b.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) + (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) && b.is_zero(perm_index));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(subt, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") - b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] - b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) - b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) - b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") - (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] - (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) - (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) - (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(subt_to, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(c("a,b,c") -= b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] - b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = c("a,b,c") - b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] - b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(sub_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) - (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) - (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) - (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile = b.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) - (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) && b.is_zero(perm_index));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scale_subt, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") - b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] - b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) - b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) - b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") - (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] - (3 * b_tile[j])));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             5 * ((2 * a("a,b,c")) - (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) - (3 * b_tile[j])));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scale_sub_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) - (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) - (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) && b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) - (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile = b.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) - (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) && b.is_zero(perm_index));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(mult, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") * b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] * b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) || b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) * b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) * b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) || b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") * (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] * (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) || b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) * (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) * (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) || b.is_zero(i));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(mult_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) * (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) * (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) || b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) * (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile = b.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) * (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) || b.is_zero(perm_index));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(mult_to, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(c("a,b,c") *= b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] * b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) || b.is_zero(i));
    }
  }

  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = c("a,b,c") * b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] * b_tile[j]);
    } else {
      BOOST_CHECK(a.is_zero(i) || b.is_zero(i));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scale_mult, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) || b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) || b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") * (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] * (3 * b_tile[j])));
    } else {
      BOOST_CHECK(a.is_zero(i) || b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             5 * ((2 * a("a,b,c")) * (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile =
          a.is_zero(i) ? F::make_zero_tile(c_tile.range()) : a.find(i).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) * (3 * b_tile[j])));
    } else {
      BOOST_CHECK(a.is_zero(i) || b.is_zero(i));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scale_mult_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& c = F::c;

  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) * (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile =
          b.is_zero(i) ? F::make_zero_tile(c_tile.range()) : b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) * (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) || b.is_zero(i));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) * (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * a.find(perm_index).get();
      auto b_tile = b.is_zero(perm_index) ? F::make_zero_tile(c_tile.range())
                                          : perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) * (3 * b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(perm_index) || b.is_zero(perm_index));
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(cont, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& w = F::w;

  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  typename F::Matrix left(m, k);
  left.fill(0);

  for (auto it = a.begin(); it != a.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  typename F::Matrix right(n, k);
  right.fill(0);

  for (auto it = b.begin(); it != b.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          right(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  typename F::Matrix result(m, n);

  result = left * right.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("i,b,c") * b("j,b,c"));
  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]));
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("i,b,c")) * b("j,b,c"));
  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 2);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("i,b,c") * (3 * b("j,b,c")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 3);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("i,b,c")) * (3 * b("j,b,c")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 6);
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(cont_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& w = F::w;

  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  typename F::Matrix left(m, k);
  left.fill(0);

  for (auto it = a.begin(); it != a.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  typename F::Matrix right(n, k);
  right.fill(0);

  for (auto it = b.begin(); it != b.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[2] * a.trange().elements_range().stride(1) +
                                i[1] * a.trange().elements_range().stride(2);

          right(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  typename F::Matrix result(m, n);

  result = left * right.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("i,b,c") * b("j,c,b"));
  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]));
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("i,b,c")) * b("j,c,b"));
  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 2);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("i,b,c") * (3 * b("j,c,b")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 3);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("i,b,c")) * (3 * b("j,c,b")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 6);
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(cont_permute_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& w = F::w;

  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  typename F::Matrix left(m, k);
  left.fill(0);

  for (auto it = a.begin(); it != a.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  typename F::Matrix right(n, k);
  right.fill(0);

  for (auto it = b.begin(); it != b.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[2] * a.trange().elements_range().stride(1) +
                                i[1] * a.trange().elements_range().stride(2);

          right(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  typename F::Matrix result(m, n);

  result = right * left.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("j,b,c") * b("i,c,b"));
  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]));
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("j,b,c")) * b("i,c,b"));
  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 2);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("j,b,c") * (3 * b("i,c,b")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 3);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("j,b,c")) * (3 * b("i,c,b")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 6);
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scale_cont, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& w = F::w;

  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  typename F::Matrix left(m, k);
  left.fill(0);

  for (auto it = a.begin(); it != a.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  typename F::Matrix right(n, k);
  right.fill(0);

  for (auto it = b.begin(); it != b.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          right(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  typename F::Matrix result(m, n);

  result = left * right.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("i,b,c") * b("j,b,c")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 5);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("i,b,c")) * b("j,b,c")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 10);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("i,b,c") * (3 * b("j,b,c"))));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 15);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("i,b,c")) * (3 * b("j,b,c"))));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 30);
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scale_cont_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& w = F::w;

  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  typename F::Matrix left(m, k);
  left.fill(0);

  for (auto it = a.begin(); it != a.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  typename F::Matrix right(n, k);
  right.fill(0);

  for (auto it = b.begin(); it != b.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[2] * a.trange().elements_range().stride(1) +
                                i[1] * a.trange().elements_range().stride(2);

          right(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  typename F::Matrix result(m, n);

  result = left * right.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("i,b,c") * b("j,c,b")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 5);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("i,b,c")) * b("j,c,b")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 10);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("i,b,c") * (3 * b("j,c,b"))));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 15);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("i,b,c")) * (3 * b("j,c,b"))));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 30);
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(scale_cont_permute_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  auto& w = F::w;

  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  typename F::Matrix left(m, k);
  left.fill(0);

  for (auto it = a.begin(); it != a.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  typename F::Matrix right(n, k);
  right.fill(0);

  for (auto it = b.begin(); it != b.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[2] * a.trange().elements_range().stride(1) +
                                i[1] * a.trange().elements_range().stride(2);

          right(r, c) = tile[i];
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  typename F::Matrix result(m, n);

  result = right * left.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("j,b,c") * b("i,c,b")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 5);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("j,b,c")) * b("i,c,b")));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 10);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("j,b,c") * (3 * b("i,c,b"))));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 15);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("j,b,c")) * (3 * b("i,c,b"))));

  for (auto it = w.begin(); it != w.end(); ++it) {
    typename F::TArray::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_EQUAL(tile[i], result(i[0], i[1]) * 30);
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(cont_non_uniform1, F, Fixtures, F) {
  // Construct the tiled range
  std::array<std::size_t, 6> tiling1 = {{0, 1, 2, 3, 4, 5}};
  std::array<std::size_t, 2> tiling2 = {{0, 40}};
  TiledRange1 tr1_1(tiling1.begin(), tiling1.end());
  TiledRange1 tr1_2(tiling2.begin(), tiling2.end());
  std::array<TiledRange1, 4> tiling4 = {{tr1_1, tr1_2, tr1_1, tr1_1}};
  TiledRange trange(tiling4.begin(), tiling4.end());

  const std::size_t m = 5;
  const std::size_t k = 40 * 5 * 5;
  const std::size_t n = 5;

  // Construct the test arguments
  auto left = F::make_array(trange);
  auto right = F::make_array(trange);

  // Construct the reference matrices
  typename F::Matrix left_ref(m, k);
  typename F::Matrix right_ref(n, k);

  // Initialize input
  F::rand_fill_matrix_and_array(left_ref, left, 23);
  F::rand_fill_matrix_and_array(right_ref, right, 42);

  // Compute the reference result
  typename F::Matrix result_ref = 5 * left_ref * right_ref.transpose();

  // Compute the result to be tested
  typename F::TArray result;
  BOOST_REQUIRE_NO_THROW(result("x,y") =
                             5 * left("x,i,j,k") * right("y,i,j,k"));

  // Check the result
  for (auto it = result.begin(); it != result.end(); ++it) {
    typename F::TArray::value_type tile = *it;
    for (Range::const_iterator rit = tile.range().begin();
         rit != tile.range().end(); ++rit) {
      const std::size_t elem_index = result.elements_range().ordinal(*rit);
      BOOST_CHECK_EQUAL(result_ref.array()(elem_index), tile[*rit]);
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(cont_non_uniform2, F, Fixtures, F) {
  // Construct the tiled range
  std::array<std::size_t, 6> tiling1 = {{0, 1, 2, 3, 4, 5}};
  std::array<std::size_t, 2> tiling2 = {{0, 40}};
  TiledRange1 tr1_1(tiling1.begin(), tiling1.end());
  TiledRange1 tr1_2(tiling2.begin(), tiling2.end());
  std::array<TiledRange1, 4> tiling4 = {{tr1_1, tr1_1, tr1_2, tr1_2}};
  TiledRange trange(tiling4.begin(), tiling4.end());

  const std::size_t m = 5;
  const std::size_t k = 5 * 40 * 40;
  const std::size_t n = 5;

  // Construct the test arguments
  auto left = F::make_array(trange);
  auto right = F::make_array(trange);

  // Construct the reference matrices
  typename F::Matrix left_ref(m, k);
  typename F::Matrix right_ref(n, k);

  // Initialize input
  F::rand_fill_matrix_and_array(left_ref, left, 23);
  F::rand_fill_matrix_and_array(right_ref, right, 42);

  // Compute the reference result
  typename F::Matrix result_ref = 5 * left_ref * right_ref.transpose();

  // Compute the result to be tested
  typename F::TArray result;
  BOOST_REQUIRE_NO_THROW(result("x,y") =
                             5 * left("x,i,j,k") * right("y,i,j,k"));

  // Check the result
  for (auto it = result.begin(); it != result.end(); ++it) {
    typename F::TArray::value_type tile = *it;
    for (Range::const_iterator rit = tile.range().begin();
         rit != tile.range().end(); ++rit) {
      const std::size_t elem_index = result.elements_range().ordinal(*rit);
      BOOST_CHECK_EQUAL(result_ref.array()(elem_index), tile[*rit]);
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(cont_plus_reduce, F, Fixtures, F) {
  // Construct the tiled range
  std::array<std::size_t, 6> tiling1 = {{0, 1, 2, 3, 4, 5}};
  std::array<std::size_t, 2> tiling2 = {{0, 40}};
  TiledRange1 tr1_1(tiling1.begin(), tiling1.end());
  TiledRange1 tr1_2(tiling2.begin(), tiling2.end());
  std::array<TiledRange1, 4> tiling4 = {{tr1_1, tr1_2, tr1_1, tr1_1}};
  TiledRange trange(tiling4.begin(), tiling4.end());

  const std::size_t m = 5;
  const std::size_t k = 40 * 5 * 5;
  const std::size_t n = 5;

  // Construct the test arrays
  auto arg1 = F::make_array(trange);
  auto arg2 = F::make_array(trange);
  auto arg3 = F::make_array(trange);
  auto arg4 = F::make_array(trange);

  // Construct the reference matrices
  typename F::Matrix arg1_ref(m, k);
  typename F::Matrix arg2_ref(n, k);
  typename F::Matrix arg3_ref(m, k);
  typename F::Matrix arg4_ref(n, k);

  // Initialize input
  F::rand_fill_matrix_and_array(arg1_ref, arg1, 23);
  F::rand_fill_matrix_and_array(arg2_ref, arg2, 42);
  F::rand_fill_matrix_and_array(arg3_ref, arg3, 79);
  F::rand_fill_matrix_and_array(arg4_ref, arg4, 19);

  // Compute the reference result
  typename F::Matrix result_ref =
      2 * (arg1_ref * arg2_ref.transpose() + arg1_ref * arg4_ref.transpose() +
           arg3_ref * arg4_ref.transpose() + arg3_ref * arg2_ref.transpose());

  // Compute the result to be tested
  typename F::TArray result;
  result("x,y") = arg1("x,i,j,k") * arg2("y,i,j,k");
  result("x,y") += arg3("x,i,j,k") * arg4("y,i,j,k");
  result("x,y") += arg1("x,i,j,k") * arg4("y,i,j,k");
  result("x,y") += arg3("x,i,j,k") * arg2("y,i,j,k");
  result("x,y") += arg3("x,i,j,k") * arg2("y,i,j,k");
  result("x,y") += arg1("x,i,j,k") * arg2("y,i,j,k");
  result("x,y") += arg3("x,i,j,k") * arg4("y,i,j,k");
  result("x,y") += arg1("x,i,j,k") * arg4("y,i,j,k");

  // Check the result
  for (auto it = result.begin(); it != result.end(); ++it) {
    typename F::TArray::value_type tile = *it;
    for (Range::const_iterator rit = tile.range().begin();
         rit != tile.range().end(); ++rit) {
      const std::size_t elem_index = result.elements_range().ordinal(*rit);
      BOOST_CHECK_EQUAL(result_ref.array()(elem_index), tile[*rit]);
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(no_alias_plus_reduce, F, Fixtures, F) {
  // Construct the tiled range
  std::array<std::size_t, 6> tiling1 = {{0, 1, 2, 3, 4, 5}};
  std::array<std::size_t, 2> tiling2 = {{0, 40}};
  TiledRange1 tr1_1(tiling1.begin(), tiling1.end());
  TiledRange1 tr1_2(tiling2.begin(), tiling2.end());
  std::array<TiledRange1, 4> tiling4 = {{tr1_1, tr1_2, tr1_1, tr1_1}};
  TiledRange trange(tiling4.begin(), tiling4.end());

  const std::size_t m = 5;
  const std::size_t k = 40 * 5 * 5;
  const std::size_t n = 5;

  // Construct the test arrays
  auto arg1 = F::make_array(trange);
  auto arg2 = F::make_array(trange);
  auto arg3 = F::make_array(trange);
  auto arg4 = F::make_array(trange);

  // Construct the reference matrices
  typename F::Matrix arg1_ref(m, k);
  typename F::Matrix arg2_ref(n, k);
  typename F::Matrix arg3_ref(m, k);
  typename F::Matrix arg4_ref(n, k);

  // Initialize input
  F::rand_fill_matrix_and_array(arg1_ref, arg1, 23);
  F::rand_fill_matrix_and_array(arg2_ref, arg2, 42);
  F::rand_fill_matrix_and_array(arg3_ref, arg3, 79);
  F::rand_fill_matrix_and_array(arg4_ref, arg4, 19);

  // Compute the reference result
  typename F::Matrix result_ref =
      2 * (arg1_ref * arg2_ref.transpose() + arg1_ref * arg4_ref.transpose() +
           arg3_ref * arg4_ref.transpose() + arg3_ref * arg2_ref.transpose());

  // Compute the result to be tested
  typename F::TArray result;
  result("x,y") = arg1("x,i,j,k") * arg2("y,i,j,k");
  result("x,y").no_alias() += arg3("x,i,j,k") * arg4("y,i,j,k");
  result("x,y").no_alias() += arg1("x,i,j,k") * arg4("y,i,j,k");
  result("x,y").no_alias() += arg3("x,i,j,k") * arg2("y,i,j,k");
  result("x,y").no_alias() += arg3("x,i,j,k") * arg2("y,i,j,k");
  result("x,y").no_alias() += arg1("x,i,j,k") * arg2("y,i,j,k");
  result("x,y").no_alias() += arg3("x,i,j,k") * arg4("y,i,j,k");
  result("x,y").no_alias() += arg1("x,i,j,k") * arg4("y,i,j,k");

  // Check the result
  for (auto it = result.begin(); it != result.end(); ++it) {
    typename F::TArray::value_type tile = *it;
    for (Range::const_iterator rit = tile.range().begin();
         rit != tile.range().end(); ++rit) {
      const std::size_t elem_index = result.elements_range().ordinal(*rit);
      BOOST_CHECK_EQUAL(result_ref.array()(elem_index), tile[*rit]);
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(outer_product, F, Fixtures, F) {
  auto& u = F::u;
  auto& v = F::v;
  auto& w = F::w;
  // Generate Eigen matrices from input arrays.
  auto ev = F::make_matrix(v);
  auto eu = F::make_matrix(u);

  // Generate the expected result
  auto ew_test = eu * ev.transpose();

  // Test that outer product works
  BOOST_REQUIRE_NO_THROW(w("i,j") = u("i") * v("j"));

  GlobalFixture::world->gop.fence();

  auto ew = F::make_matrix(w);

  BOOST_CHECK_EQUAL(ew, ew_test);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(dot, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;

  // Test the dot expression function
  typename F::element_type result = 0;
  BOOST_REQUIRE_NO_THROW(
      result = static_cast<typename F::element_type>(a("a,b,c") * b("a,b,c")));
  BOOST_REQUIRE_NO_THROW(result += a("a,b,c") * b("a,b,c"));
  BOOST_REQUIRE_NO_THROW(result -= a("a,b,c") * b("a,b,c"));
  BOOST_REQUIRE_NO_THROW(result *= a("a,b,c") * b("a,b,c"));
  BOOST_REQUIRE_NO_THROW(result = a("a,b,c").dot(b("a,b,c")).get());

  // Compute the expected value for the dot function.
  typename F::element_type expected = 0;
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    if (!a.is_zero(i) && !b.is_zero(i)) {
      auto a_tile = a.find(i).get();
      auto b_tile = b.find(i).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += a_tile[j] * b_tile[j];
    }
  }

  // Check the result of dot
  BOOST_CHECK_EQUAL(result, expected);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(
      result = (a("a,b,c") - b("a,b,c")).dot((a("a,b,c") + b("a,b,c"))).get());

  for (std::size_t i = 0ul; i < a.size(); ++i) {
    if (!a.is_zero(i) || !b.is_zero(i)) {
      auto a_tile = a.is_zero(i) ? F::make_zero_tile(a.trange().tile(i))
                                 : a.find(i).get();
      auto b_tile = b.is_zero(i) ? F::make_zero_tile(b.trange().tile(i))
                                 : b.find(i).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += (a_tile[j] - b_tile[j]) * (a_tile[j] + b_tile[j]);
    }
  }

  BOOST_CHECK_EQUAL(result, expected);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(result = (2 * a("a,b,c")).dot(3 * b("a,b,c")).get());
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    if (!a.is_zero(i) && !b.is_zero(i)) {
      auto a_tile = a.find(i).get();
      auto b_tile = b.find(i).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += 6 * (a_tile[j] * b_tile[j]);
    }
  }

  BOOST_CHECK_EQUAL(result, expected);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(
      result =
          2 *
          (a("a,b,c") - b("a,b,c")).dot(3 * (a("a,b,c") + b("a,b,c"))).get());
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    if (!a.is_zero(i) || !b.is_zero(i)) {
      auto a_tile = a.is_zero(i) ? F::make_zero_tile(a.trange().tile(i))
                                 : a.find(i).get();
      auto b_tile = b.is_zero(i) ? F::make_zero_tile(b.trange().tile(i))
                                 : b.find(i).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += 6 * (a_tile[j] - b_tile[j]) * (a_tile[j] + b_tile[j]);
    }
  }

  BOOST_CHECK_EQUAL(result, expected);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(dot_permute, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  Permutation perm({2, 1, 0});
  // Test the dot expression function
  typename F::element_type result = 0;
  BOOST_REQUIRE_NO_THROW(
      result = static_cast<typename F::element_type>(a("a,b,c") * b("c,b,a")));
  BOOST_REQUIRE_NO_THROW(result += a("a,b,c") * b("c,b,a"));
  BOOST_REQUIRE_NO_THROW(result -= a("a,b,c") * b("c,b,a"));
  BOOST_REQUIRE_NO_THROW(result *= a("a,b,c") * b("c,b,a"));
  BOOST_REQUIRE_NO_THROW(result = a("a,b,c").dot(b("c,b,a")).get());

  // Compute the expected value for the dot function.
  typename F::element_type expected = 0;
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    const size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    if (!a.is_zero(i) && !b.is_zero(perm_index)) {
      auto a_tile = a.find(i).get();
      auto b_tile = perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += a_tile[j] * b_tile[j];
    }
  }

  // Check the result of dot
  BOOST_CHECK_EQUAL(result, expected);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(
      result = (a("a,b,c") - b("c,b,a")).dot(a("a,b,c") + b("c,b,a")).get());

  // Compute the expected value for the dot function.
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    const size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    if (!a.is_zero(i) || !b.is_zero(perm_index)) {
      auto a_tile = a.is_zero(i) ? F::make_zero_tile(a.trange().tile(i))
                                 : a.find(i).get();
      auto b_tile = b.is_zero(perm_index)
                        ? perm * F::make_zero_tile(b.trange().tile(perm_index))
                        : perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += (a_tile[j] - b_tile[j]) * (a_tile[j] + b_tile[j]);
    }
  }

  // Check the result of dot
  BOOST_CHECK_EQUAL(result, expected);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(result = (2 * a("a,b,c")).dot(3 * b("c,b,a")).get());

  // Compute the expected value for the dot function.
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    const size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    if (!a.is_zero(i) && !b.is_zero(perm_index)) {
      auto a_tile = a.find(i).get();
      auto b_tile = perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += 6 * a_tile[j] * b_tile[j];
    }
  }

  // Check the result of dot
  BOOST_CHECK_EQUAL(result, expected);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(result = (2 * (a("a,b,c") - b("c,b,a")))
                                      .dot(3 * (a("a,b,c") + b("c,b,a")))
                                      .get());

  // Compute the expected value for the dot function.
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    const size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    if (!a.is_zero(i) || !b.is_zero(perm_index)) {
      auto a_tile = a.is_zero(i) ? F::make_zero_tile(a.trange().tile(i))
                                 : a.find(i).get();
      auto b_tile = b.is_zero(perm_index)
                        ? perm * F::make_zero_tile(b.trange().tile(perm_index))
                        : perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += 6 * (a_tile[j] - b_tile[j]) * (a_tile[j] + b_tile[j]);
    }
  }

  // Check the result of dot
  BOOST_CHECK_EQUAL(result, expected);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(dot_contr, F, Fixtures, F) {
  auto& a = F::a;
  auto& b = F::b;
  for (int i = 0; i != 50; ++i)
    BOOST_REQUIRE_NO_THROW(
        (a("a,b,c") * b("d,b,c")).dot(b("d,e,f") * a("a,e,f")));
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(inner_product, F, Fixtures, F) {
  // Test the inner_product expression function
  auto x = F::make_array(F::tr);
  auto y = F::make_array(F::tr);
  F::random_fill(x);
  F::random_fill(y);
  typename F::element_type result = 0;
  BOOST_REQUIRE_NO_THROW(result = x("a,b,c").inner_product(y("a,b,c")).get());

  // Compute the expected value for the dot function.
  typename F::element_type expected = 0;
  for (std::size_t i = 0ul; i < x.size(); ++i) {
    if (!x.is_zero(i) && !y.is_zero(i)) {
      auto x_tile = x.find(i).get();
      auto y_tile = y.find(i).get();

      for (std::size_t j = 0ul; j < x_tile.size(); ++j)
        expected += TiledArray::detail::conj(x_tile[j]) * y_tile[j];
    }
  }

  // Check the result of dot
  BOOST_CHECK_EQUAL(result, expected);
}

// corner case: expressions involving array with empty trange1
BOOST_FIXTURE_TEST_CASE_TEMPLATE(empty_trange1, F, Fixtures, F) {
  auto& c = F::c;
  auto& aC = F::aC;

  // unary/binary expressions
  {
    BOOST_CHECK_NO_THROW(c("a,b,c") = aC("a,b,c"));
    BOOST_CHECK_NO_THROW(c("a,b,c") += aC("a,b,c"));
    BOOST_CHECK_NO_THROW(c("a,b,c") *= aC("a,b,c"));
    BOOST_CHECK_NO_THROW(c("a,b,c") *= 2 * aC("a,b,c"));
    BOOST_CHECK_NO_THROW(c("a,b,c") += 2 * aC("a,b,c").conj());
    BOOST_CHECK_NO_THROW(c("a,b,c") = aC("a,c,b"));
    BOOST_CHECK_NO_THROW(c("a,b,c") += 2 * aC("a,c,b").conj());
    BOOST_CHECK_NO_THROW(c("a,b,c") *= 2 * aC("a,c,b").conj());
  }

  using TiledArray::eigen::iv;
  const std::array<int, 3> lobound{{0, 0, 1}};
  const std::array<int, 3> upbound{{1, 0, 2}};

  // unary/binary block expressions
  {
    BOOST_CHECK_NO_THROW(c("a,b,c") = aC("a,b,c").block(lobound, upbound));
    BOOST_CHECK_NO_THROW(c("a,b,c") +=
                         2 * aC("a,b,c").block(lobound, upbound).conj());
    BOOST_CHECK_NO_THROW(c("a,b,c") =
                             2 * conj(aC("a,c,b").block(lobound, upbound)));
  }

  // contraction expressions
  {
    std::decay_t<decltype(c)> t2, t4;
    // contraction over empty dim
    BOOST_CHECK_NO_THROW(t4("a,c,e,d") = aC("a,b,c") * aC("d,b,e"));
    // contraction over empty and nonempty dims
    BOOST_CHECK_NO_THROW(t2("a,d") = aC("a,b,c") * aC("d,b,c"));
    // contraction over nonempty dims
    BOOST_CHECK_NO_THROW(t4("b,a,e,d") = aC("a,b,c") * aC("d,e,c"));
  }

  // reduction expressions
  {
    // contraction over empty dim
    BOOST_CHECK_NO_THROW(aC("a,b,c").dot(2 * aC("a,b,c").conj()).get());
    BOOST_CHECK_EQUAL(aC("a,b,c").dot(2 * aC("a,b,c").conj()).get(), 0);
  }
}

BOOST_AUTO_TEST_SUITE_END()

#endif  // TILEDARRAY_TEST_EXPRESSIONS_IMPL_H
