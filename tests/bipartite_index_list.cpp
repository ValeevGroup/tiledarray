/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#include <numeric>
#include "TiledArray/expressions/index_list.h"
#include "tiledarray.h"
#include "unit_test_config.h"

/* General notes on testing the BipartiteIndexList class.
 *
 * - The old test suite included tests for ensuring valid characters. That
 *   functionality has been moved to the free function `is_valid_index` and is
 *   tested there.
 * - I put the BipartiteIndexLists to test into an std::map, which makes it
 * easier to loop over them and write more compact unit-tests.
 * - I split the constructor and accessor tests into individual unit tests, one
 *   per function. In the event that a bug is found/functionality changes this
 *   should make it easier to find the bug/test the functionality.
 *
 */

using namespace TiledArray;
using TiledArray::expressions::BipartiteIndexList;
using TiledArray::expressions::detail::find_common;

// Pull some typedefs from the class to make sure testing uses the right types
using value_type = typename BipartiteIndexList::value_type;
using const_reference = typename BipartiteIndexList::const_reference;
using size_type = typename BipartiteIndexList::size_type;

struct BipartiteIndexListFixture {
  World& world = get_default_world();
  std::map<value_type, BipartiteIndexList> idxs = {
      {"a,b,c,d", BipartiteIndexList("a,b,c,d")},
      {"a,i,b", BipartiteIndexList("a,i,b")},
      {"x,i,y", BipartiteIndexList("x,i,y")},
      {"a,i", BipartiteIndexList("a,i")},
      {"x,i", BipartiteIndexList("x,i")},
      {"i,a", BipartiteIndexList("i,a")},
      {"i,x", BipartiteIndexList("i,x")},
      {"i", BipartiteIndexList("i")},
      {"i,j", BipartiteIndexList("i,j")},
      {"i,j,k", BipartiteIndexList("i,j,k")},
      {"i;j", BipartiteIndexList("i;j")},
      {"i;j,k", BipartiteIndexList("i;j,k")},
      {"i,j;k", BipartiteIndexList("i,j;k")},
      {"i,j;k,l", BipartiteIndexList("i,j;k,l")}};
};

BOOST_FIXTURE_TEST_SUITE(bipartite_index_list_suite, BipartiteIndexListFixture,
                         TA_UT_LABEL_SERIAL)

/* This unit test ensures that the typedefs are what we think they are. Since no
 * template meta-programming occurs in the class these tests serve more as a
 * consistency check of the API.
 */
BOOST_AUTO_TEST_CASE(typedefs) {
  {
    constexpr bool is_same = std::is_same_v<value_type, std::string>;
    BOOST_CHECK(is_same);
  }

  {
    constexpr bool is_same =
        std::is_same_v<const_reference, const std::string&>;
    BOOST_CHECK(is_same);
  }

  {
    constexpr bool is_same = std::is_same_v<
        typename BipartiteIndexList::const_iterator,
        typename container::svector<std::string>::const_iterator>;
    BOOST_CHECK(is_same);
  }

  {
    constexpr bool is_same = std::is_same_v<size_type, std::size_t>;
    BOOST_CHECK(is_same);
  }
}

/* The default constructor creates and empty container. Here we simply make sure
 * that the accessible state of a default instance is what it is supposed to be.
 */
BOOST_AUTO_TEST_CASE(default_ctor) {
  BOOST_REQUIRE_NO_THROW(BipartiteIndexList v0);
  BipartiteIndexList v0;
  {
    bool are_same = v0.begin() == v0.end();
    BOOST_CHECK(are_same);
  }
  BOOST_CHECK_EQUAL(v0.size(), size_type{0});
  BOOST_CHECK_EQUAL(v0.first_size(), size_type{0});
  BOOST_CHECK_EQUAL(v0.second_size(), size_type{0});
  BOOST_CHECK_EQUAL(v0.size(), size_type{0});
}

/* This unit test tests the constructor which takes a string and tokenizes it.
 * The constructor ultimately calls detail::split_index, which is unit tested
 * elsewhere (and thus assumed to work). We compare the state of the resulting
 * BipartiteIndexList to the correct value (as obtained by
 * detail::split_index) for this unit test. There are a number of ways of
 * accessing the individual indices (operator[], at, iterators); here we only
 * test the at variant. The other access members are tested in their respective
 * unit tests.
 */
BOOST_AUTO_TEST_CASE(string_ctor) {
  if (world.nproc() == 1) {
    BOOST_CHECK_THROW(BipartiteIndexList("i,"), TiledArray::Exception);
  }

  for (auto&& [str, idx] : idxs) {
    auto&& [outer, inner] = TiledArray::detail::split_index(str);
    BOOST_CHECK_EQUAL(idx.size(), outer.size() + inner.size());
    BOOST_CHECK_EQUAL(idx.first_size(), outer.size());
    BOOST_CHECK_EQUAL(idx.second_size(), inner.size());
    BOOST_CHECK_EQUAL(idx.size(), outer.size() + inner.size());

    for (size_type i = 0; i < outer.size(); ++i) {
      BOOST_CHECK_EQUAL(idx.at(i), outer[i]);
    }

    for (size_type i = 0; i < inner.size(); ++i) {
      BOOST_CHECK_EQUAL(idx.at(outer.size() + i), inner[i]);
    }
  }
}

/* To test copy construction/assignment we compare the states of the original
 * and the copy with operator== (which we assume has been tested elsewhere).
 * Next we compare the addresses of the individual indices to ensure they are
 * deep copies. Finally, for copy assignment we ensure that the operation
 * supports chaining.
 */
BOOST_AUTO_TEST_CASE(copy_ctor) {
  // Default instance
  {
    BipartiteIndexList v0;
    BipartiteIndexList v1(v0);
    BOOST_CHECK_EQUAL(v0, v1);
  }

  for (auto&& [str, idx] : idxs) {
    BipartiteIndexList v1(idx);
    BOOST_CHECK_EQUAL(idx, v1);
    // Ensure deep copy
    for (size_type i = 0; i < idx.size(); ++i) BOOST_CHECK(&idx[i] != &v1[i]);
  }
}

BOOST_AUTO_TEST_CASE(copy_assignment) {
  // Default instance
  {
    BipartiteIndexList v0, v1;
    auto pv0 = &(v0 = v1);
    BOOST_CHECK_EQUAL(pv0, &v0);
    BOOST_CHECK_EQUAL(v0, v1);
  }

  for (auto&& [str, idx] : idxs) {
    BipartiteIndexList v1;
    auto pv1 = &(v1 = idx);
    BOOST_CHECK_EQUAL(pv1, &v1);
    BOOST_CHECK_EQUAL(idx, v1);
    // Ensure deep copy
    for (size_type i = 0; i < idx.size(); ++i) BOOST_CHECK(&idx[i] != &v1[i]);
  }
}

/* String assignment allows users to overwrite the state of a
 * BipartiteIndexList instance by assigning a C-string (or std::string) to
 * it. We test this operator by creating a default constructed
 * BipartiteIndexList, assigning the indices to it, and then comparing that
 * to the already created instance.
 */
BOOST_AUTO_TEST_CASE(string_assignment) {
  if (world.nproc() == 1) {
    BipartiteIndexList v1;
    BOOST_CHECK_THROW(v1.operator=("i,"), TiledArray::Exception);
  }

  for (auto&& [str, idx] : idxs) {
    BipartiteIndexList v1;
    auto pv1 = &(v1 = str);
    BOOST_CHECK_EQUAL(pv1, &v1);
    BOOST_CHECK_EQUAL(idx, v1);
  }
}

/* To test equality/inequality we consider instances which are default created,
 * as well as those holding vector, matrix, or vector-of-vector indices. Each
 * instance is compared using == and != to each other type of instance
 * (including permutations of indices)
 */
BOOST_AUTO_TEST_CASE(equality) {
  // Default instance comparisons
  {
    BipartiteIndexList v0;
    BOOST_CHECK_EQUAL(v0, BipartiteIndexList());
    BOOST_CHECK(!(v0 == BipartiteIndexList("i")));
    BOOST_CHECK(!(v0 == BipartiteIndexList("i,j")));
    BOOST_CHECK(!(v0 == BipartiteIndexList("i;j")));
  }

  // Vector Comparisons
  {
    BipartiteIndexList v0("i");
    BOOST_CHECK_EQUAL(v0, BipartiteIndexList("i"));
    BOOST_CHECK(!(v0 == BipartiteIndexList("j")));
    BOOST_CHECK(!(v0 == BipartiteIndexList("i,j")));
    BOOST_CHECK(!(v0 == BipartiteIndexList("i;j")));
  }

  // Matrix Comparisons
  {
    BipartiteIndexList v0("i,j");
    BOOST_CHECK_EQUAL(v0, BipartiteIndexList("i,j"));
    BOOST_CHECK(!(v0 == BipartiteIndexList("j,i")));
    BOOST_CHECK(!(v0 == BipartiteIndexList("i;j")));
  }

  // ToT Comparisons
  {
    BipartiteIndexList v0("i;j");
    BOOST_CHECK_EQUAL(v0, BipartiteIndexList("i;j"));
    BOOST_CHECK(!(v0 == BipartiteIndexList("j;i")));
  }
}

BOOST_AUTO_TEST_CASE(inequality) {
  // Default instance comparisons
  {
    BipartiteIndexList v0;
    BOOST_CHECK(!(v0 != BipartiteIndexList()));
    BOOST_CHECK(v0 != BipartiteIndexList("i"));
    BOOST_CHECK(v0 != BipartiteIndexList("i,j"));
    BOOST_CHECK(v0 != BipartiteIndexList("i;j"));
  }

  // Vector Comparisons
  {
    BipartiteIndexList v0("i");
    BOOST_CHECK(!(v0 != BipartiteIndexList("i")));
    BOOST_CHECK(v0 != BipartiteIndexList("j"));
    BOOST_CHECK(v0 != BipartiteIndexList("i,j"));
    BOOST_CHECK(v0 != BipartiteIndexList("i;j"));
  }

  // Matrix Comparisons
  {
    BipartiteIndexList v0("i,j");
    BOOST_CHECK(!(v0 != BipartiteIndexList("i,j")));
    BOOST_CHECK(v0 != BipartiteIndexList("j,i"));
    BOOST_CHECK(v0 != BipartiteIndexList("i;j"));
  }

  // ToT Comparisons
  {
    BipartiteIndexList v0("i;j");
    BOOST_CHECK(!(v0 != BipartiteIndexList("i;j")));
    BOOST_CHECK(v0 != BipartiteIndexList("j;i"));
  }
}

BOOST_AUTO_TEST_CASE(permute_in_place) {
  if (world.nproc() == 1) {
    BipartiteIndexList v0;
    Permutation p{0, 1};
    BOOST_CHECK_THROW(v0 *= p, TiledArray::Exception);
  }

  Permutation p({1, 2, 3, 0});
  BipartiteIndexList v1("a, b, c, d");
  auto* pv1 = &(v1 *= p);
  BOOST_CHECK_EQUAL(pv1, &v1);
  BOOST_CHECK_EQUAL(v1[0], "d");
  BOOST_CHECK_EQUAL(v1[1], "a");
  BOOST_CHECK_EQUAL(v1[2], "b");
  BOOST_CHECK_EQUAL(v1[3], "c");
}

/* To test the iterators we assume that begin/end simply return the iterators
 * which run over the object returned by data() (we also assume that the object
 * returned by data() has the correct state). Subject to these assumptions it
 * suffices to test that data().begin() and data().end() equal begin() and end()
 * respectively.
 */
BOOST_AUTO_TEST_CASE(begin_itr) {
  {
    BipartiteIndexList v0;
    bool are_same = (v0.begin() == v0.data().begin());
    BOOST_CHECK(are_same);
  }

  for (auto&& [str, idx] : idxs) {
    bool are_same = (idx.begin() == idx.data().begin());
    BOOST_CHECK(are_same);
  }
}

BOOST_AUTO_TEST_CASE(end_itr) {
  {
    BipartiteIndexList v0;
    bool are_same = (v0.end() == v0.data().end());
    BOOST_CHECK(are_same);
  }

  for (auto&& [str, idx] : idxs) {
    bool are_same = (idx.end() == idx.data().end());
    BOOST_CHECK(are_same);
  }
}

/* To test the "at" member we split the input string manually using split_index
 * and loop over the resulting outer and inner indices. This test is basically
 * identical to the one we use for subscript operator, except we also test for
 * out-of-range on the "at" member.
 */
BOOST_AUTO_TEST_CASE(at_member) {
  for (auto&& [str, idx] : idxs) {
    if (world.nproc() == 1) {
      BOOST_CHECK_THROW(idx.at(idx.size()), std::out_of_range);
    }
    auto [outer, inner] = detail::split_index(str);
    for (size_type i = 0; i < outer.size(); ++i)
      BOOST_CHECK_EQUAL(idx.at(i), outer.at(i));
    for (size_type i = 0; i < inner.size(); ++i)
      BOOST_CHECK_EQUAL(idx.at(i + outer.size()), inner.at(i));
  }
}

BOOST_AUTO_TEST_CASE(subscript_operator) {
  for (auto&& [str, idx] : idxs) {
    auto [outer, inner] = detail::split_index(str);
    for (size_type i = 0; i < outer.size(); ++i)
      BOOST_CHECK_EQUAL(idx[i], outer.at(i));
    for (size_type i = 0; i < inner.size(); ++i)
      BOOST_CHECK_EQUAL(idx[i + outer.size()], inner.at(i));
  }
}

// TODO: Test repeated index
BOOST_AUTO_TEST_CASE(modes) {
  for (auto&& [str, idx] : idxs) {
    using r_type = decltype(idx.positions(""));
    using size_type = typename BipartiteIndexList::size_type;
    auto [outer, inner] = detail::split_index(str);
    for (size_type i = 0; i < outer.size(); ++i)
      BOOST_CHECK_EQUAL(idx.positions(outer[i]), r_type{i});
    for (size_type i = 0; i < inner.size(); ++i)
      BOOST_CHECK_EQUAL(idx.positions(inner[i]), r_type{outer.size() + i});
  }
}

BOOST_AUTO_TEST_CASE(count) {
  for (auto&& [str, idx] : idxs) {
    auto [outer, inner] = detail::split_index(str);
    for (const auto& x : outer) BOOST_CHECK_EQUAL(idx.count(x), 1);
    for (const auto& x : inner) BOOST_CHECK_EQUAL(idx.count(x), 1);
    BOOST_CHECK_EQUAL(idx.count("not_an_index"), 0);
  }
}

/* To test the dim, outer_size, inner_size, and size function we simply loop
 * over the BipartiteIndexList instances and compare the results of the member
 * function to the sizes resulting from calling `split_index`, which defines how
 * a TiledArray index should be split.
 */
BOOST_AUTO_TEST_CASE(dim_fxn) {
  BOOST_CHECK_EQUAL(BipartiteIndexList{}.size(), 0);

  for (auto&& [str, idx] : idxs) {
    auto [outer, inner] = detail::split_index(str);
    BOOST_CHECK_EQUAL(idx.size(), outer.size() + inner.size());
  }
}

BOOST_AUTO_TEST_CASE(outer_size_fxn) {
  BOOST_CHECK_EQUAL(BipartiteIndexList{}.first_size(), 0);

  for (auto&& [str, idx] : idxs) {
    auto [outer, inner] = detail::split_index(str);
    BOOST_CHECK_EQUAL(idx.first_size(), outer.size());
  }
}

BOOST_AUTO_TEST_CASE(inner_size_fxn) {
  BOOST_CHECK_EQUAL(BipartiteIndexList{}.second_size(), 0);

  for (auto&& [str, idx] : idxs) {
    auto [outer, inner] = detail::split_index(str);
    BOOST_CHECK_EQUAL(idx.second_size(), inner.size());
  }
}

BOOST_AUTO_TEST_CASE(size_fxn) {
  BOOST_CHECK_EQUAL(BipartiteIndexList{}.size(), 0);

  for (auto&& [str, idx] : idxs) {
    auto [outer, inner] = detail::split_index(str);
    BOOST_CHECK_EQUAL(idx.size(), outer.size() + inner.size());
  }
}

BOOST_AUTO_TEST_CASE(is_tot_fxn) {
  BOOST_CHECK(BipartiteIndexList{}.second_size() == 0);

  for (auto&& [str, idx] : idxs) {
    auto [outer, inner] = detail::split_index(str);
    BOOST_CHECK((inner_size(idx) != 0) == detail::is_tot_index(str));
  }
}

/* The string cast operator simply concatenates the indices together using a
 * comma as glue between modes and a semicolon as glue between tensor nestings.
 * This test loops over the BipartiteIndexList instances in the test fixture,
 * converts them to std::string instances and compares the resulting string
 *
 * Note: This test exploits the fact that there's no spaces in any of the
 * indices. If an index with spaces is added, str should be run through
 * detail::remove_whitespace to compensate.
 */
BOOST_AUTO_TEST_CASE(string_cast) {
  BipartiteIndexList v;
  BOOST_CHECK_EQUAL("", static_cast<value_type>(v));

  for (auto [str, idx] : idxs) {
    BOOST_CHECK_EQUAL(str, static_cast<value_type>(idx));
  }
}

BOOST_AUTO_TEST_CASE(stream_insert) {
  {
    BipartiteIndexList v;
    std::stringstream ss;
    ss << v;
    BOOST_CHECK_EQUAL("()", ss.str());
  }

  for (auto [str, idx] : idxs) {
    std::stringstream ss;
    ss << idx;
    std::string corr = "(" + str + ")";
    BOOST_CHECK_EQUAL(corr, ss.str());
  }
}

/* To test swap we swap the contents of an already made BipartiteIndexList
 * with the contents of a defaulted BipartiteIndexList. The instance that was
 * defaulted is then compared to a copy of the originally non-defaulted instance
 * and the instance that was non-defaulted is compared to a fresh default
 * instance. Strictly speaking this only shows that we can swap defaulted and
 * non-defaulted instances.
 */
BOOST_AUTO_TEST_CASE(swap_fxn) {
  for (auto&& [str, idx] : idxs) {
    BipartiteIndexList v0, v1(idx);
    idx.swap(v0);
    BOOST_CHECK(idx == BipartiteIndexList{});
    BOOST_CHECK(v0 == v1);
  }
}

BOOST_AUTO_TEST_CASE(swap_free_fxn) {
  for (auto&& [str, idx] : idxs) {
    BipartiteIndexList v0, v1(idx);
    swap(idx, v0);
    BOOST_CHECK(idx == BipartiteIndexList{});
    BOOST_CHECK(v0 == v1);
  }
}

BOOST_AUTO_TEST_CASE(permutation_fxn) {
  if (world.nproc() == 1) {
    BipartiteIndexList v0("i, j");

    {  // not both ToT
      BipartiteIndexList v1("i;j");
      BOOST_CHECK_THROW(v1.permutation(v0), TiledArray::Exception);
    }

    {  // wrong size
      BipartiteIndexList v1("i");
      BOOST_CHECK_THROW(v1.permutation(v0), TiledArray::Exception);
    }

    {  // not a permutation
      BipartiteIndexList v1("i, a");
      BOOST_CHECK_THROW(v1.permutation(v0), TiledArray::Exception);
    }

    {  // ToTs mix outer and inner
      BipartiteIndexList v1("i,j;k,l");
      BipartiteIndexList v2("i,k;j,l");
      BOOST_CHECK_THROW(v1.permutation(v2), TiledArray::Exception);
    }
  }

  for (auto [str, idx] : idxs) {
    std::vector<size_type> perm(idx.size());
    std::iota(perm.begin(), perm.end(), 0);
    if (idx.second_size() != 0) {
      auto outer_size = idx.first_size();
      do {
        do {
          Permutation p(perm.begin(), perm.end());
          auto v = p * idx;
          BOOST_CHECK(v.permutation(idx) ==
                      BipartitePermutation(p, perm.size() - outer_size));
        } while (std::next_permutation(perm.begin() + outer_size, perm.end()));
      } while (std::next_permutation(perm.begin(), perm.begin() + outer_size));
    } else {
      do {
        Permutation p(perm.begin(), perm.end());
        auto v = p * idx;
        BOOST_CHECK(v.permutation(idx) == BipartitePermutation(p, 0));
      } while (std::next_permutation(perm.begin(), perm.end()));
    }
  }
}

BOOST_AUTO_TEST_CASE(is_permutation_fxn) {
  BipartiteIndexList v0("i, j");
  {  // Different number of variables
    BipartiteIndexList v1("i");
    BOOST_CHECK(v0.is_permutation(v1) == false);
  }

  {  // Different indices
    BipartiteIndexList v1("i, a");
    BOOST_CHECK(v1.is_permutation(v0) == false);
  }

  for (auto [str, idx] : idxs) {
    std::vector<size_type> perm(idx.size());
    std::iota(perm.begin(), perm.end(), 0);
    do {
      Permutation p(perm.begin(), perm.end());
      auto v = p * idx;
      BOOST_CHECK(v.is_permutation(idx));
    } while (std::next_permutation(perm.begin(), perm.end()));
  }
}

BOOST_AUTO_TEST_CASE(implicit_permutation) {
  Permutation p1({1, 2, 3, 0});
  BipartiteIndexList v("a,b,c,d");
  BipartiteIndexList v1 = (p1 * v);
  auto p = TiledArray::expressions::detail::var_perm(v1, v);

  BOOST_CHECK_EQUAL_COLLECTIONS(p.begin(), p.end(), p1.begin(), p1.end());
}

BOOST_AUTO_TEST_CASE(common) {
  std::pair<BipartiteIndexList::const_iterator,
            BipartiteIndexList::const_iterator>
      p1;
  std::pair<BipartiteIndexList::const_iterator,
            BipartiteIndexList::const_iterator>
      p2;
  BipartiteIndexList v("a,b,c,d");
  BipartiteIndexList v_aib("a,i,b");
  BipartiteIndexList v_xiy("x,i,y");
  BipartiteIndexList v_ai("a,i");
  BipartiteIndexList v_xi("x,i");
  BipartiteIndexList v_ia("i,a");
  BipartiteIndexList v_ix("i,x");
  BipartiteIndexList v_i("i");

  find_common(v_aib.begin(), v_aib.end(), v_xiy.begin(), v_xiy.end(), p1, p2);
  BOOST_CHECK(p1.first == v_aib.begin() + 1);
  BOOST_CHECK(p1.second == v_aib.begin() + 2);
  BOOST_CHECK(p2.first == v_xiy.begin() + 1);
  BOOST_CHECK(p2.second == v_xiy.begin() + 2);
  find_common(v_aib.begin(), v_aib.end(), v_xi.begin(), v_xi.end(), p1, p2);
  BOOST_CHECK(p1.first == v_aib.begin() + 1);
  BOOST_CHECK(p1.second == v_aib.begin() + 2);
  BOOST_CHECK(p2.first == v_xi.begin() + 1);
  BOOST_CHECK(p2.second == v_xi.begin() + 2);
  find_common(v_aib.begin(), v_aib.end(), v_ix.begin(), v_ix.end(), p1, p2);
  BOOST_CHECK(p1.first == v_aib.begin() + 1);
  BOOST_CHECK(p1.second == v_aib.begin() + 2);
  BOOST_CHECK(p2.first == v_ix.begin());
  BOOST_CHECK(p2.second == v_ix.begin() + 1);
  find_common(v_aib.begin(), v_aib.end(), v_i.begin(), v_i.end(), p1, p2);
  BOOST_CHECK(p1.first == v_aib.begin() + 1);
  BOOST_CHECK(p1.second == v_aib.begin() + 2);
  BOOST_CHECK(p2.first == v_i.begin());
  BOOST_CHECK(p2.second == v_i.begin() + 1);
  find_common(v_ai.begin(), v_ai.end(), v_xiy.begin(), v_xiy.end(), p1, p2);
  BOOST_CHECK(p1.first == v_ai.begin() + 1);
  BOOST_CHECK(p1.second == v_ai.begin() + 2);
  BOOST_CHECK(p2.first == v_xiy.begin() + 1);
  BOOST_CHECK(p2.second == v_xiy.begin() + 2);
  find_common(v_ai.begin(), v_ai.end(), v_xi.begin(), v_xi.end(), p1, p2);
  BOOST_CHECK(p1.first == v_ai.begin() + 1);
  BOOST_CHECK(p1.second == v_ai.begin() + 2);
  BOOST_CHECK(p2.first == v_xi.begin() + 1);
  BOOST_CHECK(p2.second == v_xi.begin() + 2);
  find_common(v_ai.begin(), v_ai.end(), v_ix.begin(), v_ix.end(), p1, p2);
  BOOST_CHECK(p1.first == v_ai.begin() + 1);
  BOOST_CHECK(p1.second == v_ai.begin() + 2);
  BOOST_CHECK(p2.first == v_ix.begin());
  BOOST_CHECK(p2.second == v_ix.begin() + 1);
  find_common(v_ai.begin(), v_ai.end(), v_i.begin(), v_i.end(), p1, p2);
  BOOST_CHECK(p1.first == v_ai.begin() + 1);
  BOOST_CHECK(p1.second == v_ai.begin() + 2);
  BOOST_CHECK(p2.first == v_i.begin());
  BOOST_CHECK(p2.second == v_i.begin() + 1);
  find_common(v_ia.begin(), v_ia.end(), v_xiy.begin(), v_xiy.end(), p1, p2);
  BOOST_CHECK(p1.first == v_ia.begin());
  BOOST_CHECK(p1.second == v_ia.begin() + 1);
  BOOST_CHECK(p2.first == v_xiy.begin() + 1);
  BOOST_CHECK(p2.second == v_xiy.begin() + 2);
  find_common(v_ia.begin(), v_ia.end(), v_xi.begin(), v_xi.end(), p1, p2);
  BOOST_CHECK(p1.first == v_ia.begin());
  BOOST_CHECK(p1.second == v_ia.begin() + 1);
  BOOST_CHECK(p2.first == v_xi.begin() + 1);
  BOOST_CHECK(p2.second == v_xi.begin() + 2);
  find_common(v_ia.begin(), v_ia.end(), v_ix.begin(), v_ix.end(), p1, p2);
  BOOST_CHECK(p1.first == v_ia.begin());
  BOOST_CHECK(p1.second == v_ia.begin() + 1);
  BOOST_CHECK(p2.first == v_ix.begin());
  BOOST_CHECK(p2.second == v_ix.begin() + 1);
  find_common(v_ia.begin(), v_ia.end(), v_i.begin(), v_i.end(), p1, p2);
  BOOST_CHECK(p1.first == v_ia.begin());
  BOOST_CHECK(p1.second == v_ia.begin() + 1);
  BOOST_CHECK(p2.first == v_i.begin());
  BOOST_CHECK(p2.second == v_i.begin() + 1);
  find_common(v_i.begin(), v_i.end(), v_xiy.begin(), v_xiy.end(), p1, p2);
  BOOST_CHECK(p1.first == v_i.begin());
  BOOST_CHECK(p1.second == v_i.begin() + 1);
  BOOST_CHECK(p2.first == v_xiy.begin() + 1);
  BOOST_CHECK(p2.second == v_xiy.begin() + 2);
  find_common(v_i.begin(), v_i.end(), v_xi.begin(), v_xi.end(), p1, p2);
  BOOST_CHECK(p1.first == v_i.begin());
  BOOST_CHECK(p1.second == v_i.begin() + 1);
  BOOST_CHECK(p2.first == v_xi.begin() + 1);
  BOOST_CHECK(p2.second == v_xi.begin() + 2);
  find_common(v_i.begin(), v_i.end(), v_ix.begin(), v_ix.end(), p1, p2);
  BOOST_CHECK(p1.first == v_i.begin());
  BOOST_CHECK(p1.second == v_i.begin() + 1);
  BOOST_CHECK(p2.first == v_ix.begin());
  BOOST_CHECK(p2.second == v_ix.begin() + 1);
  find_common(v_i.begin(), v_i.end(), v_i.begin(), v_i.end(), p1, p2);
  BOOST_CHECK(p1.first == v_i.begin());
  BOOST_CHECK(p1.second == v_i.begin() + 1);
  BOOST_CHECK(p2.first == v_i.begin());
  BOOST_CHECK(p2.second == v_i.begin() + 1);
  //  find_common(v_0.begin(), v_0.end(), v_i.begin(), v_i.end(), p1, p2);
  //  BOOST_CHECK(p1.first == v_0.end());
  //  BOOST_CHECK(p1.second == v_0.end());
  //  BOOST_CHECK(p2.first == v_i.end());
  //  BOOST_CHECK(p2.second == v_i.end());
}

BOOST_AUTO_TEST_SUITE_END()
