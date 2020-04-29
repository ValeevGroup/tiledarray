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

#include "TiledArray/expressions/variable_list.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;
using TiledArray::expressions::VariableList;
using TiledArray::expressions::detail::find_common;

// Pull some typedefs from the class to make sure testing uses the right types
using value_type      = typename VariableList::value_type;
using const_reference = typename VariableList::const_reference;
using size_type       = typename VariableList::size_type;

struct VariableListFixture {
  VariableListFixture()
      : v("a,b,c,d"),
        v_aib("a,i,b"),
        v_xiy("x,i,y"),
        v_ai("a,i"),
        v_xi("x,i"),
        v_ia("i,a"),
        v_ix("i,x"),
        v_i("i"),
        v_0(){}

  World& world = get_default_world();
  std::map<value_type, VariableList> idxs = {
      {"i", VariableList("i")},
      {"i,j", VariableList("i,j")},
      {"i,j,k", VariableList("i,j,k")},
      {"i;j", VariableList("i;j")},
      {"i;j,k", VariableList("i;j,k")},
      {"i,j;k", VariableList("i,j;k")},
      {"i,j;k,l", VariableList("i,j;k,l")}
  };

  VariableList v;
  const VariableList v_aib;
  const VariableList v_xiy;
  const VariableList v_ai;
  const VariableList v_xi;
  const VariableList v_ia;
  const VariableList v_ix;
  const VariableList v_i;
  const VariableList v_0;
};

BOOST_FIXTURE_TEST_SUITE(variable_list_suite, VariableListFixture)

/* This unit test ensures that the typedefs are what we think they are. Since no
 * template meta-programming occurs in the class these tests serve more as a
 * consistency check of the API.
 */
BOOST_AUTO_TEST_CASE(typedefs){
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
        typename VariableList::const_iterator,
        typename std::vector<value_type>::const_iterator>;
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
BOOST_AUTO_TEST_CASE(default_ctor){
  BOOST_REQUIRE_NO_THROW(VariableList v0);
  VariableList v0;
  {
    bool are_same = v0.begin() == v0.end();
    BOOST_CHECK(are_same);
  }
  BOOST_CHECK_EQUAL(v0.dim(), size_type{0});
  BOOST_CHECK_EQUAL(v0.outer_dim(), size_type{0});
  BOOST_CHECK_EQUAL(v0.inner_dim(), size_type{0});
  BOOST_CHECK_EQUAL(v0.size(), size_type{0});
}

/* This unit test tests the constructor which takes a string and tokenizes it.
 * The constructor ultimately calls detail::split_index, which is unit tested
 * elsewhere (and thus assumed to work). We compare the state of the resulting
 * VariableList to the correct value (as obtained by detail::split_index) for
 * this unit test. There are a number of ways of accessing the individual
 * indices (operator[], at, iterators); here we only test the at variant. The
 * other access members are tested in their respective unit tests.
 */
BOOST_AUTO_TEST_CASE(string_ctor){
  if(world.nproc() == 1){
    BOOST_CHECK_THROW(VariableList("i,"), TiledArray::Exception);
  }

  for(auto&& [str, idx] : idxs){
    auto&& [outer, inner] = TiledArray::detail::split_index(str);
    BOOST_CHECK_EQUAL(idx.dim(), outer.size() + inner.size());
    BOOST_CHECK_EQUAL(idx.outer_dim(), outer.size());
    BOOST_CHECK_EQUAL(idx.inner_dim(), inner.size());
    BOOST_CHECK_EQUAL(idx.size(), outer.size() + inner.size());

    for(size_type i = 0; i < outer.size(); ++i){
      BOOST_CHECK_EQUAL(idx.at(i), outer[i]);
    }

    for(size_type i = 0; i < inner.size(); ++i){
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
BOOST_AUTO_TEST_CASE(copy_ctor){
  // Default instance
  {
    VariableList v0;
    VariableList v1(v0);
    BOOST_CHECK_EQUAL(v0, v1);
  }

  for(auto&& [str, idx] : idxs){
    VariableList v1(idx);
    BOOST_CHECK_EQUAL(idx, v1);
    //Ensure deep copy
    for(size_type i = 0; i < idx.size(); ++i)
      BOOST_CHECK(&idx[i] != &v1[i]);
  }
}

BOOST_AUTO_TEST_CASE(copy_assignment){
  // Default instance
  {
    VariableList v0, v1;
    auto pv0 = &(v0 = v1);
    BOOST_CHECK_EQUAL(pv0, &v0);
    BOOST_CHECK_EQUAL(v0, v1);
  }

  for(auto&& [str, idx] : idxs){
    VariableList v1;
    auto pv1 = &(v1 = idx);
    BOOST_CHECK_EQUAL(pv1, &v1);
    BOOST_CHECK_EQUAL(idx, v1);
    //Ensure deep copy
    for(size_type i = 0; i < idx.size(); ++i)
      BOOST_CHECK(&idx[i] != &v1[i]);
  }
}

/* String assignment allows users to overwrite the state of a VariableList
 * instance by assigning a C-string (or std::string) to it. We test this
 * operator by creating a default constructed VariableList, assigning the
 * indices to it, and then comparing that to the already created instance.
 */
BOOST_AUTO_TEST_CASE(string_assignment){

  if(world.nproc() == 1){
    VariableList v1;
    BOOST_CHECK_THROW(v1.operator=("i,"), TiledArray::Exception);
  }

  for(auto&& [str, idx] : idxs){
    VariableList v1;
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
BOOST_AUTO_TEST_CASE(equality){
  // Default instance comparisons
  {
    VariableList v0;
    BOOST_CHECK_EQUAL(v0, VariableList());
    BOOST_CHECK(!(v0 == VariableList("i")));
    BOOST_CHECK(!(v0 == VariableList("i,j")));
    BOOST_CHECK(!(v0 == VariableList("i;j")));
  }

  // Vector Comparisons
  {
    VariableList v0("i");
    BOOST_CHECK_EQUAL(v0, VariableList("i"));
    BOOST_CHECK(!(v0 == VariableList("j")));
    BOOST_CHECK(!(v0 == VariableList("i,j")));
    BOOST_CHECK(!(v0 == VariableList("i;j")));
  }

  // Matrix Comparisons
  {
    VariableList v0("i,j");
    BOOST_CHECK_EQUAL(v0, VariableList("i,j"));
    BOOST_CHECK(!(v0 == VariableList("j,i")));
    BOOST_CHECK(!(v0 == VariableList("i;j")));
  }

  // ToT Comparisons
  {
    VariableList v0("i;j");
    BOOST_CHECK_EQUAL(v0, VariableList("i;j"));
    BOOST_CHECK(!(v0 == VariableList("j;i")));
  }
}

BOOST_AUTO_TEST_CASE(inequality){
  // Default instance comparisons
  {
    VariableList v0;
    BOOST_CHECK(!(v0 != VariableList()));
    BOOST_CHECK(v0 != VariableList("i"));
    BOOST_CHECK(v0 != VariableList("i,j"));
    BOOST_CHECK(v0 != VariableList("i;j"));
  }

  // Vector Comparisons
  {
    VariableList v0("i");
    BOOST_CHECK(!(v0 != VariableList("i")));
    BOOST_CHECK(v0 != VariableList("j"));
    BOOST_CHECK(v0 != VariableList("i,j"));
    BOOST_CHECK(v0 != VariableList("i;j"));
  }

  // Matrix Comparisons
  {
    VariableList v0("i,j");
    BOOST_CHECK(!(v0 != VariableList("i,j")));
    BOOST_CHECK(v0 != VariableList("j,i"));
    BOOST_CHECK(v0 != VariableList("i;j"));
  }

  // ToT Comparisons
  {
    VariableList v0("i;j");
    BOOST_CHECK(!(v0 != VariableList("i;j")));
    BOOST_CHECK(v0 != VariableList("j;i"));
  }
}

// TODO: More permutation testing
BOOST_AUTO_TEST_CASE(permute_in_place){
  if(world.nproc() == 1){
    VariableList v0;
    Permutation p{0, 1};
    BOOST_CHECK_THROW(v0 *= p, TiledArray::Exception);
  }
}

/* To test the iterators we assume that begin/end simply return the iterators
 * which run over the object returned by data() (we also assume that the object
 * returned by data() has the correct state). Subject to these assumptions it
 * suffices to test that data().begin() and data().end() equal begin() and end()
 * respectively.
 */
BOOST_AUTO_TEST_CASE(begin_itr){
  {
    VariableList v0;
    bool are_same = (v0.begin() == v0.data().begin());
    BOOST_CHECK(are_same);
  }

  for(auto&& [str, idx] : idxs){
    bool are_same = (idx.begin() == idx.data().begin());
    BOOST_CHECK(are_same);
  }
}

BOOST_AUTO_TEST_CASE(end_itr){
  {
    VariableList v0;
    bool are_same = (v0.end() == v0.data().end());
    BOOST_CHECK(are_same);
  }

  for(auto&& [str, idx] : idxs){
    bool are_same = (idx.end() == idx.data().end());
    BOOST_CHECK(are_same);
  }
}

BOOST_AUTO_TEST_CASE(at_member){

}


BOOST_AUTO_TEST_CASE(valid_chars) {
  // Check for valid characters in string input

  // Check letters 'a' - 'z'
  BOOST_CHECK_NO_THROW(VariableList v1("a"));
  BOOST_CHECK_NO_THROW(VariableList v1("b"));
  BOOST_CHECK_NO_THROW(VariableList v1("c"));
  BOOST_CHECK_NO_THROW(VariableList v1("d"));
  BOOST_CHECK_NO_THROW(VariableList v1("e"));
  BOOST_CHECK_NO_THROW(VariableList v1("f"));
  BOOST_CHECK_NO_THROW(VariableList v1("g"));
  BOOST_CHECK_NO_THROW(VariableList v1("h"));
  BOOST_CHECK_NO_THROW(VariableList v1("i"));
  BOOST_CHECK_NO_THROW(VariableList v1("j"));
  BOOST_CHECK_NO_THROW(VariableList v1("k"));
  BOOST_CHECK_NO_THROW(VariableList v1("l"));
  BOOST_CHECK_NO_THROW(VariableList v1("m"));
  BOOST_CHECK_NO_THROW(VariableList v1("n"));
  BOOST_CHECK_NO_THROW(VariableList v1("o"));
  BOOST_CHECK_NO_THROW(VariableList v1("p"));
  BOOST_CHECK_NO_THROW(VariableList v1("q"));
  BOOST_CHECK_NO_THROW(VariableList v1("r"));
  BOOST_CHECK_NO_THROW(VariableList v1("s"));
  BOOST_CHECK_NO_THROW(VariableList v1("t"));
  BOOST_CHECK_NO_THROW(VariableList v1("u"));
  BOOST_CHECK_NO_THROW(VariableList v1("v"));
  BOOST_CHECK_NO_THROW(VariableList v1("w"));
  BOOST_CHECK_NO_THROW(VariableList v1("x"));
  BOOST_CHECK_NO_THROW(VariableList v1("y"));
  BOOST_CHECK_NO_THROW(VariableList v1("z"));

  // Check letters 'A' - 'Z'
  BOOST_CHECK_NO_THROW(VariableList v1("A"));
  BOOST_CHECK_NO_THROW(VariableList v1("B"));
  BOOST_CHECK_NO_THROW(VariableList v1("C"));
  BOOST_CHECK_NO_THROW(VariableList v1("D"));
  BOOST_CHECK_NO_THROW(VariableList v1("E"));
  BOOST_CHECK_NO_THROW(VariableList v1("F"));
  BOOST_CHECK_NO_THROW(VariableList v1("G"));
  BOOST_CHECK_NO_THROW(VariableList v1("H"));
  BOOST_CHECK_NO_THROW(VariableList v1("I"));
  BOOST_CHECK_NO_THROW(VariableList v1("J"));
  BOOST_CHECK_NO_THROW(VariableList v1("K"));
  BOOST_CHECK_NO_THROW(VariableList v1("L"));
  BOOST_CHECK_NO_THROW(VariableList v1("M"));
  BOOST_CHECK_NO_THROW(VariableList v1("N"));
  BOOST_CHECK_NO_THROW(VariableList v1("O"));
  BOOST_CHECK_NO_THROW(VariableList v1("P"));
  BOOST_CHECK_NO_THROW(VariableList v1("Q"));
  BOOST_CHECK_NO_THROW(VariableList v1("R"));
  BOOST_CHECK_NO_THROW(VariableList v1("S"));
  BOOST_CHECK_NO_THROW(VariableList v1("T"));
  BOOST_CHECK_NO_THROW(VariableList v1("U"));
  BOOST_CHECK_NO_THROW(VariableList v1("V"));
  BOOST_CHECK_NO_THROW(VariableList v1("W"));
  BOOST_CHECK_NO_THROW(VariableList v1("X"));
  BOOST_CHECK_NO_THROW(VariableList v1("Y"));
  BOOST_CHECK_NO_THROW(VariableList v1("Z"));

  // Check characters '0' - '9'
  BOOST_CHECK_NO_THROW(VariableList v1("0"));
  BOOST_CHECK_NO_THROW(VariableList v1("1"));
  BOOST_CHECK_NO_THROW(VariableList v1("2"));
  BOOST_CHECK_NO_THROW(VariableList v1("3"));
  BOOST_CHECK_NO_THROW(VariableList v1("4"));
  BOOST_CHECK_NO_THROW(VariableList v1("5"));
  BOOST_CHECK_NO_THROW(VariableList v1("5"));
  BOOST_CHECK_NO_THROW(VariableList v1("6"));
  BOOST_CHECK_NO_THROW(VariableList v1("7"));
  BOOST_CHECK_NO_THROW(VariableList v1("8"));
  BOOST_CHECK_NO_THROW(VariableList v1("9"));

  // Check characters ',', ' ', and '\0'
  BOOST_CHECK_NO_THROW(VariableList v1("a,b"));
  BOOST_CHECK_NO_THROW(VariableList v1("a ,b"));
  BOOST_CHECK_NO_THROW(VariableList v1(""));
}

BOOST_AUTO_TEST_CASE(accessors) {
  BOOST_CHECK_EQUAL(v.dim(), 4u);   // check for variable count
  BOOST_CHECK_EQUAL(v.at(0), "a");  // check 1st variable access
  BOOST_CHECK_EQUAL(v.at(3), "d");  // check last variable access
  BOOST_CHECK_EQUAL(v[0], "a");     // check 1st variable access
  BOOST_CHECK_EQUAL(v[3], "d");     // check last variable access
  BOOST_CHECK_THROW(v.at(4),
                    std::out_of_range);  // check for out of range throw.
}

BOOST_AUTO_TEST_CASE(iterator) {
  std::array<std::string, 4> a1 = {{"a", "b", "c", "d"}};

  BOOST_CHECK_EQUAL_COLLECTIONS(v.begin(), v.end(), a1.begin(),
                                a1.end());  // check that all values are equal
  std::array<std::string, 4>::const_iterator it_a = a1.begin();
  for (VariableList::const_iterator it = v.begin(); it != v.end();
       ++it, ++it_a)  // check basic iterator functionality.
    BOOST_CHECK_EQUAL(*it, *it_a);
}

BOOST_AUTO_TEST_CASE(comparison) {
  VariableList v1("a,b,c,d");
  VariableList v2("d,b,c,a");

  BOOST_CHECK(v1 == v);  // check variable list comparison operators
  BOOST_CHECK(!(v2 == v));
  BOOST_CHECK(v2 != v);
  BOOST_CHECK(!(v1 != v));
}

BOOST_AUTO_TEST_CASE(assignment) {
  VariableList v1;
  v1 = v;
  BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), v.begin(), v.end());

  VariableList v2;
  v2 = "a,b,c,d";
  BOOST_CHECK_EQUAL_COLLECTIONS(v2.begin(), v2.end(), v.begin(), v.end());
}

BOOST_AUTO_TEST_CASE(permutation) {
  Permutation p({1, 2, 3, 0});
  VariableList v1(v);
  VariableList v2 = (p * v1);
  BOOST_CHECK_EQUAL(v2[0], "d");
  BOOST_CHECK_EQUAL(v2[1], "a");
  BOOST_CHECK_EQUAL(v2[2], "b");
  BOOST_CHECK_EQUAL(v2[3], "c");

  VariableList v3(v);
  v3 *= p;
  BOOST_CHECK_EQUAL_COLLECTIONS(v3.begin(), v3.end(), v2.begin(), v2.end());
}

BOOST_AUTO_TEST_CASE(implicit_permutation) {
  Permutation p1({1, 2, 3, 0});
  VariableList v1 = (p1 * v);
  Permutation p = TiledArray::expressions::detail::var_perm(v1, v);

  BOOST_CHECK_EQUAL_COLLECTIONS(p.begin(), p.end(), p1.begin(), p1.end());
}

BOOST_AUTO_TEST_CASE(ostream) {
  boost::test_tools::output_test_stream output;
  output << v;
  BOOST_CHECK(!output.is_empty(false));  // check for correct output.
  BOOST_CHECK(output.check_length(12, false));
  BOOST_CHECK(output.is_equal("(a, b, c, d)"));
}

BOOST_AUTO_TEST_CASE(common) {
  std::pair<VariableList::const_iterator, VariableList::const_iterator> p1;
  std::pair<VariableList::const_iterator, VariableList::const_iterator> p2;

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
  find_common(v_0.begin(), v_0.end(), v_i.begin(), v_i.end(), p1, p2);
  BOOST_CHECK(p1.first == v_0.end());
  BOOST_CHECK(p1.second == v_0.end());
  BOOST_CHECK(p2.first == v_i.end());
  BOOST_CHECK(p2.second == v_i.end());
}

BOOST_AUTO_TEST_SUITE_END()
