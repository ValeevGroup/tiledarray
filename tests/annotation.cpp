#include <TiledArray/util/annotation.h>
#include <tiledarray.h>
#include "unit_test_config.h"
using namespace TiledArray::detail;

namespace {

using string_vector_t = std::vector<std::string>;
using corr_map = std::map<std::string, std::string>;

// Single letter indices
corr_map i_idx{{"i", "i"}, {" i", "i"}, {"i ", "i"}};
corr_map j_idx{{"j", "j"}, {" j", "j"}, {"j ", "j"}};
corr_map k_idx{{"k", "k"}, {" k", "k"}, {"k ", "k"}};

auto combine_maps(const corr_map& lhs, const corr_map& rhs,
                  const std::string& joiner = "") {
  corr_map rv;
  for (auto& [lidx, lcorr] : lhs) {
    for (auto& [ridx, rcorr] : rhs) {
      rv[lidx + joiner + ridx] = lcorr + joiner + rcorr;
    }
  }
  return rv;
}

// matrix index
auto i_j_idx = combine_maps(i_idx, j_idx, ",");

// tensor-of-tensor indices
auto vov_idx = combine_maps(i_idx, j_idx, ";");
auto vom_idx = combine_maps(k_idx, i_j_idx, ";");
auto mov_idx = combine_maps(i_j_idx, k_idx, ";");
}  // namespace

/* We need to remove whitespace from all types of indices: single character,
 * multiple character, multiple mode, and tensor-of-tensor. This test suite
 * loops over some of the possible whitespace index combinations and ensures
 * that the resulting index is correct. Scenarios covered:
 *
 * - single character vector index
 * - multi-character vector index
 * - single character matrix index
 * - single character tensor index
 * - single character vector-of-vectors
 * - single character matrix-of-vectors
 * - single character vector-of-matrices
 */
BOOST_AUTO_TEST_SUITE(remove_whitespace_fxn)

BOOST_AUTO_TEST_CASE(single_character) {
  for (auto& [idx, corr] : i_idx) BOOST_CHECK(remove_whitespace(idx) == corr);
}

BOOST_AUTO_TEST_CASE(multicharacter) {
  auto ij_idx = combine_maps(i_idx, j_idx);

  for (auto& [idx, corr] : ij_idx) BOOST_CHECK(remove_whitespace(idx) == corr);
}

BOOST_AUTO_TEST_CASE(matrix_index) {
  for (auto& [idx, corr] : i_j_idx) BOOST_CHECK(remove_whitespace(idx) == corr);
}

BOOST_AUTO_TEST_CASE(tensor_index) {
  auto i_j_k_idx = combine_maps(i_j_idx, k_idx, ",");
  for (auto& [idx, corr] : i_j_k_idx)
    BOOST_CHECK(remove_whitespace(idx) == corr);
}

BOOST_AUTO_TEST_CASE(vector_of_vector_index) {
  for (auto& [idx, corr] : vov_idx) BOOST_CHECK(remove_whitespace(idx) == corr);
}

BOOST_AUTO_TEST_CASE(vector_of_matrix) {
  for (auto& [idx, corr] : vom_idx) BOOST_CHECK(remove_whitespace(idx) == corr);
}

BOOST_AUTO_TEST_CASE(matrix_of_vector) {
  for (auto& [idx, corr] : mov_idx) BOOST_CHECK(remove_whitespace(idx) == corr);
}

BOOST_AUTO_TEST_SUITE_END()

/* We really only split on commas and semicolons. We thus test that the function
 * can split correctly on commas and semicolons. Also tested are a number of
 * spacing options, the case when the delimiter does not show up, when the
 * delimiter is at the end of the index, and when there are two delimiters.
 */
BOOST_AUTO_TEST_SUITE(tokenize_index_fxn)

BOOST_AUTO_TEST_CASE(split_on_comma) {
  std::map<std::string, string_vector_t> input2corr{
      {"", string_vector_t{""}},
      {"hello world", string_vector_t{"hello world"}},
      {"hello,world", string_vector_t{"hello", "world"}},
      {" hello,world", string_vector_t{" hello", "world"}},
      {"hello ,world", string_vector_t{"hello ", "world"}},
      {"hello, world", string_vector_t{"hello", " world"}},
      {"hello,world ", string_vector_t{"hello", "world "}},
      {"hello world,", string_vector_t{"hello world", ""}},
      {"hello world, ", string_vector_t{"hello world", " "}},
      {"1,2,3", string_vector_t{"1", "2", "3"}}};
  for (auto& [str, corr] : input2corr)
    BOOST_CHECK(tokenize_index(str, ',') == corr);
}

BOOST_AUTO_TEST_CASE(split_on_semicolon) {
  std::map<std::string, string_vector_t> input2corr{
      {"", string_vector_t{""}},
      {"hello world", string_vector_t{"hello world"}},
      {"hello;world", string_vector_t{"hello", "world"}},
      {" hello;world", string_vector_t{" hello", "world"}},
      {"hello ;world", string_vector_t{"hello ", "world"}},
      {"hello; world", string_vector_t{"hello", " world"}},
      {"hello;world ", string_vector_t{"hello", "world "}},
      {"hello world;", string_vector_t{"hello world", ""}},
      {"hello world;", string_vector_t{"hello world", " "}},
      {"1;2;3", string_vector_t{"1", "2", "3"}}};
  for (auto& [str, corr] : input2corr)
    BOOST_CHECK(tokenize_index(str, ';') == corr);
}

BOOST_AUTO_TEST_SUITE_END()

/* To test the is_valid_index function we loop over the indices we've already
 * made and assert that they are indeed valid. This does not exhaustively test
 * that the entire alphabet. We also ensure that indices containing banned
 * characters, only whitespace, multiple semicolons, and anonymous index names
 * are properly marked as invalid.
 */
BOOST_AUTO_TEST_SUITE(is_valid_index_fxn)

BOOST_AUTO_TEST_CASE(valid_indices) {
  for (auto idx_set : {i_idx, i_j_idx, vov_idx, vom_idx, mov_idx}) {
    for (auto& [idx, corr] : idx_set) {
      BOOST_CHECK(is_valid_index(idx));
    }
  }
  // all valid characters forming index name
  BOOST_CHECK(
      is_valid_index("abcdefghijklmnopqrstuvwxyz,ABCDEFGHIJKLMNOPQRSTUVWXYZ;'`_"
                     "~!@#$%^&*-+.,/?:|<>[]{}"));
}

BOOST_AUTO_TEST_CASE(unallowed_character) {
  BOOST_CHECK(is_valid_index("i,\",j") == false);
  BOOST_CHECK(is_valid_index("i,\\,j") == false);
}

BOOST_AUTO_TEST_CASE(multiple_semicolons) {
  BOOST_CHECK(is_valid_index("i;j;k") == false);
}

BOOST_AUTO_TEST_CASE(at_least_one_index) {
  BOOST_CHECK(is_valid_index("") == false);
  BOOST_CHECK(is_valid_index(" ") == false);
  BOOST_CHECK(is_valid_index("     ") == false);
}

BOOST_AUTO_TEST_CASE(empty_index_name) {
  BOOST_CHECK(is_valid_index("i,") == false);
  BOOST_CHECK(is_valid_index(",i") == false);
  BOOST_CHECK(is_valid_index("i,,j") == false);
  BOOST_CHECK(is_valid_index("i;") == false);
  BOOST_CHECK(is_valid_index(";i") == false);
  BOOST_CHECK(is_valid_index("i;,j") == false);
  BOOST_CHECK(is_valid_index("i,;j") == false);
}

BOOST_AUTO_TEST_SUITE_END()

/* An index is a ToT index if it is a valid index and it contains a semicolon.
 * We loop over the already created indices asserting that they are correctly
 * identified as ToT indices, or not. Then we assume that the is_valid_index
 * function works correctly. This means we do not have to test every possibly
 * invalid index, just that an invalid index with a semicolon is not a tot
 * index.
 */
BOOST_AUTO_TEST_SUITE(is_tot_index_fxn)

BOOST_AUTO_TEST_CASE(valid_but_not_tot) {
  for (auto x : {i_idx, i_j_idx})
    for (auto idx : x) BOOST_CHECK(is_tot_index(idx.first) == false);
}

BOOST_AUTO_TEST_CASE(valid_tot_idx) {
  for (auto x : {vov_idx, vom_idx, mov_idx})
    for (auto idx : x) BOOST_CHECK(is_tot_index(idx.first));
}

BOOST_AUTO_TEST_CASE(not_valid_idx) {
  BOOST_CHECK(is_tot_index("") == false);
  BOOST_CHECK(is_tot_index(";") == false);
}

BOOST_AUTO_TEST_SUITE_END()

/* Split index: removes the whitespace, splits the index into inner and outer
 * components, and then splits the modes of each component. We know that
 * remove_whitespace, is_tot_index, and tokenize_index work by this point. So
 * all we have to do is ensure they are hooked up correctly.
 */
BOOST_AUTO_TEST_SUITE(split_index_fxn)

BOOST_AUTO_TEST_CASE(invalid_idx) {
  if (TiledArray::get_default_world().nproc() == 1)
    BOOST_CHECK_THROW(split_index("i,"), TiledArray::Exception);
}

BOOST_AUTO_TEST_CASE(non_tot) {
  std::map<std::string, std::pair<string_vector_t, string_vector_t>> inputs{
      {"i", {string_vector_t{"i"}, string_vector_t{}}},
      {"i,j", {string_vector_t{"i", "j"}, string_vector_t{}}},
      {"i,j,k", {string_vector_t{"i", "j", "k"}, string_vector_t{}}}};

  for (auto& [idx, corr] : inputs) {
    BOOST_CHECK(split_index(idx) == corr);
  }
}

BOOST_AUTO_TEST_CASE(tot) {
  std::map<std::string, std::pair<string_vector_t, string_vector_t>> inputs{
      {"i;j", {string_vector_t{"i"}, string_vector_t{"j"}}},
      {"i,j;k", {string_vector_t{"i", "j"}, string_vector_t{"k"}}},
      {"i;j,k", {string_vector_t{"i"}, string_vector_t{"j", "k"}}}};

  for (auto& [idx, corr] : inputs) BOOST_CHECK(split_index(idx) == corr);
}

BOOST_AUTO_TEST_SUITE_END()
