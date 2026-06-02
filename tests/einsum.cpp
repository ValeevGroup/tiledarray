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

#include "unit_test_config.h"

#include "TiledArray/expressions/einsum.h"
#include "tiledarray.h"
#include "tot_array_fixture.h"

#include "TiledArray/expressions/contraction_helpers.h"

#include "TiledArray/tensor/arena_einsum.h"
#include "TiledArray/tensor/arena_kernels.h"
#include "TiledArray/tensor/arena_tensor.h"

BOOST_AUTO_TEST_SUITE(einsum_manual)

namespace {
using il_trange = std::initializer_list<std::initializer_list<size_t>>;
using il_extent = std::initializer_list<size_t>;
}  // namespace

template <DeNest DeNestFlag = DeNest::False,
          ShapeComp ShapeCompFlag = ShapeComp::True, typename ArrayA,
          typename ArrayB,
          typename = std::enable_if_t<TA::detail::is_array_v<ArrayA, ArrayB>>>
bool check_manual_eval(std::string const& annot, ArrayA A, ArrayB B) {
  auto out = TA::einsum<DeNestFlag>(annot, A, B);
  auto ref = manual_eval<DeNestFlag>(annot, A, B);

  using Policy = typename decltype(out)::policy_type;
  if constexpr (ShapeCompFlag == ShapeComp::True &&
                std::is_same_v<Policy, TA::SparsePolicy>) {
    out.truncate();
  }
  return ToTArrayFixture::are_equal<ShapeCompFlag>(ref, out);
}

template <ShapeComp ShapeCompFlag, DeNest DeNestFlag = DeNest::False,
          typename ArrayA, typename ArrayB,
          typename = std::enable_if_t<TA::detail::is_array_v<ArrayA, ArrayB>>>
bool check_manual_eval(std::string const& annot, ArrayA A, ArrayB B) {
  return check_manual_eval<DeNestFlag, ShapeCompFlag>(annot, A, B);
}

template <typename Array, DeNest DeNestFlag = DeNest::False,
          ShapeComp ShapeCompFlag = ShapeComp::True>
bool check_manual_eval(std::string const& annot, il_trange trangeA,
                       il_trange trangeB) {
  static_assert(detail::is_array_v<Array> &&
                detail::is_tensor_v<typename Array::value_type>);
  auto A = random_array<Array>(TA::TiledRange(trangeA));
  auto B = random_array<Array>(TA::TiledRange(trangeB));
  return check_manual_eval<DeNestFlag, ShapeCompFlag>(annot, A, B);
}

template <typename Array, ShapeComp ShapeCompFlag,
          DeNest DeNestFlag = DeNest::False>
bool check_manual_eval(std::string const& annot, il_trange trangeA,
                       il_trange trangeB) {
  return check_manual_eval<Array, DeNestFlag, ShapeCompFlag>(annot, trangeA,
                                                             trangeB);
}

template <typename ArrayA, typename ArrayB, DeNest DeNestFlag = DeNest::False,
          ShapeComp ShapeCompFlag = ShapeComp::True>
bool check_manual_eval(std::string const& annot, il_trange trangeA,
                       il_trange trangeB, il_extent inner_extents) {
  static_assert(detail::is_array_v<ArrayA, ArrayB>);

  if constexpr (detail::is_tensor_of_tensor_v<typename ArrayA::value_type>) {
    static_assert(!detail::is_tensor_of_tensor_v<typename ArrayB::value_type>);
    return check_manual_eval<DeNestFlag, ShapeCompFlag>(
        annot, random_array<ArrayA>(trangeA, inner_extents),
        random_array<ArrayB>(trangeB));
  } else {
    static_assert(detail::is_tensor_of_tensor_v<typename ArrayB::value_type>);
    return check_manual_eval<DeNestFlag, ShapeCompFlag>(
        annot, random_array<ArrayA>(trangeA),
        random_array<ArrayB>(trangeB, inner_extents));
  }
}

template <typename ArrayA, typename ArrayB, ShapeComp ShapeCompFlag,
          DeNest DeNestFlag = DeNest::False>
bool check_manual_eval(std::string const& annot, il_trange trangeA,
                       il_trange trangeB, il_extent inner_extents) {
  return check_manual_eval<DeNestFlag, ShapeCompFlag, ArrayA, ArrayB>(
      annot, trangeA, trangeB);
}

template <typename Array, DeNest DeNestFlag = DeNest::False,
          ShapeComp ShapeCompFlag = ShapeComp::True>
bool check_manual_eval(std::string const& annot, il_trange trangeA,
                       il_trange trangeB, il_extent inner_extentsA,
                       il_extent inner_extentsB) {
  static_assert(detail::is_array_v<Array> &&
                detail::is_tensor_of_tensor_v<typename Array::value_type>);
  return check_manual_eval<DeNestFlag, ShapeCompFlag>(
      annot, random_array<Array>(trangeA, inner_extentsA),
      random_array<Array>(trangeB, inner_extentsB));
}

template <typename Array, ShapeComp ShapeCompFlag,
          DeNest DeNestFlag = DeNest::False>
bool check_manual_eval(std::string const& annot, il_trange trangeA,
                       il_trange trangeB, il_extent inner_extentsA,
                       il_extent inner_extentsB) {
  return check_manual_eval<Array, DeNestFlag, ShapeCompFlag>(
      annot, trangeA, trangeB, inner_extentsA, inner_extentsB);
}

BOOST_AUTO_TEST_CASE(contract) {
  using Array = TA::DistArray<TA::Tensor<int>>;

  BOOST_REQUIRE(check_manual_eval<Array>("ij,j->i",
                                         {{0, 2, 4}, {0, 4, 8}},  // A's trange
                                         {{0, 4, 8}}              // B's trange
                                         ));
  BOOST_REQUIRE(check_manual_eval<Array>("ik,jk->ji",
                                         {{0, 2, 4}, {0, 4, 8}},  // A's trange
                                         {{0, 3}, {0, 4, 8}}      // B's trange
                                         ));

  BOOST_REQUIRE(check_manual_eval<Array>(
      "ijkl,jm->lkmi",                      //
      {{0, 2}, {0, 4, 8}, {0, 3}, {0, 7}},  //
      {{0, 4, 8}, {0, 5}}                   //
      ));
}

BOOST_AUTO_TEST_CASE(hadamard) {
  using Array = TA::DistArray<TA::Tensor<int>>;
  BOOST_REQUIRE(check_manual_eval<Array>("i,i->i",  //
                                         {{0, 1}},  //
                                         {{0, 1}}   //
                                         ));
  BOOST_REQUIRE(check_manual_eval<Array>("i,i->i",     //
                                         {{0, 2, 4}},  //
                                         {{0, 2, 4}}   //
                                         ));

  BOOST_REQUIRE(check_manual_eval<Array>("ijk,kij->ikj",                  //
                                         {{0, 2, 4}, {0, 2, 3}, {0, 5}},  //
                                         {{0, 5}, {0, 2, 4}, {0, 2, 3}}   //
                                         ));
}

BOOST_AUTO_TEST_CASE(general) {
  using Array = TA::DistArray<TA::Tensor<int>>;
  BOOST_REQUIRE(check_manual_eval<Array>("ijk,kil->ijl",                  //
                                         {{0, 2}, {0, 3, 5}, {0, 2, 4}},  //
                                         {{0, 2, 4}, {0, 2}, {0, 1}}      //
                                         ));
  using Tensor = typename Array::value_type;
  using namespace std::string_literals;

  Tensor A(TA::Range{2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor B(TA::Range{2}, {2, 10});
  Tensor C(TA::Range{2, 3}, {2, 4, 6, 40, 50, 60});
  BOOST_REQUIRE(
      C == general_product<Tensor>(A, B, ProductSetup("ij"s, "i"s, "ij"s)));
}

BOOST_AUTO_TEST_CASE(equal_nested_ranks) {
  using ArrayToT = TA::DistArray<TA::Tensor<TA::Tensor<int>>>;

  // H;H (Hadamard outer; Hadamard inner)
  BOOST_REQUIRE(check_manual_eval<ArrayToT>("ij;mn,ji;nm->ij;mn",  //
                                            {{0, 2, 4}, {0, 3}},   //
                                            {{0, 3}, {0, 2, 4}},   //
                                            {5, 7},                //
                                            {7, 5}                 //
                                            ));

  // H;C (Hadamard outer; contraction inner)
  BOOST_REQUIRE(check_manual_eval<ArrayToT>("ij;mo,ji;on->ij;mn",  //
                                            {{0, 2, 4}, {0, 3}},   //
                                            {{0, 3}, {0, 2, 4}},   //
                                            {3, 7},                //
                                            {7, 4}                 //
                                            ));

  // H;C
  BOOST_REQUIRE(check_manual_eval<ArrayToT>("ij;mo,ji;o->ij;m",   //
                                            {{0, 2, 4}, {0, 3}},  //
                                            {{0, 3}, {0, 2, 4}},  //
                                            {3, 7},               //
                                            {7}                   //
                                            ));

  // C;C
  BOOST_REQUIRE(check_manual_eval<ArrayToT>("ik;mo,kj;on->ij;mn",    //
                                            {{0, 3, 5}, {0, 2, 4}},  //
                                            {{0, 2, 4}, {0, 2}},     //
                                            {2, 2},                  //
                                            {2, 2}));

  // C;C
  BOOST_REQUIRE(check_manual_eval<ArrayToT>("ijk;dcb,ik;bc->ij;d",     //
                                            {{0, 3}, {0, 4}, {0, 5}},  //
                                            {{0, 3}, {0, 5}},          //
                                            {2, 3, 4},                 //
                                            {4, 3}));

  // H+C;H
  BOOST_REQUIRE(
      check_manual_eval<ArrayToT>("jki;mn,ijk;nm->ij;mn",             //
                                  {{0, 2, 3}, {0, 3, 5}, {0, 1, 3}},  //
                                  {{0, 1, 3}, {0, 2, 3}, {0, 3, 5}},  //
                                  {3, 2},                             //
                                  {2, 3}));

  // H+C;C
  BOOST_REQUIRE(
      check_manual_eval<ArrayToT>("ijk;mo,kji;no->ij;nm",             //
                                  {{0, 1, 3}, {0, 2, 3}, {0, 3, 5}},  //
                                  {{0, 3, 5}, {0, 2, 3}, {0, 1, 3}},  //
                                  {3, 2},                             //
                                  {4, 2}));

  // H+C;C
  BOOST_REQUIRE(check_manual_eval<ArrayToT>("ijk;m,ijk;n->ij;nm",      //
                                            {{0, 2}, {0, 3}, {0, 2}},  //
                                            {{0, 2}, {0, 3}, {0, 2}},  //
                                            {3},                       //
                                            {2}));

  // H+C;C with permuted inner operands -- no outer permutation, so this
  // exercises the regime-A arena path; manual_eval is the independent oracle.
  BOOST_REQUIRE(check_manual_eval<ArrayToT>("ijk;om,ijk;on->ij;nm",    //
                                            {{0, 2}, {0, 3}, {0, 2}},  //
                                            {{0, 2}, {0, 3}, {0, 2}},  //
                                            {3, 2},                    //
                                            {3, 2}));

  // H+C;H with a permuted inner Hadamard operand -- no outer permutation, so
  // this exercises the regime-A arena Hadamard path against manual_eval.
  BOOST_REQUIRE(check_manual_eval<ArrayToT>("ijk;mn,ijk;nm->ij;mn",    //
                                            {{0, 2}, {0, 3}, {0, 2}},  //
                                            {{0, 2}, {0, 3}, {0, 2}},  //
                                            {4, 3},                    //
                                            {3, 4}));
  // H+C;H+C not supported

  // H;C(op)
  BOOST_REQUIRE(check_manual_eval<ArrayToT>(
      "ijk;bc,j;d->kji;dcb", {{0, 1}, {0, 1}, {0, 1}}, {{0, 1}}, {2, 3}, {4}));

  // H+C;C
  BOOST_REQUIRE(check_manual_eval<ArrayToT>(
      "jki;ad,ikj;db->ij;ab",                                  //
      {{0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}},  //
      {{0, 1, 2, 3, 4}, {0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6}},  //
      {3, 2},                                                  //
      {2, 4}));

  // H+C;C
  BOOST_REQUIRE(
      check_manual_eval<ArrayToT>("ijk;mo,kji;no->ik;nm",             //
                                  {{0, 3, 6}, {0, 1, 3}, {0, 2, 4}},  //
                                  {{0, 2, 4}, {0, 1, 3}, {0, 3, 6}},  //
                                  {3, 2},                             //
                                  {4, 2}));

  // H+C;C
  BOOST_REQUIRE(
      check_manual_eval<ArrayToT>("ijk;mo,ijk;no->ji;nm",             //
                                  {{0, 2, 5}, {0, 1, 3}, {0, 3, 4}},  //
                                  {{0, 2, 5}, {0, 1, 3}, {0, 3, 4}},  //
                                  {4, 2},                             //
                                  {3, 2}));
}

BOOST_AUTO_TEST_CASE(different_nested_ranks) {
  using ArrayT = TA::DistArray<TA::Tensor<int>>;
  using ArrayToT = TA::DistArray<TA::Tensor<TA::Tensor<int>>>;

  {
    // these tests do not involve permutation of inner tensors
    // H
    BOOST_REQUIRE(
        (check_manual_eval<ArrayToT, ArrayT>("ij;mn,ji->ji;mn",          //
                                             {{0, 2, 5}, {0, 3, 5, 9}},  //
                                             {{0, 3, 5, 9}, {0, 2, 5}},  //
                                             {2, 1})));

    // H (reversed arguments)
    BOOST_REQUIRE(
        (check_manual_eval<ArrayT, ArrayToT>("ji,ij;mn->ji;mn",          //
                                             {{0, 3, 5, 9}, {0, 2, 5}},  //
                                             {{0, 2, 5}, {0, 3, 5, 9}},  //
                                             {2, 4})));

    // C (outer product)
    BOOST_REQUIRE((check_manual_eval<ArrayToT, ArrayT>("i;mn,j->ij;mn",  //
                                                       {{0, 5}},         //
                                                       {{0, 3, 8}},      //
                                                       {3, 2})));

    // C (outer product) (reversed arguments)
    BOOST_REQUIRE((check_manual_eval<ArrayT, ArrayToT>("j,i;mn->ij;mn",  //
                                                       {{0, 3, 8}},      //
                                                       {{0, 5}},         //
                                                       {2, 2})));
  }

  // C (outer product)
  BOOST_REQUIRE((check_manual_eval<ArrayToT, ArrayT>("ik;mn,j->ijk;nm",    //
                                                     {{0, 2, 4}, {0, 4}},  //
                                                     {{0, 3, 5}},          //
                                                     {3, 2})));

  // C (outer product) (reversed arguments)
  BOOST_REQUIRE((check_manual_eval<ArrayT, ArrayToT>("jl,ik;mn->ijkl;nm",  //
                                                     {{0, 3, 5}, {0, 3}},  //
                                                     {{0, 2, 4}, {0, 4}},  //
                                                     {3, 2})));

  // H+C (outer product)
  BOOST_REQUIRE((check_manual_eval<ArrayToT, ArrayT>("ij;mn,ik->ijk;nm",      //
                                                     {{0, 2, 5}, {0, 3, 7}},  //
                                                     {{0, 2, 5}, {0, 4, 7}},  //
                                                     {2, 5})));

  // H+C (outer product) (reversed arguments)
  BOOST_REQUIRE((check_manual_eval<ArrayT, ArrayToT>("ik,ij;mn->ijk;nm",      //
                                                     {{0, 2, 5}, {0, 4, 7}},  //
                                                     {{0, 2, 5}, {0, 3, 7}},  //
                                                     {2, 5})));

  {
    // these tests do not involve permutation of inner tensors
    // H+C
    BOOST_REQUIRE(
        (check_manual_eval<ArrayToT, ArrayT>("ik;mn,ijk->ij;mn",        //
                                             {{0, 2}, {0, 3}},          //
                                             {{0, 2}, {0, 2}, {0, 3}},  //
                                             {2, 2})));

    // H+C (reversed arguments)
    BOOST_REQUIRE(
        (check_manual_eval<ArrayT, ArrayToT>("ijk,ik;mn->ij;mn",        //
                                             {{0, 2}, {0, 2}, {0, 3}},  //
                                             {{0, 2}, {0, 3}},          //
                                             {2, 2})));
  }

  // H
  BOOST_REQUIRE((check_manual_eval<ArrayToT, ArrayT>("ij;mn,ji->ji;nm",       //
                                                     {{0, 2, 4, 6}, {0, 3}},  //
                                                     {{0, 3}, {0, 2, 4, 6}},  //
                                                     {4, 2})));

  // H (reversed arguments)
  BOOST_REQUIRE((check_manual_eval<ArrayT, ArrayToT>("ji,ij;mn->ji;nm",       //
                                                     {{0, 3, 5}, {0, 2, 4}},  //
                                                     {{0, 2, 4}, {0, 3, 5}},  //
                                                     {1, 2})));

  // C
  BOOST_REQUIRE((check_manual_eval<ArrayToT, ArrayT>("ij;m,j->i;m",        //
                                                     {{0, 5}, {0, 2, 3}},  //
                                                     {{0, 2, 3}},          //
                                                     {3})));

  // C (reversed arguments)
  BOOST_REQUIRE((check_manual_eval<ArrayT, ArrayToT>("j,ij;m->i;m",     //
                                                     {{0, 2}},          //
                                                     {{0, 1}, {0, 2}},  //
                                                     {3})));

  // H+C
  BOOST_REQUIRE((
      check_manual_eval<ArrayToT, ArrayT>("ik;mn,ijk->ij;nm",                 //
                                          {{0, 2}, {0, 3, 5}},                //
                                          {{0, 2}, {0, 2, 4, 6}, {0, 3, 5}},  //
                                          {2, 2})));

  // H+C (reversed arguments)
  BOOST_REQUIRE(
      (check_manual_eval<ArrayT, ArrayToT>("ijk,ik;mn->ij;nm",        //
                                           {{0, 2}, {0, 4}, {0, 3}},  //
                                           {{0, 2}, {0, 3}},          //
                                           {2, 4})));
}

BOOST_AUTO_TEST_CASE(nested_rank_reduction) {
  using T = TA::Tensor<int>;
  using ToT = TA::Tensor<T>;
  using Array = TA::DistArray<T>;
  using ArrayToT = TA::DistArray<ToT>;
  BOOST_REQUIRE(
      (check_manual_eval<ArrayToT, DeNest::True>("ij;ab,ij;ab->ij",    //
                                                 {{0, 2, 4}, {0, 4}},  //
                                                 {{0, 2, 4}, {0, 4}},  //
                                                 {3, 2},               //
                                                 {3, 2})));
  BOOST_REQUIRE(
      (check_manual_eval<ArrayToT, DeNest::True>("ij;ab,ij;ab->i",     //
                                                 {{0, 2, 4}, {0, 4}},  //
                                                 {{0, 2, 4}, {0, 4}},  //
                                                 {3, 2},               //
                                                 {3, 2})));
  // external indices present (branch-2 generalized contraction): Hadamard i,
  // external p (from A) and q (from B), contracted-outer k, inner ab fully
  // contracted. This is the shape that motivated the rewrite (cf. the CSV-CC
  // term g*C*t*...). Exercises the ContEngine inner phantom-dot path.
  BOOST_REQUIRE((check_manual_eval<ArrayToT, DeNest::True>(
      "ipk;ab,iqk;ab->ipq",            //
      {{0, 2}, {0, 2, 3}, {0, 2, 4}},  // A outer: i, p, k
      {{0, 2}, {0, 3}, {0, 2, 4}},     // B outer: i, q, k
      {3, 2},                          // A inner: ab
      {3, 2})));                       // B inner: ab
}

BOOST_AUTO_TEST_CASE(corner_cases) {
  using T = TA::Tensor<int>;
  using ToT = TA::Tensor<T>;
  using ArrayT = TA::DistArray<T>;
  using ArrayToT = TA::DistArray<ToT>;

  BOOST_REQUIRE(check_manual_eval<ArrayT>("ia,i->ia",                   //
                                          {{0, 2, 5}, {0, 7, 11, 16}},  //
                                          {{0, 2, 5}}));

  BOOST_REQUIRE(check_manual_eval<ArrayT>("i,ai->ia",   //
                                          {{0, 2, 5}},  //
                                          {{0, 7, 11, 16}, {0, 2, 5}}));

  BOOST_REQUIRE(check_manual_eval<ArrayT>("ijk,kj->kij",                      //
                                          {{0, 2, 5}, {0, 3, 6}, {0, 2, 7}},  //
                                          {{0, 2, 7}, {0, 3, 6}}));

  BOOST_REQUIRE(check_manual_eval<ArrayT>("kj,ijk->kij",           //
                                          {{0, 2, 7}, {0, 3, 6}},  //
                                          {{0, 2, 5}, {0, 3, 6}, {0, 2, 7}}));

  BOOST_REQUIRE(check_manual_eval<ArrayToT>("kij;ab,kj;bc->kji;ac",          //
                                            {{0, 2}, {0, 3, 5}, {0, 4, 7}},  //
                                            {{0, 2}, {0, 4, 7}},             //
                                            {3, 5}, {5, 2}));

  BOOST_REQUIRE(
      (check_manual_eval<ArrayToT, ArrayT>("ijk;ab,kj->kij;ba",             //
                                           {{0, 2}, {0, 4, 6}, {0, 3, 5}},  //
                                           {{0, 3, 5}, {0, 4, 6}},          //
                                           {7, 5})));

  BOOST_REQUIRE(
      (check_manual_eval<ArrayT, ArrayToT>("ij,jik;ab->kji;ab",             //
                                           {{0, 3, 5}, {0, 3, 8}},          //
                                           {{0, 3, 8}, {0, 3, 5}, {0, 2}},  //
                                           {3, 9})));

  BOOST_REQUIRE(check_manual_eval<ArrayT>("bi,bi->i",        //
                                          {{0, 2}, {0, 4}},  //
                                          {{0, 2}, {0, 4}}));

  BOOST_REQUIRE(check_manual_eval<ArrayToT>("bi;a,bi;a->i;a",  //
                                            {{0, 2}, {0, 4}},  //
                                            {{0, 2}, {0, 4}},  //
                                            {3}, {3}));

  BOOST_REQUIRE(
      (check_manual_eval<ArrayToT, ArrayT>("jk;a,ijk->i;a",           //
                                           {{0, 2}, {0, 4}},          //
                                           {{0, 3}, {0, 2}, {0, 4}},  //
                                           {5})));

  BOOST_REQUIRE((check_manual_eval<ArrayToT, ArrayT>("bi;a,bi->i;a",       //
                                                     {{0, 4, 8}, {0, 4}},  //
                                                     {{0, 4, 8}, {0, 4}},  //
                                                     {8})));

  BOOST_REQUIRE(check_manual_eval<ArrayToT>("il;bae,il;e->li;ab",  //
                                            {{0, 2}, {0, 4}},      //
                                            {{0, 2}, {0, 4}},      //
                                            {4, 2, 3},             //
                                            {3}));

  BOOST_REQUIRE(
      check_manual_eval<ArrayToT>("ijkl;abecdf,k;e->ijl;bafdc",      //
                                  {{0, 2}, {0, 3}, {0, 4}, {0, 5}},  //
                                  {{0, 4}},                          //
                                  {2, 3, 6, 4, 5, 7},                //
                                  {6}));
}

BOOST_AUTO_TEST_SUITE_END()

using namespace TiledArray;
using namespace TiledArray::expressions;

BOOST_AUTO_TEST_SUITE(einsum_tot)

BOOST_AUTO_TEST_CASE(ik_mn_eq_ij_mn_times_jk_mn) {
  using dist_array_t = DistArray<Tensor<Tensor<double>>, DensePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {49, 73, 28, 46, 12, 83, 29, 61, 61, 98, 57, 28, 96, 57});
  Tensor<double> lhs_elem_0_1(
      Range{7, 2}, {78, 15, 69, 55, 87, 94, 28, 94, 79, 30, 26, 88, 48, 74});
  Tensor<double> lhs_elem_1_0(
      Range{7, 2}, {70, 32, 25, 71, 6, 56, 4, 13, 72, 50, 15, 95, 52, 89});
  Tensor<double> lhs_elem_1_1(
      Range{7, 2}, {12, 29, 17, 68, 37, 79, 5, 52, 13, 35, 53, 54, 78, 71});
  Tensor<double> lhs_elem_2_0(
      Range{7, 2}, {77, 39, 34, 94, 16, 82, 63, 27, 75, 12, 14, 59, 3, 14});
  Tensor<double> lhs_elem_2_1(
      Range{7, 2}, {65, 90, 37, 41, 65, 75, 59, 16, 44, 85, 86, 11, 40, 24});
  Tensor<double> lhs_elem_3_0(
      Range{7, 2}, {77, 53, 11, 6, 99, 63, 46, 68, 83, 56, 76, 86, 91, 79});
  Tensor<double> lhs_elem_3_1(
      Range{7, 2}, {56, 11, 33, 90, 36, 38, 33, 54, 60, 21, 16, 28, 6, 97});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1},
                   {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1},
                   {lhs_elem_3_0, lhs_elem_3_1}};
  TiledRange lhs_trange{{0, 2, 4}, {0, 2}};
  dist_array_t lhs(world, lhs_trange, lhs_il);
  Tensor<double> rhs_elem_0_0(
      Range{7, 2}, {53, 10, 70, 2, 14, 82, 81, 27, 22, 76, 51, 68, 77, 17});
  Tensor<double> rhs_elem_0_1(
      Range{7, 2}, {5, 99, 15, 7, 98, 85, 33, 92, 32, 91, 21, 19, 40, 54});
  Tensor<double> rhs_elem_0_2(
      Range{7, 2}, {4, 53, 38, 39, 8, 90, 43, 77, 57, 42, 92, 88, 76, 1});
  Tensor<double> rhs_elem_0_3(
      Range{7, 2}, {44, 86, 81, 43, 9, 40, 13, 6, 24, 12, 10, 60, 62, 59});
  Tensor<double> rhs_elem_0_4(
      Range{7, 2}, {35, 87, 52, 20, 55, 84, 66, 36, 41, 44, 3, 93, 91, 12});
  Tensor<double> rhs_elem_0_5(
      Range{7, 2}, {21, 23, 37, 39, 3, 25, 93, 61, 86, 11, 45, 70, 47, 30});
  Tensor<double> rhs_elem_1_0(
      Range{7, 2}, {63, 19, 60, 77, 10, 60, 68, 52, 5, 31, 44, 6, 90, 37});
  Tensor<double> rhs_elem_1_1(
      Range{7, 2}, {79, 73, 42, 83, 71, 83, 73, 32, 66, 87, 14, 76, 14, 72});
  Tensor<double> rhs_elem_1_2(
      Range{7, 2}, {80, 19, 59, 90, 73, 84, 22, 16, 1, 95, 90, 7, 97, 26});
  Tensor<double> rhs_elem_1_3(
      Range{7, 2}, {32, 55, 64, 81, 63, 97, 76, 86, 69, 61, 76, 87, 19, 72});
  Tensor<double> rhs_elem_1_4(
      Range{7, 2}, {63, 84, 5, 64, 43, 30, 78, 8, 65, 25, 99, 93, 23, 22});
  Tensor<double> rhs_elem_1_5(
      Range{7, 2}, {6, 74, 86, 41, 43, 63, 5, 83, 78, 54, 5, 1, 43, 97});
  matrix_il rhs_il{{rhs_elem_0_0, rhs_elem_0_1, rhs_elem_0_2, rhs_elem_0_3,
                    rhs_elem_0_4, rhs_elem_0_5},
                   {rhs_elem_1_0, rhs_elem_1_1, rhs_elem_1_2, rhs_elem_1_3,
                    rhs_elem_1_4, rhs_elem_1_5}};
  TiledRange rhs_trange{{0, 2}, {0, 2, 4, 6}};
  dist_array_t rhs(world, rhs_trange, rhs_il);
  Tensor<double> corr_elem_0_0(
      Range{7, 2}, {7511, 1015, 6100, 4327, 1038, 12446, 4253, 6535, 1737, 8378,
                    4051, 2432, 11712, 3707});
  Tensor<double> corr_elem_0_1(
      Range{7, 2}, {6407, 8322, 3318, 4887, 7353, 14857, 3001, 8620, 7166,
                    11528, 1561, 7220, 4512, 8406});
  Tensor<double> corr_elem_0_2(
      Range{7, 2}, {6436, 4154, 5135, 6744, 6447, 15366, 1863, 6201, 3556, 6966,
                    7584, 3080, 11952, 1981});
  Tensor<double> corr_elem_0_3(
      Range{7, 2}, {4652, 7103, 6684, 6433, 5589, 12438, 2505, 8450, 6915, 3006,
                    2546, 9336, 6864, 8691});
  Tensor<double> corr_elem_0_4(
      Range{7, 2}, {6629, 7611, 1801, 4440, 4401, 9792, 4098, 2948, 7636, 5062,
                    2745, 10788, 9840, 2312});
  Tensor<double> corr_elem_0_5(
      Range{7, 2}, {1497, 2789, 6970, 4049, 3777, 7997, 2837, 11523, 11408,
                    2698, 2695, 2048, 6576, 8888});
  Tensor<double> corr_elem_1_0(
      Range{7, 2}, {4466, 871, 2770, 5378, 454, 9332, 664, 3055, 1649, 4885,
                    3097, 6784, 11024, 4140});
  Tensor<double> corr_elem_1_1(
      Range{7, 2}, {1298, 5285, 1089, 6141, 3215, 11317, 497, 2860, 3162, 7595,
                    1057, 5909, 3172, 9918});
  Tensor<double> corr_elem_1_2(
      Range{7, 2}, {1240, 2247, 1953, 8889, 2749, 11676, 282, 1833, 4117, 5425,
                    6150, 8738, 11518, 1935});
  Tensor<double> corr_elem_1_3(
      Range{7, 2}, {3464, 4347, 3113, 8561, 2385, 9903, 432, 4550, 2625, 2735,
                    4178, 10398, 4706, 10363});
  Tensor<double> corr_elem_1_4(
      Range{7, 2}, {3206, 5220, 1385, 5772, 1921, 7074, 654, 884, 3797, 3075,
                    5292, 13857, 6526, 2630});
  Tensor<double> corr_elem_1_5(
      Range{7, 2}, {1542, 2882, 2387, 5557, 1609, 6377, 397, 5109, 7206, 2440,
                    940, 6704, 5798, 9557});
  Tensor<double> corr_elem_2_0(
      Range{7, 2}, {8176, 2100, 4600, 3345, 874, 11224, 9115, 1561, 1870, 3547,
                    4498, 4078, 3831, 1126});
  Tensor<double> corr_elem_2_1(
      Range{7, 2}, {5520, 10431, 2064, 4061, 6183, 13195, 6386, 2996, 5304,
                    8487, 1498, 1957, 680, 2484});
  Tensor<double> corr_elem_2_2(
      Range{7, 2}, {5508, 3777, 3475, 7356, 4873, 13680, 4007, 2335, 4319, 8579,
                    9028, 5269, 4108, 638});
  Tensor<double> corr_elem_2_3(
      Range{7, 2}, {5468, 8304, 5122, 7363, 4239, 10555, 5303, 1538, 4836, 5329,
                    6676, 4497, 946, 2554});
  Tensor<double> corr_elem_2_4(
      Range{7, 2}, {6790, 10953, 1953, 4504, 3675, 9138, 8760, 1100, 5935, 2653,
                    8556, 6510, 1193, 696});
  Tensor<double> corr_elem_2_5(
      Range{7, 2}, {2007, 7557, 4440, 5347, 2843, 6775, 6154, 2975, 9882, 4722,
                    1060, 4141, 1861, 2748});
  Tensor<double> corr_elem_3_0(
      Range{7, 2}, {7609, 739, 2750, 6942, 1746, 7446, 5970, 4644, 2126, 4907,
                    4580, 6016, 7547, 4932});
  Tensor<double> corr_elem_3_1(
      Range{7, 2}, {4809, 6050, 1551, 7512, 12258, 8509, 3927, 7984, 6616, 6923,
                    1820, 3762, 3724, 11250});
  Tensor<double> corr_elem_3_2(
      Range{7, 2}, {4788, 3018, 2365, 8334, 3420, 8862, 2704, 6100, 4791, 4347,
                    8432, 7764, 7498, 2601});
  Tensor<double> corr_elem_3_3(
      Range{7, 2}, {5180, 5163, 3003, 7548, 3159, 6206, 3106, 5052, 6132, 1953,
                    1976, 7596, 5756, 11645});
  Tensor<double> corr_elem_3_4(
      Range{7, 2}, {6223, 5535, 737, 5880, 6993, 6432, 5610, 2880, 7303, 2989,
                    1812, 10602, 8419, 3082});
  Tensor<double> corr_elem_3_5(
      Range{7, 2}, {1953, 2033, 3245, 3924, 1845, 3969, 4443, 8630, 11818, 1750,
                    3500, 6048, 4535, 11779});
  matrix_il corr_il{{corr_elem_0_0, corr_elem_0_1, corr_elem_0_2, corr_elem_0_3,
                     corr_elem_0_4, corr_elem_0_5},
                    {corr_elem_1_0, corr_elem_1_1, corr_elem_1_2, corr_elem_1_3,
                     corr_elem_1_4, corr_elem_1_5},
                    {corr_elem_2_0, corr_elem_2_1, corr_elem_2_2, corr_elem_2_3,
                     corr_elem_2_4, corr_elem_2_5},
                    {corr_elem_3_0, corr_elem_3_1, corr_elem_3_2, corr_elem_3_3,
                     corr_elem_3_4, corr_elem_3_5}};
  TiledRange corr_trange{{0, 2, 4}, {0, 2, 4, 6}};
  dist_array_t corr(world, corr_trange, corr_il);
  dist_array_t out = einsum(lhs("i,j;m,n"), rhs("j,k;m,n"), "i,k;m,n");
  const bool are_equal = ToTArrayFixture::are_equal(corr, out);
  BOOST_CHECK(are_equal);
}

BOOST_AUTO_TEST_CASE(ik_mn_eq_ij_mn_times_kj_mn) {
  using dist_array_t = DistArray<Tensor<Tensor<double>>, DensePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {25, 18, 67, 84, 22, 81, 6, 31, 50, 17, 61, 2, 63, 58});
  Tensor<double> lhs_elem_0_1(
      Range{7, 2}, {3, 11, 71, 19, 43, 94, 20, 93, 52, 62, 38, 59, 42, 88});
  Tensor<double> lhs_elem_1_0(
      Range{7, 2}, {43, 50, 30, 27, 97, 41, 12, 21, 57, 8, 37, 81, 26, 94});
  Tensor<double> lhs_elem_1_1(
      Range{7, 2}, {1, 84, 87, 69, 68, 26, 86, 99, 16, 88, 38, 55, 91, 56});
  Tensor<double> lhs_elem_2_0(
      Range{7, 2}, {75, 9, 24, 79, 68, 7, 33, 16, 47, 11, 3, 26, 61, 48});
  Tensor<double> lhs_elem_2_1(
      Range{7, 2}, {71, 41, 63, 70, 75, 87, 76, 31, 20, 67, 45, 84, 92, 17});
  Tensor<double> lhs_elem_3_0(
      Range{7, 2}, {7, 69, 99, 84, 66, 28, 32, 28, 30, 74, 35, 78, 89, 73});
  Tensor<double> lhs_elem_3_1(
      Range{7, 2}, {21, 91, 52, 94, 67, 63, 17, 20, 48, 88, 100, 53, 35, 29});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1},
                   {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1},
                   {lhs_elem_3_0, lhs_elem_3_1}};
  TiledRange lhs_trange{{0, 2, 4}, {0, 2}};
  dist_array_t lhs(world, lhs_trange, lhs_il);
  Tensor<double> rhs_elem_0_0(
      Range{7, 2}, {12, 52, 47, 53, 66, 87, 11, 48, 24, 83, 16, 38, 26, 39});
  Tensor<double> rhs_elem_0_1(
      Range{7, 2}, {96, 38, 53, 99, 32, 16, 11, 69, 68, 17, 38, 53, 79, 83});
  Tensor<double> rhs_elem_1_0(
      Range{7, 2}, {26, 27, 3, 57, 26, 24, 35, 11, 96, 26, 83, 94, 19, 90});
  Tensor<double> rhs_elem_1_1(
      Range{7, 2}, {83, 98, 74, 21, 27, 14, 90, 75, 98, 75, 84, 5, 42, 88});
  Tensor<double> rhs_elem_2_0(
      Range{7, 2}, {72, 77, 10, 51, 62, 81, 58, 50, 56, 43, 68, 64, 54, 82});
  Tensor<double> rhs_elem_2_1(
      Range{7, 2}, {81, 20, 46, 1, 88, 39, 45, 79, 85, 48, 72, 1, 79, 89});
  Tensor<double> rhs_elem_3_0(
      Range{7, 2}, {84, 52, 99, 71, 30, 54, 17, 70, 18, 30, 77, 16, 65, 69});
  Tensor<double> rhs_elem_3_1(
      Range{7, 2}, {49, 80, 72, 71, 18, 51, 48, 22, 72, 43, 95, 22, 41, 88});
  Tensor<double> rhs_elem_4_0(
      Range{7, 2}, {70, 14, 59, 72, 53, 91, 31, 90, 56, 58, 28, 7, 87, 82});
  Tensor<double> rhs_elem_4_1(
      Range{7, 2}, {51, 95, 69, 13, 53, 75, 30, 71, 14, 10, 17, 44, 29, 77});
  Tensor<double> rhs_elem_5_0(
      Range{7, 2}, {95, 8, 90, 64, 96, 71, 41, 3, 100, 32, 14, 42, 77, 23});
  Tensor<double> rhs_elem_5_1(
      Range{7, 2}, {75, 96, 3, 89, 8, 11, 83, 18, 19, 24, 8, 69, 72, 38});
  matrix_il rhs_il{{rhs_elem_0_0, rhs_elem_0_1}, {rhs_elem_1_0, rhs_elem_1_1},
                   {rhs_elem_2_0, rhs_elem_2_1}, {rhs_elem_3_0, rhs_elem_3_1},
                   {rhs_elem_4_0, rhs_elem_4_1}, {rhs_elem_5_0, rhs_elem_5_1}};
  TiledRange rhs_trange{{0, 2, 4, 6}, {0, 2}};
  dist_array_t rhs(world, rhs_trange, rhs_il);
  Tensor<double> corr_elem_0_0(
      Range{7, 2}, {588, 1354, 6912, 6333, 2828, 8551, 286, 7905, 4736, 2465,
                    2420, 3203, 4956, 9566});
  Tensor<double> corr_elem_0_1(
      Range{7, 2}, {899, 1564, 5455, 5187, 1733, 3260, 2010, 7316, 9896, 5092,
                    8255, 483, 2961, 12964});
  Tensor<double> corr_elem_0_2(
      Range{7, 2}, {2043, 1606, 3936, 4303, 5148, 10227, 1248, 8897, 7220, 3707,
                    6884, 187, 6720, 12588});
  Tensor<double> corr_elem_0_3(
      Range{7, 2}, {2247, 1816, 11745, 7313, 1434, 9168, 1062, 4216, 4644, 3176,
                    8307, 1330, 5817, 11746});
  Tensor<double> corr_elem_0_4(
      Range{7, 2}, {1903, 1297, 8852, 6295, 3445, 14421, 786, 9393, 3528, 1606,
                    2354, 2610, 6699, 11532});
  Tensor<double> corr_elem_0_5(
      Range{7, 2}, {2600, 1200, 6243, 7067, 2456, 6785, 1906, 1767, 5988, 2032,
                    1158, 4155, 7875, 4678});
  Tensor<double> corr_elem_1_0(
      Range{7, 2}, {612, 5792, 6021, 8262, 8578, 3983, 1078, 7839, 2456, 2160,
                    2036, 5993, 7865, 8314});
  Tensor<double> corr_elem_1_1(
      Range{7, 2}, {1201, 9582, 6528, 2988, 4358, 1348, 8160, 7656, 7040, 6808,
                    6263, 7889, 4316, 13388});
  Tensor<double> corr_elem_1_2(
      Range{7, 2}, {3177, 5530, 4302, 1446, 11998, 4335, 4566, 8871, 4552, 4568,
                    5252, 5239, 8593, 12692});
  Tensor<double> corr_elem_1_3(
      Range{7, 2}, {3661, 9320, 9234, 6816, 4134, 3540, 4332, 3648, 2178, 4024,
                    6459, 2506, 5421, 11414});
  Tensor<double> corr_elem_1_4(
      Range{7, 2}, {3061, 8680, 7773, 2841, 8745, 5681, 2952, 8919, 3416, 1344,
                    1682, 2987, 4901, 12020});
  Tensor<double> corr_elem_1_5(
      Range{7, 2}, {4160, 8464, 2961, 7869, 9856, 3197, 7630, 1845, 6004, 2368,
                    822, 7197, 8554, 4290});
  Tensor<double> corr_elem_2_0(
      Range{7, 2}, {7716, 2026, 4467, 11117, 6888, 2001, 1199, 2907, 2488, 2052,
                    1758, 5440, 8854, 3283});
  Tensor<double> corr_elem_2_1(
      Range{7, 2}, {7843, 4261, 4734, 5973, 3793, 1386, 7995, 2501, 6472, 5311,
                    4029, 2864, 5023, 5816});
  Tensor<double> corr_elem_2_2(
      Range{7, 2}, {11151, 1513, 3138, 4099, 10816, 3960, 5334, 3249, 4332,
                    3689, 3444, 1748, 10562, 5449});
  Tensor<double> corr_elem_2_3(
      Range{7, 2}, {9779, 3748, 6912, 10579, 3390, 4815, 4209, 1802, 2286, 3211,
                    4506, 2264, 7737, 4808});
  Tensor<double> corr_elem_2_4(
      Range{7, 2}, {8871, 4021, 5763, 6598, 7579, 7162, 3303, 3641, 2912, 1308,
                    849, 3878, 7975, 5245});
  Tensor<double> corr_elem_2_5(
      Range{7, 2}, {12450, 4008, 2349, 11286, 7128, 1454, 7661, 606, 5080, 1960,
                    402, 6888, 11321, 1750});
  Tensor<double> corr_elem_3_0(
      Range{7, 2}, {2100, 7046, 7409, 13758, 6500, 3444, 539, 2724, 3984, 7638,
                    4360, 5773, 5079, 5254});
  Tensor<double> corr_elem_3_1(
      Range{7, 2}, {1925, 10781, 4145, 6762, 3525, 1554, 2650, 1808, 7584, 8524,
                    11305, 7597, 3161, 9122});
  Tensor<double> corr_elem_3_2(
      Range{7, 2}, {2205, 7133, 3382, 4378, 9988, 4725, 2621, 2980, 5760, 7406,
                    9580, 5045, 7571, 8567});
  Tensor<double> corr_elem_3_3(
      Range{7, 2}, {1617, 10868, 13545, 12638, 3186, 4725, 1360, 2400, 3996,
                    6004, 12195, 2414, 7220, 7589});
  Tensor<double> corr_elem_3_4(
      Range{7, 2}, {1561, 9611, 9429, 7270, 7049, 7273, 1502, 3940, 2352, 5172,
                    2680, 2878, 8758, 8219});
  Tensor<double> corr_elem_3_5(
      Range{7, 2}, {2240, 9288, 9066, 13742, 6872, 2681, 2723, 444, 3912, 4480,
                    1290, 6933, 9373, 2781});
  matrix_il corr_il{{corr_elem_0_0, corr_elem_0_1, corr_elem_0_2, corr_elem_0_3,
                     corr_elem_0_4, corr_elem_0_5},
                    {corr_elem_1_0, corr_elem_1_1, corr_elem_1_2, corr_elem_1_3,
                     corr_elem_1_4, corr_elem_1_5},
                    {corr_elem_2_0, corr_elem_2_1, corr_elem_2_2, corr_elem_2_3,
                     corr_elem_2_4, corr_elem_2_5},
                    {corr_elem_3_0, corr_elem_3_1, corr_elem_3_2, corr_elem_3_3,
                     corr_elem_3_4, corr_elem_3_5}};
  TiledRange corr_trange{{0, 2, 4}, {0, 2, 4, 6}};
  dist_array_t corr(world, corr_trange, corr_il);
  dist_array_t out = einsum(lhs("i,j;m,n"), rhs("k,j;m,n"), "i,k;m,n");
  const bool are_equal = ToTArrayFixture::are_equal(corr, out);
  BOOST_CHECK(are_equal);
}

BOOST_AUTO_TEST_CASE(ij_mn_eq_ij_mn_times_ij_mn) {
  using dist_array_t = DistArray<Tensor<Tensor<double>>, DensePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {86, 79, 35, 52, 36, 68, 39, 28, 83, 16, 18, 74, 83, 62});
  Tensor<double> lhs_elem_0_1(Range{8, 3},
                              {99, 66, 95, 76, 38, 24, 6,  79, 85, 52, 93, 90,
                               5,  4,  41, 42, 49, 4,  42, 94, 60, 99, 69, 8});
  Tensor<double> lhs_elem_1_0(Range{8, 3},
                              {5,  40, 94, 20, 25, 34, 25, 68, 94, 80, 61, 11,
                               94, 62, 25, 2,  64, 63, 28, 24, 7,  98, 59, 30});
  Tensor<double> lhs_elem_1_1(
      Range{9, 4},
      {37, 39, 26, 63, 10, 60, 5,  84, 72, 84, 10, 85, 33, 30, 12, 54, 63, 28,
       21, 7,  27, 12, 47, 86, 58, 67, 99, 52, 92, 97, 72, 92, 10, 42, 23, 8});
  Tensor<double> lhs_elem_2_0(
      Range{9, 4},
      {81, 43, 70, 13, 91, 91, 2,  43, 85, 72, 69, 28, 20, 61, 96, 9,  80, 42,
       17, 39, 85, 50, 18, 82, 56, 14, 95, 23, 22, 90, 80, 19, 83, 93, 43, 35});
  Tensor<double> lhs_elem_2_1(
      Range{10, 5},
      {20, 80, 94, 53, 72, 67, 77, 47, 81, 30, 65, 54, 61, 55, 48, 47, 91,
       20, 64, 69, 38, 49, 24, 39, 57, 24, 2,  31, 85, 12, 25, 15, 21, 59,
       99, 84, 8,  90, 45, 95, 55, 72, 46, 98, 40, 26, 2,  64, 46, 21});
  Tensor<double> lhs_elem_3_0(
      Range{10, 5},
      {30, 40, 24, 61, 11, 46, 35, 34, 79, 18, 55, 96, 43, 5,  11, 74, 78,
       82, 19, 19, 19, 72, 17, 81, 64, 13, 55, 95, 98, 99, 28, 43, 51, 38,
       33, 26, 6,  4,  97, 3,  91, 62, 89, 42, 71, 28, 81, 98, 72, 41});
  Tensor<double> lhs_elem_3_1(
      Range{11, 6},
      {100, 81, 93, 37, 58, 97, 90, 52, 94, 48, 63, 80, 12, 12, 13, 32, 93,
       42,  6,  3,  13, 82, 46, 57, 2,  28, 95, 90, 14, 87, 29, 40, 39, 36,
       69,  33, 10, 25, 57, 21, 77, 42, 68, 46, 4,  89, 21, 71, 20, 10, 31,
       54,  64, 96, 82, 72, 17, 19, 6,  93, 22, 78, 7,  23, 52, 63});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1},
                   {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1},
                   {lhs_elem_3_0, lhs_elem_3_1}};
  TiledRange lhs_trange{{0, 2, 4}, {0, 2}};
  dist_array_t lhs(world, lhs_trange, lhs_il);
  Tensor<double> rhs_elem_0_0(
      Range{7, 2}, {51, 46, 7, 36, 56, 97, 4, 53, 67, 6, 35, 13, 62, 30});
  Tensor<double> rhs_elem_0_1(Range{8, 3},
                              {20, 83, 5, 78, 25, 77, 5,  3,  59, 54, 92, 98,
                               32, 28, 5, 94, 81, 35, 71, 87, 60, 30, 57, 26});
  Tensor<double> rhs_elem_1_0(Range{8, 3},
                              {56, 27, 42, 85, 52, 87, 70, 73, 63, 79, 33, 18,
                               8,  17, 18, 35, 97, 18, 6,  92, 14, 14, 13, 8});
  Tensor<double> rhs_elem_1_1(
      Range{9, 4},
      {32, 8,  91, 99, 88, 52, 41, 94, 14, 17, 1,  73, 43, 20, 89, 1,  63, 8,
       54, 20, 20, 37, 37, 9,  85, 38, 76, 6,  67, 52, 7,  87, 82, 33, 93, 2});
  Tensor<double> rhs_elem_2_0(
      Range{9, 4},
      {48, 66, 62, 8,  22, 94, 97, 43, 22, 20, 16, 10, 38, 40, 18, 13, 21, 25,
       73, 46, 91, 69, 13, 23, 94, 66, 66, 24, 51, 68, 89, 15, 69, 46, 9,  6});
  Tensor<double> rhs_elem_2_1(
      Range{10, 5},
      {86, 74, 40, 15, 11, 16, 2,  40, 24, 14, 95, 66, 58, 44, 30, 22, 88,
       89, 45, 89, 51, 39, 70, 19, 39, 73, 6,  30, 72, 62, 34, 44, 79, 60,
       81, 12, 68, 18, 68, 57, 83, 86, 47, 28, 20, 34, 60, 47, 99, 20});
  Tensor<double> rhs_elem_3_0(
      Range{10, 5},
      {84, 95, 77,  56, 80, 26, 91, 80, 74, 92, 76, 25, 43, 58, 28, 65, 67,
       38, 60, 1,   65, 52, 77, 34, 52, 5,  47, 77, 65, 20, 30, 51, 97, 37,
       23, 85, 100, 40, 91, 50, 14, 30, 97, 70, 83, 58, 28, 69, 26, 36});
  Tensor<double> rhs_elem_3_1(
      Range{11, 6},
      {3,  13, 87, 60, 90, 52, 7,  20, 4,  56, 41, 44, 83, 48, 41, 38, 65,
       80, 60, 63, 55, 70, 37, 9,  45, 30, 18, 27, 4,  50, 81, 45, 19, 38,
       54, 15, 71, 66, 42, 29, 42, 36, 20, 34, 79, 10, 81, 52, 85, 50, 18,
       7,  96, 84, 64, 87, 68, 55, 45, 67, 29, 14, 76, 12, 89, 86});
  matrix_il rhs_il{{rhs_elem_0_0, rhs_elem_0_1},
                   {rhs_elem_1_0, rhs_elem_1_1},
                   {rhs_elem_2_0, rhs_elem_2_1},
                   {rhs_elem_3_0, rhs_elem_3_1}};
  TiledRange rhs_trange{{0, 2, 4}, {0, 2}};
  dist_array_t rhs(world, rhs_trange, rhs_il);
  Tensor<double> corr_elem_0_0(
      Range{7, 2}, {4386, 3634, 245, 1872, 2016, 6596, 156, 1484, 5561, 96, 630,
                    962, 5146, 1860});
  Tensor<double> corr_elem_0_1(
      Range{8, 3},
      {1980, 5478, 475, 5928, 950,  1848, 30,   237,  5015, 2808, 8556, 8820,
       160,  112,  205, 3948, 3969, 140,  2982, 8178, 3600, 2970, 3933, 208});
  Tensor<double> corr_elem_1_0(
      Range{8, 3},
      {280, 1080, 3948, 1700, 1300, 2958, 1750, 4964, 5922, 6320, 2013, 198,
       752, 1054, 450,  70,   6208, 1134, 168,  2208, 98,   1372, 767,  240});
  Tensor<double> corr_elem_1_1(
      Range{9, 4},
      {1184, 312,  2366, 6237, 880,  3120, 205,  7896, 1008, 1428, 10,   6205,
       1419, 600,  1068, 54,   3969, 224,  1134, 140,  540,  444,  1739, 774,
       4930, 2546, 7524, 312,  6164, 5044, 504,  8004, 820,  1386, 2139, 16});
  Tensor<double> corr_elem_2_0(
      Range{9, 4},
      {3888, 2838, 4340, 104, 2002, 8554, 194,  1849, 1870, 1440, 1104, 280,
       760,  2440, 1728, 117, 1680, 1050, 1241, 1794, 7735, 3450, 234,  1886,
       5264, 924,  6270, 552, 1122, 6120, 7120, 285,  5727, 4278, 387,  210});
  Tensor<double> corr_elem_2_1(
      Range{10, 5},
      {1720, 5920, 3760, 795,  792,  1072, 154,  1880, 1944, 420,
       6175, 3564, 3538, 2420, 1440, 1034, 8008, 1780, 2880, 6141,
       1938, 1911, 1680, 741,  2223, 1752, 12,   930,  6120, 744,
       850,  660,  1659, 3540, 8019, 1008, 544,  1620, 3060, 5415,
       4565, 6192, 2162, 2744, 800,  884,  120,  3008, 4554, 420});
  Tensor<double> corr_elem_3_0(
      Range{10, 5},
      {2520, 3800, 1848, 3416, 880,  1196, 3185, 2720, 5846, 1656,
       4180, 2400, 1849, 290,  308,  4810, 5226, 3116, 1140, 19,
       1235, 3744, 1309, 2754, 3328, 65,   2585, 7315, 6370, 1980,
       840,  2193, 4947, 1406, 759,  2210, 600,  160,  8827, 150,
       1274, 1860, 8633, 2940, 5893, 1624, 2268, 6762, 1872, 1476});
  Tensor<double> corr_elem_3_1(
      Range{11, 6},
      {300,  1053, 8091, 2220, 5220, 5044, 630,  1040, 376,  2688, 2583,
       3520, 996,  576,  533,  1216, 6045, 3360, 360,  189,  715,  5740,
       1702, 513,  90,   840,  1710, 2430, 56,   4350, 2349, 1800, 741,
       1368, 3726, 495,  710,  1650, 2394, 609,  3234, 1512, 1360, 1564,
       316,  890,  1701, 3692, 1700, 500,  558,  378,  6144, 8064, 5248,
       6264, 1156, 1045, 270,  6231, 638,  1092, 532,  276,  4628, 5418});
  matrix_il corr_il{{corr_elem_0_0, corr_elem_0_1},
                    {corr_elem_1_0, corr_elem_1_1},
                    {corr_elem_2_0, corr_elem_2_1},
                    {corr_elem_3_0, corr_elem_3_1}};
  TiledRange corr_trange{{0, 2, 4}, {0, 2}};
  dist_array_t corr(world, corr_trange, corr_il);
  dist_array_t out = einsum(lhs("i,j;m,n"), rhs("i,j;m,n"), "i,j;m,n");
  const bool are_equal = ToTArrayFixture::are_equal(corr, out);
  BOOST_CHECK(are_equal);
}

BOOST_AUTO_TEST_CASE(ij_mn_eq_ij_mn_times_ji_mn) {
  using dist_array_t = DistArray<Tensor<Tensor<double>>, DensePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {15, 75, 54, 54, 72, 62, 97, 90, 17, 94, 19, 54, 13, 31});
  Tensor<double> lhs_elem_0_1(Range{8, 3},
                              {82, 91, 60, 11, 47, 38, 87, 13, 72, 39, 59, 90,
                               26, 38, 2,  34, 30, 32, 46, 6,  26, 92, 47, 14});
  Tensor<double> lhs_elem_1_0(Range{8, 3},
                              {53, 88, 72, 12, 58, 85, 55, 6,  50, 76, 51, 52,
                               77, 13, 4,  99, 30, 12, 16, 21, 60, 75, 55, 99});
  Tensor<double> lhs_elem_1_1(
      Range{9, 4},
      {16, 65, 6,  84, 85, 30, 97, 79, 2,  13, 4,  90, 32, 98, 88, 40, 25, 27,
       8,  50, 56, 5,  42, 11, 20, 3,  51, 55, 32, 75, 8,  25, 4,  99, 75, 50});
  Tensor<double> lhs_elem_2_0(
      Range{9, 4}, {39, 24, 23, 32, 10, 22,  94, 47, 85, 22, 77, 22,
                    92, 28, 61, 53, 21, 81,  57, 63, 37, 75, 93, 91,
                    24, 14, 56, 69, 42, 100, 17, 44, 78, 47, 33, 67});
  Tensor<double> lhs_elem_2_1(
      Range{10, 5},
      {93, 27, 38, 15, 87, 88, 48, 19, 54, 81, 6,  60, 70, 75, 1,  21, 34,
       6,  74, 26, 5,  5,  75, 21, 31, 62, 53, 18, 17, 14, 19, 33, 96, 56,
       94, 12, 30, 14, 94, 31, 25, 59, 72, 88, 66, 98, 56, 79, 11, 50});
  Tensor<double> lhs_elem_3_0(
      Range{10, 5},
      {49, 46, 13, 98, 77, 100, 23, 99, 77, 64, 10, 31, 10, 70, 30, 18, 89,
       45, 81, 24, 45, 39, 83,  31, 3,  89, 35, 93, 70, 84, 43, 26, 96, 59,
       57, 1,  3,  33, 27, 53,  33, 3,  53, 7,  80, 54, 47, 77, 62, 23});
  Tensor<double> lhs_elem_3_1(
      Range{11, 6},
      {27, 61, 27, 63, 45, 14, 80, 20, 73, 74, 74, 9,  59, 92, 5,  4,  78,
       27, 53, 94, 70, 74, 1,  48, 30, 97, 51, 42, 93, 93, 81, 94, 73, 67,
       23, 98, 58, 17, 75, 73, 92, 16, 59, 5,  82, 22, 43, 58, 68, 44, 27,
       69, 79, 42, 99, 48, 78, 18, 9,  63, 1,  50, 9,  10, 82, 39});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1},
                   {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1},
                   {lhs_elem_3_0, lhs_elem_3_1}};
  TiledRange lhs_trange{{0, 2, 4}, {0, 2}};
  dist_array_t lhs(world, lhs_trange, lhs_il);
  Tensor<double> rhs_elem_0_0(
      Range{7, 2}, {55, 2, 99, 28, 98, 27, 80, 69, 1, 66, 5, 9, 1, 80});
  Tensor<double> rhs_elem_0_1(Range{8, 3},
                              {19, 23, 52, 93, 6,  89, 68, 10, 4,  23, 24, 20,
                               99, 85, 81, 36, 82, 54, 36, 46, 26, 85, 15, 28});
  Tensor<double> rhs_elem_0_2(
      Range{9, 4},
      {57, 32, 86, 49, 55, 32, 100, 46, 2, 82, 84, 69, 63, 69, 12, 62, 21, 87,
       1,  40, 61, 56, 90, 53, 74,  72, 5, 21, 49, 97, 69, 83, 48, 38, 88, 9});
  Tensor<double> rhs_elem_0_3(
      Range{10, 5},
      {28, 7,   4,  92, 30, 7,  3,  70, 16, 51, 71, 14, 37, 33,  92, 90, 75,
       29, 52,  59, 15, 15, 96, 50, 39, 72, 22, 60, 56, 95, 45,  33, 25, 22,
       23, 100, 26, 27, 38, 88, 89, 36, 48, 46, 6,  88, 16, 100, 54, 43});
  Tensor<double> rhs_elem_1_0(
      Range{8, 3}, {55, 21, 79, 3,  77, 82, 65,  83, 66, 12, 100, 9,
                    40, 55, 8,  75, 82, 85, 100, 78, 39, 42, 65,  56});
  Tensor<double> rhs_elem_1_1(
      Range{9, 4},
      {45, 21, 58, 73, 57, 33, 27, 58, 56, 45, 88, 79, 78, 97, 23, 4,  87, 22,
       9,  21, 54, 44, 81, 98, 53, 60, 29, 70, 83, 75, 30, 56, 61, 67, 18, 61});
  Tensor<double> rhs_elem_1_2(
      Range{10, 5},
      {11, 17, 75, 95, 66, 51, 95, 79, 10, 2,  43, 3,  85, 64, 67, 50, 32,
       8,  48, 58, 35, 20, 87, 82, 40, 46, 70, 39, 46, 37, 38, 81, 87, 64,
       31, 32, 7,  14, 94, 21, 33, 75, 67, 5,  80, 80, 36, 53, 99, 93});
  Tensor<double> rhs_elem_1_3(
      Range{11, 6},
      {64, 8,  79, 99, 13,  5,  64, 76, 2,  81, 78, 89, 88, 89, 83, 99, 71,
       50, 18, 59, 91, 100, 91, 99, 20, 54, 72, 9,  43, 21, 61, 57, 18, 80,
       12, 27, 95, 31, 92,  4,  6,  59, 27, 82, 98, 32, 82, 53, 52, 8,  31,
       32, 38, 63, 32, 47,  24, 86, 64, 29, 86, 46, 96, 79, 48, 58});
  matrix_il rhs_il{{rhs_elem_0_0, rhs_elem_0_1, rhs_elem_0_2, rhs_elem_0_3},
                   {rhs_elem_1_0, rhs_elem_1_1, rhs_elem_1_2, rhs_elem_1_3}};
  TiledRange rhs_trange{{0, 2}, {0, 2, 4}};
  dist_array_t rhs(world, rhs_trange, rhs_il);
  Tensor<double> corr_elem_0_0(
      Range{7, 2}, {825, 150, 5346, 1512, 7056, 1674, 7760, 6210, 17, 6204, 95,
                    486, 13, 2480});
  Tensor<double> corr_elem_0_1(
      Range{8, 3},
      {4510, 1911, 4740, 33,   3619, 3116, 5655, 1079, 4752, 468,  5900, 810,
       1040, 2090, 16,   2550, 2460, 2720, 4600, 468,  1014, 3864, 3055, 784});
  Tensor<double> corr_elem_1_0(
      Range{8, 3},
      {1007, 2024, 3744, 1116, 348,  7565, 3740, 60,  200,  1748, 1224, 1040,
       7623, 1105, 324,  3564, 2460, 648,  576,  966, 1560, 6375, 825,  2772});
  Tensor<double> corr_elem_1_1(
      Range{9, 4},
      {720,  1365, 348,  6132, 4845, 990,  2619, 4582, 112,  585,  352,  7110,
       2496, 9506, 2024, 160,  2175, 594,  72,   1050, 3024, 220,  3402, 1078,
       1060, 180,  1479, 3850, 2656, 5625, 240,  1400, 244,  6633, 1350, 3050});
  Tensor<double> corr_elem_2_0(
      Range{9, 4},
      {2223, 768,  1978, 1568, 550,  704,  9400, 2162, 170,  1804, 6468, 1518,
       5796, 1932, 732,  3286, 441,  7047, 57,   2520, 2257, 4200, 8370, 4823,
       1776, 1008, 280,  1449, 2058, 9700, 1173, 3652, 3744, 1786, 2904, 603});
  Tensor<double> corr_elem_2_1(
      Range{10, 5},
      {1023, 459,  2850, 1425, 5742, 4488, 4560, 1501, 540,  162,
       258,  180,  5950, 4800, 67,   1050, 1088, 48,   3552, 1508,
       175,  100,  6525, 1722, 1240, 2852, 3710, 702,  782,  518,
       722,  2673, 8352, 3584, 2914, 384,  210,  196,  8836, 651,
       825,  4425, 4824, 440,  5280, 7840, 2016, 4187, 1089, 4650});
  Tensor<double> corr_elem_3_0(
      Range{10, 5}, {1372, 322, 52,   9016, 2310, 700,  69,   6930, 1232, 3264,
                     710,  434, 370,  2310, 2760, 1620, 6675, 1305, 4212, 1416,
                     675,  585, 7968, 1550, 117,  6408, 770,  5580, 3920, 7980,
                     1935, 858, 2400, 1298, 1311, 100,  78,   891,  1026, 4664,
                     2937, 108, 2544, 322,  480,  4752, 752,  7700, 3348, 989});
  Tensor<double> corr_elem_3_1(
      Range{11, 6},
      {1728, 488,  2133, 6237, 585,  70,   5120, 1520, 146,  5994, 5772,
       801,  5192, 8188, 415,  396,  5538, 1350, 954,  5546, 6370, 7400,
       91,   4752, 600,  5238, 3672, 378,  3999, 1953, 4941, 5358, 1314,
       5360, 276,  2646, 5510, 527,  6900, 292,  552,  944,  1593, 410,
       8036, 704,  3526, 3074, 3536, 352,  837,  2208, 3002, 2646, 3168,
       2256, 1872, 1548, 576,  1827, 86,   2300, 864,  790,  3936, 2262});
  matrix_il corr_il{{corr_elem_0_0, corr_elem_0_1},
                    {corr_elem_1_0, corr_elem_1_1},
                    {corr_elem_2_0, corr_elem_2_1},
                    {corr_elem_3_0, corr_elem_3_1}};
  TiledRange corr_trange{{0, 2, 4}, {0, 2}};
  dist_array_t corr(world, corr_trange, corr_il);
  dist_array_t out = einsum(lhs("i,j;m,n"), rhs("j,i;m,n"), "i,j;m,n");
  const bool are_equal = ToTArrayFixture::are_equal(corr, out);
  BOOST_CHECK(are_equal);
}

BOOST_AUTO_TEST_CASE(ijk_mn_eq_ij_mn_times_kj_mn) {
  using tot_type = DistArray<Tensor<Tensor<double>>, DensePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();

  auto random_tot = [](TA::Range const& rng) {
    TA::Range inner_rng{7, 14};
    TA::Tensor<double> t{inner_rng};
    std::generate(t.begin(), t.end(), []() -> double {
      return TA::detail::MakeRandom<double>::generate_value();
    });
    TA::Tensor<TA::Tensor<double>> result{rng};
    for (auto& e : result) e = t;
    return result;
  };

  auto random_tot_darr = [&random_tot](World& world, TiledRange const& tr) {
    tot_type result(world, tr);
    for (auto it = result.begin(); it != result.end(); ++it) {
      auto tile =
          TA::get_default_world().taskq.add(random_tot, it.make_range());
      *it = tile;
    }
    return result;
  };

  TiledRange lhs_trange{{0, 2, 4}, {0, 2, 5}};
  auto lhs = random_tot_darr(world, lhs_trange);

  TiledRange rhs_trange{{0, 2, 4, 6}, {0, 2, 5}};
  auto rhs = random_tot_darr(world, rhs_trange);
  tot_type result;
  BOOST_REQUIRE_NO_THROW(
      result = einsum(lhs("i,j;m,n"), rhs("k,j;m,n"), "i,j,k;m,n"));

  // i,j,k;m,n = i,j;m,n * k,j;m,n
  TiledRange ref_result_trange{lhs.trange().dim(0), lhs.trange().dim(1),
                               rhs.trange().dim(0)};
  tot_type ref_result(world, ref_result_trange);

  // to be able to pull remote tiles make them local AND ready
  lhs.make_replicated();
  rhs.make_replicated();
  world.gop.fence();
  auto make_tile = [&lhs, &rhs](TA::Range const& rng) {
    tot_type::value_type result_tile{rng};
    for (auto&& res_ix : result_tile.range()) {
      auto i = res_ix[0];
      auto j = res_ix[1];
      auto k = res_ix[2];

      auto lhs_tile_ix = lhs.trange().element_to_tile({i, j});
      auto lhs_tile = lhs.find_local(lhs_tile_ix).get(/* dowork = */ false);
      auto rhs_tile_ix = rhs.trange().element_to_tile({k, j});
      auto rhs_tile = rhs.find_local(rhs_tile_ix).get(/* dowork = */ false);

      auto& res_el = result_tile({i, j, k});
      auto const& lhs_el = lhs_tile({i, j});
      auto rhs_el = rhs_tile({k, j});
      res_el = lhs_el.mult(rhs_el);  // m,n * m,n -> m,n
    }
    return result_tile;
  };

  using std::begin;
  using std::end;

  for (auto it = begin(ref_result); it != end(ref_result); ++it) {
    if (ref_result.is_local(it.index())) {
      *it = world.taskq.add(make_tile, it.make_range());
    }
  }
  bool are_equal =
      ToTArrayFixture::are_equal<ShapeComp::False>(result, ref_result);
  BOOST_REQUIRE(are_equal);
}

BOOST_AUTO_TEST_CASE(xxx) {
  using dist_array_t = DistArray<Tensor<Tensor<double>>, DensePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {15, 75, 54, 54, 72, 62, 97, 90, 17, 94, 19, 54, 13, 31});
  Tensor<double> lhs_elem_0_1(Range{8, 3},
                              {82, 91, 60, 11, 47, 38, 87, 13, 72, 39, 59, 90,
                               26, 38, 2,  34, 30, 32, 46, 6,  26, 92, 47, 14});
  Tensor<double> lhs_elem_1_0(Range{8, 3},
                              {53, 88, 72, 12, 58, 85, 55, 6,  50, 76, 51, 52,
                               77, 13, 4,  99, 30, 12, 16, 21, 60, 75, 55, 99});
  Tensor<double> lhs_elem_1_1(
      Range{9, 4},
      {16, 65, 6,  84, 85, 30, 97, 79, 2,  13, 4,  90, 32, 98, 88, 40, 25, 27,
       8,  50, 56, 5,  42, 11, 20, 3,  51, 55, 32, 75, 8,  25, 4,  99, 75, 50});
  Tensor<double> lhs_elem_2_0(
      Range{9, 4}, {39, 24, 23, 32, 10, 22,  94, 47, 85, 22, 77, 22,
                    92, 28, 61, 53, 21, 81,  57, 63, 37, 75, 93, 91,
                    24, 14, 56, 69, 42, 100, 17, 44, 78, 47, 33, 67});
  Tensor<double> lhs_elem_2_1(
      Range{10, 5},
      {93, 27, 38, 15, 87, 88, 48, 19, 54, 81, 6,  60, 70, 75, 1,  21, 34,
       6,  74, 26, 5,  5,  75, 21, 31, 62, 53, 18, 17, 14, 19, 33, 96, 56,
       94, 12, 30, 14, 94, 31, 25, 59, 72, 88, 66, 98, 56, 79, 11, 50});
  Tensor<double> lhs_elem_3_0(
      Range{10, 5},
      {49, 46, 13, 98, 77, 100, 23, 99, 77, 64, 10, 31, 10, 70, 30, 18, 89,
       45, 81, 24, 45, 39, 83,  31, 3,  89, 35, 93, 70, 84, 43, 26, 96, 59,
       57, 1,  3,  33, 27, 53,  33, 3,  53, 7,  80, 54, 47, 77, 62, 23});
  Tensor<double> lhs_elem_3_1(
      Range{11, 6},
      {27, 61, 27, 63, 45, 14, 80, 20, 73, 74, 74, 9,  59, 92, 5,  4,  78,
       27, 53, 94, 70, 74, 1,  48, 30, 97, 51, 42, 93, 93, 81, 94, 73, 67,
       23, 98, 58, 17, 75, 73, 92, 16, 59, 5,  82, 22, 43, 58, 68, 44, 27,
       69, 79, 42, 99, 48, 78, 18, 9,  63, 1,  50, 9,  10, 82, 39});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1},
                   {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1},
                   {lhs_elem_3_0, lhs_elem_3_1}};
  TiledRange lhs_trange{{0, 2, 4}, {0, 2}};
  dist_array_t lhs(world, lhs_trange, lhs_il);
  Tensor<double> rhs_elem_0_0(
      Range{7, 2}, {55, 2, 99, 28, 98, 27, 80, 69, 1, 66, 5, 9, 1, 80});
  Tensor<double> rhs_elem_0_1(Range{8, 3},
                              {19, 23, 52, 93, 6,  89, 68, 10, 4,  23, 24, 20,
                               99, 85, 81, 36, 82, 54, 36, 46, 26, 85, 15, 28});
  Tensor<double> rhs_elem_0_2(
      Range{9, 4},
      {57, 32, 86, 49, 55, 32, 100, 46, 2, 82, 84, 69, 63, 69, 12, 62, 21, 87,
       1,  40, 61, 56, 90, 53, 74,  72, 5, 21, 49, 97, 69, 83, 48, 38, 88, 9});
  Tensor<double> rhs_elem_0_3(
      Range{10, 5},
      {28, 7,   4,  92, 30, 7,  3,  70, 16, 51, 71, 14, 37, 33,  92, 90, 75,
       29, 52,  59, 15, 15, 96, 50, 39, 72, 22, 60, 56, 95, 45,  33, 25, 22,
       23, 100, 26, 27, 38, 88, 89, 36, 48, 46, 6,  88, 16, 100, 54, 43});
  Tensor<double> rhs_elem_1_0(
      Range{8, 3}, {55, 21, 79, 3,  77, 82, 65,  83, 66, 12, 100, 9,
                    40, 55, 8,  75, 82, 85, 100, 78, 39, 42, 65,  56});
  Tensor<double> rhs_elem_1_1(
      Range{9, 4},
      {45, 21, 58, 73, 57, 33, 27, 58, 56, 45, 88, 79, 78, 97, 23, 4,  87, 22,
       9,  21, 54, 44, 81, 98, 53, 60, 29, 70, 83, 75, 30, 56, 61, 67, 18, 61});
  Tensor<double> rhs_elem_1_2(
      Range{10, 5},
      {11, 17, 75, 95, 66, 51, 95, 79, 10, 2,  43, 3,  85, 64, 67, 50, 32,
       8,  48, 58, 35, 20, 87, 82, 40, 46, 70, 39, 46, 37, 38, 81, 87, 64,
       31, 32, 7,  14, 94, 21, 33, 75, 67, 5,  80, 80, 36, 53, 99, 93});
  Tensor<double> rhs_elem_1_3(
      Range{11, 6},
      {64, 8,  79, 99, 13,  5,  64, 76, 2,  81, 78, 89, 88, 89, 83, 99, 71,
       50, 18, 59, 91, 100, 91, 99, 20, 54, 72, 9,  43, 21, 61, 57, 18, 80,
       12, 27, 95, 31, 92,  4,  6,  59, 27, 82, 98, 32, 82, 53, 52, 8,  31,
       32, 38, 63, 32, 47,  24, 86, 64, 29, 86, 46, 96, 79, 48, 58});
  matrix_il rhs_il{{rhs_elem_0_0, rhs_elem_0_1, rhs_elem_0_2, rhs_elem_0_3},
                   {rhs_elem_1_0, rhs_elem_1_1, rhs_elem_1_2, rhs_elem_1_3}};
  TiledRange rhs_trange{{0, 2}, {0, 2, 4}};
  dist_array_t rhs(world, rhs_trange, rhs_il);
  Tensor<double> corr_elem_0_0(
      Range{7, 2}, {825, 150, 5346, 1512, 7056, 1674, 7760, 6210, 17, 6204, 95,
                    486, 13, 2480});
  Tensor<double> corr_elem_0_1(
      Range{8, 3},
      {4510, 1911, 4740, 33,   3619, 3116, 5655, 1079, 4752, 468,  5900, 810,
       1040, 2090, 16,   2550, 2460, 2720, 4600, 468,  1014, 3864, 3055, 784});
  Tensor<double> corr_elem_1_0(
      Range{8, 3},
      {1007, 2024, 3744, 1116, 348,  7565, 3740, 60,  200,  1748, 1224, 1040,
       7623, 1105, 324,  3564, 2460, 648,  576,  966, 1560, 6375, 825,  2772});
  Tensor<double> corr_elem_1_1(
      Range{9, 4},
      {720,  1365, 348,  6132, 4845, 990,  2619, 4582, 112,  585,  352,  7110,
       2496, 9506, 2024, 160,  2175, 594,  72,   1050, 3024, 220,  3402, 1078,
       1060, 180,  1479, 3850, 2656, 5625, 240,  1400, 244,  6633, 1350, 3050});
  Tensor<double> corr_elem_2_0(
      Range{9, 4},
      {2223, 768,  1978, 1568, 550,  704,  9400, 2162, 170,  1804, 6468, 1518,
       5796, 1932, 732,  3286, 441,  7047, 57,   2520, 2257, 4200, 8370, 4823,
       1776, 1008, 280,  1449, 2058, 9700, 1173, 3652, 3744, 1786, 2904, 603});
  Tensor<double> corr_elem_2_1(
      Range{10, 5},
      {1023, 459,  2850, 1425, 5742, 4488, 4560, 1501, 540,  162,
       258,  180,  5950, 4800, 67,   1050, 1088, 48,   3552, 1508,
       175,  100,  6525, 1722, 1240, 2852, 3710, 702,  782,  518,
       722,  2673, 8352, 3584, 2914, 384,  210,  196,  8836, 651,
       825,  4425, 4824, 440,  5280, 7840, 2016, 4187, 1089, 4650});
  Tensor<double> corr_elem_3_0(
      Range{10, 5}, {1372, 322, 52,   9016, 2310, 700,  69,   6930, 1232, 3264,
                     710,  434, 370,  2310, 2760, 1620, 6675, 1305, 4212, 1416,
                     675,  585, 7968, 1550, 117,  6408, 770,  5580, 3920, 7980,
                     1935, 858, 2400, 1298, 1311, 100,  78,   891,  1026, 4664,
                     2937, 108, 2544, 322,  480,  4752, 752,  7700, 3348, 989});
  Tensor<double> corr_elem_3_1(
      Range{11, 6},
      {1728, 488,  2133, 6237, 585,  70,   5120, 1520, 146,  5994, 5772,
       801,  5192, 8188, 415,  396,  5538, 1350, 954,  5546, 6370, 7400,
       91,   4752, 600,  5238, 3672, 378,  3999, 1953, 4941, 5358, 1314,
       5360, 276,  2646, 5510, 527,  6900, 292,  552,  944,  1593, 410,
       8036, 704,  3526, 3074, 3536, 352,  837,  2208, 3002, 2646, 3168,
       2256, 1872, 1548, 576,  1827, 86,   2300, 864,  790,  3936, 2262});
  matrix_il corr_il{{corr_elem_0_0, corr_elem_0_1},
                    {corr_elem_1_0, corr_elem_1_1},
                    {corr_elem_2_0, corr_elem_2_1},
                    {corr_elem_3_0, corr_elem_3_1}};
  TiledRange corr_trange{{0, 2, 4}, {0, 2}};
  dist_array_t corr(world, corr_trange, corr_il);
  dist_array_t r0 = einsum(lhs("i,h;m,n"), rhs("h,i;m,n"), "i,h;m,n");
  dist_array_t r1;
  einsum(r1("i,h;m,n"), lhs("i,h;m,n"), rhs("h,i;m,n"));

  const bool are_equal = ToTArrayFixture::are_equal(r0, r1);
  BOOST_CHECK(are_equal);
}

BOOST_AUTO_TEST_CASE(ij_mn_eq_ij_mo_times_ji_on) {
  auto& world = TA::get_default_world();

  using Array = TA::DistArray<TA::Tensor<TA::Tensor<int>>, TA::DensePolicy>;
  using Perm = TA::Permutation;

  TA::TiledRange lhs_trng{{0, 2, 3}, {0, 1}};
  TA::TiledRange rhs_trng{{0, 1}, {0, 2, 3}};
  TA::Range lhs_inner_rng{1, 1};
  TA::Range rhs_inner_rng{1, 1};

  auto lhs = random_array<Array>(lhs_trng, lhs_inner_rng);
  auto rhs = random_array<Array>(rhs_trng, rhs_inner_rng);
  Array out;
  BOOST_REQUIRE_NO_THROW(out("i,j;m,n") = lhs("i,j;m,o") * rhs("j,i;o,n"));
}

BOOST_AUTO_TEST_CASE(ij_mn_eq_ijk_mo_times_ijk_no) {
  using Array = TA::DistArray<TA::Tensor<TA::Tensor<int>>, TA::DensePolicy>;
  using Ix = typename TA::Range::index1_type;
  using namespace std::string_literals;
  auto& world = TA::get_default_world();

  Ix const K = 2;  // the extent of contracted outer mode

  TA::Range const inner_rng{3, 7};
  TA::TiledRange const lhs_trng{
      std::initializer_list<std::initializer_list<Ix>>{
          {0, 2, 4}, {0, 2}, {0, 2}}};
  TA::TiledRange const rhs_trng(lhs_trng);
  TA::TiledRange const ref_trng{lhs_trng.dim(0), lhs_trng.dim(1)};
  TA::Range const ref_inner_rng{3, 3};  // contract(3x7,3x7) -> (3,3)
  auto lhs = random_array<Array>(lhs_trng, inner_rng);
  auto rhs = random_array<Array>(rhs_trng, inner_rng);

  //
  // manual evaluation: ij;mn = ijk;mo * ijk;no
  //
  Array ref{world, ref_trng};
  {
    lhs.make_replicated();
    rhs.make_replicated();
    world.gop.fence();

    auto make_tile = [lhs, rhs, ref_inner_rng](TA::Range const& rng) {
      using InnerT = typename Array::value_type::value_type;
      typename Array::value_type result_tile{rng};

      for (auto&& res_ix : result_tile.range()) {
        auto i = res_ix[0];
        auto j = res_ix[1];

        InnerT mn;
        for (Ix k = 0; k < K; ++k) {
          auto lhs_tile =
              lhs.find_local(lhs.trange().element_to_tile({i, j, k}))
                  .get(/*dowork = */ false);
          auto rhs_tile =
              rhs.find_local(rhs.trange().element_to_tile({i, j, k}))
                  .get(/*dowork = */ false);
          mn.add_to(tensor_contract("mo,no->mn", lhs_tile({i, j, k}),
                                    rhs_tile({i, j, k})));
        }
        result_tile({i, j}) = std::move(mn);
      }
      return result_tile;
    };
    using std::begin;
    using std::end;

    for (auto it = begin(ref); it != end(ref); ++it)
      if (ref.is_local(it.index())) {
        auto tile = world.taskq.add(make_tile, it.make_range());
        *it = tile;
      }
  }

  auto out = einsum(lhs("i,j,k;m,o"), rhs("i,j,k;n,o"), "i,j;m,n");
  bool are_equal = ToTArrayFixture::are_equal<ShapeComp::False>(ref, out);

  BOOST_CHECK(are_equal);
}

#ifdef TILEDARRAY_HAS_BTAS
BOOST_AUTO_TEST_CASE(tensor_contract) {
  using TensorT = TA::Tensor<int>;

  TA::Range const rng_A{2, 3, 4};
  TA::Range const rng_B{4, 3, 2};
  auto const A = random_tensor<TensorT>(rng_A);
  auto const B = random_tensor<TensorT>(rng_B);

  BOOST_CHECK(tensor_contract_equal("ijk,klm->ijlm", A, B));
  BOOST_CHECK(tensor_contract_equal("ijk,klm->milj", A, B));
  BOOST_CHECK(tensor_contract_equal("ijk,kjm->im", A, B));
  BOOST_CHECK(tensor_contract_equal("ijk,kli->lj", A, B));
}
#endif

// --------------------------------------------------------------------------
// strided-DGEMM ce+e (hce+e) e2e oracles
// --------------------------------------------------------------------------

// c(i,k;p,q) = sum_j a(i,j;p) * b(k,j;q): distinct externals i,k; contract
// over j; inner OUTER product p x q (no inner contraction). Runs the SAME
// numeric problem on (a) an *arena*-backed ToT DistArray (inner cells are
// ArenaTensor views) and (b) an *owning* ToT DistArray (inner cells are
// TA::Tensor), then asserts the two einsum results agree elementwise.
BOOST_AUTO_TEST_CASE(hce_e_contraction_arena_matches_owning) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();

  constexpr long I = 2, J = 2, K = 3, P = 3, Q = 4;

  TA::TiledRange a_trange{{0l, I}, {0l, J}};  // outer (i,j)
  TA::TiledRange b_trange{{0l, K}, {0l, J}};  // outer (k,j)

  auto a_val = [](long i, long j, long p) {
    return 1.0 + 0.5 * i + 0.25 * j + 0.125 * p;
  };
  auto b_val = [](long k, long j, long q) {
    return 2.0 - 0.3 * k + 0.2 * j + 0.05 * q;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{P}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long i = static_cast<long>(o / J);
      const long j = static_cast<long>(o % J);
      for (long p = 0; p < P; ++p) c.data()[p] = a_val(i, j, p);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long k = static_cast<long>(o / J);
      const long j = static_cast<long>(o % J);
      for (long q = 0; q < Q; ++q) c.data()[q] = b_val(k, j, q);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{P});
      const long i = static_cast<long>(o / J);
      const long j = static_cast<long>(o % J);
      for (long p = 0; p < P; ++p) cell.data()[p] = a_val(i, j, p);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{Q});
      const long k = static_cast<long>(o / J);
      const long j = static_cast<long>(o % J);
      for (long q = 0; q < Q; ++q) cell.data()[q] = b_val(k, j, q);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

  ArenaArr ca = einsum(ah("i,j;p"), bh("k,j;q"), "i,k;p,q");
  OwnArr co = einsum(ao("i,j;p"), bo("k,j;q"), "i,k;p,q");
  world.gop.fence();

  BOOST_REQUIRE_EQUAL(ca.trange().elements_range().volume(),
                      static_cast<std::size_t>(I * K));
  BOOST_REQUIRE_EQUAL(co.trange().elements_range().volume(),
                      static_cast<std::size_t>(I * K));

  double max_abs_diff = 0.0;
  std::size_t elements_compared = 0;
  std::size_t result_outer_cells_seen = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    const bool a_here = ca.is_local(ord) && !ca.is_zero(ord);
    const bool o_here = co.is_local(ord) && !co.is_zero(ord);
    BOOST_REQUIRE_EQUAL(a_here, o_here);
    if (!a_here) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    BOOST_REQUIRE_EQUAL(at.range().volume(), ot.range().volume());
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      BOOST_REQUIRE_EQUAL(bool(ac), !oc.empty());
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), oc.size());
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(P * Q));
      ++result_outer_cells_seen;
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double d = std::abs(ac.data()[e] - oc.data()[e]);
        if (d > max_abs_diff) max_abs_diff = d;
        ++elements_compared;
      }
    }
  }
  constexpr std::size_t expected_cells = static_cast<std::size_t>(I * K);
  constexpr std::size_t expected_elements = expected_cells * P * Q;
  world.gop.sum(result_outer_cells_seen);
  world.gop.sum(elements_compared);
  world.gop.max(max_abs_diff);
  BOOST_REQUIRE_GT(elements_compared, 0u);
  BOOST_CHECK_EQUAL(result_outer_cells_seen, expected_cells);
  BOOST_CHECK_EQUAL(elements_compared, expected_elements);
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
}

// Regime A (NO outer-external index): outer h,i = Hadamard (both operands +
// result), outer k = contracted (both operands, NOT in result), inner a1
// (left-only) x a2 (right-only) => inner OUTER-PRODUCT. h folds to nbatch=2,
// k is multi-tile {2,3}. The arena (view) path must match the owning path.
BOOST_AUTO_TEST_CASE(regime_a_hce_e_arena_matches_owning) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();

  constexpr long H = 2, I = 2, P = 3, Q = 4;
  // h: one tile extent 2 (folds to nbatch=2); i: one tile extent 2 (Hadamard);
  // k: two tiles extents {2,3}.
  TA::TiledRange a_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I},
                          TA::TiledRange1{0l, 2, 5}};  // outer (h,i,k)
  TA::TiledRange b_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I},
                          TA::TiledRange1{0l, 2, 5}};  // outer (h,i,k)

  auto a_val = [](long h, long i, long k, long a1) {
    return 1.0 + 0.5 * i + 0.25 * k + 0.125 * a1 + 0.0625 * h;
  };
  auto b_val = [](long h, long i, long k, long a2) {
    return 2.0 - 0.3 * k + 0.2 * i + 0.05 * a2 + 0.03 * h;
  };

  // Decode each cell to global (h,i,k) using the passed tile's own Range
  // (k tiles are sub-tiles, so we must use global coordinates, not ordinal/J).
  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{P}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a1 = 0; a1 < P; ++a1) c.data()[a1] = a_val(h, i, k, a1);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a2 = 0; a2 < Q; ++a2) c.data()[a2] = b_val(h, i, k, a2);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{P});
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a1 = 0; a1 < P; ++a1) cell.data()[a1] = a_val(h, i, k, a1);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{Q});
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a2 = 0; a2 < Q; ++a2) cell.data()[a2] = b_val(h, i, k, a2);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

  // h,i = Hadamard outer; k = contracted (not in result); a1 x a2 = inner OP.
  ArenaArr ca = einsum(ah("h,i,k;a1"), bh("h,i,k;a2"), "h,i;a1,a2");
  OwnArr co = einsum(ao("h,i,k;a1"), bo("h,i,k;a2"), "h,i;a1,a2");
  world.gop.fence();

  // result outer is (h,i) => 2*2 = 4 outer cells (single tile each).
  BOOST_REQUIRE_EQUAL(ca.trange().elements_range().volume(),
                      static_cast<std::size_t>(H * I));
  BOOST_REQUIRE_EQUAL(co.trange().elements_range().volume(),
                      static_cast<std::size_t>(H * I));

  double max_abs_diff = 0.0;
  std::size_t elements_compared = 0;
  std::size_t result_outer_cells_seen = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    const bool a_here = ca.is_local(ord) && !ca.is_zero(ord);
    const bool o_here = co.is_local(ord) && !co.is_zero(ord);
    BOOST_REQUIRE_EQUAL(a_here, o_here);
    if (!a_here) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    BOOST_REQUIRE_EQUAL(at.range().volume(), ot.range().volume());
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      BOOST_REQUIRE_EQUAL(bool(ac), !oc.empty());
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), oc.size());
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(P * Q));
      ++result_outer_cells_seen;
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double d = std::abs(ac.data()[e] - oc.data()[e]);
        if (d > max_abs_diff) max_abs_diff = d;
        ++elements_compared;
      }
    }
  }
  constexpr std::size_t expected_cells = static_cast<std::size_t>(H * I);
  constexpr std::size_t expected_elements = expected_cells * P * Q;
  world.gop.sum(result_outer_cells_seen);
  world.gop.sum(elements_compared);
  world.gop.max(max_abs_diff);
  BOOST_REQUIRE_GT(elements_compared, 0u);
  BOOST_CHECK_EQUAL(result_outer_cells_seen, expected_cells);
  BOOST_CHECK_EQUAL(elements_compared, expected_elements);
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
}

// Task 3.1 differential: on EXACTLY the same arena inputs, the strided reroute
// and the legacy per-cell path must produce identical results. Builds ONE
// (a,b) arena pair (same construction as regime_a_hce_e_arena_matches_owning),
// runs the SAME einsum twice flipping regime_a_strided_disabled(), and compares
// the two results cell-by-cell, element-by-element. The toggle is RESTORED to
// false before the compare loop so a failed assertion cannot leak `true` into
// later tests.
BOOST_AUTO_TEST_CASE(regime_a_hce_e_strided_equals_percell) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();

  constexpr long H = 2, I = 2, P = 3, Q = 4;
  TA::TiledRange a_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I},
                          TA::TiledRange1{0l, 2, 5}};  // outer (h,i,k)
  TA::TiledRange b_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I},
                          TA::TiledRange1{0l, 2, 5}};  // outer (h,i,k)

  auto a_val = [](long h, long i, long k, long a1) {
    return 1.0 + 0.5 * i + 0.25 * k + 0.125 * a1 + 0.0625 * h;
  };
  auto b_val = [](long h, long i, long k, long a2) {
    return 2.0 - 0.3 * k + 0.2 * i + 0.05 * a2 + 0.03 * h;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{P}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a1 = 0; a1 < P; ++a1) c.data()[a1] = a_val(h, i, k, a1);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a2 = 0; a2 < Q; ++a2) c.data()[a2] = b_val(h, i, k, a2);
    }
    return t;
  });
  world.gop.fence();

  namespace det = TiledArray::detail;
  // RAII guard: always restore the global toggle to false on scope exit, even
  // if an einsum throws -- a leaked `true` would silently disable the strided
  // path for every later test in the binary.
  struct StridedToggleGuard {
    ~StridedToggleGuard() { det::regime_a_strided_disabled() = false; }
  } toggle_guard;

  det::regime_a_strided_disabled() = false;  // strided path
  auto c_strided = TiledArray::einsum(ah("h,i,k;a1"), bh("h,i,k;a2"), "h,i;a1,a2");
  c_strided.world().gop.fence();
  det::regime_a_strided_disabled() = true;  // per-cell path
  auto c_percell = TiledArray::einsum(ah("h,i,k;a1"), bh("h,i,k;a2"), "h,i;a1,a2");
  c_percell.world().gop.fence();
  det::regime_a_strided_disabled() = false;  // RESTORE default before compares

  BOOST_REQUIRE_EQUAL(c_strided.trange().elements_range().volume(),
                      static_cast<std::size_t>(H * I));
  BOOST_REQUIRE_EQUAL(c_percell.trange().elements_range().volume(),
                      static_cast<std::size_t>(H * I));

  std::size_t elements_compared = 0;
  std::size_t result_outer_cells_seen = 0;
  const auto& tiles = c_strided.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    const bool s_here = c_strided.is_local(ord) && !c_strided.is_zero(ord);
    const bool p_here = c_percell.is_local(ord) && !c_percell.is_zero(ord);
    BOOST_REQUIRE_EQUAL(s_here, p_here);
    if (!s_here) continue;
    const ArenaOuter& st = c_strided.find(ord).get();
    const ArenaOuter& pt = c_percell.find(ord).get();
    BOOST_REQUIRE_EQUAL(st.range().volume(), pt.range().volume());
    for (std::size_t o = 0; o < st.range().volume(); ++o) {
      const ArenaInner& sc = st.data()[o];
      const ArenaInner& pc = pt.data()[o];
      BOOST_REQUIRE_EQUAL(bool(sc), bool(pc));
      if (!sc) continue;
      BOOST_REQUIRE_EQUAL(sc.size(), pc.size());
      BOOST_REQUIRE_EQUAL(sc.size(), static_cast<std::size_t>(P * Q));
      ++result_outer_cells_seen;
      for (std::size_t e = 0; e < sc.size(); ++e) {
        BOOST_CHECK_CLOSE(sc.data()[e], pc.data()[e], 1e-12);
        ++elements_compared;
      }
    }
  }
  world.gop.sum(result_outer_cells_seen);
  world.gop.sum(elements_compared);
  BOOST_REQUIRE_GT(elements_compared, 0u);
  BOOST_CHECK_EQUAL(result_outer_cells_seen, static_cast<std::size_t>(H * I));
  BOOST_CHECK_EQUAL(elements_compared,
                    static_cast<std::size_t>(H * I) * P * Q);
}

// Task 3.3 Step 1: non-canonical result inner order (a2,a1 instead of a1,a2).
// Identical operands/fills to regime_a_hce_e_arena_matches_owning, but the
// result inner is reversed, firing the result inner-permute hoist (do_perm.C)
// inside the strided path. Arena and owning both go through the same DSL, so
// they must agree regardless of the kernel's internal [P x Q] blas layout.
BOOST_AUTO_TEST_CASE(regime_a_hce_e_noncanonical_inner_matches_owning) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();

  constexpr long H = 2, I = 2, P = 3, Q = 4;  // inner a1=P, a2=Q
  TA::TiledRange a_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I},
                          TA::TiledRange1{0l, 2, 5}};  // outer (h,i,k)
  TA::TiledRange b_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I},
                          TA::TiledRange1{0l, 2, 5}};  // outer (h,i,k)

  auto a_val = [](long h, long i, long k, long a1) {
    return 1.0 + 0.5 * i + 0.25 * k + 0.125 * a1 + 0.0625 * h;
  };
  auto b_val = [](long h, long i, long k, long a2) {
    return 2.0 - 0.3 * k + 0.2 * i + 0.05 * a2 + 0.03 * h;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{P}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a1 = 0; a1 < P; ++a1) c.data()[a1] = a_val(h, i, k, a1);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a2 = 0; a2 < Q; ++a2) c.data()[a2] = b_val(h, i, k, a2);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{P});
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a1 = 0; a1 < P; ++a1) cell.data()[a1] = a_val(h, i, k, a1);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{Q});
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a2 = 0; a2 < Q; ++a2) cell.data()[a2] = b_val(h, i, k, a2);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

  // result inner reversed (a2,a1) -> inner extents (Q,P), same total.
  ArenaArr ca = einsum(ah("h,i,k;a1"), bh("h,i,k;a2"), "h,i;a2,a1");
  OwnArr co = einsum(ao("h,i,k;a1"), bo("h,i,k;a2"), "h,i;a2,a1");
  world.gop.fence();

  BOOST_REQUIRE_EQUAL(ca.trange().elements_range().volume(),
                      static_cast<std::size_t>(H * I));
  BOOST_REQUIRE_EQUAL(co.trange().elements_range().volume(),
                      static_cast<std::size_t>(H * I));

  double max_abs_diff = 0.0;
  std::size_t elements_compared = 0;
  std::size_t result_outer_cells_seen = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    const bool a_here = ca.is_local(ord) && !ca.is_zero(ord);
    const bool o_here = co.is_local(ord) && !co.is_zero(ord);
    BOOST_REQUIRE_EQUAL(a_here, o_here);
    if (!a_here) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    BOOST_REQUIRE_EQUAL(at.range().volume(), ot.range().volume());
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      BOOST_REQUIRE_EQUAL(bool(ac), !oc.empty());
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), oc.size());
      // inner is now (a2,a1) = (Q,P), same total as P*Q.
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(Q * P));
      ++result_outer_cells_seen;
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double d = std::abs(ac.data()[e] - oc.data()[e]);
        if (d > max_abs_diff) max_abs_diff = d;
        ++elements_compared;
      }
    }
  }
  world.gop.sum(result_outer_cells_seen);
  world.gop.sum(elements_compared);
  world.gop.max(max_abs_diff);
  BOOST_REQUIRE_GT(elements_compared, 0u);
  BOOST_CHECK_EQUAL(result_outer_cells_seen, static_cast<std::size_t>(H * I));
  BOOST_CHECK_EQUAL(elements_compared,
                    static_cast<std::size_t>(H * I) * Q * P);
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
}

#ifdef TA_STRIDED_DGEMM_COUNT
BOOST_AUTO_TEST_CASE(hce_e_uses_strided_path) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();

  // Hadamard mode h tiled with a single block of 2 elements => the einsum
  // driver folds h into nbatch == 2, exercising the ce+e nbatch loop.
  constexpr long H = 2, I = 2, J = 2, K = 3, P = 3, Q = 4;
  TA::TiledRange a_trange{{0l, H}, {0l, I}, {0l, J}};  // (h,i,j)
  TA::TiledRange b_trange{{0l, H}, {0l, K}, {0l, J}};  // (h,k,j)

  auto a_val = [](long h, long i, long j, long p) {
    return 1.0 + 0.5 * i + 0.25 * j + 0.125 * p + 0.0625 * h;
  };
  auto b_val = [](long h, long k, long j, long q) {
    return 2.0 - 0.3 * k + 0.2 * j + 0.05 * q + 0.03 * h;
  };

  // Both operands are plain (nbatch==1) ToT tiles over the full (h,i,j)/(h,k,j)
  // outer range; the einsum driver folds the Hadamard mode h into nbatch==2.
  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{P}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long h = static_cast<long>(o / (I * J));
      const long i = static_cast<long>((o / J) % I);
      const long j = static_cast<long>(o % J);
      for (long p = 0; p < P; ++p) c.data()[p] = a_val(h, i, j, p);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long h = static_cast<long>(o / (K * J));
      const long k = static_cast<long>((o / J) % K);
      const long j = static_cast<long>(o % J);
      for (long q = 0; q < Q; ++q) c.data()[q] = b_val(h, k, j, q);
    }
    return t;
  });
  world.gop.fence();

  TiledArray::detail::g_strided_dgemm_ce_e_calls.store(0);
  auto ca = einsum(ah("h,i,j;p"), bh("h,k,j;q"), "h,i,k;p,q");
  ca.world().gop.fence();
  BOOST_CHECK_GT(TiledArray::detail::g_strided_dgemm_ce_e_calls.load(), 0u);
}

// Regime A (hc+e) firing witness: same shape/fill as
// regime_a_hce_e_arena_matches_owning; the within-tile contraction over k must
// be routed through the strided ce+e core (counter > 0).
BOOST_AUTO_TEST_CASE(regime_a_hce_e_uses_strided_path) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();

  constexpr long H = 2, I = 2, P = 3, Q = 4;
  TA::TiledRange a_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I},
                          TA::TiledRange1{0l, 2, 5}};  // outer (h,i,k)
  TA::TiledRange b_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I},
                          TA::TiledRange1{0l, 2, 5}};  // outer (h,i,k)

  auto a_val = [](long h, long i, long k, long a1) {
    return 1.0 + 0.5 * i + 0.25 * k + 0.125 * a1 + 0.0625 * h;
  };
  auto b_val = [](long h, long i, long k, long a2) {
    return 2.0 - 0.3 * k + 0.2 * i + 0.05 * a2 + 0.03 * h;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{P}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a1 = 0; a1 < P; ++a1) c.data()[a1] = a_val(h, i, k, a1);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t /*ord*/) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a2 = 0; a2 < Q; ++a2) c.data()[a2] = b_val(h, i, k, a2);
    }
    return t;
  });
  world.gop.fence();

  TiledArray::detail::g_strided_dgemm_ce_e_calls.store(0);
  auto ca = einsum(ah("h,i,k;a1"), bh("h,i,k;a2"), "h,i;a1,a2");
  ca.world().gop.fence();
  BOOST_CHECK_GT(TiledArray::detail::g_strided_dgemm_ce_e_calls.load(), 0u);
}

// Codex must-fix #2 guard: an OWNING ToT contraction (TA::Tensor inner, NOT a
// view) must stay on the generic per-cell path -- the strided op is never
// installed (is_tensor_view_v is false), so neither counter moves. This pins
// the install-gate symmetry fix: the gate now requires view+double on ALL
// THREE operands, so a non-view operand can never instantiate the
// double-view-only kernel (which would otherwise be a hard compile error, not a
// fallback). That this translation unit compiles with owning ToT contractions
// present is itself the compile-time half of the guard.
BOOST_AUTO_TEST_CASE(owning_tot_does_not_use_strided_path) {
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();
  constexpr long H = 2, I = 2, J = 2, K = 3, P = 3, Q = 4;
  TA::TiledRange a_trange{{0l, H}, {0l, I}, {0l, J}};  // (h,i,j)
  TA::TiledRange b_trange{{0l, H}, {0l, K}, {0l, J}};  // (h,k,j)
  auto a_val = [](long h, long i, long j, long p) {
    return 1.0 + 0.5 * i + 0.25 * j + 0.125 * p + 0.0625 * h;
  };
  auto b_val = [](long h, long k, long j, long q) {
    return 2.0 - 0.3 * k + 0.2 * j + 0.05 * q + 0.03 * h;
  };
  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{P});
      const long h = static_cast<long>(o / (I * J));
      const long i = static_cast<long>((o / J) % I);
      const long j = static_cast<long>(o % J);
      for (long p = 0; p < P; ++p) cell.data()[p] = a_val(h, i, j, p);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{Q});
      const long h = static_cast<long>(o / (K * J));
      const long k = static_cast<long>((o / J) % K);
      const long j = static_cast<long>(o % J);
      for (long q = 0; q < Q; ++q) cell.data()[q] = b_val(h, k, j, q);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

  TiledArray::detail::g_strided_dgemm_ce_e_calls.store(0);
  TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  auto co = einsum(ao("h,i,j;p"), bo("h,k,j;q"), "h,i,k;p,q");
  co.world().gop.fence();
  // owning inner cells never take the view-only strided path
  BOOST_CHECK_EQUAL(TiledArray::detail::g_strided_dgemm_ce_e_calls.load(), 0u);
  BOOST_CHECK_EQUAL(TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.load(), 0u);
  // sanity: the contraction actually ran and produced the expected outer shape
  BOOST_CHECK_EQUAL(co.trange().elements_range().volume(),
                    static_cast<std::size_t>(H * I * K));
}
#endif

// Addition A (no-Hadamard, multi-external inner): both operands carry 2 inner
// external modes; the result inner cell is the flat outer product.
// c(i,k;a1,a2,a3,a4) = sum_j a(i,j;a1,a2) * b(k,j;a3,a4). Arena vs owning.
BOOST_AUTO_TEST_CASE(ce_e_multi_external_arena_matches_owning) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();
  constexpr long I = 2, J = 2, K = 2;
  constexpr long A1 = 2, A2 = 3, A3 = 2, A4 = 2;
  constexpr long P = A1 * A2, Q = A3 * A4;

  TA::TiledRange a_trange{{0l, I}, {0l, J}};
  TA::TiledRange b_trange{{0l, K}, {0l, J}};

  auto a_val = [](long i, long j, long e) {
    return 1.0 + 0.5 * i + 0.25 * j + 0.1 * e;
  };
  auto b_val = [](long k, long j, long e) {
    return 2.0 - 0.3 * k + 0.2 * j + 0.07 * e;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{A1, A2}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long i = static_cast<long>(o / J), j = static_cast<long>(o % J);
      for (long e = 0; e < P; ++e) c.data()[e] = a_val(i, j, e);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{A3, A4}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long k = static_cast<long>(o / J), j = static_cast<long>(o % J);
      for (long e = 0; e < Q; ++e) c.data()[e] = b_val(k, j, e);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{A1, A2});
      const long i = static_cast<long>(o / J), j = static_cast<long>(o % J);
      for (long e = 0; e < P; ++e) cell.data()[e] = a_val(i, j, e);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{A3, A4});
      const long k = static_cast<long>(o / J), j = static_cast<long>(o % J);
      for (long e = 0; e < Q; ++e) cell.data()[e] = b_val(k, j, e);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  TiledArray::detail::g_strided_dgemm_ce_e_calls.store(0);
#endif
  ArenaArr ca = einsum(ah("i,j;a1,a2"), bh("k,j;a3,a4"), "i,k;a1,a2,a3,a4");
  OwnArr co = einsum(ao("i,j;a1,a2"), bo("k,j;a3,a4"), "i,k;a1,a2,a3,a4");
  world.gop.fence();

  double max_abs_diff = 0.0;
  std::size_t elements_compared = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    if (!(ca.is_local(ord) && !ca.is_zero(ord))) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(P * Q));
      BOOST_REQUIRE_EQUAL(ac.size(), oc.size());
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double d = std::abs(ac.data()[e] - oc.data()[e]);
        if (d > max_abs_diff) max_abs_diff = d;
        ++elements_compared;
      }
    }
  }
  world.gop.sum(elements_compared);
  world.gop.max(max_abs_diff);
  BOOST_REQUIRE_GT(elements_compared, 0u);
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
#ifdef TA_STRIDED_DGEMM_COUNT
  BOOST_CHECK_GT(TiledArray::detail::g_strided_dgemm_ce_e_calls.load(), 0u);
#endif
}

// Addition A (Hadamard nbatch>1, multi-external inner): the case that ABORTS
// on $SRC today (nbatch==1 assert). h is a single tile of extent 2 => folds to
// nbatch==2. c(h,i,k;a1,a2,a3,a4) = sum_j a(h,i,j;a1,a2) * b(h,k,j;a3,a4).
BOOST_AUTO_TEST_CASE(hce_e_multi_external_arena_matches_owning) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();
  constexpr long H = 2, I = 2, J = 2, K = 2;
  constexpr long A1 = 2, A2 = 2, A3 = 2, A4 = 2;
  constexpr long P = A1 * A2, Q = A3 * A4;

  TA::TiledRange a_trange{{0l, H}, {0l, I}, {0l, J}};
  TA::TiledRange b_trange{{0l, H}, {0l, K}, {0l, J}};

  auto a_val = [](long h, long i, long j, long e) {
    return 1.0 + 0.5 * i + 0.25 * j + 0.1 * e + 0.0625 * h;
  };
  auto b_val = [](long h, long k, long j, long e) {
    return 2.0 - 0.3 * k + 0.2 * j + 0.07 * e + 0.03 * h;
  };

  // Both operands are plain (nbatch==1) ToT tiles over the full outer range;
  // the einsum driver folds the Hadamard mode h into nbatch==2.
  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{A1, A2}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long h = static_cast<long>(o / (I * J));
      const long i = static_cast<long>((o / J) % I), j = static_cast<long>(o % J);
      for (long e = 0; e < P; ++e) c.data()[e] = a_val(h, i, j, e);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{A3, A4}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long h = static_cast<long>(o / (K * J));
      const long k = static_cast<long>((o / J) % K), j = static_cast<long>(o % J);
      for (long e = 0; e < Q; ++e) c.data()[e] = b_val(h, k, j, e);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{A1, A2});
      const long h = static_cast<long>(o / (I * J));
      const long i = static_cast<long>((o / J) % I), j = static_cast<long>(o % J);
      for (long e = 0; e < P; ++e) cell.data()[e] = a_val(h, i, j, e);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{A3, A4});
      const long h = static_cast<long>(o / (K * J));
      const long k = static_cast<long>((o / J) % K), j = static_cast<long>(o % J);
      for (long e = 0; e < Q; ++e) cell.data()[e] = b_val(h, k, j, e);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  TiledArray::detail::g_strided_dgemm_ce_e_calls.store(0);
#endif
  ArenaArr ca =
      einsum(ah("h,i,j;a1,a2"), bh("h,k,j;a3,a4"), "h,i,k;a1,a2,a3,a4");
  OwnArr co =
      einsum(ao("h,i,j;a1,a2"), bo("h,k,j;a3,a4"), "h,i,k;a1,a2,a3,a4");
  world.gop.fence();

  double max_abs_diff = 0.0;
  std::size_t elements_compared = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    if (!(ca.is_local(ord) && !ca.is_zero(ord))) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(P * Q));
      BOOST_REQUIRE_EQUAL(ac.size(), oc.size());
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double d = std::abs(ac.data()[e] - oc.data()[e]);
        if (d > max_abs_diff) max_abs_diff = d;
        ++elements_compared;
      }
    }
  }
  world.gop.sum(elements_compared);
  world.gop.max(max_abs_diff);
  BOOST_REQUIRE_GT(elements_compared, 0u);
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
#ifdef TA_STRIDED_DGEMM_COUNT
  BOOST_CHECK_GT(TiledArray::detail::g_strided_dgemm_ce_e_calls.load(), 0u);
#endif
}

// --------------------------------------------------------------------------
// strided-DGEMM ce+ce (hce+ce) e2e oracles
// --------------------------------------------------------------------------

// c(i,m;a1) = sum_{k,a4} a(i,k;a1,a4) * b(i,m,k;a4): Hadamard i (single tile of
// 2 elements => nbatch==2), right-external m, outer-contracted k, no
// left-external (Mo==1). Arena vs owning, elementwise.
BOOST_AUTO_TEST_CASE(hce_ce_contraction_arena_matches_owning) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();
  constexpr long I = 2, M = 3, K = 2, P = 4, Q = 5;

  TA::TiledRange a_trange{{0l, I}, {0l, K}};           // outer (i,k)
  TA::TiledRange b_trange{{0l, I}, {0l, M}, {0l, K}};  // outer (i,m,k)

  auto a_val = [](long i, long k, long a1, long a4) {
    return 1.0 + 0.5 * i + 0.25 * k + 0.125 * a1 + 0.0625 * a4;
  };
  auto b_val = [](long i, long m, long k, long a4) {
    return 2.0 - 0.3 * i + 0.2 * m + 0.1 * k + 0.05 * a4;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{P, Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long i = static_cast<long>(o / K), k = static_cast<long>(o % K);
      for (long a1 = 0; a1 < P; ++a1)
        for (long a4 = 0; a4 < Q; ++a4)
          c.data()[a1 * Q + a4] = a_val(i, k, a1, a4);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long i = static_cast<long>(o / (M * K));
      const long m = static_cast<long>((o / K) % M);
      const long k = static_cast<long>(o % K);
      for (long a4 = 0; a4 < Q; ++a4) c.data()[a4] = b_val(i, m, k, a4);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{P, Q});
      const long i = static_cast<long>(o / K), k = static_cast<long>(o % K);
      for (long a1 = 0; a1 < P; ++a1)
        for (long a4 = 0; a4 < Q; ++a4)
          cell.data()[a1 * Q + a4] = a_val(i, k, a1, a4);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{Q});
      const long i = static_cast<long>(o / (M * K));
      const long m = static_cast<long>((o / K) % M);
      const long k = static_cast<long>(o % K);
      for (long a4 = 0; a4 < Q; ++a4) cell.data()[a4] = b_val(i, m, k, a4);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

  ArenaArr ca = einsum(ah("i,k;a1,a4"), bh("i,m,k;a4"), "i,m;a1");
  OwnArr co = einsum(ao("i,k;a1,a4"), bo("i,m,k;a4"), "i,m;a1");
  world.gop.fence();

  BOOST_REQUIRE_EQUAL(ca.trange().elements_range().volume(),
                      static_cast<std::size_t>(I * M));

  double max_abs_diff = 0.0;
  std::size_t elements_compared = 0, result_outer_cells_seen = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    const bool a_here = ca.is_local(ord) && !ca.is_zero(ord);
    const bool o_here = co.is_local(ord) && !co.is_zero(ord);
    BOOST_REQUIRE_EQUAL(a_here, o_here);
    if (!a_here) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      BOOST_REQUIRE_EQUAL(bool(ac), !oc.empty());
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(P));
      ++result_outer_cells_seen;
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double d = std::abs(ac.data()[e] - oc.data()[e]);
        if (d > max_abs_diff) max_abs_diff = d;
        ++elements_compared;
      }
    }
  }
  world.gop.sum(result_outer_cells_seen);
  world.gop.sum(elements_compared);
  world.gop.max(max_abs_diff);
  BOOST_CHECK_EQUAL(result_outer_cells_seen, static_cast<std::size_t>(I * M));
  BOOST_CHECK_EQUAL(elements_compared, static_cast<std::size_t>(I * M * P));
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
}

#ifdef TA_STRIDED_DGEMM_COUNT
BOOST_AUTO_TEST_CASE(hce_ce_uses_strided_path) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();
  constexpr long I = 2, M = 3, K = 2, P = 4, Q = 5;
  TA::TiledRange a_trange{{0l, I}, {0l, K}};
  TA::TiledRange b_trange{{0l, I}, {0l, M}, {0l, K}};

  auto a_val = [](long i, long k, long a1, long a4) {
    return 1.0 + 0.5 * i + 0.25 * k + 0.125 * a1 + 0.0625 * a4;
  };
  auto b_val = [](long i, long m, long k, long a4) {
    return 2.0 - 0.3 * i + 0.2 * m + 0.1 * k + 0.05 * a4;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{P, Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long i = static_cast<long>(o / K), k = static_cast<long>(o % K);
      for (long a1 = 0; a1 < P; ++a1)
        for (long a4 = 0; a4 < Q; ++a4)
          c.data()[a1 * Q + a4] = a_val(i, k, a1, a4);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long i = static_cast<long>(o / (M * K));
      const long m = static_cast<long>((o / K) % M);
      const long k = static_cast<long>(o % K);
      for (long a4 = 0; a4 < Q; ++a4) c.data()[a4] = b_val(i, m, k, a4);
    }
    return t;
  });
  world.gop.fence();

  TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  auto ca = einsum(ah("i,k;a1,a4"), bh("i,m,k;a4"), "i,m;a1");
  ca.world().gop.fence();
  BOOST_CHECK_GT(TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.load(), 0u);
}
#endif

// No-Hadamard ce+ce (nbatch==1) generality oracle. c(m;a1) = sum_{k,a4}
// a(k;a1,a4) * b(m,k;a4). The strided fused core must fire (witnessed under
// the counter) -- otherwise this oracle would pass even with the install gone.
BOOST_AUTO_TEST_CASE(ce_ce_no_hadamard_arena_matches_owning) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();
  constexpr long M = 3, K = 2, P = 4, Q = 5;

  TA::TiledRange a_trange{{0l, K}};
  TA::TiledRange b_trange{{0l, M}, {0l, K}};

  auto a_val = [](long k, long a1, long a4) {
    return 1.0 + 0.25 * k + 0.125 * a1 + 0.0625 * a4;
  };
  auto b_val = [](long m, long k, long a4) {
    return 2.0 + 0.2 * m + 0.1 * k + 0.05 * a4;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{P, Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long k = static_cast<long>(o);
      for (long a1 = 0; a1 < P; ++a1)
        for (long a4 = 0; a4 < Q; ++a4) c.data()[a1 * Q + a4] = a_val(k, a1, a4);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long m = static_cast<long>(o / K), k = static_cast<long>(o % K);
      for (long a4 = 0; a4 < Q; ++a4) c.data()[a4] = b_val(m, k, a4);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{P, Q});
      const long k = static_cast<long>(o);
      for (long a1 = 0; a1 < P; ++a1)
        for (long a4 = 0; a4 < Q; ++a4) cell.data()[a1 * Q + a4] = a_val(k, a1, a4);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{Q});
      const long m = static_cast<long>(o / K), k = static_cast<long>(o % K);
      for (long a4 = 0; a4 < Q; ++a4) cell.data()[a4] = b_val(m, k, a4);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
#endif
  ArenaArr ca = einsum(ah("k;a1,a4"), bh("m,k;a4"), "m;a1");
  OwnArr co = einsum(ao("k;a1,a4"), bo("m,k;a4"), "m;a1");
  world.gop.fence();
#ifdef TA_STRIDED_DGEMM_COUNT
  BOOST_CHECK_GT(TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.load(), 0u);
#endif

  double max_abs_diff = 0.0;
  std::size_t elements_compared = 0, result_outer_cells_seen = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    const bool a_here = ca.is_local(ord) && !ca.is_zero(ord);
    const bool o_here = co.is_local(ord) && !co.is_zero(ord);
    BOOST_REQUIRE_EQUAL(a_here, o_here);
    if (!a_here) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(P));
      ++result_outer_cells_seen;
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double d = std::abs(ac.data()[e] - oc.data()[e]);
        if (d > max_abs_diff) max_abs_diff = d;
        ++elements_compared;
      }
    }
  }
  world.gop.sum(result_outer_cells_seen);
  world.gop.max(max_abs_diff);
  BOOST_CHECK_EQUAL(result_outer_cells_seen, static_cast<std::size_t>(M));
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
}

// LEFT-clean ce+ce (either-side generalization): the LEFT operand inner is a
// pure contraction vector a(m,k;a4); the RIGHT carries the inner external
// b(k,n;a4,b1); result inner is b1. The left-oriented strided core must fire.
//   c(m,n;b1) = sum_{k,a4} a(m,k;a4) * b(k,n;a4,b1)
BOOST_AUTO_TEST_CASE(ce_ce_left_clean_arena_matches_owning) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();
  constexpr long M = 3, N = 2, K = 2, Q = 5, P = 4;  // a4=Q, b1=P

  TA::TiledRange a_trange{{0l, M}, {0l, K}};  // (m,k)
  TA::TiledRange b_trange{{0l, K}, {0l, N}};  // (k,n)

  auto a_val = [](long m, long k, long a4) {
    return 1.0 + 0.25 * m + 0.125 * k + 0.0625 * a4;
  };
  auto b_val = [](long k, long n, long a4, long b1) {
    return 2.0 + 0.2 * k + 0.1 * n + 0.05 * a4 + 0.025 * b1;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long m = static_cast<long>(o / K), k = static_cast<long>(o % K);
      for (long a4 = 0; a4 < Q; ++a4) c.data()[a4] = a_val(m, k, a4);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{Q, P}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long k = static_cast<long>(o / N), n = static_cast<long>(o % N);
      for (long a4 = 0; a4 < Q; ++a4)
        for (long b1 = 0; b1 < P; ++b1)
          c.data()[a4 * P + b1] = b_val(k, n, a4, b1);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{Q});
      const long m = static_cast<long>(o / K), k = static_cast<long>(o % K);
      for (long a4 = 0; a4 < Q; ++a4) cell.data()[a4] = a_val(m, k, a4);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{Q, P});
      const long k = static_cast<long>(o / N), n = static_cast<long>(o % N);
      for (long a4 = 0; a4 < Q; ++a4)
        for (long b1 = 0; b1 < P; ++b1)
          cell.data()[a4 * P + b1] = b_val(k, n, a4, b1);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  TiledArray::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
#endif
  ArenaArr ca = einsum(ah("m,k;a4"), bh("k,n;a4,b1"), "m,n;b1");
  OwnArr co = einsum(ao("m,k;a4"), bo("k,n;a4,b1"), "m,n;b1");
  world.gop.fence();
#ifdef TA_STRIDED_DGEMM_COUNT
  BOOST_CHECK_GT(TiledArray::detail::g_strided_dgemm_ce_ce_left_calls.load(), 0u);
#endif

  double max_abs_diff = 0.0;
  std::size_t result_outer_cells_seen = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    const bool a_here = ca.is_local(ord) && !ca.is_zero(ord);
    const bool o_here = co.is_local(ord) && !co.is_zero(ord);
    BOOST_REQUIRE_EQUAL(a_here, o_here);
    if (!a_here) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(P));
      ++result_outer_cells_seen;
      for (std::size_t e = 0; e < ac.size(); ++e)
        max_abs_diff =
            std::max(max_abs_diff, std::abs(ac.data()[e] - oc.data()[e]));
    }
  }
  world.gop.sum(result_outer_cells_seen);
  world.gop.max(max_abs_diff);
  BOOST_CHECK_EQUAL(result_outer_cells_seen, static_cast<std::size_t>(M * N));
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
}

// GENERAL ce+ce (matrix x matrix inner): BOTH operands carry an inner external
// (a1 on the left, b1 on the right) alongside the inner contraction a4. Not
// strided-castable; NEITHER the right- nor left-oriented strided core may fire,
// and the generic per-cell path must still match the owning oracle.
//   c(m,n;a1,b1) = sum_{k,a4} a(m,k;a1,a4) * b(k,n;a4,b1)
BOOST_AUTO_TEST_CASE(ce_ce_general_matrix_matrix_not_strided) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();
  constexpr long M = 3, N = 2, K = 2, P1 = 3, Q = 4, P2 = 5;  // a1,a4,b1

  TA::TiledRange a_trange{{0l, M}, {0l, K}};  // (m,k)
  TA::TiledRange b_trange{{0l, K}, {0l, N}};  // (k,n)

  auto a_val = [](long m, long k, long a1, long a4) {
    return 1.0 + 0.25 * m + 0.125 * k + 0.0625 * a1 + 0.03 * a4;
  };
  auto b_val = [](long k, long n, long a4, long b1) {
    return 2.0 + 0.2 * k + 0.1 * n + 0.05 * a4 + 0.025 * b1;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{P1, Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long m = static_cast<long>(o / K), k = static_cast<long>(o % K);
      for (long a1 = 0; a1 < P1; ++a1)
        for (long a4 = 0; a4 < Q; ++a4)
          c.data()[a1 * Q + a4] = a_val(m, k, a1, a4);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{Q, P2}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long k = static_cast<long>(o / N), n = static_cast<long>(o % N);
      for (long a4 = 0; a4 < Q; ++a4)
        for (long b1 = 0; b1 < P2; ++b1)
          c.data()[a4 * P2 + b1] = b_val(k, n, a4, b1);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{P1, Q});
      const long m = static_cast<long>(o / K), k = static_cast<long>(o % K);
      for (long a1 = 0; a1 < P1; ++a1)
        for (long a4 = 0; a4 < Q; ++a4)
          cell.data()[a1 * Q + a4] = a_val(m, k, a1, a4);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{Q, P2});
      const long k = static_cast<long>(o / N), n = static_cast<long>(o % N);
      for (long a4 = 0; a4 < Q; ++a4)
        for (long b1 = 0; b1 < P2; ++b1)
          cell.data()[a4 * P2 + b1] = b_val(k, n, a4, b1);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TiledArray::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
#endif
  ArenaArr ca = einsum(ah("m,k;a1,a4"), bh("k,n;a4,b1"), "m,n;a1,b1");
  OwnArr co = einsum(ao("m,k;a1,a4"), bo("k,n;a4,b1"), "m,n;a1,b1");
  world.gop.fence();
#ifdef TA_STRIDED_DGEMM_COUNT
  // matrix x matrix inner: neither strided orientation may fire.
  BOOST_CHECK_EQUAL(TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.load(),
                    0u);
  BOOST_CHECK_EQUAL(TiledArray::detail::g_strided_dgemm_ce_ce_left_calls.load(),
                    0u);
#endif

  double max_abs_diff = 0.0;
  std::size_t result_outer_cells_seen = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    const bool a_here = ca.is_local(ord) && !ca.is_zero(ord);
    const bool o_here = co.is_local(ord) && !co.is_zero(ord);
    BOOST_REQUIRE_EQUAL(a_here, o_here);
    if (!a_here) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(P1 * P2));
      ++result_outer_cells_seen;
      for (std::size_t e = 0; e < ac.size(); ++e)
        max_abs_diff =
            std::max(max_abs_diff, std::abs(ac.data()[e] - oc.data()[e]));
    }
  }
  world.gop.sum(result_outer_cells_seen);
  world.gop.max(max_abs_diff);
  BOOST_CHECK_EQUAL(result_outer_cells_seen, static_cast<std::size_t>(M * N));
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
}

// Addition B (left-external + Hadamard): c(h,i,k;a1) = sum_{j,a4}
// a(h,i,j;a1,a4) * b(h,j,k;a4). h Hadamard (nbatch>1), i left-external (Mo>1),
// j outer-contracted, k right-external. On $SRC this falls to the per-cell
// path (left external rejected); after Addition B it must fire AND match.
BOOST_AUTO_TEST_CASE(hce_ce_left_external_arena_matches_owning) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();
  constexpr long H = 2, I = 2, J = 2, KK = 3, P = 4, Q = 5;

  TA::TiledRange a_trange{{0l, H}, {0l, I}, {0l, J}};   // (h,i,j)
  TA::TiledRange b_trange{{0l, H}, {0l, J}, {0l, KK}};  // (h,j,k)

  auto a_val = [](long h, long i, long j, long a1, long a4) {
    return 1.0 + 0.5 * h + 0.3 * i + 0.2 * j + 0.1 * a1 + 0.05 * a4;
  };
  auto b_val = [](long h, long j, long k, long a4) {
    return 2.0 - 0.4 * h + 0.25 * j + 0.15 * k + 0.07 * a4;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{P, Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long h = static_cast<long>(o / (I * J));
      const long i = static_cast<long>((o / J) % I), j = static_cast<long>(o % J);
      for (long a1 = 0; a1 < P; ++a1)
        for (long a4 = 0; a4 < Q; ++a4)
          c.data()[a1 * Q + a4] = a_val(h, i, j, a1, a4);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const long h = static_cast<long>(o / (J * KK));
      const long j = static_cast<long>((o / KK) % J), k = static_cast<long>(o % KK);
      for (long a4 = 0; a4 < Q; ++a4) c.data()[a4] = b_val(h, j, k, a4);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{P, Q});
      const long h = static_cast<long>(o / (I * J));
      const long i = static_cast<long>((o / J) % I), j = static_cast<long>(o % J);
      for (long a1 = 0; a1 < P; ++a1)
        for (long a4 = 0; a4 < Q; ++a4)
          cell.data()[a1 * Q + a4] = a_val(h, i, j, a1, a4);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      OwnInner cell(TA::Range{Q});
      const long h = static_cast<long>(o / (J * KK));
      const long j = static_cast<long>((o / KK) % J), k = static_cast<long>(o % KK);
      for (long a4 = 0; a4 < Q; ++a4) cell.data()[a4] = b_val(h, j, k, a4);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
#endif
  ArenaArr ca = einsum(ah("h,i,j;a1,a4"), bh("h,j,k;a4"), "h,i,k;a1");
  OwnArr co = einsum(ao("h,i,j;a1,a4"), bo("h,j,k;a4"), "h,i,k;a1");
  world.gop.fence();
#ifdef TA_STRIDED_DGEMM_COUNT
  BOOST_CHECK_GT(TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.load(), 0u);
#endif

  double max_abs_diff = 0.0;
  std::size_t elements_compared = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    if (!(ca.is_local(ord) && !ca.is_zero(ord))) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(P));
      BOOST_REQUIRE_EQUAL(ac.size(), oc.size());
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double d = std::abs(ac.data()[e] - oc.data()[e]);
        if (d > max_abs_diff) max_abs_diff = d;
        ++elements_compared;
      }
    }
  }
  world.gop.sum(elements_compared);
  world.gop.max(max_abs_diff);
  BOOST_REQUIRE_GT(elements_compared, 0u);
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
}

// --------------------------------------------------------------------------
// strided-DGEMM combined equivalence matrices (REVISION: multi-external +
// left-external + nbatch>1, multi-tile contracted index, edge sizes).
// --------------------------------------------------------------------------

// ce+e combined: Hadamard h (nbatch>1) + MULTI-TILE contracted j + multi-
// external inner on both operands. c(h,i,k;a1,a2,a3) = sum_j a(h,i,j;a1,a2) *
// b(h,k,j;a3). j tiled as two blocks {2},{1} so the contracted run spans
// multiple tiles (the SUMMA K-panel stream). Arena vs owning + firing witness.
BOOST_AUTO_TEST_CASE(hce_e_combined_equivalence_strided) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();
  constexpr long H = 2, I = 2, KK = 2, A1 = 2, A2 = 2, A3 = 2;
  constexpr long P = A1 * A2, Q = A3, J = 3;  // J = 2+1 elements over two tiles

  TA::TiledRange a_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I},
                          TA::TiledRange1{0l, 2l, 3l}};  // (h,i,j)
  TA::TiledRange b_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, KK},
                          TA::TiledRange1{0l, 2l, 3l}};  // (h,k,j)

  auto a_val = [](long h, long i, long j, long e) {
    return 1.0 + 0.5 * h + 0.3 * i + 0.2 * j + 0.1 * e;
  };
  auto b_val = [](long h, long k, long j, long e) {
    return 2.0 - 0.4 * h + 0.25 * k + 0.15 * j + 0.07 * e;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{A1, A2}; });
    const auto& r = t.range();
    for (std::size_t o = 0; o < r.volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      auto idx = r.idx(o);  // (h,i,j) global
      for (long e = 0; e < P; ++e) c.data()[e] = a_val(idx[0], idx[1], idx[2], e);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{A3}; });
    const auto& r = t.range();
    for (std::size_t o = 0; o < r.volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      auto idx = r.idx(o);  // (h,k,j) global
      for (long e = 0; e < Q; ++e) c.data()[e] = b_val(idx[0], idx[1], idx[2], e);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    const auto& r = t.range();
    for (std::size_t o = 0; o < r.volume(); ++o) {
      OwnInner cell(TA::Range{A1, A2});
      auto idx = r.idx(o);
      for (long e = 0; e < P; ++e) cell.data()[e] = a_val(idx[0], idx[1], idx[2], e);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    const auto& r = t.range();
    for (std::size_t o = 0; o < r.volume(); ++o) {
      OwnInner cell(TA::Range{A3});
      auto idx = r.idx(o);
      for (long e = 0; e < Q; ++e) cell.data()[e] = b_val(idx[0], idx[1], idx[2], e);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  TiledArray::detail::g_strided_dgemm_ce_e_calls.store(0);
#endif
  ArenaArr ca = einsum(ah("h,i,j;a1,a2"), bh("h,k,j;a3"), "h,i,k;a1,a2,a3");
  OwnArr co = einsum(ao("h,i,j;a1,a2"), bo("h,k,j;a3"), "h,i,k;a1,a2,a3");
  world.gop.fence();
#ifdef TA_STRIDED_DGEMM_COUNT
  BOOST_CHECK_GT(TiledArray::detail::g_strided_dgemm_ce_e_calls.load(), 0u);
#endif

  double max_abs_diff = 0.0;
  std::size_t elements_compared = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    if (!(ca.is_local(ord) && !ca.is_zero(ord))) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(P * Q));
      BOOST_REQUIRE_EQUAL(ac.size(), oc.size());
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double d = std::abs(ac.data()[e] - oc.data()[e]);
        if (d > max_abs_diff) max_abs_diff = d;
        ++elements_compared;
      }
    }
  }
  world.gop.sum(elements_compared);
  world.gop.max(max_abs_diff);
  BOOST_REQUIRE_GT(elements_compared, 0u);
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
  (void)J;
}

// ce+ce combined: Hadamard h (nbatch>1) + left-external i (Mo>1) + MULTI-TILE
// contracted j + multi-external kept-inner a1,a2 (P>1) and contracted-inner a4.
// c(h,i,k;a1,a2) = sum_{j,a4} a(h,i,j;a1,a2,a4) * b(h,j,k;a4). Arena vs owning
// + firing witness.
BOOST_AUTO_TEST_CASE(hce_ce_combined_equivalence_strided) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;
  using OwnInner = TA::Tensor<double>;
  using OwnOuter = TA::Tensor<OwnInner>;
  using OwnArr = TA::DistArray<OwnOuter, TA::DensePolicy>;

  auto& world = TiledArray::get_default_world();
  constexpr long H = 2, I = 2, KK = 2, A1 = 2, A2 = 2, A4 = 3;
  constexpr long P = A1 * A2, Q = A4;

  TA::TiledRange a_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I},
                          TA::TiledRange1{0l, 2l, 3l}};  // (h,i,j)
  TA::TiledRange b_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, 2l, 3l},
                          TA::TiledRange1{0l, KK}};  // (h,j,k)

  auto a_val = [](long h, long i, long j, long e) {
    return 1.0 + 0.5 * h + 0.3 * i + 0.2 * j + 0.1 * e;
  };
  auto b_val = [](long h, long j, long k, long e) {
    return 2.0 - 0.4 * h + 0.25 * j + 0.15 * k + 0.07 * e;
  };

  ArenaArr ah(world, a_trange);
  ah.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{A1, A2, A4}; });
    const auto& r = t.range();
    for (std::size_t o = 0; o < r.volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      auto idx = r.idx(o);  // (h,i,j)
      for (long e = 0; e < P * Q; ++e)
        c.data()[e] = a_val(idx[0], idx[1], idx[2], e);
    }
    return t;
  });
  ArenaArr bh(world, b_trange);
  bh.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{A4}; });
    const auto& r = t.range();
    for (std::size_t o = 0; o < r.volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      auto idx = r.idx(o);  // (h,j,k)
      for (long e = 0; e < Q; ++e) c.data()[e] = b_val(idx[0], idx[1], idx[2], e);
    }
    return t;
  });
  world.gop.fence();

  OwnArr ao(world, a_trange);
  ao.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    const auto& r = t.range();
    for (std::size_t o = 0; o < r.volume(); ++o) {
      OwnInner cell(TA::Range{A1, A2, A4});
      auto idx = r.idx(o);
      for (long e = 0; e < P * Q; ++e) cell.data()[e] = a_val(idx[0], idx[1], idx[2], e);
      t.data()[o] = cell;
    }
    return t;
  });
  OwnArr bo(world, b_trange);
  bo.init_tiles([&](const TA::Range& tr) {
    OwnOuter t(tr);
    const auto& r = t.range();
    for (std::size_t o = 0; o < r.volume(); ++o) {
      OwnInner cell(TA::Range{A4});
      auto idx = r.idx(o);
      for (long e = 0; e < Q; ++e) cell.data()[e] = b_val(idx[0], idx[1], idx[2], e);
      t.data()[o] = cell;
    }
    return t;
  });
  world.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
#endif
  ArenaArr ca = einsum(ah("h,i,j;a1,a2,a4"), bh("h,j,k;a4"), "h,i,k;a1,a2");
  OwnArr co = einsum(ao("h,i,j;a1,a2,a4"), bo("h,j,k;a4"), "h,i,k;a1,a2");
  world.gop.fence();
#ifdef TA_STRIDED_DGEMM_COUNT
  BOOST_CHECK_GT(TiledArray::detail::g_strided_dgemm_ce_ce_right_calls.load(), 0u);
#endif

  double max_abs_diff = 0.0;
  std::size_t elements_compared = 0;
  const auto& tiles = ca.trange().tiles_range();
  for (std::size_t ord = 0; ord < tiles.volume(); ++ord) {
    if (!(ca.is_local(ord) && !ca.is_zero(ord))) continue;
    const ArenaOuter& at = ca.find(ord).get();
    const OwnOuter& ot = co.find(ord).get();
    for (std::size_t o = 0; o < at.range().volume(); ++o) {
      const ArenaInner& ac = at.data()[o];
      const OwnInner& oc = ot.data()[o];
      if (!ac) continue;
      BOOST_REQUIRE_EQUAL(ac.size(), static_cast<std::size_t>(P));
      BOOST_REQUIRE_EQUAL(ac.size(), oc.size());
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double d = std::abs(ac.data()[e] - oc.data()[e]);
        if (d > max_abs_diff) max_abs_diff = d;
        ++elements_compared;
      }
    }
  }
  world.gop.sum(elements_compared);
  world.gop.max(max_abs_diff);
  BOOST_REQUIRE_GT(elements_compared, 0u);
  BOOST_CHECK_LT(max_abs_diff, 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()  // einsum_tot

BOOST_AUTO_TEST_SUITE(einsum_tot_t)

BOOST_AUTO_TEST_CASE(ilkj_nm_eq_ij_mn_times_kl) {
  using t_type = DistArray<Tensor<double>, SparsePolicy>;
  using tot_type = DistArray<Tensor<Tensor<double>>, SparsePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {49, 73, 28, 46, 12, 83, 29, 61, 61, 98, 57, 28, 96, 57});
  Tensor<double> lhs_elem_0_1(
      Range{7, 2}, {78, 15, 69, 55, 87, 94, 28, 94, 79, 30, 26, 88, 48, 74});
  Tensor<double> lhs_elem_1_0(
      Range{7, 2}, {70, 32, 25, 71, 6, 56, 4, 13, 72, 50, 15, 95, 52, 89});
  Tensor<double> lhs_elem_1_1(
      Range{7, 2}, {12, 29, 17, 68, 37, 79, 5, 52, 13, 35, 53, 54, 78, 71});
  Tensor<double> lhs_elem_2_0(
      Range{7, 2}, {77, 39, 34, 94, 16, 82, 63, 27, 75, 12, 14, 59, 3, 14});
  Tensor<double> lhs_elem_2_1(
      Range{7, 2}, {65, 90, 37, 41, 65, 75, 59, 16, 44, 85, 86, 11, 40, 24});
  Tensor<double> lhs_elem_3_0(
      Range{7, 2}, {77, 53, 11, 6, 99, 63, 46, 68, 83, 56, 76, 86, 91, 79});
  Tensor<double> lhs_elem_3_1(
      Range{7, 2}, {56, 11, 33, 90, 36, 38, 33, 54, 60, 21, 16, 28, 6, 97});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1},
                   {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1},
                   {lhs_elem_3_0, lhs_elem_3_1}};
  TiledRange lhs_trange{{0, 2, 4}, {0, 2}};
  tot_type lhs(world, lhs_trange, lhs_il);

  TiledRange rhs_trange{{0, 2}, {0, 2, 4, 6}};
  t_type rhs(world, rhs_trange);
  rhs.fill_random();

  TiledRange ref_result_trange{lhs_trange.dim(0), rhs_trange.dim(1),
                               rhs_trange.dim(0), lhs_trange.dim(1)};
  tot_type ref_result(world, ref_result_trange);

  //
  // i,l,k,j;n,m = i,j;m,n * k,l
  //

  lhs.make_replicated();
  rhs.make_replicated();
  world.gop.fence();

  // why cannot lhs and rhs be captured by ref?
  auto make_tile = [lhs, rhs](TA::Range const& rng) {
    tot_type::value_type result_tile{rng};
    for (auto&& res_ix : result_tile.range()) {
      auto i = res_ix[0];
      auto l = res_ix[1];
      auto k = res_ix[2];
      auto j = res_ix[3];

      auto lhs_tile_ix = lhs.trange().element_to_tile({i, j});
      auto lhs_tile = lhs.find_local(lhs_tile_ix).get(/* dowork = */ false);

      auto rhs_tile_ix = rhs.trange().element_to_tile({k, l});
      auto rhs_tile = rhs.find_local(rhs_tile_ix).get(/* dowork = */ false);

      auto& res_el = result_tile({i, l, k, j});
      auto const& lhs_el = lhs_tile({i, j});
      auto rhs_el = rhs_tile({k, l});

      res_el = tot_type::element_type(
          lhs_el.scale(rhs_el),            // scale
          TiledArray::Permutation{1, 0});  // permute [0,1] -> [1,0]
    }
    return result_tile;
  };

  using std::begin;
  using std::end;

  for (auto it = begin(ref_result); it != end(ref_result); ++it) {
    if (ref_result.is_local(it.index())) {
      auto tile = TA::get_default_world().taskq.add(make_tile, it.make_range());
      *it = tile;
    }
  }

  tot_type result;
  BOOST_REQUIRE_NO_THROW(result("i,l,k,j;n,m") = lhs("i,j;m,n") * rhs("k,l"));

  const bool are_equal =
      ToTArrayFixture::are_equal<ShapeComp::False>(result, ref_result);
  BOOST_CHECK(are_equal);

  {  // reverse the order
    tot_type result;
    BOOST_REQUIRE_NO_THROW(result("i,l,k,j;n,m") = rhs("k,l") * lhs("i,j;m,n"));
    const bool are_equal =
        ToTArrayFixture::are_equal<ShapeComp::False>(result, ref_result);
    BOOST_CHECK(are_equal);
  }
}

BOOST_AUTO_TEST_CASE(ijk_mn_eq_ij_mn_times_jk) {
  using t_type = DistArray<Tensor<double>, SparsePolicy>;
  using tot_type = DistArray<Tensor<Tensor<double>>, SparsePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {49, 73, 28, 46, 12, 83, 29, 61, 61, 98, 57, 28, 96, 57});
  Tensor<double> lhs_elem_0_1(
      Range{7, 2}, {78, 15, 69, 55, 87, 94, 28, 94, 79, 30, 26, 88, 48, 74});
  Tensor<double> lhs_elem_1_0(
      Range{7, 2}, {70, 32, 25, 71, 6, 56, 4, 13, 72, 50, 15, 95, 52, 89});
  Tensor<double> lhs_elem_1_1(
      Range{7, 2}, {12, 29, 17, 68, 37, 79, 5, 52, 13, 35, 53, 54, 78, 71});
  Tensor<double> lhs_elem_2_0(
      Range{7, 2}, {77, 39, 34, 94, 16, 82, 63, 27, 75, 12, 14, 59, 3, 14});
  Tensor<double> lhs_elem_2_1(
      Range{7, 2}, {65, 90, 37, 41, 65, 75, 59, 16, 44, 85, 86, 11, 40, 24});
  Tensor<double> lhs_elem_3_0(
      Range{7, 2}, {77, 53, 11, 6, 99, 63, 46, 68, 83, 56, 76, 86, 91, 79});
  Tensor<double> lhs_elem_3_1(
      Range{7, 2}, {56, 11, 33, 90, 36, 38, 33, 54, 60, 21, 16, 28, 6, 97});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1},
                   {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1},
                   {lhs_elem_3_0, lhs_elem_3_1}};
  TiledRange lhs_trange{{0, 2, 4}, {0, 2}};
  tot_type lhs(world, lhs_trange, lhs_il);

  TiledRange rhs_trange{{0, 2}, {0, 2, 4, 6}};
  t_type rhs(world, rhs_trange);
  rhs.fill_random();

  // i,j;m,n * j,k => i,j,k;m,n
  TiledRange ref_result_trange{lhs_trange.dim(0), rhs_trange.dim(0),
                               rhs_trange.dim(1)};
  tot_type ref_result(world, ref_result_trange);

  lhs.make_replicated();
  rhs.make_replicated();
  world.gop.fence();

  //
  // why cannot lhs and rhs be captured by ref?
  //
  auto make_tile = [lhs, rhs](TA::Range const& rng) {
    tot_type::value_type result_tile{rng};
    for (auto&& res_ix : result_tile.range()) {
      auto i = res_ix[0];
      auto j = res_ix[1];
      auto k = res_ix[2];

      auto lhs_tile_ix = lhs.trange().element_to_tile({i, j});
      auto lhs_tile = lhs.find_local(lhs_tile_ix).get(/* dowork = */ false);

      auto rhs_tile_ix = rhs.trange().element_to_tile({j, k});
      auto rhs_tile = rhs.find_local(rhs_tile_ix).get(/* dowork = */ false);

      auto& res_el = result_tile({i, j, k});
      auto const& lhs_el = lhs_tile({i, j});
      auto rhs_el = rhs_tile({j, k});

      res_el = lhs_el.scale(rhs_el);
    }
    return result_tile;
  };

  using std::begin;
  using std::end;

  for (auto it = begin(ref_result); it != end(ref_result); ++it) {
    if (ref_result.is_local(it.index())) {
      auto tile = TA::get_default_world().taskq.add(make_tile, it.make_range());
      *it = tile;
    }
  }

  /////////////////////////////////////////////////////////
  // ToT * T

  // this is not supported by the expression layer since this is a
  // - general product w.r.t. outer indices
  // - involves ToT * T
  // tot_type result;
  // BOOST_REQUIRE_NO_THROW(result("i,j,k;m,n") = lhs("i,j;m,n") * rhs("j,k"));

  // will try to make this work
  tot_type result = einsum(lhs("i,j;m,n"), rhs("j,k"), "i,j,k;m,n");
  bool are_equal =
      ToTArrayFixture::are_equal<ShapeComp::False>(result, ref_result);
  BOOST_REQUIRE(are_equal);
  {
    result = einsum(rhs("j,k"), lhs("i,j;m,n"), "i,j,k;m,n");
    are_equal =
        ToTArrayFixture::are_equal<ShapeComp::False>(result, ref_result);
    BOOST_REQUIRE(are_equal);
  }
}

BOOST_AUTO_TEST_CASE(ij_mn_eq_ji_mn_times_ij) {
  using t_type = DistArray<Tensor<double>, SparsePolicy>;
  using tot_type = DistArray<Tensor<Tensor<double>>, SparsePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {49, 73, 28, 46, 12, 83, 29, 61, 61, 98, 57, 28, 96, 57});
  Tensor<double> lhs_elem_0_1(
      Range{7, 2}, {78, 15, 69, 55, 87, 94, 28, 94, 79, 30, 26, 88, 48, 74});
  Tensor<double> lhs_elem_1_0(
      Range{7, 2}, {70, 32, 25, 71, 6, 56, 4, 13, 72, 50, 15, 95, 52, 89});
  Tensor<double> lhs_elem_1_1(
      Range{7, 2}, {12, 29, 17, 68, 37, 79, 5, 52, 13, 35, 53, 54, 78, 71});
  Tensor<double> lhs_elem_2_0(
      Range{7, 2}, {77, 39, 34, 94, 16, 82, 63, 27, 75, 12, 14, 59, 3, 14});
  Tensor<double> lhs_elem_2_1(
      Range{7, 2}, {65, 90, 37, 41, 65, 75, 59, 16, 44, 85, 86, 11, 40, 24});
  Tensor<double> lhs_elem_3_0(
      Range{7, 2}, {77, 53, 11, 6, 99, 63, 46, 68, 83, 56, 76, 86, 91, 79});
  Tensor<double> lhs_elem_3_1(
      Range{7, 2}, {56, 11, 33, 90, 36, 38, 33, 54, 60, 21, 16, 28, 6, 97});
  Tensor<double> lhs_elem_4_0(
      Range{7, 2}, {77, 53, 11, 6, 99, 63, 46, 68, 83, 56, 76, 86, 91, 79});
  Tensor<double> lhs_elem_4_1(
      Range{7, 2}, {56, 11, 33, 90, 36, 38, 33, 54, 60, 21, 16, 28, 6, 97});
  Tensor<double> lhs_elem_5_0(
      Range{7, 2}, {77, 53, 11, 6, 99, 63, 46, 68, 83, 56, 76, 86, 91, 79});
  Tensor<double> lhs_elem_5_1(
      Range{7, 2}, {56, 11, 33, 90, 36, 38, 33, 54, 60, 21, 16, 28, 6, 97});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1}, {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1}, {lhs_elem_3_0, lhs_elem_3_1},
                   {lhs_elem_4_0, lhs_elem_4_1}, {lhs_elem_5_0, lhs_elem_5_1}};
  TiledRange lhs_trange{{0, 2, 6}, {0, 2}};
  tot_type lhs(world, lhs_trange, lhs_il);

  TiledRange rhs_trange{{0, 2}, {0, 2, 6}};
  t_type rhs(world, rhs_trange);
  rhs.fill_random();

  //
  // i,j;m,n = j,i;n,m * i,j
  //
  TiledRange ref_result_trange{rhs_trange.dim(0), rhs_trange.dim(1)};
  tot_type ref_result(world, ref_result_trange);

  lhs.make_replicated();
  rhs.make_replicated();
  world.gop.fence();

  // why cannot lhs and rhs be captured by ref?
  auto make_tile = [lhs, rhs](TA::Range const& rng) {
    tot_type::value_type result_tile{rng};
    for (auto&& res_ix : result_tile.range()) {
      auto i = res_ix[0];
      auto j = res_ix[1];

      auto lhs_tile_ix = lhs.trange().element_to_tile({j, i});
      auto lhs_tile = lhs.find_local(lhs_tile_ix).get(/* dowork */ false);

      auto rhs_tile_ix = rhs.trange().element_to_tile({i, j});
      auto rhs_tile = rhs.find_local(rhs_tile_ix).get(/* dowork */ false);

      auto& res_el = result_tile({i, j});
      auto const& lhs_el = lhs_tile({j, i});
      auto rhs_el = rhs_tile({i, j});
      res_el = tot_type::element_type(lhs_el.scale(rhs_el),          // scale
                                      TiledArray::Permutation{0, 1}  // permute
      );
    }
    return result_tile;
  };

  using std::begin;
  using std::end;

  for (auto it = begin(ref_result); it != end(ref_result); ++it) {
    if (ref_result.is_local(it.index())) {
      auto tile = TA::get_default_world().taskq.add(make_tile, it.make_range());
      *it = tile;
    }
  }

  tot_type result;
  BOOST_REQUIRE_NO_THROW(result("i,j;m,n") = lhs("j,i;m,n") * rhs("i,j"));

  const bool are_equal =
      ToTArrayFixture::are_equal<ShapeComp::False>(result, ref_result);
  BOOST_CHECK(are_equal);
}

BOOST_AUTO_TEST_SUITE_END()  // einsum_tot_t

// Eigen einsum indices
BOOST_AUTO_TEST_SUITE(einsum_index, TA_UT_LABEL_SERIAL)

#include "TiledArray/einsum/index.h"
#include "TiledArray/tensor/tensor.h"

BOOST_AUTO_TEST_CASE(einsum_index) {
  using ::Einsum::Index;
  Index<int> src({1, 2, 3, 4, 5, 6});
  Index<int> dst({2, 4, 5, 3, 6, 1});
  auto p = permutation(src, dst);
  BOOST_CHECK(p.size() == 6);
  BOOST_CHECK(apply(p, src) == dst);
  BOOST_CHECK(apply_inverse(p, dst) = src);
  using TiledArray::Range;
  using TiledArray::Tensor;
  Tensor<double> t(Range{src});
  BOOST_CHECK((t.range().rank() == 6));
  auto u = TiledArray::apply(p, t);
  BOOST_CHECK((u.range() == Range{dst}));
  auto v = TiledArray::apply_inverse(p, u);
  BOOST_CHECK((v.range() == Range{src}));
}

BOOST_AUTO_TEST_SUITE_END()  // einsum_index

#include "TiledArray/einsum/eigen.h"

template <typename... Tensors>
using common_scalar_t = std::common_type_t<typename Tensors::Scalar...>;

template <typename T>
auto abs_comparison_threshold_default() {
  if constexpr (std::is_integral_v<T>) {
    return 0;
  } else {
    return 1e-4;
  }
}

template <typename TA, typename TB>
bool isApprox(const Eigen::TensorBase<TA, Eigen::ReadOnlyAccessors>& A,
              const Eigen::TensorBase<TB, Eigen::ReadOnlyAccessors>& B,
              common_scalar_t<TA, TB> abs_comparison_threshold =
                  abs_comparison_threshold_default<common_scalar_t<TA, TB>>()) {
  Eigen::Tensor<bool, 0> r;
  if constexpr (std::is_integral_v<typename TA::Scalar> &&
                std::is_integral_v<typename TB::Scalar>) {
// Eigen::TensorBase::operator== is ambiguously defined in C++20
#if __cplusplus >= 202002L
    r = ((derived(A) - derived(B)).abs() == 0).all();
#else
    r = (derived(A) == derived(B)).all();
#endif
  } else {  // soft floating-point comparison
    r = ((derived(A) - derived(B)).abs() <= abs_comparison_threshold).all();
  }
  return r.coeffRef();
}

// Eigen einsum expressions
BOOST_AUTO_TEST_SUITE(einsum_eigen, TA_UT_LABEL_SERIAL)

template <typename T = int, typename... Args>
auto random(Args... args) {
  Eigen::Tensor<T, sizeof...(Args)> t(args...);
  t.setRandom();
  return t;
}

template <typename T, int NA, int NB, size_t NI, size_t NC>
void einsum_eigen_contract_check(Eigen::Tensor<T, NA> A, Eigen::Tensor<T, NB> B,
                                 std::string expr,
                                 std::array<std::pair<int, int>, NI> i,
                                 std::array<int, NC> p) {
  using Eigen::einsum;
  using std::tuple;
  static_assert(NC == NA + NB - 2 * NI);
  auto C = Eigen::einsum<Eigen::Tensor<T, NC>>(expr, A, B);
  BOOST_CHECK(isApprox(C, A.contract(B, i).shuffle(p)));
}

template <typename T, int NA, int NB, size_t NI, size_t NC>
void einsum_eigen_hadamard_check(
    Eigen::Tensor<T, NA> A, Eigen::Tensor<T, NB> B, std::string expr,
    std::array<std::pair<int, int>, NI> i,  // contracted pairs
    std::array<std::pair<int, int>, 1> h,   // hadamard pairs
    std::array<int, NC> p) {
  static_assert(NC == NA + NB - 2 * NI - 1);
  auto [ha, hb] = h[0];
  size_t nh = A.dimension(ha);
  size_t hc = ha;
  // decrement internal indices above h-index
  for (auto& [ia, ib] : i) {
    if (ia < ha) --hc;
    if (ia > ha) ia -= 1;
    if (ib > hb) ib -= 1;
  }
  // shuffle C to A*B order
  std::vector<int> p_ab(p.size());
  for (size_t i = 0; i < p.size(); ++i) {
    p_ab.at(p[i]) = i;
  }
  auto C = Eigen::einsum<Eigen::Tensor<T, NC>>(expr, A, B);
  // validate h-dims
  eigen_assert(nh == A.dimension(ha));
  eigen_assert(nh == B.dimension(hb));
  eigen_assert(nh == C.dimension(p_ab.at(hc)));
  // validate
  for (size_t h = 0; h < nh; ++h) {
    auto ah = A.chip(h, ha);
    auto bh = B.chip(h, hb);
    auto result = C.shuffle(p_ab).chip(h, hc);
    auto expected = ah.contract(bh, i);
    BOOST_CHECK(isApprox(result, expected));
  }
}

BOOST_AUTO_TEST_CASE(einsum_eigen_ak_bk_ab) {
  using std::array;
  using std::pair;
  einsum_eigen_contract_check(random(11, 7), random(13, 7), "ak,bk->ab",
                              array{pair{1, 1}}, array{0, 1});
}

BOOST_AUTO_TEST_CASE(einsum_eigen_ka_bk_ba) {
  using std::array;
  using std::pair;
  einsum_eigen_contract_check(random(7, 11), random(13, 7), "ka,bk->ba",
                              array{pair{0, 1}}, array{1, 0});
}

BOOST_AUTO_TEST_CASE(einsum_eigen_abi_cdi_cdab) {
  using std::array;
  using std::pair;
  einsum_eigen_contract_check(random(21, 22, 3), random(24, 25, 3),
                              "abi,cdi->cdab", array{pair{2, 2}},
                              array{2, 3, 0, 1});
}

BOOST_AUTO_TEST_CASE(einsum_eigen_icd_ai_abcd) {
  using std::array;
  using std::pair;
  einsum_eigen_contract_check(random(3, 12, 13), random(14, 15, 3),
                              "icd,bai->abcd", array{pair{0, 2}},
                              array{3, 2, 0, 1});
}

BOOST_AUTO_TEST_CASE(einsum_eigen_cdji_ibja_abcd) {
  using std::array;
  using std::pair;
  einsum_eigen_contract_check(random(14, 15, 3, 5), random(5, 12, 3, 13),
                              "cdji,ibja->abcd", array{pair{3, 0}, pair{2, 2}},
                              array{3, 2, 0, 1});
}

BOOST_AUTO_TEST_CASE(einsum_eigen_hai_hbi_hab) {
  using std::array;
  using std::pair;
  einsum_eigen_hadamard_check(random(7, 14, 3), random(7, 15, 3),
                              "hai,hbi->hab", array{pair{2, 2}},
                              array{pair{0, 0}}, array{0, 1, 2});
}

BOOST_AUTO_TEST_CASE(einsum_eigen_iah_hib_bha) {
  using std::array;
  using std::pair;
  einsum_eigen_hadamard_check(random(7, 14, 3), random(3, 7, 15),
                              "iah,hib->bha", array{pair{0, 1}},
                              array{pair{2, 0}}, array{2, 1, 0});
}

BOOST_AUTO_TEST_CASE(einsum_eigen_iah_hib_abh) {
  using std::array;
  using std::pair;
  einsum_eigen_hadamard_check(random(7, 14, 3), random(3, 7, 15),
                              "iah,hib->abh", array{pair{0, 1}},
                              array{pair{2, 0}}, array{0, 2, 1});
}

BOOST_AUTO_TEST_CASE(einsum_eigen_hi_hi_h) {
  using std::array;
  using std::pair;
  einsum_eigen_hadamard_check(random(7, 14), random(7, 14), "hi,hi->h",
                              array{pair{1, 1}}, array{pair{0, 0}}, array{0});
}

BOOST_AUTO_TEST_CASE(einsum_eigen_hji_jih_hj) {
  using std::array;
  using T = int;
  Eigen::Index nh = 3, nj = 5, ni = 4;
  auto A = random<T>(nh, nj, ni);
  auto B = random<T>(nj, ni, nh);
  using R = Eigen::Tensor<T, 2>;
  R result = einsum<R>("hji,jih->hj", A, B);
  R reference =
      einsum<Eigen::Tensor<T, 1>>(
          "hi,hi->h", A.shuffle(array{0, 1, 2}).reshape(array{nh * nj, ni}),
          B.shuffle(array{2, 0, 1}).reshape(array{nh * nj, ni}))
          .reshape(array{nh, nj});
  BOOST_CHECK(isApprox(reference, result));
}

BOOST_AUTO_TEST_SUITE_END()  // einsum_eigen

// TiledArray einsum expressions
BOOST_AUTO_TEST_SUITE(einsum_tiledarray)

using TiledArray::DensePolicy;
using TiledArray::SparsePolicy;

auto tiled_range(std::initializer_list<int> dims) {
  std::vector<TiledRange1> tr;
  for (int n : dims) {
    auto tr1 = (n < 2) ? TiledRange1{0, n} : TiledRange1{0, n / 2, n};
    // tr1 = TiledRange1{ 0, n };
    tr.push_back(tr1);
  }
  return TiledRange(tr);
}

template <typename Policy, typename T = Tensor<int>, typename... Args>
auto random(Args... args) {
  auto tr = tiled_range({args...});
  auto& world = TiledArray::get_default_world();
  TiledArray::DistArray<T, Policy> t(world, tr);
  t.fill_random();
  return t;
}

template <typename T = Tensor<int>, typename... Args>
auto sparse_zero(Args... args) {
  auto tr = tiled_range({args...});
  auto& world = TiledArray::get_default_world();
  TiledArray::SparsePolicy::shape_type shape(0.0f, tr);
  TiledArray::DistArray<T, TiledArray::SparsePolicy> t(world, tr, shape);
  t.fill(0);
  return t;
}

template <int NA, int NB, int NC, typename T, typename Policy>
void einsum_tiledarray_check(TiledArray::DistArray<T, Policy>&& A,
                             TiledArray::DistArray<T, Policy>&& B,
                             std::string expr) {
  using Eigen::Tensor;
  using U = typename T::value_type;
  using TC = Tensor<U, NC>;
  auto C = einsum(expr, A, B);
  BOOST_CHECK(rank(C) == NC);
  A.make_replicated();
  B.make_replicated();
  C.make_replicated();
  auto reference = einsum<TC>(expr, array_to_eigen_tensor<Tensor<U, NA>>(A),
                              array_to_eigen_tensor<Tensor<U, NB>>(B));
  auto result = array_to_eigen_tensor<TC>(C);
  // std::cout << "e=" << result << std::endl;
  BOOST_CHECK(isApprox(result, reference));
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_ak_bk_ab) {
  einsum_tiledarray_check<2, 2, 2>(random<SparsePolicy>(11, 7),
                                   random<SparsePolicy>(13, 7), "ak,bk->ab");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_ka_bk_ba) {
  einsum_tiledarray_check<2, 2, 2>(random<SparsePolicy>(7, 11),
                                   random<SparsePolicy>(13, 7), "ka,bk->ba");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_abi_cdi_cdab) {
  einsum_tiledarray_check<3, 3, 4>(random<SparsePolicy>(21, 22, 3),
                                   random<SparsePolicy>(24, 25, 3),
                                   "abi,cdi->cdab");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_icd_bai_abcd) {
  einsum_tiledarray_check<3, 3, 4>(random<SparsePolicy>(3, 12, 13),
                                   random<SparsePolicy>(14, 15, 3),
                                   "icd,bai->abcd");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_cdji_ibja_abcd) {
  einsum_tiledarray_check<4, 4, 4>(random<SparsePolicy>(14, 15, 3, 5),
                                   random<SparsePolicy>(5, 12, 3, 13),
                                   "cdji,ibja->abcd");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_hai_hbi_hab) {
  einsum_tiledarray_check<3, 3, 3>(random<SparsePolicy>(7, 14, 3),
                                   random<SparsePolicy>(7, 15, 3),
                                   "hai,hbi->hab");
  einsum_tiledarray_check<3, 3, 3>(sparse_zero(7, 14, 3), sparse_zero(7, 15, 3),
                                   "hai,hbi->hab");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_iah_hib_bha) {
  einsum_tiledarray_check<3, 3, 3>(random<SparsePolicy>(7, 14, 3),
                                   random<SparsePolicy>(3, 7, 15),
                                   "iah,hib->bha");
  einsum_tiledarray_check<3, 3, 3>(sparse_zero(7, 14, 3), sparse_zero(3, 7, 15),
                                   "iah,hib->bha");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_iah_hib_abh) {
  einsum_tiledarray_check<3, 3, 3>(random<SparsePolicy>(7, 14, 3),
                                   random<SparsePolicy>(3, 7, 15),
                                   "iah,hib->abh");
  einsum_tiledarray_check<3, 3, 3>(sparse_zero(7, 14, 3), sparse_zero(3, 7, 15),
                                   "iah,hib->abh");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_hiab_hci_habc) {
  einsum_tiledarray_check<4, 3, 4>(random<SparsePolicy>(9, 5, 7, 8),
                                   random<SparsePolicy>(9, 11, 5),
                                   "hiab,hci->habc");
  einsum_tiledarray_check<4, 3, 4>(sparse_zero(9, 5, 7, 8),
                                   sparse_zero(9, 11, 5), "hiab,hci->habc");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_hi_hi_h) {
  einsum_tiledarray_check<2, 2, 1>(random<SparsePolicy>(7, 14),
                                   random<SparsePolicy>(7, 14), "hi,hi->h");
  einsum_tiledarray_check<2, 2, 1>(sparse_zero(7, 14), sparse_zero(7, 14),
                                   "hi,hi->h");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_hji_jih_hj) {
  einsum_tiledarray_check<3, 3, 2>(random<SparsePolicy>(14, 7, 5),
                                   random<SparsePolicy>(7, 5, 14),
                                   "hji,jih->hj");
  einsum_tiledarray_check<3, 3, 2>(sparse_zero(14, 7, 5), sparse_zero(7, 5, 14),
                                   "hji,jih->hj");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_ik_jk_ijk) {
  einsum_tiledarray_check<2, 2, 3>(random<SparsePolicy>(7, 5),
                                   random<SparsePolicy>(14, 5), "ik,jk->ijk");
  einsum_tiledarray_check<2, 2, 3>(sparse_zero(7, 5), sparse_zero(14, 5),
                                   "ik,jk->ijk");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_replicated) {
  einsum_tiledarray_check<3, 3, 3>(replicated(random<DensePolicy>(7, 14, 3)),
                                   random<DensePolicy>(7, 15, 3),
                                   "hai,hbi->hab");
  einsum_tiledarray_check<3, 3, 3>(random<DensePolicy>(7, 14, 3),
                                   replicated(random<DensePolicy>(7, 15, 3)),
                                   "hai,hbi->hab");
  einsum_tiledarray_check<3, 3, 3>(replicated(random<DensePolicy>(7, 14, 3)),
                                   replicated(random<DensePolicy>(7, 15, 3)),
                                   "hai,hbi->hab");
  einsum_tiledarray_check<2, 2, 1>(replicated(random<SparsePolicy>(7, 14)),
                                   random<SparsePolicy>(7, 14), "hi,hi->h");
  einsum_tiledarray_check<2, 2, 1>(replicated(random<SparsePolicy>(7, 14)),
                                   replicated(random<SparsePolicy>(7, 14)),
                                   "hi,hi->h");
}

BOOST_AUTO_TEST_CASE(einsum_tiledarray_dot) {
  using TiledArray::dot;
  auto hik = random<DensePolicy>(4, 3, 5);
  auto hkj = random<DensePolicy>(4, 5, 6);
  auto hji = random<DensePolicy>(4, 6, 3);
  auto hik_hkj_hji = dot("hik,hkj,hji", hik, hkj, hji);
  auto hji_hik_hkj = dot("hji,hik,hkj", hji, hik, hkj);
  auto hkj_hji_hik = dot("hkj,hji,hik", hkj, hji, hik);
  auto AB = einsum("hik,hkj->hji", hik, hkj);
  BOOST_CHECK(dot(AB, hji) == hik_hkj_hji);
  BOOST_CHECK(hik_hkj_hji == hji_hik_hkj);
  BOOST_CHECK(hik_hkj_hji == hji_hik_hkj);
  BOOST_CHECK(hik_hkj_hji == hkj_hji_hik);
  TiledArray::get_default_world().gop.fence();
}

// BOOST_AUTO_TEST_CASE(einsum_tiledarray_dot) {
//   using TiledArray::dot;
//   auto hik = random<DensePolicy>(4,3,5);
//   auto hkj = random<DensePolicy>(4,5,6);
//   auto hji = random<DensePolicy>(4,6,3);
//   auto hik_hkj_hji = dot("hik,hkj,hji", hik, hkj, hji);
//   auto hji_hik_hkj = dot("hji,hik,hkj", hji, hik, hkj);
//   auto hkj_hji_hik = dot("hkj,hji,hik", hkj, hji, hik);
//   auto AB = einsum("hik,hkj->hji", hik, hkj);
//   BOOST_CHECK(dot(AB,hji) == hik_hkj_hji);
//   BOOST_CHECK(hik_hkj_hji == hji_hik_hkj);
//   BOOST_CHECK(hik_hkj_hji == hji_hik_hkj);
//   BOOST_CHECK(hik_hkj_hji == hkj_hji_hik);
// }

BOOST_AUTO_TEST_SUITE_END()  // einsum_tiledarray
