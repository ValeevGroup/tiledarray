#pragma once
#include <tiledarray.h>
#include "unit_test_config.h"


template <class A>
static void compare_replicated_vector(const char* context, const A& S_nd, const A& S,
                        double e) {
  // clang-format off
  BOOST_TEST_CONTEXT(context)
  ;
  // clang-format on

  const size_t n = S.size();
  BOOST_REQUIRE_EQUAL(n, S_nd.size());
  for(size_t i = 0; i < n; ++i) {
    BOOST_CHECK_SMALL(std::abs(S[i] - S_nd[i]), e);
  }
}

template <class A>
static void compare_subspace(const char* context, const A& non_dist, const A& result,
                             double e) {

  namespace TA = TiledArray;
  // clang-format off
  BOOST_TEST_CONTEXT(context)
  ;
  // clang-format on

  auto nd_eigen = TA::array_to_eigen(non_dist);
  auto rs_eigen = TA::array_to_eigen(result);

  Eigen::MatrixXd G; G = nd_eigen.adjoint() * rs_eigen;
  Eigen::MatrixXd G2; G2 = G.adjoint() * G; // Accounts for phase-flips
  const auto n = G.rows();
  auto G2_mI_nrm = (G2 - Eigen::MatrixXd::Identity(n,n)).norm();
  BOOST_CHECK_SMALL(G2_mI_nrm, e);
}

template <class A>
static void compare_eig(const char* context, const A& non_dist, const A& result,
                        double e) {
  // clang-format off
  BOOST_TEST_CONTEXT(context)
  ;
  // clang-format on

  auto [evals_nd, evecs_nd] = non_dist;
  auto [evals,    evecs   ] = result;

  compare_replicated_vector(context, evals_nd, evals, e);

  // The test problem for the unit tests has a non-degenerate spectrum
  // we only need to check for phase-flips in this check
  evecs.make_replicated(); // Need to be replicated for Eigen conversion
  evecs_nd.make_replicated();
  compare_subspace(context, evecs_nd, evecs, e);
}

template <TA::SVD::Vectors Vectors, class A>
static void compare_svd(const char* context, const A& non_dist, const A& result,
                        double e) {
  namespace TA = TiledArray;

  // clang-format off
  BOOST_TEST_CONTEXT(context)
  ;
  // clang-format on

  if constexpr (Vectors == TA::SVD::ValuesOnly) {
    compare_replicated_vector(context, non_dist, result, e);
    return;
  } else {
    const auto& S = std::get<0>(result);
    const auto& S_nd = std::get<0>(non_dist);
    compare_replicated_vector(context, S_nd, S, e);
  }

}
template <class A>
static void compare(const char* context, const A& non_dist, const A& result,
                    double e) {
  // clang-format off
  BOOST_TEST_CONTEXT(context)
  ;
  // clang-format on
  auto diff_with_non_dist = (non_dist("i,j") - result("i,j")).norm().get();
  BOOST_CHECK_SMALL(diff_with_non_dist, e);
}

template <typename T, typename F, int... Is>
static void for_each_pair_of_tuples_impl(T&& t1, T&& t2, F f,
                                         std::integer_sequence<int, Is...>) {
  auto l = {(f(std::get<Is>(t1), std::get<Is>(t2)), 0)...};
}

template <typename... Ts, typename F>
static void for_each_pair_of_tuples(std::tuple<Ts...> const& t1,
                                    std::tuple<Ts...> const& t2, F f) {
  for_each_pair_of_tuples_impl(
      t1, t2, f, std::make_integer_sequence<int, sizeof...(Ts)>());
}

template <class... As>
static void compare(const char* context, const std::tuple<As...>& non_dist,
                    const std::tuple<As...>& result, double e) {
  for_each_pair_of_tuples(non_dist, result, [&](auto& arg1, auto& arg2) {
    compare(context, arg1, arg2, e);
  });
}

