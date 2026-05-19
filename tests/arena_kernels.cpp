/// Unit tests for arena-backed ToT kernels.

#include "TiledArray/tensor/arena_kernels.h"

#include "TiledArray/tensor.h"
#include "TiledArray/tensor/arena.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <cstddef>
#include <vector>

namespace TA = TiledArray;
using inner_t = TA::Tensor<double>;
using outer_t = TA::Tensor<inner_t>;

namespace {

outer_t make_tot(std::size_t N_outer, std::size_t n_inner, double base = 1.0) {
  outer_t outer(TA::Range{static_cast<long>(N_outer)}, 1);
  for (std::size_t ord = 0; ord < N_outer; ++ord) {
    inner_t inner(TA::Range{static_cast<long>(n_inner)});
    for (std::size_t i = 0; i < n_inner; ++i)
      inner.at_ordinal(i) = base + ord * 100.0 + i;
    *(outer.data() + ord) = std::move(inner);
  }
  return outer;
}

bool tot_equal(const outer_t& a, const outer_t& b) {
  if (a.range().volume() != b.range().volume()) return false;
  for (std::size_t ord = 0; ord < a.range().volume(); ++ord) {
    const inner_t& ai = *(a.data() + ord);
    const inner_t& bi = *(b.data() + ord);
    if (ai.range().volume() != bi.range().volume()) return false;
    for (std::size_t i = 0; i < ai.range().volume(); ++i)
      if (ai.at_ordinal(i) != bi.at_ordinal(i)) return false;
  }
  return true;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(arena_kernels_suite, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(trivial_unary_clone_matches_heap_baseline) {
  outer_t src = make_tot(4, 5, 1.0);
  auto fill = [](double* dst, const double* src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = src[i];
  };
  outer_t arena_result = TA::detail::arena_trivial_unary<outer_t>(src, fill);
  BOOST_CHECK(tot_equal(arena_result, src));
}

BOOST_AUTO_TEST_CASE(trivial_unary_scale_matches_heap_baseline) {
  outer_t src = make_tot(4, 5, 1.0);
  const double factor = 2.5;
  auto fill = [factor](double* dst, const double* src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = src[i] * factor;
  };
  outer_t arena_result = TA::detail::arena_trivial_unary<outer_t>(src, fill);
  outer_t baseline(src.range(), 1);
  for (std::size_t ord = 0; ord < src.range().volume(); ++ord) {
    inner_t inner((src.data() + ord)->range());
    for (std::size_t i = 0; i < inner.range().volume(); ++i)
      inner.at_ordinal(i) = (src.data() + ord)->at_ordinal(i) * factor;
    *(baseline.data() + ord) = std::move(inner);
  }
  BOOST_CHECK(tot_equal(arena_result, baseline));
}

BOOST_AUTO_TEST_CASE(trivial_binary_add_matches_heap_baseline) {
  outer_t L = make_tot(4, 5, 1.0);
  outer_t R = make_tot(4, 5, 0.5);
  auto fill = [](double* dst, const double* l, const double* r, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = l[i] + r[i];
  };
  outer_t arena_result = TA::detail::arena_trivial_binary<outer_t>(L, R, fill);
  outer_t baseline(L.range(), 1);
  for (std::size_t ord = 0; ord < L.range().volume(); ++ord) {
    inner_t inner((L.data() + ord)->range());
    for (std::size_t i = 0; i < inner.range().volume(); ++i)
      inner.at_ordinal(i) =
          (L.data() + ord)->at_ordinal(i) + (R.data() + ord)->at_ordinal(i);
    *(baseline.data() + ord) = std::move(inner);
  }
  BOOST_CHECK(tot_equal(arena_result, baseline));
}

BOOST_AUTO_TEST_CASE(arena_outlives_kernel_call) {
  // The result data deleter co-owns the arena.
  outer_t arena_result;
  {
    outer_t src = make_tot(3, 4, 7.0);
    auto fill = [](double* dst, const double* src, std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) dst[i] = src[i];
    };
    arena_result = TA::detail::arena_trivial_unary<outer_t>(src, fill);
  }
  for (std::size_t ord = 0; ord < arena_result.range().volume(); ++ord)
    for (std::size_t i = 0; i < (arena_result.data() + ord)->range().volume();
         ++i)
      BOOST_CHECK_EQUAL((arena_result.data() + ord)->at_ordinal(i),
                        7.0 + ord * 100.0 + i);
}

BOOST_AUTO_TEST_CASE(inner_permute_transposes_each_cell) {
  // outer tile of 3 cells, each a non-uniform r x c inner matrix
  outer_t src(TA::Range{3}, 1);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const long r = 2 + static_cast<long>(ord);      // 2, 3, 4
    const long c = 3 + static_cast<long>(ord % 2);  // 3, 4, 3
    inner_t inner(TA::Range{r, c});
    for (long i = 0; i < r; ++i)
      for (long j = 0; j < c; ++j)
        inner(i, j) = 1.0 + ord * 100.0 + i * 10.0 + j;
    *(src.data() + ord) = std::move(inner);
  }
  auto result =
      TA::detail::arena_inner_permute<outer_t>(src, TA::Permutation{1, 0});
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const inner_t& s = *(src.data() + ord);
    const inner_t& d = *(result.data() + ord);
    const long r = s.range().extent(0);
    const long c = s.range().extent(1);
    BOOST_REQUIRE_EQUAL(d.range().rank(), 2u);
    BOOST_CHECK_EQUAL(d.range().extent(0), c);
    BOOST_CHECK_EQUAL(d.range().extent(1), r);
    for (long i = 0; i < r; ++i)
      for (long j = 0; j < c; ++j) BOOST_CHECK_EQUAL(d(j, i), s(i, j));
  }
}

BOOST_AUTO_TEST_CASE(inner_permute_rank3_cell) {
  outer_t src(TA::Range{2}, 1);
  const long e0 = 2, e1 = 3, e2 = 4;
  for (std::size_t ord = 0; ord < 2; ++ord) {
    inner_t inner(TA::Range{e0, e1, e2});
    for (long a = 0; a < e0; ++a)
      for (long b = 0; b < e1; ++b)
        for (long c = 0; c < e2; ++c)
          inner(a, b, c) = ord * 1000.0 + a * 100.0 + b * 10.0 + c;
    *(src.data() + ord) = std::move(inner);
  }
  // perm {2,0,1}: src dim 0->2, 1->0, 2->1 => result(b,c,a) == src(a,b,c)
  auto result =
      TA::detail::arena_inner_permute<outer_t>(src, TA::Permutation{2, 0, 1});
  for (std::size_t ord = 0; ord < 2; ++ord) {
    const inner_t& s = *(src.data() + ord);
    const inner_t& d = *(result.data() + ord);
    BOOST_CHECK_EQUAL(d.range().extent(0), e1);
    BOOST_CHECK_EQUAL(d.range().extent(1), e2);
    BOOST_CHECK_EQUAL(d.range().extent(2), e0);
    for (long a = 0; a < e0; ++a)
      for (long b = 0; b < e1; ++b)
        for (long c = 0; c < e2; ++c) BOOST_CHECK_EQUAL(d(b, c, a), s(a, b, c));
  }
}

BOOST_AUTO_TEST_SUITE_END()
