/// Arena-aware ToT trivial-op end-to-end tests (add, subt, mult, scale, clone).

#include "TiledArray/tensor.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <cstddef>
#include <cstdint>
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

/// All inner cells point into one contiguous slab (monotonic with bounded gap).
bool inners_share_one_slab(const outer_t& tot) {
  if (tot.range().volume() == 0) return true;
  const double* prev_end = nullptr;
  for (std::size_t ord = 0; ord < tot.range().volume(); ++ord) {
    const inner_t& cell = *(tot.data() + ord);
    if (cell.range().volume() == 0) continue;
    const double* cell_begin = cell.data();
    const double* cell_end = cell_begin + cell.range().volume();
    if (prev_end != nullptr && cell_begin < prev_end) return false;
    if (prev_end != nullptr &&
        static_cast<std::size_t>(cell_begin - prev_end) > 1024)
      return false;
    prev_end = cell_end;
  }
  return true;
}

}

BOOST_AUTO_TEST_SUITE(arena_tot_trivial_suite, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(scale_bit_equal_and_one_slab) {
  outer_t src = make_tot(6, 8, 1.0);
  outer_t arena_result = src.scale(2.5);
  outer_t baseline(src.range(), 1);
  for (std::size_t ord = 0; ord < src.range().volume(); ++ord) {
    inner_t inner((src.data() + ord)->range());
    for (std::size_t i = 0; i < inner.range().volume(); ++i)
      inner.at_ordinal(i) = (src.data() + ord)->at_ordinal(i) * 2.5;
    *(baseline.data() + ord) = std::move(inner);
  }
  BOOST_CHECK(tot_equal(arena_result, baseline));
  BOOST_CHECK(inners_share_one_slab(arena_result));
}

BOOST_AUTO_TEST_CASE(clone_bit_equal_and_one_slab) {
  outer_t src = make_tot(6, 8, 3.0);
  outer_t arena_result = src.clone();
  BOOST_CHECK(tot_equal(arena_result, src));
  BOOST_CHECK(inners_share_one_slab(arena_result));
}

BOOST_AUTO_TEST_CASE(add_bit_equal_and_one_slab) {
  outer_t L = make_tot(6, 8, 1.0);
  outer_t R = make_tot(6, 8, 0.5);
  outer_t arena_result = L.add(R);
  outer_t baseline(L.range(), 1);
  for (std::size_t ord = 0; ord < L.range().volume(); ++ord) {
    inner_t inner((L.data() + ord)->range());
    for (std::size_t i = 0; i < inner.range().volume(); ++i)
      inner.at_ordinal(i) = (L.data() + ord)->at_ordinal(i) +
                            (R.data() + ord)->at_ordinal(i);
    *(baseline.data() + ord) = std::move(inner);
  }
  BOOST_CHECK(tot_equal(arena_result, baseline));
  BOOST_CHECK(inners_share_one_slab(arena_result));
}

BOOST_AUTO_TEST_CASE(subt_bit_equal_and_one_slab) {
  outer_t L = make_tot(6, 8, 5.0);
  outer_t R = make_tot(6, 8, 1.0);
  outer_t arena_result = L.subt(R);
  outer_t baseline(L.range(), 1);
  for (std::size_t ord = 0; ord < L.range().volume(); ++ord) {
    inner_t inner((L.data() + ord)->range());
    for (std::size_t i = 0; i < inner.range().volume(); ++i)
      inner.at_ordinal(i) = (L.data() + ord)->at_ordinal(i) -
                            (R.data() + ord)->at_ordinal(i);
    *(baseline.data() + ord) = std::move(inner);
  }
  BOOST_CHECK(tot_equal(arena_result, baseline));
  BOOST_CHECK(inners_share_one_slab(arena_result));
}

BOOST_AUTO_TEST_CASE(mult_elementwise_bit_equal_and_one_slab) {
  outer_t L = make_tot(6, 8, 2.0);
  outer_t R = make_tot(6, 8, 0.5);
  outer_t arena_result = L.mult(R);
  outer_t baseline(L.range(), 1);
  for (std::size_t ord = 0; ord < L.range().volume(); ++ord) {
    inner_t inner((L.data() + ord)->range());
    for (std::size_t i = 0; i < inner.range().volume(); ++i)
      inner.at_ordinal(i) = (L.data() + ord)->at_ordinal(i) *
                            (R.data() + ord)->at_ordinal(i);
    *(baseline.data() + ord) = std::move(inner);
  }
  BOOST_CHECK(tot_equal(arena_result, baseline));
  BOOST_CHECK(inners_share_one_slab(arena_result));
}

BOOST_AUTO_TEST_CASE(arena_outlives_source) {
  outer_t arena_result;
  {
    outer_t src = make_tot(3, 4, 9.0);
    arena_result = src.scale(2.0);
  }
  for (std::size_t ord = 0; ord < arena_result.range().volume(); ++ord)
    for (std::size_t i = 0; i < (arena_result.data() + ord)->range().volume();
         ++i)
      BOOST_CHECK_EQUAL((arena_result.data() + ord)->at_ordinal(i),
                        (9.0 + ord * 100.0 + i) * 2.0);
}

BOOST_AUTO_TEST_SUITE_END()
