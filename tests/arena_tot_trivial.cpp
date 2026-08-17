/// Arena-aware ToT trivial-op end-to-end tests (add, subt, mult, scale, clone).

#include "TiledArray/tensor.h"
#include "TiledArray/tensor/arena_kernels.h"
#include "TiledArray/tensor/arena_tensor.h"
#include "TiledArray/tile.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace TA = TiledArray;
using inner_t = TA::Tensor<double>;
using outer_t = TA::Tensor<inner_t>;
using arena_inner_t = TA::ArenaTensor<double>;
using arena_tot_t = TA::Tensor<arena_inner_t>;

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

/// Build a real arena-backed `Tensor<ArenaTensor<double>>`: outer element e
/// gets a non-uniform-extent inner cell filled with base + 100*e + i.
arena_tot_t make_arena_tot(std::size_t N_outer, double base = 1.0) {
  TA::Range outer_range{static_cast<long>(N_outer)};
  auto inner_range_fn = [](const auto& idx) {
    const long e = static_cast<long>(idx[0]);
    return typename arena_inner_t::range_type{3 + (e % 2)};
  };
  auto fill_fn = [base](arena_inner_t& cell, const auto& idx) {
    const long e = static_cast<long>(idx[0]);
    for (std::size_t i = 0; i < cell.size(); ++i)
      cell.data()[i] = base + e * 100.0 + i;
  };
  return TA::detail::make_nested_tile<arena_tot_t>(outer_range, inner_range_fn,
                                                   fill_fn);
}

}  // namespace

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
      inner.at_ordinal(i) =
          (L.data() + ord)->at_ordinal(i) + (R.data() + ord)->at_ordinal(i);
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
      inner.at_ordinal(i) =
          (L.data() + ord)->at_ordinal(i) - (R.data() + ord)->at_ordinal(i);
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
      inner.at_ordinal(i) =
          (L.data() + ord)->at_ordinal(i) * (R.data() + ord)->at_ordinal(i);
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

// --- mismatched null-inner-cell coverage (non-arena inner) ---------------
// Same kernel (arena_trivial_binary) backs Tensor<Tensor<double>>; exercise
// the union-sparsity / implicit-zero path with mismatched per-cell nulls.
// An unassigned outer cell is a default (empty) inner Tensor.

namespace {

/// `present[ord]==false` leaves cell `ord` a null (empty) inner tensor.
outer_t make_tot_sparse(std::size_t N_outer, std::size_t n_inner, double base,
                        const std::vector<bool>& present) {
  outer_t outer(TA::Range{static_cast<long>(N_outer)}, 1);
  for (std::size_t ord = 0; ord < N_outer; ++ord) {
    if (!present[ord]) continue;  // leave default-constructed -> empty
    inner_t inner(TA::Range{static_cast<long>(n_inner)});
    for (std::size_t i = 0; i < n_inner; ++i)
      inner.at_ordinal(i) = base + ord * 100.0 + i;
    *(outer.data() + ord) = std::move(inner);
  }
  return outer;
}

// 0 = lone-left, 1&2 = both, 3 = both-null, 4 = lone-right.
const std::vector<bool> nz_L{true, true, true, false, false};
const std::vector<bool> nz_R{false, true, true, false, true};

}  // namespace

BOOST_AUTO_TEST_CASE(add_mismatched_null_inners) {
  outer_t L = make_tot_sparse(5, 4, 1.0, nz_L);
  outer_t R = make_tot_sparse(5, 4, 0.5, nz_R);
  outer_t sum = L.add(R);  // must not segfault on lone-left cell 0
  for (std::size_t ord = 0; ord < 5; ++ord) {
    const inner_t& l = *(L.data() + ord);
    const inner_t& r = *(R.data() + ord);
    const inner_t& d = *(sum.data() + ord);
    const bool hl = !l.empty(), hr = !r.empty();
    if (!hl && !hr) {
      BOOST_CHECK(d.empty());
    } else {
      BOOST_REQUIRE(!d.empty());
      for (std::size_t i = 0; i < d.range().volume(); ++i) {
        const double lv = hl ? l.at_ordinal(i) : 0.0;
        const double rv = hr ? r.at_ordinal(i) : 0.0;
        BOOST_CHECK_EQUAL(d.at_ordinal(i), lv + rv);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(subt_mismatched_null_inners) {
  outer_t L = make_tot_sparse(5, 4, 5.0, nz_L);
  outer_t R = make_tot_sparse(5, 4, 1.0, nz_R);
  outer_t diff = L.subt(R);
  for (std::size_t ord = 0; ord < 5; ++ord) {
    const inner_t& l = *(L.data() + ord);
    const inner_t& r = *(R.data() + ord);
    const inner_t& d = *(diff.data() + ord);
    const bool hl = !l.empty(), hr = !r.empty();
    if (!hl && !hr) {
      BOOST_CHECK(d.empty());
    } else {
      BOOST_REQUIRE(!d.empty());
      for (std::size_t i = 0; i < d.range().volume(); ++i) {
        const double lv = hl ? l.at_ordinal(i) : 0.0;
        const double rv = hr ? r.at_ordinal(i) : 0.0;
        BOOST_CHECK_EQUAL(d.at_ordinal(i), lv - rv);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(mult_mismatched_null_inners) {
  outer_t L = make_tot_sparse(5, 4, 2.0, nz_L);
  outer_t R = make_tot_sparse(5, 4, 0.5, nz_R);
  outer_t prod = L.mult(R);
  for (std::size_t ord = 0; ord < 5; ++ord) {
    const inner_t& l = *(L.data() + ord);
    const inner_t& r = *(R.data() + ord);
    const inner_t& d = *(prod.data() + ord);
    if (!l.empty() && !r.empty()) {
      BOOST_REQUIRE(!d.empty());
      for (std::size_t i = 0; i < d.range().volume(); ++i)
        BOOST_CHECK_EQUAL(d.at_ordinal(i), l.at_ordinal(i) * r.at_ordinal(i));
    } else if (!d.empty()) {
      for (std::size_t i = 0; i < d.range().volume(); ++i)
        BOOST_CHECK_EQUAL(d.at_ordinal(i), 0.0);
    }
  }
}

// --- new: Tile<flat scalar> x Tensor<ArenaTensor> cross-type mult --------
// Exercises the new free-function overload in tile.h that lets a flat,
// Tile<T>-wrapped scalar tensor multiply against a real arena-backed ToT
// (Tensor<ArenaTensor<U>>) tensor -- the actual failing case from
// TA::einsum<Tile<CLR>, ArrayToT> (Tile<CLR> plays the role of the flat
// left-hand tile here; TA::Tensor<double> is used as the CPU stand-in,
// since the overload dispatches purely on Tile<Left>::numeric_type, not on
// the specific wrapped tensor type).
BOOST_AUTO_TEST_CASE(mult_flat_tile_times_arena_tot) {
  const std::size_t N_outer = 5;
  arena_tot_t right = make_arena_tot(N_outer, 2.0);

  TA::Tensor<double> scalars(TA::Range{static_cast<long>(N_outer)});
  for (std::size_t e = 0; e < N_outer; ++e)
    scalars.at_ordinal(e) = 1.5 + static_cast<double>(e);
  TA::Tile<TA::Tensor<double>> left(scalars);

  arena_tot_t prod = TA::mult(left, right);

  BOOST_REQUIRE_EQUAL(prod.range().volume(), right.range().volume());
  for (std::size_t ord = 0; ord < N_outer; ++ord) {
    const arena_inner_t& src = right.data()[ord];
    const arena_inner_t& dst = prod.data()[ord];
    BOOST_REQUIRE_EQUAL(dst.size(), src.size());
    const double s = scalars.at_ordinal(ord);
    for (std::size_t i = 0; i < dst.size(); ++i)
      BOOST_CHECK_EQUAL(dst.data()[i], s * src.data()[i]);
  }
}

BOOST_AUTO_TEST_CASE(mult_flat_tile_times_arena_tot_empty_operands) {
  arena_tot_t empty_right;
  TA::Tile<TA::Tensor<double>> nonempty_left(TA::Tensor<double>(TA::Range{3}));
  BOOST_CHECK(TA::mult(nonempty_left, empty_right).empty());

  arena_tot_t nonempty_right = make_arena_tot(3, 1.0);
  TA::Tile<TA::Tensor<double>> empty_left;
  BOOST_CHECK(TA::mult(empty_left, nonempty_right).empty());
}

BOOST_AUTO_TEST_SUITE_END()
