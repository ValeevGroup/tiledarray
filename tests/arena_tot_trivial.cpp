/// Arena-aware ToT trivial-op end-to-end tests (add, subt, mult, scale, clone).

#include "TiledArray/tensor.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
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

// --- null destination cell in an IN-PLACE binary op ----------------------
// `Tensor<ArenaTensor>` inner cells are non-owning views: a null cell has no
// storage and cannot allocate. An in-place `add_to` therefore cannot adopt a
// populated right-hand cell, and silently returning would drop it. The
// in-place ops must detect this and fall back to the value-returning
// (union-sparsity) kernel.

namespace {

using arena_inner_t = TA::ArenaTensor<double, TA::Range>;
using arena_outer_t = TA::Tensor<arena_inner_t>;

// The in-place null-cell fallback is gated on exactly these two traits (see
// `Tensor::binary_needs_view_cell_fallback_v`). If either went false for the
// arena ToT type the fallback would silently vanish, so pin them here.
static_assert(TA::detail::is_tensor_of_tensor_v<arena_outer_t>);
static_assert(TA::is_tensor_view_v<arena_inner_t>);
static_assert(!TA::is_tensor_view_v<TA::Tensor<double>>);

/// `present[ord]==false` requests a zero-volume inner range, which
/// `arena_outer_init` turns into a deliberately-null inner cell.
arena_outer_t make_arena_tot_sparse(std::size_t N_outer, std::size_t n_inner,
                                    double base,
                                    const std::vector<bool>& present) {
  auto range_fn = [&present, n_inner](std::size_t ord) {
    return present[ord] ? TA::Range{static_cast<long>(n_inner)} : TA::Range{};
  };
  arena_outer_t t = TA::detail::arena_outer_init<arena_outer_t>(
      TA::Range{static_cast<long>(N_outer)}, 1, range_fn, alignof(double),
      /*zero_init=*/true);
  for (std::size_t ord = 0; ord < N_outer; ++ord) {
    arena_inner_t& c = t.data()[ord];
    if (c.empty()) continue;
    for (std::size_t i = 0; i < n_inner; ++i)
      c.data()[i] = base + ord * 100.0 + i;
  }
  return t;
}

/// Elementwise check against the implicit-zero (union sparsity) reference:
/// every element of `got` must equal `op(l, r)` with an absent operand read as
/// zero. This checks the *value*, not the representation: a null cell and an
/// explicit zero cell both denote zero, so a null cell in `got` is accepted
/// wherever every expected element is zero (which is what an annihilating
/// `mult` produces for a cell present in only one operand).
template <typename Op>
void check_union(const arena_outer_t& got, const arena_outer_t& L,
                 const arena_outer_t& R, Op op) {
  BOOST_REQUIRE_EQUAL(got.range().volume(), L.range().volume());
  for (std::size_t ord = 0; ord < L.range().volume(); ++ord) {
    const arena_inner_t& l = L.data()[ord];
    const arena_inner_t& r = R.data()[ord];
    const arena_inner_t& d = got.data()[ord];
    const bool hl = !l.empty(), hr = !r.empty();
    if (!hl && !hr) {
      BOOST_CHECK(d.empty());
      continue;
    }
    const std::size_t n = hl ? l.size() : r.size();
    auto expected = [&](std::size_t i) {
      const double lv = hl ? l.data()[i] : 0.0;
      const double rv = hr ? r.data()[i] : 0.0;
      return op(lv, rv);
    };
    if (d.empty()) {
      // a null cell is the sparse spelling of zero -- fine iff zero is right
      for (std::size_t i = 0; i < n; ++i) BOOST_CHECK_EQUAL(0.0, expected(i));
      continue;
    }
    BOOST_REQUIRE_EQUAL(d.size(), n);
    for (std::size_t i = 0; i < n; ++i)
      BOOST_CHECK_EQUAL(d.data()[i], expected(i));
  }
}

const std::vector<bool> all_null_5{false, false, false, false, false};
const std::vector<bool> all_present_5{true, true, true, true, true};

}  // namespace

// Baseline: the *value-returning* kernel already handles an all-null left.
BOOST_AUTO_TEST_CASE(arena_add_all_null_left_cells) {
  arena_outer_t L = make_arena_tot_sparse(5, 4, 1.0, all_null_5);
  arena_outer_t R = make_arena_tot_sparse(5, 4, 0.5, all_present_5);
  check_union(L.add(R), L, R, [](double a, double b) { return a + b; });
}

// The in-place kernel must not drop the right-hand cells.
BOOST_AUTO_TEST_CASE(arena_add_to_all_null_left_cells) {
  arena_outer_t L = make_arena_tot_sparse(5, 4, 1.0, all_null_5);
  arena_outer_t R = make_arena_tot_sparse(5, 4, 0.5, all_present_5);
  arena_outer_t t = L.clone();
  t.add_to(R);
  check_union(t, L, R, [](double a, double b) { return a + b; });
}

BOOST_AUTO_TEST_CASE(arena_add_to_mismatched_null_inners) {
  arena_outer_t L = make_arena_tot_sparse(5, 4, 1.0, nz_L);
  arena_outer_t R = make_arena_tot_sparse(5, 4, 0.5, nz_R);
  arena_outer_t t = L.clone();
  t.add_to(R);
  check_union(t, L, R, [](double a, double b) { return a + b; });
}

BOOST_AUTO_TEST_CASE(arena_subt_to_mismatched_null_inners) {
  arena_outer_t L = make_arena_tot_sparse(5, 4, 5.0, nz_L);
  arena_outer_t R = make_arena_tot_sparse(5, 4, 1.0, nz_R);
  arena_outer_t t = L.clone();
  t.subt_to(R);
  check_union(t, L, R, [](double a, double b) { return a - b; });
}

// `nz_L`/`nz_R` disagree in both directions: cell 0 is left-only (the in-place
// kernel must zero it) and cell 4 is right-only, which is what drives
// `Tensor::mult_to` onto its tensor-of-view fallback.
BOOST_AUTO_TEST_CASE(arena_mult_to_mismatched_null_inners) {
  arena_outer_t L = make_arena_tot_sparse(5, 4, 2.0, nz_L);
  arena_outer_t R = make_arena_tot_sparse(5, 4, 0.5, nz_R);
  arena_outer_t t = L.clone();
  t.mult_to(R);
  check_union(t, L, R, [](double a, double b) { return a * b; });
}

BOOST_AUTO_TEST_CASE(arena_add_to_scaled_mismatched_null_inners) {
  arena_outer_t L = make_arena_tot_sparse(5, 4, 1.0, nz_L);
  arena_outer_t R = make_arena_tot_sparse(5, 4, 0.5, nz_R);
  arena_outer_t t = L.clone();
  t.add_to(R, 2.0);  // legacy semantics: (l + r) * factor
  check_union(t, L, R, [](double a, double b) { return (a + b) * 2.0; });
}

BOOST_AUTO_TEST_CASE(arena_axpy_to_mismatched_null_inners) {
  arena_outer_t L = make_arena_tot_sparse(5, 4, 1.0, nz_L);
  arena_outer_t R = make_arena_tot_sparse(5, 4, 0.5, nz_R);
  arena_outer_t t = L.clone();
  t.axpy_to(R, 2.0);
  check_union(t, L, R, [](double a, double b) { return a + b * 2.0; });
}

// The free per-cell kernels are deliberately asymmetric about a null
// destination: the additive ops would drop the source, so they refuse
// loudly; `mult_to` annihilates to zero, which a null cell already
// denotes, so it stays a no-op.
BOOST_AUTO_TEST_CASE(arena_cell_kernels_null_destination_policy) {
  arena_outer_t L = make_arena_tot_sparse(2, 4, 1.0, {false, true});
  arena_outer_t R = make_arena_tot_sparse(2, 4, 0.5, {true, false});
  arena_inner_t& null_dst = L.data()[0];
  const arena_inner_t& live_src = R.data()[0];
  BOOST_REQUIRE(null_dst.empty());
  BOOST_REQUIRE(!live_src.empty());
  // additive ops: the source would be dropped, so refuse loudly
  BOOST_CHECK_THROW(TiledArray::add_to(null_dst, live_src),
                    TiledArray::Exception);
  BOOST_CHECK_THROW(TiledArray::subt_to(null_dst, live_src),
                    TiledArray::Exception);
  BOOST_CHECK_THROW(TiledArray::axpy_to(null_dst, live_src, 2.0),
                    TiledArray::Exception);
  // mult annihilates: 0 * src == 0, which the null destination already
  // denotes, so nothing is lost and this is a legitimate no-op
  BOOST_CHECK_NO_THROW(TiledArray::mult_to(null_dst, live_src));
  BOOST_CHECK(null_dst.empty());

  // a null *source* is a no-op for the additive ops ...
  arena_inner_t& live_dst = L.data()[1];
  const arena_inner_t& null_src = R.data()[1];
  BOOST_REQUIRE(null_src.empty());
  const double before = live_dst.data()[0];
  BOOST_CHECK_NO_THROW(TiledArray::add_to(live_dst, null_src));
  BOOST_CHECK_EQUAL(live_dst.data()[0], before);
  // ... but zeroes the destination for mult (dst *= 0)
  BOOST_CHECK_NO_THROW(TiledArray::mult_to(live_dst, null_src));
  BOOST_CHECK_EQUAL(live_dst.data()[0], 0.0);
}

// --- compile-only regression: non-view tensors never instantiate the -----
// tensor-of-view fallback. `Tensor<std::complex<double>>` accepts an `int`
// factor in place -- `(l -= r) *= 2` goes through
// `complex<double>::operator*=(const double&)` -- but the value-returning
// `subt(right, int)` the fallback would call is ill-formed, because the
// binary `operator*(const complex<_Tp>&, const _Tp&)` cannot deduce `_Tp`
// from a `complex<double>` / `int` pair. Guarding the fallback with
// `if constexpr` keeps it uninstantiated here; without that guard this test
// case does not compile. (This is the shape MPQC instantiates.)
BOOST_AUTO_TEST_CASE(complex_tile_inplace_ops_with_int_factor) {
  using ztile_t = TA::Tensor<std::complex<double>>;
  const TA::Range r{4l};
  auto make = [&r](double v) {
    ztile_t t(r);
    for (std::size_t i = 0; i < r.volume(); ++i)
      t.at_ordinal(i) = std::complex<double>(v + i, -v);
    return t;
  };
  const ztile_t L = make(1.0), R = make(0.5);

  ztile_t a = L.clone();
  a.add_to(R, int{2});  // legacy semantics: (l + r) * factor
  ztile_t d = L.clone();
  d.subt_to(R, int{2});
  ztile_t m = L.clone();
  m.mult_to(R, int{2});
  // `axpy_to`'s own in-place body is `l += r * factor`, so a mixed
  // complex/int pair is ill-formed there independently of the fallback --
  // exercise it with a scalar the elementwise body accepts.
  ztile_t x = L.clone();
  x.axpy_to(R, 2.0);

  for (std::size_t i = 0; i < r.volume(); ++i) {
    const std::complex<double> l = L.at_ordinal(i), rr = R.at_ordinal(i);
    BOOST_CHECK_EQUAL(a.at_ordinal(i), (l + rr) * 2.0);
    BOOST_CHECK_EQUAL(d.at_ordinal(i), (l - rr) * 2.0);
    BOOST_CHECK_EQUAL(m.at_ordinal(i), (l * rr) * 2.0);
    BOOST_CHECK_EQUAL(x.at_ordinal(i), l + rr * 2.0);
  }
}

// --- regression: `add` on an empty (default-constructed) left outer -------
BOOST_AUTO_TEST_CASE(add_empty_left_outer_returns_right) {
  outer_t L;  // default-constructed -> empty outer
  outer_t R = make_tot(3, 4, 7.0);
  BOOST_REQUIRE(L.empty());
  BOOST_CHECK(tot_equal(L.add(R), R));
}

// --- expression level: a permuted operand must not turn the add in place --
// `ArrayEvalImpl::is_consumable()` is true for any permuted operand, which
// makes `BinaryWrapper` rewrite `A(perm) + B` as `A(perm).add_to(B)`. With
// all of A's inner cells null that in-place add used to drop B entirely.
BOOST_AUTO_TEST_CASE(expr_permuted_add_null_left_cells_keeps_right) {
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::SparsePolicy>;

  auto& world = TA::get_default_world();
  constexpr long P = 2, Q = 3;
  TA::TiledRange trange{{0l, 2l, 4l}, {0l, 2l, 4l}};  // 2x2 tiles, 4x4 outer

  TA::Tensor<float> norms(trange.tiles_range(), 1.0f);
  TA::SparseShape<float> shape(norms, trange);

  auto b_val = [](long i, long j, long a, long b) {
    return 1.0 + 10.0 * i + 100.0 * j + 0.5 * a + 0.25 * b;
  };

  // A: every tile present, every inner cell null (an all-zero ToT)
  ArenaArr A(world, trange, shape);
  A.init_tiles([](const TA::Range& tr) {
    return TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [](std::size_t) { return TA::Range{}; });
  });
  // B: every tile present, every inner cell populated
  ArenaArr B(world, trange, shape);
  B.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t =
        TA::detail::arena_outer_init<ArenaOuter>(tr, 1, [](std::size_t) {
          return TA::Range{P, Q};
        });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (c.empty()) continue;
      const auto idx = tr.idx(o);
      for (long a = 0; a < P; ++a)
        for (long b = 0; b < Q; ++b)
          c.data()[a * Q + b] = b_val(idx[0], idx[1], a, b);
    }
    return t;
  });
  world.gop.fence();

  // A is transposed on the outer indices only (arena cells reject an inner
  // permutation); since A is all zeros the sum must reproduce B exactly.
  ArenaArr C;
  C("i,j;a,b") = A("j,i;a,b") + B("i,j;a,b");
  world.gop.fence();

  ArenaArr D;  // control: no permutation
  D("i,j;a,b") = A("i,j;a,b") + B("i,j;a,b");
  world.gop.fence();

  for (const auto& out : {std::ref(C), std::ref(D)}) {
    const ArenaArr& got = out.get();
    for (std::size_t t = 0; t < trange.tiles_range().volume(); ++t) {
      BOOST_REQUIRE(!got.is_zero(t));
      ArenaOuter gt = got.find(t).get();
      const TA::Range& tr = trange.make_tile_range(t);
      BOOST_REQUIRE_EQUAL(gt.range().volume(), tr.volume());
      for (std::size_t o = 0; o < gt.range().volume(); ++o) {
        const ArenaInner& c = gt.data()[o];
        BOOST_REQUIRE(!c.empty());
        const auto idx = tr.idx(o);
        for (long a = 0; a < P; ++a)
          for (long b = 0; b < Q; ++b)
            BOOST_CHECK_EQUAL(c.data()[a * Q + b], b_val(idx[0], idx[1], a, b));
      }
    }
  }
}

// --- empty-operand guards on the in-place binary ops ---------------------
// An empty tensor denotes zero in these ops -- that is already how `add_to`
// and the unscaled `mult_to` treat it. `subt_to` (both overloads), the scaled
// `add_to` and the scaled `mult_to` were missing one or both guards, so they
// fell through into `inplace_binary`, whose `!empty()` precondition is a
// TA_ASSERT: it throws in a Debug build and, compiled out in Release, walks a
// null `data()` or silently yields an empty (== zero) result. Same
// silent-data-loss shape as the null-cell bug, one function over.

BOOST_AUTO_TEST_CASE(subt_to_empty_left_yields_negated_right) {
  outer_t L;  // default-constructed -> empty outer == zero
  outer_t R = make_tot(3, 4, 7.0);
  BOOST_REQUIRE(L.empty());
  const outer_t expected = R.scale(-1.0);
  L.subt_to(R);
  BOOST_CHECK(tot_equal(L, expected));
}

BOOST_AUTO_TEST_CASE(subt_to_scaled_empty_left_yields_negated_scaled_right) {
  outer_t L;
  outer_t R = make_tot(3, 4, 7.0);
  const outer_t expected = R.scale(-2.0);
  L.subt_to(R, 2.0);  // legacy semantics: (l - r) * factor
  BOOST_CHECK(tot_equal(L, expected));
}

BOOST_AUTO_TEST_CASE(add_to_scaled_empty_left_yields_scaled_right) {
  outer_t L;
  outer_t R = make_tot(3, 4, 7.0);
  const outer_t expected = R.scale(2.0);
  L.add_to(R, 2.0);  // legacy semantics: (l + r) * factor
  BOOST_CHECK(tot_equal(L, expected));
}

BOOST_AUTO_TEST_CASE(add_to_scaled_empty_right_scales_left) {
  outer_t L = make_tot(3, 4, 5.0);
  const outer_t expected = L.scale(2.0);
  const outer_t empty_right;
  L.add_to(empty_right, 2.0);
  BOOST_CHECK(tot_equal(L, expected));
}

BOOST_AUTO_TEST_CASE(mult_to_scaled_empty_right_is_zero) {
  outer_t L = make_tot(3, 4, 5.0);
  const outer_t empty_right;
  L.mult_to(empty_right, 2.0);
  // matches the unscaled `mult_to`, which spells the zero product as an
  // empty result
  BOOST_CHECK(L.empty());
}

// --- non-contiguous (block-view) right operand ---------------------------
// `Tensor::block()` yields a `TensorInterface<T, BlockRange>`, whose cells are
// strided in the parent's storage. Both `inplace_binary_drops_cells` and the
// arena value-returning kernels must address it through its range; reading it
// linearly would silently pick up the wrong cells rather than fail.

namespace {

/// 2-D outer arena ToT; `present[ord]==false` gives a deliberately-null cell.
arena_outer_t make_arena_tot_2d(long n0, long n1, std::size_t n_inner,
                                double base, const std::vector<bool>& present) {
  auto range_fn = [&present, n_inner](std::size_t ord) {
    return present[ord] ? TA::Range{static_cast<long>(n_inner)} : TA::Range{};
  };
  arena_outer_t t = TA::detail::arena_outer_init<arena_outer_t>(
      TA::Range{n0, n1}, 1, range_fn, alignof(double), /*zero_init=*/true);
  const std::size_t N = static_cast<std::size_t>(n0 * n1);
  for (std::size_t ord = 0; ord < N; ++ord) {
    arena_inner_t& c = t.data()[ord];
    if (c.empty()) continue;
    for (std::size_t i = 0; i < n_inner; ++i)
      c.data()[i] = base + ord * 100.0 + i;
  }
  return t;
}

}  // namespace

BOOST_AUTO_TEST_CASE(arena_add_to_noncontiguous_right_block) {
  // R is 2x3, so its leading 2x2 sub-block has row stride 3: block cell (1,0)
  // is R ordinal 3, not 2. Reading the block linearly would take R's ordinal 2
  // instead, which is exactly what this test is here to catch.
  const arena_outer_t R =
      make_arena_tot_2d(2, 3, 4, 0.5, {true, true, true, true, true, true});
  auto blk = R.block({0l, 0l}, {2l, 2l});
  static_assert(!TA::detail::is_contiguous_tensor<decltype(blk)>::value,
                "a BlockRange view must be non-contiguous for this test to "
                "exercise the strided path");

  // L is 2x2 with cell (0,1) null, so the fallback fires and must not drop the
  // block's populated cell there.
  const arena_outer_t L =
      make_arena_tot_2d(2, 2, 4, 1.0, {true, false, true, true});
  arena_outer_t t = L.clone();
  t.add_to(blk);

  BOOST_REQUIRE_EQUAL(t.range().volume(), 4u);
  for (std::size_t ord = 0; ord < 4; ++ord) {
    const arena_inner_t& lc = L.data()[ord];
    // the block's own view of cell `ord` -- strided, via its range
    const arena_inner_t& rc = blk.data()[blk.range().ordinal(ord)];
    const arena_inner_t& got = t.data()[ord];
    BOOST_REQUIRE(!rc.empty());
    BOOST_REQUIRE(!got.empty());
    for (std::size_t i = 0; i < got.size(); ++i) {
      const double lv = lc.empty() ? 0.0 : lc.data()[i];
      BOOST_CHECK_EQUAL(got.data()[i], lv + rc.data()[i]);
    }
  }
}

// --- mult uses intersection sparsity -------------------------------------
// Multiplication annihilates, so the minimal correct result sparsity is the
// *intersection* of the operands' cell patterns: a cell absent from either
// operand has an identically zero product, which a null cell already denotes.
// `nz_L`/`nz_R` disagree in both directions (cell 0 is right-only, cells 3/4
// differ), so these pin the rule rather than pass vacuously.

BOOST_AUTO_TEST_CASE(arena_mult_uses_intersection_sparsity) {
  const arena_outer_t L = make_arena_tot_sparse(5, 4, 2.0, nz_L);
  const arena_outer_t R = make_arena_tot_sparse(5, 4, 0.5, nz_R);
  const arena_outer_t prod = L.mult(R);
  for (std::size_t ord = 0; ord < 5; ++ord) {
    const arena_inner_t& l = L.data()[ord];
    const arena_inner_t& r = R.data()[ord];
    const arena_inner_t& d = prod.data()[ord];
    if (!l.empty() && !r.empty()) {
      BOOST_REQUIRE(!d.empty());
      for (std::size_t i = 0; i < d.size(); ++i)
        BOOST_CHECK_EQUAL(d.data()[i], l.data()[i] * r.data()[i]);
    } else {
      // union sparsity would have emitted an explicit zero slab here
      BOOST_CHECK(d.empty());
    }
  }
}

// The in-place op must not densify either: a null destination cell stays null
// (its product is zero), and a populated cell against a null source is zeroed
// in place because a view cannot free its own storage.
BOOST_AUTO_TEST_CASE(arena_mult_to_does_not_densify_null_cells) {
  const arena_outer_t L = make_arena_tot_sparse(5, 4, 2.0, nz_L);
  const arena_outer_t R = make_arena_tot_sparse(5, 4, 0.5, nz_R);
  arena_outer_t t = L.clone();
  t.mult_to(R);
  for (std::size_t ord = 0; ord < 5; ++ord) {
    const arena_inner_t& l = L.data()[ord];
    const arena_inner_t& r = R.data()[ord];
    const arena_inner_t& d = t.data()[ord];
    if (l.empty()) {
      BOOST_CHECK(d.empty());
    } else if (r.empty()) {
      BOOST_REQUIRE(!d.empty());
      for (std::size_t i = 0; i < d.size(); ++i)
        BOOST_CHECK_EQUAL(d.data()[i], 0.0);
    } else {
      BOOST_REQUIRE(!d.empty());
      for (std::size_t i = 0; i < d.size(); ++i)
        BOOST_CHECK_EQUAL(d.data()[i], l.data()[i] * r.data()[i]);
    }
  }
}

// --- an empty-`this` fast path must not alias, let alone mutate, the source
// `detail::clone_or_cast` deep-copies only when `Right` *is* `Tensor`; for any
// other nested-tensor `Right` (e.g. the `TensorInterface<..., BlockRange>` that
// `block()` yields) it copies the cell *handles*, which for a view cell aliases
// the source's storage. Scaling or negating the copy then writes through to a
// const operand.

BOOST_AUTO_TEST_CASE(subt_to_empty_left_must_not_mutate_source) {
  arena_outer_t P =
      make_arena_tot_2d(2, 3, 4, 1.0, {true, true, true, true, true, true});
  const double before = P.data()[0].data()[0];
  arena_outer_t t;
  t.subt_to(P.block({0l, 0l}, {2l, 2l}));
  BOOST_CHECK_EQUAL(P.data()[0].data()[0], before);
  BOOST_CHECK_EQUAL(t.data()[0].data()[0], -before);
}

BOOST_AUTO_TEST_CASE(subt_to_scaled_empty_left_must_not_mutate_source) {
  arena_outer_t P =
      make_arena_tot_2d(2, 3, 4, 1.0, {true, true, true, true, true, true});
  const double before = P.data()[0].data()[0];
  arena_outer_t t;
  t.subt_to(P.block({0l, 0l}, {2l, 2l}), 2.0);
  BOOST_CHECK_EQUAL(P.data()[0].data()[0], before);
  BOOST_CHECK_EQUAL(t.data()[0].data()[0], -before * 2.0);
}

BOOST_AUTO_TEST_CASE(add_to_scaled_empty_left_must_not_mutate_source) {
  arena_outer_t P =
      make_arena_tot_2d(2, 3, 4, 1.0, {true, true, true, true, true, true});
  const double before = P.data()[0].data()[0];
  arena_outer_t t;
  t.add_to(P.block({0l, 0l}, {2l, 2l}), 3.0);
  BOOST_CHECK_EQUAL(P.data()[0].data()[0], before);
  BOOST_CHECK_EQUAL(t.data()[0].data()[0], before * 3.0);
}

BOOST_AUTO_TEST_CASE(axpy_to_empty_left_must_not_mutate_source) {
  arena_outer_t P =
      make_arena_tot_2d(2, 3, 4, 1.0, {true, true, true, true, true, true});
  const double before = P.data()[0].data()[0];
  arena_outer_t t;
  t.axpy_to(P.block({0l, 0l}, {2l, 2l}), 2.0);
  BOOST_CHECK_EQUAL(P.data()[0].data()[0], before);
}

// the value-returning empty-left exits have the same hazard: the result must
// own its cells, or the caller's later mutation corrupts the source
BOOST_AUTO_TEST_CASE(add_empty_left_result_must_not_alias_source) {
  arena_outer_t P =
      make_arena_tot_2d(2, 3, 4, 1.0, {true, true, true, true, true, true});
  const double before = P.data()[0].data()[0];
  const arena_outer_t empty_left;
  arena_outer_t r = empty_left.add(P.block({0l, 0l}, {2l, 2l}));
  r.scale_to(-1.0);
  BOOST_CHECK_EQUAL(P.data()[0].data()[0], before);
}

// --- intersection sparsity also governs the plain owning ToT --------------
// `Tensor::mult`'s intersection rule is applied in the `is_ta_tensor_v` branch
// too, so it is a behavior change for `TA::Tensor<TA::Tensor<T>>`, not only for
// arena tiles. Pin it: the pre-existing mult_mismatched_null_inners is written
// permissively and passes under either rule.
BOOST_AUTO_TEST_CASE(plain_tot_mult_uses_intersection_sparsity) {
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
    } else {
      BOOST_CHECK(d.empty());
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
