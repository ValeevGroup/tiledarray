/// Tests for the arena-backed factory that builds an outer tile of
/// `ArenaTensor` cells: SIMD-aligned data, null cells for zero-volume
/// shapes, monotonic slab layout, slab survives factory scope.

#include "TiledArray/tensor/arena_kernels.h"

#include "TiledArray/tensor.h"
#include "TiledArray/tensor/arena_einsum.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <optional>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace TA = TiledArray;
using Inner = TA::ArenaTensor<double, TA::Range>;
using Outer = TA::Tensor<Inner>;

BOOST_AUTO_TEST_SUITE(arena_tensor_kernels_suite, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(builds_outer_with_uniform_inners) {
  TA::Range outer_r{4};
  auto shape_fn = [](std::size_t /*ord*/) { return TA::Range{8}; };
  Outer outer = TA::detail::arena_outer_init<Outer>(outer_r, 1, shape_fn);
  BOOST_REQUIRE_EQUAL(outer.range().volume(), 4u);
  for (std::size_t ord = 0; ord < 4; ++ord) {
    Inner& inner = outer.data()[ord];
    BOOST_CHECK(bool(inner));
    BOOST_CHECK_EQUAL(inner.size(), 8u);
    auto addr = reinterpret_cast<std::uintptr_t>(inner.data());
    BOOST_CHECK_EQUAL(addr % TA::kInnerSimdAlign, 0u);
  }
}

BOOST_AUTO_TEST_CASE(zero_volume_shapes_yield_null_inners) {
  TA::Range outer_r{4};
  auto shape_fn = [](std::size_t ord) {
    return ord % 2 == 0 ? TA::Range{4} : TA::Range();
  };
  Outer outer = TA::detail::arena_outer_init<Outer>(outer_r, 1, shape_fn);
  for (std::size_t ord = 0; ord < 4; ++ord) {
    Inner& inner = outer.data()[ord];
    if (ord % 2 == 0) {
      BOOST_CHECK(bool(inner));
      BOOST_CHECK_EQUAL(inner.size(), 4u);
    } else {
      BOOST_CHECK(!inner);
    }
  }
}

BOOST_AUTO_TEST_CASE(non_null_cells_share_one_monotonic_slab) {
  TA::Range outer_r{6};
  auto shape_fn = [](std::size_t /*ord*/) { return TA::Range{6}; };
  Outer outer = TA::detail::arena_outer_init<Outer>(outer_r, 1, shape_fn);
  const double* prev_end = nullptr;
  for (std::size_t ord = 0; ord < 6; ++ord) {
    Inner& inner = outer.data()[ord];
    const double* begin = inner.data();
    const double* end = begin + inner.size();
    if (prev_end != nullptr) {
      BOOST_CHECK(begin >= prev_end);
      // Gap bounded by one cell stride (cache-line-floor or SIMD-driven).
      const std::size_t gap =
          static_cast<std::size_t>(begin - prev_end) * sizeof(double);
      BOOST_CHECK_LE(gap,
                     TA::detail::kArenaCachelineAlign + Inner::cell_size(0));
    }
    prev_end = end;
  }
}

BOOST_AUTO_TEST_CASE(outer_outlives_factory_scope) {
  Outer outer;
  {
    TA::Range outer_r{3};
    auto shape_fn = [](std::size_t /*ord*/) { return TA::Range{4}; };
    outer = TA::detail::arena_outer_init<Outer>(outer_r, 1, shape_fn);
  }
  for (std::size_t ord = 0; ord < 3; ++ord) {
    Inner& inner = outer.data()[ord];
    TA::fill(inner, double(ord + 1));
  }
  for (std::size_t ord = 0; ord < 3; ++ord) {
    Inner& inner = outer.data()[ord];
    for (std::size_t i = 0; i < inner.size(); ++i)
      BOOST_CHECK_EQUAL(inner.data()[i], double(ord + 1));
  }
}

BOOST_AUTO_TEST_CASE(jagged_inner_shapes_round_trip) {
  TA::Range outer_r{4};
  std::vector<long> sizes = {3, 5, 0, 7};
  auto shape_fn = [&](std::size_t ord) {
    return sizes[ord] == 0 ? TA::Range() : TA::Range{sizes[ord]};
  };
  Outer outer = TA::detail::arena_outer_init<Outer>(outer_r, 1, shape_fn);
  for (std::size_t ord = 0; ord < 4; ++ord) {
    Inner& inner = outer.data()[ord];
    if (sizes[ord] == 0) {
      BOOST_CHECK(!inner);
    } else {
      BOOST_REQUIRE(bool(inner));
      BOOST_CHECK_EQUAL(inner.size(), static_cast<std::size_t>(sizes[ord]));
      auto addr = reinterpret_cast<std::uintptr_t>(inner.data());
      BOOST_CHECK_EQUAL(addr % TA::kInnerSimdAlign, 0u);
    }
  }
}

BOOST_AUTO_TEST_CASE(empty_outer_range_yields_no_slab) {
  TA::Range outer_r{0};
  auto shape_fn = [](std::size_t /*ord*/) { return TA::Range{4}; };
  Outer outer = TA::detail::arena_outer_init<Outer>(outer_r, 1, shape_fn);
  BOOST_CHECK_EQUAL(outer.range().volume(), 0u);
}

BOOST_AUTO_TEST_CASE(all_null_outer_works) {
  TA::Range outer_r{5};
  auto shape_fn = [](std::size_t /*ord*/) { return TA::Range(); };
  Outer outer = TA::detail::arena_outer_init<Outer>(outer_r, 1, shape_fn);
  for (std::size_t ord = 0; ord < 5; ++ord) BOOST_CHECK(!outer.data()[ord]);
}

namespace {

/// Build an outer with uniform inners filled by an ordinal-dependent rule.
Outer make_outer(std::size_t n_outer, std::size_t n_inner, double base) {
  TA::Range outer_r{static_cast<long>(n_outer)};
  auto shape_fn = [n_inner](std::size_t /*ord*/) {
    return TA::Range{static_cast<long>(n_inner)};
  };
  Outer outer = TA::detail::arena_outer_init<Outer>(outer_r, 1, shape_fn);
  for (std::size_t ord = 0; ord < n_outer; ++ord) {
    Inner& inner = outer.data()[ord];
    for (std::size_t i = 0; i < inner.size(); ++i)
      inner.data()[i] = base + ord * 100.0 + i;
  }
  return outer;
}

bool outers_equal(const Outer& a, const Outer& b) {
  if (a.range().volume() != b.range().volume()) return false;
  for (std::size_t ord = 0; ord < a.range().volume(); ++ord) {
    const Inner& ai = a.data()[ord];
    const Inner& bi = b.data()[ord];
    if (bool(ai) != bool(bi)) return false;
    if (!ai) continue;
    if (ai.size() != bi.size()) return false;
    for (std::size_t i = 0; i < ai.size(); ++i)
      if (ai.data()[i] != bi.data()[i]) return false;
  }
  return true;
}

}  // namespace

BOOST_AUTO_TEST_CASE(builder_matches_up_front_baseline) {
  // incremental one-pass construction of an ArenaTensor-celled outer tile
  Outer baseline = make_outer(4, 8, 1.0);
  TA::detail::ArenaToTBuilder<Outer> b(TA::Range{4});
  for (std::size_t ord = 0; ord < 4; ++ord) {
    Inner& cell = b.emplace(ord, TA::Range{8});
    for (std::size_t i = 0; i < 8; ++i)
      cell.data()[i] = 1.0 + ord * 100.0 + double(i);
  }
  Outer built = std::move(b).finish();
  BOOST_CHECK(outers_equal(built, baseline));
}

BOOST_AUTO_TEST_CASE(builder_rolls_over_to_multiple_pages) {
  const std::size_t N = 10;
  TA::detail::ArenaToTBuilder<Outer> b(TA::Range{static_cast<long>(N)}, 1,
                                       false,
                                       /*page_size=*/Inner::cell_size(8) * 4);
  for (std::size_t ord = 0; ord < N; ++ord) {
    Inner& cell = b.emplace(ord, TA::Range{8});
    for (std::size_t i = 0; i < 8; ++i)
      cell.data()[i] = 1.0 + ord * 100.0 + double(i);
  }
  BOOST_CHECK_GT(b.arena().page_count(), 1u);
  Outer built = std::move(b).finish();
  BOOST_CHECK(outers_equal(built, make_outer(N, 8, 1.0)));
}

BOOST_AUTO_TEST_CASE(builder_single_cell_uses_one_exact_page) {
  // corner case (b): a lone ArenaTensor cell -> one exactly-sized page
  TA::detail::ArenaToTBuilder<Outer> b(TA::Range{1});
  Inner& cell = b.emplace(0, TA::Range{7});
  for (std::size_t i = 0; i < 7; ++i) cell.data()[i] = double(i);
  BOOST_CHECK_EQUAL(b.arena().page_count(), 1u);
  BOOST_CHECK_EQUAL(b.arena().bytes_reserved(), Inner::cell_size(7));
  Outer built = std::move(b).finish();
  BOOST_REQUIRE_EQUAL(built.range().volume(), 1u);
  BOOST_REQUIRE(bool(built.data()[0]));
  for (std::size_t i = 0; i < 7; ++i)
    BOOST_CHECK_EQUAL(built.data()[0].data()[i], double(i));
}

BOOST_AUTO_TEST_CASE(compact_coalesces_a_multipage_tile) {
  const std::size_t N = 9;
  TA::detail::ArenaToTBuilder<Outer> b(TA::Range{static_cast<long>(N)}, 1,
                                       false, Inner::cell_size(6) * 4);
  for (std::size_t ord = 0; ord < N; ++ord) {
    Inner& cell = b.emplace(ord, TA::Range{6});
    for (std::size_t i = 0; i < 6; ++i)
      cell.data()[i] = 1.0 + ord * 100.0 + double(i);
  }
  BOOST_CHECK_GT(b.arena().page_count(), 1u);
  Outer multipage = std::move(b).finish();
  Outer compacted = TA::detail::arena_compact(multipage);
  BOOST_CHECK(outers_equal(compacted, multipage));
  BOOST_CHECK(outers_equal(compacted, make_outer(N, 6, 1.0)));
}

BOOST_AUTO_TEST_CASE(arena_tensor_is_a_tensor_but_a_view) {
  // ArenaTensor is registered as is_tensor_helper / is_contiguous_tensor so
  // kernel paths treat it like Tensor<double>; the `is_tensor_view` trait
  // opts it out of value-returning member-call paths (which require
  // allocation a view cannot do).
  static_assert(TA::detail::is_tensor_helper<Inner>::value);
  static_assert(TA::detail::is_contiguous_tensor<Inner>::value);
  static_assert(TA::detail::is_tensor<Inner>::value);
  static_assert(TA::is_tensor_view_v<Inner>);
  static_assert(TA::is_arena_tensor_v<Inner>);
  // ta_ops_match_tensor (value-returning ops gate) is now false for views.
  static_assert(!TA::detail::ta_ops_match_tensor_v<Inner>);
  // ta_ops_match_tensor_inplace (in-place ops gate) is true.
  static_assert(TA::detail::ta_ops_match_tensor_inplace_v<Inner>);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(trivial_clone_inner_round_trip) {
  Outer src = make_outer(4, 5, 1.0);
  Outer copy = src.clone();
  BOOST_CHECK(outers_equal(copy, src));
  // Independent slab: mutating copy doesn't affect src.
  copy.data()[0].data()[0] = -1.0;
  BOOST_CHECK_EQUAL(src.data()[0].data()[0], 1.0);
}

BOOST_AUTO_TEST_CASE(trivial_scale_inner_multiplies) {
  Outer src = make_outer(3, 4, 1.0);
  Outer scaled = src.scale(2.5);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const Inner& sinner = src.data()[ord];
    const Inner& dinner = scaled.data()[ord];
    BOOST_REQUIRE_EQUAL(dinner.size(), sinner.size());
    for (std::size_t i = 0; i < sinner.size(); ++i)
      BOOST_CHECK_EQUAL(dinner.data()[i], sinner.data()[i] * 2.5);
  }
}

BOOST_AUTO_TEST_CASE(trivial_add_inner_accumulates) {
  Outer L = make_outer(3, 4, 1.0);
  Outer R = make_outer(3, 4, 0.5);
  Outer sum = L.add(R);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const Inner& l = L.data()[ord];
    const Inner& r = R.data()[ord];
    const Inner& d = sum.data()[ord];
    for (std::size_t i = 0; i < l.size(); ++i)
      BOOST_CHECK_EQUAL(d.data()[i], l.data()[i] + r.data()[i]);
  }
}

BOOST_AUTO_TEST_CASE(trivial_subt_inner_subtracts) {
  Outer L = make_outer(3, 4, 5.0);
  Outer R = make_outer(3, 4, 1.0);
  Outer diff = L.subt(R);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const Inner& l = L.data()[ord];
    const Inner& r = R.data()[ord];
    const Inner& d = diff.data()[ord];
    for (std::size_t i = 0; i < l.size(); ++i)
      BOOST_CHECK_EQUAL(d.data()[i], l.data()[i] - r.data()[i]);
  }
}

BOOST_AUTO_TEST_CASE(trivial_mult_inner_elementwise) {
  Outer L = make_outer(3, 4, 2.0);
  Outer R = make_outer(3, 4, 0.5);
  Outer prod = L.mult(R);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const Inner& l = L.data()[ord];
    const Inner& r = R.data()[ord];
    const Inner& d = prod.data()[ord];
    for (std::size_t i = 0; i < l.size(); ++i)
      BOOST_CHECK_EQUAL(d.data()[i], l.data()[i] * r.data()[i]);
  }
}

BOOST_AUTO_TEST_CASE(contraction_arena_plan_reserve_and_construct_inner) {
  // Verify ContractionArenaPlan's inner-tensor dispatch builds the right
  // outer/inner shapes and SIMD-aligns each non-null inner cell.
  TA::math::GemmHelper outer_gh(TA::math::blas::NoTranspose,
                                TA::math::blas::NoTranspose, 2, 2, 2);
  TA::math::GemmHelper inner_gh(TA::math::blas::NoTranspose,
                                TA::math::blas::NoTranspose, 2, 2, 2);
  auto left = TA::detail::arena_outer_init<Outer>(TA::Range{2, 3}, 1,
                                                  [](std::size_t /*ord*/) {
                                                    return TA::Range{4, 5};
                                                  });
  auto right = TA::detail::arena_outer_init<Outer>(TA::Range{3, 4}, 1,
                                                   [](std::size_t /*ord*/) {
                                                     return TA::Range{5, 6};
                                                   });
  TA::detail::ArenaInnerShapePlan inner_plan{
      TA::detail::ArenaInnerShapeKind::gemm_result_range,
      std::make_optional(inner_gh)};
  TA::detail::ContractionArenaPlan<Outer, Outer, Outer> plan(inner_plan);
  Outer result = plan.reserve_and_construct(left, right, outer_gh);
  // Outer result: 2x4 = 8 cells; each inner: 4x6 = 24 elements.
  BOOST_REQUIRE_EQUAL(result.range().volume(), 8u);
  for (std::size_t ord = 0; ord < 8; ++ord) {
    const Inner& inner = result.data()[ord];
    BOOST_REQUIRE(bool(inner));
    BOOST_CHECK_EQUAL(inner.size(), 24u);
    auto addr = reinterpret_cast<std::uintptr_t>(inner.data());
    BOOST_CHECK_EQUAL(addr % TA::kInnerSimdAlign, 0u);
  }
}

BOOST_AUTO_TEST_CASE(outer_gemm_with_arena_tensor_contraction) {
  // End-to-end: arena-allocate result via the plan, then run TA::Tensor's
  // outer gemm with a custom elem_muladd_op that calls the free gemm CPO
  // for ArenaTensor inners. Verifies the full chain reserve_and_construct
  // -> outer iteration -> inner BLAS gemm.
  TA::math::GemmHelper outer_gh(TA::math::blas::NoTranspose,
                                TA::math::blas::NoTranspose, 2, 2, 2);
  TA::math::GemmHelper inner_gh(TA::math::blas::NoTranspose,
                                TA::math::blas::NoTranspose, 2, 2, 2);
  // A[2,3] outer of <4,5> inners (each 1.0); B[3,4] outer of <5,6> inners
  // (each 2.0). C[2,4] outer of <4,6> inners; each entry =
  //   sum over outer k in [0,3) of sum over inner k in [0,5) of 1.0*2.0
  //   = 3 * 5 * 2.0 = 30.0
  auto left = TA::detail::arena_outer_init<Outer>(TA::Range{2, 3}, 1,
                                                  [](std::size_t /*ord*/) {
                                                    return TA::Range{4, 5};
                                                  });
  auto right = TA::detail::arena_outer_init<Outer>(TA::Range{3, 4}, 1,
                                                   [](std::size_t /*ord*/) {
                                                     return TA::Range{5, 6};
                                                   });
  for (std::size_t i = 0; i < left.range().volume(); ++i)
    TA::fill(left.data()[i], 1.0);
  for (std::size_t i = 0; i < right.range().volume(); ++i)
    TA::fill(right.data()[i], 2.0);

  TA::detail::ArenaInnerShapePlan inner_plan{
      TA::detail::ArenaInnerShapeKind::gemm_result_range,
      std::make_optional(inner_gh)};
  TA::detail::ContractionArenaPlan<Outer, Outer, Outer> plan(inner_plan);
  Outer result = plan.reserve_and_construct(left, right, outer_gh);

  auto elem_muladd = [&inner_gh](Inner& r, const Inner& l, const Inner& rr) {
    TA::gemm(r, l, rr, 1.0, inner_gh);
  };
  result.gemm(left, right, outer_gh, elem_muladd);

  for (std::size_t ord = 0; ord < result.range().volume(); ++ord) {
    const Inner& inner = result.data()[ord];
    BOOST_REQUIRE(bool(inner));
    BOOST_REQUIRE_EQUAL(inner.size(), 24u);
    for (std::size_t e = 0; e < 24; ++e)
      BOOST_CHECK_CLOSE(inner.data()[e], 30.0, 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(trivial_ops_preserve_null_cells) {
  // Outer with mixed null and non-null inners; trivial ops should propagate
  // null cells through to the result.
  TA::Range outer_r{4};
  auto shape_fn = [](std::size_t ord) {
    return ord % 2 == 0 ? TA::Range{4} : TA::Range();
  };
  Outer src = TA::detail::arena_outer_init<Outer>(outer_r, 1, shape_fn);
  for (std::size_t ord = 0; ord < 4; ++ord) {
    Inner& inner = src.data()[ord];
    if (inner) {
      for (std::size_t i = 0; i < inner.size(); ++i) inner.data()[i] = 1.0;
    }
  }
  Outer scaled = src.scale(3.0);
  for (std::size_t ord = 0; ord < 4; ++ord) {
    const Inner& d = scaled.data()[ord];
    if (ord % 2 == 0) {
      BOOST_REQUIRE(bool(d));
      for (std::size_t i = 0; i < d.size(); ++i)
        BOOST_CHECK_EQUAL(d.data()[i], 3.0);
    } else {
      BOOST_CHECK(!d);
    }
  }
}

// Outer-tile serialize round-trip: exercises the arena-aware path in
// TA::Tensor::serialize directly via an in-memory archive. The slab
// gets rebuilt on load.
BOOST_AUTO_TEST_CASE(outer_tile_serialize_round_trip_arena_tensor) {
  // Build an outer with jagged inner shapes including one null cell.
  Outer src =
      TA::detail::arena_outer_init<Outer>(TA::Range{4}, 1, [](std::size_t ord) {
        if (ord == 2) return TA::Range();  // null cell
        return TA::Range{static_cast<long>(3 + ord)};
      });
  // Fill non-null cells with ord-dependent values.
  for (std::size_t ord = 0; ord < 4; ++ord) {
    Inner& cell = src.data()[ord];
    if (cell) {
      for (std::size_t i = 0; i < cell.size(); ++i)
        cell.data()[i] = double(ord * 100 + i);
    }
  }

  const std::size_t buf_size = 1 << 16;
  std::vector<unsigned char> buf(buf_size);
  madness::archive::BufferOutputArchive oar(buf.data(), buf_size);
  BOOST_REQUIRE_NO_THROW(oar & src);
  const std::size_t nbyte = oar.size();
  oar.close();

  Outer dst;
  madness::archive::BufferInputArchive iar(buf.data(), nbyte);
  BOOST_REQUIRE_NO_THROW(iar & dst);
  iar.close();

  // Verify outer shape, null/non-null flags, inner shapes, element values.
  BOOST_REQUIRE_EQUAL(dst.range().volume(), src.range().volume());
  for (std::size_t ord = 0; ord < src.range().volume(); ++ord) {
    const Inner& s = src.data()[ord];
    const Inner& d = dst.data()[ord];
    BOOST_REQUIRE_EQUAL(bool(s), bool(d));
    if (!s) continue;
    BOOST_REQUIRE_EQUAL(s.size(), d.size());
    for (std::size_t i = 0; i < s.size(); ++i)
      BOOST_CHECK_EQUAL(d.data()[i], s.data()[i]);
    // The loaded cell's data pointer is SIMD-aligned via
    // arena_outer_init.
    auto addr = reinterpret_cast<std::uintptr_t>(d.data());
    BOOST_CHECK_EQUAL(addr % TA::kInnerSimdAlign, 0u);
  }
}

// DistArray-level test: forces `TA::DistArray<TA::Tensor<ArenaTensor>>` to
// instantiate, exercising arena-aware serialization at the outer-tile
// boundary. Serial-only (no @distributed).
BOOST_AUTO_TEST_CASE(distarray_arena_tensor_construct_and_init_tiles) {
  using Array = TA::DistArray<Outer, TA::DensePolicy>;
  auto& world = TA::get_default_world();
  TA::TiledRange tr{TA::TiledRange1{0, 2, 4}};
  Array A(world, tr);
  A.init_tiles([](const TA::Range& tile_range) {
    return TA::detail::arena_outer_init<Outer>(
        tile_range, 1, [](std::size_t /*ord*/) { return TA::Range{3}; });
  });
  world.gop.fence();
  BOOST_CHECK_EQUAL(A.trange().tiles_range().volume(), 2u);
  if (A.is_local(0)) {
    Outer tile = A.find(0).get();
    BOOST_CHECK_EQUAL(tile.range().volume(), 2u);
    for (std::size_t i = 0; i < tile.range().volume(); ++i) {
      const Inner& cell = tile.data()[i];
      BOOST_REQUIRE(bool(cell));
      BOOST_CHECK_EQUAL(cell.size(), 3u);
    }
  }
}

// DistArray-level incremental construction: each outer tile is built with
// ArenaToTBuilder *inside* the init_tiles callback -- inner cells are sized
// and filled one at a time, with no up-front range_fn. This needs no new
// DistArray API: init_tiles already supplies a per-tile callback. Serial-only.
BOOST_AUTO_TEST_CASE(distarray_arena_tensor_incremental_init_tiles) {
  using Array = TA::DistArray<Outer, TA::DensePolicy>;
  auto& world = TA::get_default_world();
  TA::TiledRange tr{TA::TiledRange1{0, 2, 4}};
  Array A(world, tr);
  A.init_tiles([](const TA::Range& tile_range) {
    TA::detail::ArenaToTBuilder<Outer> b(tile_range);
    const std::size_t n = tile_range.volume();
    for (std::size_t ord = 0; ord < n; ++ord) {
      // inner extent discovered per cell (jagged) -- no pre-walk
      const std::size_t inner = 2 + ord;
      Inner& cell = b.emplace(ord, TA::Range{static_cast<long>(inner)});
      for (std::size_t i = 0; i < inner; ++i)
        cell.data()[i] = double(ord * 10 + i);
    }
    return std::move(b).finish();
  });
  world.gop.fence();
  BOOST_CHECK_EQUAL(A.trange().tiles_range().volume(), 2u);
  for (std::size_t t = 0; t < 2; ++t) {
    if (!A.is_local(t)) continue;
    Outer tile = A.find(t).get();
    const std::size_t n = tile.range().volume();
    for (std::size_t ord = 0; ord < n; ++ord) {
      const Inner& cell = tile.data()[ord];
      BOOST_REQUIRE(bool(cell));
      BOOST_CHECK_EQUAL(cell.size(), 2u + ord);
      for (std::size_t i = 0; i < cell.size(); ++i)
        BOOST_CHECK_EQUAL(cell.data()[i], double(ord * 10 + i));
    }
  }
}

// Mixed scalar/ArenaTensor outer Hadamard: each scalar-side outer cell
// multiplies the corresponding ArenaTensor-side inner element-wise.
// Exercises Tensor<ArenaTensor>::mult(Tensor<scalar>) and the symmetric
// Tensor<scalar>::mult(Tensor<ArenaTensor>).
BOOST_AUTO_TEST_CASE(mixed_outer_mult_scalar_times_arena) {
  using Scalars = TA::Tensor<double>;
  // 3 outer cells, each inner of size 4, base value 1.0 + ord*100 + i.
  Outer A = make_outer(3, 4, 1.0);
  Scalars S(TA::Range{3});
  S.at_ordinal(0) = 2.0;
  S.at_ordinal(1) = -1.5;
  S.at_ordinal(2) = 0.25;

  // Tensor<ArenaTensor> * Tensor<scalar>
  Outer prod_as = A.mult(S);
  BOOST_REQUIRE_EQUAL(prod_as.range().volume(), 3u);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const Inner& a = A.data()[ord];
    const Inner& d = prod_as.data()[ord];
    BOOST_REQUIRE(bool(d));
    BOOST_REQUIRE_EQUAL(d.size(), a.size());
    // Result must be independent of the source slab.
    BOOST_CHECK_NE(d.data(), a.data());
    for (std::size_t i = 0; i < a.size(); ++i)
      BOOST_CHECK_CLOSE(d.data()[i], a.data()[i] * S.at_ordinal(ord), 1e-12);
  }

  // Tensor<scalar> * Tensor<ArenaTensor>
  Outer prod_sa = S.mult(A);
  BOOST_REQUIRE_EQUAL(prod_sa.range().volume(), 3u);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const Inner& a = A.data()[ord];
    const Inner& d = prod_sa.data()[ord];
    BOOST_REQUIRE(bool(d));
    BOOST_REQUIRE_EQUAL(d.size(), a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      BOOST_CHECK_CLOSE(d.data()[i], S.at_ordinal(ord) * a.data()[i], 1e-12);
  }
}

// Mixed mult preserves null cells coming from the arena side.
BOOST_AUTO_TEST_CASE(mixed_outer_mult_preserves_null_cells) {
  using Scalars = TA::Tensor<double>;
  TA::Range outer_r{4};
  auto shape_fn = [](std::size_t ord) {
    return ord % 2 == 0 ? TA::Range{4} : TA::Range();
  };
  Outer A = TA::detail::arena_outer_init<Outer>(outer_r, 1, shape_fn);
  for (std::size_t ord = 0; ord < 4; ++ord) {
    Inner& inner = A.data()[ord];
    if (inner)
      for (std::size_t i = 0; i < inner.size(); ++i) inner.data()[i] = 1.0;
  }
  Scalars S(TA::Range{4});
  S.at_ordinal(0) = 3.0;
  S.at_ordinal(1) = 7.0;
  S.at_ordinal(2) = -2.0;
  S.at_ordinal(3) = 11.0;

  Outer prod = A.mult(S);
  for (std::size_t ord = 0; ord < 4; ++ord) {
    const Inner& d = prod.data()[ord];
    if (ord % 2 == 0) {
      BOOST_REQUIRE(bool(d));
      for (std::size_t i = 0; i < d.size(); ++i)
        BOOST_CHECK_CLOSE(d.data()[i], 1.0 * S.at_ordinal(ord), 1e-12);
    } else {
      BOOST_CHECK(!d);
    }
  }
}

// Mixed scalar/ArenaTensor add/subt: scalar broadcast across each inner.
BOOST_AUTO_TEST_CASE(mixed_outer_add_subt_scalar_and_arena) {
  using Scalars = TA::Tensor<double>;
  Outer A = make_outer(3, 4, 1.0);
  Scalars S(TA::Range{3});
  S.at_ordinal(0) = 10.0;
  S.at_ordinal(1) = -2.0;
  S.at_ordinal(2) = 0.5;

  // ToT + scalar  → broadcast scalar across each inner element.
  Outer sum_as = A.add(S);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const Inner& a = A.data()[ord];
    const Inner& d = sum_as.data()[ord];
    BOOST_REQUIRE(bool(d));
    for (std::size_t i = 0; i < a.size(); ++i)
      BOOST_CHECK_CLOSE(d.data()[i], a.data()[i] + S.at_ordinal(ord), 1e-12);
  }
  // scalar + ToT  → symmetric.
  Outer sum_sa = S.add(A);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const Inner& a = A.data()[ord];
    const Inner& d = sum_sa.data()[ord];
    BOOST_REQUIRE(bool(d));
    for (std::size_t i = 0; i < a.size(); ++i)
      BOOST_CHECK_CLOSE(d.data()[i], S.at_ordinal(ord) + a.data()[i], 1e-12);
  }
  // ToT - scalar  → subtract per-cell scalar.
  Outer diff_as = A.subt(S);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const Inner& a = A.data()[ord];
    const Inner& d = diff_as.data()[ord];
    BOOST_REQUIRE(bool(d));
    for (std::size_t i = 0; i < a.size(); ++i)
      BOOST_CHECK_CLOSE(d.data()[i], a.data()[i] - S.at_ordinal(ord), 1e-12);
  }
  // scalar - ToT.
  Outer diff_sa = S.subt(A);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const Inner& a = A.data()[ord];
    const Inner& d = diff_sa.data()[ord];
    BOOST_REQUIRE(bool(d));
    for (std::size_t i = 0; i < a.size(); ++i)
      BOOST_CHECK_CLOSE(d.data()[i], S.at_ordinal(ord) - a.data()[i], 1e-12);
  }
}

// `Tensor<ArenaTensor>` should support the same reductions as a flat tensor
// (sum / product / squared_norm / min / max), routed through TA's ToT
// reduce path via the `is_tensor_of_tensor_helper` extension.
// Sanity-check the trait flip:
//  - is_arena_tensor_v<ArenaTensor<...>> must be true
//  - is_tensor_of_tensor_v<Tensor<ArenaTensor>> must be true (was false)
//  - is_tensor_v<Tensor<ArenaTensor>> must be false (was true)
static_assert(TA::is_arena_tensor_v<Inner>);
static_assert(TA::detail::is_tensor_of_tensor_v<Outer>);
static_assert(!TA::detail::is_tensor_v<Outer>);
// And view-aware in-place ops must work for Tensor<ArenaTensor>.
// Confirm prerequisite traits hold:
static_assert(TA::is_tensor_view_v<Inner>, "ArenaTensor must be a tensor view");
static_assert(TA::is_tensor_view_v<typename Outer::value_type>,
              "Outer's value_type (ArenaTensor) must be a tensor view");

// Spot-check that the legacy in-place ops which use `is_tensor<Right>`
// SFINAE *do not* match for `Tensor<ArenaTensor>` after the trait flip.
// If they did, instantiating them would fail (no operator-= on
// ArenaTensor). Probe via `has_member_function_subt_to_anyreturn_v`.

// Smoke test: in-place ops on Tensor<ArenaTensor> compile and execute.
BOOST_AUTO_TEST_CASE(tot_inplace_ops_smoketest) {
  Outer a = TA::detail::arena_outer_init<Outer>(
      TA::Range{2}, 1, [](std::size_t) { return TA::Range{3}; });
  for (std::size_t ord = 0; ord < 2; ++ord)
    for (std::size_t i = 0; i < 3; ++i) a.data()[ord].data()[i] = 1.0;
  Outer b = TA::detail::arena_outer_init<Outer>(
      TA::Range{2}, 1, [](std::size_t) { return TA::Range{3}; });
  for (std::size_t ord = 0; ord < 2; ++ord)
    for (std::size_t i = 0; i < 3; ++i) b.data()[ord].data()[i] = 2.0;
  a.add_to(b);  // expect 3.0 elements
  for (std::size_t ord = 0; ord < 2; ++ord)
    for (std::size_t i = 0; i < 3; ++i)
      BOOST_CHECK_CLOSE(a.data()[ord].data()[i], 3.0, 1e-12);
  a.subt_to(b);  // back to 1.0
  for (std::size_t ord = 0; ord < 2; ++ord)
    for (std::size_t i = 0; i < 3; ++i)
      BOOST_CHECK_CLOSE(a.data()[ord].data()[i], 1.0, 1e-12);
  a.mult_to(b);  // 2.0
  for (std::size_t ord = 0; ord < 2; ++ord)
    for (std::size_t i = 0; i < 3; ++i)
      BOOST_CHECK_CLOSE(a.data()[ord].data()[i], 2.0, 1e-12);
  a.scale_to(0.5);  // 1.0
  for (std::size_t ord = 0; ord < 2; ++ord)
    for (std::size_t i = 0; i < 3; ++i)
      BOOST_CHECK_CLOSE(a.data()[ord].data()[i], 1.0, 1e-12);
  a.neg_to();  // -1.0
  for (std::size_t ord = 0; ord < 2; ++ord)
    for (std::size_t i = 0; i < 3; ++i)
      BOOST_CHECK_CLOSE(a.data()[ord].data()[i], -1.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(tot_reductions_match_flat_aggregate) {
  using Inner = TA::ArenaTensor<double, TA::Range>;
  using Outer = TA::Tensor<Inner>;
  Outer a = TA::detail::arena_outer_init<Outer>(
      TA::Range{3}, 1, [](std::size_t /*ord*/) { return TA::Range{4}; });
  double expected_sum = 0.0;
  double expected_product = 1.0;
  double expected_sq_norm = 0.0;
  for (std::size_t ord = 0; ord < 3; ++ord) {
    Inner& inner = a.data()[ord];
    for (std::size_t i = 0; i < inner.size(); ++i) {
      const double v = 1.0 + ord * 10.0 + i;  // deterministic, all positive
      inner.data()[i] = v;
      expected_sum += v;
      expected_product *= v;
      expected_sq_norm += v * v;
    }
  }
  BOOST_CHECK_CLOSE(a.sum(), expected_sum, 1e-12);
  BOOST_CHECK_CLOSE(a.product(), expected_product, 1e-12);
  BOOST_CHECK_CLOSE(a.squared_norm(), expected_sq_norm, 1e-12);
  BOOST_CHECK_CLOSE(a.norm(), std::sqrt(expected_sq_norm), 1e-12);
}

// axpy_to on Tensor<ArenaTensor>: verifies axpy semantics
// (factor scales only the added operand, not the existing result) —
// distinct from add_to(right, factor) which is `(result + right) * factor`.
BOOST_AUTO_TEST_CASE(tot_axpy_to_accumulates_scaled_operand) {
  Outer result = make_outer(3, 4, 10.0);
  std::vector<std::vector<double>> initial(3, std::vector<double>(4));
  for (std::size_t ord = 0; ord < 3; ++ord)
    for (std::size_t i = 0; i < 4; ++i)
      initial[ord][i] = result.data()[ord].data()[i];
  Outer arg = make_outer(3, 4, 1.0);
  const double factor = 0.5;
  using TiledArray::axpy_to;
  axpy_to(result, arg, factor);
  for (std::size_t ord = 0; ord < 3; ++ord) {
    const Inner& a = arg.data()[ord];
    const Inner& d = result.data()[ord];
    for (std::size_t i = 0; i < a.size(); ++i)
      BOOST_CHECK_CLOSE(d.data()[i], initial[ord][i] + a.data()[i] * factor,
                        1e-12);
  }
}

BOOST_AUTO_TEST_SUITE_END()
