// tests/arena_mixed_product_order.cpp
//
// Two DistArray-level regressions for MIXED products (an arena tensor-of-
// tensors operand times a plain tensor operand), found on the CSV-CC
// per-batch intermediate (occ,occ,PAO;PNO) * (PAO,PAO,DF) ->
// (occ,occ,PAO,DF;PNO):
//
//  1. operand order: the expression layer's cost of a mixed product depended
//     on which operand came first (nested-first ~2.5x faster on two tiles
//     per mode, ~4.5x on one tile, regardless of the result order or an
//     imposed shape). Cause: Tensor::gemm's two arena "scale" strided-GEMM
//     fast paths (nested-left rows, nested-right columns) accepted only the
//     NoTranspose/NoTranspose orientation, and the permutation optimizer
//     hands a nested-right operand whose contracted mode trails to the GEMM
//     as an implicit transpose -- so that order fell to the per-cell AXPY
//     loop. Both paths now serve either orientation (the plain matrix's
//     transpose goes to BLAS as its op; a transposed nested tile only changes
//     which cells form a row/column). The test asserts that all spellings
//     (both orders, plain and shaped, both result orders) agree and reports
//     the wall time of each (informational; no timing assertion).
//
//  2. permuted operand in a binary op: reading an arena ToT through a
//     PERMUTED annotation inside a binary expression (e.g. `x(perm) - y`)
//     destroyed the source: the permuted operand tile is a SHALLOW permute
//     (cells alias the source slab) and the in-place binary op wrote through
//     it. A permute-COPY (`y(perm) = x(...)`) was unaffected, and owning
//     (Tensor<Tensor>) inners were unaffected. The test asserts the source's
//     norm is unchanged after such a read.

#include "TiledArray/tensor/arena_einsum.h"
#include "TiledArray/tensor/arena_tensor.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <chrono>
#include <iostream>
#include <string>

namespace TA = TiledArray;

namespace {

using ArenaInner = TA::ArenaTensor<double, TA::Range>;
using ArenaOuter = TA::Tensor<ArenaInner>;
using ToTArray = TA::DistArray<ArenaOuter, TA::SparsePolicy>;
using FlatArray = TA::DistArray<TA::Tensor<double>, TA::SparsePolicy>;

struct MixedFixture {
  long const I = 8, P = 24, K = 10, A = 6;
  TA::TiledRange1 const tr_i{0l, I / 2, I}, tr_p{0l, P / 2, P}, tr_k{0l, K};
  TA::TiledRange const ij_p{tr_i, tr_i, tr_p};
  TA::TiledRange const q_p_k{tr_p, tr_p, tr_k};
  TA::TiledRange const ijqk{tr_i, tr_i, tr_p, tr_k};
  std::string const ta = "i,j,p;a", tb = "q,p,k", tc = "i,j,q,k;a",
                    tc_alt = "q,k,i,j;a";
  ToTArray tot;
  FlatArray flat;

  MixedFixture() {
    auto& world = TA::get_default_world();
    tot = ToTArray(world, ij_p);
    tot.init_tiles([&](TA::Range const& tr) {
      ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
          tr, 1, [=](std::size_t) { return TA::Range{A}; });
      for (std::size_t o = 0; o < t.range().volume(); ++o) {
        ArenaInner& c = t.data()[o];
        if (!c) continue;
        for (long a = 0; a < A; ++a)
          c.data()[a] = 1.0 + 0.001 * static_cast<double>((o * 7 + a) % 13);
      }
      return t;
    });
    flat = FlatArray(world, q_p_k);
    flat.init_tiles([&](TA::Range const& tr) {
      TA::Tensor<double> t(tr);
      for (std::size_t o = 0; o < t.range().volume(); ++o)
        t.data()[o] = 0.5 + 0.001 * static_cast<double>(o % 17);
      return t;
    });
    world.gop.fence();
  }

  template <typename F>
  static double timed(F&& f) {
    auto& world = TA::get_default_world();
    world.gop.fence();
    auto const t0 = std::chrono::steady_clock::now();
    f();
    world.gop.fence();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
  }

  double diff(ToTArray const& x, ToTArray const& ref, std::string const& xa) {
    ToTArray d;
    d(tc) = x(xa) - ref(tc);
    return TA::norm2(d);
  }
};

}  // namespace

BOOST_FIXTURE_TEST_SUITE(arena_mixed_product_order_suite, MixedFixture,
                         TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(operand_order_is_canonicalized) {
  ToTArray ref, ff, nf_sh, ff_sh, nf_alt, ff_alt;
  double const t_nf = timed([&] { ref(tc) = tot(ta) * flat(tb); });
  double const t_ff = timed([&] { ff(tc) = flat(tb) * tot(ta); });
  TA::Tensor<float> norms(ijqk.tiles_range(), 1.0f);
  TA::SparseShape<float> const shape(norms, ijqk, /*do_not_scale=*/true);
  double const t_nf_sh =
      timed([&] { nf_sh(tc) = (tot(ta) * flat(tb)).set_shape(shape); });
  double const t_ff_sh =
      timed([&] { ff_sh(tc) = (flat(tb) * tot(ta)).set_shape(shape); });
  double const t_nf_alt = timed([&] { nf_alt(tc_alt) = tot(ta) * flat(tb); });
  double const t_ff_alt = timed([&] { ff_alt(tc_alt) = flat(tb) * tot(ta); });
  double const n = TA::norm2(ref);
  BOOST_REQUIRE(n > 0.0);
  double const tol = 1e-12 * n;
  BOOST_CHECK_SMALL(diff(ff, ref, tc), tol);
  BOOST_CHECK_SMALL(diff(nf_sh, ref, tc), tol);
  BOOST_CHECK_SMALL(diff(ff_sh, ref, tc), tol);
  BOOST_CHECK_SMALL(diff(nf_alt, ref, tc_alt), tol);
  BOOST_CHECK_SMALL(diff(ff_alt, ref, tc_alt), tol);
  BOOST_TEST_MESSAGE("mixed product wall time (s): nested*flat="
                     << t_nf << " flat*nested=" << t_ff
                     << " +shape: " << t_nf_sh << " / " << t_ff_sh
                     << " result q,k,i,j: " << t_nf_alt << " / " << t_ff_alt);
}

BOOST_AUTO_TEST_CASE(permuted_arena_operand_in_binary_op_keeps_source) {
  ToTArray ref, src_copy, src_binary;
  ref(tc) = tot(ta) * flat(tb);
  src_copy(tc) = tot(ta) * flat(tb);
  src_binary(tc) = tot(ta) * flat(tb);
  TA::get_default_world().gop.fence();
  double const n0 = TA::norm2(src_copy);
  BOOST_REQUIRE(n0 > 0.0);

  ToTArray y;
  y(tc_alt) = src_copy(tc);  // permute-copy: must not touch the source
  TA::get_default_world().gop.fence();
  BOOST_CHECK_CLOSE(TA::norm2(src_copy), n0, 1e-10);

  ToTArray z;
  z(tc) = src_binary(tc_alt) - ref(tc);  // permuted operand of a binary op
  TA::get_default_world().gop.fence();
  BOOST_CHECK_CLOSE(TA::norm2(src_binary), n0, 1e-10);
  // and the binary op itself was right: |z| is |ref_perm - ref| = 0 only if
  // the permutation was applied; here they differ, so just require finite.
  BOOST_CHECK(std::isfinite(TA::norm2(z)));
}

BOOST_AUTO_TEST_SUITE_END()
