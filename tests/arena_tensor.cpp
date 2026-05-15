/// Unit tests for TiledArray::ArenaTensor: null state, view copy/move,
/// foreign-tensor assignment, in-place CPOs, materialize.

#include "TiledArray/tensor/arena_tensor.h"

#include "TiledArray/external/btas.h"
#include "TiledArray/tensor.h"
#include "TiledArray/tensor/tensor_map.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <btas/tensorview.h>
#include <btas/zb/range.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace TA = TiledArray;
// Tests use TA::Range explicitly so the standalone target for materialize()
// is the natural TA::Tensor<double>; the type's default range is
// btas::zb::RangeNd, which pairs with btas::Tensor as the standalone.
using Inner = TA::ArenaTensor<double, TA::Range>;

namespace {

/// Holds an over-aligned byte buffer big enough for a single `Inner` cell of
/// `n` elements.
struct CellBuf {
  std::vector<std::byte> bytes;
  std::byte* aligned_ptr = nullptr;

  explicit CellBuf(std::size_t n_elems) {
    const std::size_t total = Inner::cell_size(n_elems);
    const std::size_t algn = Inner::cell_alignment();
    bytes.assign(total + algn, std::byte{0});
    auto base = reinterpret_cast<std::uintptr_t>(bytes.data());
    auto aligned = (base + algn - 1) & ~(algn - 1);
    aligned_ptr = reinterpret_cast<std::byte*>(aligned);
  }
};

}  // namespace

BOOST_AUTO_TEST_SUITE(arena_tensor_suite, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(sizeof_is_one_pointer) {
  BOOST_CHECK_EQUAL(sizeof(Inner), sizeof(void*));
}

BOOST_AUTO_TEST_CASE(sizeof_invariant_across_range_parameterizations) {
  // `ArenaTensor`'s footprint must be one pointer regardless of the range
  // template parameter -- this is the original motivation for the type. The
  // default `btas::zb::RangeNd<>` (~14 B + alignment) and `TA::Range`
  // (~300 B) both go behind the same `Cell*` indirection.
  static_assert(sizeof(TA::ArenaTensor<double>) == sizeof(void*),
                "default-range ArenaTensor<double> must be one pointer");
  static_assert(
      sizeof(TA::ArenaTensor<double, ::btas::zb::RangeNd<>>) == sizeof(void*),
      "zb::RangeNd ArenaTensor<double> must be one pointer");
  static_assert(sizeof(TA::ArenaTensor<double, TA::Range>) == sizeof(void*),
                "TA::Range ArenaTensor<double> must be one pointer");
  // Different element type same story.
  static_assert(sizeof(TA::ArenaTensor<float>) == sizeof(void*));
  static_assert(sizeof(TA::ArenaTensor<std::complex<double>>) == sizeof(void*));
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(element_data_is_simd_aligned) {
  // data_alignment() should be at least kInnerSimdAlign; cell_alignment()
  // should propagate that so the element pointer is SIMD-aligned.
  BOOST_CHECK(Inner::data_alignment() >= TA::kInnerSimdAlign);
  BOOST_CHECK_EQUAL(Inner::data_alignment() % TA::kInnerSimdAlign, 0u);
  BOOST_CHECK(Inner::cell_alignment() >= Inner::data_alignment());
  CellBuf buf(8);
  Inner x =
      TA::detail::make_arena_tensor_in<double>(buf.aligned_ptr, TA::Range{8});
  auto addr = reinterpret_cast<std::uintptr_t>(x.data());
  BOOST_CHECK_EQUAL(addr % TA::kInnerSimdAlign, 0u);
}

BOOST_AUTO_TEST_CASE(default_constructed_is_null) {
  Inner x;
  BOOST_CHECK(!x);
  BOOST_CHECK(x.empty());
  BOOST_CHECK_EQUAL(x.size(), 0u);
  BOOST_CHECK(x.data() == nullptr);
}

BOOST_AUTO_TEST_CASE(make_arena_tensor_zero_initialized) {
  CellBuf buf(6);
  Inner x =
      TA::detail::make_arena_tensor_in<double>(buf.aligned_ptr, TA::Range{6});
  BOOST_REQUIRE(bool(x));
  BOOST_CHECK(!x.empty());
  BOOST_CHECK_EQUAL(x.size(), 6u);
  for (std::size_t i = 0; i < x.size(); ++i)
    BOOST_CHECK_EQUAL(x.data()[i], 0.0);
}

BOOST_AUTO_TEST_CASE(copy_construction_yields_alias) {
  CellBuf buf(4);
  Inner x =
      TA::detail::make_arena_tensor_in<double>(buf.aligned_ptr, TA::Range{4});
  Inner y = x;
  BOOST_CHECK(bool(x));
  BOOST_CHECK(bool(y));
  BOOST_CHECK_EQUAL(x.data(), y.data());
  y.data()[0] = 42.0;
  BOOST_CHECK_EQUAL(x.data()[0], 42.0);
}

BOOST_AUTO_TEST_CASE(move_leaves_source_null) {
  CellBuf buf(4);
  Inner x =
      TA::detail::make_arena_tensor_in<double>(buf.aligned_ptr, TA::Range{4});
  Inner y = std::move(x);
  BOOST_CHECK(!x);
  BOOST_CHECK(bool(y));
  BOOST_CHECK_EQUAL(y.size(), 4u);
}

BOOST_AUTO_TEST_CASE(operator_assign_from_ta_tensor_copies_elements) {
  CellBuf buf(5);
  Inner x =
      TA::detail::make_arena_tensor_in<double>(buf.aligned_ptr, TA::Range{5});
  TA::Tensor<double> src(TA::Range{5}, 0.0);
  for (std::size_t i = 0; i < 5; ++i) src.data()[i] = double(i + 1);
  x = src;
  for (std::size_t i = 0; i < 5; ++i)
    BOOST_CHECK_EQUAL(x.data()[i], double(i + 1));
}

BOOST_AUTO_TEST_CASE(zero_fills_with_zeros) {
  CellBuf buf(4);
  Inner x =
      TA::detail::make_arena_tensor_in<double>(buf.aligned_ptr, TA::Range{4});
  for (std::size_t i = 0; i < 4; ++i) x.data()[i] = 7.0;
  TA::zero(x);
  for (std::size_t i = 0; i < 4; ++i) BOOST_CHECK_EQUAL(x.data()[i], 0.0);
}

BOOST_AUTO_TEST_CASE(fill_sets_all_elements) {
  CellBuf buf(4);
  Inner x =
      TA::detail::make_arena_tensor_in<double>(buf.aligned_ptr, TA::Range{4});
  TA::fill(x, 3.5);
  for (std::size_t i = 0; i < 4; ++i) BOOST_CHECK_EQUAL(x.data()[i], 3.5);
}

BOOST_AUTO_TEST_CASE(scale_to_multiplies_in_place) {
  CellBuf buf(4);
  Inner x =
      TA::detail::make_arena_tensor_in<double>(buf.aligned_ptr, TA::Range{4});
  TA::fill(x, 2.0);
  TA::scale_to(x, 3.0);
  for (std::size_t i = 0; i < 4; ++i) BOOST_CHECK_EQUAL(x.data()[i], 6.0);
}

BOOST_AUTO_TEST_CASE(add_to_accumulates) {
  CellBuf bd(4), bs(4);
  Inner dst =
      TA::detail::make_arena_tensor_in<double>(bd.aligned_ptr, TA::Range{4});
  Inner src =
      TA::detail::make_arena_tensor_in<double>(bs.aligned_ptr, TA::Range{4});
  TA::fill(dst, 1.0);
  TA::fill(src, 2.0);
  TA::add_to(dst, src);
  for (std::size_t i = 0; i < 4; ++i) BOOST_CHECK_EQUAL(dst.data()[i], 3.0);
}

BOOST_AUTO_TEST_CASE(subt_to_subtracts) {
  CellBuf bd(4), bs(4);
  Inner dst =
      TA::detail::make_arena_tensor_in<double>(bd.aligned_ptr, TA::Range{4});
  Inner src =
      TA::detail::make_arena_tensor_in<double>(bs.aligned_ptr, TA::Range{4});
  TA::fill(dst, 5.0);
  TA::fill(src, 2.0);
  TA::subt_to(dst, src);
  for (std::size_t i = 0; i < 4; ++i) BOOST_CHECK_EQUAL(dst.data()[i], 3.0);
}

BOOST_AUTO_TEST_CASE(mult_to_does_elementwise) {
  CellBuf bd(4), bs(4);
  Inner dst =
      TA::detail::make_arena_tensor_in<double>(bd.aligned_ptr, TA::Range{4});
  Inner src =
      TA::detail::make_arena_tensor_in<double>(bs.aligned_ptr, TA::Range{4});
  TA::fill(dst, 4.0);
  TA::fill(src, 0.5);
  TA::mult_to(dst, src);
  for (std::size_t i = 0; i < 4; ++i) BOOST_CHECK_EQUAL(dst.data()[i], 2.0);
}

BOOST_AUTO_TEST_CASE(axpy_scales_and_adds) {
  CellBuf bd(4), bs(4);
  Inner dst =
      TA::detail::make_arena_tensor_in<double>(bd.aligned_ptr, TA::Range{4});
  Inner src =
      TA::detail::make_arena_tensor_in<double>(bs.aligned_ptr, TA::Range{4});
  TA::fill(dst, 1.0);
  TA::fill(src, 2.0);
  TA::axpy(dst, 3.0, src);
  for (std::size_t i = 0; i < 4; ++i) BOOST_CHECK_EQUAL(dst.data()[i], 7.0);
}

BOOST_AUTO_TEST_CASE(squared_norm_sums_squares) {
  CellBuf buf(3);
  Inner x =
      TA::detail::make_arena_tensor_in<double>(buf.aligned_ptr, TA::Range{3});
  x.data()[0] = 1.0;
  x.data()[1] = 2.0;
  x.data()[2] = 2.0;
  BOOST_CHECK_EQUAL(TA::squared_norm(x), 9.0);
}

BOOST_AUTO_TEST_CASE(in_place_cpos_no_op_on_null) {
  Inner null;
  TA::zero(null);
  TA::fill(null, 1.0);
  TA::scale_to(null, 2.0);
  TA::add_to(null, null);
  BOOST_CHECK_EQUAL(TA::squared_norm(null), 0.0);
}

BOOST_AUTO_TEST_CASE(materialize_returns_independent_standalone) {
  CellBuf buf(4);
  Inner x =
      TA::detail::make_arena_tensor_in<double>(buf.aligned_ptr, TA::Range{4});
  for (std::size_t i = 0; i < 4; ++i) x.data()[i] = double(i);
  auto standalone = TA::materialize<TA::Tensor<double>>(x);
  BOOST_REQUIRE_EQUAL(standalone.range().volume(), 4u);
  for (std::size_t i = 0; i < 4; ++i)
    BOOST_CHECK_EQUAL(standalone.data()[i], double(i));
  standalone.data()[0] = 99.0;
  BOOST_CHECK_EQUAL(x.data()[0], 0.0);
}

BOOST_AUTO_TEST_CASE(materialize_null_yields_empty_standalone) {
  Inner null;
  auto standalone = TA::materialize<TA::Tensor<double>>(null);
  BOOST_CHECK(standalone.empty());
}

BOOST_AUTO_TEST_CASE(is_arena_tensor_v_predicate) {
  static_assert(TA::is_arena_tensor_v<Inner>);
  static_assert(!TA::is_arena_tensor_v<TA::Tensor<double>>);
  static_assert(!TA::is_arena_tensor_v<double>);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(is_tensor_view_v_predicate) {
  // ArenaTensor is a view that lacks value-returning member arithmetic --
  // it cannot allocate on its own. `is_tensor_view_v` is the predicate that
  // opts such types out of value-returning operator dispatch.
  static_assert(TA::is_tensor_view_v<Inner>);
  // btas::TensorView is also a view without member arithmetic.
  static_assert(TA::is_tensor_view_v<btas::TensorView<double>>);
  // TA::TensorMap (TensorInterface) is non-owning, but DOES provide
  // value-returning member arithmetic (it materializes a fresh tensor), so
  // it is intentionally NOT in `is_tensor_view`.
  static_assert(!TA::is_tensor_view_v<TA::TensorMap<double>>);
  static_assert(!TA::is_tensor_view_v<TA::TensorConstMap<double>>);
  // Value-semantic tensors and scalars are not views.
  static_assert(!TA::is_tensor_view_v<TA::Tensor<double>>);
  static_assert(!TA::is_tensor_view_v<btas::Tensor<double>>);
  static_assert(!TA::is_tensor_view_v<double>);
  // Layering: is_arena_tensor_v implies is_tensor_view_v.
  static_assert(!TA::is_arena_tensor_v<TA::TensorMap<double>>);
  static_assert(!TA::is_arena_tensor_v<btas::TensorView<double>>);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(gemm_inner_matrix_product) {
  // C[3,5] += A[3,4] * B[4,5]; A is 1..12 row-major, B is 0.0,0.5,...,9.5.
  CellBuf bl(12), br(20), bc(15);
  Inner left =
      TA::detail::make_arena_tensor_in<double>(bl.aligned_ptr, TA::Range{3, 4});
  Inner right =
      TA::detail::make_arena_tensor_in<double>(br.aligned_ptr, TA::Range{4, 5});
  Inner result =
      TA::detail::make_arena_tensor_in<double>(bc.aligned_ptr, TA::Range{3, 5});
  for (int i = 0; i < 12; ++i) left.data()[i] = double(i + 1);
  for (int i = 0; i < 20; ++i) right.data()[i] = 0.5 * double(i);
  TA::zero(result);

  TA::math::GemmHelper helper(TA::math::blas::NoTranspose,
                              TA::math::blas::NoTranspose, 2, 2, 2);
  TA::gemm(result, left, right, 1.0, helper);

  // Row-major reference: ref[i,k] = sum_j A[i,j] * B[j,k].
  double ref[15] = {0};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 5; ++k)
      for (int j = 0; j < 4; ++j)
        ref[i * 5 + k] += left.data()[i * 4 + j] * right.data()[j * 5 + k];
  for (int i = 0; i < 15; ++i)
    BOOST_CHECK_CLOSE(result.data()[i], ref[i], 1e-12);
}

BOOST_AUTO_TEST_CASE(gemm_inner_accumulates_into_result) {
  // C starts at known nonzero, gemm accumulates (beta=1).
  CellBuf bl(4), br(4), bc(4);
  Inner left =
      TA::detail::make_arena_tensor_in<double>(bl.aligned_ptr, TA::Range{2, 2});
  Inner right =
      TA::detail::make_arena_tensor_in<double>(br.aligned_ptr, TA::Range{2, 2});
  Inner result =
      TA::detail::make_arena_tensor_in<double>(bc.aligned_ptr, TA::Range{2, 2});
  TA::fill(left, 1.0);
  TA::fill(right, 2.0);
  TA::fill(result, 10.0);  // preload

  TA::math::GemmHelper helper(TA::math::blas::NoTranspose,
                              TA::math::blas::NoTranspose, 2, 2, 2);
  TA::gemm(result, left, right, 1.0, helper);
  // Each result entry: 10 (preload) + 2 (sum_j 1*2 over j=0..1) = 14.
  for (int i = 0; i < 4; ++i) BOOST_CHECK_CLOSE(result.data()[i], 14.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(gemm_inner_skips_when_operand_null) {
  // Null operands -> result unchanged (no-op).
  CellBuf bc(4);
  Inner result =
      TA::detail::make_arena_tensor_in<double>(bc.aligned_ptr, TA::Range{2, 2});
  TA::fill(result, 7.0);
  Inner null_inner;
  TA::math::GemmHelper helper(TA::math::blas::NoTranspose,
                              TA::math::blas::NoTranspose, 2, 2, 2);
  TA::gemm(result, null_inner, null_inner, 1.0, helper);
  for (int i = 0; i < 4; ++i) BOOST_CHECK_CLOSE(result.data()[i], 7.0, 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()
