#ifndef TILEDARRAY_MATH_LIBXSMM_GEMM_H__INCLUDED
#define TILEDARRAY_MATH_LIBXSMM_GEMM_H__INCLUDED

/// \file libxsmm_gemm.h
/// Optional libxsmm fast path for small strided tensor-of-tensors micro-GEMMs.
/// This header is DECLARATION-ONLY: the implementation lives in libxsmm_gemm.cpp,
/// which is the single translation unit that includes <libxsmm.h>. Keeping the
/// libxsmm include out of this header is deliberate -- <libxsmm.h> pulls in
/// libxsmm_macros.h, whose macros otherwise leak into TiledArray headers that
/// transitively include this one (e.g. arena_einsum.h -> ... -> math/vector_op.h,
/// breaking detail::is_scalar_v). Callers just see a plain function.

#include <cstdint>

// N.B. namespace TiledArray::detail (NOT TiledArray::math::detail): introducing
// a TiledArray::math::detail namespace would hijack unqualified `detail::` name
// lookup inside TiledArray::math headers (e.g. vector_op.h's detail::is_scalar_v,
// which lives in TiledArray::detail), breaking their compilation.
namespace TiledArray::detail {

/// Computes C(m x n) [+]= alpha * op_a(A) . op_b(B) in **row-major** layout with
/// leading dims lda/ldb/ldc, i.e. exactly TiledArray::math::blas::gemm (double)
/// semantics. \p trans_a / \p trans_b are the transpose flags (true == Trans).
///
/// \return true iff libxsmm performed the GEMM. Returns false (caller must fall
/// back to blas::gemm) when: built without libxsmm (TILEDARRAY_HAS_LIBXSMM
/// undefined), the runtime switch `TA_LIBXSMM=0` is set, max(m,n,k) > 64,
/// alpha != 1, beta not in {0,1}, or libxsmm could not JIT this shape.
bool libxsmm_gemm_le64(bool trans_a, bool trans_b, std::int64_t m,
                       std::int64_t n, std::int64_t k, double alpha,
                       const double* a, std::int64_t lda, const double* b,
                       std::int64_t ldb, double beta, double* c,
                       std::int64_t ldc);

}  // namespace TiledArray::detail

#endif  // TILEDARRAY_MATH_LIBXSMM_GEMM_H__INCLUDED
