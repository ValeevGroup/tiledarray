/// \file libxsmm_gemm.cpp
/// The ONLY translation unit that includes <libxsmm.h>. Isolating the libxsmm
/// include here keeps its macros (libxsmm_macros.h) from leaking into any
/// TiledArray header. When TILEDARRAY_HAS_LIBXSMM is undefined, libxsmm_gemm_le64
/// is still defined here as a `return false` stub so callers link unconditionally.

#include "TiledArray/math/libxsmm_gemm.h"

#ifdef TILEDARRAY_HAS_LIBXSMM
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <libxsmm.h>
#endif

namespace TiledArray::detail {

/// max(M,N,K) cutoff above which we keep using the vendor BLAS.
static constexpr std::int64_t libxsmm_gemm_max_dim = 64;

#ifdef TILEDARRAY_HAS_LIBXSMM
/// Runtime master switch for the libxsmm fast path. Even in a libxsmm-enabled
/// build, exporting `TA_LIBXSMM=0` (also accepts `off`/`OFF`/`false`/`no`)
/// routes EVERY strided micro-GEMM back through the vendor BLAS path, i.e.
/// libxsmm_gemm_le64() returns false for all shapes. Unset or any other value
/// => libxsmm ON. Read from the environment once, on first use, and cached.
static bool libxsmm_runtime_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("TA_LIBXSMM");
    if (v == nullptr || *v == '\0') return true;  // default ON when compiled in
    return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 ||
             std::strcmp(v, "OFF") == 0 || std::strcmp(v, "false") == 0 ||
             std::strcmp(v, "no") == 0);
  }();
  return enabled;
}
#endif

bool libxsmm_gemm_le64(bool trans_a, bool trans_b, std::int64_t m,
                       std::int64_t n, std::int64_t k, double alpha,
                       const double* a, std::int64_t lda, const double* b,
                       std::int64_t ldb, double beta, double* c,
                       std::int64_t ldc) {
#ifdef TILEDARRAY_HAS_LIBXSMM
  // Runtime master switch: TA_LIBXSMM=0 sends everything back to vendor BLAS.
  if (!libxsmm_runtime_enabled()) return false;
  // libxsmm only for small shapes; max(M,N,K) <= 64.
  if (m > libxsmm_gemm_max_dim || n > libxsmm_gemm_max_dim ||
      k > libxsmm_gemm_max_dim)
    return false;
  // libxsmm SMM has no alpha and only beta in {0,1} (LIBXSMM_GEMM_NO_BYPASS).
  if (alpha != 1.0) return false;
  if (beta != 0.0 && beta != 1.0) return false;
  // libxsmm_blasint is 32-bit; refuse leading dims that would narrow silently.
  // (M,N,K are already <=64; lda/ldb/ldc are strides and unbounded in general.)
  constexpr std::int64_t bi_max =
      static_cast<std::int64_t>(std::numeric_limits<libxsmm_blasint>::max());
  if (lda > bi_max || ldb > bi_max || ldc > bi_max) return false;

  static std::once_flag init_flag;
  std::call_once(init_flag, [] {
    // libxsmm's own verbose dispatch/JIT stats are part of the profiling
    // result, so fold them into TA_PROFILE: when profiling is on and the user
    // has NOT pinned LIBXSMM_VERBOSE explicitly, enable libxsmm verbosity here,
    // BEFORE libxsmm_init() parses the environment. TA_PROFILE>=1 -> a concise
    // exit summary (version + registry "gemm=<n>" kernel count); TA_PROFILE>=2
    // -> verbose per-kernel JIT events. Must run before libxsmm_init().
    if (std::getenv("LIBXSMM_VERBOSE") == nullptr) {
      const char* p = std::getenv("TA_PROFILE");
      const int lvl = (p != nullptr) ? std::atoi(p) : 0;
      if (lvl >= 2)
        setenv("LIBXSMM_VERBOSE", "3", /*overwrite=*/0);
      else if (lvl >= 1)
        setenv("LIBXSMM_VERBOSE", "2", /*overwrite=*/0);
    }
    libxsmm_init();
  });

  // Mirror blas::gemm's row-major -> col-major mapping: it realizes the result
  // as a column-major GEMM (op_b, op_a, n, m, k) with operands (b, a) swapped.
  // libxsmm is column-major, so: A'=b (ld=ldb), B'=a (ld=lda), dims (n, m, k),
  // TRANS_A from op_b, TRANS_B from op_a.
  const libxsmm_bitfield flags =
      (trans_b ? LIBXSMM_GEMM_FLAG_TRANS_A : 0) |
      (trans_a ? LIBXSMM_GEMM_FLAG_TRANS_B : 0) |
      (beta == 0.0 ? LIBXSMM_GEMM_FLAG_BETA_0 : 0);

  const libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(
      static_cast<libxsmm_blasint>(n), static_cast<libxsmm_blasint>(m),
      static_cast<libxsmm_blasint>(k), static_cast<libxsmm_blasint>(ldb),
      static_cast<libxsmm_blasint>(lda), static_cast<libxsmm_blasint>(ldc),
      LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
      LIBXSMM_DATATYPE_F64);

  const libxsmm_gemmfunction kernel = libxsmm_dispatch_gemm(
      shape, flags, static_cast<libxsmm_bitfield>(LIBXSMM_GEMM_PREFETCH_NONE));
  if (kernel == nullptr) return false;  // shape not JIT-able -> fall back

  libxsmm_gemm_param param;
  std::memset(&param, 0, sizeof param);
  param.a.primary = const_cast<double*>(b);  // A' = b
  param.b.primary = const_cast<double*>(a);  // B' = a
  param.c.primary = c;
  kernel(&param);
  return true;
#else
  (void)trans_a; (void)trans_b; (void)m; (void)n; (void)k; (void)alpha;
  (void)a; (void)lda; (void)b; (void)ldb; (void)beta; (void)c; (void)ldc;
  return false;
#endif
}

}  // namespace TiledArray::detail
