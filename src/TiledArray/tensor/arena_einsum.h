/// Arena-aware ToT einsum: plans, fused kernels, and dispatch.

#ifndef TILEDARRAY_TENSOR_ARENA_EINSUM_H__INCLUDED
#define TILEDARRAY_TENSOR_ARENA_EINSUM_H__INCLUDED

#include "TiledArray/error.h"
#include "TiledArray/math/gemm_helper.h"
#include "TiledArray/permutation.h"
#include "TiledArray/tensor/arena.h"
#include "TiledArray/tensor/arena_kernels.h"
#include "TiledArray/tensor/kernels.h"
#include "TiledArray/tensor/type_traits.h"
#include "TiledArray/util/annotation.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#if defined(_MSC_VER) && _MSC_VER < 1937  // VS 2022 < 17.7
#define TA_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define TA_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

namespace TiledArray::detail {

/// Env-gated (TA_STRIDED_DGEMM_VERBOSE) toggle for the strided-DGEMM install
/// logger. Reads the environment once. Set TA_STRIDED_DGEMM_VERBOSE=1 to have
/// the ContEngine print, per ToT contraction, whether a strided-DGEMM regime
/// (hce+e / hc+e / hce+ce) FIRES or REVERTS to the generic by-cell path.
inline bool strided_dgemm_verbose() {
  static const bool enabled = [] {
    const char* e = std::getenv("TA_STRIDED_DGEMM_VERBOSE");
    return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
  }();
  return enabled;
}

/// One-line install-decision logger for the strided-DGEMM regimes. No-op unless
/// strided_dgemm_verbose() (i.e. TA_STRIDED_DGEMM_VERBOSE) is set.
inline void strided_dgemm_log(const char* msg) {
  if (strided_dgemm_verbose()) std::cerr << "[strided-dgemm] " << msg << '\n';
}

/// ===========================================================================
/// GEMM-vs-op timing instrumentation (Amdahl profiling).
///
/// Accumulates the wall-clock nanoseconds spent INSIDE blas::gemm in the
/// strided-DGEMM regimes, separated by regime (ce+e = inner outer-product,
/// ce+ce = inner contraction). The inner-cell loops in each kernel are serial
/// within a single Product op, so summing per-call durations across all ops is
/// directly comparable to the summed per-op "Eval | Product | <ns>ns" trace
/// time (same aggregation across MADNESS task threads). The ratio
/// gemm_ns / product_ns is the Amdahl compute (kernel) fraction.
///
/// Env-gated by TA_GEMM_TIMING=1: when unset the timer takes no clock samples
/// and touches no atomics (zero overhead on production runs). The per-regime
/// totals are printed to stderr at process exit.
inline bool gemm_timing_enabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("TA_GEMM_TIMING");
    return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
  }();
  return enabled;
}

inline std::atomic<std::uint64_t> g_gemm_ns_ce_e{0};
inline std::atomic<std::uint64_t> g_gemm_ns_ce_ce{0};
inline std::atomic<std::uint64_t> g_gemm_calls_ce_e{0};
inline std::atomic<std::uint64_t> g_gemm_calls_ce_ce{0};

/// RAII timer scoping a single blas::gemm. No-op (no clock read, no atomic
/// touch) unless TA_GEMM_TIMING is set.
class ScopedGemmTimer {
 public:
  ScopedGemmTimer(std::atomic<std::uint64_t>& ns_acc,
                  std::atomic<std::uint64_t>& call_acc)
      : ns_(gemm_timing_enabled() ? &ns_acc : nullptr), calls_(&call_acc) {
    if (ns_) t0_ = std::chrono::steady_clock::now();
  }
  ~ScopedGemmTimer() {
    if (!ns_) return;
    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0_)
                        .count();
    ns_->fetch_add(static_cast<std::uint64_t>(dt), std::memory_order_relaxed);
    calls_->fetch_add(1, std::memory_order_relaxed);
  }
  ScopedGemmTimer(const ScopedGemmTimer&) = delete;
  ScopedGemmTimer& operator=(const ScopedGemmTimer&) = delete;

 private:
  std::atomic<std::uint64_t>* ns_;
  std::atomic<std::uint64_t>* calls_;
  std::chrono::steady_clock::time_point t0_;
};

/// ---------------------------------------------------------------------------
/// Per-shape GEMM histogram for the ce+e regime (inner outer-product).
///
/// Each ce+e strided GEMM is a (P x Q) result accumulated over K, i.e.
/// gemm dims M=P, N=Q, K=K. We bucket calls by the exact (P,Q,K) triple and
/// tally count + wall ns per bucket, so we can see whether the inner extents
/// in a given molecule are large enough to feed an efficient DGEMM (the bench
/// used K~256; CSV/PNO domains in small molecules may be far smaller).
///
/// To stay lock-free in the hot loop, each thread accumulates into its own
/// heap-allocated map (intentionally leaked so the pointer survives MADNESS
/// worker-thread teardown); the exit dumper merges all registered maps.
struct GemmShapeMap {
  // key = M,N,K packed (21 bits each); value = {count, total ns}.
  std::unordered_map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> m;
};

/// A per-regime shape registry: a mutex + a list of per-thread maps merged at
/// exit. Two instances exist (ce+e, ce+ce).
struct ShapeRegistry {
  std::mutex mtx;
  std::vector<GemmShapeMap*> maps;
};
inline ShapeRegistry g_ce_e_shapes;
inline ShapeRegistry g_ce_ce_shapes;

/// This thread's bucket map for the given registry (heap-allocated + leaked so
/// the pointer survives MADNESS worker-thread teardown). The per-thread lookup
/// list holds at most two entries (ce+e, ce+ce), so the linear scan is trivial.
inline GemmShapeMap& tls_shapes(ShapeRegistry& reg) {
  thread_local std::vector<std::pair<ShapeRegistry*, GemmShapeMap*>> mine;
  for (auto& p : mine)
    if (p.first == &reg) return *p.second;
  auto* a = new GemmShapeMap();
  {
    std::lock_guard<std::mutex> lk(reg.mtx);
    reg.maps.push_back(a);
  }
  mine.emplace_back(&reg, a);
  return *a;
}

inline std::uint64_t pack_shape(std::size_t M, std::size_t N, std::size_t K) {
  constexpr std::uint64_t MASK = (std::uint64_t{1} << 21) - 1;  // up to 2,097,151
  return ((static_cast<std::uint64_t>(M) & MASK) << 42) |
         ((static_cast<std::uint64_t>(N) & MASK) << 21) |
         (static_cast<std::uint64_t>(K) & MASK);
}

/// Records one GEMM's (M,N,K) shape into the calling thread's bucket for `reg`.
inline void record_shape(ShapeRegistry& reg, std::size_t M, std::size_t N,
                         std::size_t K, std::uint64_t ns) {
  auto& e = tls_shapes(reg).m[pack_shape(M, N, K)];
  e.first += 1;
  e.second += ns;
}

/// RAII timer wrapping one blas::gemm that accumulates wall ns into the given
/// regime totals AND records the (M,N,K) shape into the given registry. No-op
/// (no clock read, no map touch) unless TA_GEMM_TIMING is set.
class ScopedShapedGemmTimer {
 public:
  ScopedShapedGemmTimer(std::atomic<std::uint64_t>& ns_acc,
                        std::atomic<std::uint64_t>& call_acc, ShapeRegistry& reg,
                        std::size_t M, std::size_t N, std::size_t K)
      : on_(gemm_timing_enabled()),
        ns_(&ns_acc),
        calls_(&call_acc),
        reg_(&reg),
        M_(M),
        N_(N),
        K_(K) {
    if (on_) t0_ = std::chrono::steady_clock::now();
  }
  ~ScopedShapedGemmTimer() {
    if (!on_) return;
    const auto dt = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0_)
            .count());
    ns_->fetch_add(dt, std::memory_order_relaxed);
    calls_->fetch_add(1, std::memory_order_relaxed);
    record_shape(*reg_, M_, N_, K_, dt);
  }
  ScopedShapedGemmTimer(const ScopedShapedGemmTimer&) = delete;
  ScopedShapedGemmTimer& operator=(const ScopedShapedGemmTimer&) = delete;

 private:
  bool on_;
  std::atomic<std::uint64_t>* ns_;
  std::atomic<std::uint64_t>* calls_;
  ShapeRegistry* reg_;
  std::size_t M_, N_, K_;
  std::chrono::steady_clock::time_point t0_;
};

/// ---------------------------------------------------------------------------
/// Phase decomposition of the ce+ce kernel time (where does the non-GEMM
/// overhead go?). All gated by TA_GEMM_TIMING; printed at exit.
///   kernel_total = whole arena_strided_dgemm_ce_ce_{right,left} body
///   gemm         = g_gemm_ns_ce_ce (the blas::gemm calls, timed elsewhere)
///   check        = the per-(b,m/n) presence+stride cleanliness verification
///   fallback     = the per-cell GEMV scalar path taken when a run is not clean
///   loop residual= kernel_total - gemm - check - fallback (cell iteration,
///                  offset math, result-pointer setup)
/// Separately, dispatch/scheduling OUTSIDE the kernel is the per-op eval-trace
/// "Product" time minus kernel_total (computed post-hoc from the trace).
inline std::atomic<std::uint64_t> g_kernel_ns_ce_ce{0};
inline std::atomic<std::uint64_t> g_check_ns_ce_ce{0};
inline std::atomic<std::uint64_t> g_fallback_ns_ce_ce{0};

/// Scoped timer accumulating wall ns of its lexical scope into `acc`. No-op
/// unless TA_GEMM_TIMING is set.
class ScopedPhaseTimer {
 public:
  explicit ScopedPhaseTimer(std::atomic<std::uint64_t>& acc)
      : acc_(gemm_timing_enabled() ? &acc : nullptr) {
    if (acc_) t0_ = std::chrono::steady_clock::now();
  }
  ~ScopedPhaseTimer() {
    if (!acc_) return;
    acc_->fetch_add(static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0_)
                            .count()),
                    std::memory_order_relaxed);
  }
  ScopedPhaseTimer(const ScopedPhaseTimer&) = delete;
  ScopedPhaseTimer& operator=(const ScopedPhaseTimer&) = delete;

 private:
  std::atomic<std::uint64_t>* acc_;
  std::chrono::steady_clock::time_point t0_;
};

/// Manual start/stop variant for a region that can't be lexically scoped (the
/// cleanliness check writes locals consumed after it).
inline std::chrono::steady_clock::time_point phase_start() {
  return gemm_timing_enabled() ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
}
inline void phase_stop(std::atomic<std::uint64_t>& acc,
                       std::chrono::steady_clock::time_point t0) {
  if (!gemm_timing_enabled()) return;
  acc.fetch_add(static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count()),
                std::memory_order_relaxed);
}

/// ---------------------------------------------------------------------------
/// Why did a ce+ce run fall back to scalar GEMV? Diagnose the rejected run by
/// re-walking it (gate order: presence -> uniform size -> constant stride).
///   1 = absent   : a cell in the run is missing (sparsity / screened out)
///   2 = nonuniform: present, but inner-cell sizes differ along the run
///   3 = stride    : present + uniform, but cells are NOT at a constant
///                   page-jump-free stride (the strided-DGEMM precondition)
///   0 = run looks clean (so the rejection came from the OTHER operand run)
inline std::atomic<std::uint64_t> g_fall_runs_ce_ce{0};
inline std::atomic<std::uint64_t> g_fall_res_absent_ce_ce{0};      // 1
inline std::atomic<std::uint64_t> g_fall_res_nonuniform_ce_ce{0};  // 2
inline std::atomic<std::uint64_t> g_fall_res_stride_ce_ce{0};      // 3
inline std::atomic<std::uint64_t> g_fall_op_absent_ce_ce{0};       // 13
inline std::atomic<std::uint64_t> g_fall_op_nonuniform_ce_ce{0};   // 14
inline std::atomic<std::uint64_t> g_fall_op_stride_ce_ce{0};       // 15
inline std::atomic<std::uint64_t> g_fall_op_acrossk_ce_ce{0};      // 16
inline std::atomic<std::uint64_t> g_fall_both_clean_ce_ce{0};      // 17

/// Classify a strided run: 1=absent, 2=nonuniform size, 3=bad stride, 0=clean.
template <typename GetCell>
inline int classify_run(GetCell getcell, std::size_t n) {
  if (n == 0) return 0;
  long s0 = -1;
  const double* base = nullptr;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& c = getcell(i);
    if (!c) return 1;  // absent
    const long sz = static_cast<long>(c.size());
    if (s0 < 0) {
      s0 = sz;
      base = c.data();
    } else if (sz != s0) {
      return 2;  // nonuniform
    }
  }
  if (n <= 1 || s0 <= 0) return 0;  // single cell: no stride to violate
  const long st = static_cast<long>(getcell(1).data() - base);
  if (st < s0) return 3;  // page-jump / overlap
  for (std::size_t i = 0; i < n; ++i)
    if (getcell(i).data() != base + static_cast<std::ptrdiff_t>(i) * st)
      return 3;  // non-constant stride
  return 0;
}

/// Diagnose the OPERAND side (called when the result run is clean). getR(k,i)
/// is the strided operand run (length `nrun`, per outer-contraction k); getL(k)
/// is the per-k single (non-strided) operand cell, expected size P*Q. Returns
/// 13=absent/size, 14=nonuniform, 15=bad stride within a k, 16=stride varies
/// across k, 17=clean (gate rejected a run this re-check finds valid).
template <typename GetR, typename GetL>
inline int classify_operand(GetR getR, GetL getL, std::size_t nrun,
                            std::size_t nK, long P) {
  const auto& r00 = getR(0, 0);
  if (!r00) return 13;
  const long Q = static_cast<long>(r00.size());
  if (Q <= 0) return 13;
  long sR = -1;
  for (std::size_t k = 0; k < nK; ++k) {
    const auto& lk = getL(k);
    if (!lk || static_cast<long>(lk.size()) != P * Q) return 13;  // single-cell
    long s0 = -1;
    const double* base = nullptr;
    for (std::size_t i = 0; i < nrun; ++i) {
      const auto& c = getR(k, i);
      if (!c) return 13;
      const long sz = static_cast<long>(c.size());
      if (s0 < 0) {
        s0 = sz;
        base = c.data();
      } else if (sz != s0) {
        return 14;
      }
    }
    if (s0 != Q) return 14;  // run size != Q (cross-k size mismatch)
    if (nrun > 1) {
      const long sk = static_cast<long>(getR(k, 1).data() - base);
      if (sk < Q) return 15;
      for (std::size_t i = 0; i < nrun; ++i)
        if (getR(k, i).data() != base + static_cast<std::ptrdiff_t>(i) * sk)
          return 15;
      if (k == 0)
        sR = sk;
      else if (sk != sR)
        return 16;  // stride varies across k
    }
  }
  return 17;  // both runs look clean to this re-check
}

// How many rejected runs would become a valid strided GEMM if we simply
// SKIPPED the absent outer-contraction (k) slabs (correct under beta=1)? A run
// is k-skip-rescuable when the result strided run is clean AND every k whose
// single-cell operand is present has a clean strided operand run -- i.e. the
// only defect is absent-k, not a hole in the strided (μ̃/m) dimension.
inline std::atomic<std::uint64_t> g_fall_kskip_ce_ce{0};
inline std::atomic<std::uint64_t> g_fall_kskip_present_ce_ce{0};  // present-k count

template <typename GetC, typename GetR, typename GetL>
inline bool kskip_rescuable(GetC getC, GetR getR, GetL getL, std::size_t nrun,
                            std::size_t nK, long P, std::size_t& present_k) {
  present_k = 0;
  if (classify_run(getC, nrun) != 0) return false;  // strided result hole
  long Q = -1;
  for (std::size_t k = 0; k < nK; ++k) {
    const auto& sc = getL(k);
    if (!sc) continue;  // absent single-cell operand -> skip this k (beta=1)
    const long scsz = static_cast<long>(sc.size());
    auto runk = [&](std::size_t i) -> decltype(getR(k, i)) {
      return getR(k, i);
    };
    if (classify_run(runk, nrun) != 0) return false;  // strided operand hole
    const auto& r0 = getR(k, 0);
    const long q = static_cast<long>(r0.size());
    if (q <= 0 || scsz != P * q) return false;  // size mismatch
    if (Q < 0) Q = q;
    else if (q != Q) return false;
    ++present_k;
  }
  return present_k > 0;
}

// Stronger test: can we recover the run by GATHERING the present strided
// indices (allowing μ̃-holes) into ONE strided GEMM? Requires: (a) the present
// result cells are at a uniform packed stride, and (b) for every present-k the
// operand slab has those SAME present indices, also uniform-stride, size Q.
// This is exactly the "result & operand empties are aligned" case -- the
// sparsity is shared (driven by the domain), so a single gather handles both.
inline std::atomic<std::uint64_t> g_fall_gather_ce_ce{0};
inline std::atomic<std::uint64_t> g_fall_gather_misalign_ce_ce{0};  // holes, but
                                                                    // unaligned

template <typename GetC, typename GetR, typename GetL>
inline int gather_rescuable(GetC getC, GetR getR, GetL getL, std::size_t nrun,
                            std::size_t nK, long P) {
  // returns 1 = gatherable (aligned), 0 = not (misaligned/irregular).
  if (nrun == 0 || nrun > 1024) return 0;
  std::size_t pres[1024];
  std::size_t np = 0;
  const double* rbase = nullptr;
  for (std::size_t i = 0; i < nrun; ++i) {
    const auto& c = getC(i);
    if (!c) continue;
    if (static_cast<long>(c.size()) != P) return 0;  // result inner nonuniform
    if (np == 0) rbase = c.data();
    pres[np++] = i;
  }
  if (np == 0) return 0;
  long rstride = -1;
  if (np > 1) {
    rstride = static_cast<long>(getC(pres[1]).data() - rbase);
    if (rstride < P) return 0;
    for (std::size_t j = 0; j < np; ++j)
      if (getC(pres[j]).data() != rbase + static_cast<std::ptrdiff_t>(j) * rstride)
        return 0;  // present result cells not at uniform packed stride
  }
  long Q = -1;
  bool any_k = false;
  for (std::size_t k = 0; k < nK; ++k) {
    if (!getL(k)) continue;  // absent single-cell -> skip k (β=1)
    const double* ob = nullptr;
    long os = -1;
    for (std::size_t j = 0; j < np; ++j) {
      const auto& oc = getR(k, pres[j]);
      if (!oc) return 0;  // operand hole at a result-present index -> misaligned
      const long q = static_cast<long>(oc.size());
      if (Q < 0) Q = q;
      else if (q != Q) return 0;
      if (j == 0) ob = oc.data();
      else if (j == 1) {
        os = static_cast<long>(oc.data() - ob);
        if (os < Q) return 0;
      }
      if (os >= 0 &&
          oc.data() != ob + static_cast<std::ptrdiff_t>(j) * os)
        return 0;
    }
    if (Q <= 0 || static_cast<long>(getL(k).size()) != P * Q) return 0;
    any_k = true;
  }
  return any_k ? 1 : 0;
}

// Simulate the per-k segmented strided GEMM: for each present k, walk the
// strided axis and count maximal contiguous (present + uniform-stride) segments.
// Reports how many segment-GEMMs the scheme issues and their length (=BLAS M)
// distribution -- the make-or-break metric (M>1 GEMM vs M=1 GEMV).
inline std::atomic<std::uint64_t> g_seg_calls_ce_ce{0};   // # segment GEMMs
inline std::atomic<std::uint64_t> g_seg_cells_ce_ce{0};   // Σ segment lengths
inline std::atomic<std::uint64_t> g_seg_len1_ce_ce{0};    // length-1 (GEMV)
inline std::atomic<std::uint64_t> g_seg_len2_ce_ce{0};
inline std::atomic<std::uint64_t> g_seg_len3_4_ce_ce{0};
inline std::atomic<std::uint64_t> g_seg_len5_8_ce_ce{0};
inline std::atomic<std::uint64_t> g_seg_len9p_ce_ce{0};

template <typename GetC, typename GetR, typename GetL>
inline void measure_segments(GetC getC, GetR getR, GetL getL, std::size_t nrun,
                             std::size_t nK, long P) {
  for (std::size_t k = 0; k < nK; ++k) {
    const auto& sc = getL(k);
    if (!sc) continue;  // skip absent-k
    std::size_t mu = 0;
    while (mu < nrun) {
      const auto& c0 = getC(mu);
      const auto& r0 = getR(k, mu);
      if (!c0 || !r0 || static_cast<long>(c0.size()) != P) {
        ++mu;
        continue;
      }
      const long Q = static_cast<long>(r0.size());
      if (Q <= 0 || static_cast<long>(sc.size()) != P * Q) {
        ++mu;
        continue;
      }
      const double* cb = c0.data();
      const double* rb = r0.data();
      std::size_t end = mu + 1;
      long sC = -1, sR = -1;
      while (end < nrun) {
        const auto& ce = getC(end);
        const auto& re = getR(k, end);
        if (!ce || !re) break;
        if (static_cast<long>(ce.size()) != P ||
            static_cast<long>(re.size()) != Q)
          break;
        const long dc = static_cast<long>(ce.data() - cb);
        const long dr = static_cast<long>(re.data() - rb);
        const long off = static_cast<long>(end - mu);
        if (off == 1) {
          sC = dc;
          sR = dr;
          if (sC < P || sR < Q) break;
        } else if (dc != off * sC || dr != off * sR) {
          break;
        }
        ++end;
      }
      const std::size_t len = end - mu;
      g_seg_calls_ce_ce.fetch_add(1, std::memory_order_relaxed);
      g_seg_cells_ce_ce.fetch_add(len, std::memory_order_relaxed);
      if (len == 1) g_seg_len1_ce_ce.fetch_add(1, std::memory_order_relaxed);
      else if (len == 2) g_seg_len2_ce_ce.fetch_add(1, std::memory_order_relaxed);
      else if (len <= 4) g_seg_len3_4_ce_ce.fetch_add(1, std::memory_order_relaxed);
      else if (len <= 8) g_seg_len5_8_ce_ce.fetch_add(1, std::memory_order_relaxed);
      else g_seg_len9p_ce_ce.fetch_add(1, std::memory_order_relaxed);
      mu = end;
    }
  }
}

inline void record_ce_ce_fallback(int why) {
  if (!gemm_timing_enabled()) return;
  g_fall_runs_ce_ce.fetch_add(1, std::memory_order_relaxed);
  switch (why) {
    case 1: g_fall_res_absent_ce_ce.fetch_add(1, std::memory_order_relaxed); break;
    case 2: g_fall_res_nonuniform_ce_ce.fetch_add(1, std::memory_order_relaxed); break;
    case 3: g_fall_res_stride_ce_ce.fetch_add(1, std::memory_order_relaxed); break;
    case 13: g_fall_op_absent_ce_ce.fetch_add(1, std::memory_order_relaxed); break;
    case 14: g_fall_op_nonuniform_ce_ce.fetch_add(1, std::memory_order_relaxed); break;
    case 15: g_fall_op_stride_ce_ce.fetch_add(1, std::memory_order_relaxed); break;
    case 16: g_fall_op_acrossk_ce_ce.fetch_add(1, std::memory_order_relaxed); break;
    default: g_fall_both_clean_ce_ce.fetch_add(1, std::memory_order_relaxed); break;
  }
}

// ---------------------------------------------------------------------------
// ce+e phase timers + fallback diagnosis (mirror of the ce+ce instrumentation).
// In ce+e a "run" is one result cell (m,n); the clean check is over the k-slabs
// of L (stride ldA) and R (stride ldB). Fallback reasons classify those k-runs:
//   L-run: 1 absent / 2 nonuniform / 3 stride;  R-run: 11 / 12 / 13;  17 clean.
inline std::atomic<std::uint64_t> g_kernel_ns_ce_e{0};
inline std::atomic<std::uint64_t> g_check_ns_ce_e{0};
inline std::atomic<std::uint64_t> g_fallback_ns_ce_e{0};
inline std::atomic<std::uint64_t> g_e_fall_runs{0};
inline std::atomic<std::uint64_t> g_e_l_absent{0};
inline std::atomic<std::uint64_t> g_e_l_nonuniform{0};
inline std::atomic<std::uint64_t> g_e_l_stride{0};
inline std::atomic<std::uint64_t> g_e_r_absent{0};
inline std::atomic<std::uint64_t> g_e_r_nonuniform{0};
inline std::atomic<std::uint64_t> g_e_r_stride{0};
inline std::atomic<std::uint64_t> g_e_both_clean{0};

inline void record_ce_e_fallback(int why) {
  if (!gemm_timing_enabled()) return;
  g_e_fall_runs.fetch_add(1, std::memory_order_relaxed);
  switch (why) {
    case 1: g_e_l_absent.fetch_add(1, std::memory_order_relaxed); break;
    case 2: g_e_l_nonuniform.fetch_add(1, std::memory_order_relaxed); break;
    case 3: g_e_l_stride.fetch_add(1, std::memory_order_relaxed); break;
    case 11: g_e_r_absent.fetch_add(1, std::memory_order_relaxed); break;
    case 12: g_e_r_nonuniform.fetch_add(1, std::memory_order_relaxed); break;
    case 13: g_e_r_stride.fetch_add(1, std::memory_order_relaxed); break;
    default: g_e_both_clean.fetch_add(1, std::memory_order_relaxed); break;
  }
}

// ---------------------------------------------------------------------------
// Coverage: clean (strided-GEMM) vs fallback (scalar) work. Clean FLOPs and
// time come from the shape histogram + g_gemm_ns_*; here we also count clean
// runs (ce+ce, where one run issues nK gemms) and accumulate the scalar
// fallback FLOPs so we can report exactly what fraction of each regime's
// arithmetic the strided path captures today.
inline std::atomic<std::uint64_t> g_clean_runs_ce_ce{0};
inline std::atomic<std::uint64_t> g_fall_flops_ce_e{0};
inline std::atomic<std::uint64_t> g_fall_flops_ce_ce{0};

/// Dumps the per-regime GEMM-time totals at process exit when TA_GEMM_TIMING
/// is set. The single inline instance is constructed after <iostream>'s static
/// init, hence destroyed before std::cerr.
struct GemmTimingDumper {
  ~GemmTimingDumper() {
    if (!gemm_timing_enabled()) return;
    const auto ce_e = g_gemm_ns_ce_e.load(std::memory_order_relaxed);
    const auto ce_ce = g_gemm_ns_ce_ce.load(std::memory_order_relaxed);
    std::cerr << "[gemm-timing] ce+e  GEMM: " << (ce_e / 1e9) << " s  ("
              << g_gemm_calls_ce_e.load(std::memory_order_relaxed)
              << " gemm calls)\n";
    std::cerr << "[gemm-timing] ce+ce GEMM: " << (ce_ce / 1e9) << " s  ("
              << g_gemm_calls_ce_ce.load(std::memory_order_relaxed)
              << " gemm calls)\n";
    std::cerr << "[gemm-timing] total GEMM: " << ((ce_e + ce_ce) / 1e9)
              << " s\n";
    auto L = [](std::atomic<std::uint64_t>& a) {
      return a.load(std::memory_order_relaxed);
    };

    // ---- kernel-internal phase decomposition (per regime) ----
    auto dump_phases = [](const char* tag, std::uint64_t kn, std::uint64_t gm,
                          std::uint64_t ck, std::uint64_t fbt) {
      if (kn == 0) return;
      const auto resid =
          kn > (gm + ck + fbt) ? kn - (gm + ck + fbt) : std::uint64_t{0};
      auto pct = [&](std::uint64_t x) { return 100.0 * x / kn; };
      std::cerr << "[" << tag << "-phases] kernel total   : " << (kn / 1e9)
                << " s\n";
      std::cerr << "[" << tag << "-phases]   gemm         : " << (gm / 1e9)
                << " s  (" << pct(gm) << "%)\n";
      std::cerr << "[" << tag << "-phases]   clean-check  : " << (ck / 1e9)
                << " s  (" << pct(ck) << "%)\n";
      std::cerr << "[" << tag << "-phases]   fallback     : " << (fbt / 1e9)
                << " s  (" << pct(fbt) << "%)\n";
      std::cerr << "[" << tag << "-phases]   loop residual: " << (resid / 1e9)
                << " s  (" << pct(resid) << "%)\n";
    };
    dump_phases("ce+e", L(g_kernel_ns_ce_e), ce_e, L(g_check_ns_ce_e),
                L(g_fallback_ns_ce_e));
    dump_phases("ce+ce", L(g_kernel_ns_ce_ce), ce_ce, L(g_check_ns_ce_ce),
                L(g_fallback_ns_ce_ce));

    // ---- shape histograms (return clean strided-GEMM FLOPs per regime) ----
    const double clean_flops_e = dump_shapes("ce+e", g_ce_e_shapes);
    const double clean_flops_ce = dump_shapes("ce+ce", g_ce_ce_shapes);

    // ---- coverage: what fraction of each regime's arithmetic & time the
    //      strided GEMM captures today, vs the scalar fallback ----
    auto cov = [](const char* tag, double clean_flops, std::uint64_t clean_ns,
                  std::uint64_t clean_runs, std::uint64_t fall_flops_u,
                  std::uint64_t fall_ns, std::uint64_t fall_runs) {
      const double ff = static_cast<double>(fall_flops_u);
      auto p = [](double a, double b) { return b > 0 ? 100.0 * a / b : 0.0; };
      std::cerr << "[coverage " << tag << "] clean GEMM: " << std::fixed
                << std::setprecision(2) << (clean_flops / 1e9) << " GFLOP, "
                << (clean_ns / 1e9) << " s, " << clean_runs << " runs\n";
      std::cerr << "[coverage " << tag << "] fallback  : " << (ff / 1e9)
                << " GFLOP, " << (fall_ns / 1e9) << " s, " << fall_runs
                << " runs\n";
      std::cerr << "[coverage " << tag
                << "] FLOP coverage = " << p(clean_flops, clean_flops + ff)
                << "%   time coverage = " << p(clean_ns, clean_ns + fall_ns)
                << "%\n"
                << std::defaultfloat;
    };
    cov("ce+e", clean_flops_e, ce_e, L(g_gemm_calls_ce_e), L(g_fall_flops_ce_e),
        L(g_fallback_ns_ce_e), L(g_e_fall_runs));
    cov("ce+ce", clean_flops_ce, ce_ce, L(g_clean_runs_ce_ce),
        L(g_fall_flops_ce_ce), L(g_fallback_ns_ce_ce), L(g_fall_runs_ce_ce));

    // ---- ce+e fallback reasons (which k-run failed the strided gate) ----
    const auto efr = L(g_e_fall_runs);
    if (efr > 0) {
      auto fp = [&](std::uint64_t x) { return 100.0 * x / efr; };
      std::cerr << "[ce+e-fallback] total rejected cells: " << efr << "\n";
      std::cerr << "[ce+e-fallback]   L-run absent     : " << L(g_e_l_absent)
                << "  (" << fp(L(g_e_l_absent)) << "%)\n";
      std::cerr << "[ce+e-fallback]   L-run nonuniform : " << L(g_e_l_nonuniform)
                << "  (" << fp(L(g_e_l_nonuniform)) << "%)\n";
      std::cerr << "[ce+e-fallback]   L-run bad stride : " << L(g_e_l_stride)
                << "  (" << fp(L(g_e_l_stride)) << "%)\n";
      std::cerr << "[ce+e-fallback]   R-run absent     : " << L(g_e_r_absent)
                << "  (" << fp(L(g_e_r_absent)) << "%)\n";
      std::cerr << "[ce+e-fallback]   R-run nonuniform : " << L(g_e_r_nonuniform)
                << "  (" << fp(L(g_e_r_nonuniform)) << "%)\n";
      std::cerr << "[ce+e-fallback]   R-run bad stride : " << L(g_e_r_stride)
                << "  (" << fp(L(g_e_r_stride)) << "%)\n";
      std::cerr << "[ce+e-fallback]   both runs clean  : " << L(g_e_both_clean)
                << "  (" << fp(L(g_e_both_clean)) << "%)\n";
    }

    // ---- ce+ce fallback reasons (result-run then operand-run diagnosis) ----
    const auto fruns = L(g_fall_runs_ce_ce);
    if (fruns > 0) {
      auto fp = [&](std::uint64_t x) { return 100.0 * x / fruns; };
      std::cerr << "[ce+ce-fallback] total rejected runs: " << fruns << "\n";
      std::cerr << "[ce+ce-fallback]   result absent (sparse): "
                << L(g_fall_res_absent_ce_ce) << "  ("
                << fp(L(g_fall_res_absent_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-fallback]   result nonuniform     : "
                << L(g_fall_res_nonuniform_ce_ce) << "  ("
                << fp(L(g_fall_res_nonuniform_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-fallback]   result bad stride     : "
                << L(g_fall_res_stride_ce_ce) << "  ("
                << fp(L(g_fall_res_stride_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-fallback]   operand absent/size   : "
                << L(g_fall_op_absent_ce_ce) << "  ("
                << fp(L(g_fall_op_absent_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-fallback]   operand nonuniform    : "
                << L(g_fall_op_nonuniform_ce_ce) << "  ("
                << fp(L(g_fall_op_nonuniform_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-fallback]   operand bad stride    : "
                << L(g_fall_op_stride_ce_ce) << "  ("
                << fp(L(g_fall_op_stride_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-fallback]   operand stride X k    : "
                << L(g_fall_op_acrossk_ce_ce) << "  ("
                << fp(L(g_fall_op_acrossk_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-fallback]   both runs clean (!)   : "
                << L(g_fall_both_clean_ce_ce) << "  ("
                << fp(L(g_fall_both_clean_ce_ce)) << "%)\n";
      const auto ks = L(g_fall_kskip_ce_ce);
      const auto kp = L(g_fall_kskip_present_ce_ce);
      const auto gr = L(g_fall_gather_ce_ce);
      std::cerr << "[ce+ce-fallback]   >> rescuable by k-skip (no μ̃ holes): "
                << ks << "  (" << fp(ks) << "%); "
                << (ks ? static_cast<double>(kp) / ks : 0.0)
                << " present-k/run avg\n";
      std::cerr << "[ce+ce-fallback]   >> rescuable by μ̃-gather (aligned holes): "
                << gr << "  (" << fp(gr) << "% of rejected runs) -- present "
                   "result & operand μ̃ aligned at uniform stride => ONE "
                   "strided GEMM over gathered cells\n";
    }
    // Per-k segmented strided GEMM (the contiguous-sub-run scheme): how many
    // segment-GEMMs would it issue on the fallback runs, and how long (BLAS M)?
    const auto sc = L(g_seg_calls_ce_ce);
    if (sc > 0) {
      const auto sl = L(g_seg_cells_ce_ce);
      auto sp = [&](std::uint64_t x) { return 100.0 * x / sc; };
      std::cerr << "[ce+ce-segment] segment-GEMMs the scheme would issue: " << sc
                << "  (covering " << sl << " present cells, mean M="
                << (sc ? static_cast<double>(sl) / sc : 0.0) << ")\n";
      std::cerr << "[ce+ce-segment]   length distribution (M=segment len):\n";
      std::cerr << "[ce+ce-segment]     M=1 (GEMV) : " << L(g_seg_len1_ce_ce)
                << "  (" << sp(L(g_seg_len1_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-segment]     M=2        : " << L(g_seg_len2_ce_ce)
                << "  (" << sp(L(g_seg_len2_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-segment]     M=3-4      : " << L(g_seg_len3_4_ce_ce)
                << "  (" << sp(L(g_seg_len3_4_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-segment]     M=5-8      : " << L(g_seg_len5_8_ce_ce)
                << "  (" << sp(L(g_seg_len5_8_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-segment]     M>=9       : " << L(g_seg_len9p_ce_ce)
                << "  (" << sp(L(g_seg_len9p_ce_ce)) << "%)\n";
      std::cerr << "[ce+ce-segment]   (today's clean path issues "
                << L(g_gemm_calls_ce_ce)
                << " full-run GEMMs; this would ADD the above on fallback runs)\n";
    }
  }

  // Per-call-weighted distribution of a single GEMM dimension (M, N, or K),
  // sorted by total time (descending) so the dominant values come first.
  static void dump_dim_dist(
      const char* tag, const char* dim,
      const std::map<std::size_t, std::pair<std::uint64_t, std::uint64_t>>& by,
      std::uint64_t tot_ns) {
    struct E {
      std::size_t val;
      std::uint64_t calls, ns;
    };
    std::vector<E> v;
    v.reserve(by.size());
    for (const auto& kv : by)
      v.push_back(E{kv.first, kv.second.first, kv.second.second});
    std::sort(v.begin(), v.end(),
              [](const E& a, const E& b) { return a.ns > b.ns; });
    std::cerr << "[gemm-shapes " << tag << "] " << dim
              << " distribution (by %time desc; val: calls, ns_total, %time):\n";
    for (const auto& e : v) {
      const double pct = tot_ns > 0 ? 100.0 * e.ns / tot_ns : 0.0;
      std::cerr << "[gemm-shapes " << tag << "]   " << dim << "=" << std::setw(4)
                << e.val << "  " << std::setw(10) << e.calls << " calls  "
                << std::setw(13) << e.ns << "  " << std::fixed
                << std::setprecision(1) << std::setw(5) << pct << "%\n"
                << std::defaultfloat;
    }
  }

  static double dump_shapes(const char* tag, ShapeRegistry& reg) {
    // Merge all per-thread maps.
    std::unordered_map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>>
        merged;
    {
      std::lock_guard<std::mutex> lk(reg.mtx);
      for (auto* a : reg.maps)
        for (const auto& kv : a->m) {
          auto& e = merged[kv.first];
          e.first += kv.second.first;
          e.second += kv.second.second;
        }
    }
    if (merged.empty()) return 0.0;
    constexpr std::uint64_t MASK = (std::uint64_t{1} << 21) - 1;
    struct Row {
      std::size_t M, N, K;
      std::uint64_t calls, ns;
    };
    std::vector<Row> rows;
    rows.reserve(merged.size());
    for (const auto& kv : merged) {
      const std::uint64_t k = kv.first;
      rows.push_back(Row{static_cast<std::size_t>((k >> 42) & MASK),
                         static_cast<std::size_t>((k >> 21) & MASK),
                         static_cast<std::size_t>(k & MASK), kv.second.first,
                         kv.second.second});
    }
    // Sort by total ns descending (dominant shapes first).
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.ns > b.ns; });
    std::cerr << "[gemm-shapes " << tag << "] distinct (M,N,K) shapes: "
              << rows.size() << "\n";
    std::cerr << "[gemm-shapes " << tag << "] "
              << "M       N       K     calls       ns_total      "
                 "ns/call   GFLOP/s   %time\n";
    std::uint64_t tot_ns = 0;
    for (const auto& r : rows) tot_ns += r.ns;
    std::size_t shown = 0;
    for (const auto& r : rows) {
      const double flops = 2.0 * r.M * r.N * r.K * static_cast<double>(r.calls);
      const double gflops = r.ns > 0 ? flops / r.ns : 0.0;  // 2MNK*calls / ns
      const double pct = tot_ns > 0 ? 100.0 * r.ns / tot_ns : 0.0;
      if (shown < 40)
        std::cerr << "[gemm-shapes " << tag << "] " << std::setw(6) << r.M
                  << "  " << std::setw(6) << r.N << "  " << std::setw(5) << r.K
                  << "  " << std::setw(9) << r.calls << "  " << std::setw(13)
                  << r.ns << "  " << std::setw(8) << (r.ns / r.calls) << "  "
                  << std::setw(7) << std::fixed << std::setprecision(2) << gflops
                  << "  " << std::setw(5) << std::setprecision(1) << pct
                  << "%\n"
                  << std::defaultfloat;
      else if (shown == 40)
        std::cerr << "[gemm-shapes " << tag << "] ... (" << (rows.size() - 40)
                  << " more shapes omitted)\n";
      ++shown;
    }

    // ---- aggregate summaries over ALL shapes ----
    std::uint64_t tot_calls = 0;
    double tot_flops = 0.0;
    double sum_M = 0.0, sum_N = 0.0, sum_K = 0.0;  // call-weighted
    std::map<std::size_t, std::pair<std::uint64_t, std::uint64_t>> byK, byM, byN;
    for (const auto& r : rows) {
      tot_calls += r.calls;
      tot_flops += 2.0 * r.M * r.N * r.K * static_cast<double>(r.calls);
      sum_M += static_cast<double>(r.M) * r.calls;
      sum_N += static_cast<double>(r.N) * r.calls;
      sum_K += static_cast<double>(r.K) * r.calls;
      byK[r.K].first += r.calls;
      byK[r.K].second += r.ns;
      byM[r.M].first += r.calls;
      byM[r.M].second += r.ns;
      byN[r.N].first += r.calls;
      byN[r.N].second += r.ns;
    }
    const double eff_gflops = tot_ns > 0 ? tot_flops / tot_ns : 0.0;
    std::cerr << "[gemm-shapes " << tag << "] ---- aggregate over all "
              << rows.size() << " shapes ----\n";
    std::cerr << "[gemm-shapes " << tag << "] total calls=" << tot_calls
              << "  total GFLOP=" << std::fixed << std::setprecision(2)
              << (tot_flops / 1e9) << "  total ns=" << tot_ns
              << "  effective GFLOP/s=" << eff_gflops << std::defaultfloat
              << "\n";
    std::cerr << "[gemm-shapes " << tag << "] call-weighted mean: M="
              << (tot_calls ? sum_M / tot_calls : 0.0)
              << "  N=" << (tot_calls ? sum_N / tot_calls : 0.0)
              << "  K=" << (tot_calls ? sum_K / tot_calls : 0.0) << "\n";
    dump_dim_dist(tag, "K", byK, tot_ns);
    dump_dim_dist(tag, "M", byM, tot_ns);
    dump_dim_dist(tag, "N", byN, tot_ns);
    return tot_flops;
  }
};
inline GemmTimingDumper g_gemm_timing_dumper;

/// Specifies how an inner-cell range is derived from operand inner cells.
enum class ArenaInnerShapeKind {
  left_range,         // Hadamard inner; Scale tot_x_t
  right_range,        // Scale t_x_tot
  gemm_result_range,  // inner Contraction (uses inner_gh)
  unit_range  // phantom-unit denest: a unit-extent [1]^phantom_rank cell
              // independent of operand inner ranges (the inner product is a
              // flat dot; see RegimeAInnerKind::phantom_dot)
};

/// Inner-shape derivation plan: kind + (optional) inner GemmHelper.
struct ArenaInnerShapePlan {
  ArenaInnerShapeKind kind;
  std::optional<math::GemmHelper> inner_gh;  // only for gemm_result_range
  std::size_t phantom_rank = 0;              // only for unit_range

  /// Derives one result inner range from operand inner cells.
  template <typename ResultInnerRange, typename LInner, typename RInner>
  ResultInnerRange make(const LInner& l, const RInner& r) const {
    switch (kind) {
      case ArenaInnerShapeKind::left_range:
        return l.range();
      case ArenaInnerShapeKind::right_range:
        return r.range();
      case ArenaInnerShapeKind::gemm_result_range:
        TA_ASSERT(inner_gh.has_value());
        return inner_gh->template make_result_range<ResultInnerRange>(
            l.range(), r.range());
      case ArenaInnerShapeKind::unit_range: {
        container::vector<std::size_t> ext(phantom_rank, std::size_t{1});
        return ResultInnerRange(ext);
      }
    }
    TA_ASSERT(false);
    return ResultInnerRange{};
  }
};

/// Derives result ranges and constructs non-empty inner cells in one arena
/// slab.
template <typename Result, typename Left, typename Right>
class ContractionArenaPlan {
 public:
  /// Stores the inner shape plan used to construct result cells.
  explicit ContractionArenaPlan(ArenaInnerShapePlan p)
      : inner_plan_(std::move(p)) {}

  /// Constructs a result tile whose non-empty inner cells alias arena storage.
  Result reserve_and_construct(const Left& left, const Right& right,
                               const math::GemmHelper& outer_gh) const;

  /// Grows an already-constructed result tile in place so it covers every
  /// inner cell implied by this `left`/`right` K-panel. A SUMMA reduction
  /// shapes the result from its first K-panel only; a later panel of a
  /// contracted-dimension-sparse ToT operand can touch inner cells the first
  /// panel left null, so each subsequent panel must extend the result.
  void grow_to_cover(Result& result, const Left& left, const Right& right,
                     const math::GemmHelper& outer_gh) const;

 private:
  /// Per-output-cell inner ranges implied by one `left`/`right` K-panel.
  /// Deduced return type: spelling `Result::value_type::range_type` in the
  /// declaration would make the whole class ill-formed for a non-ToT
  /// `Result`, but `make_contraction_arena_plan` names this class in its
  /// return type unconditionally (and returns nullopt for non-ToT).
  auto operand_inner_ranges(const Left& left, const Right& right,
                            const math::GemmHelper& outer_gh) const;

  ArenaInnerShapePlan inner_plan_{};
};

/// True when `T` is a `TA::Tensor` outer whose inner cells the arena
/// machinery knows how to allocate (legacy `TA::Tensor` ToT inner or the
/// pinned-view `ArenaTensor`). Doesn't require `is_tensor_of_tensor_v` --
/// `ArenaTensor` is deliberately not registered as `is_tensor_helper`, so
/// trait propagation can't reach it that way.
template <typename T>
inline constexpr bool is_arena_eligible_outer_v =
    is_ta_tensor_v<T> &&
    (is_ta_tensor_v<typename T::value_type> ||
     ::TiledArray::is_arena_tensor_v<typename T::value_type>);

/// True when `T` is an inner-cell type that the arena machinery treats as
/// tensor-shaped (as opposed to a scalar in mixed Scale ops). Covers the
/// legacy `TA::Tensor` inner and the pinned `ArenaTensor`. Used by the
/// regime-A `accumulate` dispatch to distinguish the tensor-inner branches
/// from the scalar-inner ones in `scale_left`/`scale_right` cases.
template <typename T>
inline constexpr bool is_arena_inner_cell_v =
    is_ta_tensor_v<T> || ::TiledArray::is_arena_tensor_v<T>;

/// True when the result is an arena-eligible outer; gates the arena
/// allocation path in cont_engine.
template <typename Result, typename Left, typename Right>
inline constexpr bool is_contraction_arena_tot_v =
    is_arena_eligible_outer_v<Result>;

/// Stores an arena plan for ToT results and std::monostate otherwise.
template <typename Result, typename Left, typename Right>
using arena_plan_storage_t =
    std::conditional_t<is_contraction_arena_tot_v<Result, Left, Right>,
                       std::optional<ContractionArenaPlan<Result, Left, Right>>,
                       std::monostate>;

/// Builds a contraction arena plan when the result and inner permutation allow
/// it.
template <typename Result, typename Left, typename Right>
auto make_contraction_arena_plan(ArenaInnerShapeKind inner_kind,
                                 std::optional<math::GemmHelper> inner_gh,
                                 const Permutation& inner_perm,
                                 std::size_t phantom_rank = 0)
    -> std::optional<ContractionArenaPlan<Result, Left, Right>> {
  if (arena_disabled()) return std::nullopt;
  if constexpr (!is_contraction_arena_tot_v<Result, Left, Right>) {
    return std::nullopt;
  } else {
    if (bool(inner_perm) && !inner_perm.is_identity()) return std::nullopt;
    if (inner_kind != ArenaInnerShapeKind::gemm_result_range)
      inner_gh.reset();
    else if (!inner_gh.has_value())
      return std::nullopt;
    return std::optional<ContractionArenaPlan<Result, Left, Right>>(
        std::in_place,
        ArenaInnerShapePlan{inner_kind, std::move(inner_gh), phantom_rank});
  }
}

/// Per-output-cell inner ranges implied by one `left`/`right` K-panel.
template <typename Result, typename Left, typename Right>
auto ContractionArenaPlan<Result, Left, Right>::operand_inner_ranges(
    const Left& left, const Right& right,
    const math::GemmHelper& outer_gh) const {
  using inner_t = typename Result::value_type;
  using inner_range_t = typename inner_t::range_type;
  using integer = math::blas::integer;

  integer M, N, K;
  outer_gh.compute_matrix_sizes(M, N, K, left.range(), right.range());
  const integer lda = (outer_gh.left_op() == math::blas::NoTranspose) ? K : M;
  const integer ldb = (outer_gh.right_op() == math::blas::NoTranspose) ? N : K;
  TA_ASSERT(left.nbatch() == right.nbatch());
  const std::size_t batch_sz = static_cast<std::size_t>(left.nbatch());
  const std::size_t mn =
      static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

  auto range_for = [&](std::size_t ord) -> inner_range_t {
    if (mn == 0) return inner_range_t{};
    const integer b = static_cast<integer>(ord / mn);
    const integer rem = static_cast<integer>(ord % mn);
    const integer m = rem / N;
    const integer n = rem % N;

    if (inner_plan_.kind == ArenaInnerShapeKind::left_range) {
      if constexpr (is_arena_eligible_outer_v<Left>) {
        const auto* lbase = left.batch_data(static_cast<std::size_t>(b));
        for (integer k = 0; k != K; ++k) {
          const auto aoff = (outer_gh.left_op() == math::blas::NoTranspose)
                                ? m * lda + k
                                : k * lda + m;
          const auto& lc = *(lbase + aoff);
          if (!lc.empty()) return lc.range();
        }
      }
      return inner_range_t{};
    }
    if (inner_plan_.kind == ArenaInnerShapeKind::right_range) {
      if constexpr (is_arena_eligible_outer_v<Right>) {
        const auto* rbase = right.batch_data(static_cast<std::size_t>(b));
        for (integer k = 0; k != K; ++k) {
          const auto boff = (outer_gh.right_op() == math::blas::NoTranspose)
                                ? k * ldb + n
                                : n * ldb + k;
          const auto& rc = *(rbase + boff);
          if (!rc.empty()) return rc.range();
        }
      }
      return inner_range_t{};
    }
    // gemm_result_range needs both operands to be ToT.
    if constexpr (is_arena_eligible_outer_v<Left> &&
                  is_arena_eligible_outer_v<Right>) {
      const auto* lbase = left.batch_data(static_cast<std::size_t>(b));
      const auto* rbase = right.batch_data(static_cast<std::size_t>(b));
      for (integer k = 0; k != K; ++k) {
        const auto aoff = (outer_gh.left_op() == math::blas::NoTranspose)
                              ? m * lda + k
                              : k * lda + m;
        const auto boff = (outer_gh.right_op() == math::blas::NoTranspose)
                              ? k * ldb + n
                              : n * ldb + k;
        const auto& lc = *(lbase + aoff);
        const auto& rc = *(rbase + boff);
        if (lc.empty() || rc.empty()) continue;
        return inner_plan_.template make<inner_range_t>(lc, rc);
      }
    }
    return inner_range_t{};
  };

  std::vector<inner_range_t> ranges;
  const std::size_t N_cells = mn * batch_sz;
  ranges.reserve(N_cells);
  for (std::size_t ord = 0; ord < N_cells; ++ord)
    ranges.emplace_back(range_for(ord));
  return ranges;
}

/// Reserves arena storage and constructs the result tensor-of-tensor tile.
template <typename Result, typename Left, typename Right>
Result ContractionArenaPlan<Result, Left, Right>::reserve_and_construct(
    const Left& left, const Right& right,
    const math::GemmHelper& outer_gh) const {
  using inner_range_t = typename Result::value_type::range_type;
  auto outer_range =
      outer_gh.template make_result_range<typename Result::range_type>(
          left.range(), right.range());
  TA_ASSERT(left.nbatch() == right.nbatch());
  const std::size_t batch_sz = static_cast<std::size_t>(left.nbatch());
  const auto ranges = operand_inner_ranges(left, right, outer_gh);
  // arena_outer_init dispatches internally on the inner-cell type.
  return detail::arena_outer_init<Result>(
      outer_range, batch_sz,
      [&ranges](std::size_t ord) -> inner_range_t { return ranges[ord]; });
}

/// Grows an already-constructed result tile to cover this K-panel's cells.
template <typename Result, typename Left, typename Right>
void ContractionArenaPlan<Result, Left, Right>::grow_to_cover(
    Result& result, const Left& left, const Right& right,
    const math::GemmHelper& outer_gh) const {
  using inner_range_t = typename Result::value_type::range_type;
  const auto ranges = operand_inner_ranges(left, right, outer_gh);
  detail::arena_tot_grow_inplace(
      result,
      [&ranges](std::size_t ord) -> inner_range_t { return ranges[ord]; });
}

/// Accumulates a contraction into an already-allocated result cell.
template <typename Result, typename Left, typename Right, typename Scalar>
void fused_contraction_inplace(Result& result, const Left& left,
                               const Right& right, Scalar alpha,
                               const math::GemmHelper& gh) {
  if (left.empty() || right.empty()) return;
  TA_ASSERT(!result.empty());
  // Free `gemm` CPO, not the member: `ArenaTensor` (a view) provides only the
  // free in-place overload, while `TA::Tensor` is reached via the
  // `tile_interface.h` CPO that forwards to its member.
  gemm(result, left, right, alpha, gh);
}

/// Accumulates an elementwise product into an already-allocated result cell.
template <typename Result, typename Left, typename Right>
void fused_hadamard_inplace(Result& result, const Left& left,
                            const Right& right) {
  if (left.empty() || right.empty()) return;
  TA_ASSERT(!result.empty());
  inplace_tensor_op(
      [](typename Result::value_type& MADNESS_RESTRICT r,
         const typename Left::value_type& MADNESS_RESTRICT l,
         const typename Right::value_type& MADNESS_RESTRICT rr) {
        r += l * rr;
      },
      result, left, right);
}

/// Accumulates a scaled elementwise product into an allocated result cell.
template <typename Result, typename Left, typename Right, typename Scalar>
void fused_hadamard_scaled_inplace(Result& result, const Left& left,
                                   const Right& right, Scalar factor) {
  if (left.empty() || right.empty()) return;
  TA_ASSERT(!result.empty());
  // Preserve historical grouping: r += (l * rr) * factor.
  inplace_tensor_op(
      [factor](typename Result::value_type& MADNESS_RESTRICT r,
               const typename Left::value_type& MADNESS_RESTRICT l,
               const typename Right::value_type& MADNESS_RESTRICT rr) {
        r += (l * rr) * factor;
      },
      result, left, right);
}

/// Accumulates a ToT cell scaled by a scalar right operand.
template <typename Result, typename Left, typename Scalar>
void fused_scale_tot_x_t_inplace(Result& result, const Left& left,
                                 const Scalar& s) {
  if (left.empty()) return;
  TA_ASSERT(!result.empty());
  inplace_tensor_op(
      [s](typename Result::value_type& MADNESS_RESTRICT r,
          const typename Left::value_type& MADNESS_RESTRICT l) { r += l * s; },
      result, left);
}

/// Accumulates a ToT right operand scaled by a scalar left operand.
template <typename Result, typename Scalar, typename Right>
void fused_scale_t_x_tot_inplace(Result& result, const Scalar& s,
                                 const Right& right) {
  if (right.empty()) return;
  TA_ASSERT(!result.empty());
  inplace_tensor_op(
      [s](typename Result::value_type& MADNESS_RESTRICT r,
          const typename Right::value_type& MADNESS_RESTRICT rr) {
        r += rr * s;
      },
      result, right);
}

#ifdef TA_STRIDED_DGEMM_COUNT
inline std::atomic<std::size_t> g_strided_dgemm_ce_e_calls{0};
#endif

/// ce+e strided-DGEMM core (inner OUTER-PRODUCT), looped over the Hadamard-
/// folded nbatch. For each batch b and result cell (m,n):
///   C[m,n](p,q) += factor * sum_k L[m,k](p) * R[k,n](q)
/// as ONE P x Q DGEMM riding the outer-contracted k into BLAS K via the
/// inter-cell slab stride (zero-copy) when the k-run is "clean" (all cells
/// present, uniform inner size, single constant stride); else an inline per-k
/// rank-1 fallback for THAT cell only. Orientation-aware (left_op/right_op pick
/// per-(m,n,k) offsets). M=left-external, N=right-external, K=outer-contracted.
template <typename ResultOuter, typename LeftOuter, typename RightOuter>
void arena_strided_dgemm_ce_e(ResultOuter& C, const LeftOuter& L,
                              const RightOuter& R, std::size_t M, std::size_t N,
                              std::size_t K, math::blas::Op left_op,
                              math::blas::Op right_op, double factor) {
  namespace blas = TiledArray::math::blas;
  using integer = blas::integer;
  static_assert(is_tensor_view_v<typename ResultOuter::value_type> &&
                    is_tensor_view_v<typename LeftOuter::value_type> &&
                    is_tensor_view_v<typename RightOuter::value_type>,
                "arena_strided_dgemm_ce_e: arena (view) inner cells only");
  static_assert(
      std::is_same_v<typename ResultOuter::value_type::numeric_type, double> &&
          std::is_same_v<typename LeftOuter::value_type::numeric_type, double> &&
          std::is_same_v<typename RightOuter::value_type::numeric_type, double>,
      "arena_strided_dgemm_ce_e: double inner storage only");
  if (M == 0 || N == 0 || K == 0) return;
  const std::size_t nbatch = static_cast<std::size_t>(C.nbatch());
  if (nbatch == 0) return;
  const bool shape_ok =
      (C.range().volume() == M * N && L.range().volume() == M * K &&
       R.range().volume() == K * N &&
       static_cast<std::size_t>(L.nbatch()) == nbatch &&
       static_cast<std::size_t>(R.nbatch()) == nbatch);
  TA_ASSERT(shape_ok);
  if (!shape_ok) return;
  ScopedPhaseTimer _kernel_timer(g_kernel_ns_ce_e);
  const std::size_t lda = (left_op == blas::NoTranspose) ? K : M;
  const std::size_t ldb = (right_op == blas::NoTranspose) ? N : K;
  auto a_off = [&](std::size_t m, std::size_t k) {
    return (left_op == blas::NoTranspose) ? m * lda + k : k * lda + m;
  };
  auto b_off = [&](std::size_t k, std::size_t n) {
    return (right_op == blas::NoTranspose) ? k * ldb + n : n * ldb + k;
  };
  const auto* lc = L.data();
  const auto* rc = R.data();
  auto* cc = C.data();
  for (std::size_t b = 0; b < nbatch; ++b) {
    const std::size_t cbase = b * M * N;
    const std::size_t lbase = b * M * K;
    const std::size_t rbase = b * K * N;
    for (std::size_t m = 0; m < M; ++m) {
      for (std::size_t n = 0; n < N; ++n) {
        auto& Cc = cc[cbase + m * N + n];
        if (!Cc) continue;
        const auto _check_t0 = phase_start();
        const auto& l0 = lc[lbase + a_off(m, 0)];
        const auto& r0 = rc[rbase + b_off(0, n)];
        long P = l0 ? static_cast<long>(l0.size()) : -1;
        long Q = r0 ? static_cast<long>(r0.size()) : -1;
        bool clean = (P > 0 && Q > 0 && static_cast<long>(Cc.size()) == P * Q);
        // presence-first: verify every k-cell present + uniform size BEFORE
        // any .data() pointer subtraction.
        for (std::size_t k = 0; clean && k < K; ++k) {
          const auto& lk = lc[lbase + a_off(m, k)];
          const auto& rk = rc[rbase + b_off(k, n)];
          if (!lk || static_cast<long>(lk.size()) != P) clean = false;
          else if (!rk || static_cast<long>(rk.size()) != Q) clean = false;
        }
        long ldA = P, ldB = Q;
        if (clean && K > 1) {
          ldA = static_cast<long>(lc[lbase + a_off(m, 1)].data() - l0.data());
          ldB = static_cast<long>(rc[rbase + b_off(1, n)].data() - r0.data());
          if (ldA < P || ldB < Q) clean = false;
          for (std::size_t k = 0; clean && k < K; ++k) {
            if (lc[lbase + a_off(m, k)].data() !=
                l0.data() + static_cast<std::ptrdiff_t>(k) * ldA)
              clean = false;
            else if (rc[rbase + b_off(k, n)].data() !=
                     r0.data() + static_cast<std::ptrdiff_t>(k) * ldB)
              clean = false;
          }
        }
        phase_stop(g_check_ns_ce_e, _check_t0);
        if (clean) {
          // C(P x Q) += factor * Lmat(P x K) . Rmat^T... realized as
          // gemm(Transpose, NoTranspose): A=K x P slab, B=K x Q slab.
          {
            ScopedShapedGemmTimer _gt(g_gemm_ns_ce_e, g_gemm_calls_ce_e,
                                      g_ce_e_shapes, P, Q, K);
            blas::gemm(blas::Transpose, blas::NoTranspose,
                       /*M=*/static_cast<integer>(P),
                       /*N=*/static_cast<integer>(Q),
                       /*K=*/static_cast<integer>(K), factor,
                       /*A=*/l0.data(), /*lda=*/static_cast<integer>(ldA),
                       /*B=*/r0.data(), /*ldb=*/static_cast<integer>(ldB),
                       /*beta=*/1.0,
                       /*C=*/Cc.data(), /*ldc=*/static_cast<integer>(Q));
          }
#ifdef TA_STRIDED_DGEMM_COUNT
          g_strided_dgemm_ce_e_calls.fetch_add(1, std::memory_order_relaxed);
#endif
        } else {
          ScopedPhaseTimer _fb_timer(g_fallback_ns_ce_e);
          if (gemm_timing_enabled()) {
            int why = classify_run(
                [&](std::size_t k) -> const typename LeftOuter::value_type& {
                  return lc[lbase + a_off(m, k)];
                },
                K);
            if (why == 0) {
              const int wr = classify_run(
                  [&](std::size_t k) -> const typename RightOuter::value_type& {
                    return rc[rbase + b_off(k, n)];
                  },
                  K);
              why = (wr == 0) ? 17 : 10 + wr;
            }
            record_ce_e_fallback(why);
          }
          // inline per-k rank-1 fallback for THIS cell (computed once)
          double* c = Cc.data();
          std::uint64_t _fl = 0;
          for (std::size_t k = 0; k < K; ++k) {
            const auto& lk = lc[lbase + a_off(m, k)];
            const auto& rk = rc[rbase + b_off(k, n)];
            if (!lk || !rk) continue;
            const std::size_t pp = lk.size(), qq = rk.size();
            if (static_cast<long>(Cc.size()) != static_cast<long>(pp * qq))
              continue;
            const double* lp = lk.data();
            const double* rp = rk.data();
            _fl += 2ull * pp * qq;
            for (std::size_t p = 0; p < pp; ++p)
              for (std::size_t q = 0; q < qq; ++q)
                c[p * qq + q] += factor * lp[p] * rp[q];
          }
          if (gemm_timing_enabled())
            g_fall_flops_ce_e.fetch_add(_fl, std::memory_order_relaxed);
        }
      }
    }
  }
}

#ifdef TA_STRIDED_DGEMM_COUNT
inline std::atomic<std::size_t> g_strided_dgemm_ce_ce_right_calls{0};
#endif

/// ce+ce strided-DGEMM core (inner CONTRACTION; ride right-external μ̃ into BLAS
/// M). ORIENTATION-AWARE (offsets derived from left_op/right_op of the OUTER
/// GemmHelper, exactly as arena_strided_dgemm_ce_e) and Hadamard-agnostic:
/// Hadamard is carried as nbatch by the einsum driver, so a thin nbatch loop
/// wraps a fixed-Hadamard ce+ce core and serves both hce+ce (nbatch>1) and the
/// no-Hadamard ce+ce (nbatch==1). Do NOT assert nbatch==1.
///
/// Outer GemmHelper mapping: Mo = left outer-external, No = right outer-external
/// = Mμ, Ko = outer-contracted = nK. The left-external `m` (Mo) is an OUTER loop
/// around the per-(b) body; R (a function of k,μ̃ only) is reused across m.
/// For each batch b, each left-external m, and each outer-contraction cell Κ=k:
///   C̃[m,μ̃, a_1] += factor * Σ_{a_4} R[k,μ̃](a_4) · L[m,k](a_1,a_4)
/// realized as ONE M=μ̃ × N=a_1 × K=a_4 DGEMM riding μ̃ into BLAS M via the
/// (empirically measured) inter-μ̃-cell slab stride (zero-copy), looping k with
/// beta=1. Mo==1 reduces to the original ce+ce kernel exactly. If a per-(b,m)
/// run is not clean, an inline per-cell GEMV fallback handles THAT (b,m) only
/// (each cell once -> no double-count). C must be pre-shaped (a_1-major); the
/// result outer is (m, μ̃) row-major (left-then-right concatenation, matching
/// make_result_range). Accumulates into C (beta=1).
template <typename ResultOuter, typename LeftOuter, typename RightOuter>
void arena_strided_dgemm_ce_ce_right(ResultOuter& C, const LeftOuter& L,
                               const RightOuter& R, std::size_t Mo,
                               std::size_t No, std::size_t Ko,
                               math::blas::Op left_op, math::blas::Op right_op,
                               double factor,
                               bool left_inner_transposed = false) {
  // left_inner_transposed: the external-carrying LEFT inner cell is stored
  // (a4,a1)=Q x P (matrix_transpose) instead of canonical (a1,a4)=P x Q. Folded
  // into the inner GEMM via transb (zero-copy); the right contraction-vector
  // side must remain canonical (gated upstream).
  namespace blas = TiledArray::math::blas;
  using integer = blas::integer;
  static_assert(is_tensor_view_v<typename ResultOuter::value_type> &&
                    is_tensor_view_v<typename LeftOuter::value_type> &&
                    is_tensor_view_v<typename RightOuter::value_type>,
                "arena_strided_dgemm_ce_ce_right: arena (view) inner cells only");
  static_assert(
      std::is_same_v<typename ResultOuter::value_type::numeric_type, double> &&
          std::is_same_v<typename LeftOuter::value_type::numeric_type, double> &&
          std::is_same_v<typename RightOuter::value_type::numeric_type, double>,
      "arena_strided_dgemm_ce_ce_right: double inner storage only");
  const std::size_t Mmu = No;  // right outer-external rides BLAS M
  const std::size_t nK = Ko;   // outer-contracted is looped with beta=1
  const std::size_t nbatch = static_cast<std::size_t>(C.nbatch());
  if (nbatch == 0 || Mmu == 0 || nK == 0 || Mo == 0) return;
  // structural + self-defense (a mis-gated shape falls back / no-ops, never
  // miscomputes). The left-external Mo>=1 is supported (rides as an outer loop).
  const bool shape_ok =
      (C.range().volume() == Mo * Mmu && L.range().volume() == Mo * nK &&
       R.range().volume() == Mmu * nK &&
       static_cast<std::size_t>(L.nbatch()) == nbatch &&
       static_cast<std::size_t>(R.nbatch()) == nbatch);
  // If the structural invariant is violated, do nothing rather than form
  // out-of-bounds cell offsets below. This is only reachable via a mis-gate.
  TA_ASSERT(shape_ok);
  if (!shape_ok) return;
  ScopedPhaseTimer _kernel_timer(g_kernel_ns_ce_ce);
  // orientation-aware outer offsets (mirror arena_strided_dgemm_ce_e a_off/b_off)
  const std::size_t ldb_o = (right_op == blas::NoTranspose) ? No : Ko;
  auto r_off = [&](std::size_t k, std::size_t mu) {
    return (right_op == blas::NoTranspose) ? k * ldb_o + mu : mu * ldb_o + k;
  };
  // L: 2-D (m,k) offset (orientation-aware). l_off(k)==k only held for Mo==1.
  const std::size_t lda_o = (left_op == blas::NoTranspose) ? Ko : Mo;
  auto l_off = [&](std::size_t m, std::size_t k) {
    return (left_op == blas::NoTranspose) ? m * lda_o + k : k * lda_o + m;
  };
  // result outer (Mo x Mμ) row-major: (m, μ̃) = m*Mmu + mu.
  auto c_off = [&](std::size_t m, std::size_t mu) { return m * Mmu + mu; };
  const auto* lc = L.data();
  const auto* rc = R.data();
  auto* cc = C.data();
  for (std::size_t b = 0; b < nbatch; ++b) {
    const std::size_t cbase = b * Mo * Mmu;
    const std::size_t rbase = b * Mmu * nK;
    const std::size_t lbase = b * Mo * nK;
    for (std::size_t m = 0; m < Mo; ++m) {
      // Per-k segment walker (replaces the old all-or-nothing clean gate). For
      // each present left single-cell operand L[m,k], walk the μ̃ axis and emit
      // one strided GEMM per maximal contiguous segment of present, size-matched
      // (C.size==P, R.size==Q), uniformly-strided cells; skip holes. β=1
      // accumulates across k AND across segments. A fully-dense run yields one
      // full-run segment == the old clean GEMM (T1 regression). A genuine size
      // mismatch L[m,k] != P*Q drops to a tiny scalar path for that k only.
      const auto _check_t0 = phase_start();
      // Result inner free index P from the FIRST PRESENT C[m,μ̃] (the run's
      // leading cell may be a hole); Q (operand contraction) is discovered per k
      // from the first present R[k,μ̃]. P is uniform across present result cells
      // by construction (each present segment re-checks C.size==P below).
      long P = -1;
      for (std::size_t mu = 0; mu < Mmu; ++mu) {
        const auto& cmu = cc[cbase + c_off(m, mu)];
        if (cmu) {
          P = static_cast<long>(cmu.size());
          break;
        }
      }
      phase_stop(g_check_ns_ce_ce, _check_t0);
      if (P <= 0) continue;  // result run entirely absent: nothing to write

      // Gated diagnosis: log the segment-length distribution this (b,m) would
      // issue (there is no separate scalar fallback path to attribute now).
      if (gemm_timing_enabled()) {
        auto getC = [&](std::size_t mu) -> const typename ResultOuter::value_type& {
          return cc[cbase + c_off(m, mu)];
        };
        auto getR = [&](std::size_t k, std::size_t mu)
            -> const typename RightOuter::value_type& {
          return rc[rbase + r_off(k, mu)];
        };
        auto getL = [&](std::size_t k) -> const typename LeftOuter::value_type& {
          return lc[lbase + l_off(m, k)];
        };
        int why = classify_run(getC, Mmu);
        if (why == 0) why = classify_operand(getR, getL, Mmu, nK, P);
        if (why != 0) record_ce_ce_fallback(why);
        measure_segments(getC, getR, getL, Mmu, nK, P);
      }

      for (std::size_t k = 0; k < nK; ++k) {
        const auto& lk = lc[lbase + l_off(m, k)];
        if (!lk) continue;  // absent left single-cell operand: skip k (β=1)
        // Discover Q from the first present R[k,μ̃] for this k.
        long Q = -1;
        for (std::size_t mu = 0; mu < Mmu; ++mu) {
          const auto& rmu = rc[rbase + r_off(k, mu)];
          if (rmu) {
            Q = static_cast<long>(rmu.size());
            break;
          }
        }
        if (Q <= 0) continue;  // no present operand cell for this k

        // Defensive: genuine size mismatch L[m,k] != P*Q -> scalar this k only
        // (a strided segment GEMM would be ill-shaped). Each present (μ̃) cell
        // once; β=1. Mirrors the old inline GEMV exactly.
        if (static_cast<long>(lk.size()) != P * Q) {
          ScopedPhaseTimer _fb_timer(g_fallback_ns_ce_ce);
          std::uint64_t _fl = 0;
          const double* l = lk.data();
          for (std::size_t mu = 0; mu < Mmu; ++mu) {
            auto& Cc = cc[cbase + c_off(m, mu)];
            const auto& rk = rc[rbase + r_off(k, mu)];
            if (!Cc || !rk) continue;
            const long Pl = static_cast<long>(Cc.size());
            const long Ql = static_cast<long>(rk.size());
            if (Ql == 0 || static_cast<long>(lk.size()) != Pl * Ql) continue;
            _fl += 2ull * static_cast<std::uint64_t>(Pl) * Ql;
            double* c = Cc.data();
            const double* rr = rk.data();
            for (long a1 = 0; a1 < Pl; ++a1) {
              double acc = 0;
              if (left_inner_transposed) {
                for (long a4 = 0; a4 < Ql; ++a4) acc += l[a4 * Pl + a1] * rr[a4];
              } else {
                const double* lr = l + a1 * Ql;
                for (long a4 = 0; a4 < Ql; ++a4) acc += lr[a4] * rr[a4];
              }
              c[a1] += factor * acc;
            }
          }
          if (gemm_timing_enabled())
            g_fall_flops_ce_ce.fetch_add(_fl, std::memory_order_relaxed);
          continue;
        }

        const double* Lk = lk.data();  // P x Q (or Q x P if transposed)
        std::size_t mu = 0;
        while (mu < Mmu) {
          const auto& rc0 = rc[rbase + r_off(k, mu)];
          auto& cc0 = cc[cbase + c_off(m, mu)];
          // skip holes / size-mismatched cells (cannot join a P/Q segment).
          if (!rc0 || !cc0 || static_cast<long>(cc0.size()) != P ||
              static_cast<long>(rc0.size()) != Q) {
            ++mu;
            continue;
          }
          const double* rstart = rc0.data();  // segment μ̃-run base on R, stride sR
          double* cstart = cc0.data();        // segment μ̃-run base on C, stride sC
          // Grow the maximal segment, recomputing the strides locally (never
          // reuse a run-wide stale stride).
          std::size_t end = mu + 1;
          long sR = -1, sC = -1;
          while (end < Mmu) {
            const auto& rce = rc[rbase + r_off(k, end)];
            const auto& cce = cc[cbase + c_off(m, end)];
            if (!rce || !cce) break;
            if (static_cast<long>(cce.size()) != P ||
                static_cast<long>(rce.size()) != Q)
              break;
            const long dR = static_cast<long>(rce.data() - rstart);
            const long dC = static_cast<long>(cce.data() - cstart);
            const long off = static_cast<long>(end - mu);
            if (off == 1) {
              sR = dR;
              sC = dC;
              if (sR < Q || sC < P) break;  // page-jump / overlap
            } else if (dR != off * sR || dC != off * sC) {
              break;
            }
            ++end;
          }
          const std::size_t Mseg = end - mu;
          const long ldR = (Mseg > 1) ? sR : Q;
          const long ldC = (Mseg > 1) ? sC : P;
          // C(Mseg x P) += factor * R̃(Mseg x Q) · op(L) ; contract a_4(=Q).
          // op(L) is (a4,a1)=Q x P: canonical L is P x Q used transposed
          // (transb=T, ldb=Q); a matrix_transpose left inner is already Q x P,
          // fed transb=N with ldb=P (zero-copy). Threaded identically per
          // segment to the old clean GEMM.
          {
            ScopedShapedGemmTimer _gt(g_gemm_ns_ce_ce, g_gemm_calls_ce_ce,
                                      g_ce_ce_shapes, Mseg, P, Q);
            blas::gemm(
                blas::NoTranspose,
                left_inner_transposed ? blas::NoTranspose : blas::Transpose,
                /*M=*/static_cast<integer>(Mseg),
                /*N=*/static_cast<integer>(P),
                /*K=*/static_cast<integer>(Q), factor,
                /*A=*/rstart, /*lda=*/static_cast<integer>(ldR),
                /*B=*/Lk,
                /*ldb=*/static_cast<integer>(left_inner_transposed ? P : Q),
                /*beta=*/1.0,
                /*C=*/cstart, /*ldc=*/static_cast<integer>(ldC));
          }
#ifdef TA_STRIDED_DGEMM_COUNT
          g_strided_dgemm_ce_ce_right_calls.fetch_add(1,
                                                      std::memory_order_relaxed);
#endif
          mu = end;
        }
      }
    }
  }
}

#ifdef TA_STRIDED_DGEMM_COUNT
inline std::atomic<std::size_t> g_strided_dgemm_ce_ce_left_calls{0};
#endif

/// ce+ce strided-DGEMM core, LEFT-clean mirror of
/// arena_strided_dgemm_ce_ce_right. Here the LEFT operand inner cell is the pure
/// contraction vector L[m,k](a4) (no inner external) and the RIGHT operand
/// carries the inner external R[k,n](a4,b1); the result inner is the right
/// inner-external b1. Rides the LEFT outer-external `m` (Mo) into BLAS M and
/// loops the RIGHT outer-external `n` (No) as an OUTER loop; L (a function of
/// m,k) supplies the strided BLAS-M rows. For each batch b, each right-external
/// n, and each outer-contraction cell k:
///   C[m,n](b1) += factor * Σ_{a4} L[m,k](a4) · R[k,n](a4,b1)
/// realized as ONE M=Mo × N=P(=b1) × K=Q(=a4) DGEMM riding `m` into BLAS M via
/// the inter-m-cell slab stride (zero-copy), looping k with beta=1. If a
/// per-(b,n) run is not clean, an inline per-cell fallback handles THAT (b,n)
/// only (each cell once -> no double-count). Orientation-aware (l_off/r_off from
/// left_op/right_op of the OUTER GemmHelper, exactly as the right core). C must
/// be pre-shaped; the result outer is (m, n) row-major. Accumulates (beta=1).
template <typename ResultOuter, typename LeftOuter, typename RightOuter>
void arena_strided_dgemm_ce_ce_left(ResultOuter& C, const LeftOuter& L,
                                    const RightOuter& R, std::size_t Mo,
                                    std::size_t No, std::size_t Ko,
                                    math::blas::Op left_op,
                                    math::blas::Op right_op, double factor,
                                    bool right_inner_transposed = false) {
  // right_inner_transposed: the external-carrying RIGHT inner cell is stored
  // (b1,a4)=P x Q (matrix_transpose) instead of canonical (a4,b1)=Q x P. Folded
  // into the inner GEMM via transb (zero-copy); the left contraction-vector
  // side must remain canonical (gated upstream).
  namespace blas = TiledArray::math::blas;
  using integer = blas::integer;
  static_assert(is_tensor_view_v<typename ResultOuter::value_type> &&
                    is_tensor_view_v<typename LeftOuter::value_type> &&
                    is_tensor_view_v<typename RightOuter::value_type>,
                "arena_strided_dgemm_ce_ce_left: arena (view) inner cells only");
  static_assert(
      std::is_same_v<typename ResultOuter::value_type::numeric_type, double> &&
          std::is_same_v<typename LeftOuter::value_type::numeric_type, double> &&
          std::is_same_v<typename RightOuter::value_type::numeric_type, double>,
      "arena_strided_dgemm_ce_ce_left: double inner storage only");
  const std::size_t nK = Ko;  // outer-contracted, looped with beta=1
  const std::size_t nbatch = static_cast<std::size_t>(C.nbatch());
  if (nbatch == 0 || Mo == 0 || nK == 0 || No == 0) return;
  const bool shape_ok =
      (C.range().volume() == Mo * No && L.range().volume() == Mo * nK &&
       R.range().volume() == No * nK &&
       static_cast<std::size_t>(L.nbatch()) == nbatch &&
       static_cast<std::size_t>(R.nbatch()) == nbatch);
  TA_ASSERT(shape_ok);
  if (!shape_ok) return;
  ScopedPhaseTimer _kernel_timer(g_kernel_ns_ce_ce);
  // orientation-aware outer offsets (mirror arena_strided_dgemm_ce_ce_right)
  const std::size_t lda_o = (left_op == blas::NoTranspose) ? Ko : Mo;
  auto l_off = [&](std::size_t m, std::size_t k) {
    return (left_op == blas::NoTranspose) ? m * lda_o + k : k * lda_o + m;
  };
  const std::size_t ldb_o = (right_op == blas::NoTranspose) ? No : Ko;
  auto r_off = [&](std::size_t k, std::size_t n) {
    return (right_op == blas::NoTranspose) ? k * ldb_o + n : n * ldb_o + k;
  };
  // result outer (Mo x No) row-major: (m, n) = m*No + n.
  auto c_off = [&](std::size_t m, std::size_t n) { return m * No + n; };
  const auto* lc = L.data();
  const auto* rc = R.data();
  auto* cc = C.data();
  for (std::size_t b = 0; b < nbatch; ++b) {
    const std::size_t cbase = b * Mo * No;
    const std::size_t lbase = b * Mo * nK;
    const std::size_t rbase = b * No * nK;
    for (std::size_t n = 0; n < No; ++n) {  // right-external outer loop
      const auto _check_t0 = phase_start();
      const auto& c0 = cc[cbase + c_off(0, n)];
      long P = c0 ? static_cast<long>(c0.size()) : -1;  // result inner b1
      bool clean = (P > 0);
      // C m-run (fixed n): uniform size P, constant stride sC>=P (page-jump
      // guard). Presence-first before any cell-0->1 pointer subtraction.
      long sC = P;
      if (clean && Mo > 1) {
        for (std::size_t m = 0; clean && m < Mo; ++m) {
          const auto& cm = cc[cbase + c_off(m, n)];
          if (!cm || static_cast<long>(cm.size()) != P) clean = false;
        }
        if (clean) {
          sC = static_cast<long>(cc[cbase + c_off(1, n)].data() - c0.data());
          if (sC < P) clean = false;
          for (std::size_t m = 0; clean && m < Mo; ++m) {
            if (cc[cbase + c_off(m, n)].data() !=
                c0.data() + static_cast<std::ptrdiff_t>(m) * sC)
              clean = false;
          }
        }
      }
      // Q from L[m=0,k=0]; R[k,n] size Q*P; L m-run per k uniform Q at constant
      // stride sA>=Q (uniform across k) -- page-jump guard on L too.
      const auto* l00 = clean ? &lc[lbase + l_off(0, 0)] : nullptr;
      long Q = (l00 && *l00) ? static_cast<long>(l00->size()) : -1;
      if (Q <= 0) clean = false;
      long sA = Q;
      for (std::size_t k = 0; clean && k < nK; ++k) {
        const auto& rk = rc[rbase + r_off(k, n)];
        if (!rk || static_cast<long>(rk.size()) != P * Q) {
          clean = false;
          break;
        }
        const auto& lk0 = lc[lbase + l_off(0, k)];
        if (!lk0 || static_cast<long>(lk0.size()) != Q) {
          clean = false;
          break;
        }
        if (Mo > 1) {
          for (std::size_t m = 0; clean && m < Mo; ++m) {
            const auto& lm = lc[lbase + l_off(m, k)];
            if (!lm || static_cast<long>(lm.size()) != Q) clean = false;
          }
          if (!clean) break;
          const long sAk =
              static_cast<long>(lc[lbase + l_off(1, k)].data() - lk0.data());
          if (sAk < Q) {
            clean = false;
            break;
          }
          if (k == 0) sA = sAk;
          else if (sAk != sA) {
            clean = false;
            break;
          }
          for (std::size_t m = 0; clean && m < Mo; ++m) {
            if (lc[lbase + l_off(m, k)].data() !=
                lk0.data() + static_cast<std::ptrdiff_t>(m) * sA)
              clean = false;
          }
        }
      }
      phase_stop(g_check_ns_ce_ce, _check_t0);
      if (clean) {
        if (gemm_timing_enabled())
          g_clean_runs_ce_ce.fetch_add(1, std::memory_order_relaxed);
        double* Cd = cc[cbase + c_off(0, n)].data();  // m-run base, stride sC
        for (std::size_t k = 0; k < nK; ++k) {
          const double* Ak =
              lc[lbase + l_off(0, k)].data();  // m-run base for k, stride sA
          const double* Bk = rc[rbase + r_off(k, n)].data();
          // C(Mo x P) += factor * A(Mo x Q) . B(Q x P) ; contract a4(=Q).
          // Canonical B is (a4,b1)=Q x P row-major (transb=N, ldb=P). When the
          // right inner cell is matrix_transpose it is stored (b1,a4)=P x Q
          // row-major, so feed transb=T with ldb=Q (op(B)=(a4,b1)); zero-copy.
          {
            ScopedShapedGemmTimer _gt(g_gemm_ns_ce_ce, g_gemm_calls_ce_ce,
                                      g_ce_ce_shapes, Mo, P, Q);
            blas::gemm(
                blas::NoTranspose,
                right_inner_transposed ? blas::Transpose : blas::NoTranspose,
                /*M=*/static_cast<integer>(Mo),
                /*N=*/static_cast<integer>(P),
                /*K=*/static_cast<integer>(Q), factor,
                /*A=*/Ak, /*lda=*/static_cast<integer>(sA),
                /*B=*/Bk,
                /*ldb=*/static_cast<integer>(right_inner_transposed ? Q : P),
                /*beta=*/1.0,
                /*C=*/Cd, /*ldc=*/static_cast<integer>(sC));
          }
#ifdef TA_STRIDED_DGEMM_COUNT
          g_strided_dgemm_ce_ce_left_calls.fetch_add(1,
                                                     std::memory_order_relaxed);
#endif
        }
      } else {
        ScopedPhaseTimer _fb_timer(g_fallback_ns_ce_ce);
        if (gemm_timing_enabled()) {
          int why = classify_run(
              [&](std::size_t m) -> const typename ResultOuter::value_type& {
                return cc[cbase + c_off(m, n)];
              },
              Mo);
          if (why == 0) {
            const auto& c0d = cc[cbase + c_off(0, n)];
            const long Pr = c0d ? static_cast<long>(c0d.size()) : -1;
            // strided operand here is L (m-run); single-cell operand is R[k,n].
            why = classify_operand(
                [&](std::size_t k, std::size_t m)
                    -> const typename LeftOuter::value_type& {
                  return lc[lbase + l_off(m, k)];
                },
                [&](std::size_t k) -> const typename RightOuter::value_type& {
                  return rc[rbase + r_off(k, n)];
                },
                Mo, nK, Pr);
          }
          record_ce_ce_fallback(why);
          std::size_t pk = 0;
          const auto& c0k = cc[cbase + c_off(0, n)];
          const long Pk = c0k ? static_cast<long>(c0k.size()) : -1;
          if (Pk > 0 &&
              kskip_rescuable(
                  [&](std::size_t m)
                      -> const typename ResultOuter::value_type& {
                    return cc[cbase + c_off(m, n)];
                  },
                  [&](std::size_t k, std::size_t m)
                      -> const typename LeftOuter::value_type& {
                    return lc[lbase + l_off(m, k)];
                  },
                  [&](std::size_t k) -> const typename RightOuter::value_type& {
                    return rc[rbase + r_off(k, n)];
                  },
                  Mo, nK, Pk, pk)) {
            g_fall_kskip_ce_ce.fetch_add(1, std::memory_order_relaxed);
            g_fall_kskip_present_ce_ce.fetch_add(pk, std::memory_order_relaxed);
          }
          if (Pk > 0 &&
              gather_rescuable(
                  [&](std::size_t m)
                      -> const typename ResultOuter::value_type& {
                    return cc[cbase + c_off(m, n)];
                  },
                  [&](std::size_t k, std::size_t m)
                      -> const typename LeftOuter::value_type& {
                    return lc[lbase + l_off(m, k)];
                  },
                  [&](std::size_t k) -> const typename RightOuter::value_type& {
                    return rc[rbase + r_off(k, n)];
                  },
                  Mo, nK, Pk))
            g_fall_gather_ce_ce.fetch_add(1, std::memory_order_relaxed);
          if (Pk > 0)
            measure_segments(
                [&](std::size_t m)
                    -> const typename ResultOuter::value_type& {
                  return cc[cbase + c_off(m, n)];
                },
                [&](std::size_t k, std::size_t m)
                    -> const typename LeftOuter::value_type& {
                  return lc[lbase + l_off(m, k)];
                },
                [&](std::size_t k) -> const typename RightOuter::value_type& {
                  return rc[rbase + r_off(k, n)];
                },
                Mo, nK, Pk);
        }
        std::uint64_t _fl = 0;
        // inline per-cell fallback for THIS (b,n) (each cell once)
        for (std::size_t m = 0; m < Mo; ++m) {
          auto& Cc = cc[cbase + c_off(m, n)];
          if (!Cc) continue;
          const long Pl = static_cast<long>(Cc.size());
          double* c = Cc.data();
          for (std::size_t k = 0; k < nK; ++k) {
            const auto& lk = lc[lbase + l_off(m, k)];
            const auto& rk = rc[rbase + r_off(k, n)];
            if (!lk || !rk) continue;
            const long Ql = static_cast<long>(lk.size());
            if (Ql == 0 || static_cast<long>(rk.size()) != Ql * Pl) continue;
            _fl += 2ull * static_cast<std::uint64_t>(Pl) * Ql;
            const double* a = lk.data();   // Ql vector
            const double* bd = rk.data();  // canonical Ql x Pl row-major
            // canonical B(a4,p) = bd[a4*Pl + p]; transposed (b1,a4)=Pl x Ql
            // row-major B(a4,p) = bd[p*Ql + a4].
            for (long a4 = 0; a4 < Ql; ++a4) {
              const double av = a[a4];
              if (right_inner_transposed) {
                for (long p = 0; p < Pl; ++p)
                  c[p] += factor * av * bd[p * Ql + a4];
              } else {
                const double* br = bd + a4 * Pl;
                for (long p = 0; p < Pl; ++p) c[p] += factor * av * br[p];
              }
            }
          }
        }
        if (gemm_timing_enabled())
          g_fall_flops_ce_ce.fetch_add(_fl, std::memory_order_relaxed);
      }
    }
  }
}

/// Creates a fused contraction callback.
template <typename Result, typename Left, typename Right, typename Op>
auto make_fused_contraction_lambda(Op contrreduce_op) {
  return
      [contrreduce_op](Result& result, const Left& left, const Right& right) {
        TA_ASSERT(!contrreduce_op.perm());
        fused_contraction_inplace(result, left, right, contrreduce_op.factor(),
                                  contrreduce_op.gemm_helper());
      };
}

/// Hadamard-outer, contraction-inner ToT x ToT product into a fresh arena
/// tile. `left` and `right` share the (Hadamard) outer layout; each result
/// outer cell is the inner GEMM of the corresponding left/right inner cells,
/// shaped by `inner_gh`. `cell_op(result_cell, left_cell, right_cell)` runs
/// the per-cell in-place contraction (e.g. the make_fused_contraction_lambda
/// callback). The per-cell op is perm-free; a non-identity `inner_perm`
/// permutes the result cells' inner modes as a slab-level post-pass.
template <typename Result, typename Left, typename Right, typename CellOp>
Result arena_hadamard_inner_contract(const Left& left, const Right& right,
                                     const math::GemmHelper& inner_gh,
                                     const CellOp& cell_op,
                                     const Permutation& inner_perm) {
  using inner_range_t = typename Result::value_type::range_type;
  TA_ASSERT(left.range().volume() == right.range().volume());
  TA_ASSERT(left.nbatch() == right.nbatch());
  const std::size_t N_cells = left.range().volume() * left.nbatch();
  auto range_fn = [&left, &right, &inner_gh](std::size_t ord) -> inner_range_t {
    const auto& lc = left.data()[ord];
    const auto& rc = right.data()[ord];
    if (lc.empty() || rc.empty()) return inner_range_t{};
    return inner_gh.template make_result_range<inner_range_t>(lc.range(),
                                                              rc.range());
  };
  Result result =
      arena_outer_init<Result>(left.range(), left.nbatch(), range_fn);
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    if (result.data()[ord].empty()) continue;
    cell_op(result.data()[ord], left.data()[ord], right.data()[ord]);
  }
  if (inner_perm && !inner_perm.is_identity())
    result = arena_inner_permute<Result>(result, inner_perm);
  return result;
}

/// Hadamard-outer, phantom-unit-denest-inner ToT x ToT product into a fresh
/// arena tile. Like arena_hadamard_inner_contract, but each result outer cell
/// is a unit-extent [1]^phantom_rank cell (the inner product is a full
/// contraction = a flat dot; there are no real result inner modes). `cell_op`
/// (the phantom-dot per-cell op) fills the pre-shaped unit cell. No inner
/// permutation: phantom modes are all unit-extent.
template <typename Result, typename Left, typename Right, typename CellOp>
Result arena_hadamard_phantom_dot(const Left& left, const Right& right,
                                  std::size_t phantom_rank,
                                  const CellOp& cell_op) {
  using inner_range_t = typename Result::value_type::range_type;
  TA_ASSERT(left.range().volume() == right.range().volume());
  TA_ASSERT(left.nbatch() == right.nbatch());
  const std::size_t N_cells = left.range().volume() * left.nbatch();
  const container::vector<std::size_t> unit_ext(phantom_rank, std::size_t{1});
  auto range_fn = [&left, &right, &unit_ext](std::size_t ord) -> inner_range_t {
    const auto& lc = left.data()[ord];
    const auto& rc = right.data()[ord];
    if (lc.empty() || rc.empty()) return inner_range_t{};
    return inner_range_t(unit_ext);
  };
  Result result =
      arena_outer_init<Result>(left.range(), left.nbatch(), range_fn);
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    if (result.data()[ord].empty()) continue;
    cell_op(result.data()[ord], left.data()[ord], right.data()[ord]);
  }
  return result;
}

/// Creates a fused Hadamard callback.
template <typename Result, typename Left, typename Right>
auto make_fused_hadamard_lambda() {
  return [](Result& result, const Left& left, const Right& right) {
    fused_hadamard_inplace(result, left, right);
  };
}

/// Creates a fused scaled-Hadamard callback.
template <typename Result, typename Left, typename Right, typename Scalar>
auto make_fused_hadamard_scaled_lambda(Scalar factor) {
  return [factor](Result& result, const Left& left, const Right& right) {
    fused_hadamard_scaled_inplace(result, left, right, factor);
  };
}

/// Creates a fused ToT-times-scalar callback.
template <typename Result, typename Left, typename Right>
auto make_fused_scale_tot_x_t_lambda() {
  return [](Result& result, const Left& left, const Right& right) {
    fused_scale_tot_x_t_inplace(result, left, right);
  };
}

/// Creates a fused scalar-times-ToT callback.
template <typename Result, typename Left, typename Right>
auto make_fused_scale_t_x_tot_lambda() {
  return [](Result& result, const Left& left, const Right& right) {
    fused_scale_t_x_tot_inplace(result, left, right);
  };
}

/// Discriminates the per-cell operation used by the arena regime-A path.
enum class RegimeAInnerKind {
  hadamard,
  contraction,
  scale_left,   // ToT × plain T → ToT (right operand contributes scalars)
  scale_right,  // plain T × ToT → ToT (left operand contributes scalars)
  phantom_dot   // full inner contraction (dot) into a unit-extent result cell;
                // the result keeps only phantom-unit inner modes (see
                // is_phantom_unit_label). Operand cells are read flat, so no
                // operand carries the phantom mode and no GEMM rank match is
  // required. Realizes the ToT×ToT→plain-T (DeNest) inner product.
};

/// Permute the extents of `src` by `perm` and materialize a range of type
/// `RangeT`. Generic over the inner-cell range types regime-A einsum sees:
/// `TA::Range` (legacy `Tensor<Tensor>` inners) and `btas::zb::RangeNd`
/// (`Tensor<ArenaTensor>` inners). `Permutation * Range` only exists for
/// `TA::Range`, so the permutation is applied to a plain extent vector and
/// the target range is rebuilt from the result.
template <typename RangeT, typename SrcRange>
RangeT arena_make_permuted_range(const TiledArray::Permutation& perm,
                                 const SrcRange& src) {
  const std::size_t rank = src.rank();
  const auto& src_ext = src.extent();
  container::svector<std::size_t> ext(rank);
  for (std::size_t d = 0; d < rank; ++d)
    ext[d] = static_cast<std::size_t>(src_ext[d]);
  if (perm && !perm.is_identity()) {
    TA_ASSERT(perm.size() == rank);
    return RangeT(perm * ext);
  }
  return RangeT(ext);
}

/// Holds the inner operation plan for arena regime-A dispatch.
template <typename Result, typename A, typename B, typename Inner>
struct RegimeAArenaPlan {
  using Annot = ::Einsum::Index<std::string>;

  bool active = false;
  RegimeAInnerKind kind = RegimeAInnerKind::hadamard;

  // Exactly one plan optional is engaged; optionals avoid default construction.
  std::optional<TensorHadamardPlan<Annot>> h_plan{};
  std::optional<TensorContractionPlan<Annot>> c_plan{};

  // For kind == phantom_dot: the number of phantom-unit result modes (the rank
  // of the unit-extent result inner cell, e.g. 1 for `⊗₁`).
  std::size_t phantom_rank = 0;

  /// Derives the result inner range from a non-empty input-cell pair.
  template <typename InnerRange, typename LRange, typename RRange>
  InnerRange derive_inner_range(const LRange& l_range,
                                const RRange& r_range) const {
    switch (kind) {
      case RegimeAInnerKind::hadamard:
        TA_ASSERT(h_plan.has_value());
        return arena_make_permuted_range<InnerRange>(h_plan->perm.AC, l_range);
      case RegimeAInnerKind::contraction: {
        TA_ASSERT(c_plan.has_value());
        const auto& p = *c_plan;
        using PlanIndices = std::remove_cvref_t<decltype(p.A)>;
        using PlanIndex = typename PlanIndices::value_type;
        using Extent =
            std::remove_cv_t<typename decltype(std::declval<TiledArray::Range>()
                                                   .extent())::value_type>;
        using ExtentMap = ::Einsum::index::IndexMap<PlanIndex, Extent>;
        ExtentMap extent = (ExtentMap{p.A, l_range.extent()} |
                            ExtentMap{p.B, r_range.extent()});
        container::vector<Extent> rng;
        rng.reserve(p.e.size());
        for (auto&& ix : p.e) rng.emplace_back(extent[ix]);
        return InnerRange(rng);
      }
      case RegimeAInnerKind::scale_left:
        // Scale-left preserves the ToT operand's inner range.
        return InnerRange(l_range);
      case RegimeAInnerKind::scale_right:
        return InnerRange(r_range);
      case RegimeAInnerKind::phantom_dot: {
        // The result keeps only phantom-unit modes: a rank-`phantom_rank`,
        // all-unit-extent cell (e.g. [1] for `⊗₁`).
        container::vector<std::size_t> ext(phantom_rank, std::size_t{1});
        return InnerRange(ext);
      }
    }
    TA_ASSERT(false && "RegimeAInnerKind: unhandled kind");
    return InnerRange{};
  }

  /// Accumulates one input-cell pair into the result cell.
  template <typename ResultCell, typename LCell, typename RCell>
  void accumulate(ResultCell& r, const LCell& l, const RCell& rr) const {
    switch (kind) {
      case RegimeAInnerKind::hadamard: {
        if constexpr (is_arena_inner_cell_v<LCell> &&
                      is_arena_inner_cell_v<RCell>) {
          if (l.empty() || rr.empty()) return;
          TA_ASSERT(h_plan.has_value());
          // run_regime_a_arena has already hoisted any operand inner
          // permutation, so l and rr are both in C-layout: the per-cell op
          // is a flat r += l * rr on congruent cells.
          fused_hadamard_inplace(r, l, rr);
        }
        return;
      }
      case RegimeAInnerKind::contraction: {
        if constexpr (is_arena_inner_cell_v<LCell> &&
                      is_arena_inner_cell_v<RCell>) {
          if (l.empty() || rr.empty()) return;
          TA_ASSERT(c_plan.has_value());
          // run_regime_a_arena has already hoisted any operand inner
          // permutation, so l and rr are in canonical (blas_layout) order:
          // the per-cell op is a single canonical GEMM into r with beta=1.
          // Uniform for TA::Tensor and ArenaTensor cells (free `gemm` CPO).
          using Scalar = typename std::remove_cv_t<ResultCell>::numeric_type;
          fused_contraction_inplace(r, l, rr, Scalar{1}, c_plan->gemm_helper);
        }
        return;
      }
      case RegimeAInnerKind::scale_left: {
        // Scale-left receives a ToT inner cell and a scalar.
        if constexpr (is_arena_inner_cell_v<LCell> &&
                      !is_arena_inner_cell_v<RCell>) {
          if (l.empty()) return;
          fused_scale_tot_x_t_inplace(r, l, rr);
        }
        return;
      }
      case RegimeAInnerKind::scale_right: {
        if constexpr (!is_arena_inner_cell_v<LCell> &&
                      is_arena_inner_cell_v<RCell>) {
          if (rr.empty()) return;
          fused_scale_t_x_tot_inplace(r, l, rr);
        }
        return;
      }
      case RegimeAInnerKind::phantom_dot: {
        if constexpr (is_arena_inner_cell_v<LCell> &&
                      is_arena_inner_cell_v<RCell>) {
          if (l.empty() || rr.empty()) return;
          // Full inner contraction with only phantom-unit modes surviving: a
          // flat (non-conjugating) dot of the operand cells -- the same value a
          // GEMM with M=N=1,K=vol would compute -- accumulated into the lone
          // element of the unit-extent result cell. Reads operands flat, so no
          // operand need carry the phantom mode and no rank match is required;
          // uniform for TA::Tensor and ArenaTensor cells.
          using Numeric = typename std::remove_cv_t<ResultCell>::numeric_type;
          const std::size_t n = l.range().volume();
          TA_ASSERT(n == rr.range().volume());
          const auto* MADNESS_RESTRICT lp = l.data();
          const auto* MADNESS_RESTRICT rp = rr.data();
          Numeric acc{0};
          for (std::size_t j = 0; j < n; ++j) acc += lp[j] * rp[j];
          r.data()[0] += acc;
        }
        return;
      }
    }
  }
};

/// Builds an arena regime-A plan when result and permutation constraints allow
/// it.
template <typename Result, typename A, typename B, typename Inner,
          typename PermT>
auto make_regime_a_arena_plan(const A& a, const B& b, const Inner& inner,
                              const PermT& inner_perm)
    -> RegimeAArenaPlan<Result, A, B, Inner> {
  using Plan = RegimeAArenaPlan<Result, A, B, Inner>;
  Plan plan;
  if (arena_disabled()) return plan;
  if constexpr (!is_arena_eligible_outer_v<Result>) {
    return plan;
  } else {
    // `inner_perm` (== C.permutation at the call site) is the result *outer*
    // permutation. run_regime_a_arena applies it itself via tile.permute(pc)
    // -- byte-identical to the legacy non-arena path, and supported for an
    // arena ToT via arena_permute_shallow -- so it does not gate the plan.
    // Inner-operand and inner-result permutations are likewise handled, by
    // hoisting them to slab-level arena_inner_permute rewrites (see below).
    (void)inner_perm;

    using ArrayA_t = std::remove_cvref_t<decltype(a.array)>;
    using ArrayB_t = std::remove_cvref_t<decltype(b.array)>;
    // "Tot" here means "tile is a ToT-like thing whose inner cell is the
    // tensor we want to operate on"; covers both legacy TA::Tensor inners
    // and pinned ArenaTensor inners.
    constexpr bool a_is_tot =
        is_arena_eligible_outer_v<typename ArrayA_t::value_type>;
    constexpr bool b_is_tot =
        is_arena_eligible_outer_v<typename ArrayB_t::value_type>;

    if constexpr (a_is_tot && b_is_tot) {
      if (static_cast<bool>(inner.h)) {
        plan.kind = RegimeAInnerKind::hadamard;
        plan.h_plan.emplace(inner.A, inner.B, inner.C);
        // A non-canonical inner Hadamard (h_plan.perm.{AC,BC} non-identity)
        // is handled the same way as a non-canonical inner contraction:
        // run_regime_a_arena hoists each operand inner permutation to a
        // slab-level rewrite (arena_inner_permute) so both operands reach
        // C-layout before the per-cell flat r += l * rr. No need to bail.
      } else if (bool(inner.C) && [&] {
                   for (const auto& lbl : inner.C)
                     if (!::TiledArray::detail::is_phantom_unit_label(lbl))
                       return false;
                   return true;
                 }()) {
        // Phantom-unit denest: every surviving result inner mode is a phantom
        // unit (⊗ₙ), i.e. the real inner modes are fully contracted. Realize
        // the inner product as a flat dot into a unit-extent result cell --
        // operands are read flat, so neither needs to carry the phantom mode
        // (no GEMM, no TensorContractionPlan rank match). See accumulate().
        plan.kind = RegimeAInnerKind::phantom_dot;
        plan.phantom_rank = inner.C.size();
      } else {
        plan.kind = RegimeAInnerKind::contraction;
        plan.c_plan.emplace(inner.A, inner.B, inner.C);
        // A non-canonical inner contraction (c_plan.do_perm.{A,B,C} set --
        // e.g. M/K- or M/N-interleaved inner annotations that are not
        // GEMM-absorbable transposes) is still handled: run_regime_a_arena
        // hoists each operand inner permutation, and the result inner
        // permutation, to slab-level rewrites (arena_inner_permute), leaving
        // the per-cell op a single canonical GEMM. No need to bail here.
      }
    } else if constexpr (a_is_tot && !b_is_tot) {
      plan.kind = RegimeAInnerKind::scale_left;
    } else if constexpr (!a_is_tot && b_is_tot) {
      plan.kind = RegimeAInnerKind::scale_right;
    } else {
      return plan;
    }
    plan.active = true;
    (void)a;
    (void)b;
    return plan;
  }
}

/// Kill switch for the regime-A hc+e strided-DGEMM reuse path: when true,
/// run_regime_a_arena keeps the legacy per-cell accumulate. Test/bench hook
/// for the strided-vs-per-cell differential (correctness) and the perf
/// measurement; production default is false (strided on). Mirrors
/// arena_disabled().
inline bool& regime_a_strided_disabled() {
  static bool flag = false;
  return flag;
}

/// Runs the arena regime-A path for one H-slice when the plan is active.
template <typename Plan, typename HIndex, typename TermA, typename TermB,
          typename TermC, typename LocalTiles, typename Tiles, typename Trange>
bool run_regime_a_arena(const Plan& plan, const HIndex& h, std::size_t batch,
                        const TermA& A, const TermB& B, const TermC& C,
                        LocalTiles& C_local_tiles, const Tiles& tiles,
                        const Trange& trange) {
  if (!plan.active) return false;

  using ResultTensor = typename LocalTiles::value_type::second_type;
  // Guard avoids naming inner-cell APIs for non-ToT instantiations.
  using ArrayA_t = std::remove_cvref_t<decltype(A.array)>;
  using ArrayB_t = std::remove_cvref_t<decltype(B.array)>;
  // ToT-like in the regime-A sense: tile is an arena-eligible outer
  // (legacy TA::Tensor inner or pinned ArenaTensor inner).
  constexpr bool a_is_tot =
      is_arena_eligible_outer_v<typename ArrayA_t::value_type>;
  constexpr bool b_is_tot =
      is_arena_eligible_outer_v<typename ArrayB_t::value_type>;
  if constexpr (!is_arena_eligible_outer_v<ResultTensor> ||
                (!a_is_tot && !b_is_tot)) {
    (void)h;
    (void)batch;
    (void)A;
    (void)B;
    (void)C;
    (void)C_local_tiles;
    (void)tiles;
    (void)trange;
    return false;
  } else {
    using InnerT = typename ResultTensor::value_type;
    using InnerRange = typename InnerT::range_type;

    const auto& pa = A.permutation;
    const auto& pb = B.permutation;
    const auto& pc = C.permutation;
    auto const c = apply(pc, h);

    if constexpr (a_is_tot && b_is_tot) {
      using IIndex = ::Einsum::index::Index<std::size_t>;
      // hc+e reuse gate: the result/operand inner cells must be the kernel's
      // (view + double) inner type; mirror arena_strided_dgemm_ce_e's
      // static_assert so non-view / non-double ToT keep the per-cell path.
      using LInnerT = typename ArrayA_t::value_type::value_type;
      using RInnerT = typename ArrayB_t::value_type::value_type;
      constexpr bool ce_e_kernel_ok =
          is_tensor_view_v<InnerT> && is_tensor_view_v<LInnerT> &&
          is_tensor_view_v<RInnerT> &&
          std::is_same_v<typename InnerT::numeric_type, double> &&
          std::is_same_v<typename LInnerT::numeric_type, double> &&
          std::is_same_v<typename RInnerT::numeric_type, double>;
      // Inner OUTER-PRODUCT (K_inner==0) is the strided-reusable shape; any
      // inner contraction (hc+ce) stays per-cell (two-level stride). The
      // runtime toggle lets tests/benches force the per-cell path.
      const bool hce_e_strided =
          plan.kind == RegimeAInnerKind::contraction && plan.c_plan &&
          plan.c_plan->gemm_helper.num_contract_ranks() == 0 &&
          !regime_a_strided_disabled();
      auto range_for = [&](std::size_t k) -> InnerRange {
        if (k >= batch) return InnerRange{};
        for (IIndex i : tiles) {
          const auto pahi_inv = apply_inverse(pa, h + i);
          const auto pbhi_inv = apply_inverse(pb, h + i);
          if (A.array.is_zero(pahi_inv) || B.array.is_zero(pbhi_inv)) continue;
          auto ai = A.array.find(pahi_inv).get();
          auto bi = B.array.find(pbhi_inv).get();
          if (pa) ai = ai.permute(pa);
          if (pb) bi = bi.permute(pb);
          auto shape = trange.tile(i);
          ai = ai.reshape(shape, batch);
          bi = bi.reshape(shape, batch);
          auto aik = ai.batch(k);
          auto bik = bi.batch(k);
          auto vol = aik.total_size();
          TA_ASSERT(vol == bik.total_size());
          for (decltype(vol) j = 0; j < vol; ++j) {
            const auto& l_inner = aik.data()[j];
            const auto& r_inner = bik.data()[j];
            if (l_inner.empty() || r_inner.empty()) continue;
            return plan.template derive_inner_range<InnerRange>(
                l_inner.range(), r_inner.range());
          }
        }
        return InnerRange{};
      };

      ResultTensor tile = arena_outer_init<ResultTensor>(
          TiledArray::Range{batch}, /*batch_sz=*/1, range_for);

      for (IIndex i : tiles) {
        const auto pahi_inv = apply_inverse(pa, h + i);
        const auto pbhi_inv = apply_inverse(pb, h + i);
        if (A.array.is_zero(pahi_inv) || B.array.is_zero(pbhi_inv)) continue;
        auto ai = A.array.find(pahi_inv).get();
        auto bi = B.array.find(pbhi_inv).get();
        if (pa) ai = ai.permute(pa);
        if (pb) bi = bi.permute(pb);
        // Hoist a non-canonical inner op's operand inner permutations to
        // slab-level rewrites, so the per-cell op below stays canonical:
        // contraction -> a single canonical GEMM; Hadamard -> a flat
        // r += l * rr on congruent C-layout cells. No per-cell view permute.
        if (plan.kind == RegimeAInnerKind::contraction) {
          const auto& cp = *plan.c_plan;
          if (cp.do_perm.A)
            ai = arena_inner_permute<decltype(ai)>(ai, cp.perm.A);
          if (cp.do_perm.B)
            bi = arena_inner_permute<decltype(bi)>(bi, cp.perm.B);
        } else if (plan.kind == RegimeAInnerKind::hadamard) {
          const auto& hp = *plan.h_plan;
          if (!hp.perm.AC.is_identity())
            ai = arena_inner_permute<decltype(ai)>(ai, hp.perm.AC);
          if (!hp.perm.BC.is_identity())
            bi = arena_inner_permute<decltype(bi)>(bi, hp.perm.BC);
        }
        auto shape = trange.tile(i);
        ai = ai.reshape(shape, batch);
        bi = bi.reshape(shape, batch);
        if constexpr (ce_e_kernel_ok) {
          if (hce_e_strided) {
            // hc+e: ride the within-tile contraction cells into BLAS K via the
            // landed ce+e core. M=N=1, K=vol; kernel nbatch == Hadamard batch.
            // cview shares tile's data_ (storage-aliasing reshape), so the
            // kernel's beta=1 writes accumulate into tile across the i-loop.
            namespace blas = TiledArray::math::blas;
            const std::size_t Kvol =
                static_cast<std::size_t>(trange.tile(i).volume());
            auto cview = tile.reshape(TiledArray::Range{1}, batch);
            arena_strided_dgemm_ce_e(cview, ai, bi, /*M=*/std::size_t{1},
                                     /*N=*/std::size_t{1}, /*K=*/Kvol,
                                     blas::NoTranspose, blas::NoTranspose,
                                     /*factor=*/1.0);
            continue;  // tile-i contribution complete
          }
        }
        for (std::size_t k = 0; k < batch; ++k) {
          auto& cell = tile({k});
          if (cell.empty()) continue;
          auto aik = ai.batch(k);
          auto bik = bi.batch(k);
          auto vol = aik.total_size();
          TA_ASSERT(vol == bik.total_size());
          for (decltype(vol) j = 0; j < vol; ++j) {
            const auto& l_inner = aik.data()[j];
            const auto& r_inner = bik.data()[j];
            plan.accumulate(cell, l_inner, r_inner);
          }
        }
      }

      // Hoist the result inner permutation: cells were accumulated in
      // blas_layout (e) order; rewrite the slab to the C inner order.
      if (plan.kind == RegimeAInnerKind::contraction && plan.c_plan->do_perm.C)
        tile =
            arena_inner_permute<ResultTensor>(tile, plan.c_plan->perm.C.inv());
      auto shape = apply_inverse(pc, C.array.trange().tile(c));
      tile = tile.reshape(shape);
      if (pc) tile = tile.permute(pc);
      C_local_tiles.emplace_back(std::move(c), std::move(tile));
      return true;
    } else {
      // Scale path has exactly one ToT operand and one scalar-cell operand.
      using IIndex = ::Einsum::index::Index<std::size_t>;
      auto range_for = [&](std::size_t k) -> InnerRange {
        if (k >= batch) return InnerRange{};
        for (IIndex i : tiles) {
          const auto pahi_inv = apply_inverse(pa, h + i);
          const auto pbhi_inv = apply_inverse(pb, h + i);
          if (A.array.is_zero(pahi_inv) || B.array.is_zero(pbhi_inv)) continue;
          auto ai = A.array.find(pahi_inv).get();
          auto bi = B.array.find(pbhi_inv).get();
          if (pa) ai = ai.permute(pa);
          if (pb) bi = bi.permute(pb);
          auto shape = trange.tile(i);
          ai = ai.reshape(shape, batch);
          bi = bi.reshape(shape, batch);
          auto aik = ai.batch(k);
          auto bik = bi.batch(k);
          if constexpr (a_is_tot) {
            auto vol = aik.total_size();
            for (decltype(vol) j = 0; j < vol; ++j) {
              const auto& l_inner = aik.data()[j];
              if (l_inner.empty()) continue;
              return InnerRange(l_inner.range());
            }
          } else {
            auto vol = bik.total_size();
            for (decltype(vol) j = 0; j < vol; ++j) {
              const auto& r_inner = bik.data()[j];
              if (r_inner.empty()) continue;
              return InnerRange(r_inner.range());
            }
          }
        }
        return InnerRange{};
      };

      ResultTensor tile = arena_outer_init<ResultTensor>(
          TiledArray::Range{batch}, /*batch_sz=*/1, range_for);

      for (IIndex i : tiles) {
        const auto pahi_inv = apply_inverse(pa, h + i);
        const auto pbhi_inv = apply_inverse(pb, h + i);
        if (A.array.is_zero(pahi_inv) || B.array.is_zero(pbhi_inv)) continue;
        auto ai = A.array.find(pahi_inv).get();
        auto bi = B.array.find(pbhi_inv).get();
        if (pa) ai = ai.permute(pa);
        if (pb) bi = bi.permute(pb);
        auto shape = trange.tile(i);
        ai = ai.reshape(shape, batch);
        bi = bi.reshape(shape, batch);
        for (std::size_t k = 0; k < batch; ++k) {
          auto& cell = tile({k});
          if (cell.empty()) continue;
          auto aik = ai.batch(k);
          auto bik = bi.batch(k);
          auto vol = aik.total_size();
          TA_ASSERT(vol == bik.total_size());
          for (decltype(vol) j = 0; j < vol; ++j) {
            const auto& l_elem = aik.data()[j];
            const auto& r_elem = bik.data()[j];
            plan.accumulate(cell, l_elem, r_elem);
          }
        }
      }

      auto shape = apply_inverse(pc, C.array.trange().tile(c));
      tile = tile.reshape(shape);
      if (pc) tile = tile.permute(pc);
      C_local_tiles.emplace_back(std::move(c), std::move(tile));
      return true;
    }
  }
}

}  // namespace TiledArray::detail

#endif  // TILEDARRAY_TENSOR_ARENA_EINSUM_H__INCLUDED
