/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
 *
 *  util/retile_probe.h
 *
 *  Lock-free, runtime-gated wall-clock attribution probe for the SUMMA retile
 *  path. Separates time in the GEMM kernel from time in operand permute-in,
 *  operand repack-in, result carve-out, and result permute-back. Each worker
 *  thread accumulates into its own heap-allocated RetileCounters (plain ints,
 *  NO atomics); the registry mutex is touched only at thread registration and
 *  at the exit dump. Enabled by TA_RETILE_PROBE (any non-empty value != "0").
 */

#ifndef TILEDARRAY_UTIL_RETILE_PROBE_H__INCLUDED
#define TILEDARRAY_UTIL_RETILE_PROBE_H__INCLUDED

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string_view>
#include <vector>

namespace TiledArray::detail {

enum class RetileBucket : std::size_t {
  Gemm = 0,     ///< the contraction kernel (ContractReduce / BatchedContractReduce)
  PermuteIn,    ///< operand canonicalization permute (Tensor::permute)
  RepackIn,     ///< operand gather/pack/carve-pack into coarse tiles
  CarveOut,     ///< result carve/merge back into the user's (U) tiling
  PermuteBack,  ///< result re-permute to the einsum output index order
  COUNT
};

inline const char* retile_bucket_name(RetileBucket b) {
  switch (b) {
    case RetileBucket::Gemm: return "gemm";
    case RetileBucket::PermuteIn: return "permute_in";
    case RetileBucket::RepackIn: return "repack_in";
    case RetileBucket::CarveOut: return "carve_out";
    case RetileBucket::PermuteBack: return "permute_back";
    default: return "?";
  }
}

struct RetileCounters {
  std::array<std::uint64_t, static_cast<std::size_t>(RetileBucket::COUNT)> ns{};
  std::array<std::uint64_t, static_cast<std::size_t>(RetileBucket::COUNT)>
      calls{};
  void merge(const RetileCounters& o) {
    for (std::size_t i = 0; i < ns.size(); ++i) {
      ns[i] += o.ns[i];
      calls[i] += o.calls[i];
    }
  }
};

/// Test override: -1 => consult env; 0 => force off; 1 => force on.
inline int& retile_probe_override() {
  static int v = -1;
  return v;
}
inline void set_retile_probe_enabled_for_testing(bool on) {
  retile_probe_override() = on ? 1 : 0;
}
inline void clear_retile_probe_testing_override() {
  retile_probe_override() = -1;
}

/// Runtime gate. Test override wins; otherwise the cached TA_RETILE_PROBE read.
inline bool retile_probe_enabled() {
  const int ov = retile_probe_override();
  if (ov >= 0) return ov != 0;
  static const bool env = [] {
    const char* e = std::getenv("TA_RETILE_PROBE");
    return e != nullptr && e[0] != '\0' && std::string_view(e) != "0";
  }();
  return env;
}

/// A mutex + the list of per-thread counter blocks, merged at snapshot/exit.
struct RetileRegistry {
  std::mutex mtx;
  std::vector<RetileCounters*> threads;
  static RetileRegistry& instance() {
    static RetileRegistry r;
    return r;
  }
};

/// This thread's counter block. Heap-allocated and intentionally leaked so the
/// pointer survives MADNESS worker-thread teardown (matches arena_einsum.h's
/// tls_shapes). Registration takes the mutex exactly once per thread.
inline RetileCounters& tls_retile() {
  thread_local RetileCounters* mine = [] {
    auto* c = new RetileCounters();
    auto& reg = RetileRegistry::instance();
    std::lock_guard<std::mutex> lk(reg.mtx);
    reg.threads.push_back(c);
    return c;
  }();
  return *mine;
}

/// Reentrancy depth: >0 while a PermuteBack op is on this thread's stack, so the
/// nested Tensor::permute does not also charge PermuteIn (the time belongs to
/// PermuteBack). Plain thread-local int — no atomics.
inline int& tls_permute_back_depth() {
  thread_local int d = 0;
  return d;
}

/// RAII region timer. on iff retile_probe_enabled() && extra_gate. The Gemm and
/// PermuteBack seams pass extra_gate=true (env only); RepackIn/CarveOut pass
/// plan_.active; PermuteIn passes (tls_permute_back_depth()==0).
struct RetileTimer {
  RetileBucket b_;
  bool on_;
  std::chrono::steady_clock::time_point t0_;
  explicit RetileTimer(RetileBucket b, bool extra_gate = true)
      : b_(b), on_(retile_probe_enabled() && extra_gate) {
    if (on_) t0_ = std::chrono::steady_clock::now();
  }
  ~RetileTimer() {
    if (!on_) return;
    const auto dt = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0_)
            .count());
    auto& c = tls_retile();
    c.ns[static_cast<std::size_t>(b_)] += dt;
    c.calls[static_cast<std::size_t>(b_)] += 1;
  }
  RetileTimer(const RetileTimer&) = delete;
  RetileTimer& operator=(const RetileTimer&) = delete;
};

/// RAII for a result permute-back op: times the whole op into PermuteBack AND
/// raises the reentrancy depth so a nested Tensor::permute stays off PermuteIn.
struct PermuteBackScope {
  bool on_;
  std::chrono::steady_clock::time_point t0_;
  PermuteBackScope() : on_(retile_probe_enabled()) {
    if (on_) {
      ++tls_permute_back_depth();
      t0_ = std::chrono::steady_clock::now();
    }
  }
  ~PermuteBackScope() {
    if (!on_) return;
    const auto dt = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0_)
            .count());
    --tls_permute_back_depth();
    auto& c = tls_retile();
    c.ns[static_cast<std::size_t>(RetileBucket::PermuteBack)] += dt;
    c.calls[static_cast<std::size_t>(RetileBucket::PermuteBack)] += 1;
  }
  PermuteBackScope(const PermuteBackScope&) = delete;
  PermuteBackScope& operator=(const PermuteBackScope&) = delete;
};

/// Merge of all registered per-thread blocks. Call only at a quiescent point
/// (after a fence) — it reads the per-thread ints non-atomically.
inline RetileCounters retile_probe_snapshot() {
  auto& reg = RetileRegistry::instance();
  std::lock_guard<std::mutex> lk(reg.mtx);
  RetileCounters total;
  for (auto* c : reg.threads) total.merge(*c);
  return total;
}

/// Zero all registered per-thread blocks. Test-only; call between fences.
inline void retile_probe_reset_for_testing() {
  auto& reg = RetileRegistry::instance();
  std::lock_guard<std::mutex> lk(reg.mtx);
  for (auto* c : reg.threads) *c = RetileCounters{};
}

/// Prints the merged buckets to stderr at process exit when enabled. inline =>
/// single instance across TUs; <iostream> guarantees std::cerr outlives it.
struct RetileProbeDumper {
  ~RetileProbeDumper() {
    if (!retile_probe_enabled()) return;
    const auto t = retile_probe_snapshot();
    bool any = false;
    for (auto v : t.calls)
      if (v) any = true;
    if (!any) return;
    auto s = [](std::uint64_t n) { return n / 1e9; };
    std::ostream& os = std::cerr;
    os << "\n============ TA retile probe (TA_RETILE_PROBE) ============\n";
    for (std::size_t i = 0; i < t.ns.size(); ++i) {
      os << "  " << std::setw(12) << std::left
         << retile_bucket_name(static_cast<RetileBucket>(i)) << "  "
         << std::setw(10) << std::right << std::fixed << std::setprecision(6)
         << s(t.ns[i]) << " s   x" << t.calls[i] << "\n";
    }
    os << "==========================================================\n";
    os.flush();
  }
};
inline RetileProbeDumper g_retile_probe_dumper;

}  // namespace TiledArray::detail

#endif  // TILEDARRAY_UTIL_RETILE_PROBE_H__INCLUDED
