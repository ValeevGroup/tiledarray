/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
 *
 *  einsum/einsum_instrument.h
 *
 *  Lightweight, runtime-gated attribution profiler for the generalized
 *  batched-contraction einsum (Hadamard indices coexisting with
 *  external/contracted indices, including tensor-of-tensor operands).
 *
 *  Goal: separate time spent in the *machinery* of the per-Hadamard-tile
 *  sub-World scheme (MPI_Comm_split, sub-World construction/teardown,
 *  make_array/retile, harvest, entry fence) from time spent in the actual
 *  numeric contraction. Enabled by setting TA_EINSUM_INSTRUMENT to any
 *  non-empty value other than "0"; zero (modulo a cached bool load) overhead
 *  when off. Results are dumped to std::cerr at static teardown.
 */

#ifndef TILEDARRAY_EINSUM_EINSUM_INSTRUMENT_H__INCLUDED
#define TILEDARRAY_EINSUM_EINSUM_INSTRUMENT_H__INCLUDED

#include "TiledArray/util/time.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace TiledArray::detail {

/// Runtime gate for the einsum attribution profiler. Toggled from the
/// environment via TA_EINSUM_INSTRUMENT (any non-empty value other than "0"
/// enables). Mirrors einsum_hadamard_local_fastpath_disabled().
inline bool einsum_instrument_enabled() {
  static const bool flag = [] {
    const char *e = std::getenv("TA_EINSUM_INSTRUMENT");
    return e != nullptr && e[0] != '\0' && std::string_view(e) != "0";
  }();
  return flag;
}

/// Time buckets attributed per einsum call. NUMERICS is the only bucket that
/// is genuine flops; the rest is per-Hadamard-tile / per-call machinery.
enum class EinsumBucket : std::size_t {
  EntryFence = 0,  ///< the blocking world.gop.fence() at function entry
  Setup,           ///< index algebra, range maps, trange, inner-op build
  CommSplitWorld,  ///< MPI_Comm_split + sub-World construction
  Retile,          ///< make_array of the per-slice input sub-arrays
  ContractFence,   ///< expr-assign enqueue + per-slice sub-World fence
                   ///< (numerics for the sub-World path live here)
  Harvest,         ///< extracting completed result tiles from the sub-array
  LocalKernel,     ///< direct local contraction (arena / element-op gemm)
  Teardown,        ///< build_C_array + final sub-World fences
  COUNT
};

inline const char *einsum_bucket_name(EinsumBucket b) {
  switch (b) {
    case EinsumBucket::EntryFence:
      return "entry_fence";
    case EinsumBucket::Setup:
      return "setup";
    case EinsumBucket::CommSplitWorld:
      return "commsplit+world";
    case EinsumBucket::Retile:
      return "retile/make_array";
    case EinsumBucket::ContractFence:
      return "contract+fence";
    case EinsumBucket::Harvest:
      return "harvest";
    case EinsumBucket::LocalKernel:
      return "local_kernel";
    case EinsumBucket::Teardown:
      return "teardown";
    default:
      return "?";
  }
}

struct EinsumProfileEntry {
  std::uint64_t calls = 0;
  std::uint64_t slices = 0;       ///< total Hadamard slices iterated (owned)
  std::uint64_t subworlds = 0;    ///< total sub-Worlds constructed
  std::uint64_t localslices = 0;  ///< slices handled by a local kernel
  std::array<std::int64_t, static_cast<std::size_t>(EinsumBucket::COUNT)> ns{};

  void merge(const EinsumProfileEntry &o) {
    calls += o.calls;
    slices += o.slices;
    subworlds += o.subworlds;
    localslices += o.localslices;
    for (std::size_t k = 0; k < ns.size(); ++k) ns[k] += o.ns[k];
  }
  std::int64_t total_ns() const {
    std::int64_t t = 0;
    for (auto v : ns) t += v;
    return t;
  }
};

/// Process-wide accumulator, keyed by "<branch> | <shape annotation>". Dumps a
/// sorted attribution table to std::cerr at static teardown when enabled.
class EinsumProfiler {
 public:
  static EinsumProfiler &instance() {
    static EinsumProfiler p;
    return p;
  }

  void merge(const std::string &key, const EinsumProfileEntry &e) {
    std::lock_guard<std::mutex> g(mtx_);
    by_key_[key].merge(e);
  }

  ~EinsumProfiler() {
    if (einsum_instrument_enabled()) dump(std::cerr);
  }

  void dump(std::ostream &os) {
    std::lock_guard<std::mutex> g(mtx_);
    if (by_key_.empty()) return;
    // collect and sort by total time descending
    std::vector<std::pair<std::string, EinsumProfileEntry>> rows(
        by_key_.begin(), by_key_.end());
    std::sort(rows.begin(), rows.end(), [](auto const &a, auto const &b) {
      return a.second.total_ns() > b.second.total_ns();
    });
    std::int64_t grand = 0;
    EinsumProfileEntry tot;
    for (auto const &[k, e] : rows) {
      grand += e.total_ns();
      tot.merge(e);
    }
    auto s = [](std::int64_t ns) { return ns / 1e9; };
    os << "\n================ TA einsum attribution (TA_EINSUM_INSTRUMENT) "
          "================\n";
    os << "total einsum-region time: " << s(grand) << " s over " << tot.calls
       << " calls, " << tot.slices << " slices, " << tot.subworlds
       << " sub-Worlds, " << tot.localslices << " local slices\n";
    // aggregate bucket breakdown
    os << "-- aggregate by bucket --\n";
    for (std::size_t k = 0; k < tot.ns.size(); ++k) {
      if (tot.ns[k] == 0) continue;
      os << "  " << std::setw(18) << std::left
         << einsum_bucket_name(static_cast<EinsumBucket>(k)) << "  "
         << std::setw(9) << std::right << s(tot.ns[k]) << " s  ("
         << std::setw(5) << std::fixed << std::setprecision(1)
         << (grand ? 100.0 * tot.ns[k] / grand : 0.0) << "%)\n";
    }
    // per-key rows
    os << "-- by contraction (branch | shape), sorted by total time --\n";
    for (auto const &[k, e] : rows) {
      std::int64_t t = e.total_ns();
      os << "  " << s(t) << " s  x" << e.calls << "  slices=" << e.slices
         << " subW=" << e.subworlds << " local=" << e.localslices << "\n      "
         << k << "\n      ";
      for (std::size_t b = 0; b < e.ns.size(); ++b) {
        if (e.ns[b] == 0) continue;
        os << einsum_bucket_name(static_cast<EinsumBucket>(b)) << "="
           << std::fixed << std::setprecision(1)
           << (t ? 100.0 * e.ns[b] / t : 0.0) << "% ";
      }
      os << "\n";
    }
    os << "============================================================"
          "====================\n";
    os.flush();
  }

 private:
  std::mutex mtx_;
  std::map<std::string, EinsumProfileEntry> by_key_;
};

/// Per-call accumulator. Construct once near the top of an einsum call; its
/// destructor merges into the process-wide profiler. No-op when disabled.
struct EinsumCall {
  bool active;
  std::string label;         ///< shape annotation (a;A * b;B -> c;C)
  const char *branch = "?";  ///< which einsum branch handled this call
  EinsumProfileEntry e;

  explicit EinsumCall(std::string lbl)
      : active(einsum_instrument_enabled()), label(std::move(lbl)) {
    if (active) e.calls = 1;
  }
  ~EinsumCall() {
    if (active)
      EinsumProfiler::instance().merge(std::string(branch) + " | " + label, e);
  }
  void add(EinsumBucket b, std::int64_t ns) {
    if (active) e.ns[static_cast<std::size_t>(b)] += ns;
  }
  void add_slices(std::uint64_t n) {
    if (active) e.slices += n;
  }
  void add_subworld() {
    if (active) ++e.subworlds;
  }
  void add_localslice() {
    if (active) ++e.localslices;
  }
};

/// RAII region timer; adds elapsed wall time to a bucket of an EinsumCall.
struct EinsumTimer {
  EinsumCall *call;
  EinsumBucket bucket;
  bool on;
  time_point t0;
  EinsumTimer(EinsumCall &c, EinsumBucket b)
      : call(&c), bucket(b), on(c.active), t0(on ? now() : time_point{}) {}
  ~EinsumTimer() {
    if (on) call->add(bucket, duration_in_ns(t0, now()));
  }
};

}  // namespace TiledArray::detail

#endif  // TILEDARRAY_EINSUM_EINSUM_INSTRUMENT_H__INCLUDED
