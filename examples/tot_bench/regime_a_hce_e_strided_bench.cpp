// regime_a_hce_e_strided_bench.cpp
// ---------------------------------------------------------------------------
// End-to-end einsum-DSL benchmark for the regime-A `hc+e` ToT product on
// ArenaTensor inner cells. Unlike the sibling tile/BLAS-level benches
// (op[ab]_strided_arena_dgemm.cpp, which call BLAS directly off an op_dump),
// this driver times the SAME `einsum(...)` over the SAME arena operands under
// the two states of the runtime kill switch
// `TiledArray::detail::regime_a_strided_disabled()`:
//
//   strided   (disabled=false): the ce+e core fuses the outer-contraction k
//             into ONE strided DGEMM per result cell (M=N=1, K=tile-volume).
//   per-cell  (disabled=true) : the legacy per-cell rank-1 `dger` loop.
//
// The ONLY variable between the two timings is that toggle, so the measured
// ratio isolates the kernel swap (rank-1 dger loop -> one strided DGEMM). It
// is NOT an arena-vs-owning comparison: both paths read the identical arena
// data and pay the identical einsum driver / scheduling overhead (which
// compresses the ratio -- that is honest and expected).
//
// The einsum:
//   c("h,i; a1,a2") = a("h,i,k; a1") * b("h,i,k; a2")
//     h,i = Hadamard outer (both operands + result)
//     k   = outer-contracted (both operands, NOT in result)
//     a1 (left-only) x a2 (right-only) = inner OUTER-PRODUCT
//
// Problem is sized to mimic C6H14: large outer-contraction k (multi-tile,
// a few hundred cells), moderate inner a1/a2, small Hadamard folding to
// nbatch >= 2.
// ---------------------------------------------------------------------------

#include <tiledarray.h>
#include <TiledArray/math/blas.h>

#include <TiledArray/expressions/einsum.h>
#include <TiledArray/tensor/arena_einsum.h>
#include <TiledArray/tensor/arena_tensor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace TA = TiledArray;

using clock_type = std::chrono::steady_clock;
static double ms_since(clock_type::time_point t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() -
                                                              t0)
             .count() /
         1.0e6;
}

// ===========================================================================
// CLI
// ===========================================================================

struct Cli {
  int reps = 20;        // timed reps per path
  int warmup = 3;       // untimed warmup reps per path
  int k_tiles = 8;      // number of k tiles
  int k_tile = 32;      // extent of each k tile
  int inner = 24;       // |a1| = |a2|
};

static void usage() {
  std::fprintf(stderr,
               "regime_a_hce_e_strided_bench\n"
               "  --reps R       timed reps per path        (default 20)\n"
               "  --warmup W      untimed warmup reps        (default 3)\n"
               "  --k_tiles N    number of k tiles          (default 8)\n"
               "  --k_tile E     extent of each k tile       (default 32)\n"
               "  --inner P      |a1| = |a2|                 (default 24)\n");
}

static Cli parse_cli(int argc, char** argv) {
  Cli c;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--reps")
      c.reps = std::stoi(need());
    else if (a == "--warmup")
      c.warmup = std::stoi(need());
    else if (a == "--k_tiles")
      c.k_tiles = std::stoi(need());
    else if (a == "--k_tile")
      c.k_tile = std::stoi(need());
    else if (a == "--inner")
      c.inner = std::stoi(need());
    else if (a == "-h" || a == "--help") {
      usage();
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown flag: %s\n", a.c_str());
      usage();
      std::exit(1);
    }
  }
  return c;
}

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char** argv) {
  Cli cli = parse_cli(argc, argv);
  auto& world = TA_SCOPED_INITIALIZE(argc, argv);

  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;

  const long H = 2, I = 2;             // Hadamard outer (fold to nbatch = H*I = 4)
  const long P = cli.inner;            // |a1|
  const long Q = cli.inner;            // |a2|
  const long nbatch = H * I;
  const long Kcells = static_cast<long>(cli.k_tiles) * cli.k_tile;

  // k as an explicit multi-boundary TiledRange1: k_tiles tiles of k_tile each.
  std::vector<long> kbounds;
  kbounds.reserve(cli.k_tiles + 1);
  for (int t = 0; t <= cli.k_tiles; ++t)
    kbounds.push_back(static_cast<long>(t) * cli.k_tile);
  TA::TiledRange1 ktr1(kbounds.begin(), kbounds.end());

  // outer (h,i,k): h one tile extent H, i one tile extent I, k multi-tile.
  TA::TiledRange a_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I}, ktr1};
  TA::TiledRange b_trange{TA::TiledRange1{0l, H}, TA::TiledRange1{0l, I}, ktr1};

  // Smooth deterministic fill (a function of global coordinates).
  auto a_val = [](long h, long i, long k, long a1) {
    return 1.0 + 0.5 * i + 0.25 * std::sin(0.13 * k) + 0.125 * a1 + 0.0625 * h;
  };
  auto b_val = [](long h, long i, long k, long a2) {
    return 2.0 - 0.3 * std::cos(0.07 * k) + 0.2 * i + 0.05 * a2 + 0.03 * h;
  };

  // ---- Construct a, b once (arena ToT) ----
  ArenaArr a(world, a_trange);
  a.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [&](std::size_t /*ord*/) { return TA::Range{P}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a1 = 0; a1 < P; ++a1) c.data()[a1] = a_val(h, i, k, a1);
    }
    return t;
  });
  ArenaArr b(world, b_trange);
  b.init_tiles([&](const TA::Range& tr) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        tr, 1, [&](std::size_t /*ord*/) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto idx = tr.idx(o);
      const long h = static_cast<long>(idx[0]);
      const long i = static_cast<long>(idx[1]);
      const long k = static_cast<long>(idx[2]);
      for (long a2 = 0; a2 < Q; ++a2) c.data()[a2] = b_val(h, i, k, a2);
    }
    return t;
  });
  world.gop.fence();

  std::printf("=== regime-A hc+e einsum strided-vs-per-cell bench ===\n");
  std::printf(
      "shape: h=%ld(x1 tile) i=%ld(x1 tile) k=%ld (%d tiles x %d)  "
      "a1=%ld a2=%ld  nbatch=%ld\n",
      H, I, Kcells, cli.k_tiles, cli.k_tile, P, Q, nbatch);
  std::printf("reps=%d warmup=%d\n", cli.reps, cli.warmup);

  // ---- Warmup both paths (untimed: JIT / page-fault / threadpool warmup) ----
  for (int w = 0; w < cli.warmup; ++w) {
    TA::detail::regime_a_strided_disabled() = false;
    {
      auto c = einsum(a("h,i,k;a1"), b("h,i,k;a2"), "h,i;a1,a2");
      c.world().gop.fence();
    }
    TA::detail::regime_a_strided_disabled() = true;
    {
      auto c = einsum(a("h,i,k;a1"), b("h,i,k;a2"), "h,i;a1,a2");
      c.world().gop.fence();
    }
  }
  TA::detail::regime_a_strided_disabled() = false;
  world.gop.fence();

  auto median = [](std::vector<double> v) -> double {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if (n == 0) return 0.0;
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  };

  // ---- Timed: strided path (disabled = false) ----
  TA::detail::regime_a_strided_disabled() = false;
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_e_calls.store(0);
#endif
  std::vector<double> strided_ms;
  strided_ms.reserve(cli.reps);
  for (int r = 0; r < cli.reps; ++r) {
    auto t0 = clock_type::now();
    auto c = einsum(a("h,i,k;a1"), b("h,i,k;a2"), "h,i;a1,a2");
    c.world().gop.fence();
    strided_ms.push_back(ms_since(t0));
  }
#ifdef TA_STRIDED_DGEMM_COUNT
  const std::size_t strided_calls =
      TA::detail::g_strided_dgemm_ce_e_calls.load();
#endif
  const double t_strided_min =
      *std::min_element(strided_ms.begin(), strided_ms.end());
  const double t_strided_med = median(strided_ms);

  // ---- Timed: per-cell path (disabled = true) ----
  TA::detail::regime_a_strided_disabled() = true;
  std::vector<double> percell_ms;
  percell_ms.reserve(cli.reps);
  for (int r = 0; r < cli.reps; ++r) {
    auto t0 = clock_type::now();
    auto c = einsum(a("h,i,k;a1"), b("h,i,k;a2"), "h,i;a1,a2");
    c.world().gop.fence();
    percell_ms.push_back(ms_since(t0));
  }
  const double t_percell_min =
      *std::min_element(percell_ms.begin(), percell_ms.end());
  const double t_percell_med = median(percell_ms);

  // ---- Restore production default ----
  TA::detail::regime_a_strided_disabled() = false;

  // ---- Report ----
  std::printf("\n--- results (per einsum call, ms) ---\n");
  std::printf("t_percell : min=%8.4f ms  median=%8.4f ms\n", t_percell_min,
              t_percell_med);
  std::printf("t_strided : min=%8.4f ms  median=%8.4f ms\n", t_strided_min,
              t_strided_med);
  const double speedup_min =
      t_strided_min > 0.0 ? t_percell_min / t_strided_min : 0.0;
  const double speedup_med =
      t_strided_med > 0.0 ? t_percell_med / t_strided_med : 0.0;
  std::printf("speedup   : min=%6.3fx  median=%6.3fx  (t_percell / t_strided)\n",
              speedup_min, speedup_med);

#ifdef TA_STRIDED_DGEMM_COUNT
  std::printf("\n--- firing witness (TA_STRIDED_DGEMM_COUNT) ---\n");
  std::printf("g_strided_dgemm_ce_e_calls = %zu  (over %d strided reps)\n",
              strided_calls, cli.reps);
  if (strided_calls == 0) {
    std::fprintf(stderr,
                 "ERROR: strided DGEMM never fired -- reported numbers would "
                 "reflect a silent fallback, not the strided path.\n");
    std::abort();
  }
  std::printf("OK: strided DGEMM fired (counter > 0).\n");
#endif

  return 0;
}
