// ce_ce_segmented_strided_bench.cpp
// ---------------------------------------------------------------------------
// Tile/BLAS-level benchmark for the hce+ce per-k SEGMENTED strided DGEMM
// (arena_strided_dgemm_ce_ce_right). It times the SAME kernel on the SAME
// hole-containing arena operands under the two states of the runtime kill
// switch TiledArray::detail::ce_ce_strided_disabled():
//
//   segmented (disabled=false): per k, walk μ̃ and emit one strided GEMM per
//             maximal contiguous present+uniform-stride segment; skip holes.
//   per-cell  (disabled=true) : the legacy path -- one length-Q GEMV per
//             present (μ̃) cell (what TA did before, reverting to per-cell
//             whenever results/operands contained holes).
//
// The ONLY variable between the two timings is that toggle, so the ratio
// isolates the kernel strategy swap. Operands model the measured CSV-CCk
// fallback regime: present cells are CLUSTERED (mean segment length ~ --cluster)
// and per-k MISALIGNED (each k shifts its hole phase), the pattern the old
// all-or-nothing gate fell back to scalar on. The right-kernel walker is
// identical to the left's, so this speedup represents hce+ce overall.
// ---------------------------------------------------------------------------

#include <tiledarray.h>
#include <TiledArray/math/blas.h>
#include <TiledArray/tensor/arena_einsum.h>
#include <TiledArray/tensor/arena_tensor.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace TA = TiledArray;
namespace tablas = TA::math::blas;

using Inner = TA::ArenaTensor<double, TA::Range>;
using Outer = TA::Tensor<Inner>;

using clock_type = std::chrono::steady_clock;
static double ms_since(clock_type::time_point t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() -
                                                              t0)
             .count() /
         1.0e6;
}

struct Cli {
  int reps = 30;       // timed reps per path
  int warmup = 5;      // untimed warmup reps per path
  long Mmu = 256;      // strided axis (right external) -> BLAS M
  long nK = 8;         // outer-contraction slabs (looped, beta=1)
  long P = 16;         // result inner free (a1)
  long Q = 16;         // contraction inner (a4)
  long cluster = 6;    // mean present-run length (~ mean segment M)
  double c_fraction = 0.0;  // fraction of otherwise-present C cells to drop to holes (sparse result)
};

static void usage() {
  std::fprintf(stderr,
               "ce_ce_segmented_strided_bench\n"
               "  --reps R       timed reps per path     (default 30)\n"
               "  --warmup W     untimed warmup reps      (default 5)\n"
               "  --Mmu N        strided axis extent      (default 256)\n"
               "  --nK N         outer-contraction slabs  (default 8)\n"
               "  --P N          result inner free a1     (default 16)\n"
               "  --Q N          contraction inner a4     (default 16)\n"
               "  --cluster N    mean present-run length  (default 6)\n"
               "  --c_fraction F fraction of C cells dropped to holes (default 0.0)\n");
}

static Cli parse_cli(int argc, char** argv) {
  Cli c;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&]() -> std::string {
      if (i + 1 >= argc) { usage(); std::exit(1); }
      return argv[++i];
    };
    if (a == "--reps") c.reps = std::stoi(need());
    else if (a == "--warmup") c.warmup = std::stoi(need());
    else if (a == "--Mmu") c.Mmu = std::stol(need());
    else if (a == "--nK") c.nK = std::stol(need());
    else if (a == "--P") c.P = std::stol(need());
    else if (a == "--Q") c.Q = std::stol(need());
    else if (a == "--cluster") c.cluster = std::stol(need());
    else if (a == "--c_fraction") c.c_fraction = std::stod(need());
    else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
    else { std::fprintf(stderr, "unknown flag: %s\n", a.c_str()); usage();
           std::exit(1); }
  }
  return c;
}

// Build an arena Outer with holes: dense_shape(o) unless is_hole(o) -> Range{}.
static Outer make_sparse(const TA::Range& outer_range, std::size_t nbatch,
                         const std::function<TA::Range(std::size_t)>& dense_shape,
                         const std::function<bool(std::size_t)>& is_hole,
                         double base) {
  Outer t = TA::detail::arena_outer_init<Outer>(
      outer_range, nbatch,
      [&](std::size_t o) { return is_hole(o) ? TA::Range{} : dense_shape(o); });
  for (std::size_t o = 0; o < t.range().volume() * nbatch; ++o) {
    Inner& c = t.data()[o];
    if (!c) continue;
    for (std::size_t e = 0; e < c.size(); ++e) c.data()[e] = base + 0.001 * o + e;
  }
  return t;
}

int main(int argc, char** argv) {
  Cli cli = parse_cli(argc, argv);
  if (cli.reps < 1) { std::fprintf(stderr, "--reps must be >= 1\n"); return 1; }
  auto& world = TA_SCOPED_INITIALIZE(argc, argv);
  (void)world;

  const std::size_t Mo = 1;
  const std::size_t Mmu = static_cast<std::size_t>(cli.Mmu);
  const std::size_t nK = static_cast<std::size_t>(cli.nK);
  const long P = cli.P, Q = cli.Q;
  const long cl = std::max<long>(1, cli.cluster);

  // Clustered + per-k-misaligned presence on R[mu,k] (canonical mu slow, k fast,
  // ordinal o = mu*nK + k). A cell is a hole when, within its k-shifted phase,
  // it falls in the 1-wide gap after each run of length `cl`. period = cl + 1.
  auto rhole = [&](std::size_t o) {
    const std::size_t mu = o / nK, k = o % nK;
    const long period = cl + 1;
    const long phase = (static_cast<long>(mu) + static_cast<long>(k) * 2) % period;
    return phase == cl;  // the single gap cell each period
  };
  const double c_frac = std::max(0.0, std::min(1.0, cli.c_fraction));
  // C[mu] present iff present for at least one k (the union the kernel writes),
  // optionally thinned: a deterministic fraction c_frac of otherwise-present
  // cells are dropped to holes to model a genuinely SPARSE result. A hole C cell
  // is absent regardless of operand presence (its (k,mu) contributions skip).
  auto chole = [&](std::size_t o) {
    bool union_present = false;
    for (std::size_t k = 0; k < nK; ++k)
      if (!rhole(o * nK + k)) { union_present = true; break; }
    if (!union_present) return true;            // absent for all k -> hole
    if (c_frac > 0.0) {
      const std::size_t h = (o * 2654435761ull) & 0xffffull;  // cheap hash
      if (static_cast<double>(h) / 65536.0 < c_frac) return true;
    }
    return false;
  };

  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK}, 1, [&](std::size_t) {
        return TA::Range{static_cast<std::size_t>(P),
                         static_cast<std::size_t>(Q)};
      });
  for (std::size_t o = 0; o < L.range().volume(); ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.001 * o + e;
  Outer R = make_sparse(TA::Range{Mmu, nK}, 1,
                        [&](std::size_t){ return TA::Range{static_cast<std::size_t>(Q)}; },
                        rhole, 2.0);
  Outer Ctemplate = make_sparse(TA::Range{Mmu}, 1,
                        [&](std::size_t){ return TA::Range{static_cast<std::size_t>(P)}; },
                        chole, 0.0);

  std::size_t present_C = 0;
  for (std::size_t o = 0; o < Ctemplate.range().volume(); ++o)
    if (Ctemplate.data()[o]) ++present_C;
  std::size_t present_R = 0;
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    if (R.data()[o]) ++present_R;

  std::printf("=== hce+ce segmented-vs-per-cell strided DGEMM bench ===\n");
  std::printf("Mmu=%zu nK=%zu P=%ld Q=%ld cluster=%ld  "
              "present C=%zu/%zu  present R=%zu/%zu\n",
              Mmu, nK, P, Q, cl, present_C, Ctemplate.range().volume(),
              present_R, R.range().volume());
  std::printf("reps=%d warmup=%d\n", cli.reps, cli.warmup);

  // FLOP estimate: 2*P*Q per present (mu,k) contributing cell.
  double flop = 0.0;
  for (std::size_t mu = 0; mu < Mmu; ++mu)
    for (std::size_t k = 0; k < nK; ++k)
      if (R.data()[mu * nK + k] && Ctemplate.data()[mu])
        flop += 2.0 * P * Q;

  auto zero_C = [&](Outer& C) {
    for (std::size_t o = 0; o < C.range().volume(); ++o) {
      Inner& c = C.data()[o];
      if (!c) continue;
      for (std::size_t e = 0; e < c.size(); ++e) c.data()[e] = 0.0;
    }
  };
  auto make_C = [&]() {
    return make_sparse(TA::Range{Mmu}, 1,
        [&](std::size_t){ return TA::Range{static_cast<std::size_t>(P)}; },
        chole, 0.0);
  };
  auto median = [](std::vector<double> v) -> double {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if (n == 0) return 0.0;
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  };
  // Time ONLY the kernel call. C is allocated once per path and re-zeroed
  // OUTSIDE the timed window each rep (beta=1 needs a zero start), so the
  // measured time isolates the segment-walker vs per-cell strategy, not the
  // per-rep tile allocation (which is identical on both sides and would
  // otherwise dominate this ~0.1 ms kernel and compress the ratio).
  auto time_path = [&](bool disabled) {
    Outer C = make_C();
    TA::detail::ce_ce_strided_disabled() = disabled;
    for (int w = 0; w < cli.warmup; ++w) { zero_C(C);
      TA::detail::arena_strided_dgemm_ce_ce_right(
          C, L, R, Mo, Mmu, nK, tablas::NoTranspose, tablas::Transpose, 1.0); }
    std::vector<double> ms;
    ms.reserve(cli.reps);
    for (int r = 0; r < cli.reps; ++r) {
      zero_C(C);  // untimed
      auto t0 = clock_type::now();
      TA::detail::arena_strided_dgemm_ce_ce_right(
          C, L, R, Mo, Mmu, nK, tablas::NoTranspose, tablas::Transpose, 1.0);
      ms.push_back(ms_since(t0));
    }
    return ms;
  };

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
#endif
  auto seg_ms = time_path(/*disabled=*/false);
#ifdef TA_STRIDED_DGEMM_COUNT
  const std::size_t seg_calls =
      TA::detail::g_strided_dgemm_ce_ce_right_calls.load();
#endif
  auto pc_ms = time_path(/*disabled=*/true);
  TA::detail::ce_ce_strided_disabled() = false;  // restore production default

  const double seg_min = *std::min_element(seg_ms.begin(), seg_ms.end());
  const double seg_med = median(seg_ms);
  const double pc_min = *std::min_element(pc_ms.begin(), pc_ms.end());
  const double pc_med = median(pc_ms);

  std::printf("\n--- results (per kernel call, ms) ---\n");
  std::printf("per-cell  : min=%9.5f ms  median=%9.5f ms  (%.2f GFLOP/s)\n",
              pc_min, pc_med, flop / (pc_min * 1e6));
  std::printf("segmented : min=%9.5f ms  median=%9.5f ms  (%.2f GFLOP/s)\n",
              seg_min, seg_med, flop / (seg_min * 1e6));
  std::printf("speedup   : min=%6.3fx  median=%6.3fx  (per-cell / segmented)\n",
              seg_min > 0 ? pc_min / seg_min : 0.0,
              seg_med > 0 ? pc_med / seg_med : 0.0);

#ifdef TA_STRIDED_DGEMM_COUNT
  std::printf("\n--- firing witness (TA_STRIDED_DGEMM_COUNT) ---\n");
  std::printf("segment GEMMs over %d+%d (warmup+timed) reps = %zu  "
              "(mean %.1f per rep)\n",
              cli.warmup, cli.reps, seg_calls,
              double(seg_calls) / double(cli.warmup + cli.reps));
  if (seg_calls == 0) {
    std::fprintf(stderr, "ERROR: segmented path never issued a GEMM -- the "
                 "reported speedup would reflect a silent fallback.\n");
    std::abort();
  }
  std::printf("OK: segmented path fired (counter > 0).\n");
#endif
  return 0;
}
