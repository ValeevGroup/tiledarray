// batched_contraction_attribution.cpp
// ---------------------------------------------------------------------------
// Attribution benchmark for the *plain* (non-ToT) batched contraction
//
//     C(b,i,k) = A(b,i,j) * B(b,j,k)
//
// where `b` is a Hadamard ("batch") index (present in A, B, and C), `i`/`k`
// are external indices, and `j` is contracted. This is the case that today
// can only be expressed via TA::einsum, and whose performance the batch index
// being "hidden" inside the TA::Tensor tile (folded into Tensor::nbatch) is
// suspected to hurt.
//
// The benchmark separates two cost centers:
//
//   RAW    : the irreducible work. For every one of the B = bt*be batch
//            elements we issue exactly ONE BLAS dgemm (I x J)*(J x K) on the
//            already-local tile data -- i.e. precisely the per-batch-element
//            GEMM granularity that einsum's Tensor::gemm() ultimately runs,
//            but with ZERO TiledArray driver overhead. RAW therefore already
//            includes the "many small GEMMs, no batched-BLAS" inefficiency
//            (the P1 headroom).
//
//   EINSUM : the production path, einsum(A("b,i,j"), B("b,j,k"), "b,i,k").
//
//   => (EINSUM - RAW) isolates the TA/einsum *machinery*: blocking find().get()
//      of every tile, eager tile.permute(), reshape-into-nbatch, make_array of
//      per-slice sub-arrays, the per-Hadamard-tile MPI_Comm_split + sub-World,
//      and the per-slice fences (the P2/P3/P4 headroom).
//
// The granularity SWEEP holds the total batch B and the total flop count
// FIXED while moving B from "few big batch tiles" to "many tiny batch tiles".
// Flops are constant across the sweep, so any rise in EINSUM time with the
// number of batch tiles is machinery overhead that scales with the number of
// Hadamard tiles -- the signature of the per-H-tile comm-split/sub-World path.
//
// NOTE: the RAW baseline materializes data locally and is meaningful at np=1
// (single rank); it is the node-local flops reference. The EINSUM timing is
// valid at any world size -- run at np=2 to watch the per-H-tile collective
// machinery show up.
// ---------------------------------------------------------------------------

#include <TiledArray/math/blas.h>
#include <tiledarray.h>

#include <TiledArray/einsum/tiledarray.h>
#include <TiledArray/expressions/einsum.h>

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
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock_type::now() - t0)
             .count() /
         1.0e6;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  if (n == 0) return 0.0;
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ===========================================================================
// CLI
// ===========================================================================

struct Cli {
  int reps = 20;   // timed reps per path
  int warmup = 3;  // untimed warmup reps
  int bt = 16;     // number of batch (Hadamard) tiles
  int be = 4;      // extent of each batch tile  (total batch B = bt*be)
  int I = 32;      // external index i extent (one tile)
  int J = 32;      // contracted index j extent (one tile)
  int K = 32;      // external index k extent (one tile)
  int sweep = 0;   // if !=0, run the constant-B granularity sweep
};

static void usage() {
  std::fprintf(stderr,
               "batched_contraction_attribution\n"
               "  C(b,i,k) = A(b,i,j) * B(b,j,k)   (b=Hadamard, j=contracted)\n"
               "  --reps R    timed reps per path        (default 20)\n"
               "  --warmup W  untimed warmup reps         (default 3)\n"
               "  --bt N      number of batch tiles       (default 16)\n"
               "  --be E      extent per batch tile       (default 4)\n"
               "  --i / --j / --k  matrix extents         (default 32)\n"
               "  --sweep     run constant-B granularity sweep (B = bt*be held "
               "fixed)\n");
}

static Cli parse_cli(int argc, char** argv) {
  Cli c;
  for (int a = 1; a < argc; ++a) {
    std::string s = argv[a];
    auto need = [&]() -> std::string {
      if (a + 1 >= argc) {
        usage();
        std::exit(1);
      }
      return argv[++a];
    };
    if (s == "--reps")
      c.reps = std::stoi(need());
    else if (s == "--warmup")
      c.warmup = std::stoi(need());
    else if (s == "--bt")
      c.bt = std::stoi(need());
    else if (s == "--be")
      c.be = std::stoi(need());
    else if (s == "--i")
      c.I = std::stoi(need());
    else if (s == "--j")
      c.J = std::stoi(need());
    else if (s == "--k")
      c.K = std::stoi(need());
    else if (s == "--sweep")
      c.sweep = 1;
    else if (s == "-h" || s == "--help") {
      usage();
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown flag: %s\n", s.c_str());
      usage();
      std::exit(1);
    }
  }
  return c;
}

// ===========================================================================
// helpers
// ===========================================================================

using Arr = TA::DistArray<TA::Tensor<double>, TA::DensePolicy>;

static TA::TiledRange1 batch_tr1(int bt, int be) {
  std::vector<long> bounds;
  bounds.reserve(bt + 1);
  for (int t = 0; t <= bt; ++t) bounds.push_back(static_cast<long>(t) * be);
  return TA::TiledRange1(bounds.begin(), bounds.end());
}

// deterministic fills (function of global coords) so RAW and EINSUM agree
static double a_val(long b, long i, long j) {
  return 1.0 + 0.5 * std::sin(0.013 * (b + 1) * (i + 1)) + 0.1 * (j + 1);
}
static double b_val(long b, long j, long k) {
  return 2.0 - 0.3 * std::cos(0.017 * (b + 1) * (k + 1)) + 0.05 * (j + 1);
}

static Arr make_A(TA::World& world, int bt, int be, int I, int J) {
  TA::TiledRange tr{batch_tr1(bt, be), TA::TiledRange1{0l, I},
                    TA::TiledRange1{0l, J}};
  Arr a(world, tr);
  a.init_tiles([&](const TA::Range& r) {
    TA::Tensor<double> t(r);
    const auto lo = r.lobound();
    const auto ex = r.extent();
    std::size_t o = 0;
    for (long b = lo[0]; b < lo[0] + (long)ex[0]; ++b)
      for (long i = lo[1]; i < lo[1] + (long)ex[1]; ++i)
        for (long j = lo[2]; j < lo[2] + (long)ex[2]; ++j)
          t.data()[o++] = a_val(b, i, j);
    return t;
  });
  return a;
}

static Arr make_B(TA::World& world, int bt, int be, int J, int K) {
  TA::TiledRange tr{batch_tr1(bt, be), TA::TiledRange1{0l, J},
                    TA::TiledRange1{0l, K}};
  Arr b(world, tr);
  b.init_tiles([&](const TA::Range& r) {
    TA::Tensor<double> t(r);
    const auto lo = r.lobound();
    const auto ex = r.extent();
    std::size_t o = 0;
    for (long bb = lo[0]; bb < lo[0] + (long)ex[0]; ++bb)
      for (long j = lo[1]; j < lo[1] + (long)ex[1]; ++j)
        for (long k = lo[2]; k < lo[2] + (long)ex[2]; ++k)
          t.data()[o++] = b_val(bb, j, k);
    return t;
  });
  return b;
}

// RAW baseline: one BLAS dgemm per batch element on already-local tile data.
// Returns a checksum (to defeat DCE) via out-param; time is measured outside.
static double raw_once(const Arr& A, const Arr& B, int I, int J, int K,
                       std::vector<double>& cbuf) {
  namespace blas = TiledArray::math::blas;
  double checksum = 0.0;
  const auto& tr = A.trange();
  const std::size_t nbtiles = tr.dim(0).tile_extent();
  for (std::size_t bt = 0; bt < nbtiles; ++bt) {
    // batch tile index (bt, 0, 0)
    std::array<std::size_t, 3> tidx{bt, 0, 0};
    if (!A.is_local(tidx)) continue;
    auto at = A.find_local(tidx).get();
    auto bt_tile = B.find_local(tidx).get();
    const double* ap = at.data();
    const double* bp = bt_tile.data();
    const std::size_t be = at.range().extent()[0];
    for (std::size_t e = 0; e < be; ++e) {
      const double* ae = ap + e * (std::size_t)I * J;   // I x J
      const double* be_ = bp + e * (std::size_t)J * K;  // J x K
      // C(I,K) = A(I,J) * B(J,K), row-major
      blas::gemm(blas::Op::NoTrans, blas::Op::NoTrans, I, K, J, 1.0, ae, J, be_,
                 K, 0.0, cbuf.data(), K);
      checksum += cbuf[0] + cbuf[(std::size_t)I * K - 1];
    }
  }
  return checksum;
}

// peak-ish reference: throughput of a single large square dgemm, to
// contextualize how far the small per-batch GEMMs are from the machine's
// achievable rate.
static double peak_gemm_gflops(int n, int reps) {
  namespace blas = TiledArray::math::blas;
  std::vector<double> a((std::size_t)n * n, 1.0001),
      b((std::size_t)n * n, 0.9999), c((std::size_t)n * n, 0.0);
  // warmup
  blas::gemm(blas::Op::NoTrans, blas::Op::NoTrans, n, n, n, 1.0, a.data(), n,
             b.data(), n, 0.0, c.data(), n);
  std::vector<double> ms;
  for (int r = 0; r < reps; ++r) {
    auto t0 = clock_type::now();
    blas::gemm(blas::Op::NoTrans, blas::Op::NoTrans, n, n, n, 1.0, a.data(), n,
               b.data(), n, 0.0, c.data(), n);
    ms.push_back(ms_since(t0));
  }
  const double flop = 2.0 * (double)n * n * n;
  const double t = median(ms) / 1.0e3;
  return flop / t / 1.0e9;
}

struct Result {
  double legacy_med, fused_med, raw_med;
};

// Time the einsum path under a given value of the fused-Hadamard toggle.
static double time_einsum(TA::World& world, const Arr& A, const Arr& B,
                          bool fused, int reps, int warmup) {
  TA::detail::einsum_hadamard_local_fastpath_disabled() = !fused;
  for (int w = 0; w < warmup; ++w) {
    auto c = einsum(A("b,i,j"), B("b,j,k"), "b,i,k");
    c.world().gop.fence();
  }
  world.gop.fence();
  std::vector<double> ms;
  ms.reserve(reps);
  for (int r = 0; r < reps; ++r) {
    auto t0 = clock_type::now();
    auto c = einsum(A("b,i,j"), B("b,j,k"), "b,i,k");
    c.world().gop.fence();
    ms.push_back(ms_since(t0));
  }
  TA::detail::einsum_hadamard_local_fastpath_disabled() =
      false;  // restore default
  return median(ms);
}

static Result run_case(TA::World& world, const Cli& cli, int bt, int be,
                       bool quiet) {
  const int I = cli.I, J = cli.J, K = cli.K;
  Arr A = make_A(world, bt, be, I, J);
  Arr B = make_B(world, bt, be, J, K);
  world.gop.fence();

  const double flop = 2.0 * (double)bt * be * I * J * K;

  const double legacy =
      time_einsum(world, A, B, /*fused=*/false, cli.reps, cli.warmup);
  const double fused =
      time_einsum(world, A, B, /*fused=*/true, cli.reps, cli.warmup);

  // RAW timed (node-local; meaningful at np=1)
  std::vector<double> cbuf((std::size_t)I * K, 0.0);
  for (int w = 0; w < cli.warmup; ++w) raw_once(A, B, I, J, K, cbuf);
  std::vector<double> r_ms;
  r_ms.reserve(cli.reps);
  volatile double sink = 0.0;
  for (int r = 0; r < cli.reps; ++r) {
    auto t0 = clock_type::now();
    sink += raw_once(A, B, I, J, K, cbuf);
    r_ms.push_back(ms_since(t0));
  }
  (void)sink;

  const double rm = median(r_ms);
  if (!quiet && world.rank() == 0) {
    auto gf = [&](double ms) { return ms > 0 ? flop / (ms / 1e3) / 1e9 : 0.0; };
    std::printf(
        "bt=%-4d be=%-3d B=%-6d | legacy %9.4f ms (%6.2f GF/s)  P4b %9.4f ms "
        "(%6.2f GF/s)  RAW %8.4f ms | speedup %5.2fx  ovhd L=%5.1fx "
        "P4b=%5.1fx\n",
        bt, be, bt * be, legacy, gf(legacy), fused, gf(fused), rm,
        fused > 0 ? legacy / fused : 0.0, rm > 0 ? legacy / rm : 0.0,
        rm > 0 ? fused / rm : 0.0);
  }
  return {legacy, fused, rm};
}

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char** argv) {
  Cli cli = parse_cli(argc, argv);
  auto& world = TA_SCOPED_INITIALIZE(argc, argv);

  if (world.rank() == 0) {
    std::printf(
        "=== batched contraction attribution: C(b,i,k)=A(b,i,j)*B(b,j,k) "
        "===\n");
    std::printf("world.size=%d  reps=%d warmup=%d  i=%d j=%d k=%d\n",
                world.size(), cli.reps, cli.warmup, cli.I, cli.J, cli.K);
    std::printf(
        "legacy = comm-split sub-World per Hadamard tile; P4b = parent-"
        "World slices + single fence;\nRAW = one BLAS dgemm per batch "
        "elem (no TA driver). speedup = legacy/P4b.\n\n");
  }

  if (!cli.sweep) {
    run_case(world, cli, cli.bt, cli.be, /*quiet=*/false);
  } else {
    // constant total batch B = bt*be; vary granularity from coarse to fine.
    const int B = cli.bt * cli.be;
    std::vector<std::pair<int, int>> tilings;
    for (int be = B; be >= 1; be /= 2) {
      if (B % be != 0) continue;
      tilings.emplace_back(B / be, be);  // (bt, be)
    }
    if (world.rank() == 0)
      std::printf(
          "--- granularity sweep (total batch B=%d, flops CONSTANT) ---\n", B);
    for (auto [bt, be] : tilings) run_case(world, cli, bt, be, /*quiet=*/false);
    if (world.rank() == 0)
      std::printf(
          "\n(Reading: flops are identical across rows; rising EINSUM time as\n"
          " bt grows = per-Hadamard-tile machinery cost. RAW isolates the\n"
          " small-GEMM/flops floor.)\n");
  }

  if (world.rank() == 0) {
    const double pk = peak_gemm_gflops(1024, 5);
    std::printf(
        "\nmachine ref: single 1024^3 dgemm = %.1f GF/s (throughput ceiling)\n",
        pk);
  }

  return 0;
}
