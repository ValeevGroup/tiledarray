/// Shared bench helpers for arena-vs-heap case binaries.

#pragma once

#include <tiledarray.h>
#include <TiledArray/einsum/tiledarray.h>
#include <TiledArray/tensor/arena.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory_resource>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cases {

namespace TA = ::TiledArray;

using inner_t = TA::Tensor<double>;
using tile_t = TA::Tensor<inner_t>;
using ToT = TA::DistArray<tile_t, TA::DensePolicy>;
using Plain = TA::DistArray<inner_t, TA::DensePolicy>;

inline int& g_tile_grid() {
  static int v = 7;
  return v;
}

/// Stores the h-dimension scale set by --h-scale.
inline int& g_h_scale() {
  static int v = 1;
  return v;
}

inline std::vector<std::size_t> tile_breaks(int n, int ntiles) {
  if (ntiles <= 1 || n <= 0)
    return {0, static_cast<std::size_t>(std::max(n, 0))};
  std::vector<std::size_t> b;
  b.reserve(ntiles + 1);
  const int chunk = n / ntiles;
  for (int t = 0; t < ntiles; ++t) {
    b.push_back(static_cast<std::size_t>(t * chunk));
  }
  b.push_back(static_cast<std::size_t>(n));
  std::vector<std::size_t> uniq;
  for (auto x : b)
    if (uniq.empty() || uniq.back() != x) uniq.push_back(x);
  return uniq;
}

inline TA::TiledRange1 tr1_dim(int n) {
  auto b = tile_breaks(n, g_tile_grid());
  return TA::TiledRange1(b.begin(), b.end());
}

inline TA::TiledRange tr3(int a, int b, int c) {
  return TA::TiledRange{tr1_dim(a), tr1_dim(b), tr1_dim(c)};
}

inline TA::TiledRange tr4(int a, int b, int c, int d) {
  return TA::TiledRange{tr1_dim(a), tr1_dim(b), tr1_dim(c), tr1_dim(d)};
}

/// Builds a 3-D slab-backed jagged ToT.
template <typename Fn>
ToT make_tot_3d_jagged_slab(TA::World& world, int A, int B, int C,
                            double offset, Fn inner_fn) {
  ToT out(world, tr3(A, B, C));
  out.init_tiles([offset, inner_fn](const TA::Range& tile_range) {
    const std::size_t n_cells = tile_range.volume();
    std::vector<TA::Range> ranges;
    ranges.reserve(n_cells);
    std::vector<std::size_t> cell_offsets(n_cells);
    std::size_t total_elems = 0;
    {
      std::size_t ord = 0;
      for (auto outer_idx : tile_range) {
        const long o0 = static_cast<long>(outer_idx[0]);
        const long o1 = static_cast<long>(outer_idx[1]);
        const long o2 = static_cast<long>(outer_idx[2]);
        TA::Range ir = inner_fn(o0, o1, o2);
        cell_offsets[ord] = total_elems;
        const std::size_t vol = ir.volume();
        const std::size_t padded = (vol + 7) & ~std::size_t{7};
        total_elems += padded;
        ranges.push_back(std::move(ir));
        ++ord;
      }
    }

    std::shared_ptr<double[]> slab;
    if (total_elems > 0) {
      void* raw = nullptr;
      if (posix_memalign(&raw, 64, total_elems * sizeof(double)) != 0) {
        std::abort();
      }
      slab = std::shared_ptr<double[]>(static_cast<double*>(raw),
                                       [](double* p) { std::free(p); });
    }

    tile_t tile(tile_range);
    std::size_t ord = 0;
    for (auto outer_idx : tile_range) {
      const long o0 = static_cast<long>(outer_idx[0]);
      const long o1 = static_cast<long>(outer_idx[1]);
      const long o2 = static_cast<long>(outer_idx[2]);
      auto& ir = ranges[ord];
      const std::size_t vol = ir.volume();
      if (vol == 0) {
        *(tile.data() + ord) = inner_t{};
      } else {
        std::shared_ptr<double[]> alias(slab,
                                        slab.get() + cell_offsets[ord]);
        for (std::size_t k = 0; k < vol; ++k)
          alias[k] = offset + 1e-4 * static_cast<double>(
                                          o0 * 100000 + o1 * 1000 + o2 * 100 + k);
        *(tile.data() + ord) = inner_t(ir, std::move(alias));
      }
      ++ord;
    }
    return tile;
  });
  world.gop.fence();
  return out;
}

/// Builds a 3-D heap-scattered jagged ToT.
template <typename Fn>
ToT make_tot_3d_jagged(TA::World& world, int A, int B, int C, double offset,
                       Fn inner_fn) {
  ToT out(world, tr3(A, B, C));
  out.init_tiles([offset, inner_fn](const TA::Range& tile_range) {
    tile_t tile(tile_range);
    std::size_t ord = 0;
    for (auto outer_idx : tile_range) {
      const long o0 = static_cast<long>(outer_idx[0]);
      const long o1 = static_cast<long>(outer_idx[1]);
      const long o2 = static_cast<long>(outer_idx[2]);
      TA::Range ir = inner_fn(o0, o1, o2);
      const std::size_t vol = ir.volume();
      if (vol == 0) {
        *(tile.data() + ord) = inner_t{};
      } else {
        inner_t inner(ir);
        for (std::size_t k = 0; k < vol; ++k)
          inner.at_ordinal(k) =
              offset + 1e-4 * static_cast<double>(
                                  o0 * 100000 + o1 * 1000 + o2 * 100 + k);
        *(tile.data() + ord) = std::move(inner);
      }
      ++ord;
    }
    return tile;
  });
  world.gop.fence();
  return out;
}

/// Builds a 4-D slab-backed jagged ToT.
template <typename Fn>
ToT make_tot_4d_jagged_slab(TA::World& world, int A, int B, int C, int D,
                            double offset, Fn inner_fn) {
  ToT out(world, tr4(A, B, C, D));
  out.init_tiles([offset, inner_fn](const TA::Range& tile_range) {
    const std::size_t n_cells = tile_range.volume();
    std::vector<TA::Range> ranges;
    ranges.reserve(n_cells);
    std::vector<std::size_t> cell_offsets(n_cells);
    std::size_t total_elems = 0;
    {
      std::size_t ord = 0;
      for (auto outer_idx : tile_range) {
        const long o0 = static_cast<long>(outer_idx[0]);
        const long o1 = static_cast<long>(outer_idx[1]);
        const long o2 = static_cast<long>(outer_idx[2]);
        const long o3 = static_cast<long>(outer_idx[3]);
        TA::Range ir = inner_fn(o0, o1, o2, o3);
        cell_offsets[ord] = total_elems;
        const std::size_t vol = ir.volume();
        const std::size_t padded = (vol + 7) & ~std::size_t{7};
        total_elems += padded;
        ranges.push_back(std::move(ir));
        ++ord;
      }
    }
    std::shared_ptr<double[]> slab;
    if (total_elems > 0) {
      void* raw = nullptr;
      if (posix_memalign(&raw, 64, total_elems * sizeof(double)) != 0) {
        std::abort();
      }
      slab = std::shared_ptr<double[]>(static_cast<double*>(raw),
                                       [](double* p) { std::free(p); });
    }
    tile_t tile(tile_range);
    std::size_t ord = 0;
    for (auto outer_idx : tile_range) {
      const long o0 = static_cast<long>(outer_idx[0]);
      const long o1 = static_cast<long>(outer_idx[1]);
      const long o2 = static_cast<long>(outer_idx[2]);
      const long o3 = static_cast<long>(outer_idx[3]);
      auto& ir = ranges[ord];
      const std::size_t vol = ir.volume();
      if (vol == 0) {
        *(tile.data() + ord) = inner_t{};
      } else {
        std::shared_ptr<double[]> alias(slab,
                                        slab.get() + cell_offsets[ord]);
        for (std::size_t k = 0; k < vol; ++k)
          alias[k] = offset + 1e-4 * static_cast<double>(
                                          o0 * 1000000 + o1 * 10000 +
                                          o2 * 100 + o3 * 10 + k);
        *(tile.data() + ord) = inner_t(ir, std::move(alias));
      }
      ++ord;
    }
    return tile;
  });
  world.gop.fence();
  return out;
}

/// Builds a 4-D heap-scattered jagged ToT.
template <typename Fn>
ToT make_tot_4d_jagged(TA::World& world, int A, int B, int C, int D,
                       double offset, Fn inner_fn) {
  ToT out(world, tr4(A, B, C, D));
  out.init_tiles([offset, inner_fn](const TA::Range& tile_range) {
    tile_t tile(tile_range);
    std::size_t ord = 0;
    for (auto outer_idx : tile_range) {
      const long o0 = static_cast<long>(outer_idx[0]);
      const long o1 = static_cast<long>(outer_idx[1]);
      const long o2 = static_cast<long>(outer_idx[2]);
      const long o3 = static_cast<long>(outer_idx[3]);
      TA::Range ir = inner_fn(o0, o1, o2, o3);
      const std::size_t vol = ir.volume();
      if (vol == 0) {
        *(tile.data() + ord) = inner_t{};
      } else {
        inner_t inner(ir);
        for (std::size_t k = 0; k < vol; ++k)
          inner.at_ordinal(k) =
              offset + 1e-4 * static_cast<double>(
                                  o0 * 1000000 + o1 * 10000 + o2 * 100 +
                                  o3 * 10 + k);
        *(tile.data() + ord) = std::move(inner);
      }
      ++ord;
    }
    return tile;
  });
  world.gop.fence();
  return out;
}

inline Plain make_plain_3d(TA::World& world, int A, int B, int C,
                           double offset) {
  Plain out(world, tr3(A, B, C));
  out.init_tiles([offset](const TA::Range& r) {
    inner_t tile(r);
    for (std::size_t k = 0; k < r.volume(); ++k)
      tile.at_ordinal(k) = offset + 1e-3 * static_cast<double>(k);
    return tile;
  });
  world.gop.fence();
  return out;
}

inline double max_abs_diff(const ToT& a, const ToT& b) {
  if (a.trange() != b.trange()) return 1e30;
  double mx = 0.0;
  const auto& tr = a.trange();
  for (auto t = tr.tiles_range().begin(); t != tr.tiles_range().end(); ++t) {
    if (!a.is_local(*t)) continue;
    auto ta = a.find(*t).get();
    auto tb = b.find(*t).get();
    if (ta.range().volume() != tb.range().volume()) return 1e30;
    for (std::size_t ord = 0; ord < ta.range().volume(); ++ord) {
      const auto& ia = *(ta.data() + ord);
      const auto& ib = *(tb.data() + ord);
      if (ia.range().volume() != ib.range().volume()) {
        if (ia.range().volume() == 0 || ib.range().volume() == 0) {
          mx = std::max(mx, 1.0);
          continue;
        }
        return 1e30;
      }
      for (std::size_t k = 0; k < ia.range().volume(); ++k) {
        double d = std::abs(ia.at_ordinal(k) - ib.at_ordinal(k));
        if (d > mx) mx = d;
      }
    }
  }
  return mx;
}

struct RunResult {
  double wall_ns_min = 0.0;
  double wall_ns_med = 0.0;
  ToT result;
  bool ok = true;
  std::string err;
};

template <typename Runner>
RunResult time_run(TA::World& world, Runner&& run, bool disable_arena,
                   int repeats) {
  RunResult R;
  std::vector<double> ns;
  ns.reserve(repeats);
  for (int r = 0; r < repeats; ++r) {
    TA::detail::arena_disabled() = disable_arena;
    world.gop.fence();
    auto t0 = std::chrono::steady_clock::now();
    try {
      R.result = run();
      world.gop.fence();
    } catch (std::exception& e) {
      R.ok = false;
      R.err = e.what();
      return R;
    } catch (...) {
      R.ok = false;
      R.err = "unknown";
      return R;
    }
    auto t1 = std::chrono::steady_clock::now();
    ns.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  }
  std::sort(ns.begin(), ns.end());
  R.wall_ns_min = ns.front();
  R.wall_ns_med = ns[ns.size() / 2];
  return R;
}

/// Runs a case binary by building operands once and timing one mode.
template <typename Build, typename Run>
int run_case_main(int argc, char** argv, const char* case_name, Build build,
                  Run run) {
  // Heap and arena timings must run in separate processes to avoid allocator/cache bias.
  std::string mode;
  int repeats = 3;
  bool quiet = false;
  int tile_grid = 7;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (a == "--repeat" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (a == "--tile-grid" && i + 1 < argc) {
      tile_grid = std::max(1, std::atoi(argv[++i]));
    } else if (a == "--h-scale" && i + 1 < argc) {
      g_h_scale() = std::max(1, std::atoi(argv[++i]));
    } else if (a == "--quiet") {
      quiet = true;
    } else if (a == "-h" || a == "--help") {
      std::cout
          << "Usage: " << argv[0]
          << " --mode {heap|arena} [--tile-grid G] [--h-scale S] "
             "[--repeat R] [--quiet]\n"
             "MAD_NUM_THREADS env var controls thread count.\n"
             "Note: --mode is required. heap and arena MUST be benchmarked\n"
             "in separate processes — running both in one process biases the\n"
             "second run via allocator fragmentation and cache residue.\n";
      return 0;
    }
  }
  if (mode != "heap" && mode != "arena") {
    std::cerr << "error: --mode must be 'heap' or 'arena' (got '"
              << mode << "')\n";
    return 2;
  }
  g_tile_grid() = tile_grid;

  TA::World& world = TA::initialize(argc, argv);

  const char* threads_env = std::getenv("MAD_NUM_THREADS");
  std::string threads_label = threads_env ? threads_env : "default";

  std::cout << "case,mode,tile_grid,threads,wall_ns_min,wall_ns_med,verified\n";

  if (!quiet) {
    std::cerr << "# " << case_name << " tile_grid=" << tile_grid
              << " h_scale=" << g_h_scale()
              << " threads=" << threads_label << "\n";
  }

  auto operands = build(world);

  auto emit = [&](const char* m, const RunResult& R, const std::string& v) {
    if (!R.ok) {
      std::cout << case_name << "," << m << "," << tile_grid << ","
                << threads_label << ",NA,NA,err:" << R.err << "\n";
      return;
    }
    std::cout << case_name << "," << m << "," << tile_grid << ","
              << threads_label << "," << static_cast<long long>(R.wall_ns_min)
              << "," << static_cast<long long>(R.wall_ns_med) << "," << v
              << "\n";
  };

  if (mode == "heap") {
    auto Rh = time_run(
        world, [&]() { return run(operands); }, true,
        repeats);
    emit("heap", Rh, "single");
    if (!quiet) {
      std::cerr << "  heap=" << Rh.wall_ns_med / 1e6 << "ms\n";
    }
  } else {
    auto Ra = time_run(
        world, [&]() { return run(operands); }, false,
        repeats);
    emit("arena", Ra, "single");
    if (!quiet) {
      std::cerr << "  arena=" << Ra.wall_ns_med / 1e6 << "ms\n";
    }
  }

  std::cout.flush();
  TA::detail::arena_disabled() = false;
  TA::finalize();
  return 0;
}

/// Runs a case binary with separate heap-scatter and arena-slab input builders.
template <typename BuildHeap, typename BuildArena, typename Run>
int run_case_main_split(int argc, char** argv, const char* case_name,
                        BuildHeap build_heap, BuildArena build_arena,
                        Run run) {
  // Heap and arena timings must run in separate processes to avoid allocator/cache bias.
  std::string mode;
  int repeats = 3;
  bool quiet = false;
  int tile_grid = 7;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (a == "--repeat" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (a == "--tile-grid" && i + 1 < argc) {
      tile_grid = std::max(1, std::atoi(argv[++i]));
    } else if (a == "--h-scale" && i + 1 < argc) {
      g_h_scale() = std::max(1, std::atoi(argv[++i]));
    } else if (a == "--quiet") {
      quiet = true;
    } else if (a == "-h" || a == "--help") {
      std::cout << "Usage: " << argv[0]
                << " --mode {heap|arena} [--tile-grid G] [--h-scale S] "
                   "[--repeat R] [--quiet]\n"
                   "Heap mode uses scattered (legacy) inputs; arena mode "
                   "uses slab-backed inputs.\n"
                   "Note: --mode is required. heap and arena MUST be "
                   "benchmarked in separate\n"
                   "processes — running both in one process biases the "
                   "second run via allocator\n"
                   "fragmentation and cache residue.\n";
      return 0;
    }
  }
  if (mode != "heap" && mode != "arena") {
    std::cerr << "error: --mode must be 'heap' or 'arena' (got '"
              << mode << "')\n";
    return 2;
  }
  g_tile_grid() = tile_grid;

  TA::World& world = TA::initialize(argc, argv);

  const char* threads_env = std::getenv("MAD_NUM_THREADS");
  std::string threads_label = threads_env ? threads_env : "default";

  std::cout << "case,mode,tile_grid,threads,wall_ns_min,wall_ns_med,verified\n";

  if (!quiet) {
    std::cerr << "# " << case_name << " tile_grid=" << tile_grid
              << " h_scale=" << g_h_scale()
              << " threads=" << threads_label
              << " (split inputs: heap=scatter, arena=slab)\n";
  }

  auto emit = [&](const char* m, const RunResult& R, const std::string& v) {
    if (!R.ok) {
      std::cout << case_name << "," << m << "," << tile_grid << ","
                << threads_label << ",NA,NA,err:" << R.err << "\n";
      return;
    }
    std::cout << case_name << "," << m << "," << tile_grid << ","
              << threads_label << "," << static_cast<long long>(R.wall_ns_min)
              << "," << static_cast<long long>(R.wall_ns_med) << "," << v
              << "\n";
  };

  if (mode == "heap") {
    auto operands = build_heap(world);
    auto Rh = time_run(
        world, [&]() { return run(operands); }, true,
        repeats);
    emit("heap", Rh, "single");
    if (!quiet) std::cerr << "  heap=" << Rh.wall_ns_med / 1e6 << "ms\n";
  } else {
    auto operands = build_arena(world);
    auto Ra = time_run(
        world, [&]() { return run(operands); }, false,
        repeats);
    emit("arena", Ra, "single");
    if (!quiet) std::cerr << "  arena=" << Ra.wall_ns_med / 1e6 << "ms\n";
  }

  std::cout.flush();
  TA::detail::arena_disabled() = false;
  TA::finalize();
  return 0;
}

}
