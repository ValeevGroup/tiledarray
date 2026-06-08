// opa_strided_arena_dgemm.cpp
// ---------------------------------------------------------------------------
// Standalone tile/BLAS-level benchmark for the op_A ToT x ToT -> ToT
// product, on ArenaTensor inner cells. Companion to opb_strided_arena_dgemm.cpp.
//
//   op_A:  I(i_2,i_1,Κ; a_1,a_4) * I(i_2,i_1,μ̃,Κ; a_4) -> I(μ̃,i_2,i_1; a_1)
//          Hadamard = {i_1,i_2}, outer-contracted = {Κ}, outer-external = {μ̃},
//          inner-contracted = {a_4}, inner-external (kept) = {a_1}.
//          P = |a_1|, Q = |a_4|  depend (jaggedly) on the Hadamard slice.
//
//   C(μ̃,i_2,i_1; a_1) = sum_Κ sum_{a_4} L(i_2,i_1,Κ; a_1,a_4)*R(i_2,i_1,μ̃,Κ; a_4)
//
// Unlike op_B (inner OUTER-product), op_A's inner part is a real CONTRACTION
// over a_4, and the full reduction spans an OUTER index (Κ) AND an INNER index
// (a_4). That means the *direct* op_B-style fusion of the outer contraction Κ
// would have to merge (Κ ⊗ a_4) into a single BLAS K axis -- a two-level stride
// no single leading-dimension can express -> a pack/deep-clone is required.
//
// But a DIFFERENT, ZERO-COPY fusion exists: ride the outer EXTERNAL μ̃ into the
// GEMM M axis (one outer index, via inter-cell stride), keep a_4 as the
// contiguous inner K, keep a_1 as inner N, and loop the outer contraction Κ
// with beta-accumulation:
//
//   for Κ:  C̃[μ̃, a_1] += R̃_Κ[μ̃, a_4] · L_Κ[a_1, a_4]^T      (M=|μ̃|, N=P, K=Q)
//
// This benchmark compares four ways to evaluate one Hadamard slice, all reading
// already-fusable arena slabs (operands laid contracted/contiguous-friendly):
//
//   current_gemm : Mμ·nK tiny (P x 1, K=Q) GEMMs -- exactly what TA dispatches
//                  today (per result cell, per Κ: one inner gemm with N=1).
//   current_gemv : Mμ·nK BLAS dgemv calls (the natural mat-vec primitive).
//   strided        : nK strided GEMMs (μ̃ ridden into M), zero-copy, beta-accum.
//   packed       : pack (Κ,a_4) contiguous then ONE GEMM (M=Mμ,N=P,K=nK·Q);
//                  the pack cost (= the deep-clone tradeoff) is timed too.
//
// Work units {P,Q,Mμ,nK} per Hadamard slice are reconstructed from op_dump.txt.
// (Idealization: the μ̃ x Κ block per slice is treated as dense; Mμ and nK are
//  the per-slice nonzero μ̃ / Κ counts. The shape distribution -- which is what
//  drives the GEMV->GEMM win -- is faithful.)
// ---------------------------------------------------------------------------

#include <tiledarray.h>
#include <TiledArray/math/blas.h>
#include <TiledArray/tensor/arena_kernels.h>
#include <TiledArray/tensor/arena_tensor.h>

#include <blas.hh>
#include <btas/zb/range.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace TA = TiledArray;

using Inner = TA::ArenaTensor<double>;
using InnerRange = typename Inner::range_type;
using Outer = TA::Tensor<Inner>;

using clock_type = std::chrono::steady_clock;
static double ms_since(clock_type::time_point t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() -
                                                              t0)
             .count() /
         1.0e6;
}

// ===========================================================================
// op_dump.txt parser (same structure as opb_strided_arena_dgemm.cpp)
// ===========================================================================

struct InnerRangeKey {
  std::vector<long> lo, hi;
};
struct TileSpec {
  long outer_vol = 0;
  std::vector<InnerRangeKey> distinct;
  std::vector<int> cell_labels;
};
struct ArraySpec {
  int outer_rank = 0;
  std::vector<std::vector<long>> dim_tile_bounds;
  std::map<std::size_t, TileSpec> tiles;
};
struct DumpEntry {
  std::string annot_left, annot_right, annot_out;
  double t_einsum = 0;
  ArraySpec L, R, C;
};

static std::vector<long> parse_long_list(std::string s) {
  std::vector<long> v;
  std::string buf;
  for (char c : s) {
    if (c == '[' || c == ' ') continue;
    if (c == ',' || c == ']') {
      if (!buf.empty()) {
        v.push_back(std::stol(buf));
        buf.clear();
      }
      if (c == ']') break;
    } else
      buf.push_back(c);
  }
  if (!buf.empty()) v.push_back(std::stol(buf));
  return v;
}
static std::string get_kv(std::string const& line, std::string const& key) {
  auto pos = line.find(key + "=");
  if (pos == std::string::npos) return "";
  pos += key.size() + 1;
  auto end = line.find_first_of(" \t\n", pos);
  return line.substr(pos, end == std::string::npos ? std::string::npos
                                                   : end - pos);
}
static std::string get_kv_bracket(std::string const& line,
                                  std::string const& key) {
  auto pos = line.find(key + "=[");
  if (pos == std::string::npos) return "";
  pos += key.size() + 1;
  auto end = line.find(']', pos);
  if (end == std::string::npos) return "";
  return line.substr(pos, end - pos + 1);
}
struct CellRun {
  std::size_t lo, hi;
  int label;
};
static std::vector<CellRun> parse_cells_rle(std::string const& s) {
  std::vector<CellRun> runs;
  std::size_t i = 0;
  while (i < s.size()) {
    if (s[i] != '[') {
      ++i;
      continue;
    }
    auto end = s.find(']', i);
    if (end == std::string::npos) break;
    auto inner = s.substr(i + 1, end - i - 1);
    auto dotdot = inner.find("..");
    auto colon = inner.find(':');
    if (dotdot == std::string::npos || colon == std::string::npos) {
      i = end + 1;
      continue;
    }
    CellRun r;
    r.lo = std::stoull(inner.substr(0, dotdot));
    r.hi = std::stoull(inner.substr(dotdot + 2, colon - dotdot - 2));
    auto lab = inner.substr(colon + 1);
    r.label = (lab == "E") ? -1 : std::stoi(lab.substr(1));
    runs.push_back(r);
    i = end + 1;
  }
  return runs;
}
static std::vector<DumpEntry> parse_dump(std::string const& path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<DumpEntry> entries;
  DumpEntry* cur = nullptr;
  ArraySpec* arr = nullptr;
  TileSpec* tile = nullptr;
  std::string line;
  bool in_tiles = false;
  while (std::getline(f, line)) {
    if (line.rfind("===== OP DUMP id=", 0) == 0) {
      entries.emplace_back();
      cur = &entries.back();
      auto t = line.find("t_einsum=");
      if (t != std::string::npos) cur->t_einsum = std::stod(line.substr(t + 9));
      arr = nullptr;
      tile = nullptr;
      in_tiles = false;
    } else if (!cur) {
      continue;
    } else if (line.rfind("annot_left=", 0) == 0) {
      cur->annot_left = line.substr(11);
    } else if (line.rfind("annot_right=", 0) == 0) {
      cur->annot_right = line.substr(12);
    } else if (line.rfind("annot_out=", 0) == 0) {
      cur->annot_out = line.substr(10);
    } else if (line.rfind("L.outer_rank=", 0) == 0) {
      arr = &cur->L;
      arr->outer_rank = std::stoi(line.substr(13));
      in_tiles = false;
    } else if (line.rfind("R.outer_rank=", 0) == 0) {
      arr = &cur->R;
      arr->outer_rank = std::stoi(line.substr(13));
      in_tiles = false;
    } else if (line.rfind("C.outer_rank=", 0) == 0) {
      arr = &cur->C;
      arr->outer_rank = std::stoi(line.substr(13));
      in_tiles = false;
    } else if (arr && line.find(".dim[") != std::string::npos &&
               line.find("tile_bounds=") != std::string::npos) {
      arr->dim_tile_bounds.push_back(
          parse_long_list(get_kv_bracket(line, "tile_bounds")));
    } else if (arr && line.find(".tiles_BEGIN") != std::string::npos) {
      in_tiles = true;
      tile = nullptr;
    } else if (arr && line.find(".tiles_END") != std::string::npos) {
      in_tiles = false;
      tile = nullptr;
    } else if (in_tiles && arr && line.rfind("  ord=", 0) == 0) {
      std::size_t ord = std::stoull(get_kv(line, "ord"));
      auto& ts = arr->tiles[ord];
      ts.outer_vol = std::stol(get_kv(line, "outer_vol"));
      ts.cell_labels.assign(ts.outer_vol, -1);
      tile = &ts;
    } else if (in_tiles && tile && line.rfind("    range[", 0) == 0) {
      InnerRangeKey key;
      key.lo = parse_long_list(get_kv_bracket(line, "inner_lo"));
      key.hi = parse_long_list(get_kv_bracket(line, "inner_hi"));
      tile->distinct.push_back(std::move(key));
    } else if (in_tiles && tile && line.rfind("    cells_rle=", 0) == 0) {
      for (auto& r : parse_cells_rle(line.substr(14)))
        for (std::size_t k = r.lo; k <= r.hi && k < tile->cell_labels.size();
             ++k)
          tile->cell_labels[k] = r.label;
    }
  }
  for (auto& e : entries)
    for (auto* a : {&e.L, &e.R, &e.C})
      for (auto& kv : a->tiles) {
        auto& ts = kv.second;
        bool any = false;
        for (int l : ts.cell_labels)
          if (l >= 0) {
            any = true;
            break;
          }
        if (!any && !ts.distinct.empty())
          std::fill(ts.cell_labels.begin(), ts.cell_labels.end(), 0);
      }
  return entries;
}

// ===========================================================================
// Fast cell-presence queries
// ===========================================================================

struct DimTiling {
  std::vector<long> bounds;
  std::vector<int> elem_to_tile;
  long extent() const { return bounds.empty() ? 0 : bounds.back(); }
  int ntiles() const { return static_cast<int>(bounds.size()) - 1; }
};
static DimTiling make_dim_tiling(std::vector<long> const& bounds) {
  DimTiling dt;
  dt.bounds = bounds;
  long n = bounds.empty() ? 0 : bounds.back();
  dt.elem_to_tile.assign(n, 0);
  for (int t = 0; t + 1 < (int)bounds.size(); ++t)
    for (long e = bounds[t]; e < bounds[t + 1]; ++e) dt.elem_to_tile[e] = t;
  return dt;
}
struct OperandIndex {
  ArraySpec const* spec = nullptr;
  std::vector<DimTiling> dims;
  std::vector<std::size_t> tile_strides;
  void build(ArraySpec const& s) {
    spec = &s;
    dims.clear();
    for (auto const& b : s.dim_tile_bounds) dims.push_back(make_dim_tiling(b));
    int rank = s.outer_rank;
    tile_strides.assign(rank, 1);
    for (int d = rank - 2; d >= 0; --d)
      tile_strides[d] = tile_strides[d + 1] * dims[d + 1].ntiles();
  }
  int rank() const { return spec->outer_rank; }
  long dim_extent(int d) const { return dims[d].extent(); }
  int query(std::vector<long> const& idx,
            std::vector<long>* out_hi = nullptr) const {
    int rank = spec->outer_rank;
    std::size_t tord = 0;
    std::vector<long> tile_lo(rank), tile_ext(rank);
    for (int d = 0; d < rank; ++d) {
      int t = dims[d].elem_to_tile[idx[d]];
      tord += static_cast<std::size_t>(t) * tile_strides[d];
      tile_lo[d] = dims[d].bounds[t];
      tile_ext[d] = dims[d].bounds[t + 1] - dims[d].bounds[t];
    }
    auto it = spec->tiles.find(tord);
    if (it == spec->tiles.end()) return -1;
    auto const& ts = it->second;
    std::size_t k = 0;
    for (int d = 0; d < rank; ++d)
      k = k * static_cast<std::size_t>(tile_ext[d]) + (idx[d] - tile_lo[d]);
    if (k >= ts.cell_labels.size()) return -1;
    int lbl = ts.cell_labels[k];
    if (lbl >= 0 && out_hi) *out_hi = ts.distinct[lbl].hi;
    return lbl;
  }
};

// ===========================================================================
// Annotation classification
// ===========================================================================

static void split_annot(std::string const& annot, std::vector<std::string>& o,
                        std::vector<std::string>& in) {
  o.clear();
  in.clear();
  auto semi = annot.find(';');
  std::string os = annot.substr(0, semi);
  std::string is = (semi == std::string::npos) ? "" : annot.substr(semi + 1);
  auto split = [](std::string const& s, std::vector<std::string>& out) {
    std::string buf;
    for (char c : s) {
      if (c == ',') {
        if (!buf.empty()) out.push_back(buf);
        buf.clear();
      } else
        buf.push_back(c);
    }
    if (!buf.empty()) out.push_back(buf);
  };
  split(os, o);
  split(is, in);
}
static int find_idx(std::vector<std::string> const& v, std::string const& s) {
  for (int i = 0; i < (int)v.size(); ++i)
    if (v[i] == s) return i;
  return -1;
}

// ===========================================================================
// Per-Hadamard-slice work unit
// ===========================================================================

struct WorkUnit {
  long P, Q, Mmu, nK;
};

// Ready-to-fuse arena operands for one slice.
struct SliceOperand {
  Outer Lslab;   // outer {nK},        inner {P,Q}: L_Κ[a_1,a_4]
  Outer Rslab;   // outer {nK, Mμ},    inner {Q}  : R(Κ,μ̃)[a_4], μ̃ fastest
  Outer Cslab;   // outer {Mμ},        inner {P}  : C(μ̃)[a_1]
  long P, Q, Mmu, nK;
  std::ptrdiff_t sR;  // inter-μ̃-cell stride within a Κ block (elements)
  std::ptrdiff_t sC;  // inter-μ̃-cell stride of C (elements)
  std::vector<double> ref;  // golden Mμ*P reference
  // scratch pack buffers (reused across reps for the packed path)
  std::vector<double> Lpacked, Rpacked;
};

static void build_slice(SliceOperand& s, WorkUnit const& wu,
                        std::mt19937_64& rng) {
  s.P = wu.P;
  s.Q = wu.Q;
  s.Mmu = wu.Mmu;
  s.nK = wu.nK;
  const long P = wu.P, Q = wu.Q, Mmu = wu.Mmu, nK = wu.nK;

  s.Lslab = TA::detail::arena_outer_init<Outer>(
      TA::Range{static_cast<std::size_t>(nK)}, 1,
      [P, Q](std::size_t) { return InnerRange{P, Q}; }, /*zero_init=*/false);
  s.Rslab = TA::detail::arena_outer_init<Outer>(
      TA::Range{static_cast<std::size_t>(nK), static_cast<std::size_t>(Mmu)}, 1,
      [Q](std::size_t) { return InnerRange{Q}; }, /*zero_init=*/false);
  s.Cslab = TA::detail::arena_outer_init<Outer>(
      TA::Range{static_cast<std::size_t>(Mmu)}, 1,
      [P](std::size_t) { return InnerRange{P}; }, /*zero_init=*/true);

  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (long k = 0; k < nK; ++k) {
    double* l = s.Lslab.data()[k].data();
    for (long e = 0; e < P * Q; ++e) l[e] = dist(rng);
    for (long mu = 0; mu < Mmu; ++mu) {
      double* r = s.Rslab.data()[k * Mmu + mu].data();
      for (long q = 0; q < Q; ++q) r[q] = dist(rng);
    }
  }
  // strides (constant within a uniform-Q / uniform-P slice)
  s.sR = (Mmu > 1)
             ? (s.Rslab.data()[1].data() - s.Rslab.data()[0].data())  // μ̃ fast
             : Q;
  s.sC = (Mmu > 1) ? (s.Cslab.data()[1].data() - s.Cslab.data()[0].data()) : P;

  // golden reference: C[μ̃,a_1] = sum_Κ sum_{a_4} L_Κ[a_1,a_4] * R(Κ,μ̃)[a_4]
  s.ref.assign(static_cast<std::size_t>(Mmu) * P, 0.0);
  for (long k = 0; k < nK; ++k) {
    const double* l = s.Lslab.data()[k].data();  // P x Q row-major
    for (long mu = 0; mu < Mmu; ++mu) {
      const double* r = s.Rslab.data()[k * Mmu + mu].data();  // Q
      double* c = &s.ref[mu * P];
      for (long a1 = 0; a1 < P; ++a1) {
        double acc = 0;
        const double* lr = l + a1 * Q;
        for (long a4 = 0; a4 < Q; ++a4) acc += lr[a4] * r[a4];
        c[a1] += acc;
      }
    }
  }
  s.Lpacked.assign(static_cast<std::size_t>(P) * nK * Q, 0.0);
  s.Rpacked.assign(static_cast<std::size_t>(Mmu) * nK * Q, 0.0);
}

// ===========================================================================
// The four evaluation paths (one slice)
// ===========================================================================

namespace tamb = TiledArray::math::blas;
using integer = tamb::integer;

static void zero_C(SliceOperand& s) {
  for (long mu = 0; mu < s.Mmu; ++mu)
    std::memset(s.Cslab.data()[mu].data(), 0, sizeof(double) * s.P);
}

// current_gemm: Mμ·nK tiny (P x 1, K=Q) GEMMs -- TA's actual inner-gemm shape.
static void eval_current_gemm(SliceOperand& s) {
  const integer P = s.P, Q = s.Q;
  for (long k = 0; k < s.nK; ++k) {
    const double* Lk = s.Lslab.data()[k].data();  // P x Q
    for (long mu = 0; mu < s.Mmu; ++mu) {
      const double* r = s.Rslab.data()[k * s.Mmu + mu].data();  // Q
      double* c = s.Cslab.data()[mu].data();                    // P
      // C(P x 1) += L(P x Q) * R(Q x 1)
      tamb::gemm(tamb::Op::NoTrans, tamb::Op::NoTrans, /*M=*/P, /*N=*/1,
                 /*K=*/Q, 1.0, /*A=*/Lk, /*lda=*/Q, /*B=*/r, /*ldb=*/1,
                 /*beta=*/1.0, /*C=*/c, /*ldc=*/1);
    }
  }
}

// current_gemv: Mμ·nK BLAS dgemv calls.
static void eval_current_gemv(SliceOperand& s) {
  const integer P = s.P, Q = s.Q;
  for (long k = 0; k < s.nK; ++k) {
    const double* Lk = s.Lslab.data()[k].data();
    for (long mu = 0; mu < s.Mmu; ++mu) {
      const double* r = s.Rslab.data()[k * s.Mmu + mu].data();
      double* c = s.Cslab.data()[mu].data();
      // y(P) += L(P x Q) * x(Q)
      ::blas::gemv(::blas::Layout::RowMajor, ::blas::Op::NoTrans, P, Q, 1.0, Lk,
                   Q, r, 1, 1.0, c, 1);
    }
  }
}

// strided: nK strided GEMMs, μ̃ ridden into M, a_4 the contiguous inner K.
//   C̃[μ̃,a_1] += R̃_Κ[μ̃,a_4] · L_Κ[a_1,a_4]^T
//   R̃_Κ: M x K = Mμ x Q, lda = sR (μ̃ rows strided; a_4 contiguous)
//   L_Κ: stored a_1 x a_4 = N x K, op=Trans, ldb = Q
//   C̃ : M x N = Mμ x P, ldc = sC
static void eval_strided(SliceOperand& s) {
  const integer Mmu = s.Mmu, P = s.P, Q = s.Q;
  for (long k = 0; k < s.nK; ++k) {
    const double* Rk = s.Rslab.data()[k * s.Mmu].data();  // base of Κ block
    const double* Lk = s.Lslab.data()[k].data();
    double* C = s.Cslab.data()[0].data();
    tamb::gemm(tamb::Op::NoTrans, tamb::Op::Trans, /*M=*/Mmu, /*N=*/P, /*K=*/Q,
               1.0, /*A=*/Rk, /*lda=*/static_cast<integer>(s.sR), /*B=*/Lk,
               /*ldb=*/Q, /*beta=*/(k == 0 ? 0.0 : 1.0), /*C=*/C,
               /*ldc=*/static_cast<integer>(s.sC));
  }
}

// packed: pack (Κ,a_4) contiguous, then ONE GEMM. Pack cost is timed (this is
// the deep-clone tradeoff). C̃[μ̃,a_1] = Rp[μ̃,(Κ,a_4)] · Lp[a_1,(Κ,a_4)]^T
static void eval_packed(SliceOperand& s) {
  const integer Mmu = s.Mmu, P = s.P, Q = s.Q, nK = s.nK;
  const integer KK = nK * Q;
  double* Lp = s.Lpacked.data();  // P x KK row-major
  double* Rp = s.Rpacked.data();  // Mμ x KK row-major
  for (long k = 0; k < nK; ++k) {
    const double* Lk = s.Lslab.data()[k].data();  // P x Q
    for (long a1 = 0; a1 < P; ++a1)
      std::memcpy(Lp + a1 * KK + k * Q, Lk + a1 * Q, sizeof(double) * Q);
    for (long mu = 0; mu < Mmu; ++mu) {
      const double* r = s.Rslab.data()[k * Mmu + mu].data();
      std::memcpy(Rp + mu * KK + k * Q, r, sizeof(double) * Q);
    }
  }
  double* C = s.Cslab.data()[0].data();
  tamb::gemm(tamb::Op::NoTrans, tamb::Op::Trans, /*M=*/Mmu, /*N=*/P, /*K=*/KK,
             1.0, /*A=*/Rp, /*lda=*/KK, /*B=*/Lp, /*ldb=*/KK, /*beta=*/0.0,
             /*C=*/C, /*ldc=*/static_cast<integer>(s.sC));
}

static double max_abs_diff_ref(SliceOperand const& s) {
  double d = 0;
  for (long mu = 0; mu < s.Mmu; ++mu) {
    const double* c = s.Cslab.data()[mu].data();
    const double* ref = &s.ref[mu * s.P];
    for (long a1 = 0; a1 < s.P; ++a1) d = std::max(d, std::abs(c[a1] - ref[a1]));
  }
  return d;
}

// ===========================================================================
// CLI
// ===========================================================================

struct Cli {
  std::string dump =
      "/Users/zhihaodeng/packages/mpqc4/agent/experiments/C6H14/profile/"
      "op_dump.txt";
  long max_slices = 16;  // timed-pool cap (slices can be large); 0 = all
  int repeats = 5;
  int warmup = 1;
  int seed = 42;
  std::string mode = "all";  // current_gemm|current_gemv|strided|packed|all
  bool check = true;
};
static void usage() {
  std::fprintf(stderr,
               "opa_strided_arena_dgemm\n"
               "  --dump PATH        op_dump.txt path\n"
               "  --max_slices N     timed-pool cap (0=all)  (default 16)\n"
               "  --repeats R        timed reps              (default 5)\n"
               "  --warmup W         untimed warmup reps      (default 1)\n"
               "  --seed S           RNG seed                 (default 42)\n"
               "  --mode M  current_gemm|current_gemv|strided|packed|all\n"
               "  --no_check         skip correctness check\n");
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
    if (a == "--dump")
      c.dump = need();
    else if (a == "--max_slices")
      c.max_slices = std::stol(need());
    else if (a == "--repeats")
      c.repeats = std::stoi(need());
    else if (a == "--warmup")
      c.warmup = std::stoi(need());
    else if (a == "--seed")
      c.seed = std::stoi(need());
    else if (a == "--mode")
      c.mode = need();
    else if (a == "--no_check")
      c.check = false;
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
  (void)world;

  std::printf("=== op_A strided-vs-current arena DGEMM bench ===\n");
  std::printf("dump=%s\n", cli.dump.c_str());
  auto dump = parse_dump(cli.dump);

  // op_A: result annotation has exactly 1 inner index.
  auto n_inner = [](std::string const& annot) {
    auto p = annot.find(';');
    if (p == std::string::npos) return 0;
    auto in = annot.substr(p + 1);
    if (in.empty()) return 0;
    int n = 1;
    for (char c : in)
      if (c == ',') ++n;
    return n;
  };
  DumpEntry const* op = nullptr;
  for (auto const& e : dump)
    if (n_inner(e.annot_out) == 1) {
      op = &e;
      break;
    }
  if (!op) {
    std::fprintf(stderr, "ERROR: no op_A (1 inner index) in dump\n");
    return 1;
  }

  std::vector<std::string> Lo, Li, Ro, Ri, Co, Ci;
  split_annot(op->annot_left, Lo, Li);
  split_annot(op->annot_right, Ro, Ri);
  split_annot(op->annot_out, Co, Ci);

  std::vector<std::string> hadamard, contracted, ext_left, ext_right;
  for (auto const& s : Lo) {
    bool inR = find_idx(Ro, s) >= 0, inC = find_idx(Co, s) >= 0;
    if (inR && inC)
      hadamard.push_back(s);
    else if (inR && !inC)
      contracted.push_back(s);
    else if (!inR && inC)
      ext_left.push_back(s);
  }
  for (auto const& s : Ro)
    if (find_idx(Lo, s) < 0 && find_idx(Co, s) >= 0) ext_right.push_back(s);

  auto join = [](std::vector<std::string> const& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
    return s;
  };
  std::printf("  annot_left  = %s\n", op->annot_left.c_str());
  std::printf("  annot_right = %s\n", op->annot_right.c_str());
  std::printf("  annot_out   = %s\n", op->annot_out.c_str());
  std::printf(
      "  hadamard={%s} outer-contracted={%s} ext_left={%s} ext_right={%s}\n",
      join(hadamard).c_str(), join(contracted).c_str(), join(ext_left).c_str(),
      join(ext_right).c_str());
  std::printf("  inner-contracted={%s} inner-kept={%s}\n",
              [&] {  // R inner not in C inner = contracted
                std::string s;
                for (auto const& x : Ri)
                  if (find_idx(Ci, x) < 0) s += (s.empty() ? "" : ",") + x;
                return s;
              }()
                  .c_str(),
              join(Ci).c_str());

  if (contracted.size() != 1 || ext_right.size() != 1 || hadamard.empty()) {
    std::fprintf(stderr,
                 "ERROR: expected op_A form (1 outer-contracted, 1 outer-right-"
                 "external, >=1 Hadamard); got c=%zu er=%zu h=%zu\n",
                 contracted.size(), ext_right.size(), hadamard.size());
    return 1;
  }

  OperandIndex Lidx, Ridx, Cidx;
  Lidx.build(op->L);
  Ridx.build(op->R);
  Cidx.build(op->C);
  (void)Ridx;

  // position maps (index name -> dim position) for L and C queries
  std::unordered_map<std::string, int> Lpos, Cpos;
  for (int d = 0; d < (int)Lo.size(); ++d) Lpos[Lo[d]] = d;
  for (int d = 0; d < (int)Co.size(); ++d) Cpos[Co[d]] = d;

  const std::string Kname = contracted[0];   // Κ
  const std::string MUname = ext_right[0];    // μ̃
  const long Kext = Lidx.dim_extent(Lpos[Kname]);
  const long MUext = Cidx.dim_extent(Cpos[MUname]);
  std::vector<long> hext;
  for (auto const& hn : hadamard) hext.push_back(Lidx.dim_extent(Lpos[hn]));
  std::printf("  Κ extent=%ld   μ̃ extent=%ld   hadamard extents=", Kext, MUext);
  for (long h : hext) std::printf("%ld ", h);
  std::printf("\n");

  // --- reconstruct per-Hadamard-slice work units ---
  std::printf("\nreconstructing per-slice work units from dump ...\n");
  auto t_recon = clock_type::now();
  std::vector<WorkUnit> units;
  long total_slices = 0, total_result_cells = 0, total_current_calls = 0,
       total_strided_calls = 0;
  double total_flops = 0;
  std::map<long, long> P_hist, Mmu_hist, nK_hist;

  long n_had = 1;
  for (long h : hext) n_had *= h;
  std::vector<long> hidx(hadamard.size());
  std::vector<long> lq(Lo.size()), cq(Co.size()), hi;
  for (long hlin = 0; hlin < n_had; ++hlin) {
    long rem = hlin;
    for (int d = (int)hadamard.size() - 1; d >= 0; --d) {
      hidx[d] = rem % hext[d];
      rem /= hext[d];
    }
    // place hadamard values into L and C query templates
    for (int d = 0; d < (int)hadamard.size(); ++d) {
      lq[Lpos[hadamard[d]]] = hidx[d];
      cq[Cpos[hadamard[d]]] = hidx[d];
    }
    // P,Q and nK from L(i_2,i_1,Κ)
    long P = -1, Q = -1, nK = 0;
    for (long k = 0; k < Kext; ++k) {
      lq[Lpos[Kname]] = k;
      int lbl = Lidx.query(lq, &hi);
      if (lbl >= 0) {
        ++nK;
        if (P < 0) {
          P = hi[0];
          Q = (hi.size() > 1 ? hi[1] : hi[0]);
        }
      }
    }
    if (nK == 0 || P <= 0 || Q <= 0) continue;
    // Mμ from C(μ̃,i_2,i_1)
    long Mmu = 0;
    for (long mu = 0; mu < MUext; ++mu) {
      cq[Cpos[MUname]] = mu;
      if (Cidx.query(cq) >= 0) ++Mmu;
    }
    if (Mmu == 0) continue;
    units.push_back({P, Q, Mmu, nK});
    ++total_slices;
    total_result_cells += Mmu;
    total_current_calls += Mmu * nK;
    total_strided_calls += nK;
    total_flops += 2.0 * P * Q * Mmu * nK;
    ++P_hist[P];
    ++Mmu_hist[Mmu];
    ++nK_hist[nK];
  }
  std::printf("  reconstructed %ld active Hadamard slices in %.1f ms\n",
              total_slices, ms_since(t_recon));
  if (units.empty()) {
    std::fprintf(stderr, "ERROR: no work units\n");
    return 1;
  }

  std::printf("\n--- FAITHFUL whole-op work (from dump) ---\n");
  std::printf("  Hadamard slices        : %ld\n", total_slices);
  std::printf("  result cells (=ΣMμ)    : %ld\n", total_result_cells);
  std::printf("  current calls (=ΣMμ·nK): %ld\n", total_current_calls);
  std::printf("  strided calls   (=ΣnK)   : %ld   (reduction %.1fx)\n",
              total_strided_calls,
              double(total_current_calls) / double(total_strided_calls));
  std::printf("  packed calls  (=slices): %ld\n", total_slices);
  std::printf("  total flops            : %.3e\n", total_flops);
  auto dump_hist = [](const char* lbl, std::map<long, long> const& h) {
    std::printf("  %-22s: ", lbl);
    int n = 0;
    for (auto const& kv : h) {
      std::printf("%ld:%ld  ", kv.first, kv.second);
      if (++n >= 16) {
        std::printf("...");
        break;
      }
    }
    std::printf("\n");
  };
  dump_hist("P(=|a_1|) dist", P_hist);
  dump_hist("Mμ dist", Mmu_hist);
  dump_hist("nK dist", nK_hist);

  // --- bounded timed sample (stride-sampled to span the distribution) ---
  long n_sample =
      (cli.max_slices <= 0) ? total_slices
                            : std::min(cli.max_slices, total_slices);
  std::vector<WorkUnit> sample;
  sample.reserve(n_sample);
  double sample_flops = 0;
  long sample_current_calls = 0, sample_strided_calls = 0;
  {
    double step = double(total_slices) / double(n_sample);
    for (long s = 0; s < n_sample; ++s) {
      long i = std::min<long>(total_slices - 1, (long)std::llround(s * step));
      sample.push_back(units[i]);
      sample_flops += 2.0 * units[i].P * units[i].Q * units[i].Mmu * units[i].nK;
      sample_current_calls += units[i].Mmu * units[i].nK;
      sample_strided_calls += units[i].nK;
    }
  }
  const double extrap = double(total_slices) / double(n_sample);
  std::printf("\n--- timed sample ---\n");
  std::printf("  sampling %ld / %ld slices (extrapolation x%.2f)\n", n_sample,
              total_slices, extrap);

  std::mt19937_64 rng(cli.seed);
  std::printf("  building arena slice pool ...\n");
  auto t_pool = clock_type::now();
  std::vector<SliceOperand> pool(n_sample);
  for (long s = 0; s < n_sample; ++s) build_slice(pool[s], sample[s], rng);
  std::printf("  pool built in %.1f ms\n", ms_since(t_pool));

  if (cli.check) {
    auto check_mode = [&](const char* name, void (*fn)(SliceOperand&)) {
      double d = 0;
      for (auto& s : pool) {
        zero_C(s);
        fn(s);
        d = std::max(d, max_abs_diff_ref(s));
      }
      std::printf("  check %-13s max_abs_diff=%.3e  %s\n", name, d,
                  d < 1e-9 ? "pass" : "FAIL");
      return d < 1e-9;
    };
    bool ok = true;
    ok &= check_mode("current_gemm", eval_current_gemm);
    ok &= check_mode("current_gemv", eval_current_gemv);
    ok &= check_mode("strided", eval_strided);
    ok &= check_mode("packed", eval_packed);
    if (!ok) {
      std::fprintf(stderr, "correctness FAILED -- aborting timing\n");
      return 2;
    }
  }

  std::printf("\nresults (min/median over %d reps; sample of %ld slices)\n",
              cli.repeats, n_sample);
  double g_cg = 0, g_cv = 0, g_fu = 0, g_pk = 0;
  auto run = [&](const char* name, void (*fn)(SliceOperand&), long calls,
                 double& slot) {
    if (cli.mode != "all" && cli.mode != name) return;
    for (int w = 0; w < cli.warmup; ++w)
      for (auto& s : pool) {
        zero_C(s);
        fn(s);
      }
    std::vector<double> times;
    for (int r = 0; r < cli.repeats; ++r) {
      for (auto& s : pool) zero_C(s);
      auto t0 = clock_type::now();
      for (auto& s : pool) fn(s);
      times.push_back(ms_since(t0));
    }
    std::sort(times.begin(), times.end());
    double mn = times.front(), md = times[times.size() / 2];
    double gf = (sample_flops / 1e9) / (mn / 1e3);
    slot = mn;
    std::printf(
        "  %-13s  min=%8.2f ms  median=%8.2f ms  %7.2f GFLOPS  calls=%ld   "
        "(whole-op est: %8.1f ms)\n",
        name, mn, md, gf, calls, mn * extrap);
  };
  run("current_gemm", eval_current_gemm, sample_current_calls, g_cg);
  run("current_gemv", eval_current_gemv, sample_current_calls, g_cv);
  run("strided", eval_strided, sample_strided_calls, g_fu);
  run("packed", eval_packed, n_sample, g_pk);

  if (cli.mode == "all") {
    std::printf("\n--- speedups (min time) ---\n");
    if (g_fu > 0) {
      std::printf("  strided  vs current_gemm : %.2fx\n", g_cg / g_fu);
      std::printf("  strided  vs current_gemv : %.2fx\n", g_cv / g_fu);
    }
    if (g_pk > 0) {
      std::printf("  packed vs current_gemm : %.2fx\n", g_cg / g_pk);
      std::printf("  packed vs strided        : %.2fx  (pack incl.)\n",
                  g_fu / g_pk);
    }
  }
  std::printf("\n");
  return 0;
}
