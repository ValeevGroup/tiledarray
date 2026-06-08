// opb_strided_arena_dgemm.cpp
// ---------------------------------------------------------------------------
// Standalone tile/BLAS-level benchmark for the op_B ToT x ToT -> ToT
// outer-contraction with an inner OUTER-PRODUCT, on ArenaTensor inner cells.
//
//   op_B:  I(i_2,i_1,μ̃,Κ; a_1) * I(μ̃,i_1,i_2; a_4) -> I(i_2,i_1,Κ; a_1,a_4)
//          Hadamard = {i_1,i_2}, contracted = {μ̃} (=J), ext-left = {Κ}
//          P = |a_1|, Q = |a_4|  depend (jaggedly) on the Hadamard slice.
//
//   Per result cell (i_2,i_1,Κ):
//       C(p,q) = sum_{μ̃} L(i_2,i_1,μ̃,Κ)(p) * R(μ̃,i_1,i_2)(q)
//   i.e. an accumulation over J=|μ̃| rank-1 outer products into a P x Q block.
//
// We compare three ways to evaluate one result cell, ALL reading the SAME
// already-fusable arena slabs (operands built contracted-index-fastest, so the
// J reduced cells are one contiguous, constant-stride run -- the "already
// permuted" precondition; achieving that layout from the real scattered operand
// is a separate cost, not measured here):
//
//   SEQ-gemm : per cell, J tiny PxQ GEMMs with K=1, beta=1 (TA's current
//              per-cell inner-op shape).
//   SEQ-ger  : per cell, J BLAS dger rank-1 updates.
//   FUSED    : per cell, ONE PxQ GEMM with K=J, riding μ̃ into the BLAS K axis
//              by passing the inter-cell slab stride as the leading dimension
//              (zero-copy). #BLAS calls drops from J to 1 per cell.
//
// Work units {P,Q,J} are reconstructed FAITHFULLY from op_dump.txt (jagged P,Q
// per Hadamard slice, real per-cell J after sparsity). The full distribution is
// reported; timing runs over a bounded sample (--max_cells) and is extrapolated
// to the whole op.
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
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace TA = TiledArray;

using Inner = TA::ArenaTensor<double>;  // default btas::zb range, matches MPQC
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
// op_dump.txt parser (subset of cck_bottleneck_bench.cpp -- structure only)
// ===========================================================================

struct InnerRangeKey {
  std::vector<long> lo, hi;
};

struct TileSpec {
  long outer_vol = 0;
  std::vector<InnerRangeKey> distinct;
  std::vector<int> cell_labels;  // size outer_vol; -1=empty else index distinct
};

struct ArraySpec {
  int outer_rank = 0;
  std::vector<std::vector<long>> dim_tile_bounds;  // per outer dim
  std::map<std::size_t, TileSpec> tiles;           // ord -> tile
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
  int label;  // -1 = E
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
  // Tiles with a single distinct range and no emitted RLE are fully uniform.
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
// Fast cell-presence / inner-range queries over an ArraySpec
// ===========================================================================

// A flattened tiled-range for one outer dim: element -> tile index, and the
// per-tile element extent, all from the dump's tile_bounds.
struct DimTiling {
  std::vector<long> bounds;          // size ntiles+1
  std::vector<int> elem_to_tile;     // size n_elem
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

// Indexable view of one operand: maps a full element multi-index (in this
// operand's dim order) to (present?, inner-range hi extents).
struct OperandIndex {
  ArraySpec const* spec = nullptr;
  std::vector<DimTiling> dims;
  std::vector<std::size_t> tile_strides;  // row-major over tile grid

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

  // Returns the cell label (>=0 present, -1 empty/absent) and, if present and
  // out_hi != nullptr, fills the inner-range upper bounds.
  int query(std::vector<long> const& idx,
            std::vector<long>* out_hi = nullptr) const {
    int rank = spec->outer_rank;
    std::size_t tord = 0;
    // tile ordinal
    std::vector<long> tile_lo(rank);
    std::vector<long> tile_ext(rank);
    for (int d = 0; d < rank; ++d) {
      int t = dims[d].elem_to_tile[idx[d]];
      tord += static_cast<std::size_t>(t) * tile_strides[d];
      tile_lo[d] = dims[d].bounds[t];
      tile_ext[d] = dims[d].bounds[t + 1] - dims[d].bounds[t];
    }
    auto it = spec->tiles.find(tord);
    if (it == spec->tiles.end()) return -1;
    auto const& ts = it->second;
    // within-tile row-major ordinal
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
// Annotation role classification
// ===========================================================================

// Split "a,b,c;x,y" -> outer={a,b,c}, inner={x,y}.
static void split_annot(std::string const& annot, std::vector<std::string>& outer,
                        std::vector<std::string>& inner) {
  outer.clear();
  inner.clear();
  auto semi = annot.find(';');
  std::string o = annot.substr(0, semi);
  std::string in = (semi == std::string::npos) ? "" : annot.substr(semi + 1);
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
  split(o, outer);
  split(in, inner);
}

static int find_idx(std::vector<std::string> const& v, std::string const& s) {
  for (int i = 0; i < (int)v.size(); ++i)
    if (v[i] == s) return i;
  return -1;
}

// ===========================================================================
// Reconstructed work unit
// ===========================================================================

struct WorkUnit {
  long P, Q, J;
};

// ===========================================================================
// Arena slab operand for one work unit (built once, ready-to-fuse)
// ===========================================================================

struct CellOperand {
  Outer Lslab;   // outer {J}, inner {P}: J cells of size P, μ̃-fastest
  Outer Rslab;   // outer {J}, inner {Q}
  Outer Ccell;   // outer {1}, inner {P,Q}
  long P, Q, J;
  std::ptrdiff_t sL, sR;  // inter-cell strides (elements); ld for strided GEMM
  double* Cdata;
  std::vector<double> ref;  // golden P*Q reference
};

static void build_cell_operand(CellOperand& op, WorkUnit const& wu,
                               std::mt19937_64& rng) {
  op.P = wu.P;
  op.Q = wu.Q;
  op.J = wu.J;
  const long P = wu.P, Q = wu.Q, J = wu.J;
  op.Lslab = TA::detail::arena_outer_init<Outer>(
      TA::Range{static_cast<std::size_t>(J)}, 1,
      [P](std::size_t) { return InnerRange{P}; }, /*zero_init=*/false);
  op.Rslab = TA::detail::arena_outer_init<Outer>(
      TA::Range{static_cast<std::size_t>(J)}, 1,
      [Q](std::size_t) { return InnerRange{Q}; }, /*zero_init=*/false);
  op.Ccell = TA::detail::arena_outer_init<Outer>(
      TA::Range{1}, 1,
      [P, Q](std::size_t) { return InnerRange{P, Q}; }, /*zero_init=*/true);

  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (long j = 0; j < J; ++j) {
    double* l = op.Lslab.data()[j].data();
    for (long p = 0; p < P; ++p) l[p] = dist(rng);
    double* r = op.Rslab.data()[j].data();
    for (long q = 0; q < Q; ++q) r[q] = dist(rng);
  }
  op.Cdata = op.Ccell.data()[0].data();
  op.sL = (J > 1) ? (op.Lslab.data()[1].data() - op.Lslab.data()[0].data()) : P;
  op.sR = (J > 1) ? (op.Rslab.data()[1].data() - op.Rslab.data()[0].data()) : Q;

  // Golden reference: C[p,q] = sum_j L_j[p] * R_j[q].
  op.ref.assign(static_cast<std::size_t>(P) * Q, 0.0);
  for (long j = 0; j < J; ++j) {
    const double* l = op.Lslab.data()[j].data();
    const double* r = op.Rslab.data()[j].data();
    for (long p = 0; p < P; ++p)
      for (long q = 0; q < Q; ++q) op.ref[p * Q + q] += l[p] * r[q];
  }
}

// ===========================================================================
// The three evaluation paths (one result cell)
// ===========================================================================

namespace tamb = TiledArray::math::blas;  // TA's row-major gemm wrapper
using integer = tamb::integer;

static void zero_C(CellOperand& op) {
  std::memset(op.Cdata, 0, sizeof(double) * op.P * op.Q);
}

// SEQ-gemm: J tiny PxQ GEMMs, K=1, beta=1 (C pre-zeroed by caller).
static void eval_seq_gemm(CellOperand& op) {
  const integer P = op.P, Q = op.Q;
  for (long j = 0; j < op.J; ++j) {
    const double* Lj = op.Lslab.data()[j].data();
    const double* Rj = op.Rslab.data()[j].data();
    tamb::gemm(tamb::Op::NoTrans, tamb::Op::NoTrans, /*M=*/P, /*N=*/Q, /*K=*/1,
               1.0, /*A=*/Lj, /*lda=*/1, /*B=*/Rj, /*ldb=*/Q, /*beta=*/1.0,
               /*C=*/op.Cdata, /*ldc=*/Q);
  }
}

// SEQ-ger: J BLAS dger rank-1 updates (C pre-zeroed by caller).
static void eval_seq_ger(CellOperand& op) {
  const integer P = op.P, Q = op.Q;
  for (long j = 0; j < op.J; ++j) {
    const double* Lj = op.Lslab.data()[j].data();
    const double* Rj = op.Rslab.data()[j].data();
    ::blas::ger(::blas::Layout::RowMajor, P, Q, 1.0, Lj, 1, Rj, 1, op.Cdata, Q);
  }
}

// FUSED: one PxQ GEMM, K=J, μ̃ ridden into K via the inter-cell slab stride as
// leading dimension. C(PxQ) = A~(PxJ) . B~(JxQ), beta=0 (full sum in one GEMM).
//   A~[p,j] = Lslab_j[p] -> column-major PxJ (col stride sL) = row-major JxP^T
//             => pass op_a = Trans on stored JxP (lda=sL)
//   B~[j,q] = Rslab_j[q] -> row-major JxQ (row stride sR)  => op_b = NoTrans
static void eval_strided(CellOperand& op) {
  const integer P = op.P, Q = op.Q, J = op.J;
  tamb::gemm(tamb::Op::Trans, tamb::Op::NoTrans, /*M=*/P, /*N=*/Q, /*K=*/J, 1.0,
             /*A=*/op.Lslab.data()[0].data(), /*lda=*/op.sL,
             /*B=*/op.Rslab.data()[0].data(), /*ldb=*/op.sR, /*beta=*/0.0,
             /*C=*/op.Cdata, /*ldc=*/Q);
}

static double max_abs_diff_ref(CellOperand const& op) {
  double d = 0;
  for (std::size_t e = 0; e < op.ref.size(); ++e)
    d = std::max(d, std::abs(op.Cdata[e] - op.ref[e]));
  return d;
}

// ===========================================================================
// CLI
// ===========================================================================

struct Cli {
  std::string dump =
      "/Users/zhihaodeng/packages/mpqc4/agent/experiments/C6H14/profile/"
      "op_dump.txt";
  long max_cells = 20000;  // timed-pool cap; 0 = all
  int repeats = 5;
  int warmup = 1;
  int seed = 42;
  std::string mode = "all";  // seq_gemm | seq_ger | strided | all
  bool check = true;
};

static void usage() {
  std::fprintf(stderr,
               "opb_strided_arena_dgemm\n"
               "  --dump PATH        op_dump.txt path\n"
               "  --max_cells N      timed-pool cap (0=all)  (default 20000)\n"
               "  --repeats R        timed reps             (default 5)\n"
               "  --warmup W         untimed warmup reps     (default 1)\n"
               "  --seed S           RNG seed                (default 42)\n"
               "  --mode M           seq_gemm|seq_ger|strided|all (default all)\n"
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
    else if (a == "--max_cells")
      c.max_cells = std::stol(need());
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

  std::printf("=== op_B strided-vs-sequential arena DGEMM bench ===\n");
  std::printf("dump=%s\n", cli.dump.c_str());

  auto dump = parse_dump(cli.dump);

  // Select op_B: result annotation has 2 inner indices.
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
    if (n_inner(e.annot_out) == 2) {
      op = &e;
      break;
    }
  if (!op) {
    std::fprintf(stderr, "ERROR: no op_B (2 inner indices) in dump\n");
    return 1;
  }

  // --- classify outer index roles from annotations ---
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
    for (std::size_t i = 0; i < v.size(); ++i)
      s += (i ? "," : "") + v[i];
    return s;
  };
  std::printf("  annot_left  = %s\n", op->annot_left.c_str());
  std::printf("  annot_right = %s\n", op->annot_right.c_str());
  std::printf("  annot_out   = %s\n", op->annot_out.c_str());
  std::printf("  hadamard={%s} contracted={%s} ext_left={%s} ext_right={%s}\n",
              join(hadamard).c_str(), join(contracted).c_str(),
              join(ext_left).c_str(), join(ext_right).c_str());

  if (contracted.size() != 1) {
    std::fprintf(stderr,
                 "ERROR: this bench handles exactly one contracted outer index "
                 "(got %zu)\n",
                 contracted.size());
    return 1;
  }
  if (!ext_right.empty()) {
    std::fprintf(stderr,
                 "ERROR: this bench assumes no pure right-external outer index "
                 "(got %zu)\n",
                 ext_right.size());
    return 1;
  }

  OperandIndex Lidx, Ridx, Cidx;
  Lidx.build(op->L);
  Ridx.build(op->R);
  Cidx.build(op->C);

  const std::string mu = contracted[0];
  const int muL = find_idx(Lo, mu);  // μ̃ position in L dim order
  const int muR = find_idx(Ro, mu);  // μ̃ position in R dim order
  const long Jext = Lidx.dim_extent(muL);
  std::printf("  contracted '%s' extent J=%ld\n", mu.c_str(), Jext);

  // Maps from index name -> dim position for assembling cross-operand queries.
  std::unordered_map<std::string, int> Cpos;
  for (int d = 0; d < (int)Co.size(); ++d) Cpos[Co[d]] = d;

  // --- faithful work-unit reconstruction: one per nonzero C cell ---
  std::printf("\nreconstructing work units from dump ...\n");
  auto t_recon = clock_type::now();
  std::vector<WorkUnit> units;
  long total_cells = 0, total_calls_seq = 0;
  double total_flops = 0;
  std::map<long, long> J_hist;  // J -> count
  std::map<long, long> PQ_hist; // P -> count (P==Q here, report P)

  std::vector<long> cidx(Co.size()), lq(Lo.size()), rq(Ro.size()), hi;
  for (auto const& kv : op->C.tiles) {
    std::size_t ord = kv.first;
    TileSpec const& ts = kv.second;
    // decode tile ord -> tile multi-index over C tile grid
    std::vector<long> tmi(Cidx.rank());
    {
      std::size_t rem = ord;
      for (int d = 0; d < Cidx.rank(); ++d) {
        tmi[d] = rem / Cidx.tile_strides[d];
        rem %= Cidx.tile_strides[d];
      }
    }
    std::vector<long> tlo(Cidx.rank()), text(Cidx.rank());
    for (int d = 0; d < Cidx.rank(); ++d) {
      tlo[d] = Cidx.dims[d].bounds[tmi[d]];
      text[d] = Cidx.dims[d].bounds[tmi[d] + 1] - tlo[d];
    }
    for (long k = 0; k < ts.outer_vol; ++k) {
      int lbl = ts.cell_labels[k];
      if (lbl < 0) continue;
      // within-tile k -> element multi-index
      std::size_t rem = k;
      for (int d = Cidx.rank() - 1; d >= 0; --d) {
        cidx[d] = tlo[d] + (rem % text[d]);
        rem /= text[d];
      }
      const auto& cr = ts.distinct[lbl].hi;  // [P, Q]
      const long P = cr[0], Q = cr.size() > 1 ? cr[1] : cr[0];
      // assemble L and R query templates from this C cell's hadamard+ext values
      for (int d = 0; d < (int)Lo.size(); ++d) {
        auto it = Cpos.find(Lo[d]);
        if (it != Cpos.end()) lq[d] = cidx[it->second];
      }
      for (int d = 0; d < (int)Ro.size(); ++d) {
        auto it = Cpos.find(Ro[d]);
        if (it != Cpos.end()) rq[d] = cidx[it->second];
      }
      long J = 0;
      for (long m = 0; m < Jext; ++m) {
        lq[muL] = m;
        rq[muR] = m;
        if (Lidx.query(lq) >= 0 && Ridx.query(rq) >= 0) ++J;
      }
      if (J == 0) continue;
      units.push_back({P, Q, J});
      ++total_cells;
      total_calls_seq += J;
      total_flops += 2.0 * P * Q * J;
      ++J_hist[J];
      ++PQ_hist[P];
    }
  }
  std::printf("  reconstructed %ld nonzero result cells in %.1f ms\n",
              total_cells, ms_since(t_recon));
  if (units.empty()) {
    std::fprintf(stderr, "ERROR: no work units\n");
    return 1;
  }

  // --- faithful totals (whole op) ---
  std::printf("\n--- FAITHFUL whole-op work (from dump) ---\n");
  std::printf("  result cells          : %ld\n", total_cells);
  std::printf("  BLAS calls  SEQ (=ΣJ) : %ld\n", total_calls_seq);
  std::printf("  BLAS calls  FUSED     : %ld   (reduction %.1fx)\n",
              total_cells, double(total_calls_seq) / double(total_cells));
  std::printf("  total flops           : %.3e\n", total_flops);
  std::printf("  J distribution        : ");
  for (auto const& kv : J_hist) std::printf("J=%ld:%ld  ", kv.first, kv.second);
  std::printf("\n  P(=Q) distribution    : ");
  for (auto const& kv : PQ_hist) std::printf("P=%ld:%ld  ", kv.first, kv.second);
  std::printf("\n");

  // --- bounded timed sample ---
  long n_sample =
      (cli.max_cells <= 0) ? total_cells : std::min(cli.max_cells, total_cells);
  // even, deterministic stride sample so the timed subset matches the full
  // (P,Q,J) distribution rather than just the first tile.
  std::vector<WorkUnit> sample;
  sample.reserve(n_sample);
  double sample_flops = 0;
  long sample_calls_seq = 0;
  {
    double step = double(total_cells) / double(n_sample);
    for (long s = 0; s < n_sample; ++s) {
      long i = std::min<long>(total_cells - 1, (long)std::llround(s * step));
      sample.push_back(units[i]);
      sample_flops += 2.0 * units[i].P * units[i].Q * units[i].J;
      sample_calls_seq += units[i].J;
    }
  }
  const double extrap = double(total_cells) / double(n_sample);
  std::printf("\n--- timed sample ---\n");
  std::printf("  sampling %ld / %ld cells (extrapolation x%.2f)\n", n_sample,
              total_cells, extrap);

  // build arena operand pool (untimed)
  std::mt19937_64 rng(cli.seed);
  std::printf("  building arena operand pool ...\n");
  auto t_pool = clock_type::now();
  std::vector<CellOperand> pool(n_sample);
  for (long s = 0; s < n_sample; ++s) build_cell_operand(pool[s], sample[s], rng);
  std::printf("  pool built in %.1f ms\n", ms_since(t_pool));

  // correctness
  if (cli.check) {
    auto check_mode = [&](const char* name, void (*fn)(CellOperand&)) {
      double d = 0;
      for (auto& op2 : pool) {
        zero_C(op2);
        fn(op2);
        d = std::max(d, max_abs_diff_ref(op2));
      }
      std::printf("  check %-9s max_abs_diff=%.3e  %s\n", name, d,
                  d < 1e-9 ? "pass" : "FAIL");
      return d < 1e-9;
    };
    bool ok = true;
    ok &= check_mode("seq_gemm", eval_seq_gemm);
    ok &= check_mode("seq_ger", eval_seq_ger);
    ok &= check_mode("strided", eval_strided);
    if (!ok) {
      std::fprintf(stderr, "correctness FAILED -- aborting timing\n");
      return 2;
    }
  }

  // timing
  std::printf("\nresults (min/median over %d reps; sample of %ld cells)\n",
              cli.repeats, n_sample);
  static double g_seq_gemm = 0, g_seq_ger = 0, g_strided = 0;
  auto run_capture = [&](const char* name, void (*fn)(CellOperand&), long calls,
                         double& slot) {
    if (cli.mode != "all" && cli.mode != name) return;
    for (int w = 0; w < cli.warmup; ++w)
      for (auto& op2 : pool) {
        zero_C(op2);
        fn(op2);
      }
    std::vector<double> times;
    for (int r = 0; r < cli.repeats; ++r) {
      for (auto& op2 : pool) zero_C(op2);
      auto t0 = clock_type::now();
      for (auto& op2 : pool) fn(op2);
      times.push_back(ms_since(t0));
    }
    std::sort(times.begin(), times.end());
    double mn = times.front(), md = times[times.size() / 2];
    double gf = (sample_flops / 1e9) / (mn / 1e3);
    slot = mn;
    std::printf(
        "  %-9s  min=%8.2f ms  median=%8.2f ms  %7.2f GFLOPS  calls=%ld   "
        "(whole-op est: %8.1f ms)\n",
        name, mn, md, gf, calls, mn * extrap);
  };
  run_capture("seq_gemm", eval_seq_gemm, sample_calls_seq, g_seq_gemm);
  run_capture("seq_ger", eval_seq_ger, sample_calls_seq, g_seq_ger);
  run_capture("strided", eval_strided, n_sample, g_strided);

  if (cli.mode == "all") {
    std::printf("\n--- speedups (min time) ---\n");
    if (g_strided > 0) {
      std::printf("  strided vs seq_gemm : %.2fx\n", g_seq_gemm / g_strided);
      std::printf("  strided vs seq_ger  : %.2fx\n", g_seq_ger / g_strided);
    }
  }
  std::printf("\n");
  return 0;
}
