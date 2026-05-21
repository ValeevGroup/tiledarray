/// 4d_e: outer-4D x outer-3D with one Hadamard, one contracted, three free.

#include "case_common.h"

#include <cmath>

namespace c = cases;

namespace {

/// Deterministic truncated-exponential inner-size, mean ~10, cap 50.
inline long a_size(long p, long q) {
  unsigned long h =
      (static_cast<unsigned long>(p) * 73ULL +
       static_cast<unsigned long>(q) * 113ULL + 17ULL) * 2654435761ULL;
  double u = static_cast<double>(h & 0x7FFFFFFFUL) /
             static_cast<double>(0x80000000UL);
  double x = -10.0 * std::log(1.0 - u);
  if (x > 50.0) x = 50.0;
  return static_cast<long>(x);
}

}  // namespace

struct Ops {
  c::ToT lhs;
  c::ToT rhs;
};

int main(int argc, char** argv) {
  constexpr int I = 20;
  constexpr int M = 50;
  constexpr int K = 100;

  auto sl = [](long q, long p, long /*m*/, long /*k*/) {
    return TiledArray::Range{a_size(p, q)};
  };
  auto sr = [](long r, long q, long /*m*/) {
    return TiledArray::Range{a_size(q, r)};
  };

  return c::run_case_main_split(
      argc, argv, "4d_e",
      [&](TiledArray::World& w) {
        Ops ops;
        ops.lhs = c::make_tot_4d_jagged(w, I, I, M, K, 1.0, sl);
        ops.rhs = c::make_tot_3d_jagged(w, I, I, M, 100.0, sr);
        return ops;
      },
      [&](TiledArray::World& w) {
        Ops ops;
        ops.lhs = c::make_tot_4d_jagged_slab(w, I, I, M, K, 1.0, sl);
        ops.rhs = c::make_tot_3d_jagged_slab(w, I, I, M, 100.0, sr);
        return ops;
      },
      [&](Ops& ops) {
        return TiledArray::einsum(ops.lhs("q,p,m,k;s"),
                                   ops.rhs("r,q,m;t"),
                                   "p,r,q,k;s,t");
      });
}
