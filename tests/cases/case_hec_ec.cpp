/// hec_ec: A(h,i,j;m,p) * B(h,j,k;p,n) -> C(h,i,k;m,n); inner contracts p.

#include "case_common.h"

namespace c = cases;

struct Ops {
  c::ToT lhs;
  c::ToT rhs;
};

int main(int argc, char** argv) {
  constexpr int N = 60;
  auto sl = [](long /*h*/, long i, long j) {
    return TiledArray::Range{i, j};
  };
  auto sr = [](long /*h*/, long j, long k) {
    return TiledArray::Range{j, k};
  };
  return c::run_case_main_split(
      argc, argv, "hec_ec",
      [&](TiledArray::World& w) {
        Ops ops;
        ops.lhs = c::make_tot_3d_jagged(w, N, N, N, 1.0, sl);
        ops.rhs = c::make_tot_3d_jagged(w, N, N, N, 100.0, sr);
        return ops;
      },
      [&](TiledArray::World& w) {
        Ops ops;
        ops.lhs = c::make_tot_3d_jagged_slab(w, N, N, N, 1.0, sl);
        ops.rhs = c::make_tot_3d_jagged_slab(w, N, N, N, 100.0, sr);
        return ops;
      },
      [&](Ops& ops) {
        return TiledArray::einsum(ops.lhs("h,i,j;m,p"), ops.rhs("h,j,k;p,n"),
                                  "h,i,k;m,n");
      });
}
