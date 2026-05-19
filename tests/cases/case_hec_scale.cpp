/// hec_scale: A(h,i,j;m,n) * B_plain(h,j,k) -> C(h,i,k;m,n); inner scale.

#include "case_common.h"

namespace c = cases;

struct Ops {
  c::ToT lhs;
  c::Plain rhs;
};

int main(int argc, char** argv) {
  constexpr int N = 56;
  auto sl = [](long /*h*/, long i, long /*j*/) {
    return TiledArray::Range{i, i};
  };
  return c::run_case_main_split(
      argc, argv, "hec_scale",
      [&](TiledArray::World& w) {
        Ops ops;
        ops.lhs = c::make_tot_3d_jagged(w, N, N, N, 1.0, sl);
        ops.rhs = c::make_plain_3d(w, N, N, N, 0.5);
        return ops;
      },
      [&](TiledArray::World& w) {
        Ops ops;
        ops.lhs = c::make_tot_3d_jagged_slab(w, N, N, N, 1.0, sl);
        ops.rhs = c::make_plain_3d(w, N, N, N, 0.5);
        return ops;
      },
      [&](Ops& ops) {
        return TiledArray::einsum(ops.lhs("h,i,j;m,n"), ops.rhs("h,j,k"),
                                  "h,i,k;m,n");
      });
}
