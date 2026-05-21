/// hec_h: A(h,i,j;m,n) * B(h,j,k;m,n) -> C(h,i,k;m,n); inner = (h, h).

#include "case_common.h"

namespace c = cases;

struct Ops {
  c::ToT lhs;
  c::ToT rhs;
};

int main(int argc, char** argv) {
  constexpr int N = 56;
  auto sf = [](long h, long /*o1*/, long /*o2*/) {
    return TiledArray::Range{h, h};
  };
  return c::run_case_main_split(
      argc, argv, "hec_h",
      [&](TiledArray::World& w) {
        Ops ops;
        ops.lhs = c::make_tot_3d_jagged(w, N, N, N, /*offset=*/1.0, sf);
        ops.rhs = c::make_tot_3d_jagged(w, N, N, N, /*offset=*/100.0, sf);
        return ops;
      },
      [&](TiledArray::World& w) {
        Ops ops;
        ops.lhs = c::make_tot_3d_jagged_slab(w, N, N, N, /*offset=*/1.0, sf);
        ops.rhs = c::make_tot_3d_jagged_slab(w, N, N, N, /*offset=*/100.0, sf);
        return ops;
      },
      [&](Ops& ops) {
        return TiledArray::einsum(ops.lhs("h,i,j;m,n"), ops.rhs("h,j,k;m,n"),
                                  "h,i,k;m,n");
      });
}
