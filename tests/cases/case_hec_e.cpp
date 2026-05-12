/// hec_e: A(h,i,j;m) * B(h,j,k;n) -> C(h,i,k;m,n); inner outer-product (i, k).

#include "case_common.h"

namespace c = cases;

struct Ops {
  c::ToT lhs;
  c::ToT rhs;
};

int main(int argc, char** argv) {
  constexpr int N = 30;
  auto sl = [](long /*h*/, long i, long /*j*/) {
    return TiledArray::Range{i};
  };
  auto sr = [](long /*h*/, long /*j*/, long k) {
    return TiledArray::Range{k};
  };
  return c::run_case_main_split(
      argc, argv, "hec_e",
      [&](TiledArray::World& w) {
        const int H = N * c::g_h_scale();
        Ops ops;
        ops.lhs = c::make_tot_3d_jagged(w, H, N, N, 1.0, sl);
        ops.rhs = c::make_tot_3d_jagged(w, H, N, N, 100.0, sr);
        return ops;
      },
      [&](TiledArray::World& w) {
        const int H = N * c::g_h_scale();
        Ops ops;
        ops.lhs = c::make_tot_3d_jagged_slab(w, H, N, N, 1.0, sl);
        ops.rhs = c::make_tot_3d_jagged_slab(w, H, N, N, 100.0, sr);
        return ops;
      },
      [&](Ops& ops) {
        return TiledArray::einsum(ops.lhs("h,i,j;m"), ops.rhs("h,j,k;n"),
                                  "h,i,k;m,n");
      });
}
