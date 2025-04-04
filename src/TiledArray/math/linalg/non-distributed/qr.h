#ifndef TILEDARRAY_MATH_LINALG_NON_DISTRIBUTED_QR_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_NON_DISTRIBUTED_QR_H__INCLUDED

#include <TiledArray/config.h>

#include <TiledArray/conversions/eigen.h>
#include <TiledArray/math/linalg/rank-local.h>
#include <TiledArray/math/linalg/util.h>

namespace TiledArray::math::linalg::non_distributed {

template <bool QOnly, typename ArrayV>
auto householder_qr(const ArrayV& V, TiledRange q_trange = TiledRange(),
                    TiledRange r_trange = TiledRange()) {
  (void)detail::array_traits<ArrayV>{};
  auto& world = V.world();
  auto V_eig = detail::make_matrix(V);
  decltype(V_eig) R_eig;
  TA_LAPACK_ON_RANK_ZERO(householder_qr<QOnly>, world, V_eig, R_eig);
  world.gop.broadcast_serializable(V_eig, 0);
  if (q_trange.rank() == 0) q_trange = V.trange();
  auto Q = eigen_to_array<ArrayV>(world, q_trange, V_eig);
  if constexpr (not QOnly) {
    world.gop.broadcast_serializable(R_eig, 0);
    if (r_trange.rank() == 0) {
      // Generate a TRange based on column tiling of V
      auto col_tiling = V.trange().dim(1);
      r_trange = TiledRange({col_tiling, col_tiling});
    }
    auto R = eigen_to_array<ArrayV>(world, r_trange, R_eig);
    return std::make_tuple(Q, R);
  } else {
    return Q;
  }
}

template <typename ArrayA, typename ArrayB, typename T = ArrayB::numeric_type>
auto qr_solve(const ArrayA& A, const ArrayB& B,
              const TiledArray::detail::real_t<T> cond = 1e8,
              TiledRange x_trange = TiledRange()) {
  (void)detail::array_traits<ArrayB>{};
  auto& world = B.world();
  auto A_eig = detail::make_matrix(A);
  auto B_eig = detail::make_matrix(B);
  TA_LAPACK_ON_RANK_ZERO(qr_solve, world, A_eig, B_eig, cond);
  world.gop.broadcast_serializable(A_eig, 0);
  world.gop.broadcast_serializable(B_eig, 0);
  if (x_trange.rank() == 0) x_trange = B.trange();
  auto X = eigen_to_array<ArrayB>(world, x_trange, B_eig);
  return X;
}

}  // namespace TiledArray::math::linalg::non_distributed

#endif
