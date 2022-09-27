#ifndef TILEDARRAY_MATH_LINALG_QR_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_QR_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK
#include <TiledArray/math/linalg/scalapack/qr.h>
#endif
#include <TiledArray/math/linalg/basic.h>
#include <TiledArray/math/linalg/non-distributed/qr.h>
#include <TiledArray/util/threads.h>

#include <TiledArray/math/linalg/cholesky.h>

namespace TiledArray::math::linalg {

template <bool QOnly, typename ArrayV>
auto householder_qr(const ArrayV& V, TiledRange q_trange = TiledRange(),
                    TiledRange r_trange = TiledRange()) {
  TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG(
      householder_qr<QOnly>(V, q_trange, r_trange), V);
}

template <bool QOnly, typename ArrayV>
auto cholesky_qr(const ArrayV& V, TiledRange r_trange = TiledRange()) {
  TA_MAX_THREADS;
  // Form Grammian
  ArrayV G;
  G("i,j") = V("k,i").conj() * V("k,j");

  // Obtain Cholesky L and its inverse
  auto [L, Linv] = cholesky_linv<true>(G, r_trange);

  // Q = V * L**-H
  ArrayV Q;
  Q("i,j") = V("i,k") * Linv("j,k").conj();

  if constexpr (not QOnly) {
    // R = L**H
    ArrayV R;
    R("i,j") = L("j,i");
    return std::make_tuple(Q, R);
  } else
    return Q;
}

}  // namespace TiledArray::math::linalg

namespace TiledArray {
using TiledArray::math::linalg::cholesky_qr;
using TiledArray::math::linalg::householder_qr;
}  // namespace TiledArray
#endif
