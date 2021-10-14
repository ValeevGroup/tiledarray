#ifndef TILEDARRAY_MATH_LINALG_QR_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_QR_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK
//#include <TiledArray/math/linalg/scalapack/qr.h>
#endif
#include <TiledArray/math/linalg/non-distributed/qr.h>
#include <TiledArray/util/threads.h>

namespace TiledArray::math::linalg {

template <typename ArrayV, bool QOnly = false>
auto householder_qr( const ArrayV& V, TiledRange q_trange = TiledRange(),
                     TiledRange r_trange = TiledRange() ) {
  TA_MAX_THREADS;
#if TILEDARRAY_HAS_SCALAPACK
  if (V.world().size() > 1 && V.elements_range().volume() > 10000000) {
    TA_EXCEPTION("ScaLAPACK Householder QR NYI");
    //return scalapack::householder_qr<ArrayV,QOnly>( V, q_trange, r_trange );
  }
#endif
  return non_distributed::householder_qr<ArrayV,QOnly>( V, q_trange, r_trange );
}

}  // namespace TiledArray::math::linalg

namespace TiledArray {
  using TiledArray::math::linalg::householder_qr;
}  // namespace TiledArray
#endif
