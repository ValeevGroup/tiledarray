#ifndef TILEDARRAY_MATH_LINALG_SCALAPACK_QR_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SCALAPACK_QR_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK

#include <TiledArray/math/linalg/scalapack/util.h>

#include <scalapackpp/factorizations/geqrf.hpp>
#include <scalapackpp/householder/generate_q_householder.hpp>
#include <scalapackpp/lacpy.hpp>

namespace TiledArray::math::linalg::scalapack {

template <bool QOnly, typename ArrayV>
auto householder_qr( const ArrayV& V, TiledRange q_trange = TiledRange(),
                     TiledRange r_trange = TiledRange(),
                     size_t NB = default_block_size(),
                     size_t MB = default_block_size()) {

  using value_type = typename ArrayV::element_type;

  auto& world = V.world();
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence();  // stage ScaLAPACK execution
  auto V_sca = scalapack::array_to_block_cyclic(V, grid, MB, NB);
  world.gop.fence();  // stage ScaLAPACK execution

  auto [M, N] = V_sca.dims();
  auto K = std::min(M,N);
  auto [V_Mloc, V_Nloc] = V_sca.dist().get_local_dims(M, N);
  auto desc_v = V_sca.dist().descinit_noerror(M, N, V_Mloc);


  std::vector<value_type>
    TAU_local( scalapackpp::local_col_from_desc( K, desc_v ) );

  // Perform QR factorization -> Obtain reflectors + R in UT
  auto info = scalapackpp::pgeqrf( M, N, V_sca.local_mat().data(), 1, 1, desc_v, TAU_local.data() );
  if(info) TA_EXCEPTION("GEQRF FAILED");

  ArrayV R; // Uninitialized R matrix

  if constexpr (not QOnly) {
    BlockCyclicMatrix<value_type> R_sca( world, grid, K, N, MB, NB );
    auto [R_Mloc, R_Nloc] = R_sca.dist().get_local_dims(K, N);
    auto desc_r = R_sca.dist().descinit_noerror(K, N, R_Mloc);

    // Extract R from the upper triangle of V
    R_sca.local_mat().fill(0.);
    scalapackpp::placpy( scalapackpp::Uplo::Upper, K, N, 
      V_sca.local_mat().data(), 1, 1, desc_v,
      R_sca.local_mat().data(), 1, 1, desc_r );
    
    if (r_trange.rank() == 0) {
      // Generate a TRange based on column tiling of V
      auto col_tiling = V.trange().dim(1);
      r_trange = TiledRange( {col_tiling, col_tiling} );
    }

    world.gop.fence();
    R = scalapack::block_cyclic_to_array<ArrayV>( R_sca, r_trange );
    world.gop.fence();
  }

  // Generate Q
  info = scalapackpp::generate_q_householder( M, N, K, V_sca.local_mat().data(), 1, 1, desc_v,
    TAU_local.data() );
  if(info) TA_EXCEPTION("GENQ FAILED");

  if(q_trange.rank() == 0) q_trange = V.trange();
  world.gop.fence();
  auto Q = scalapack::block_cyclic_to_array<ArrayV>( V_sca, q_trange );
  world.gop.fence();

  if constexpr (QOnly) return Q;
  else                 return std::make_tuple( Q, R );
}

} // namespace TiledArray::math::linalg::scalapack

#endif // TILEDARRAY_HAS_SCALAPACK
#endif // HEADER GUARD
