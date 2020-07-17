/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020 Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  David Williams-Young
 *  Computational Research Division, Lawrence Berkeley National Laboratory
 *
 *  lu.h
 *  Created:    19 June, 2020
 *
 */
#ifndef TILEDARRAY_MATH_SCALAPACK_LU_H__INCLUDED
#define TILEDARRAY_MATH_SCALAPACK_LU_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK

#include <TiledArray/conversions/block_cyclic.h>
#include <TiledArray/math/scalapack/util.h>

#include <scalapackpp/factorizations/getrf.hpp>
#include <scalapackpp/linear_systems/gesv.hpp>
#include <scalapackpp/matrix_inverse/getri.hpp>

namespace TiledArray {

/**
 *  @brief Solve a linear system via LU factorization
 */
template <typename ArrayA, typename ArrayB>
auto lu_solve( const ArrayA& A, const ArrayB& B, size_t NB = 128, size_t MB = 128,
  TiledRange x_trange = TiledRange() ) {

  using value_type = typename ArrayA::element_type;
  static_assert(std::is_same_v<value_type,typename ArrayB::element_type>);

  auto& world = A.world();
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence(); // stage ScaLAPACK execution
  auto A_sca = array_to_block_cyclic( A, grid, MB, NB );
  auto B_sca = array_to_block_cyclic( B, grid, MB, NB );
  world.gop.fence(); // stage ScaLAPACK execution

  auto [M, N]      = A_sca.dims();
  if( M != N )
    throw std::runtime_error("A must be square for LU Solve");
  auto [B_N, NRHS] = B_sca.dims();
  if( B_N != N )
    throw std::runtime_error("A and B dims must agree");

  auto [A_Mloc, A_Nloc] = A_sca.dist().get_local_dims(N, N);
  auto desc_a = A_sca.dist().descinit_noerror(N, N, A_Mloc);

  auto [B_Mloc, B_Nloc] = B_sca.dist().get_local_dims(N, NRHS);
  auto desc_b = B_sca.dist().descinit_noerror(N, NRHS, B_Mloc);

  std::vector<scalapackpp::scalapack_int> IPIV( A_Mloc + MB );

  auto info = scalapackpp::pgesv( N, NRHS,
    A_sca.local_mat().data(), 1, 1, desc_a, IPIV.data(),
    B_sca.local_mat().data(), 1, 1, desc_b );
  if (info) throw std::runtime_error("LU Solve Failed");

  if( x_trange.rank() == 0 ) x_trange = B.trange();

  world.gop.fence();
  auto X = block_cyclic_to_array<ArrayB>( B_sca, x_trange );
  world.gop.fence();

  return X;

}

/**
 *  @brief Invert a matrix via LU
 */
template <typename Array>
auto lu_inv( const Array& A, size_t NB = 128, size_t MB = 128,
  TiledRange ainv_trange = TiledRange() ) {

  using value_type = typename Array::element_type;

  auto& world = A.world();
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence(); // stage ScaLAPACK execution
  auto A_sca = array_to_block_cyclic( A, grid, MB, NB );
  world.gop.fence(); // stage ScaLAPACK execution

  auto [M, N]      = A_sca.dims();
  if( M != N )
    throw std::runtime_error("A must be square for LU Inverse");

  auto [A_Mloc, A_Nloc] = A_sca.dist().get_local_dims(N, N);
  auto desc_a = A_sca.dist().descinit_noerror(N, N, A_Mloc);


  std::vector<scalapackpp::scalapack_int> IPIV( A_Mloc + MB );

  {
  auto info = scalapackpp::pgetrf( N, N,
    A_sca.local_mat().data(), 1, 1, desc_a, IPIV.data() );
  if (info) throw std::runtime_error("LU Failed");
  }

  {
  auto info = scalapackpp::pgetri( N, 
    A_sca.local_mat().data(), 1, 1, desc_a, IPIV.data() );
  if (info) throw std::runtime_error("LU Inverse Failed");
  }

  if( ainv_trange.rank() == 0 ) ainv_trange = A.trange();

  world.gop.fence();
  auto Ainv = block_cyclic_to_array<Array>( A_sca, ainv_trange );
  world.gop.fence();

  return Ainv;

}





} // namespace TiledArray

#endif // TILEDARRAY_HAS_SCALAPACK
#endif // TILEDARRAY_MATH_SCALAPACK_H__INCLUDED


