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
 *  chol.h
 *  Created:    8 June, 2020
 *
 */
#ifndef TILEDARRAY_MATH_SCALAPACK_CHOL_H__INCLUDED
#define TILEDARRAY_MATH_SCALAPACK_CHOL_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK

#include <TiledArray/conversions/block_cyclic.h>
#include <scalapackpp/factorizations/potrf.hpp>
#include <scalapackpp/matrix_inverse/trtri.hpp>
#include <scalapackpp/linear_systems/posv.hpp>
#include <scalapackpp/linear_systems/trtrs.hpp>

namespace TiledArray {

namespace detail {

template <typename T>
void scalapack_zero_triangle( 
  blacspp::Triangle tri, BlockCyclicMatrix<T>& A, bool zero_diag = false 
) {

  auto zero_el = [&]( size_t I, size_t J ) {
    if( A.dist().i_own(I,J) ) {
      auto [i,j] = A.dist().local_indx(I,J);
      A.local_mat()(i,j) = 0.;
    }
  };

  auto [M,N] = A.dims();

  // Zero the lower triangle
  if( tri == blacspp::Triangle::Lower ) {

    if( zero_diag )
      for( size_t j = 0; j < N; ++j )
      for( size_t i = j; i < M; ++i )
        zero_el( i,j );
    else
      for( size_t j = 0;   j < N; ++j )
      for( size_t i = j+1; i < M; ++i )
        zero_el( i,j );

  // Zero the upper triangle
  } else {

    if( zero_diag )
      for( size_t j = 0; j < N;  ++j )
      for( size_t i = 0; i <= std::min(j,M); ++i )
        zero_el( i,j );
    else
      for( size_t j = 0; j < N; ++j )
      for( size_t i = 0; i < std::min(j,M); ++i )
        zero_el( i,j );

  }
}

}

/**
 *  @brief Compute the Cholesky factorization of a HPD rank-2 tensor
 *
 *  A(i,j) = L(i,k) * conj(L(j,k))
 *
 *  Example Usage:
 *
 *  auto L = cholesky(A, ...)
 *
 *  @tparam Array Input array type, must be convertable to BlockCyclicMatrix
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] NB          ScaLAPACK blocking factor. Defaults to 128
 *  @param[in] l_trange    TiledRange for resulting Cholesky factor. If left empty,
 *                         will default to array.trange()
 *
 *  @returns The lower triangular Cholesky factor L in TA format
 */
template <typename Array>
auto cholesky( const Array& A, size_t NB = 128, TiledRange l_trange = TiledRange() ) {

  using value_type = typename Array::element_type;

  auto& world = A.world();
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence(); // stage ScaLAPACK execution
  auto matrix = array_to_block_cyclic( A, grid, NB, NB );
  world.gop.fence(); // stage ScaLAPACK execution

  auto [M, N] = matrix.dims();
  if( M != N )
    throw std::runtime_error("Matrix must be square for Cholesky");

  auto [Mloc, Nloc] = matrix.dist().get_local_dims(N, N);
  auto desc = matrix.dist().descinit_noerror(N, N, Mloc);

  auto info = scalapackpp::ppotrf( blacspp::Triangle::Lower, N,
    matrix.local_mat().data(), 1, 1, desc );
  if (info) throw std::runtime_error("Cholesky Failed");

  // Zero out the upper triangle
  detail::scalapack_zero_triangle( blacspp::Triangle::Upper, matrix );

  if( l_trange.rank() == 0 ) l_trange = A.trange();

  world.gop.fence();
  auto L = block_cyclic_to_array<Array>( matrix, l_trange );
  world.gop.fence();


  return L;

}







/**
 *  @brief Compute the inverse of the Cholesky factor of an HPD rank-2 tensor.
 *  Optinally return the Cholesky factor itself
 *
 *  A(i,j) = L(i,k) * conj(L(j,k)) -> compute Linv
 *
 *  Example Usage:
 *
 *  auto Linv     = cholesky_Linv(A, ...)
 *  auto [L,Linv] = cholesky_Linv<decltype(A),true>(A, ...)
 *
 *  @tparam Array Input array type, must be convertable to BlockCyclicMatrix
 *  @tparam RetL  Whether or not to return the cholesky factor
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] NB          ScaLAPACK blocking factor. Defaults to 128
 *  @param[in] l_trange    TiledRange for resulting inverse Cholesky factor. 
 *                         If left empty, will default to array.trange()
 *
 *  @returns The inverse lower triangular Cholesky factor in TA format
 */
template <typename Array, bool RetL = false>
auto cholesky_linv( const Array& A, size_t NB = 128, TiledRange l_trange = TiledRange() ) {

  using value_type = typename Array::element_type;

  auto& world = A.world();
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence(); // stage ScaLAPACK execution
  auto matrix = array_to_block_cyclic( A, grid, NB, NB );
  world.gop.fence(); // stage ScaLAPACK execution

  auto [M, N] = matrix.dims();
  if( M != N )
    throw std::runtime_error("Matrix must be square for Cholesky");

  auto [Mloc, Nloc] = matrix.dist().get_local_dims(N, N);
  auto desc = matrix.dist().descinit_noerror(N, N, Mloc);

  auto info = scalapackpp::ppotrf( blacspp::Triangle::Lower, N,
    matrix.local_mat().data(), 1, 1, desc );
  if (info) throw std::runtime_error("Cholesky Failed");

  // Zero out the upper triangle
  detail::scalapack_zero_triangle( blacspp::Triangle::Upper, matrix );

  // Copy L if needed
  std::shared_ptr<BlockCyclicMatrix<value_type>> L_sca = nullptr;
  if constexpr (RetL) {
    L_sca = std::make_shared<BlockCyclicMatrix<value_type>>(
      world, grid, N, N, NB, NB
    );
    L_sca->local_mat() = matrix.local_mat();
  }

  // Compute inverse
  info = scalapackpp::ptrtri( blacspp::Triangle::Lower, 
    blacspp::Diagonal::NonUnit, N, matrix.local_mat().data(), 1, 1, desc );
  if (info) throw std::runtime_error("TRTRI Failed");


  if( l_trange.rank() == 0 ) l_trange = A.trange();

  world.gop.fence();
  auto Linv = block_cyclic_to_array<Array>( matrix, l_trange );
  world.gop.fence();


  if constexpr (RetL) {
    auto L = block_cyclic_to_array<Array>( *L_sca, l_trange);
    world.gop.fence();
    return std::tuple( L, Linv );
  } else {
    return Linv;
  }

}


template <typename Array>
auto cholesky_solve( const Array& A, const Array& B, size_t NB = 128, 
  TiledRange x_trange = TiledRange() ) {

  auto& world = A.world();
  /*
  if( world != B.world() ) {
    throw std::runtime_error("A and B must be distributed on same MADWorld context");
  }
  */
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence(); // stage ScaLAPACK execution
  auto A_sca = array_to_block_cyclic( A, grid, NB, NB );
  auto B_sca = array_to_block_cyclic( B, grid, NB, NB );
  world.gop.fence(); // stage ScaLAPACK execution

  auto [M, N] = A_sca.dims();
  if( M != N )
    throw std::runtime_error("A must be square for Cholesky Solve");

  auto [B_N, NRHS] = B_sca.dims();
  if( B_N != N )
    throw std::runtime_error("A and B dims must agree");


  scalapackpp::scalapack_desc desc_a, desc_b;
  {
  auto [Mloc, Nloc] = A_sca.dist().get_local_dims(N, N);
  desc_a = A_sca.dist().descinit_noerror(N, N, Mloc);
  }

  {
  auto [Mloc, Nloc] = B_sca.dist().get_local_dims(N, NRHS);
  desc_b = B_sca.dist().descinit_noerror(N, NRHS, Mloc);
  }

  auto info = scalapackpp::pposv( blacspp::Triangle::Lower, N, NRHS,
    A_sca.local_mat().data(), 1, 1, desc_a, B_sca.local_mat().data(),
    1, 1, desc_b );
  if (info) throw std::runtime_error("Cholesky Solve Failed");

  if( x_trange.rank() == 0 ) x_trange = B.trange();

  world.gop.fence();
  auto X = block_cyclic_to_array<Array>( B_sca, x_trange );
  world.gop.fence();

  return X;
}




template <typename Array>
auto cholesky_lsolve( scalapackpp::TransposeFlag trans, 
  const Array& A, const Array& B, size_t NB = 128, 
  TiledRange l_trange = TiledRange(),
  TiledRange x_trange = TiledRange() ) {

  auto& world = A.world();
  /*
  if( world != B.world() ) {
    throw std::runtime_error("A and B must be distributed on same MADWorld context");
  }
  */
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence(); // stage ScaLAPACK execution
  auto A_sca = array_to_block_cyclic( A, grid, NB, NB );
  auto B_sca = array_to_block_cyclic( B, grid, NB, NB );
  world.gop.fence(); // stage ScaLAPACK execution

  auto [M, N] = A_sca.dims();
  if( M != N )
    throw std::runtime_error("A must be square for Cholesky Solve");

  auto [B_N, NRHS] = B_sca.dims();
  if( B_N != N )
    throw std::runtime_error("A and B dims must agree");


  scalapackpp::scalapack_desc desc_a, desc_b;
  {
  auto [Mloc, Nloc] = A_sca.dist().get_local_dims(N, N);
  desc_a = A_sca.dist().descinit_noerror(N, N, Mloc);
  }

  {
  auto [Mloc, Nloc] = B_sca.dist().get_local_dims(N, NRHS);
  desc_b = B_sca.dist().descinit_noerror(N, NRHS, Mloc);
  }

  auto info = scalapackpp::ppotrf( blacspp::Triangle::Lower, N,
    A_sca.local_mat().data(), 1, 1, desc_a );
  if (info) throw std::runtime_error("Cholesky Failed");

  info = scalapackpp::ptrtrs( blacspp::Triangle::Lower, trans, 
    blacspp::Diagonal::NonUnit, N, NRHS, A_sca.local_mat().data(), 1, 1, desc_a,
    B_sca.local_mat().data(), 1, 1, desc_b );
  if (info) throw std::runtime_error("TRTRS Failed");

  // Zero out the upper triangle
  detail::scalapack_zero_triangle( blacspp::Triangle::Upper, A_sca );

  if( l_trange.rank() == 0 ) l_trange = A.trange();
  if( x_trange.rank() == 0 ) x_trange = B.trange();

  world.gop.fence();
  auto L = block_cyclic_to_array<Array>( A_sca, l_trange );
  auto X = block_cyclic_to_array<Array>( B_sca, x_trange );
  world.gop.fence();

  return std::tuple(L, X);
}

} // namespace TiledArray

#endif // TILEDARRAY_HAS_SCALAPACK
#endif // TILEDARRAY_MATH_SCALAPACK_H__INCLUDED

