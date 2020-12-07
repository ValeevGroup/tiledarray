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
 *  cholesky.h
 *  Created:    8 June, 2020
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_SCALAPACK_CHOL_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SCALAPACK_CHOL_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK

#include <TiledArray/math/linalg/scalapack/util.h>
#include <TiledArray/math/linalg/forward.h>

#include <scalapackpp/factorizations/potrf.hpp>
#include <scalapackpp/linear_systems/posv.hpp>
#include <scalapackpp/linear_systems/trtrs.hpp>
#include <scalapackpp/matrix_inverse/trtri.hpp>

namespace TiledArray::math::linalg::scalapack {

/**
 *  @brief Compute the Cholesky factorization of a HPD rank-2 tensor
 *
 *  A(i,j) = L(i,k) * conj(L(j,k))
 *
 *  Example Usage:
 *
 *  auto L = cholesky(A, ...)
 *
 *  @tparam Array Input array type, must be convertible to BlockCyclicMatrix
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] l_trange    TiledRange for resulting Cholesky factor. If left
 * empty, will default to array.trange()
 *  @param[in] NB          ScaLAPACK block size. Defaults to 128
 *
 *  @returns The lower triangular Cholesky factor L in TA format
 */
template <typename Array>
auto cholesky(const Array& A, TiledRange l_trange = TiledRange(),
              size_t NB = default_block_size()) {
  auto& world = A.world();
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence();  // stage ScaLAPACK execution
  auto matrix = scalapack::array_to_block_cyclic(A, grid, NB, NB);
  world.gop.fence();  // stage ScaLAPACK execution

  auto [M, N] = matrix.dims();
  if (M != N) TA_EXCEPTION("Matrix must be square for Cholesky");

  auto [Mloc, Nloc] = matrix.dist().get_local_dims(N, N);
  auto desc = matrix.dist().descinit_noerror(N, N, Mloc);

  auto info = scalapackpp::ppotrf(blacspp::Triangle::Lower, N,
                                  matrix.local_mat().data(), 1, 1, desc);
  if (info) TA_EXCEPTION("Cholesky Failed");

  // Zero out the upper triangle
  zero_triangle(blacspp::Triangle::Upper, matrix);

  if (l_trange.rank() == 0) l_trange = A.trange();

  world.gop.fence();
  auto L = scalapack::block_cyclic_to_array<Array>(matrix, l_trange);
  world.gop.fence();

  return L;
}

/**
 *  @brief Compute the inverse of the Cholesky factor of an HPD rank-2 tensor.
 *  Optionally return the Cholesky factor itself
 *
 *  A(i,j) = L(i,k) * conj(L(j,k)) -> compute Linv
 *
 *  Example Usage:
 *
 *  auto Linv     = cholesky_Linv(A, ...)
 *  auto [L,Linv] = cholesky_Linv<decltype(A),true>(A, ...)
 *
 *  @tparam Array Input array type, must be convertible to BlockCyclicMatrix
 *  @tparam Both  Whether or not to return the cholesky factor
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] l_trange    TiledRange for resulting inverse Cholesky factor.
 *                         If left empty, will default to array.trange()
 *  @param[in] NB          ScaLAPACK block size. Defaults to 128
 *
 *  @returns The inverse lower triangular Cholesky factor in TA format
 */
  template <bool Both, typename Array>
auto cholesky_linv(const Array& A, TiledRange l_trange = TiledRange(),
                   size_t NB = default_block_size()) {
  using value_type = typename Array::element_type;

  auto& world = A.world();
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence();  // stage ScaLAPACK execution
  auto matrix = scalapack::array_to_block_cyclic(A, grid, NB, NB);
  world.gop.fence();  // stage ScaLAPACK execution

  auto [M, N] = matrix.dims();
  if (M != N) TA_EXCEPTION("Matrix must be square for Cholesky");

  auto [Mloc, Nloc] = matrix.dist().get_local_dims(N, N);
  auto desc = matrix.dist().descinit_noerror(N, N, Mloc);

  auto info = scalapackpp::ppotrf(blacspp::Triangle::Lower, N,
                                  matrix.local_mat().data(), 1, 1, desc);
  if (info) TA_EXCEPTION("Cholesky Failed");

  // Zero out the upper triangle
  zero_triangle(blacspp::Triangle::Upper, matrix);

  // Copy L if needed
  std::shared_ptr<scalapack::BlockCyclicMatrix<value_type>> L_sca = nullptr;
  if constexpr (Both) {
    L_sca = std::make_shared<scalapack::BlockCyclicMatrix<value_type>>(
        world, grid, N, N, NB, NB);
    L_sca->local_mat() = matrix.local_mat();
  }

  // Compute inverse
  info =
      scalapackpp::ptrtri(blacspp::Triangle::Lower, blacspp::Diagonal::NonUnit,
                          N, matrix.local_mat().data(), 1, 1, desc);
  if (info) TA_EXCEPTION("TRTRI Failed");

  if (l_trange.rank() == 0) l_trange = A.trange();

  world.gop.fence();
  auto Linv = scalapack::block_cyclic_to_array<Array>(matrix, l_trange);
  world.gop.fence();

  if constexpr (Both) {
    auto L = scalapack::block_cyclic_to_array<Array>(*L_sca, l_trange);
    world.gop.fence();
    return std::tuple(L, Linv);
  } else {
    return Linv;
  }
}

template <typename Array>
auto cholesky_solve(const Array& A, const Array& B,
                    TiledRange x_trange = TiledRange(),
                    size_t NB = default_block_size()) {
  auto& world = A.world();
  /*
  if( world != B.world() ) {
    TA_EXCEPTION("A and B must be distributed on same MADWorld context");
  }
  */
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence();  // stage ScaLAPACK execution
  auto A_sca = scalapack::array_to_block_cyclic(A, grid, NB, NB);
  auto B_sca = scalapack::array_to_block_cyclic(B, grid, NB, NB);
  world.gop.fence();  // stage ScaLAPACK execution

  auto [M, N] = A_sca.dims();
  if (M != N) TA_EXCEPTION("A must be square for Cholesky Solve");

  auto [B_N, NRHS] = B_sca.dims();
  if (B_N != N) TA_EXCEPTION("A and B dims must agree");

  scalapackpp::scalapack_desc desc_a, desc_b;
  {
    auto [Mloc, Nloc] = A_sca.dist().get_local_dims(N, N);
    desc_a = A_sca.dist().descinit_noerror(N, N, Mloc);
  }

  {
    auto [Mloc, Nloc] = B_sca.dist().get_local_dims(N, NRHS);
    desc_b = B_sca.dist().descinit_noerror(N, NRHS, Mloc);
  }

  auto info = scalapackpp::pposv(blacspp::Triangle::Lower, N, NRHS,
                                 A_sca.local_mat().data(), 1, 1, desc_a,
                                 B_sca.local_mat().data(), 1, 1, desc_b);
  if (info) TA_EXCEPTION("Cholesky Solve Failed");

  if (x_trange.rank() == 0) x_trange = B.trange();

  world.gop.fence();
  auto X = scalapack::block_cyclic_to_array<Array>(B_sca, x_trange);
  world.gop.fence();

  return X;
}

template <typename Array>
auto cholesky_lsolve(TransposeFlag trans, const Array& A, const Array& B,
                     TiledRange l_trange = TiledRange(),
                     TiledRange x_trange = TiledRange(),
                     size_t NB = default_block_size()) {
  auto& world = A.world();
  /*
  if( world != B.world() ) {
    TA_EXCEPTION("A and B must be distributed on same MADWorld context");
  }
  */
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence();  // stage ScaLAPACK execution
  auto A_sca = scalapack::array_to_block_cyclic(A, grid, NB, NB);
  auto B_sca = scalapack::array_to_block_cyclic(B, grid, NB, NB);
  world.gop.fence();  // stage ScaLAPACK execution

  auto [M, N] = A_sca.dims();
  if (M != N) TA_EXCEPTION("A must be square for Cholesky Solve");

  auto [B_N, NRHS] = B_sca.dims();
  if (B_N != N) TA_EXCEPTION("A and B dims must agree");

  scalapackpp::scalapack_desc desc_a, desc_b;
  {
    auto [Mloc, Nloc] = A_sca.dist().get_local_dims(N, N);
    desc_a = A_sca.dist().descinit_noerror(N, N, Mloc);
  }

  {
    auto [Mloc, Nloc] = B_sca.dist().get_local_dims(N, NRHS);
    desc_b = B_sca.dist().descinit_noerror(N, NRHS, Mloc);
  }

  auto info = scalapackpp::ppotrf(blacspp::Triangle::Lower, N,
                                  A_sca.local_mat().data(), 1, 1, desc_a);
  if (info) TA_EXCEPTION("Cholesky Failed");

  info = scalapackpp::ptrtrs(
      blacspp::Triangle::Lower, to_scalapackpp_transposeflag(trans),
      blacspp::Diagonal::NonUnit, N, NRHS, A_sca.local_mat().data(), 1, 1,
      desc_a, B_sca.local_mat().data(), 1, 1, desc_b);
  if (info) TA_EXCEPTION("TRTRS Failed");

  // Zero out the upper triangle
  zero_triangle(blacspp::Triangle::Upper, A_sca);

  if (l_trange.rank() == 0) l_trange = A.trange();
  if (x_trange.rank() == 0) x_trange = B.trange();

  world.gop.fence();
  auto L = scalapack::block_cyclic_to_array<Array>(A_sca, l_trange);
  auto X = scalapack::block_cyclic_to_array<Array>(B_sca, x_trange);
  world.gop.fence();

  return std::tuple(L, X);
}

}  // namespace TiledArray::math::linalg::scalapack

#endif  // TILEDARRAY_HAS_SCALAPACK
#endif  // TILEDARRAY_MATH_LINALG_SCALAPACK_CHOL_H__INCLUDED
