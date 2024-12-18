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
 *  heig.h
 *  Created:  13 May,  2020
 *  Edited:    8 June, 2020
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_SCALAPACK_HEIG_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SCALAPACK_HEIG_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK

#include <TiledArray/math/linalg/scalapack/util.h>

#include <scalapackpp/eigenvalue_problem/gevp.hpp>
#include <scalapackpp/eigenvalue_problem/sevp.hpp>

namespace TiledArray::math::linalg::scalapack {

/**
 *  @brief Solve the standard eigenvalue problem with ScaLAPACK
 *
 *  A(i,k) X(k,j) = X(i,j) E(j)
 *
 *  Example Usage:
 *
 *  auto [E, X] = heig(A, ...)
 *
 *  @tparam Array Input array type, must be convertible to BlockCyclicMatrix
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] evec_trange TiledRange for resulting eigenvectors. If left empty,
 *                         will default to array.trange()
 *  @param[in] NB          ScaLAPACK block size. Defaults to 128
 *
 *  @returns A tuple containing the eigenvalues and eigenvectors of input array
 *  as std::vector and in TA format, respectively.
 */
template <typename Array>
auto heig(const Array& A, TiledRange evec_trange = TiledRange(),
          size_t NB = default_block_size()) {
  using value_type = typename Array::numeric_type;
  using real_type = scalapackpp::detail::real_t<value_type>;

  auto& world = A.world();
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  // auto world_comm = MPI_COMM_WORLD;
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence();  // stage ScaLAPACK execution
  auto matrix = scalapack::array_to_block_cyclic(A, grid, NB, NB);
  world.gop.fence();  // stage ScaLAPACK execution

  auto [M, N] = matrix.dims();
  if (M != N) TA_EXCEPTION("Matrix must be square for EVP");

  auto [Mloc, Nloc] = matrix.dist().get_local_dims(N, N);
  auto desc = matrix.dist().descinit_noerror(N, N, Mloc);

  std::vector<real_type> evals(N);
  scalapack::BlockCyclicMatrix<value_type> evecs(world, grid, N, N, NB, NB);

  auto info = scalapackpp::hereig(
      scalapackpp::Job::Vec, blacspp::Uplo::Lower, N, matrix.local_mat().data(),
      1, 1, desc, evals.data(), evecs.local_mat().data(), 1, 1, desc);
  if (info) TA_EXCEPTION("EVP Failed");

  if (evec_trange.rank() == 0) evec_trange = A.trange();

  world.gop.fence();
  auto evecs_ta = scalapack::block_cyclic_to_array<Array>(evecs, evec_trange);
  world.gop.fence();

  return std::tuple(evals, evecs_ta);
}

/**
 *  @brief Solve the generalized eigenvalue problem with ScaLAPACK
 *
 *  A(i,k) X(k,j) = B(i,k) X(k,j) E(j)
 *
 *  with
 *
 *  X(k,i) B(k,l) X(l,j) = I(i,j)
 *
 *  Example Usage:
 *
 *  auto [E, X] = heig(A, B, ...)
 *
 *  @tparam Array Input array type, must be convertible to BlockCyclicMatrix
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] B           Metric
 *  @param[in] evec_trange TiledRange for resulting eigenvectors. If left empty,
 *                         will default to array.trange()
 *  @param[in] NB          ScaLAPACK block size. Defaults to 128
 *
 *  @returns A tuple containing the eigenvalues and eigenvectors of input array
 *  as std::vector and in TA format, respectively.
 */
template <typename ArrayA, typename ArrayB, typename EVecType = ArrayA>
auto heig(const ArrayA& A, const ArrayB& B,
          TiledRange evec_trange = TiledRange(),
          size_t NB = default_block_size()) {
  using value_type = typename ArrayA::numeric_type;
  static_assert(std::is_same_v<typename ArrayB::numeric_type, value_type>);
  using real_type = scalapackpp::detail::real_t<value_type>;

  auto& world = A.world();
  auto world_comm = world.mpi.comm().Get_mpi_comm();
  // auto world_comm = MPI_COMM_WORLD;
  blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

  world.gop.fence();  // stage ScaLAPACK execution
  auto A_sca = scalapack::array_to_block_cyclic(A, grid, NB, NB);
  auto B_sca = scalapack::array_to_block_cyclic(B, grid, NB, NB);
  world.gop.fence();  // stage ScaLAPACK execution

  auto [M, N] = A_sca.dims();
  if (M != N) TA_EXCEPTION("Matrix must be square for EVP");

  auto [B_M, B_N] = B_sca.dims();
  if (B_M != M or B_N != N)
    TA_EXCEPTION("A and B must have the same dimensions");

  auto [Mloc, Nloc] = A_sca.dist().get_local_dims(N, N);
  auto desc = A_sca.dist().descinit_noerror(N, N, Mloc);

  std::vector<real_type> evals(N);
  scalapack::BlockCyclicMatrix<value_type> evecs(world, grid, N, N, NB, NB);

  auto info = scalapackpp::hereig_gen(
      scalapackpp::Job::Vec, blacspp::Uplo::Lower, N, A_sca.local_mat().data(),
      1, 1, desc, B_sca.local_mat().data(), 1, 1, desc, evals.data(),
      evecs.local_mat().data(), 1, 1, desc);
  if (info) TA_EXCEPTION("EVP Failed");

  if (evec_trange.rank() == 0) evec_trange = A.trange();

  world.gop.fence();
  auto evecs_ta =
      scalapack::block_cyclic_to_array<EVecType>(evecs, evec_trange);
  world.gop.fence();

  return std::tuple(evals, evecs_ta);
}

}  // namespace TiledArray::math::linalg::scalapack

#endif  // TILEDARRAY_HAS_SCALAPACK
#endif  // TILEDARRAY_MATH_LINALG_SCALAPACK_HEIG_H__INCLUDED
