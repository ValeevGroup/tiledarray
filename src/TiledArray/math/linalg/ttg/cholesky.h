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
 *  Eduard Valeyev
 *
 *  cholesky.h
 *  Created:    26 July, 2022
 *
 */

#ifndef TILEDARRAY_MATH_LINALG_TTG_CHOL_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_TTG_CHOL_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_TTG

#include <TiledArray/math/linalg/forward.h>
#include <TiledArray/math/linalg/ttg/util.h>

#include <ttg/../../examples/potrf/potrf.h>
#include <ttg/../../examples/potrf/trtri_L.h>

namespace TiledArray::math::linalg::ttg {

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
auto cholesky(const Array& A, TiledRange l_trange = {},
              size_t NB = default_block_size()) {
  using value_type = typename Array::element_type;
  using Tile = typename Array::value_type;
  using Policy = typename Array::policy_type;
  using shape_type = typename Policy::shape_type;
  constexpr auto is_dense_policy = is_dense_v<Policy>;

  ::ttg::Edge<Key2, MatrixTile<double>> input;
  ::ttg::Edge<Key2, MatrixTile<double>> output;
  const auto A_descr = MatrixDescriptor<Tile, Policy>(
      A);  // make_potrf_ttg keeps references to this, must outlive all work
  auto potrf_ttg = potrf::make_potrf_ttg(A_descr, input, output,
                                         /* defer_write = */ true);

  // make result
  shape_type L_shape;
  if constexpr (!is_dense_policy)
    L_shape = shape_type(
        math::linalg::detail::symmetric_matrix_shape<lapack::Uplo::Lower,
                                                     float>(1),
        A.trange());
  Array L(A.world(), A.trange(), L_shape,
          A.pmap());  // potrf produces a dense result for now
  auto store_potrf_ttg =
      make_writer_ttg<lapack::Layout::ColMajor, lapack::Uplo::Lower>(
          L, output, /* defer_write = */ true);

  [[maybe_unused]] auto connected = make_graph_executable(potrf_ttg.get());

  // uncomment to trace
  ::ttg::trace_on();

  // start
  ::ttg::execute();

  // *now* "connect" input data to TTG
  // TTG expect lower triangle of the matrix, and col-major tiles
  flow_matrix_to_tt<lapack::Layout::ColMajor, lapack::Uplo::Lower>(
      A, potrf_ttg.get());

  A.world().gop.fence();
  ::ttg::fence();

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
 *  @param[in] NB          target block size. Defaults to 128
 *
 *  @returns The inverse lower triangular Cholesky factor in TA format
 */
template <bool Both, typename Array>
auto cholesky_linv(const Array& A, TiledRange l_trange = {},
                   size_t NB = default_block_size()) {
  using value_type = typename Array::element_type;
  using Tile = typename Array::value_type;
  using Policy = typename Array::policy_type;
  using shape_type = typename Policy::shape_type;
  constexpr auto is_dense_policy = is_dense_v<Policy>;

  ::ttg::Edge<Key2, MatrixTile<double>> input;
  ::ttg::Edge<Key2, MatrixTile<double>> potrf2trtri;
  ::ttg::Edge<Key2, MatrixTile<double>> potrf_output;
  ::ttg::Edge<Key2, MatrixTile<double>> trtri_output;
  const auto A_descr = MatrixDescriptor<Tile, Policy>(
      A);  // make_potrf_ttg and make_trtri_ttg keep references to this, must
           // outlive all work
  auto portf_out_edge =
      Both ? ::ttg::fuse(potrf2trtri, potrf_output) : potrf2trtri;
  auto potrf_ttg = potrf::make_potrf_ttg(A_descr, input, portf_out_edge,
                                         /* defer_write = */ true);
  auto trtri_ttg = trtri_LOWER::make_trtri_ttg(A_descr, lapack::Diag::NonUnit,
                                               potrf2trtri, trtri_output,
                                               /* defer_write = */ true);

  // make result(s)
  shape_type L_shape;
  if constexpr (!is_dense_policy)
    L_shape = shape_type(
        math::linalg::detail::symmetric_matrix_shape<lapack::Uplo::Lower,
                                                     float>(1),
        A.trange(),
        /* per-element norm values already */ true);
  Array L;
  if constexpr (Both)
    L = Array(A.world(), A.trange(), L_shape,
              A.pmap());  // trtri produces a dense result for now
  Array Linv(A.world(), A.trange(), L_shape,  // Linv has same shape as L
             A.pmap());  // trtri produces a dense result for now
  std::unique_ptr<::ttg::TTBase> store_potrf_tt_ptr;
  if constexpr (Both) {
    auto store_potrf_ttg =
        make_writer_ttg<lapack::Layout::ColMajor, lapack::Uplo::Lower>(
            L, potrf_output, /* defer_write = */ true);
    store_potrf_tt_ptr = std::move(store_potrf_ttg);
  }
  auto store_trtri_ttg =
      make_writer_ttg<lapack::Layout::ColMajor, lapack::Uplo::Lower>(
          Linv, trtri_output, /* defer_write = */ true);

  [[maybe_unused]] auto connected = make_graph_executable(trtri_ttg.get());

  // uncomment to trace
  ::ttg::trace_on();

  // start
  ::ttg::execute();

  // *now* "connect" input data to TTG
  // TTG expect lower triangle of the matrix, and col-major tiles
  flow_matrix_to_tt<lapack::Layout::ColMajor, lapack::Uplo::Lower>(
      A, potrf_ttg.get());

  A.world().gop.fence();
  ::ttg::fence();

  // Copy L if needed
  if constexpr (Both) {
    return std::tuple(L, Linv);
  } else
    return Linv;
}

}  // namespace TiledArray::math::linalg::ttg

#endif  // TILEDARRAY_HAS_TTG
#endif  // TILEDARRAY_MATH_LINALG_TTG_CHOL_H__INCLUDED
