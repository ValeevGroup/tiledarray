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
 *  util.h
 *  Created:    19 October, 2020
 *
 */
#ifndef TILEDARRAY_ALGEBRA_LAPACK_UTIL_H__INCLUDED
#define TILEDARRAY_ALGEBRA_LAPACK_UTIL_H__INCLUDED

#include <TiledArray/config.h>
#include <TiledArray/conversions/eigen.h>

namespace TiledArray {
namespace lapack {

#define MADNESS_DISPATCH_LAPACK_FN(name, args...)                        \
  if constexpr (std::is_same_v<numeric_type, double>)                    \
    d##name##_(args);                                                    \
  else if constexpr (std::is_same_v<numeric_type, float>)                \
    s##name##_(args);                                                    \
  else if constexpr (std::is_same_v<numeric_type, std::complex<double>>) \
    z##name##_(args);                                                    \
  else if constexpr (std::is_same_v<numeric_type, std::complex<float>>)  \
    c##name##_(args);                                                    \
  else                                                                   \
    std::abort();

namespace detail {

template <typename Tile, typename Policy>
auto to_eigen(const DistArray<Tile, Policy>& A) {
  auto A_repl = A;
  A_repl.make_replicated();
  return array_to_eigen<Tile, Policy, Eigen::ColMajor>(A_repl);
}

template <typename ContiguousTensor,
          typename = std::enable_if_t<
              TiledArray::detail::is_contiguous_tensor_v<ContiguousTensor>>>
auto to_eigen(const ContiguousTensor& A) {
  using numeric_type = TiledArray::detail::numeric_t<ContiguousTensor>;
  static_assert(
      std::is_same_v<numeric_type, typename ContiguousTensor::value_type>,
      "TA::lapack::{cholesky*} are only usable with a ContiguousTensor of "
      "scalar types");
  TA_ASSERT(A.range().rank() == 1 || A.range().rank() == 2);
  using colmajor_matrix_type = Eigen::Matrix<numeric_type, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::ColMajor>;
  colmajor_matrix_type result(A.range().extent(0),
                              A.range().rank() == 2 ? A.range().extent(1) : 1);
  constexpr const auto layout =
      TiledArray::detail::ordinal_trait<ContiguousTensor>::layout;
  if (layout == TiledArray::OrdinalType::RowMajor) {
    using rowmajor_matrix_type = Eigen::Matrix<numeric_type, Eigen::Dynamic,
                                               Eigen::Dynamic, Eigen::RowMajor>;
    auto result_map = Eigen::Map<const rowmajor_matrix_type>(
        A.data(), result.rows(), result.cols());
    result = result_map;
  } else if constexpr (layout == TiledArray::OrdinalType::ColMajor) {
    using rowmajor_matrix_type = Eigen::Matrix<numeric_type, Eigen::Dynamic,
                                               Eigen::Dynamic, Eigen::ColMajor>;
    auto result_map = Eigen::Map<const rowmajor_matrix_type>(
        A.data(), result.rows(), result.cols());
    result = result_map;
  } else
    abort();
  return result;
}

template <typename ContiguousTensor, typename Scalar, int RowsAtCompileTime,
          int ColsAtCompileTime, int Options, int MaxRowsAtCompileTime,
          int MaxColsAtCompileTime,
          typename = std::enable_if_t<
              TiledArray::detail::is_contiguous_tensor_v<ContiguousTensor>>>
auto from_eigen(
    const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options,
                        MaxRowsAtCompileTime, MaxColsAtCompileTime>& A,
    typename ContiguousTensor::range_type range = {}) {
  using numeric_type = TiledArray::detail::numeric_t<ContiguousTensor>;
  static_assert(
      std::is_same_v<numeric_type, typename ContiguousTensor::value_type>,
      "TA::lapack::{cholesky*} are only usable with a ContiguousTensor of "
      "scalar types");
  using range_type = typename ContiguousTensor::range_type;
  if (range.rank() == 0)
    range = range_type(A.rows(), A.cols());
  else
    TA_ASSERT(A.rows() * A.cols() == range.volume());
  ContiguousTensor result(range);
  auto result_map = eigen_map(result, A.rows(), A.cols());
  result_map = A;
  return result;
}

template <typename Derived>
void zero_out_upper_triangle(Eigen::MatrixBase<Derived>& A) {
  A.template triangularView<Eigen::StrictlyUpper>().setZero();
}

}  // namespace detail

}  // namespace lapack
}  // namespace TiledArray

#endif  // TILEDARRAY_ALGEBRA_LAPACK_UTIL_H__INCLUDED
