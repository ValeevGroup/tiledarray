/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  Department of Chemistry, Virginia Tech
 *
 *  util.h
 *  May 20, 2013
 *
 */

#ifndef TILEDARRAY_MATH_LINALG_UTIL_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_UTIL_H__INCLUDED

#include <TiledArray/config.h>
#include "TiledArray/dist_array.h"
#include <TiledArray/conversions/eigen.h>

namespace TiledArray::math::linalg::detail {

template<class A>
struct array_traits {
  using scalar_type = typename A::scalar_type;
  using numeric_type = typename A::numeric_type;
  static const bool complex = !std::is_same_v<scalar_type, numeric_type>;
  static_assert(std::is_same_v<numeric_type, typename A::element_type>,
                "TA::linalg is only usable with a DistArray of scalar types");
};


template <typename Tile, typename Policy>
auto make_matrix(const DistArray<Tile, Policy>& A) {
  auto A_repl = A;
  A_repl.make_replicated();
  return array_to_eigen<Tile, Policy, Eigen::ColMajor>(A_repl);
}

template <typename ContiguousTensor,
          typename = std::enable_if_t<
              TiledArray::detail::is_contiguous_tensor_v<ContiguousTensor>>>
auto make_matrix(const ContiguousTensor& A) {
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
      TiledArray::detail::ordinal_traits<ContiguousTensor>::type;
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

template <typename ContiguousTensor, typename Scalar, int ... Options,
          typename = std::enable_if_t<
              TiledArray::detail::is_contiguous_tensor_v<ContiguousTensor>>>
auto make_array(const Eigen::Matrix<Scalar, Options...>& A,
                typename ContiguousTensor::range_type range = {}) {
  using numeric_type = TiledArray::detail::numeric_t<ContiguousTensor>;
  static_assert(
      std::is_same_v<numeric_type, typename ContiguousTensor::value_type>,
      "TA::math::linalg is only usable with a ContiguousTensor of scalar types"
  );
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

}  // namespace TiledArray::math::linalg

#endif  // TILEDARRAY_MATH_LINALG_UTIL_H__INCLUDED
