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

#ifndef TILEDARRAY_MATH_LINALG_BASIC_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_BASIC_H__INCLUDED

#include "TiledArray/dist_array.h"
#include "TiledArray/external/eigen.h"

namespace TiledArray::math::linalg {

// freestanding adaptors for DistArray needed by solvers like DIIS

template <typename Tile, typename Policy>
inline void vec_multiply(DistArray<Tile, Policy>& a1,
                         const DistArray<Tile, Policy>& a2) {
  auto vars = TiledArray::detail::dummy_annotation(rank(a1));
  a1(vars) = a1(vars) * a2(vars);
}

template <typename Tile, typename Policy, typename S>
inline void scale(DistArray<Tile, Policy>& a, S scaling_factor) {
  using numeric_type = typename DistArray<Tile, Policy>::numeric_type;
  auto vars = TiledArray::detail::dummy_annotation(rank(a));
  a(vars) = numeric_type(scaling_factor) * a(vars);
}

template <typename Tile, typename Policy>
inline void zero(DistArray<Tile, Policy>& a) {
  scale(a, 0);
}

template <typename Tile, typename Policy, typename S>
inline void axpy(DistArray<Tile, Policy>& y, S alpha,
                 const DistArray<Tile, Policy>& x) {
  using numeric_type = typename DistArray<Tile, Policy>::numeric_type;
  auto vars = TiledArray::detail::dummy_annotation(rank(y));
  y(vars) = y(vars) + numeric_type(alpha) * x(vars);
}

}  // namespace TiledArray::math::linalg

namespace Eigen {

// freestanding adaptors for Eigen::MatrixBase needed by solvers like DIIS

template <typename Derived>
inline void vec_multiply(Eigen::MatrixBase<Derived>& a1,
                         const Eigen::MatrixBase<Derived>& a2) {
  a1.array() *= a2.array();
}

template <typename Derived, typename S>
inline void scale(Eigen::MatrixBase<Derived>& a, S scaling_factor) {
  using numeric_type = typename Eigen::MatrixBase<Derived>::value_type;
  a.array() *= numeric_type(scaling_factor);
}

template <typename Derived>
inline void zero(Eigen::MatrixBase<Derived>& a) {
  a = Derived::Zero(a.rows(), a.cols());
}

template <typename Derived, typename S>
inline void axpy(Eigen::MatrixBase<Derived>& y, S alpha,
                 const Eigen::MatrixBase<Derived>& x) {
  using numeric_type = typename Eigen::MatrixBase<Derived>::value_type;
  y.array() += numeric_type(alpha) * x.array();
}

template <typename Derived>
inline auto dot(const Eigen::MatrixBase<Derived>& l,
                const Eigen::MatrixBase<Derived>& r) {
  return l.adjoint().dot(r);
}

template <typename Derived>
inline auto inner_product(const Eigen::MatrixBase<Derived>& l,
                          const Eigen::MatrixBase<Derived>& r) {
  return l.dot(r);
}

template <typename Derived>
inline auto norm2(const Eigen::MatrixBase<Derived>& m) {
  return m.template lpNorm<2>();
}

}  // namespace Eigen

#endif  // TILEDARRAY_MATH_LINALG_BASIC_H__INCLUDED
