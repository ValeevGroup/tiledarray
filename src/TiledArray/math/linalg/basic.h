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

#include "TiledArray/math/linalg/forward.h"

#include "TiledArray/conversions/concat.h"
#include "TiledArray/dist_array.h"
#include "TiledArray/external/eigen.h"

namespace TiledArray::math::linalg {

namespace detail {

template <typename Tile, typename Policy>
inline bool prefer_distributed(const DistArray<Tile, Policy>& matrix) {
  const auto prefer_distributed =
      get_linalg_backend() == LinearAlgebraBackend::BestAvailable &&
      matrix.elements_range().volume() >=
          get_linalg_crossover_to_distributed() &&
      matrix.world().size() > 1;
  return prefer_distributed;
}

template <lapack::Uplo Uplo, typename T = float>
struct symmetric_matrix_shape {
  symmetric_matrix_shape(T v) : v_(v) {}
  T operator()(const Range::index_type& idx) const {
    TA_ASSERT(idx.size() == 2);
    if constexpr (Uplo == lapack::Uplo::Lower) {
      return (idx[0] >= idx[1]) ? v_ : 0.;
    } else if constexpr (Uplo == lapack::Uplo::Upper) {
      return (idx[0] <= idx[1]) ? v_ : 0.;
    } else {  // Uplo == lapack::Uplo::General
      return v_;
    }
  }

  T v_;
};

}  // namespace detail

namespace non_distributed {}
namespace scalapack {}
namespace ttg {}

}  // namespace TiledArray::math::linalg

namespace TiledArray {

// freestanding adaptors for DistArray needed by solvers like DIIS

template <typename Tile, typename Policy>
inline void vec_multiply(DistArray<Tile, Policy>& a1,
                         const DistArray<Tile, Policy>& a2) {
  auto vars = TiledArray::detail::dummy_annotation(rank(a1));
  a1.make_tsrexpr(vars) = a1.make_tsrexpr(vars) * a2.make_tsrexpr(vars);
}

template <typename Tile, typename Policy, typename S>
inline void scale(DistArray<Tile, Policy>& a, S scaling_factor) {
  using numeric_type = typename DistArray<Tile, Policy>::numeric_type;
  auto vars = TiledArray::detail::dummy_annotation(rank(a));
  a.make_tsrexpr(vars) = numeric_type(scaling_factor) * a.make_tsrexpr(vars);
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
  y.make_tsrexpr(vars) =
      y.make_tsrexpr(vars) + numeric_type(alpha) * x.make_tsrexpr(vars);
}

/// selector for concat
enum class Concat : char { Row = 'R', Col = 'C', Both = 'B' };

/// generic TiledArray::concat adapted for matrices
template <typename Tile, typename Policy>
inline DistArray<Tile, Policy> concat(const DistArray<Tile, Policy>& a,
                                      const DistArray<Tile, Policy>& b,
                                      Concat C) {
  TA_ASSERT(a.trange().rank() == 2);
  TA_ASSERT(b.trange().rank() == 2);
  switch (C) {
    case Concat::Row:
      return TiledArray::concat<Tile, Policy>({a, b},
                                              std::vector<bool>{true, false});
    case Concat::Col:
      return TiledArray::concat<Tile, Policy>({a, b},
                                              std::vector<bool>{false, true});
    case Concat::Both:
      return TiledArray::concat<Tile, Policy>({a, b},
                                              std::vector<bool>{true, true});
  }
}

using TiledArray::math::linalg::get_linalg_backend;
using TiledArray::math::linalg::get_linalg_crossover_to_distributed;
using TiledArray::math::linalg::LinearAlgebraBackend;
using TiledArray::math::linalg::set_linalg_backend;
using TiledArray::math::linalg::set_linalg_crossover_to_distributed;

}  // namespace TiledArray

namespace Eigen {

// freestanding adaptors for Eigen::MatrixBase and Eigen::Block
// needed by solvers like DIIS

template <typename Derived>
inline void vec_multiply(Eigen::MatrixBase<Derived>& a1,
                         const Eigen::MatrixBase<Derived>& a2) {
  a1.array() *= a2.array();
}

template <typename XprType1, int BlockRows1, int BlockCols1, bool InnerPanel1,
          typename XprType2, int BlockRows2, int BlockCols2, bool InnerPanel2>
inline void vec_multiply(
    Eigen::Block<XprType1, BlockRows1, BlockCols1, InnerPanel1>& a1,
    const Eigen::Block<XprType2, BlockRows2, BlockCols2, InnerPanel2>& a2) {
  a1.array() *= a2.array();
}

template <typename Derived, typename S>
inline void scale(Eigen::MatrixBase<Derived>& a, S scaling_factor) {
  using numeric_type = typename Eigen::MatrixBase<Derived>::value_type;
  a.array() *= numeric_type(scaling_factor);
}

template <typename XprType1, int BlockRows1, int BlockCols1, bool InnerPanel1,
          typename S>
inline void scale(
    Eigen::Block<XprType1, BlockRows1, BlockCols1, InnerPanel1>& a,
    S scaling_factor) {
  using numeric_type = typename Eigen::Block<XprType1, BlockRows1, BlockCols1,
                                             InnerPanel1>::value_type;
  a.array() *= numeric_type(scaling_factor);
}

template <typename Derived>
inline void zero(Eigen::MatrixBase<Derived>& a) {
  a.fill(0);
}

template <typename XprType1, int BlockRows1, int BlockCols1, bool InnerPanel1>
inline void zero(
    Eigen::Block<XprType1, BlockRows1, BlockCols1, InnerPanel1>& a) {
  a.fill(0);
}

template <typename Derived, typename S>
inline void axpy(Eigen::MatrixBase<Derived>& y, S alpha,
                 const Eigen::MatrixBase<Derived>& x) {
  using numeric_type = typename Eigen::MatrixBase<Derived>::value_type;
  y.array() += numeric_type(alpha) * x.array();
}

template <typename XprType1, int BlockRows1, int BlockCols1, bool InnerPanel1,
          typename XprType2, int BlockRows2, int BlockCols2, bool InnerPanel2,
          typename S>
inline void axpy(
    Eigen::Block<XprType1, BlockRows1, BlockCols1, InnerPanel1>& y, S alpha,
    const Eigen::Block<XprType2, BlockRows2, BlockCols2, InnerPanel2>& x) {
  using numeric_type = typename Eigen::Block<XprType2, BlockRows2, BlockCols2,
                                             InnerPanel2>::value_type;
  y.array() += numeric_type(alpha) * x.array();
}

template <typename Derived>
inline auto dot(const Eigen::MatrixBase<Derived>& l,
                const Eigen::MatrixBase<Derived>& r) {
  return l.adjoint().dot(r);
}

template <typename XprType1, int BlockRows1, int BlockCols1, bool InnerPanel1,
          typename XprType2, int BlockRows2, int BlockCols2, bool InnerPanel2>
inline auto dot(
    const Eigen::Block<XprType1, BlockRows1, BlockCols1, InnerPanel1>& l,
    const Eigen::Block<XprType2, BlockRows2, BlockCols2, InnerPanel2>& r) {
  return l.adjoint().dot(r);
}

template <typename Derived>
inline auto inner_product(const Eigen::MatrixBase<Derived>& l,
                          const Eigen::MatrixBase<Derived>& r) {
  return l.dot(r);
}

template <typename XprType1, int BlockRows1, int BlockCols1, bool InnerPanel1,
          typename XprType2, int BlockRows2, int BlockCols2, bool InnerPanel2>
inline auto inner_product(
    const Eigen::Block<XprType1, BlockRows1, BlockCols1, InnerPanel1>& l,
    const Eigen::Block<XprType2, BlockRows2, BlockCols2, InnerPanel2>& r) {
  return l.dot(r);
}

template <typename Derived>
inline auto norm2(const Eigen::MatrixBase<Derived>& m) {
  return m.template lpNorm<2>();
}

template <typename XprType1, int BlockRows1, int BlockCols1, bool InnerPanel1>
inline auto norm2(
    const Eigen::Block<XprType1, BlockRows1, BlockCols1, InnerPanel1>& m) {
  return m.template lpNorm<2>();
}

}  // namespace Eigen

#ifndef TILEDARRAY_MATH_LINALG_DISPATCH_W_TTG
#if (TILEDARRAY_HAS_TTG && TILEDARRAY_HAS_SCALAPACK)
#define TILEDARRAY_MATH_LINALG_DISPATCH_W_TTG(FN, MATRIX)           \
  TA_MAX_THREADS;                                                   \
  if (get_linalg_backend() == LinearAlgebraBackend::TTG ||          \
      TiledArray::math::linalg::detail::prefer_distributed(MATRIX)) \
    return TiledArray::math::linalg::ttg::FN;                       \
  if (get_linalg_backend() == LinearAlgebraBackend::ScaLAPACK ||    \
      TiledArray::math::linalg::detail::prefer_distributed(MATRIX)) \
    return scalapack::FN;                                           \
  return non_distributed::FN;
#elif (TILEDARRAY_HAS_TTG && !TILEDARRAY_HAS_SCALAPACK)
#define TILEDARRAY_MATH_LINALG_DISPATCH_W_TTG(FN, MATRIX)               \
  TA_MAX_THREADS;                                                       \
  if (get_linalg_backend() == LinearAlgebraBackend::TTG ||              \
      TiledArray::math::linalg::detail::prefer_distributed(MATRIX))     \
    return TiledArray::math::linalg::ttg::FN;                           \
  if (get_linalg_backend() == LinearAlgebraBackend::ScaLAPACK)          \
    TA_EXCEPTION("ScaLAPACK lineear algebra backend is not available"); \
  return non_distributed::FN;
#elif !TILEDARRAY_HAS_TTG && TILEDARRAY_HAS_SCALAPACK
#define TILEDARRAY_MATH_LINALG_DISPATCH_W_TTG(FN, MATRIX)           \
  TA_MAX_THREADS;                                                   \
  if (get_linalg_backend() == LinearAlgebraBackend::TTG)            \
    TA_EXCEPTION("TTG linear algebra backend is not available");    \
  if (get_linalg_backend() == LinearAlgebraBackend::ScaLAPACK ||    \
      TiledArray::math::linalg::detail::prefer_distributed(MATRIX)) \
    return scalapack::FN;                                           \
  return non_distributed::FN;
#else  // !TILEDARRAY_HAS_TTG && !TILEDARRAY_HAS_SCALAPACK
#define TILEDARRAY_MATH_LINALG_DISPATCH_W_TTG(FN, MATRIX)               \
  TA_MAX_THREADS;                                                       \
  if (get_linalg_backend() == LinearAlgebraBackend::TTG)                \
    TA_EXCEPTION("TTG linear algebra backend is not available");        \
  if (get_linalg_backend() == LinearAlgebraBackend::ScaLAPACK)          \
    TA_EXCEPTION("ScaLAPACK lineear algebra backend is not available"); \
  return non_distributed::FN;
#endif  // !TILEDARRAY_HAS_TTG && !TILEDARRAY_HAS_SCALAPACK
#endif  // defined(TILEDARRAY_MATH_LINALG_DISPATCH_W_TTG)

#ifndef TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG_STRINGIFY
#define TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG_STRINGIFY(FN) #FN
#endif  // defined(TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG_STRINGIFY)

#ifndef TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG
#if TILEDARRAY_HAS_TTG && TILEDARRAY_HAS_SCALAPACK
#define TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG(FN, MATRIX)          \
  TA_MAX_THREADS;                                                   \
  if (get_linalg_backend() == LinearAlgebraBackend::TTG)            \
    TA_EXCEPTION(TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG_STRINGIFY(  \
        FN) " is not provided by the TTG backend");                 \
  if (get_linalg_backend() == LinearAlgebraBackend::ScaLAPACK ||    \
      TiledArray::math::linalg::detail::prefer_distributed(MATRIX)) \
    return scalapack::FN;                                           \
  return non_distributed::FN;
#elif TILEDARRAY_HAS_TTG && !TILEDARRAY_HAS_SCALAPACK
#define TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG(FN, MATRIX)              \
  TA_MAX_THREADS;                                                       \
  if (get_linalg_backend() == LinearAlgebraBackend::TTG)                \
    TA_EXCEPTION(TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG_STRINGIFY(      \
        FN) " is not provided by the TTG backend");                     \
  if (get_linalg_backend() == LinearAlgebraBackend::ScaLAPACK)          \
    TA_EXCEPTION("ScaLAPACK lineear algebra backend is not available"); \
  return non_distributed::FN;
#elif !TILEDARRAY_HAS_TTG && TILEDARRAY_HAS_SCALAPACK
#define TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG(FN, MATRIX)          \
  TA_MAX_THREADS;                                                   \
  if (get_linalg_backend() == LinearAlgebraBackend::TTG)            \
    TA_EXCEPTION("TTG linear algebra backend is not available");    \
  if (get_linalg_backend() == LinearAlgebraBackend::ScaLAPACK ||    \
      TiledArray::math::linalg::detail::prefer_distributed(MATRIX)) \
    return scalapack::FN;                                           \
  return non_distributed::FN;
#else  // !TILEDARRAY_HAS_TTG && !TILEDARRAY_HAS_SCALAPACK
#define TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG(FN, MATRIX)              \
  TA_MAX_THREADS;                                                       \
  if (get_linalg_backend() == LinearAlgebraBackend::TTG)                \
    TA_EXCEPTION("TTG linear algebra backend is not available");        \
  if (get_linalg_backend() == LinearAlgebraBackend::ScaLAPACK)          \
    TA_EXCEPTION("ScaLAPACK lineear algebra backend is not available"); \
  return non_distributed::FN;
#endif  // !TILEDARRAY_HAS_TTG && !TILEDARRAY_HAS_SCALAPACK
#endif  // defined(TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG)

#endif  // TILEDARRAY_MATH_LINALG_BASIC_H__INCLUDED
