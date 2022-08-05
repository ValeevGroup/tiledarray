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

/// known linear algebra backends
enum LinearAlgebraBackend {
  /// choose the best that's available, taking into consideration the problem
  /// size and # of ranks
  BestAvailable,
  /// LAPACK on rank 0, followed by broadcast
  LAPACK,
  /// ScaLAPACK
  ScaLAPACK,
  /// TTG (currently only provides cholesky and cholesky_linv)
  TTG
};

namespace detail {

inline LinearAlgebraBackend& linalg_backend_accessor() {
  static LinearAlgebraBackend backend = LinearAlgebraBackend::BestAvailable;
  return backend;
}

inline std::size_t& linalg_crossover_to_distributed_accessor() {
  static std::size_t crossover_to_distributed = 2048 * 2048;
  return crossover_to_distributed;
}

}  // namespace detail

/// @return the linear algebra backend to use for matrix algebra
/// @note the default is LinearAlgebraBackend::BestAvailable
inline LinearAlgebraBackend get_linalg_backend() {
  return detail::linalg_backend_accessor();
}

/// @param[in] b the linear algebra backend to use after this
/// @note this is a collective call over the default world
inline void set_linalg_backend(LinearAlgebraBackend b) {
  get_default_world().gop.fence();
  detail::linalg_backend_accessor() = b;
}

/// @return the crossover-to-distributed threshold specifies the matrix volume
/// for which to switch to the distributed-memory backend (if available) for
/// linear algebra solvers
inline std::size_t get_linalg_crossover_to_distributed() {
  return detail::linalg_crossover_to_distributed_accessor();
}

/// @param[in] c the crossover-to-distributed threshold to use following this
/// call
/// @note this is a collective call over the default world
inline void set_linalg_crossover_to_distributed(std::size_t c) {
  get_default_world().gop.fence();
  detail::linalg_crossover_to_distributed_accessor() = c;
}

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

namespace non_distributed {}
namespace scalapack {}
namespace ttg {}

}  // namespace TiledArray::math::linalg

namespace TiledArray {
using TiledArray::math::linalg::get_linalg_backend;
using TiledArray::math::linalg::get_linalg_crossover_to_distributed;
using TiledArray::math::linalg::LinearAlgebraBackend;
using TiledArray::math::linalg::set_linalg_backend;
using TiledArray::math::linalg::set_linalg_crossover_to_distributed;
}  // namespace TiledArray

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

#ifndef TILEDARRAY_MATH_LINALG_DISPATCH_W_TTG
#if TILEDARRAY_HAS_TTG && TILEDARRAY_HAS_SCALAPACK
#define TILEDARRAY_MATH_LINALG_DISPATCH_W_TTG(FN, MATRIX)           \
  TA_MAX_THREADS;                                                   \
  if (get_linalg_backend() == LinearAlgebraBackend::TTG ||          \
      TiledArray::math::linalg::detail::prefer_distributed(MATRIX)) \
    return TiledArray::math::linalg::ttg::FN;                       \
  if (get_linalg_backend() == LinearAlgebraBackend::ScaLAPACK ||    \
      TiledArray::math::linalg::detail::prefer_distributed(MATRIX)) \
    return scalapack::FN;                                           \
  return non_distributed::FN;
#elif TILEDARRAY_HAS_TTG && !TILEDARRAY_HAS_SCALAPACK
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
