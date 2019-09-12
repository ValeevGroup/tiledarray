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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  eigen.h
 *  Sep 16, 2013
 *
 */

#ifndef TILEDARRAY_MATH_EIGEN_H__INCLUDED
#define TILEDARRAY_MATH_EIGEN_H__INCLUDED

#include <madness/config.h>
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC system_header
#endif
#if HAVE_INTEL_MKL
# define EIGEN_USE_MKL_ALL 1
//# define MKL_DIRECT_CALL 1
#endif
#include <Eigen/Core>
#include <Eigen/QR>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <TiledArray/error.h>

namespace TiledArray {
  namespace math {


    /// Construct a const Eigen::Map object for a given Tensor object

    /// \tparam T The element type
    /// \param t The buffer pointer
    /// \param m The number of rows in the result matrix
    /// \param n The number of columns in the result matrix
    /// \return An m x n Eigen matrix map for \c tensor
    /// \throw TiledArray::Exception When m * n is not equal to \c tensor size
    template <typename T>
    inline Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
    eigen_map(const T* t, const std::size_t m, const std::size_t n) {
      TA_ASSERT(t);
      TA_ASSERT(m > 0);
      TA_ASSERT(n > 0);
      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::RowMajor>, Eigen::AutoAlign>(t, m, n);
    }

    /// Construct an Eigen::Map object for a given Tensor object

    /// \tparam T The tensor element type
    /// \param t The tensor object
    /// \param m The number of rows in the result matrix
    /// \param n The number of columns in the result matrix
    /// \return An m x n Eigen matrix map for \c tensor
    /// \throw TiledArray::Exception When m * n is not equal to \c tensor size
    template <typename T>
    inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
    eigen_map(T* t, const std::size_t m, const std::size_t n) {
      TA_ASSERT(t);
      TA_ASSERT(m > 0);
      TA_ASSERT(n > 0);
      return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::RowMajor>, Eigen::AutoAlign>(t, m, n);
    }

    /// Construct a const Eigen::Map object for a given Tensor object

    /// \tparam T The element type
    /// \param t The vector pointer
    /// \param n The number of elements in the result matrix
    /// \return An n element Eigen vector map for \c tensor
    /// \throw TiledArray::Exception When n is not equal to \c tensor size
    template <typename T>
    inline Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::AutoAlign>
    eigen_map(const T* t, const std::size_t n) {
      TA_ASSERT(t);
      TA_ASSERT(n > 0);
      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::AutoAlign>(t, n);
    }

    /// Construct an Eigen::Map object for a given Tensor object

    /// \tparam T The element type
    /// \param t The vector pointer
    /// \param n The number of elements in the result matrix
    /// \return An n element Eigen vector map for \c tensor
    /// \throw TiledArray::Exception When m * n is not equal to \c tensor size
    template <typename T>
    inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::AutoAlign>
    eigen_map(T* t, const std::size_t n) {
      TA_ASSERT(t);
      TA_ASSERT(n > 0);
      return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::AutoAlign>(t, n);
    }

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_MATH_EIGEN_H__INCLUDED
