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
 *  May 02, 2015
 *
 */

#ifndef TILEDARRAY_CONVERSIONS_EIGEN_H__INCLUDED
#define TILEDARRAY_CONVERSIONS_EIGEN_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/external/madness.h>
#include <TiledArray/math/eigen.h>
#include <TiledArray/pmap/replicated_pmap.h>
#include <TiledArray/tensor.h>
#include <tiledarray_fwd.h>
#include <cstdint>
#include "TiledArray/dist_array.h"

namespace TiledArray {

// Convenience typedefs
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    EigenMatrixXd;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    EigenMatrixXf;
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
    EigenMatrixXcd;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
    EigenMatrixXcf;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    EigenMatrixXi;
typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    EigenMatrixXl;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> EigenVectorXd;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> EigenVectorXf;
typedef Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic> EigenVectorXcd;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic> EigenVectorXcf;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> EigenVectorXi;
typedef Eigen::Matrix<long, Eigen::Dynamic, 1> EigenVectorXl;

/// Construct a const Eigen::Map object for a given Tensor object

/// \tparam T A contiguous tensor type, e.g. TiledArray::Tensor ; namely, \c
/// TiledArray::detail::is_contiguous_tensor_v<T> must be true \tparam Storage
/// the tensor layout, either Eigen::RowMajor (default) or Eigen::ColMajor
/// \param tensor The tensor object, laid out according to Storage
/// \param m The number of rows in the result matrix
/// \param n The number of columns in the result matrix
/// \return An m x n Eigen matrix map for \c tensor
/// \throw TiledArray::Exception When m * n is not equal to \c tensor size
template <typename T, Eigen::StorageOptions Storage = Eigen::RowMajor,
          std::enable_if_t<detail::is_contiguous_tensor_v<T>>* = nullptr>
inline Eigen::Map<const Eigen::Matrix<typename T::value_type, Eigen::Dynamic,
                                      Eigen::Dynamic, Storage>,
                  Eigen::AutoAlign>
eigen_map(const T& tensor, const std::size_t m, const std::size_t n) {
  TA_ASSERT((m * n) == tensor.size());

  return Eigen::Map<const Eigen::Matrix<typename T::value_type, Eigen::Dynamic,
                                        Eigen::Dynamic, Storage>,
                    Eigen::AutoAlign>(tensor.data(), m, n);
}

/// Construct an Eigen::Map object for a given Tensor object

/// \tparam T A contiguous tensor type, e.g. TiledArray::Tensor ; namely, \c
/// TiledArray::detail::is_contiguous_tensor_v<T> must be true \tparam Storage
/// the tensor layout, either Eigen::RowMajor (default) or Eigen::ColMajor
/// \param tensor The tensor object, laid out according to Storage
/// \param m The number of rows in the result matrix
/// \param n The number of columns in the result matrix
/// \return An m x n Eigen matrix map for \c tensor
/// \throw TiledArray::Exception When m * n is not equal to \c tensor size
template <typename T, Eigen::StorageOptions Storage = Eigen::RowMajor,
          std::enable_if_t<detail::is_contiguous_tensor_v<T>>* = nullptr>
inline Eigen::Map<Eigen::Matrix<typename T::value_type, Eigen::Dynamic,
                                Eigen::Dynamic, Storage>,
                  Eigen::AutoAlign>
eigen_map(T& tensor, const std::size_t m, const std::size_t n) {
  TA_ASSERT((m * n) == tensor.size());

  return Eigen::Map<Eigen::Matrix<typename T::value_type, Eigen::Dynamic,
                                  Eigen::Dynamic, Storage>,
                    Eigen::AutoAlign>(tensor.data(), m, n);
}

/// Construct a const Eigen::Map object for a given Tensor object

/// \tparam T A contiguous tensor type, e.g. TiledArray::Tensor ; namely, \c
/// TiledArray::detail::is_contiguous_tensor_v<T> must be true \param tensor The
/// tensor object \param n The number of elements in the result matrix \return
/// An n element Eigen vector map for \c tensor \throw TiledArray::Exception
/// When n is not equal to \c tensor size
template <typename T,
          std::enable_if_t<detail::is_contiguous_tensor_v<T>>* = nullptr>
inline Eigen::Map<
    const Eigen::Matrix<typename T::value_type, Eigen::Dynamic, 1>,
    Eigen::AutoAlign>
eigen_map(const T& tensor, const std::size_t n) {
  TA_ASSERT(n == tensor.size());

  return Eigen::Map<
      const Eigen::Matrix<typename T::value_type, Eigen::Dynamic, 1>,
      Eigen::AutoAlign>(tensor.data(), n);
}

/// Construct an Eigen::Map object for a given Tensor object

/// \tparam T A tensor type, e.g. TiledArray::Tensor
/// \param tensor The tensor object
/// \param n The number of elements in the result matrix
/// \return An n element Eigen vector map for \c tensor
/// \throw TiledArray::Exception When n is not equal to \c tensor size
template <typename T,
          std::enable_if_t<detail::is_contiguous_tensor_v<T>>* = nullptr>
inline Eigen::Map<Eigen::Matrix<typename T::value_type, Eigen::Dynamic, 1>,
                  Eigen::AutoAlign>
eigen_map(T& tensor, const std::size_t n) {
  TA_ASSERT(n == tensor.size());

  return Eigen::Map<Eigen::Matrix<typename T::value_type, Eigen::Dynamic, 1>,
                    Eigen::AutoAlign>(tensor.data(), n);
}

/// Construct a const Eigen::Map object for a given Tensor object

/// The dimensions of the result tensor are extracted from the tensor itself
/// \tparam T A contiguous tensor type, e.g. TiledArray::Tensor ; namely, \c
/// TiledArray::detail::is_contiguous_tensor_v<T> must be true \tparam Storage
/// the tensor layout, either Eigen::RowMajor (default) or Eigen::ColMajor
/// \param tensor The tensor object, laid out according to Storage
/// \return An Eigen matrix map for \c tensor
/// \throw TiledArray::Exception When \c tensor dimensions are not equal to 2
/// or 1.
template <typename T, Eigen::StorageOptions Storage = Eigen::RowMajor,
          std::enable_if_t<detail::is_contiguous_tensor_v<T>>* = nullptr>
inline Eigen::Map<const Eigen::Matrix<typename T::value_type, Eigen::Dynamic,
                                      Eigen::Dynamic, Storage>,
                  Eigen::AutoAlign>
eigen_map(const T& tensor) {
  TA_ASSERT((tensor.range().rank() == 2u) || (tensor.range().rank() == 1u));
  const auto* MADNESS_RESTRICT const tensor_extent =
      tensor.range().extent_data();
  return eigen_map<T, Storage>(
      tensor, tensor_extent[0],
      (tensor.range().rank() == 2u ? tensor_extent[1] : 1ul));
}

/// Construct an Eigen::Map object for a given Tensor object

/// The dimensions of the result tensor are extracted from the tensor itself
/// \tparam T A contiguous tensor type, e.g. TiledArray::Tensor ; namely, \c
/// TiledArray::detail::is_contiguous_tensor_v<T> must be true \tparam Storage
/// the tensor layout, either Eigen::RowMajor (default) or Eigen::ColMajor
/// \param tensor The tensor object, laid out according to Storage
/// \return An Eigen matrix map for \c tensor
/// \throw When \c tensor dimensions are not equal to 2 or 1.
template <typename T, Eigen::StorageOptions Storage = Eigen::RowMajor,
          std::enable_if_t<detail::is_contiguous_tensor_v<T>>* = nullptr>
inline Eigen::Map<Eigen::Matrix<typename T::value_type, Eigen::Dynamic,
                                Eigen::Dynamic, Storage>,
                  Eigen::AutoAlign>
eigen_map(T& tensor) {
  TA_ASSERT((tensor.range().rank() == 2u) || (tensor.range().rank() == 1u));
  const auto* MADNESS_RESTRICT const tensor_extent =
      tensor.range().extent_data();
  return eigen_map<T, Storage>(
      tensor, tensor_extent[0],
      (tensor.range().rank() == 2u ? tensor_extent[1] : 1ul));
}

/// Copy a block of an Eigen matrix into a tensor

/// A block of \c matrix will be copied into \c tensor. The block
/// dimensions will be determined by the dimensions of the tensor's range.
/// \tparam T A tensor type, e.g. TiledArray::Tensor
/// \tparam Derived The derived type of an Eigen matrix
/// \param[in] matrix The object that will be assigned the content of \c tensor
/// \param[out] tensor The object that will be assigned the content of \c matrix
/// \throw TiledArray::Exception When the dimensions of \c tensor are not equal
/// to 1 or 2.
/// \throw TiledArray::Exception When the range of \c tensor is outside the
/// range of \c matrix .
template <typename T, typename Derived,
          std::enable_if_t<detail::is_contiguous_tensor_v<T>>* = nullptr>
inline void eigen_submatrix_to_tensor(const Eigen::MatrixBase<Derived>& matrix,
                                      T& tensor) {
  [[maybe_unused]] typedef typename T::index1_type size_type;
  TA_ASSERT((tensor.range().rank() == 2u) || (tensor.range().rank() == 1u));

  // Get pointers to the tensor range data
  const auto* MADNESS_RESTRICT const tensor_lower =
      tensor.range().lobound_data();
  const auto* MADNESS_RESTRICT const tensor_upper =
      tensor.range().upbound_data();
  const auto* MADNESS_RESTRICT const tensor_extent =
      tensor.range().extent_data();

  if (tensor.range().rank() == 2u) {
    // Get tensor range data
    const std::size_t tensor_lower_0 = tensor_lower[0];
    const std::size_t tensor_lower_1 = tensor_lower[1];
    [[maybe_unused]] const std::size_t tensor_upper_0 = tensor_upper[0];
    [[maybe_unused]] const std::size_t tensor_upper_1 = tensor_upper[1];
    const std::size_t tensor_extent_0 = tensor_extent[0];
    const std::size_t tensor_extent_1 = tensor_extent[1];

    TA_ASSERT(tensor_upper_0 <= size_type(matrix.rows()));
    TA_ASSERT(tensor_upper_1 <= size_type(matrix.cols()));

    // Copy matrix
    eigen_map(tensor, tensor_extent_0, tensor_extent_1) = matrix.block(
        tensor_lower_0, tensor_lower_1, tensor_extent_0, tensor_extent_1);
  } else {
    // Get tensor range data
    const std::size_t tensor_lower_0 = tensor_lower[0];
    [[maybe_unused]] const std::size_t tensor_upper_0 = tensor_upper[0];
    const std::size_t tensor_extent_0 = tensor_extent[0];

    // Check that matrix is a vector.
    TA_ASSERT((matrix.rows() == 1) || (matrix.cols() == 1));

    if (matrix.rows() == 1) {
      TA_ASSERT(tensor_upper_0 <= size_type(matrix.cols()));

      // Copy the row vector to tensor
      eigen_map(tensor, 1, tensor_extent_0) =
          matrix.block(0, tensor_lower_0, 1, tensor_extent_0);
    } else {
      TA_ASSERT(tensor_upper_0 <= size_type(matrix.rows()));

      // Copy the column vector to tensor
      eigen_map(tensor, tensor_extent_0, 1) =
          matrix.block(tensor_lower_0, 0, tensor_extent_0, 1);
    }
  }
}

/// Copy the content of a tensor into an Eigen matrix block

/// The content of tensor will be copied into a block of matrix. The block
/// dimensions will be determined by the dimensions of the tensor's range.
/// \tparam T A tensor type, e.g. TiledArray::Tensor
/// \tparam Derived The derived type of an Eigen matrix
/// \param[in] tensor The object that will be copied to \c matrix
/// \param[out] matrix The object that will be assigned the content of \c tensor
/// \throw TiledArray::Exception When the dimensions of \c tensor are not equal
/// to 1 or 2.
/// \throw TiledArray::Exception When the range of \c tensor is outside the
/// range of \c matrix .
template <typename T, typename Derived,
          std::enable_if_t<detail::is_contiguous_tensor_v<T>>* = nullptr>
inline void tensor_to_eigen_submatrix(const T& tensor,
                                      Eigen::MatrixBase<Derived>& matrix) {
  [[maybe_unused]] typedef typename T::index1_type size_type;
  TA_ASSERT((tensor.range().rank() == 2u) || (tensor.range().rank() == 1u));

  // Get pointers to the tensor range data
  const auto* MADNESS_RESTRICT const tensor_lower =
      tensor.range().lobound_data();
  const auto* MADNESS_RESTRICT const tensor_upper =
      tensor.range().upbound_data();
  const auto* MADNESS_RESTRICT const tensor_extent =
      tensor.range().extent_data();

  if (tensor.range().rank() == 2) {
    // Get tensor range data
    const std::size_t tensor_lower_0 = tensor_lower[0];
    const std::size_t tensor_lower_1 = tensor_lower[1];
    [[maybe_unused]] const std::size_t tensor_upper_0 = tensor_upper[0];
    [[maybe_unused]] const std::size_t tensor_upper_1 = tensor_upper[1];
    const std::size_t tensor_extent_0 = tensor_extent[0];
    const std::size_t tensor_extent_1 = tensor_extent[1];

    TA_ASSERT(tensor_upper_0 <= size_type(matrix.rows()));
    TA_ASSERT(tensor_upper_1 <= size_type(matrix.cols()));

    // Copy tensor into matrix
    matrix.block(tensor_lower_0, tensor_lower_1, tensor_extent_0,
                 tensor_extent_1) =
        eigen_map(tensor, tensor_extent_0, tensor_extent_1);
  } else {
    // Get tensor range data
    const std::size_t tensor_lower_0 = tensor_lower[0];
    [[maybe_unused]] const std::size_t tensor_upper_0 = tensor_upper[0];
    const std::size_t tensor_extent_0 = tensor_extent[0];

    TA_ASSERT((matrix.rows() == 1) || (matrix.cols() == 1));

    if (matrix.rows() == 1) {
      TA_ASSERT(tensor_upper_0 <= size_type(matrix.cols()));

      // Copy tensor into row vector
      matrix.block(0, tensor_lower_0, 1, tensor_extent_0) =
          eigen_map(tensor, 1, tensor_extent_0);
    } else {
      TA_ASSERT(tensor_upper_0 <= size_type(matrix.rows()));

      // Copy tensor into column vector
      matrix.block(tensor_lower_0, 0, tensor_extent_0, 1) =
          eigen_map(tensor, tensor_extent_0, 1);
    }
  }
}

namespace detail {

/// Task function for converting Eigen submatrix to a tensor

/// \tparam A Array type
/// \tparam Derived The matrix type
/// \param matrix The matrix that will be copied
/// \param array The array that will hold the result
/// \param i The index of the tile to be copied
/// \param counter The task counter
template <typename A, typename Derived>
void counted_eigen_submatrix_to_tensor(const Eigen::MatrixBase<Derived>* matrix,
                                       A* array,
                                       const typename A::ordinal_type i,
                                       madness::AtomicInt* counter) {
  typename A::value_type tensor(array->trange().make_tile_range(i));
  eigen_submatrix_to_tensor(*matrix, tensor);
  array->set(i, tensor);
  (*counter)++;
}

/// Task function for assigning a tensor to an Eigen submatrix

/// \tparam Derived The matrix type
/// \tparam T Tensor type
/// \param matrix The matrix to be assigned
/// \param tensor The tensor to be copied
/// \param counter The task counter
template <typename Derived, typename T>
void counted_tensor_to_eigen_submatrix(const T& tensor,
                                       Eigen::MatrixBase<Derived>* matrix,
                                       madness::AtomicInt* counter) {
  tensor_to_eigen_submatrix(tensor, *matrix);
  (*counter)++;
}

}  // namespace detail

// clang-format off
/// Convert an Eigen matrix into an Array object

/// This function will copy the content of \c matrix into an \c Array object
/// that is tiled according to the \c trange object. The copy operation is
/// done in parallel, and this function will block until all elements of
/// \c matrix have been copied into the result array tiles.
/// Each tile is created
/// using the local contents of \c matrix, hence
/// it is your responsibility to ensure that the data in \c matrix
/// is distributed correctly among the ranks. If in doubt, you should replicate
/// \c matrix among the ranks prior to calling this.
///
/// Usage:
/// \code
/// Eigen::MatrixXd m(100, 100);
/// // Fill m with data ...
///
/// // Create a range for the new array object
/// std::vector<std::size_t> blocks;
/// for(std::size_t i = 0ul; i <= 100ul; i += 10ul)
///   blocks.push_back(i);
/// std::array<TiledArray::TiledRange1, 2> blocks2 =
///     {{ TiledArray::TiledRange1(blocks.begin(), blocks.end()),
///        TiledArray::TiledRange1(blocks.begin(), blocks.end()) }};
/// TiledArray::TiledRange trange(blocks2.begin(), blocks2.end());
///
/// // Create an Array from an Eigen matrix.
/// TiledArray::Array<double, 2> array =
///     eigen_to_array<TiledArray::Array<double, 2> >(world, trange, m);
/// \endcode
/// \tparam A The array type
/// \tparam Derived The Eigen matrix derived type
/// \param world The world where the array will live
/// \param trange The tiled range of the new array
/// \param matrix The Eigen matrix to be copied
/// \param replicated if true, the result will be replicated
/// [default = true].
/// \param pmap the process map object [default=null]; initialized to the
/// default if \p replicated is false, or a replicated pmap if \p replicated
/// is true; ignored if \p replicated is true and \c world.size()>1
/// \return An \c Array object that is a copy of \c matrix
// clang-format on
template <typename A, typename Derived>
A eigen_to_array(World& world, const typename A::trange_type& trange,
                 const Eigen::MatrixBase<Derived>& matrix,
                 bool replicated = false,
                 std::shared_ptr<typename A::pmap_interface> pmap = {}) {
  typedef typename A::index1_type size_type;
  // Check that trange matches the dimensions of other
  if ((matrix.cols() > 1) && (matrix.rows() > 1)) {
    TA_USER_ASSERT(trange.tiles_range().rank() == 2,
                   "TiledArray::eigen_to_array(): The number of dimensions in "
                   "trange is not equal to that of the Eigen matrix.");
    TA_USER_ASSERT(
        trange.elements_range().extent(0) == size_type(matrix.rows()),
        "TiledArray::eigen_to_array(): The number of rows in trange is not "
        "equal to the number of rows in the Eigen matrix.");
    TA_USER_ASSERT(
        trange.elements_range().extent(1) == size_type(matrix.cols()),
        "TiledArray::eigen_to_array(): The number of columns in trange is not "
        "equal to the number of columns in the Eigen matrix.");
  } else {
    TA_USER_ASSERT(trange.tiles_range().rank() == 1,
                   "TiledArray::eigen_to_array(): The number of dimensions in "
                   "trange must match that of the Eigen matrix.");
    TA_USER_ASSERT(
        trange.elements_range().extent(0) == size_type(matrix.size()),
        "TiledArray::eigen_to_array(): The size of trange must be equal to the "
        "matrix size.");
  }

  // Create a new tensor
  if (replicated && (world.size() > 1))
    pmap = std::static_pointer_cast<typename A::pmap_interface>(
        std::make_shared<detail::ReplicatedPmap>(
            world, trange.tiles_range().volume()));
  A array = (pmap ? A(world, trange, pmap) : A(world, trange));

  // Spawn tasks to copy Eigen to an Array
  madness::AtomicInt counter;
  counter = 0;
  std::int64_t n = 0;
  for (std::size_t i = 0; i < array.size(); ++i) {
    if (array.is_local(i)) {
      world.taskq.add(&detail::counted_eigen_submatrix_to_tensor<A, Derived>,
                      &matrix, &array, i, &counter);
      ++n;
    }
  }

  // Wait until the write tasks are complete
  array.world().await([&counter, n]() { return counter == n; });

  // truncate, n.b. this can replace the wait above
  array.truncate();

  return array;
}

// clang-format off
/// Convert an Array object into an Eigen matrix object

/// This function will copy the content of an \c Array object into matrix. The
/// copy operation is done in parallel, and this function will block until
/// all elements of \c array have been copied into the result matrix. The size
/// of world must be exactly equal to 1, or \c array must be a replicated
/// object.
/// Usage:
/// \code
/// TiledArray::Array<double, 2> array(world, trange);
/// // Set tiles of array ...
///
/// Eigen::MatrixXd m = array_to_eigen(array);
/// \endcode
/// \tparam Tile The array tile type
/// \tparam EigenStorageOrder The storage order of the resulting Eigen::Matrix
///      object; the default is Eigen::ColMajor, i.e. the column-major storage
/// \param array The array to be converted. It must be replicated if using 2 or
/// more World ranks.
/// \return an Eigen matrix; it will contain same data on each
/// World rank.
/// \throw TiledArray::Exception When world size is greater than 1
/// and \c array is not replicated.
/// \throw TiledArray::Exception When the number
/// of dimensions of \c array is not equal to 1 or 2.
// clang-format on
template <typename Tile, typename Policy,
          unsigned int EigenStorageOrder = Eigen::ColMajor>
Eigen::Matrix<typename Tile::value_type, Eigen::Dynamic, Eigen::Dynamic,
              EigenStorageOrder>
array_to_eigen(const DistArray<Tile, Policy>& array) {
  typedef Eigen::Matrix<typename Tile::value_type, Eigen::Dynamic,
                        Eigen::Dynamic, EigenStorageOrder>
      EigenMatrix;

  const auto rank = array.trange().tiles_range().rank();

  // Check that the array will fit in a matrix or vector
  TA_USER_ASSERT((rank == 2u) || (rank == 1u),
                 "TiledArray::array_to_eigen(): The array dimensions must be "
                 "equal to 1 or 2.");

  // Check that this is not a distributed computing environment or that the
  // array is replicated
  if (!array.pmap()->is_replicated())
    TA_USER_ASSERT(array.world().size() == 1,
                   "TiledArray::array_to_eigen(): non-replicated Array cannot "
                   "be assigned to an Eigen::Matrix when the number of World "
                   "ranks is greater than 1.");

  // Construct the Eigen matrix
  const auto* MADNESS_RESTRICT const array_extent =
      array.trange().elements_range().extent_data();
  // if array is sparse must initialize to zero
  EigenMatrix matrix =
      EigenMatrix::Zero(array_extent[0], (rank == 2 ? array_extent[1] : 1));

  // Spawn tasks to copy array tiles to the Eigen matrix
  madness::AtomicInt counter;
  counter = 0;
  int n = 0;
  for (std::size_t i = 0; i < array.size(); ++i) {
    if (!array.is_zero(i)) {
      array.world().taskq.add(
          &detail::counted_tensor_to_eigen_submatrix<
              EigenMatrix, typename DistArray<Tile, Policy>::value_type>,
          array.find(i), &matrix, &counter);
      ++n;
    }
  }

  // Wait until the above tasks are complete. Tasks will be processed by this
  // thread while waiting.
  array.world().await([&counter, n]() { return counter == n; });

  return matrix;
}

/// Convert a row-major matrix buffer into an Array object

/// This function will copy the content of \c buffer into an \c Array object
/// that is tiled according to the \c trange object. The copy operation is
/// done in parallel, and this function will block until all elements of
/// \c matrix have been copied into the result array tiles.
/// Each tile is created
/// using the local contents of \c matrix, hence
/// it is your responsibility to ensure that the data in \c matrix
/// is distributed correctly among the ranks. If in doubt, you should replicate
/// \c matrix among the ranks prior to calling this.
///
/// Usage:
/// \code
/// double* buffer = new double[100 * 100];
/// // Fill buffer with data ...
///
/// // Create a range for the new array object
/// std::vector<std::size_t> blocks;
/// for(std::size_t i = 0ul; i <= 100ul; i += 10ul)
///   blocks.push_back(i);
/// std::array<TiledArray::TiledRange1, 2> blocks2 =
///     {{ TiledArray::TiledRange1(blocks.begin(), blocks.end()),
///        TiledArray::TiledRange1(blocks.begin(), blocks.end()) }};
/// TiledArray::TiledRange trange(blocks2.begin(), blocks2.end());
///
/// // Create an Array from an Eigen matrix.
/// TiledArray::Array<double, 2> array =
///     row_major_buffer_to_array<TiledArray::Array<double, 2> >(world, trange,
///     buffer, 100, 100);
///
/// delete [] buffer;
/// \endcode
/// \tparam A The array type
/// \param world The world where the array will live
/// \param trange The tiled range of the new array
/// \param buffer The row-major matrix buffer to be copied
/// \param m The number of rows in the matrix
/// \param n The number of columns in the matrix
/// \param replicated if true, the result will be replicated
/// [default = true].
/// \param pmap the process map object [default=null]; initialized to the
/// default if \p replicated is false, or a replicated pmap if \p replicated
/// is true; ignored if \p replicated is true and \c world.size()>1
/// \return An \c Array object that is a copy of \c matrix
/// \throw TiledArray::Exception When \c m and \c n are not equal to the
/// number of rows or columns in tiled range.
template <typename A>
inline A row_major_buffer_to_array(
    World& world, const typename A::trange_type& trange,
    const typename A::value_type::value_type* buffer, const std::size_t m,
    const std::size_t n, const bool replicated = false,
    std::shared_ptr<typename A::pmap_interface> pmap = {}) {
  TA_USER_ASSERT(trange.elements_range().extent(0) == m,
                 "TiledArray::eigen_to_array(): The number of rows in trange "
                 "is not equal to m.");
  TA_USER_ASSERT(trange.elements_range().extent(1) == n,
                 "TiledArray::eigen_to_array(): The number of columns in "
                 "trange is not equal to n.");

  typedef Eigen::Matrix<typename A::value_type::value_type, Eigen::Dynamic,
                        Eigen::Dynamic, Eigen::RowMajor>
      matrix_type;
  return eigen_to_array(
      world, trange,
      Eigen::Map<const matrix_type, Eigen::AutoAlign>(buffer, m, n), replicated,
      pmap);
}

/// Convert a column-major matrix buffer into an Array object

/// This function will copy the content of \c buffer into an \c Array object
/// that is tiled according to the \c trange object. The copy operation is
/// done in parallel, and this function will block until all elements of
/// \c matrix have been copied into the result array tiles.
/// Each tile is created
/// using the local contents of \c matrix, hence
/// it is your responsibility to ensure that the data in \c matrix
/// is distributed correctly among the ranks. If in doubt, you should replicate
/// \c matrix among the ranks prior to calling this.
///
/// Usage:
/// \code
/// double* buffer = new double[100 * 100];
/// // Fill buffer with data ...
///
/// // Create a range for the new array object
/// std::vector<std::size_t> blocks;
/// for(std::size_t i = 0ul; i <= 100ul; i += 10ul)
///   blocks.push_back(i);
/// std::array<TiledArray::TiledRange1, 2> blocks2 =
///     {{ TiledArray::TiledRange1(blocks.begin(), blocks.end()),
///        TiledArray::TiledRange1(blocks.begin(), blocks.end()) }};
/// TiledArray::TiledRange trange(blocks2.begin(), blocks2.end());
///
/// // Create an Array from an Eigen matrix.
/// TiledArray::Array<double, 2> array =
///     column_major_buffer_to_array<TiledArray::Array<double, 2> >(world,
///     trange, buffer, 100, 100);
///
/// delete [] buffer;
/// \endcode
/// \tparam A The array type
/// \param world The world where the array will live
/// \param trange The tiled range of the new array
/// \param buffer The row-major matrix buffer to be copied
/// \param m The number of rows in the matrix
/// \param n The number of columns in the matrix
/// \param replicated if true, the result will be replicated
/// [default = true].
/// \param pmap the process map object [default=null]; initialized to the
/// default if \p replicated is false, or a replicated pmap if \p replicated
/// is true; ignored if \p replicated is true and \c world.size()>1
/// \return An \c Array object that is a copy of \c matrix
/// \throw TiledArray::Exception When \c m and \c n are not equal to the
/// number of rows or columns in tiled range.
template <typename A>
inline A column_major_buffer_to_array(
    World& world, const typename A::trange_type& trange,
    const typename A::value_type::value_type* buffer, const std::size_t m,
    const std::size_t n, const bool replicated = false,
    std::shared_ptr<typename A::pmap_interface> pmap = {}) {
  TA_USER_ASSERT(trange.elements_range().extent(0) == m,
                 "TiledArray::eigen_to_array(): The number of rows in trange "
                 "is not equal to m.");
  TA_USER_ASSERT(trange.elements_range().extent(1) == n,
                 "TiledArray::eigen_to_array(): The number of columns in "
                 "trange is not equal to n.");

  typedef Eigen::Matrix<typename A::value_type::value_type, Eigen::Dynamic,
                        Eigen::Dynamic, Eigen::ColMajor>
      matrix_type;
  return eigen_to_array(
      world, trange,
      Eigen::Map<const matrix_type, Eigen::AutoAlign>(buffer, m, n), replicated,
      pmap);
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CONVERSIONS_EIGEN_H__INCLUDED
