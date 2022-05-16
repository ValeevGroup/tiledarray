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
#include <TiledArray/external/eigen.h>
#include <TiledArray/external/madness.h>
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

    TA_ASSERT(tensor_upper_0 <= std::size_t(matrix.rows()));
    TA_ASSERT(tensor_upper_1 <= std::size_t(matrix.cols()));

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
      TA_ASSERT(tensor_upper_0 <= std::size_t(matrix.cols()));

      // Copy the row vector to tensor
      eigen_map(tensor, 1, tensor_extent_0) =
          matrix.block(0, tensor_lower_0, 1, tensor_extent_0);
    } else {
      TA_ASSERT(tensor_upper_0 <= std::size_t(matrix.rows()));

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

    TA_ASSERT(tensor_upper_0 <= std::size_t(matrix.rows()));
    TA_ASSERT(tensor_upper_1 <= std::size_t(matrix.cols()));

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
      TA_ASSERT(tensor_upper_0 <= std::size_t(matrix.cols()));

      // Copy tensor into row vector
      matrix.block(0, tensor_lower_0, 1, tensor_extent_0) =
          eigen_map(tensor, 1, tensor_extent_0);
    } else {
      TA_ASSERT(tensor_upper_0 <= std::size_t(matrix.rows()));

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
/// auto array =
///     eigen_to_array<TA::TSpArrayD> >(world, trange, m);
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
  const auto rank  = trange.tiles_range().rank();

  TA_ASSERT(rank == 1 || rank == 2 && 
      "TiledArray::eigen_to_array(): The number of dimensions in "
      "trange must match that of the Eigen matrix.");

  if(rank == 2) {
    TA_ASSERT(
        trange.elements_range().extent(0) == size_type(matrix.rows()) &&
        "TiledArray::eigen_to_array(): The number of rows in trange is not "
        "equal to the number of rows in the Eigen matrix.");
    TA_ASSERT(
        trange.elements_range().extent(1) == size_type(matrix.cols()) &&
        "TiledArray::eigen_to_array(): The number of columns in trange is not "
        "equal to the number of columns in the Eigen matrix.");
  } else {
    TA_ASSERT(
        trange.elements_range().extent(0) == size_type(matrix.size()) &&
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
/// TA::TSpArrayD array(world, trange);
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
  TA_ASSERT((rank == 2u) ||
            (rank == 1u) &&
                "TiledArray::array_to_eigen(): The array dimensions must be "
                "equal to 1 or 2.");

  // Check that this is not a distributed computing environment or that the
  // array is replicated
  if (!array.pmap()->is_replicated())
    TA_ASSERT(array.world().size() == 1 &&
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
/// auto array =
///     row_major_buffer_to_array<TA::TSpArrayD>(world, trange,
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
  TA_ASSERT(trange.elements_range().extent(0) == m &&
            "TiledArray::eigen_to_array(): The number of rows in trange "
            "is not equal to m.");
  TA_ASSERT(trange.elements_range().extent(1) == n &&
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
/// auto array =
///     column_major_buffer_to_array<TA::TSpArrayD>(world,
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
  TA_ASSERT(trange.elements_range().extent(0) == m &&
            "TiledArray::eigen_to_array(): The number of rows in trange "
            "is not equal to m.");
  TA_ASSERT(trange.elements_range().extent(1) == n &&
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
/*
///////////////// Eigen::Tensor conversions ////////////////////////////////////

// clang-format off
/// Copy a block of a Eigen::Tensor into a (row-major) TiledArray::Tensor

/// A block of Eigen::Tensor \c src will be copied into TiledArray::Tensor \c
/// dst. The block dimensions will be determined by the dimensions of the range
/// of \c dst .
/// \tparam T The tensor element type
/// \tparam NumIndices_ The order of \p src
/// \tparam Options_
/// \tparam IndexType_
/// \tparam Tensor_ A tensor type (e.g., TiledArray::Tensor or btas::Tensor,
///         optionally wrapped into TiledArray::Tile)
/// \param[in] src The source object; its subblock defined by the {lower,upper}
///            bounds \c {dst.lobound(),dst.upbound()} will be copied to \c dst
/// \param[out] dst The object that will contain the contents of the
///             corresponding subblock of src
/// \throw TiledArray::Exception When the dimensions of \c src and \c dst do not
///        match.
// clang-format on
template <typename T, int NumIndices_, int Options_, typename IndexType_,
          typename Tensor_>
inline void eigen_subtensor_to_tensor(
    const Eigen::Tensor<T, NumIndices_, Options_, IndexType_>& src,
    Tensor_& dst) {
  TA_ASSERT(dst.range().rank() == NumIndices_);

  auto to_array = [](const auto& seq) {
    TA_ASSERT(seq.size() == NumIndices_);
    std::array<IndexType_, NumIndices_> result;
    std::copy(seq.begin(), seq.end(), result.begin());
    return result;
  };

  [[maybe_unused]] auto reverse_extent_indices = []() {
    std::array<IndexType_, NumIndices_> result;
    std::iota(result.rbegin(), result.rend(), 0);
    return result;
  };

  const auto& dst_range = dst.range();
  auto src_block =
      src.slice(to_array(dst_range.lobound()), to_array(dst_range.extent()));
  auto dst_eigen_map = Eigen::TensorMap<
      Eigen::Tensor<T, NumIndices_, Eigen::RowMajor, IndexType_>>(
      dst.data(), to_array(dst_range.extent()));
  if constexpr (static_cast<int>(std::decay_t<decltype(src)>::Layout) ==
                static_cast<int>(Eigen::ColMajor))
    dst_eigen_map = src_block.swap_layout().shuffle(reverse_extent_indices());
  else
    dst_eigen_map = src_block;
}

// clang-format off
/// Copy a (row-major) TiledArray::Tensor into a block of a Eigen::Tensor

/// TiledArray::Tensor \c src will be copied into a block of Eigen::Tensor
/// \c dst. The block dimensions will be determined by
/// the dimensions of the range of \c src .
/// \tparam Tensor_ A tensor type (e.g., TiledArray::Tensor or btas::Tensor,
///         optionally wrapped into TiledArray::Tile)
/// \tparam T The tensor element type
/// \tparam NumIndices_ The order of \p dst
/// \tparam Options_
/// \tparam IndexType_
/// \param[in] src The source object whose contents will be copied into
///            a subblock of \c dst
/// \param[out] dst The destination object; its subblock defined by the
///             {lower,upper} bounds \c {src.lobound(),src.upbound()} will be
///             overwritten with the content of \c src
/// \throw TiledArray::Exception When the dimensions
///        of \c src and \c dst do not match.
// clang-format on
template <typename Tensor_, typename T, int NumIndices_, int Options_,
          typename IndexType_>
inline void tensor_to_eigen_subtensor(
    const Tensor_& src,
    Eigen::Tensor<T, NumIndices_, Options_, IndexType_>& dst) {
  TA_ASSERT(src.range().rank() == NumIndices_);

  auto to_array = [](const auto& seq) {
    TA_ASSERT(seq.size() == NumIndices_);
    std::array<IndexType_, NumIndices_> result;
    std::copy(seq.begin(), seq.end(), result.begin());
    return result;
  };

  [[maybe_unused]] auto reverse_extent_indices = []() {
    std::array<IndexType_, NumIndices_> result;
    std::iota(result.rbegin(), result.rend(), 0);
    return result;
  };

  const auto& src_range = src.range();
  auto dst_block =
      dst.slice(to_array(src_range.lobound()), to_array(src_range.extent()));
  auto src_eigen_map = Eigen::TensorMap<
      Eigen::Tensor<const T, NumIndices_, Eigen::RowMajor, IndexType_>>(
      src.data(), to_array(src_range.extent()));
  if constexpr (static_cast<int>(std::decay_t<decltype(dst)>::Layout) ==
                static_cast<int>(Eigen::ColMajor))
    dst_block = src_eigen_map.swap_layout().shuffle(reverse_extent_indices());
  else
    dst_block = src_eigen_map;
}

namespace detail {

/// Task function for converting Eigen::Tensor subblock to a
/// TiledArray::DistArray

/// \tparam DistArray_ a TiledArray::DistArray type
/// \tparam Eigen_Tensor_ an Eigen::Tensor type
/// \param src The btas::Tensor object whose block will be copied
/// \param dst The array that will hold the result
/// \param i The index of the tile to be copied
/// \param counter The task counter
/// \internal OK to use bare ptrs as args as long as the user blocks on the
/// counter.
template <typename DistArray_, typename Eigen_Tensor_>
void counted_eigen_subtensor_to_tensor(const Eigen_Tensor_* src,
                                       DistArray_* dst,
                                       const typename Range::index_type i,
                                       madness::AtomicInt* counter) {
  typename DistArray_::value_type tensor(dst->trange().make_tile_range(i));
  eigen_subtensor_to_tensor(*src, tensor);
  dst->set(i, tensor);
  (*counter)++;
}

/// Task function for assigning a tensor to an Eigen subtensor

/// \tparam TA_Tensor_ a TiledArray::Tensor type
/// \tparam Eigen_Tensor_ an Eigen::Tensor type
/// \param src The source tensor
/// \param dst The destination tensor
/// \param counter The task counter
template <typename TA_Tensor_, typename Eigen_Tensor_>
void counted_tensor_to_eigen_subtensor(const TA_Tensor_& src,
                                       Eigen_Tensor_* dst,
                                       madness::AtomicInt* counter) {
  tensor_to_eigen_subtensor(src, *dst);
  (*counter)++;
}

template <bool sparse>
auto make_ta_shape(World& world, const TiledArray::TiledRange& trange);

template <>
inline auto make_ta_shape<true>(World& world,
                                const TiledArray::TiledRange& trange) {
  TiledArray::Tensor<float> tile_norms(trange.tiles_range(),
                                       std::numeric_limits<float>::max());
  return TiledArray::SparseShape<float>(world, tile_norms, trange);
}

template <>
inline auto make_ta_shape<false>(World&, const TiledArray::TiledRange&) {
  return TiledArray::DenseShape{};
}

}  // namespace detail

/// Convert a Eigen::Tensor object into a TiledArray::DistArray object

/// This function will copy the contents of \c src into a \c DistArray_ object
/// that is tiled according to the \c trange object. If the \c DistArray_ object
/// has sparse policy, a sparse map with large norm is created to ensure all the
/// values from \c src copy to the \c DistArray_ object. The copy operation is
/// done in parallel, and this function will block until all elements of
/// \c src have been copied into the result array tiles.
/// Each tile is created
/// using the local contents of \c src, hence
/// it is your responsibility to ensure that the data in \c src
/// is distributed correctly among the ranks. If in doubt, you should replicate
/// \c src among the ranks prior to calling this.
///
/// Upon completion,
/// if the \c DistArray_ object has sparse policy truncate() is called.\n
/// Usage:
/// \code
/// Eigen::Tensor<double, 3> src(100, 100, 100);
/// // Fill src with data ...
///
/// // Create a range for the new array object
/// std::vector<std::size_t> blocks;
/// for(std::size_t i = 0ul; i <= 100ul; i += 10ul)
///   blocks.push_back(i);
/// std::array<TiledArray::TiledRange1, 3> blocks3 =
///     {{ TiledArray::TiledRange1(blocks.begin(), blocks.end()),
///        TiledArray::TiledRange1(blocks.begin(), blocks.end()),
///        TiledArray::TiledRange1(blocks.begin(), blocks.end()) }};
/// TiledArray::TiledRange trange(blocks3.begin(), blocks3.end());
///
/// // Create an Array from the source btas::Tensor object
/// TiledArray::TArrayD array =
///     eigen_tensor_to_array<decltype(array)>(world, trange, src);
/// \endcode
/// \tparam DistArray_ a TiledArray::DistArray type
/// \tparam NumIndices_ The order of \p dst
/// \tparam Options_
/// \tparam IndexType_
/// \param[in,out] world The world where the result array will live
/// \param[in] trange The tiled range of the new array
/// \param[in] src The Eigen::Tensor object whose contents will be
/// copied to the result.
/// \param replicated if true, the result will be replicated
///        [default = false].
/// \param pmap the process map object [default=null]; initialized to the
/// default if \p replicated is false, or a replicated pmap if \p replicated
/// is true; ignored if \p replicated is true and \c world.size()>1
/// \return A \c DistArray_ object that is a copy of \c src
/// \throw TiledArray::Exception When world size is greater than 1
/// \note If using 2 or more World ranks, set \c replicated=true and make sure
/// \c matrix is the same on each rank!
template <typename DistArray_, typename T, int NumIndices_, int Options_,
          typename IndexType_>
DistArray_ eigen_tensor_to_array(
    World& world, const TiledArray::TiledRange& trange,
    const Eigen::Tensor<T, NumIndices_, Options_, IndexType_>& src,
    bool replicated = false,
    std::shared_ptr<typename DistArray_::pmap_interface> pmap = {}) {
  // Test preconditions
  const auto rank = trange.tiles_range().rank();
  TA_ASSERT(rank == NumIndices_ &&
            "TiledArray::eigen_tensor_to_array(): rank of destination "
            "trange does not match the rank of source Eigen tensor.");
  auto dst_range_extents = trange.elements_range().extent();
  for (std::remove_const_t<decltype(rank)> d = 0; d != rank; ++d) {
    TA_ASSERT(dst_range_extents[d] == src.dimension(d) &&
              "TiledArray::eigen_tensor_to_array(): source dimension does "
              "not match destination dimension.");
  }

  using Tensor_ = Eigen::Tensor<T, NumIndices_, Options_, IndexType_>;
  using Policy_ = typename DistArray_::policy_type;
  const auto is_sparse = !is_dense_v<Policy_>;

  // Make a shape, only used if making a sparse array
  using Shape_ = typename DistArray_::shape_type;
  Shape_ shape = detail::make_ta_shape<is_sparse>(world, trange);

  // Create a new tensor
  if (replicated && (world.size() > 1))
    pmap = std::static_pointer_cast<typename DistArray_::pmap_interface>(
        std::make_shared<detail::ReplicatedPmap>(
            world, trange.tiles_range().volume()));
  DistArray_ array = (pmap ? DistArray_(world, trange, shape, pmap)
                           : DistArray_(world, trange, shape));

  // Spawn copy tasks
  madness::AtomicInt counter;
  counter = 0;
  std::int64_t n = 0;
  for (auto&& acc : array) {
    world.taskq.add(
        &detail::counted_eigen_subtensor_to_tensor<DistArray_, Tensor_>, &src,
        &array, acc.index(), &counter);
    ++n;
  }

  // Wait until the write tasks are complete
  array.world().await([&counter, n]() { return counter == n; });

  // Analyze tiles norms and truncate based on sparse policy
  if (is_sparse) truncate(array);

  return array;
}

/// Convert a TiledArray::DistArray object into a Eigen::Tensor object

/// This function will copy the contents of \c src into a \c Eigen::Tensor
/// object. The copy operation is done in parallel, and this function will block
/// until all elements of \c src have been copied into the result array tiles.
/// The size of \c src.world().size() must be equal to 1 or \c src must be a
/// replicated TiledArray::DistArray. Usage:
/// \code
/// TiledArray::TArrayD
/// array(world, trange);
/// // Set tiles of array ...
///
/// auto t = array_to_eigen_tensor(array);
/// \endcode
/// \tparam Tile the tile type of \c src
/// \tparam Policy the policy type of \c src
/// \param[in] src The TiledArray::DistArray<Tile,Policy> object whose contents
/// will be copied to the result.
/// \return A \c Eigen::Tensor object that is a copy of \c src
/// \throw TiledArray::Exception When world size is greater than
///        1 and \c src is not replicated
/// \param[in] target_rank the rank on which to create the Eigen:Tensor
///            containing the data of \c src ; if \c target_rank=-1 then
///            create the Eigen::Tensor on every rank (this requires
///            that \c src.is_replicated()==true )
/// \return Eigen::Tensor object containing the data of \c src , if my rank
/// equals
///         \c target_rank or \c target_rank==-1 ,
///         default-initialized Eigen::Tensor otherwise.
template <typename Tensor, typename Tile, typename Policy>
Tensor array_to_eigen_tensor(const TiledArray::DistArray<Tile, Policy>& src,
                             int target_rank = -1) {
  // Test preconditions
  if (target_rank == -1 && src.world().size() > 1 &&
      !src.pmap()->is_replicated())
    TA_ASSERT(
        src.world().size() == 1 &&
        "TiledArray::array_to_eigen_tensor(): a non-replicated array can only "
        "be converted to a Eigen::Tensor on every rank if the number of World "
        "ranks is 1.");

  using result_type = Tensor;

  // Construct the result
  if (target_rank == -1 || src.world().rank() == target_rank) {
    // if array is sparse must initialize to zero
    result_type result(src.trange().elements_range().extent());
    result.setZero();

    // Spawn tasks to copy array tiles to btas::Tensor
    madness::AtomicInt counter;
    counter = 0;
    int n = 0;
    for (std::size_t i = 0; i < src.size(); ++i) {
      if (!src.is_zero(i)) {
        src.world().taskq.add(
            &detail::counted_tensor_to_eigen_subtensor<Tile, result_type>,
            src.find(i), &result, &counter);
        ++n;
      }
    }

    // Wait until the write tasks are complete
    src.world().await([&counter, n]() { return counter == n; });

    return result;
  } else  // else
    return result_type{};
}
*/
}  // namespace TiledArray

#endif  // TILEDARRAY_CONVERSIONS_EIGEN_H__INCLUDED
