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
 */

#ifndef TILEDARRAY_EIGEN_H__INCLUDED
#define TILEDARRAY_EIGEN_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/array.h>
#include <Eigen/Core>

namespace TiledArray {

  // Convenience typedefs
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixXd;
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixXf;
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixXcd;
  typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixXcf;
  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixXi;
  typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixXl;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::RowMajor> EigenVectorXd;
  typedef Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::RowMajor> EigenVectorXf;
  typedef Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic, Eigen::RowMajor> EigenVectorXcd;
  typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor> EigenVectorXcf;
  typedef Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::RowMajor> EigenVectorXi;
  typedef Eigen::Matrix<long, Eigen::Dynamic, 1, Eigen::RowMajor> EigenVectorXl;


  /// Construct a const Eigen::Map object for a given Tensor object

  /// \tparam T The tensor element type
  /// \tparam A The tensor allocator type
  /// \param tensor The tensor object
  /// \param m The number of rows in the result matrix
  /// \param n The number of columns in the result matrix
  /// \return An m x n Eigen matrix map for \c tensor
  /// \throw TiledArray::Exception When m * n is not equal to \c tensor size
  template <typename T, typename A>
  inline Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
  eigen_map(const Tensor<T, A>& tensor, const std::size_t m, const std::size_t n) {
    TA_ASSERT((m * n) == tensor.size());

    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
        Eigen::RowMajor>, Eigen::AutoAlign>(tensor.data(), m, n);
  }

  /// Construct an Eigen::Map object for a given Tensor object

  /// \tparam T The tensor element type
  /// \tparam A The tensor allocator type
  /// \param tensor The tensor object
  /// \param m The number of rows in the result matrix
  /// \param n The number of columns in the result matrix
  /// \return An m x n Eigen matrix map for \c tensor
  /// \throw TiledArray::Exception When m * n is not equal to \c tensor size
  template <typename T, typename A>
  inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
  eigen_map(Tensor<T, A>& tensor, const std::size_t m, const std::size_t n) {
    TA_ASSERT((m * n) == tensor.size());

    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
        Eigen::RowMajor>, Eigen::AutoAlign>(tensor.data(), m, n);
  }

  /// Construct a const Eigen::Map object for a given Tensor object

  /// The dimensions of the result tensor
  /// \tparam T The tensor element type
  /// \tparam A The tensor allocator type
  /// \param tensor The tensor object
  /// \return An Eigen matrix map for \c tensor
  /// \throw TiledArray::Exception When \c tensor dimensions are not equal to 2 or 1.
  template <typename T, typename A>
  inline Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
  eigen_map(const Tensor<T, A>& tensor) {
    TA_ASSERT((tensor.range().dim() == 2u) || (tensor.range().dim() == 1u));

    return eigen_map(tensor, tensor.range().size()[0],
            (tensor.range().dim() == 2u ? tensor.range().size()[1] : 1ul));
  }

  /// Construct an Eigen::Map object for a given Tensor object

  /// \tparam T The tensor element type
  /// \tparam A The tensor allocator type
  /// \param tensor The tensor object
  /// \return An Eigen matrix map for \c tensor
  /// \throw When \c tensor dimensions are not equal to 2 or 1.
  template <typename T, typename A>
  inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
  eigen_map(Tensor<T, A>& tensor) {
    TA_ASSERT((tensor.range().dim() == 2u) || (tensor.range().dim() == 1u));

    return eigen_map(tensor, tensor.range().size()[0],
            (tensor.range().dim() == 2u ? tensor.range().size()[1] : 1ul));
  }

  /// Copy a block of an Eigen matrix into a tensor

  /// A block of \c matrix will be copied into \c tensor. The block
  /// dimensions will be determined by the dimensions of the tensor's range.
  /// \tparam T The tensor element type
  /// \tparam A The tensor allocator type
  /// \tparam Derived The derived type of an Eigen matrix
  /// \param matrix The matrix that will be assigned the content of \c tensor
  /// \throw TiledArray::Exception When the dimensions of \c tensor are not equal
  /// to 1 or 2.
  template <typename T, typename A, typename Derived>
  inline void eigen_submatrix_to_tensor(const Eigen::MatrixBase<Derived>& matrix, Tensor<T, A>& tensor) {
    if(matrix.rows() == 1) {
      TA_ASSERT(tensor.range().dim() == 1u);
      TA_ASSERT(tensor.range().finish()[0] <= matrix.cols());

      // Copy vector to tensor
      eigen_map(tensor, 1, tensor.range().size()[0]) =
          matrix.block(0, tensor.range().start()[0], 1, tensor.range().size()[0]);
    } else if(matrix.cols() == 1) {
      TA_ASSERT(tensor.range().dim() == 1u);
      TA_ASSERT(tensor.range().finish()[0] <= matrix.rows());

      // Copy vector to tensor
      eigen_map(tensor, tensor.range().size()[0], 1) =
          matrix.block(tensor.range().start()[0], 0, tensor.range().size()[0], 1);
    } else {
      TA_ASSERT(tensor.range().dim() == 2u);
      TA_ASSERT(tensor.range().finish()[0] <= matrix.rows());
      TA_ASSERT(tensor.range().finish()[1] <= matrix.cols());

      // Copy matrix
      eigen_map(tensor, tensor.range().size()[0], tensor.range().size()[1]) =
          matrix.block(tensor.range().start()[0], tensor.range().start()[1],
          tensor.range().size()[0], tensor.range().size()[1]);
    }
  }

  /// Copy the content of a tensor into an Eigen matrix block

  /// The content of tensor will be copied into a block of matrix. The block
  /// dimensions will be determined by the dimensions of the tensor's range.
  /// \tparam T The tensor element type
  /// \tparam A The tensor allocator type
  /// \tparam Derived The derived type of an Eigen matrix
  /// \param matrix The matrix that will be assigned the content of \c tensor
  /// \throw TiledArray::Exception When the dimensions of \c tensor are not equal
  /// to 1 or 2.
  template <typename T, typename A, typename Derived>
  inline void tensor_to_eigen_submatrix(const Tensor<T, A>& tensor, Eigen::MatrixBase<Derived>& matrix) {
    if(matrix.rows() == 1) {
      TA_ASSERT(tensor.range().dim() == 1u);
      TA_ASSERT(tensor.range().finish()[0] <= matrix.cols());

      // Copy vector to tensor
      matrix.block(0, tensor.range().start()[0], 1, tensor.range().size()[0]) =
          eigen_map(tensor, 1, tensor.range().size()[0]);
    } else if(matrix.cols() == 1) {
      TA_ASSERT(tensor.range().dim() == 1u);
      TA_ASSERT(tensor.range().finish()[0] <= matrix.rows());

      // Copy vector to tensor
      matrix.block(tensor.range().start()[0], 0, tensor.range().size()[0], 1) =
          eigen_map(tensor, tensor.range().size()[0], 1);
    } else {
      TA_ASSERT(tensor.range().dim() == 2u);
      TA_ASSERT(tensor.range().finish()[0] <= matrix.rows());
      TA_ASSERT(tensor.range().finish()[1] <= matrix.cols());

      // Copy matrix
      matrix.block(tensor.range().start()[0], tensor.range().start()[1],
          tensor.range().size()[0], tensor.range().size()[1]) =
              eigen_map(tensor, tensor.range().size()[0], tensor.range().size()[1]);
    }
  }

  namespace detail {

    /// Task function for converting Eigen submatrix to a tensor

    /// \tparam T Tensor type
    /// \tparam Derived The matrix type
    /// \param tensor The tensor to be copied
    /// \param matrix The matrix to be assigned
    /// \param counter The task counter
    template <typename A, typename Derived>
    void counted_eigen_submatrix_to_tensor(const Eigen::MatrixBase<Derived>* matrix,
        A& array, const typename A::size_type i, madness::AtomicInt* counter)
    {
      typename A::value_type tensor(array.trange().make_tile_range(i));
      eigen_submatrix_to_tensor(*matrix, tensor);
      array.set(i, tensor);
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
        Eigen::MatrixBase<Derived>* matrix, madness::AtomicInt* counter)
    {
      tensor_to_eigen_submatrix(tensor, *matrix);
      (*counter)++;
    }

    /// Counter probe used to check for the completion of a set of tasks
    class CounterProbe {
    private:
      const madness::AtomicInt& counter_; ///< Counter incremented by the set of tasks
      const long n_; ///< The total number of tasks

    public:
      /// Constructor

      /// \param counter The task completion counter
      /// \param n The total number of tasks
      CounterProbe(const madness::AtomicInt& counter, const long n) :
        counter_(counter), n_(n)
      { }

      /// Probe function

      /// \return \c true when the counter is equal to the number of tasks
      bool operator()() const { return counter_ == n_; }
    }; // class CounterProbe

  } // namespace detail

  /// Convert an Eigen matrix into an Array object

  /// This function will copy the content of \c matrix into an \c Array object
  /// that is tiled according to the \c trange object. The copy operation is
  /// done in parallel, and this function will block until all elements of
  /// \c matrix have been copied into the result array tiles. The size of
  /// \c array.get_world().size() must be equal to 1 or \c replicate must be
  /// equal to \c true . If \c replicate is \c true, it is your responsibility
  /// to ensure that the data in matrix is identical on all nodes.
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
  /// \param replicated \c true indicates that the result array should be a
  /// replicated array [default = false].
  /// \return An \c Array object that is a copy of \c matrix
  /// \throw TiledArray::Exception When world size is greater than 1
  /// \note This function will only work in non-distributed environments. If you
  /// need to convert an Eigen matrix to an \c Array object, you must implement
  /// it yourself. However, you may use \c eigen_submatrix_to_tensor to make
  /// writing such an algorithm easier.
  template <typename A, typename Derived>
  A eigen_to_array(madness::World& world, const typename A::trange_type& trange,
      const Eigen::MatrixBase<Derived>& matrix, bool replicated = false)
  {
    // Check that trange matches the dimensions of other
    if((matrix.cols() > 1) && (matrix.rows() > 1)) {
      TA_USER_ASSERT(trange.tiles().dim() == 2,
          "TiledArray::eigen_to_array(): The number of dimensions in trange is not equal to that of the Eigen matrix.");
      TA_USER_ASSERT(trange.elements().size()[0] == matrix.rows(),
          "TiledArray::eigen_to_array(): The number of rows in trange is not equal to the number of rows in the Eigen matrix.");
      TA_USER_ASSERT(trange.elements().size()[1] == matrix.cols(),
          "TiledArray::eigen_to_array(): The number of columns in trange is not equal to the number of columns in the Eigen matrix.");
    } else {
      TA_USER_ASSERT(trange.tiles().dim() == 1,
          "TiledArray::eigen_to_array(): The number of dimensions in trange must match that of the Eigen matrix.");
      TA_USER_ASSERT(trange.elements().size()[0] == matrix.size(),
          "TiledArray::eigen_to_array(): The size of trange must be equal to the matrix size.");
    }

    // Check that this is not a distributed computing environment
    if(! replicated)
      TA_USER_ASSERT(world.size() == 1,
          "An array cannot be assigned with an Eigen::Matrix when the number of MPI processes is greater than 1.");

    // Create a new tensor
    A array = (replicated && (world.size() > 1) ?
        A(world, trange, std::static_pointer_cast<typename A::pmap_interface>(
            std::shared_ptr<detail::ReplicatedPmap>(new detail::ReplicatedPmap(world, trange.tiles().volume())))) :
        A(world, trange));

    // Spawn tasks to copy Eigen to an Array
    madness::AtomicInt counter;
    counter = 0;
    std::size_t n = 0;
    for(std::size_t i = 0; i < array.size(); ++i) {
      world.taskq.add(& detail::counted_eigen_submatrix_to_tensor<A, Derived>,
          &matrix, array, i, &counter);
      ++n;
    }

    // Wait until the write tasks are complete
    detail::CounterProbe probe(counter, n);
    array.get_world().await(probe);

    return array;
  }

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
  /// \tparam T The element type of the array
  /// \tparam DIM The array dimension
  /// \tparam Tile The array tile type
  /// \param array The array to be converted
  /// \throw TiledArray::Exception When world size is greater than 1 and
  /// \c array is not replicated.
  /// \throw TiledArray::Exception When the number of dimensions of \c array
  /// is not equal to 1 or 2.
  /// \note This function will only work in non-distributed environments.
  template <typename T, unsigned int DIM, typename Tile>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
  array_to_eigen(const Array<T, DIM, Tile>& array) {
    // Check that the array will fit in a matrix or vector
    TA_USER_ASSERT((DIM == 2u) || (DIM == 1u),
        "TiledArray::array_to_eigen(): The array dimensions must be equal to 1 or 2.");

    // Check that this is not a distributed computing environment or that the
    // array is replicated
    if(! array.get_pmap()->is_replicated())
      TA_USER_ASSERT(array.get_world().size() == 1,
          "TiledArray::array_to_eigen(): Array cannot be assigned with an Eigen::Matrix when the number of MPI processes is greater than 1.");

    // Construct the Eigen matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
        matrix(array.trange().elements().size()[0],
            (DIM == 2 ? array.trange().elements().size()[1] : 1));

    // Spawn tasks to copy array tiles to the Eigen matrix
    madness::AtomicInt counter;
    counter = 0;
    std::size_t n = 0;
    for(std::size_t i = 0; i < array.size(); ++i) {
      if(! array.is_zero(i)) {
        array.get_world().taskq.add(
            & detail::counted_tensor_to_eigen_submatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
            typename Array<T, DIM, Tile>::value_type>,
            array.find(i), &matrix, &counter);
        ++n;
      }
    }

    // Wait until the above tasks are complete. Tasks will be processed by this
    // thread while waiting.
    detail::CounterProbe probe(counter, n);
    array.get_world().await(probe);

    return matrix;
  }

} // namespace TiledArray

#endif // TILEDARRAY_EIGEN_H__INCLUDED
