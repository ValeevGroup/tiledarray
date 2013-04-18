#ifndef TILEDARRAY_EIGEN_H__INCLUDED
#define TILEDARRAY_EIGEN_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/array.h>
#include <Eigen/Core>

namespace TiledArray {
  namespace detail {

    template <typename T, typename A>
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
    map_tensor(const Tensor<T, A>& tensor) {
      TA_ASSERT(tensor.range().dim() <= 2u);
      TA_ASSERT(tensor.range().dim() > 0u);

      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::RowMajor>, Eigen::AutoAlign>(tensor.data(), tensor.range().size()[0],
              (tensor.range().dim() == 2u ? tensor.range().size()[1] : 1ul));
    }

    template <typename T, typename A>
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
    map_tensor(Tensor<T, A>& tensor) {
      TA_ASSERT(tensor.range().dim() <= 2u);
      TA_ASSERT(tensor.range().dim() > 0u);

      return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::RowMajor>, Eigen::AutoAlign>(tensor.data(), tensor.range().size()[0],
              (tensor.range().dim() == 2u ? tensor.range().size()[1] : 1ul));
    }

  } // namespace detail

  template <typename T, typename A, typename Derived>
  void eigen_submatrix_to_tensor(Tensor<T, A>& tensor, const Eigen::MatrixBase<Derived>& matrix) {
    if((matrix.rows() == 1) || (matrix.cols() == 1)) {
      TA_ASSERT(tensor.range().dim() == 1u);

      // Copy vector to tensor
      detail::map_tensor(tensor) = matrix.segment(tensor.range().start()[0],
          tensor.range().size()[0]);

    } else {
      TA_ASSERT(tensor.range().dim() == 2u);
      // Copy matrix
      detail::map_tensor(tensor) = matrix.block(tensor.range().start()[0],
          tensor.range().start()[1], tensor.range().size()[0], tensor.range().size()[1]);
    }
  }

  template <typename T, typename A, typename Derived>
  void tensor_to_eigen_submatrix(Eigen::MatrixBase<Derived>& matrix, const Tensor<T, A>& tensor) {
    if((matrix.rows() == 1) || (matrix.cols() == 1)) {
      TA_ASSERT(tensor.range().dim() == 1u);

      // Copy vector to tensor
      matrix.segment(tensor.range().start()[0], tensor.range().size()[0]) =
          detail::map_tensor(tensor);

    } else {
      TA_ASSERT(tensor.range().dim() == 2u);
      // Copy matrix
      matrix.block(tensor.range().start()[0], tensor.range().start()[1],
          tensor.range().size()[0], tensor.range().size()[1]) = detail::map_tensor(tensor);
    }
  }

  namespace detail {

    template <typename T, typename A, typename Derived>
    void counted_eigen_submatrix_to_tensor(Tensor<T, A>& tensor,
        const Eigen::MatrixBase<Derived>& matrix, madness::AtomicInt* counter)
    {
      eigen_submatrix_to_tensor(tensor, matrix);
      (*counter)++;
    }

    template <typename T, typename A, typename Derived>
    void counted_tensor_to_eigen_submatrix(Eigen::MatrixBase<Derived>& matrix,
        const Tensor<T, A>& tensor, madness::AtomicInt* counter)
    {
      tensor_to_eigen_submatrix(matrix, tensor);
      (*counter)++;
    }

  } // namespace detail

  template <typename A, typename Derived>
  A& eigen_to_array(madness::World& world, const typename A::trange_type& trange,
      const Eigen::MatrixBase<Derived>& matrix)
  {
    // Check that trange matches the dimensions of other
    if((matrix.cols() > 1) && (matrix.rows() > 1)) {
      TA_ASSERT(trange.dim() == 2);
      TA_ASSERT(trange.elements().size()[0] == matrix.rows());
      TA_ASSERT(trange.elements().size()[1] == matrix.cols());
    } else {
      TA_ASSERT(trange.dim() == 1);
      TA_ASSERT(trange.elements().size()[0] == matrix.size());
    }

    // Check that this is not a distributed computing environment
    if(world.size() > 1) {
      std::cerr << "!!! ERROR TiledArray: TiledArray::Array cannot be assigned with an Eigen::Matrix when the number of MPI processes is greater than 1.\n";
      TA_EXCEPTION("Eigen::Matrix => TiledArray::Array when MPI size > 1");
    }

    // Create a new tensor
    A array(world, trange);

    // Spawn tasks to copy Eigen to an Array
    madness::AtomicInt counter;
    counter = 0;
    std::size_t n = 0;
    for(std::size_t i = 0; i < array.size(); ++i) {
      if(! array.is_zero(i)) {
        array.get_world().taskq.add(& detail::counted_eigen_submatrix_to_tensor,
            array.find(i), matrix, &counter);
        ++n;
      }
    }

    return array;
  }

  template <typename T, unsigned int DIM, typename Tile>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
  array_to_eigen(const Array<T, DIM, Tile>& array) {
    // Check that the array will fit in a matrix or vector
    TA_ASSERT((DIM <= 2) && (DIM > 0));

    // Check that this is not a distributed computing environment
    if(array.get_world().size() > 1) {
      std::cerr << "!!! ERROR TiledArray: TiledArray::Array cannot be assigned with an Eigen::Matrix when the number of MPI processes is greater than 1.\n";
      TA_EXCEPTION("Eigen::Matrix => TiledArray::Array when MPI size > 1");
    }
    // Construct the Eigen matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
        matrix(array.trange().elements().size()[0],
            (DIM == 2 ? array.trange().elements()[1] : 1));

    // Spawn tasks to copy Eigen to an Array
    madness::AtomicInt counter;
    counter = 0;
    std::size_t n = 0;
    for(std::size_t i = 0; i < array.size(); ++i) {
      if(! array.is_zero(i)) {
        array.get_world().taskq.add(& detail::counted_tensor_to_eigen_submatrix,
            array.find(i), matrix, &counter);
        ++n;
      }
    }

    return matrix;
  }

}


#endif // TILEDARRAY_EIGEN_H__INCLUDED
