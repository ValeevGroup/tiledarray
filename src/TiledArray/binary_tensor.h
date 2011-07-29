#ifndef TILEDARRAY_BINARY_TENSOR_H__INCLUDED
#define TILEDARRAY_BINARY_TENSOR_H__INCLUDED


#include <TiledArray/permutation.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/eval_tensor.h>
#include <TiledArray/transform_iterator.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/contraction.h>
#include <Eigen/Core>
#include <functional>

namespace TiledArray {
  namespace expressions {

    template <typename, typename, typename>
    class BinaryTensor;

    template <typename LeftArg, typename RightArg, typename Op>
    struct TensorTraits<BinaryTensor<LeftArg, RightArg, Op> > {
      typedef typename TensorSize::size_type size_type;
      typedef typename TensorSize::size_array size_array;
      typedef typename madness::result_of<Op>::type value_type;
      typedef TiledArray::detail::BinaryTransformIterator<typename LeftArg::const_iterator,
          typename RightArg::const_iterator, Op> const_iterator;
      typedef value_type const_reference;
    }; // struct TensorTraits<EvalTensor<T, A> >

    template <typename LeftArg, typename RightArg, typename Op>
    struct Eval<BinaryTensor<LeftArg, RightArg, Op> > {
      typedef EvalTensor<typename madness::result_of<Op>::type> type;
    }; // struct Eval<BinaryTensor<LeftArg, RightArg, Op> >

    /// Tensor that is composed from two argument tensors

    /// The tensor elements are constructed using a binary transformation
    /// operation.
    /// \tparam LeftArg The left-hand argument type
    /// \tparam RightArg The right-hand argument type
    /// \tparam Op The binary transform operator type.
    template <typename LeftArg, typename RightArg, typename Op>
    class BinaryTensor : public ReadableTensor<BinaryTensor<LeftArg, RightArg, Op> > {
    public:
      typedef ReadableTensor<BinaryTensor<LeftArg, RightArg, Op> > base;
      typedef BinaryTensor<LeftArg, RightArg, Op> BinaryTensor_;
      typedef LeftArg left_tensor_type;
      typedef RightArg right_tensor_type;

      typedef typename TensorSize::size_type size_type; ///< Tensor size type
      typedef typename TensorSize::size_array size_array; ///< Tensor size array type
      typedef typename madness::result_of<Op>::type value_type; ///< The tensor value type
      typedef TiledArray::detail::BinaryTransformIterator<typename left_tensor_type::const_iterator,
          typename right_tensor_type::const_iterator, Op> const_iterator; ///< Tensor const iterator
      typedef value_type const_reference; ///< Element constant reference type
      typedef DenseStorage<value_type> storage_type; /// The storage type for this object

      typedef Op op_type; ///< The transform operation type

      /// Construct a binary tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      BinaryTensor(const left_tensor_type& left, const right_tensor_type& right, const op_type& op) :
        left_(&left), right_(&right), size_(left.size(), left.order()), op_(op)
      {
        TA_ASSERT(left.order() == right.order());
        TA_ASSERT(std::equal(left.size().begin(), left.size().end(), right.size().begin()));
      }

      BinaryTensor(const BinaryTensor_& other) :
        left_(other.left_), right_(other.right_), size_(other.size_), op_(other.op_)
      { }

      BinaryTensor_& operator=(const BinaryTensor_& other) {
        left_ = other.left_;
        right_ = other.right_;
        size_ = other.size_;
        op_ = other.op_;

        return *this;
      }


      /// Evaluate this tensor

      /// \return An evaluated tensor object
      EvalTensor<value_type> eval() const {
        typename EvalTensor<value_type>::storage_type data(volume(), begin());
        return EvalTensor<value_type>(size(), order(), data);
      }

      /// Evaluate this tensor and store the results in \c dest

      /// \tparam Dest The destination object type
      /// \param dest The destination object
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(volume() == dest.volume());
        std::copy(begin(), end(), dest.begin());
      }

      /// Tensor dimension accessor

      /// \return The number of dimensions
      unsigned int dim() const { return size_.dim(); }

      /// Data ordering

      /// \return The data ordering type
      TiledArray::detail::DimensionOrderType order() const { return size_.order(); }

      /// Tensor dimension size accessor

      /// \return An array that contains the sizes of each tensor dimension
      const size_array& size() const { return size_.size(); }

      /// Tensor volume

      /// \return The total number of elements in the tensor
      size_type volume() const { return size_.volume(); }

      /// Iterator factory

      /// \return An iterator to the first data element
      const_iterator begin() const { return TiledArray::detail::make_tran_it(left_->begin(), right_->begin(), op_); }

      /// Iterator factory

      /// \return An iterator to the last data element }
      const_iterator end() const { return TiledArray::detail::make_tran_it(left_->end(), right_->end(), op_); }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const {
        return op_(*left_[i], *right_[i]);
      }

    private:
      const left_tensor_type* left_; ///< Left argument
      const right_tensor_type* right_; ///< Right argument
      TensorSize size_; ///< Tensor size info
      op_type op_; ///< Transform operation
    }; // class BinaryTensor


  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_BINARY_TENSOR_H__INCLUDED
