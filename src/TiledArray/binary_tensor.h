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
      typedef typename LeftArg::size_type size_type;
      typedef typename LeftArg::range_type range_type;
      typedef typename madness::detail::result_of<Op>::type value_type;
      typedef TiledArray::detail::BinaryTransformIterator<typename LeftArg::const_iterator,
          typename RightArg::const_iterator, Op> const_iterator;
      typedef value_type const_reference;
    }; // struct TensorTraits<EvalTensor<T, A> >

    template <typename LeftArg, typename RightArg, typename Op>
    struct Eval<BinaryTensor<LeftArg, RightArg, Op> > {
      typedef EvalTensor<typename madness::detail::result_of<Op>::type> type;
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
      typedef BinaryTensor<LeftArg, RightArg, Op>  BinaryTensor_;
      typedef LeftArg left_tensor_type;
      typedef RightArg right_tensor_type;
      TILEDARRAY_READABLE_TENSOR_INHEIRATE_TYPEDEF(ReadableTensor<BinaryTensor_>, BinaryTensor_)
      typedef DenseStorage<value_type> storage_type; /// The storage type for this object
      typedef Op op_type; ///< The transform operation type

    private:
      // Not allowed
      BinaryTensor_& operator=(const BinaryTensor_&);

    public:

      /// Construct a binary tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      /// \throw TiledArray::Exception When left and right argument orders,
      /// dimensions, or sizes are not equal.
      BinaryTensor(typename TensorArg<left_tensor_type>::type left, typename TensorArg<right_tensor_type>::type right, const op_type& op) :
        left_(left), right_(right), op_(op)
      {
        TA_ASSERT(left.range() == right.range());
      }

      BinaryTensor(const BinaryTensor_& other) :
        left_(other.left_), right_(other.right_), op_(other.op_)
      { }

      /// Evaluate this tensor

      /// \return An evaluated tensor object
      EvalTensor<value_type> eval() const {
        typename EvalTensor<value_type>::storage_type data(size(), begin());
        return EvalTensor<value_type>(range(), data);
      }

      /// Evaluate this tensor and store the results in \c dest

      /// \tparam Dest The destination object type
      /// \param dest The destination object
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(size() == dest.size());
        std::copy(begin(), end(), dest.begin());
      }

      /// Tensor dimension accessor

      /// \return The number of dimensions
      unsigned int dim() const {
        return left_.dim();
      }

      /// Data ordering

      /// \return The data ordering type
      TiledArray::detail::DimensionOrderType order() const {
        return left_.order();
      }

      /// Tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const {
        return left_.range();
      }

      /// Tensor size accessor

      /// \return The total number of elements in the tensor
      size_type size() const {
        return left_.size();
      }

      /// Iterator factory

      /// \return An iterator to the first data element
      const_iterator begin() const {
        return TiledArray::detail::make_tran_it(left_.begin(), right_.begin(), op_);
      }

      /// Iterator factory

      /// \return An iterator to the last data element }
      const_iterator end() const {
        return TiledArray::detail::make_tran_it(left_.end(), right_.end(), op_);
      }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const {
        return op_(left_[i], right_[i]);
      }

      void check_dependancies(madness::TaskInterface* task) const {
        left_.check_dependancies(task);
        right_.check_dependancies(task);
      }

    private:
      typename TensorMem<left_tensor_type>::type left_; ///< Left argument
      typename TensorMem<right_tensor_type>::type right_; ///< Right argument
      op_type op_; ///< Transform operation
    }; // class BinaryTensor


  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_BINARY_TENSOR_H__INCLUDED
