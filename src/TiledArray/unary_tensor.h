#ifndef TILEDARRAY_UNARY_TENSOR_H__INCLUDED
#define TILEDARRAY_UNARY_TENSOR_H__INCLUDED

#include <TiledArray/eval_tensor.h>
#include <TiledArray/transform_iterator.h>
#include <TiledArray/type_traits.h>
#include <Eigen/Core>
#include <functional>

namespace TiledArray {
  namespace expressions {

    template <typename, typename>
    class UnaryTensor;

    template <typename Arg, typename Op>
    struct TensorTraits<UnaryTensor<Arg, Op> > {
      typedef typename TensorSize::size_type size_type;
      typedef typename TensorSize::size_array size_array;
      typedef typename madness::result_of<Op>::type value_type;
      typedef TiledArray::detail::UnaryTransformIterator<typename Arg::const_iterator,
          Op> const_iterator; ///< Tensor const iterator
      typedef value_type const_reference;
    }; // struct TensorTraits<UnaryTensor<Arg, Op> >

    template <typename Arg, typename Op>
    struct Eval<UnaryTensor<Arg, Op> > {
      typedef EvalTensor<typename madness::result_of<Op>::type> type;
    }; // struct Eval<UnaryTensor<Arg, Op> >


    /// Tensor that is composed from an argument tensor

    /// The tensor elements are constructed using a unary transformation
    /// operation.
    /// \tparam Arg The argument type
    /// \tparam Op The Unary transform operator type.
    template <typename Arg, typename Op>
    class UnaryTensor : public ReadableTensor<UnaryTensor<Arg, Op> > {
    public:
      typedef UnaryTensor<Arg, Op> UnaryTensor_;
      typedef Arg arg_tensor_type;
      TILEDARRAY_READABLE_TENSOR_INHEIRATE_TYPEDEF(ReadableTensor<UnaryTensor_>, UnaryTensor_);
      typedef DenseStorage<value_type> storage_type; /// The storage type for this object
      typedef Op op_type; ///< The transform operation type

      /// Construct a binary tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      UnaryTensor(const arg_tensor_type& arg, const op_type& op) :
        arg_(&arg), size_(arg.size(), arg.order()), op_(op)
      {}

      UnaryTensor(const UnaryTensor_& other) :
        arg_(other.arg_), size_(other.size_), op_(other.op_)
      { }

      UnaryTensor_& operator=(const UnaryTensor_& other) {
        arg_ = other.arg_;
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
      const_iterator begin() const { return TiledArray::detail::make_tran_it(arg_->begin(), op_); }

      /// Iterator factory

      /// \return An iterator to the last data element }
      const_iterator end() const { return TiledArray::detail::make_tran_it(arg_->end(), op_); }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const {
        return op_((*arg_)[i]);
      }

    private:
      const arg_tensor_type* arg_; ///< Argument
      TensorSize size_; ///< Tensor size info
      op_type op_; ///< Transform operation
    }; // class UnaryTensor

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_UNARY_TENSOR_H__INCLUDED
