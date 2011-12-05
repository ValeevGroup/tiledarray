#ifndef TILEDARRAY_BINARY_TENSOR_H__INCLUDED
#define TILEDARRAY_BINARY_TENSOR_H__INCLUDED

#include <TiledArray/tensor.h> // for Tensor, StaticRange, and DynamicRange

namespace TiledArray {
  namespace expressions {

    template <typename, typename, typename>
    class BinaryTensor;

    namespace detail {

      /// Select the range type

      /// This helper class selects a range for binary operations. It favors
      /// \c StaticRange over \c DynamicRange to avoid the dynamic memory
      /// allocations used in \c DynamicRange.
      /// \tparam LRange The left tiled range type
      /// \tparam RRange The right tiled range type
      template <typename LRange, typename RRange>
      struct range_select {
        typedef LRange type; ///< The range type to use

        /// Select the range object

        /// \tparam L The left tensor object type
        /// \tparam R The right tensor object type
        /// \param l The left tensor object
        /// \param r The right tensor object
        /// \return A const reference to the either the \c l or \c r range
        /// object
        template <typename L, typename R>
        static const type& range(const L& l, const R&) {
          return l.range();
        }
      };

      template <typename CS>
      struct range_select<DynamicRange, StaticRange<CS> > {
        typedef StaticRange<CS> type;

        template <typename L, typename R>
        static const type& range(const L&, const R& r) {
          return r.range();
        }
      };

      template <typename L, typename R, typename O>
      BinaryTensor<L, R, O> make_binary_tensor(const L& l, const R& r, const O& o) {
        return BinaryTensor<L, R, O>(l, r, o);
      }

    } // namespace detail

    template <typename LeftArg, typename RightArg, typename Op>
    struct TensorTraits<BinaryTensor<LeftArg, RightArg, Op> > {
      typedef typename detail::range_select<typename LeftArg::range_type,
          typename RightArg::range_type>::type range_type;
      typedef typename madness::detail::result_of<Op>::type value_type;
      typedef value_type const_reference;
    }; // struct TensorTraits<EvalTensor<T, A> >

    template <typename LeftArg, typename RightArg, typename Op>
    struct Eval<BinaryTensor<LeftArg, RightArg, Op> > {
      typedef Tensor<typename madness::detail::result_of<Op>::type,
          typename detail::range_select<typename LeftArg::range_type,
          typename RightArg::range_type>::type> type;
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
      TILEDARRAY_READABLE_TENSOR_INHERIT_TYPEDEF(ReadableTensor<BinaryTensor_>, BinaryTensor_)
      typedef Op op_type; ///< The transform operation type

    private:
      // Not allowed
      BinaryTensor_& operator=(const BinaryTensor_&);

    public:

      /// Construct a binary tensor op

      /// The argument may be of type \c Arg or \c FutureTensor<Arg::eval_type>
      /// \tparam L Left tensor argument type
      /// \tparam R Right tensor argument type
      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      /// \throw TiledArray::Exception When left and right argument orders,
      /// dimensions, or sizes are not equal.
      BinaryTensor(const left_tensor_type& left, const right_tensor_type& right, const op_type& op) :
        left_(left), right_(right), op_(op)
      {
        TA_ASSERT(left.range() == right.range());
      }

      BinaryTensor(const BinaryTensor_& other) :
        left_(other.left_), right_(other.right_), op_(other.op_)
      { }

      /// Evaluate this tensor

      /// \return An evaluated tensor object
      eval_type eval() const { return *this; }

      /// Evaluate this tensor and store the results in \c dest

      /// \tparam Dest The destination object type
      /// \param dest The destination object
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(size() == dest.size());
        const size_type s = size();
        for(size_type i = 0; i < s; ++i)
          dest[i] = operator[](i);
      }

      /// Tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const {
        return detail::range_select<typename LeftArg::range_type,
            typename RightArg::range_type>::range(left_, right_);
      }

      /// Tensor size accessor

      /// \return The total number of elements in the tensor
      size_type size() const {
        return left_.size();
      }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const {
        return op_(left_[i], right_[i]);
      }

    private:
      const left_tensor_type& left_; ///< Left argument
      const right_tensor_type& right_; ///< Right argument
      op_type op_; ///< Transform operation
    }; // class BinaryTensor


  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_BINARY_TENSOR_H__INCLUDED
