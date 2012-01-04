#ifndef TILEDARRAY_BINARY_TENSOR_H__INCLUDED
#define TILEDARRAY_BINARY_TENSOR_H__INCLUDED

#include <TiledArray/tensor.h> // for Tensor, StaticRange, and DynamicRange

namespace TiledArray {
  namespace expressions {

    template <typename, typename, typename>
    class BinaryTensor;

    /// Binary tensor factory function

    /// Construct a BinaryTensor object. The constructed object will apply \c op
    /// to each element of \c left and \c right.
    /// \tparam LExp Left expression type
    /// \tparam RExp Right expression type
    /// \tparam Op Binary operation type
    /// \param left The left expression object
    /// \param right The right expression object
    /// \param op The binary element operation
    /// \return A \c BinaryTensor object
    template <typename LExp, typename RExp, typename Op>
    inline BinaryTensor<LExp, RExp, Op> make_binary_tensor(const ReadableTensor<LExp>& left,
        const ReadableTensor<RExp>& right, const Op& op)
    { return BinaryTensor<LExp, RExp, Op>(left.derived(), right.derived(), op); }

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
      typedef ReadableTensor<BinaryTensor_> base;
      typedef typename base::size_type size_type;
      typedef typename base::range_type range_type;
      typedef typename base::eval_type eval_type;
      typedef typename base::value_type value_type;
      typedef typename base::const_reference const_reference;
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
