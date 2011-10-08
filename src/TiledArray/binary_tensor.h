#ifndef TILEDARRAY_BINARY_TENSOR_H__INCLUDED
#define TILEDARRAY_BINARY_TENSOR_H__INCLUDED


#include <TiledArray/permutation.h>
#include <TiledArray/eval_tensor.h>
#include <TiledArray/transform_iterator.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/arg_tensor.h>
#include <TiledArray/range.h>
#include <Eigen/Core>
#include <functional>

namespace TiledArray {
  namespace expressions {

    template <typename, typename, typename>
    class BinaryTensor;

    namespace detail {
      template <typename LRange, typename RRange>
      struct range_select {
        typedef DynamicRange type;

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

      template <typename CS>
      struct range_select<StaticRange<CS>, DynamicRange> {
        typedef StaticRange<CS> type;

        template <typename L, typename R>
        static const type& range(const L& l, const R&) {
          return l.range();
        }
      };

      template <typename CS>
      struct range_select<StaticRange<CS>, StaticRange<CS> > {
        typedef StaticRange<CS> type;

        template <typename L, typename R>
        static const type& range(const L& l, const R&) {
          return l.range();
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
      typedef EvalTensor<typename madness::detail::result_of<Op>::type,
          typename LeftArg::range_type> type;
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
      typedef ArgTensor<LeftArg> left_tensor_type;
      typedef ArgTensor<RightArg> right_tensor_type;
      TILEDARRAY_READABLE_TENSOR_INHERIT_TYPEDEF(ReadableTensor<BinaryTensor_>, BinaryTensor_)
      typedef Op op_type; ///< The transform operation type
      typedef EvalTensor<value_type, range_type> eval_type;

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
      BinaryTensor(const LeftArg& left, const RightArg& right, const op_type& op) :
        left_(left), right_(right), op_(op)
      {
        TA_ASSERT(left.range() == right.range());
      }

      BinaryTensor(const BinaryTensor_& other) :
        left_(other.left_), right_(other.right_), op_(other.op_)
      { }

      /// Evaluate this tensor

      /// \return An evaluated tensor object
      eval_type eval() const {
        return eval_type(*this);
      }

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

      void check_dependancies(madness::TaskInterface* task) const {
        left_.check_dependancies(task);
        right_.check_dependancies(task);
      }

    private:
      left_tensor_type left_; ///< Left argument
      right_tensor_type right_; ///< Right argument
      op_type op_; ///< Transform operation
    }; // class BinaryTensor


  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_BINARY_TENSOR_H__INCLUDED
