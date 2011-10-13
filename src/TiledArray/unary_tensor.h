#ifndef TILEDARRAY_UNARY_TENSOR_H__INCLUDED
#define TILEDARRAY_UNARY_TENSOR_H__INCLUDED

#include <TiledArray/tensor.h>
#include <TiledArray/transform_iterator.h>
#include <TiledArray/type_traits.h>
//#include <TiledArray/arg_tensor.h>
#include <Eigen/Core>
#include <functional>

namespace TiledArray {
  namespace expressions {

    template <typename, typename> class UnaryTensor;

    template <typename Arg, typename Op>
    struct TensorTraits<UnaryTensor<Arg, Op> > {
      typedef typename Arg::range_type range_type;
      typedef typename madness::detail::result_of<Op>::type value_type;
      typedef value_type const_reference;
    }; // struct TensorTraits<UnaryTensor<Arg, Op> >

    template <typename Arg, typename Op>
    struct Eval<UnaryTensor<Arg, Op> > {
      typedef Tensor<typename madness::detail::result_of<Op>::type,
          typename Arg::range_type> type;
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
      TILEDARRAY_READABLE_TENSOR_INHERIT_TYPEDEF(ReadableTensor<UnaryTensor_>, UnaryTensor_);
      typedef DenseStorage<value_type> storage_type; /// The storage type for this object
      typedef Op op_type; ///< The transform operation type
      typedef Tensor<value_type, range_type> eval_type;

    private:
      // Not allowed
      UnaryTensor_& operator=(const UnaryTensor_&);

    public:

      /// Construct a unary tensor op

      /// The argument may be of type \c Arg or \c FutureTensor<Arg::eval_type>
      /// \tparam T Tensor argument type
      /// \param arg The argument
      /// \param op The element transform operation
      template <typename T>
      UnaryTensor(const T& arg, const op_type& op) :
        arg_(arg), op_(op)
      { }

      UnaryTensor(const UnaryTensor_& other) :
        arg_(other.arg_), op_(other.op_)
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
      const range_type& range() const { return arg_.range(); }

      /// Tensor size

      /// \return The number of elements in the tensor
      size_type size() const { return arg_.size(); }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const {
        return op_(arg_[i]);
      }

    private:
      const arg_tensor_type& arg_; ///< Argument
      op_type op_; ///< Transform operation
    }; // class UnaryTensor

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_UNARY_TENSOR_H__INCLUDED
