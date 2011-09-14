#ifndef TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED

//#include <TiledArray/annotated_array.h>
#include <TiledArray/array_base.h>
#include <TiledArray/unary_tensor.h>
#include <TiledArray/distributed_storage.h>

namespace TiledArray {
  namespace expressions {

    template <typename, typename>
    class UnaryTiledTensor;

    template <typename Arg, typename Op>
    struct TensorTraits<UnaryTiledTensor<Arg, Op> > {
      typedef typename Arg::size_type size_type;
      typedef typename Arg::size_array size_array;
      typedef typename Arg::trange_type trange_type;
      typedef UnaryTensor<typename Arg::value_type, Op> value_type;
      typedef TiledArray::detail::UnaryTransformIterator<typename Arg::const_iterator,
          Op> const_iterator; ///< Tensor const iterator
      typedef value_type const_reference;
    }; // struct TensorTraits<UnaryTiledTensor<Arg, Op> >

    template <typename Arg, typename Op>
    struct Eval<UnaryTiledTensor<Arg, Op> > {
      typedef UnaryTiledTensor<Arg, Op> type;
    }; // struct Eval<UnaryTiledTensor<Arg, Op> >


    /// Tensor that is composed from an argument tensor

    /// The tensor elements are constructed using a unary transformation
    /// operation.
    /// \tparam Arg The argument type
    /// \tparam Op The Unary transform operator type.
    template <typename Arg, typename Op>
    class UnaryTiledTensor : public ReadableTiledTensor<UnaryTiledTensor<Arg, Op> > {
    public:
      typedef UnaryTiledTensor<Arg, Op> UnaryTiledTensor_;
      typedef Arg arg_tensor_type;
      TILEDARRAY_READABLE_TILED_TENSOR_INHEIRATE_TYPEDEF(ReadableTiledTensor<UnaryTiledTensor_>, UnaryTiledTensor_);
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type; /// The storage type for this object
      typedef Op op_type; ///< The transform operation type

    private:
      // Not allowed
      UnaryTiledTensor(const UnaryTiledTensor_&);
      UnaryTiledTensor_& operator=(const UnaryTiledTensor_&);

    public:

      /// Construct a binary tiled tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      UnaryTiledTensor(const arg_tensor_type& arg, const op_type& op) :
        arg_(arg), op_(op)
      { }

      /// Evaluate this tensor and store the results in \c dest

      /// \tparam Dest The destination object type
      /// \param dest The destination object
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(volume() == dest.volume());
      }

      /// Tensor dimension accessor

      /// \return The number of dimensions
      unsigned int dim() const {
        return arg_.dim();
      }

      /// Data ordering

      /// \return The data ordering type
      TiledArray::detail::DimensionOrderType order() const {
        return arg_.order();
      }

      /// Tensor dimension size accessor

      /// \return An array that contains the sizes of each tensor dimension
      const size_array& size() const {
        return arg_.size();
      }

      /// Tensor volume

      /// \return The total number of elements in the tensor
      size_type volume() const { return arg_.volume(); }

      const trange_type& trange() const { return arg_.trange(); }

      /// Iterator factory

      /// \return An iterator to the first data element
      const_iterator begin() const {
        return TiledArray::detail::make_tran_it(arg_.begin(), op_);
      }

      /// Iterator factory

      /// \return An iterator to the last data element }
      const_iterator end() const {
        return TiledArray::detail::make_tran_it(arg_.end(), op_);
      }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const {
        return op_((*arg_)[i]);
      }

      std::shared_ptr<pmap_interface> get_pmap() const { return arg_.get_pmap(); }

    private:
      arg_tensor_type arg_; ///< Argument
      op_type op_; ///< Element transform operation
    }; // class UnaryTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED
