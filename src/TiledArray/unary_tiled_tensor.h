#ifndef TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED

//#include <TiledArray/annotated_array.h>
#include <TiledArray/array_base.h>
#include <TiledArray/unary_tensor.h>
#include <TiledArray/eval_task.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename, typename>
    class UnaryTiledTensor;

    namespace detail {

      /// Tile generator functor
      template <typename Arg, typename Op>
      class MakeUnaryTensor {
      public:
        MakeUnaryTensor(const Op& op) : op_(op) { }

        typedef const Arg& argument_type;
        typedef UnaryTensor<Arg, Op> result_type;

        result_type operator()(argument_type arg_tile) const {
          return result_type(arg_tile, op_);
        }

      private:
        Op op_;
      }; // struct MakeFutTensor

    }  // namespace detail


    template <typename Arg, typename Op>
    struct TensorTraits<UnaryTiledTensor<Arg, Op> > {
      typedef typename Arg::range_type range_type;
      typedef typename Arg::trange_type trange_type;
      typedef UnaryTensor<typename Arg::value_type, Op> value_type;
      typedef TiledArray::detail::UnaryTransformIterator<typename Arg::const_iterator,
          detail::MakeUnaryTensor<typename Arg::value_type, Op> > const_iterator; ///< Tensor const iterator
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
      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<UnaryTiledTensor_>, UnaryTiledTensor_);

    private:
      // Not allowed
      UnaryTiledTensor_& operator=(const UnaryTiledTensor_&);

      typedef detail::MakeUnaryTensor<typename Arg::value_type, Op> op_type; ///< The transform operation type

    public:

      using base::get_world;
      using base::get_remote;

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      UnaryTiledTensor(const arg_tensor_type& arg, const Op& op) :
        arg_(arg), op_(op)
      { }

      /// Copy constructor

      /// \param other The unary tensor to be copied
      UnaryTiledTensor(const UnaryTiledTensor_& other) :
        arg_(other.arg_), op_(other.op_)
      { }


      /// Evaluate tensor

      /// \return The evaluated tensor
      const UnaryTiledTensor_& eval() const { return *this; }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(range() == dest.range());

        // Add result tiles to dest and wait for all tiles to be added.
        madness::Future<bool> done =
            get_world().taskq.for_each(madness::Range<const_iterator>(begin(),
            end(), 8), detail::EvalTo<Dest, const_iterator>(dest));
        done.get();
      }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const range_type& range() const { return arg_.range(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const { return arg_.size(); }

      /// Query a tile owner

      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      ProcessID owner(size_type i) const { return arg_.owner(i); }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      bool is_local(size_type i) const { return arg_.is_local(i); }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      bool is_zero(size_type i) const { return arg_.is_zero(i); }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      std::shared_ptr<pmap_interface> get_pmap() const { return arg_.get_pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return arg_.is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const { return arg_.get_shape(); }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      trange_type trange() const { return arg_.trange(); }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference operator[](size_type i) const { return op_(arg_[i]); }


      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return const_iterator(arg_.begin(), op_); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return const_iterator(arg_.end(), op_); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return arg_.vars(); }


    private:
      arg_tensor_type arg_; ///< Argument
      op_type op_; ///< Element transform operation
    }; // class UnaryTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED
