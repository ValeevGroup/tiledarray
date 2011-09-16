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
      typedef typename Arg::size_type size_type;
      typedef typename Arg::size_array size_array;
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
      TILEDARRAY_READABLE_TILED_TENSOR_INHEIRATE_TYPEDEF(ReadableTiledTensor<UnaryTiledTensor_>, UnaryTiledTensor_);

    private:
      // Not allowed
      UnaryTiledTensor_& operator=(const UnaryTiledTensor_&);

      typedef detail::MakeUnaryTensor<typename Arg::value_type, Op> op_type; ///< The transform operation type

    public:


      /// Construct a binary tiled tensor op

      /// \param left The left argument
      /// \param right The right argument
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
        TA_ASSERT(dim() == dest.dim());
        TA_ASSERT(std::equal(size().begin(), size().end(), dest.size().begin()));

        // Add result tiles to dest and wait for all tiles to be added.
        madness::Future<bool> done =
            get_world().taskq.for_each(madness::Range<const_iterator>(begin(),
            end(), 8), detail::EvalTo<Dest, const_iterator>(dest));
        done.get();
      }


      /// Tensor dimension accessor

      /// \return The number of dimension in the tensor
      unsigned int dim() const { return arg_.dim(); }


      /// Tensor data and tile ordering accessor

      /// \return The tensor data and tile ordering
      TiledArray::detail::DimensionOrderType order() const { return arg_.order(); }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const size_array& size() const { return arg_.range().size(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type volume() const { return arg_.range().volume(); }

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

      /// World object accessor

      /// \return A reference to the world where tensor lives
      madness::World& get_world() const { return arg_.get_world(); }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      std::shared_ptr<pmap_interface> get_pmap() const { return arg_.get_pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return arg_.is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const { return arg_.get_shape(); }

      // Tile dimension info

      /// Tile tensor size array accessor

      /// \param i The tile index
      /// \return The size array of tile \c i
      size_array size(size_type i) const { return arg_.size(i); }

      /// Tile tensor volume accessor

      /// \param i The tile index
      /// \return The number of elements in tile \c i
      size_type volume(size_type i) const { return arg_.volume(i); }

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
