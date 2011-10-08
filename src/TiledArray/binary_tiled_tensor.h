#ifndef TILEDARRAY_BINARY_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_BINARY_TILED_TENSOR_H__INCLUDED

//#include <TiledArray/annotated_array.h>
#include <TiledArray/array_base.h>
#include <TiledArray/binary_tensor.h>
#include <TiledArray/eval_task.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename, typename, typename>
    class BinaryTiledTensor;

    namespace detail {

      /// Tile generator functor
      template <typename Left, typename Right, typename Op>
      class MakeBinaryTensor {
      public:
        MakeBinaryTensor(const Op& op) : op_(op) { }

        typedef const Left& first_argument_type;
        typedef const Right& second_argument_type;
        typedef BinaryTensor<Left, Right, Op> result_type;

        result_type operator()(first_argument_type left, second_argument_type right) const {
          return result_type(left, right, op_);
        }

      private:
        Op op_;
      }; // struct MakeFutTensor

    }  // namespace detail


    template <typename Left, typename Right, typename Op>
    struct TensorTraits<BinaryTiledTensor<Left, Right, Op> > {
      typedef typename Left::range_type range_type;
      typedef typename Left::trange_type trange_type;
      typedef BinaryTensor<typename Left::value_type, typename Right::value_type, Op> value_type;
      typedef TiledArray::detail::BinaryTransformIterator<typename Left::const_iterator,
          typename Right::const_iterator, detail::MakeBinaryTensor<typename Left::value_type,
          typename Right::value_type, Op> > const_iterator; ///< Tensor const iterator
      typedef value_type const_reference;
    }; // struct TensorTraits<BinaryTiledTensor<Arg, Op> >

    template <typename Left, typename Right, typename Op>
    struct Eval<BinaryTiledTensor<Left, Right, Op> > {
      typedef BinaryTiledTensor<Left, Right, Op> type;
    }; // struct Eval<BinaryTiledTensor<Arg, Op> >


    /// Tensor that is composed from an argument tensor

    /// The tensor elements are constructed using a unary transformation
    /// operation.
    /// \tparam Arg The argument type
    /// \tparam Op The Unary transform operator type.
    template <typename Left, typename Right, typename Op>
    class BinaryTiledTensor : public ReadableTiledTensor<BinaryTiledTensor<Left, Right, Op> > {
    public:
      typedef BinaryTiledTensor<Left, Right, Op> BinaryTiledTensor_;
      typedef Left left_tensor_type;
      typedef Right right_tensor_type;
      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<BinaryTiledTensor_>, BinaryTiledTensor_);

    private:
      // Not allowed
      BinaryTiledTensor_& operator=(const BinaryTiledTensor_&);

      typedef detail::MakeBinaryTensor<typename Left::value_type,
          typename Right::value_type, Op> op_type; ///< The transform operation type

    public:

      using base::get_world;
      using base::get_remote;

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      BinaryTiledTensor(const left_tensor_type& left, const right_tensor_type& right, const Op& op) :
        left_(left), right_(right),
        shape_((left.is_dense() || right.is_dense() ? 0 : left_.get_shape() | right_.get_shape())),
        op_(op)
      { }

      /// Copy constructor

      /// \param other The unary tensor to be copied
      BinaryTiledTensor(const BinaryTiledTensor_& other) :
        left_(other.left_), right_(other.right_), shape_(other.shape_), op_(other.op_)
      { }


      /// Evaluate tensor

      /// \return The evaluated tensor
      const BinaryTiledTensor_& eval() const { return *this; }

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
      const range_type& range() const { return left_.range(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const { return left_.size(); }

      /// Query a tile owner

      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      ProcessID owner(size_type i) const { return left_.owner(i); }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      bool is_local(size_type i) const { return left_.is_local(i); }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      bool is_zero(size_type i) const {
        TA_ASSERT(! is_dense());
        TA_ASSERT(range().includes(i));
        return ! (shape_[i]);
      }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      std::shared_ptr<pmap_interface> get_pmap() const { return left_.get_pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return left_.is_dense() || right_.is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const { return shape_; }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      trange_type trange() const { return left_.trange(); }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference get_local(size_type i) const {
        TA_ASSERT(left_.is_local(i));
        TA_ASSERT(right_.is_local(i));
        return op_(left_.get_local(i), right_.get_local(i));
      }


      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return const_iterator(left_.begin(), right_.begin(), op_); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return const_iterator(left_.end(), right_.end(), op_); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return left_.vars(); }


    private:
      left_tensor_type left_; ///< Left argument
      right_tensor_type right_; ///< Right argument
      TiledArray::detail::Bitset<> shape_;
      op_type op_; ///< Element transform operation
    }; // class BinaryTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_BINARY_TILED_TENSOR_H__INCLUDED
