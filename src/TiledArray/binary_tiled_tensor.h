#ifndef TILEDARRAY_BINARY_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_BINARY_TILED_TENSOR_H__INCLUDED

#include <TiledArray/array_base.h>
//#include <TiledArray/tiled_range.h>
#include <TiledArray/binary_tensor.h>
#include <TiledArray/unary_tensor.h>
#include <TiledArray/distributed_storage.h>

namespace TiledArray {

  template <typename> class StaticTiledRange;
  class DynamicTiledRange;

  namespace expressions {

    // Forward declaration
    template <typename, typename, typename>
    class BinaryTiledTensor;

    namespace detail {

      /// Select the tiled range type

      /// This helper class selects a tiled range for binary operations. It favors
      /// \c StaticTiledRange over \c DynamicTiledRange to avoid the dynamic memory
      /// allocations used in \c DynamicTiledRange.
      /// \tparam LRange The left tiled range type
      /// \tparam RRange The right tiled range type
      template <typename LRange, typename RRange>
      struct trange_select {
        typedef LRange type; ///< The tiled range type to use

        /// Select the tiled range object

        /// \tparam L The left tiled tensor object type
        /// \tparam R The right tiled tensor object type
        /// \param l The left tiled tensor object
        /// \param r The right tiled tensor object
        /// \return A const reference to the either the \c l or \c r tiled range
        /// object
        template <typename L, typename R>
        static inline const type& trange(const L& l, const R&) {
          return l.trange();
        }
      };

      template <typename CS>
      struct trange_select<DynamicTiledRange, StaticTiledRange<CS> > {
        typedef StaticTiledRange<CS> type;

        template <typename L, typename R>
        static inline const type& trange(const L&, const R& r) {
          return r.trange();
        }
      };

    } // namespace detail

    template <typename Left, typename Right, typename Op>
    struct TensorTraits<BinaryTiledTensor<Left, Right, Op> > {
      typedef typename detail::range_select<typename Left::range_type,
          typename Right::range_type>::type range_type;
      typedef typename detail::trange_select<typename Left::trange_type,
          typename Right::trange_type>::type trange_type;
      typedef typename Eval<BinaryTensor<typename Left::value_type,
          typename Right::value_type, Op> >::type value_type;
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;
      typedef typename storage_type::const_iterator const_iterator; ///< Tensor const iterator
      typedef typename storage_type::future const_reference;
    }; // struct TensorTraits<BinaryTiledTensor<Arg, Op> >

    /// Tensor that is composed from two argument tensors

    /// The tensor tiles are constructed with \c BinaryTensor. A binary operator
    /// is used to transform the individual elements of the tiles.
    /// \tparam Left The left argument type
    /// \tparam Right The right argument type
    /// \tparam Op The binary transform operator type.
    template <typename Left, typename Right, typename Op>
    class BinaryTiledTensor : public ReadableTiledTensor<BinaryTiledTensor<Left, Right, Op> > {
    public:
      typedef BinaryTiledTensor<Left, Right, Op> BinaryTiledTensor_;
      typedef Left left_tensor_type;
      typedef Right right_tensor_type;
      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<BinaryTiledTensor_>, BinaryTiledTensor_);
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type; /// The storage type for this object

    private:
      // Not allowed
      BinaryTiledTensor_& operator=(const BinaryTiledTensor_&);

      // These eval functions are used as task functions to evaluate the tiles
      // of this tiled tensor. The three different versions are needed for cases
      // where one of the tiles may be zero.

      static value_type eval_tensor(const typename left_tensor_type::value_type& left,
          const typename right_tensor_type::value_type& right, const Op& op) {
        return BinaryTensor<typename left_tensor_type::value_type,
            typename right_tensor_type::value_type, Op>(left, right, op);
      }

      static value_type eval_tensor_left(const typename left_tensor_type::value_type& left, const Op& op) {
        return UnaryTensor<typename left_tensor_type::value_type, std::binder2nd<Op> >(left,
            std::binder2nd<Op>(op, 0));
      }

      static value_type eval_tensor_right(const typename right_tensor_type::value_type& right, const Op& op) {
        return UnaryTensor<typename right_tensor_type::value_type, std::binder1st<Op> >(right,
            std::binder1st<Op>(op, 0));
      }

    public:

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      BinaryTiledTensor(const left_tensor_type& left, const right_tensor_type& right, const Op& op) :
        left_(left), right_(right),
        shape_((left.is_dense() || right.is_dense() ? 0 : left_.get_shape() | right_.get_shape())),
        data_(new storage_type(left.get_world(), left.size(), left.get_pmap(), false),
            madness::make_deferred_deleter<storage_type>(left.get_world()))
      {
        // Iterate over local left tiles and generate binary tile tasks
        for(typename left_tensor_type::const_iterator it = left.begin(); it != left.end(); ++it) {
          if(right.is_zero(it.index())) {
            // Add a task where the right tile is zero and left tile is non-zero
            madness::Future<value_type> value = get_world().taskq.add(& eval_tensor_left,
                *it, op);
            data_->set(it.index(), value);
          } else {
            // Add a task where both the left and right tiles are non-zero
            madness::Future<value_type> value = get_world().taskq.add(& eval_tensor,
                *it, right[it.index()], op);
            data_->set(it.index(), value);
          }
        }

        // Iterate over local right tiles and generate binary tile tasks
        for(typename right_tensor_type::const_iterator it = right.begin(); it != right.end(); ++it) {
          if(left.is_zero(it.index())) {
            // Add tasks where the left tile is zero and the right is non-zero
            madness::Future<value_type> value = get_world().taskq.add(& eval_tensor_right,
                *it, op);
            data_->set(it.index(), value);
          }
          // Note: The previous loop will handle non-zero left tiles
        }
        data_->process_pending();
      }

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      BinaryTiledTensor(const BinaryTiledTensor_& other) :
          left_(other.left_), right_(other.right_),
          shape_(other.shape_),
          data_(other.data_)
      { }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(range() == dest.range());

        // Add result tiles to dest
        for(const_iterator it = begin(); it != end(); ++it)
          dest.set(it.index(), *it);
      }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const range_type& range() const {
        return detail::range_select<typename left_tensor_type::range_type,
            typename right_tensor_type::range_type>::range(left_, right_);
      }

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
        TA_ASSERT(range().includes(i));
        if(is_dense())
          return false;
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
      const TiledArray::detail::Bitset<>& get_shape() const {
        TA_ASSERT(! is_dense());
        return shape_;
      }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const {
        return detail::trange_select<typename left_tensor_type::trange_type,
          typename right_tensor_type::trange_type>::trange(left_, right_);
      }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference operator[](size_type i) const {
        TA_ASSERT(! is_zero(i));
        if(is_local(i)) {
          typename storage_type::const_accessor acc;
          data_->insert(acc, i);
          return acc->second;
        }

        return data_->find(i, true);
      }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return data_->begin(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return data_->end(); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return left_.vars(); }

      madness::World& get_world() const { return data_->get_world(); }

    private:
      const left_tensor_type& left_; ///< Left argument
      const right_tensor_type& right_; ///< Right argument
      TiledArray::detail::Bitset<> shape_;
      std::shared_ptr<storage_type> data_;
    }; // class BinaryTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_BINARY_TILED_TENSOR_H__INCLUDED
