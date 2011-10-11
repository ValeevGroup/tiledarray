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

    template <typename Left, typename Right, typename Op>
    struct TensorTraits<BinaryTiledTensor<Left, Right, Op> > {
      typedef typename Left::range_type range_type;
      typedef typename Left::trange_type trange_type;
      typedef typename Left::value_type value_type;
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;
      typedef typename storage_type::const_iterator const_iterator; ///< Tensor const iterator
      typedef typename storage_type::future const_reference;
    }; // struct TensorTraits<BinaryTiledTensor<Arg, Op> >

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
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type; /// The storage type for this object

    private:
      // Not allowed
      BinaryTiledTensor(const BinaryTiledTensor_&);
      BinaryTiledTensor_& operator=(const BinaryTiledTensor_&);

      static value_type eval_tensor(const typename left_tensor_type::value_type& left,
          const typename right_tensor_type::value_type& right, const Op& op) {
        return value_type(BinaryTensor<typename left_tensor_type::value_type,
            typename right_tensor_type::value_type, Op>(left, right, op));
      }

    public:

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      BinaryTiledTensor(const left_tensor_type& left, const right_tensor_type& right, const Op& op) :
        left_(left), right_(right),
        shape_((left.is_dense() || right.is_dense() ? 0 : left_.get_shape() | right_.get_shape())),
        data_(new storage_type(left.get_world(), left.size(), left.get_pmap(), false),
            madness::make_deferred_deleter<storage_type>(left.get_word()))
      {

        for(typename left_tensor_type::const_iterator it = left.begin(); it != left.end(); ++it) {
          if(right.is_zero(it.index())) {
            madness::Future<value_type> value = get_world().taskq.add(& eval_tensor,
                *it, left_tensor_type::value_type(), op);
            data_.set(it.index(), value);
          } else {
            madness::Future<value_type> value = get_world().taskq.add(& eval_tensor,
                *it, right[it.index()], op);
            data_.set(it.index(), value);
          }
        }
        for(typename right_tensor_type::const_iterator it = right.begin(); it != right.end(); ++it) {
          if(! left.is_zero(it.index())) {
            madness::Future<value_type> value = get_world().taskq.add(& eval_tensor,
                left_tensor_type::value_type(), *it, op);
            data_.set(it.index(), value);
          }
        }
        data_.process_pending();
      }

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
      const TiledArray::detail::Bitset<>& get_shape() const { return shape_; }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      trange_type trange() const { return left_.trange(); }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference operator[](size_type i) const {
        TA_ASSERT(! is_zero(i));
        if(is_local(i)) {
          typename storage_type::const_accessor acc;
          data_.insert(acc, i);
          return acc->second;
        }

        return data_.find(i, true);
      }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return data_.begin(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return data_.end(); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return left_.vars(); }

      madness::World get_world() const { return data_.get_world(); }

    private:
      const left_tensor_type& left_; ///< Left argument
      const right_tensor_type& right_; ///< Right argument
      TiledArray::detail::Bitset<> shape_;
      std::shared_ptr<storage_type> data_;
    }; // class BinaryTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_BINARY_TILED_TENSOR_H__INCLUDED
