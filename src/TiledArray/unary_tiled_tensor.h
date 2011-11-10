#ifndef TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED

//#include <TiledArray/annotated_array.h>
#include <TiledArray/array_base.h>
#include <TiledArray/unary_tensor.h>
#include <TiledArray/distributed_storage.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename, typename>
    class UnaryTiledTensor;

    template <typename Arg, typename Op>
    struct TensorTraits<UnaryTiledTensor<Arg, Op> > {
      typedef typename Arg::range_type range_type;
      typedef typename Arg::trange_type trange_type;
      typedef typename Arg::value_type value_type;
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;
      typedef typename storage_type::const_iterator const_iterator; ///< Tensor const iterator
      typedef typename storage_type::future const_reference;
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
    class UnaryTiledTensor : public ReadableTiledTensor<UnaryTiledTensor<Arg, Op> >{
    public:
      typedef UnaryTiledTensor<Arg, Op> UnaryTiledTensor_;
      typedef Arg arg_tensor_type;
      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<UnaryTiledTensor_>, UnaryTiledTensor_);
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;

    private:
      // Not allowed
      UnaryTiledTensor_& operator=(const UnaryTiledTensor_&);

      static value_type eval_tensor(Op op, const typename arg_tensor_type::value_type& arg) {
        return value_type(UnaryTensor<typename arg_tensor_type::value_type, Op>(arg, op));
      }

    public:

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      UnaryTiledTensor(const arg_tensor_type& arg, const Op& op) :
        arg_(arg),
        data_(new storage_type(arg.get_world(), arg.size(), arg.get_pmap(), false),
            madness::make_deferred_deleter<storage_type>(arg.get_world()))
      {
        for(typename arg_tensor_type::const_iterator it = arg.begin(); it != arg.end(); ++it) {
          madness::Future<value_type> value = get_world().taskq.add(& eval_tensor, op, *it);
          data_->set(it.index(), value);
        }
        data_->process_pending();
      }

      /// Copy constructor
      UnaryTiledTensor(const UnaryTiledTensor_& other) :
        arg_(other.arg_),
        data_(other.data_)
      { }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(range() == dest.range());

        for(const_iterator it = begin(); it != end(); ++it)
          dest.set(it.index(), *it);
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
      ProcessID owner(size_type i) const { return data_->owner(i); }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      bool is_local(size_type i) const { return data_->is_local(i); }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      bool is_zero(size_type i) const { return arg_.is_zero(i); }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      const std::shared_ptr<pmap_interface>& get_pmap() const { return data_->get_pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return arg_.is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const { return arg_.get_shape(); }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const { return arg_.trange(); }

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

        return data_->find(i);
      }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return data_->begin(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return data_->end(); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return arg_.vars(); }

      madness::World& get_world() const { return data_->get_world(); }


    private:
      const arg_tensor_type& arg_; ///< Argument
      std::shared_ptr<storage_type> data_;
    }; // class UnaryTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED
