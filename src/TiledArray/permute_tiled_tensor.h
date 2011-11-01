#ifndef TILEDARRAY_PERMUTE_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_PERMUTE_TILED_TENSOR_H__INCLUDED

#include <TiledArray/array_base.h>
#include <TiledArray/permute_tensor.h>
#include <TiledArray/distributed_storage.h>

namespace TiledArray {
  namespace expressions {

    template <typename, unsigned int>
    class PermuteTiledTensor;

    template <typename Arg, unsigned int DIM>
    struct TensorTraits<PermuteTiledTensor<Arg, DIM> > {
      typedef typename Arg::range_type range_type;
      typedef typename Arg::trange_type trange_type;
      typedef typename Arg::value_type value_type;
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;
      typedef typename storage_type::const_iterator const_iterator; ///< Tensor const iterator
      typedef typename storage_type::future const_reference;
    }; // struct TensorTraits<PermuteTiledTensor<Arg, Op> >

    /// Tensor that is composed from an argument tensor

    /// The tensor elements are constructed using a unary transformation
    /// operation.
    /// \tparam Arg The argument type
    /// \tparam Op The Unary transform operator type.
    template <typename Arg, unsigned int DIM>
    class PermuteTiledTensor : public ReadableTiledTensor<PermuteTiledTensor<Arg, DIM> > {
    public:
      typedef PermuteTiledTensor<Arg, DIM> PermuteTiledTensor_;
      typedef Arg arg_tensor_type;
      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<PermuteTiledTensor_>, PermuteTiledTensor_);
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type; /// The storage type for this object
      typedef Permutation<DIM> perm_type;

    private:
      // Not allowed
      PermuteTiledTensor(const PermuteTiledTensor_&);
      PermuteTiledTensor_& operator=(const PermuteTiledTensor_&);

      static value_type eval_tensor(const perm_type& p, const typename arg_tensor_type::value_type& value) {
        return PermuteTensor<typename arg_tensor_type::value_type, DIM>(value, p);
      }

    public:

      /// Construct a permute tiled tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      PermuteTiledTensor(const arg_tensor_type& arg, const perm_type& p) :
          perm_(p),
          arg_(arg),
          trange_(p ^ arg.trange()),
          shape_((arg_.is_dense() ? 0 : arg_.size())),
          data_(new storage_type(arg.get_world(), arg.size(), arg.get_pmap(), false),
              madness::make_deferred_deleter<storage_type>(arg.get_world()))
      {
        // Initialize the shape
        if(! arg_.is_dense())
          init_shape();

        // Initialize the tiles
        for(typename arg_tensor_type::const_iterator it = arg.begin(); it != arg.end(); ++it) {
          madness::Future<value_type> value = get_world().taskq.add(& eval_tensor, p, *it);
          data_->set(range().ord(p ^ arg.range().idx(it.index())), value);
        }
        data_->process_pending();
      }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(range() == dest.range());

        // Add result tiles to dest and wait for all tiles to be added.
        for(const_iterator it = begin(); it != end(); ++it)
          dest.set(it.index(), *it);
      }


      /// The tile tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const { return trange_.tiles(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const { return trange_.tiles().volume(); }

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
      bool is_zero(size_type i) const {
        TA_ASSERT(trange_.includes(i));
        if(is_dense())
          return false;

        return shape_[i];
      }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      std::shared_ptr<pmap_interface> get_pmap() const { return data_->get_pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return arg_.is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const { return shape_; }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      trange_type trange() const { return trange_; }

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
      const VariableList& vars() const { return arg_.vars(); }

      madness::World& get_world() const { return data_->get_world(); }


    private:

//      template <typename CS>
//      struct init_shape_helper {
//
//        static const TiledArray::detail::Bitset<>::block_type count =
//            8 * sizeof(TiledArray::detail::Bitset<>::block_type);
//
//        init_shape_helper(arg_tensor_type& arg, TiledArray::detail::Bitset<>& shape, const typename CS::size_array& invp_weight) :
//            arg_(arg), shape_(shape), invp_weight_(invp_weight)
//        { }
//
//
//
//        bool operator() (std::size_t b) {
//          if(arg_.get_shape().get()[b]) {
//
//            typename CS::index index(0);
//            const typename CS::index start(0);
//
//            std::size_t first = b * count;
//            const std::size_t last = first + count;
//            for(; first < last; ++first, CS::increment_coordinate(index, start, arg_.size()))
//              if(arg_.get_shape()[first])
//                shape_.set(CS::calc_ordinal(index, invp_weight_));
//          }
//
//          return true;
//        }
//      private:
//
//        arg_tensor_type& arg_; ///< Argument
//        TiledArray::detail::Bitset<>& shape_;
//        typename range_type::size_array ip_weight = ip ^ range_.weight();
//        const typename range_type::index ip_start = ip ^ arg_.range().start();
//      }; // struct permute_shape_helper
//
//      template <typename CS>
//      madness::Future<bool> init_shape(TiledArray::detail::Bitset<>& result) {
//        const perm_type ip = -perm_;
//        typename range_type::size_array ip_weight = ip ^ range_.weight();
//        const typename range_type::index ip_start = ip ^ arg_.range().start();
//        madness::Future<bool> done = get_world().taskq.for_each(
//            madness::Range<std::size_t>(0, shape_.num_blocks(), 8),
//            init_shape_helper<CS>(arg_, shape_, invp_weight));
//        return done;
//      }

      void init_shape() {
        // Construct the inverse permuted weight and size for this tensor
        const perm_type ip = -perm_;
        typename range_type::size_array ip_weight = ip ^ range().weight();
        const typename range_type::index ip_start = ip ^ arg_.range().start();

        // Coordinated iterator for the argument object range
        typename arg_tensor_type::range_type::const_iterator arg_range_it =
            arg_.range().begin();

        // permute the data
        for(std::size_t i = 0; i < arg_.size(); ++i, ++arg_range_it)
          if(arg_.get_shape()[i])
            shape_.set(TiledArray::detail::calc_ordinal(*arg_range_it, ip_weight, ip_start));
      }

      perm_type perm_; ///< Transform operation
      const arg_tensor_type& arg_; ///< Argument
      trange_type trange_; ///< Tensor tiled range
      TiledArray::detail::Bitset<> shape_;
      std::shared_ptr<storage_type> data_; ///< Tile container
    }; // class PermuteTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_PERMUTE_TILED_TENSOR_H__INCLUDED
