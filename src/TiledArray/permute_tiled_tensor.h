#ifndef TILEDARRAY_PERMUTE_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_PERMUTE_TILED_TENSOR_H__INCLUDED

#include <TiledArray/future_tensor.h>
#include <TiledArray/array_base.h>
#include <TiledArray/permute_tensor.h>
#include <TiledArray/distributed_storage.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename, unsigned int>
    class PermuteTiledTensor;

    namespace detail {

      /// Tile generator functor
      template <typename Arg, unsigned int DIM>
      class MakePermuteTensor {
      public:
        MakePermuteTensor(const Permutation<DIM>& perm) : perm_(perm) { }

        typedef const Arg& argument_type;
        typedef PermuteTensor<Arg, DIM> result_type;

        result_type operator()(argument_type arg_tile) const {
          return result_type(arg_tile, perm_);
        }

      private:
        const Permutation<DIM>& perm_;
      }; // struct MakeFutTensor

    }  // namespace detail


    template <typename Arg, unsigned int DIM>
    struct TensorTraits<PermuteTiledTensor<Arg, DIM> > {
      typedef typename Arg::range_type range_type;
      typedef typename Arg::trange_type trange_type;
      typedef PermuteTensor<typename Arg::value_type, DIM> value_type;
      typedef FutureTensor<EvalTensor<typename value_type::value_type> > remote_type;
      typedef TiledArray::detail::UnaryTransformIterator<typename Arg::const_iterator,
          detail::MakePermuteTensor<typename Arg::value_type, DIM> > const_iterator; ///< Tensor const iterator
      typedef value_type const_reference;
    }; // struct TensorTraits<PermuteTiledTensor<Arg, Op> >

    template <typename Arg, unsigned int DIM>
    struct Eval<PermuteTiledTensor<Arg, DIM> > {
      typedef PermuteTiledTensor<Arg, DIM> type;
    }; // struct Eval<PermuteTiledTensor<Arg, Op> >


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
      TILEDARRAY_READABLE_TILED_TENSOR_INHEIRATE_TYPEDEF(ReadableTiledTensor<PermuteTiledTensor_>, PermuteTiledTensor_);
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type; /// The storage type for this object
      typedef Permutation<DIM> perm_type;

    private:
      // Not allowed
      PermuteTiledTensor_& operator=(const PermuteTiledTensor_&);

      typedef detail::MakePermuteTensor<typename Arg::value_type, DIM> op_type; ///< The transform operation type

      struct InitTiles {

        InitTiles(storage_type& data, const op_type& op) : data_(data), op_(op) { }

        bool operator()(const typename Arg::const_iterator& it) {
          typename storage_type::const_accessor acc;
          data_.insert(acc, it.index(), op_(*it));
          return true;
        }

      private:
        storage_type& data_;
        const op_type& op_;
      };

    public:

      using base::get_world;

      /// Construct a permute tiled tensor op

      /// \param left The left argument
      /// \param right The right argument
      /// \param op The element transform operation
      PermuteTiledTensor(const arg_tensor_type& arg, const perm_type& p) :
          base(arg.get_world()),
          perm_(p),
          arg_(arg),
          range_(p ^ arg.range()),
          trange_(p ^ arg.trange()),
          shape_((arg_.is_dense() ? arg_.size() : 0)),
          data_(arg.get_world(), arg.size(), arg.get_pmap(), false)
      {
        // Initialize the shape
        init_shape();

        // Initialize the tiles
        // Add result tiles to dest and wait for all tiles to be added.
        madness::Future<bool> done =
            get_world().taskq.for_each(madness::Range<const_iterator>(arg.begin(),
            arg.end(), 8), InitTiles(data_, op_type(p, data_)));
        done.get();
      }

      /// Copy constructor

      /// \param other The unary tensor to be copied
      PermuteTiledTensor(const PermuteTiledTensor_& other) :
        base(other.get_world()),
        perm_(other.perm_),
        arg_(other.arg_),
        range_(other.range_),
        trange_(other.trange_),
        shape_(other.shape_),
        data_(other.get_world(), other.volume(), other.get_pmap(), false)
      { }


      /// Evaluate tensor

      /// \return The evaluated tensor
      const PermuteTiledTensor_& eval() const { return *this; }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(range_.dim() == dest.range().dim());
        TA_ASSERT(std::equal(size().begin(), size().end(), dest.size().begin()));

        // Add result tiles to dest and wait for all tiles to be added.
        madness::Future<bool> done =
            get_world().taskq.for_each(madness::Range<const_iterator>(begin(),
            end(), 8), detail::EvalTo<Dest, const_iterator>(dest));
        done.get();
      }


      /// The tile tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const { return range_; }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const { return range_.size(); }

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
      bool is_zero(size_type i) const {
        TA_ASSERT(trange_.includes(i));
        if(is_dense())
          return true;

        return shape_[i];
      }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      std::shared_ptr<pmap_interface> get_pmap() const { return arg_.get_pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return arg_.is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const { return shape_; }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      trange_type trange() const { return trange_; }

      /// Local tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference get_local(size_type i) const {
        TA_ASSERT(is_local(i));
        return data_[i].get();
      }

      /// Local tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      remote_type get_remote(size_type i) const {
        TA_ASSERT(! is_local(i));
        return remote_type(data_[i]);
      }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return data_.begin(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return data_.end(); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return arg_.vars(); }


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
        typename range_type::size_array ip_weight = ip ^ range_.weight();
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
      arg_tensor_type arg_; ///< Argument
      range_type range_; ///< Tensor size info
      trange_type trange_; ///< Tensor tiled range
      TiledArray::detail::Bitset<> shape_;
      mutable storage_type data_; ///< Tile container
      op_type op_;
    }; // class PermuteTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_PERMUTE_TILED_TENSOR_H__INCLUDED
