#ifndef TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED

//#include <TiledArray/annotated_array.h>
#include <TiledArray/array_base.h>
#include <TiledArray/unary_tensor.h>
#include <TiledArray/distributed_storage.h>
#include <TiledArray/array.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename, typename>
    class UnaryTiledTensor;

    template <typename Exp, typename Op>
    UnaryTiledTensor<Exp, Op> make_unary_tiled_tensor(const ReadableTiledTensor<Exp>& arg, const Op& op) {
      return UnaryTiledTensor<Exp, Op>(arg.derived(), op);
    }

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

    namespace detail {

      /// Tensor that is composed from an argument tensor

      /// The tensor elements are constructed using a unary transformation
      /// operation.
      /// \tparam Arg The argument type
      /// \tparam Op The Unary transform operator type.
      template <typename Arg, typename Op>
      class UnaryTiledTensorImpl {
      public:
        typedef UnaryTiledTensor<Arg, Op> UnaryTiledTensor_;
        typedef UnaryTiledTensorImpl<Arg, Op> UnaryTiledTensorImpl_;
        typedef Arg arg_tensor_type;
        typedef ReadableTiledTensor<UnaryTiledTensor_> base;
        typedef typename base::size_type size_type;
        typedef typename base::range_type range_type;
        typedef typename base::eval_type eval_type;
        typedef typename base::pmap_interface pmap_interface;
        typedef typename base::trange_type trange_type;
        typedef typename base::value_type value_type;
        typedef typename base::const_reference const_reference;
        typedef typename base::const_iterator const_iterator;
        typedef TiledArray::detail::DistributedStorage<value_type> storage_type;

      private:
        // Not allowed
        UnaryTiledTensorImpl(const UnaryTiledTensorImpl_& other);
        UnaryTiledTensorImpl_& operator=(const UnaryTiledTensorImpl_&);

        class Eval {
        public:
          typedef typename pmap_interface::const_iterator iterator;
          typedef typename arg_tensor_type::value_type arg_type;

          typedef const iterator& argument_type;
          typedef bool result_type;

          Eval(const std::shared_ptr<UnaryTiledTensorImpl_>& pimpl) : pimpl_(pimpl) { }

          Eval(const Eval& other) : pimpl_(other.pimpl_) { }

          Eval& operator=(const Eval& other) const {
            pimpl_ = other.pimpl_;
            return *this;
          }

          result_type operator()(argument_type it) const  {
            if(! pimpl_->is_zero(*it)) {
              madness::Future<value_type> value = pimpl_->get_world().taskq.add(& eval,
                  pimpl_->arg_.move(*it), pimpl_->op_);
              pimpl_->data_.set(*it, value);
            }

            return true;
          }

          template <typename Archive>
          void serialize(const Archive& ar) { TA_ASSERT(false); }

        private:

          static value_type eval(const arg_type& arg, const Op& op) {
            return make_unary_tensor(arg, op);
          }

          std::shared_ptr<UnaryTiledTensorImpl_> pimpl_;
        }; // class Eval

        /// Task function for generating tile evaluation tasks.

        /// The two parameters are given by futures that ensure the child
        /// arguments have completed before spawning tile tasks.
        /// \note: This task cannot return until all other \c for_each() tasks
        /// have completed. get() blocks this task until for_each() is done
        /// while still processing tasks.
        bool generate_tasks(const std::shared_ptr<UnaryTiledTensorImpl_>& me, bool) const {

          // This task cannot return until all other for_each tasks have completed.
          // Tasks are still being processed.
          return get_world().taskq.for_each(
              madness::Range<typename pmap_interface::const_iterator>(
                  arg_.get_pmap()->begin(), arg_.get_pmap()->end(), 8), Eval(me)).get();
        }

      public:

        /// Construct a unary tiled tensor op

        /// \param arg The argument
        /// \param op The element transform operation
        UnaryTiledTensorImpl(const arg_tensor_type& arg, const Op& op) :
            arg_(arg),
            data_(arg.get_world(), arg.size()),
            op_(op)
        { }


        void set_pmap(const std::shared_ptr<pmap_interface>& pmap) {
          data_.init(pmap);
        }

        /// Evaluate tensor to destination

        /// \tparam Dest The destination tensor type
        /// \param dest The destination to evaluate this tensor to
        template <typename Dest>
        void eval_to(Dest& dest) const {
          TA_ASSERT(range() == dest.range());

          typename pmap_interface::const_iterator end = data_.get_pmap()->end();
          for(typename pmap_interface::const_iterator it = data_.get_pmap()->begin(); it != end; ++it)
            if(! is_zero(*it))
              dest.set(*it, move(*it));
        }

        /// Evaluate the argument

        /// \return A future to a bool that will be set once the argument has
        /// been evaluated.
        madness::Future<bool> eval_arg(const VariableList& v, const std::shared_ptr<pmap_interface>& pmap) {
          return arg_.eval(v, pmap);
        }

        static madness::Future<bool> generate_tiles(const std::shared_ptr<UnaryTiledTensorImpl_>& me,
            madness::Future<bool> arg_done)
        {
          return me->get_world().taskq.add(*me, & UnaryTiledTensorImpl_::generate_tasks,
              me, arg_done, madness::TaskAttributes::hipri());
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
        ProcessID owner(size_type i) const { return data_.owner(i); }

        /// Query for a locally owned tile

        /// \param i The tile index to query
        /// \return \c true if the tile is owned by this node, otherwise \c false
        bool is_local(size_type i) const { return data_.is_local(i); }

        /// Query for a zero tile

        /// \param i The tile index to query
        /// \return \c true if the tile is zero, otherwise \c false
        bool is_zero(size_type i) const { return arg_.is_zero(i); }

        /// Tensor process map accessor

        /// \return A shared pointer to the process map of this tensor
        const std::shared_ptr<pmap_interface>& get_pmap() const { return data_.get_pmap(); }

        /// Query the density of the tensor

        /// \return \c true if the tensor is dense, otherwise false
        bool is_dense() const { return arg_.is_dense(); }

        /// Tensor shape accessor

        /// \return A reference to the tensor shape map
        TiledArray::detail::Bitset<> get_shape() const { return arg_.get_shape(); }

        /// Tiled range accessor

        /// \return The tiled range of the tensor
        const trange_type& trange() const { return arg_.trange(); }

        /// Tile accessor

        /// \param i The tile index
        /// \return Tile \c i
        const_reference operator[](size_type i) const {
          TA_ASSERT(! is_zero(i));
          return data_[i];
        }

        /// Tile move

        /// Tile is removed after it is set.
        /// \param i The tile index
        /// \return Tile \c i
        const_reference move(size_type i) const {
          TA_ASSERT(! is_zero(i));
          return data_.move(i);
        }

        /// Array begin iterator

        /// \return A const iterator to the first element of the array.
        const_iterator begin() const { return data_.begin(); }

        /// Array end iterator

        /// \return A const iterator to one past the last element of the array.
        const_iterator end() const { return data_.end(); }

        /// Variable annotation for the array.
        const VariableList& vars() const { return arg_.vars(); }

        madness::World& get_world() const { return data_.get_world(); }

        /// Clear the tile data

        /// Remove all tiles from the tensor.
        /// \note: Any tiles will remain in memory until the last reference
        /// is destroyed.
        void clear() { data_.clear(); }

      private:
        arg_tensor_type arg_; ///< Argument
        storage_type data_; ///< Distributed data container
        Op op_; ///< The unary element opertation
      }; // class UnaryTiledTensorImpl

    } // namespace detail


    /// Tensor that is composed from an argument tensor

    /// The tensor elements are constructed using a unary transformation
    /// operation.
    /// \tparam Arg The argument type
    /// \tparam Op The Unary transform operator type.
    template <typename Arg, typename Op>
    class UnaryTiledTensor : public ReadableTiledTensor<UnaryTiledTensor<Arg, Op> >{
    public:
      typedef UnaryTiledTensor<Arg, Op> UnaryTiledTensor_;
      typedef detail::UnaryTiledTensorImpl<Arg, Op> impl_type;
      typedef Arg arg_tensor_type;
      typedef ReadableTiledTensor<UnaryTiledTensor_> base;
      typedef typename base::size_type size_type;
      typedef typename base::range_type range_type;
      typedef typename base::eval_type eval_type;
      typedef typename base::pmap_interface pmap_interface;
      typedef typename base::trange_type trange_type;
      typedef typename base::value_type value_type;
      typedef typename base::const_reference const_reference;
      typedef typename base::const_iterator const_iterator;
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;

      UnaryTiledTensor() : pimpl_() { }

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      UnaryTiledTensor(const arg_tensor_type& arg, const Op& op) :
        pimpl_(new impl_type(arg, op),
            madness::make_deferred_deleter<impl_type>(arg.get_world()))
      { }

      /// Copy constructor

      /// Create a shallow copy of \c other .
      /// \param other The object to be copied.
      UnaryTiledTensor(const UnaryTiledTensor_& other) :
          pimpl_(other.pimpl_)
      { }

      /// Assignement operator

      /// Create a shallow copy of \c other
      /// \param other The object to be copied
      /// \return A reference to this object
      UnaryTiledTensor& operator=(const UnaryTiledTensor_& other) {
        pimpl_ = other.pimpl_;
        return *this;
      }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(pimpl_);
        pimpl_->eval_to(dest);
      }

      /// Evaluate this tiled tensor object with the given result variable list

      /// V is the data layout that the parent tiled tensor operation expects.
      /// The returned future will be evaluated once the tensor has been evaluated.
      /// \param v The expected data layout of this tensor.
      /// \return A Future bool that will be assigned once this tensor has been
      /// evaluated.
      madness::Future<bool> eval(const VariableList& v, const std::shared_ptr<pmap_interface>& pmap) {
        TA_ASSERT(pimpl_);
        pimpl_->set_pmap(pmap);
        return impl_type::generate_tiles(pimpl_, pimpl_->eval_arg(v, pmap->clone()));
      }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const range_type& range() const {
        TA_ASSERT(pimpl_);
        return pimpl_->range();
      }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const {
        TA_ASSERT(pimpl_);
        return pimpl_->size();
      }

      /// Query a tile owner

      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      ProcessID owner(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->owner(i);
      }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      bool is_local(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->is_local(i);
      }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      bool is_zero(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->is_zero(i);
      }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      const std::shared_ptr<pmap_interface>& get_pmap() const {
        TA_ASSERT(pimpl_);
        return pimpl_->get_pmap();
      }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const {
        TA_ASSERT(pimpl_);
        return pimpl_->is_dense();
      }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<> get_shape() const {
        TA_ASSERT(pimpl_);
        return pimpl_->get_shape();
      }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const {
        TA_ASSERT(pimpl_);
        return pimpl_->trange();
      }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference operator[](size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->operator[](i);
      }

      /// Tile move

      /// Tile is removed after it is set.
      /// \param i The tile index
      /// \return Tile \c i
      const_reference move(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->move(i);
      }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const {
        TA_ASSERT(pimpl_);
        return pimpl_->begin();
      }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const {
        TA_ASSERT(pimpl_);
        return pimpl_->end();
      }

      /// Variable annotation for the array.
      const VariableList& vars() const {
        TA_ASSERT(pimpl_);
        return pimpl_->vars();
      }

      madness::World& get_world() const {
        TA_ASSERT(pimpl_);
        return pimpl_->get_world();
      }

      /// Release tensor data

      /// Clear all tensor data from memory. This is equivalent to
      /// \c UnaryTiledTensor().swap(*this) .
      void release() {
        if(pimpl_) {
          pimpl_->clear();
          pimpl_.reset();
        }
      }

      template <typename Archive>
      void serialize(const Archive&) { TA_ASSERT(false); }

    private:
      std::shared_ptr<impl_type> pimpl_;
    }; // class UnaryTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED
