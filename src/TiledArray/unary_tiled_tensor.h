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

    namespace {

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
        TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<UnaryTiledTensor_>, UnaryTiledTensor_);
        typedef TiledArray::detail::DistributedStorage<value_type> storage_type;

      private:
        // Not allowed
        UnaryTiledTensorImpl(const UnaryTiledTensorImpl_& other);
        UnaryTiledTensorImpl_& operator=(const UnaryTiledTensorImpl_&);

        class Eval {
        public:
          typedef typename arg_tensor_type::const_iterator iterator;
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
            madness::Future<value_type> value = pimpl_->get_world().taskq.add(& eval,
                *it, pimpl_->op_);
            pimpl_->data_.set(it.index(), value);

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
          madness::Future<bool> done = get_world().taskq.for_each(
              madness::Range<typename arg_tensor_type::const_iterator>(
                  arg_.begin(), arg_.end(), 8), Eval(me));

          // This task cannot return until all other for_each tasks have completed.
          // Tasks are still being processed.
          done.get();

          return true;
        }

      public:

        /// Construct a unary tiled tensor op

        /// \param arg The argument
        /// \param op The element transform operation
        UnaryTiledTensorImpl(const arg_tensor_type& arg, const Op& op) :
            arg_(arg),
            data_(arg.get_world(), arg.size(), arg.get_pmap(), true),
            op_(op)
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

        madness::Future<bool> eval_arg(const VariableList& v) {
          return arg_.eval(v);
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

        /// Array begin iterator

        /// \return A const iterator to the first element of the array.
        const_iterator begin() const { return data_.begin(); }

        /// Array end iterator

        /// \return A const iterator to one past the last element of the array.
        const_iterator end() const { return data_.end(); }

        /// Variable annotation for the array.
        const VariableList& vars() const { return arg_.vars(); }

        madness::World& get_world() const { return data_.get_world(); }


      private:
        arg_tensor_type arg_; ///< Argument
        storage_type data_; ///< Distributed data container
        Op op_; ///< The unary element opertation
      }; // class UnaryTiledTensorImpl

    } // namespace


    /// Tensor that is composed from an argument tensor

    /// The tensor elements are constructed using a unary transformation
    /// operation.
    /// \tparam Arg The argument type
    /// \tparam Op The Unary transform operator type.
    template <typename Arg, typename Op>
    class UnaryTiledTensor : public ReadableTiledTensor<UnaryTiledTensor<Arg, Op> >{
    public:
      typedef UnaryTiledTensor<Arg, Op> UnaryTiledTensor_;
      typedef UnaryTiledTensorImpl<Arg, Op> impl_type;
      typedef Arg arg_tensor_type;
      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<UnaryTiledTensor_>, UnaryTiledTensor_);
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;

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
      void eval_to(Dest& dest) const { pimpl_->eval_to(dest); }

      /// Evaluate this tiled tensor object with the given result variable list

      /// V is the data layout that the parent tiled tensor operation expects.
      /// The returned future will be evaluated once the tensor has been evaluated.
      /// \param v The expected data layout of this tensor.
      /// \return A Future bool that will be assigned once this tensor has been
      /// evaluated.
      madness::Future<bool> eval(const VariableList& v) {
        return impl_type::generate_tiles(pimpl_, pimpl_->eval_arg(v));
      }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const range_type& range() const { return pimpl_->range(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const { return pimpl_->size(); }

      /// Query a tile owner

      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      ProcessID owner(size_type i) const { return pimpl_->owner(i); }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      bool is_local(size_type i) const { return pimpl_->is_local(i); }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      bool is_zero(size_type i) const { return pimpl_->is_zero(i); }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      const std::shared_ptr<pmap_interface>& get_pmap() const { return pimpl_->get_pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return pimpl_->is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<> get_shape() const { return pimpl_->get_shape(); }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const { return pimpl_->trange(); }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference operator[](size_type i) const { return pimpl_->operator[](i); }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return pimpl_->begin(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return pimpl_->end(); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return pimpl_->vars(); }

      madness::World& get_world() const { return pimpl_->get_world(); }

      template <typename T, typename CS>
      operator Array<T, CS>()  {
        madness::Future<bool> eval_done = eval(vars());
        eval_done.get();
        if(is_dense()) {
          Array<T, CS> result(get_world(), trange(), get_pmap());
          eval_to(result);
          return result;
        } else {
          Array<T, CS> result(get_world(), trange(), get_shape(), get_pmap());
          eval_to(result);
          return result;
        }
      }

    private:
      std::shared_ptr<impl_type> pimpl_;
    }; // class UnaryTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_UNARY_TILED_TENSOR_H__INCLUDED
