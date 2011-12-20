#ifndef TILEDARRAY_BINARY_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_BINARY_TILED_TENSOR_H__INCLUDED

#include <TiledArray/array_base.h>
//#include <TiledArray/tiled_range.h>
#include <TiledArray/binary_tensor.h>
#include <TiledArray/unary_tensor.h>
#include <TiledArray/permute_tensor.h>
#include <TiledArray/distributed_storage.h>
#include <TiledArray/bitset.h>
#include <TiledArray/array.h>

namespace TiledArray {

  // Forward declarations

  template <typename> class StaticTiledRange;
  class DynamicTiledRange;

  namespace expressions {

    // Forward declaration
    template <typename, typename, typename>
    class BinaryTiledTensor;

    namespace {

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

    } // namespace


    template <typename Left, typename Right, typename Op>
    struct TensorTraits<BinaryTiledTensor<Left, Right, Op> > {
      typedef typename detail::range_select<typename Left::range_type,
          typename Right::range_type>::type range_type;
      typedef typename trange_select<typename Left::trange_type,
          typename Right::trange_type>::type trange_type;
      typedef typename Eval<BinaryTensor<typename Left::value_type,
          typename Right::value_type, Op> >::type value_type;
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;
      typedef typename storage_type::const_iterator const_iterator; ///< Tensor const iterator
      typedef typename storage_type::future const_reference;
    }; // struct TensorTraits<BinaryTiledTensor<Arg, Op> >

    namespace {

      /// Tensor that is composed from two argument tensors

      /// The tensor tiles are constructed with \c BinaryTensor. A binary operator
      /// is used to transform the individual elements of the tiles.
      /// \tparam Left The left argument type
      /// \tparam Right The right argument type
      /// \tparam Op The binary transform operator type.
      template <typename Left, typename Right, typename Op>
      class BinaryTiledTensorImpl {
      public:
        typedef BinaryTiledTensorImpl<Left, Right, Op> BinaryTiledTensorImpl_;
        typedef BinaryTiledTensor<Left, Right, Op> BinaryTiledTensor_;
        typedef Left left_tensor_type;
        typedef Right right_tensor_type;
        TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<BinaryTiledTensor_>, BinaryTiledTensor_);
        typedef TiledArray::detail::DistributedStorage<value_type> storage_type; /// The storage type for this object

      private:
        // Not allowed
        BinaryTiledTensorImpl_& operator=(const BinaryTiledTensorImpl_&);
        BinaryTiledTensorImpl(const BinaryTiledTensorImpl_&);

        /// Tile and task generator

        /// This object is passed to the parallel for_each function in MADNESS.
        /// It generates tasks that evaluates the tile for this tensor.
        /// \tparam The operations type, which is an instantiation of PermOps.
        class EvalLeft {
        public:
          typedef typename left_tensor_type::const_iterator iterator;
          typedef typename left_tensor_type::value_type left_arg_type;
          typedef typename right_tensor_type::value_type right_arg_type;

          EvalLeft(const std::shared_ptr<BinaryTiledTensorImpl_>& pimpl) :
              pimpl_(pimpl)
          { }

          EvalLeft(const EvalLeft& other) :
              pimpl_(other.pimpl_)
          { }

          EvalLeft& operator=(const EvalLeft& other) const {
            pimpl_ = other.pimpl_;
            return *this;
          }

          bool operator()(const iterator& it) const  {
            if(pimpl_->right_.is_zero(it.index())) {
              // Add a task where the right tile is zero and left tile is non-zero
              madness::Future<value_type> value =
                  pimpl_->get_world().taskq.add(& EvalLeft::eval_left, *it, pimpl_->op_);
              pimpl_->data_.set(it.index(), value);
            } else {
              // Add a task where both the left and right tiles are non-zero
              madness::Future<value_type> value =
                  pimpl_->get_world().taskq.add(& EvalLeft::eval, *it,
                      pimpl_->right_[it.index()], pimpl_->op_);
              pimpl_->data_.set(it.index(), value);
            }

            return true;
          }

          template <typename Archive>
          void serialize(const Archive& ar) { TA_ASSERT(false); }

        private:

          static value_type eval(const left_arg_type& left, const right_arg_type& right, const Op& op) {
            return make_binary_tensor(left, right, op);
          }

          static value_type eval_left(const typename left_tensor_type::value_type& left, const Op& op) {
            return make_unary_tensor(left, std::bind2nd(op,
                typename left_tensor_type::value_type::value_type(0)));
          }

          std::shared_ptr<BinaryTiledTensorImpl_> pimpl_; ///< pimpl to the owning expression object
        }; // class EvalLeft

        class EvalRight {
        public:
          typedef typename right_tensor_type::const_iterator iterator;
          typedef typename left_tensor_type::value_type left_arg_type;
          typedef typename right_tensor_type::value_type right_arg_type;

          typedef const iterator& argument_type;
          typedef bool result_type;

          EvalRight(const std::shared_ptr<BinaryTiledTensorImpl_>& pimpl) : pimpl_(pimpl) { }

          EvalRight(const EvalRight& other) : pimpl_(other.pimpl_) { }

          EvalRight& operator=(const EvalRight& other) const {
            pimpl_ = other.pimpl_;
            return *this;
          }

          result_type operator()(argument_type it) const  {
            if(pimpl_->left_.is_zero(it.index())) {
              // Add a task where the right tile is zero and left tile is non-zero
              madness::Future<value_type> value =
                  pimpl_->get_world().taskq.add(& EvalRight::eval_right, *it,
                  pimpl_->op_);
              pimpl_->data_.set(it.index(), value);
            }

            return true;
          }

          template <typename Archive>
          void serialize(const Archive& ar) { TA_ASSERT(false); }

        private:

          static value_type eval_right(const typename right_tensor_type::value_type& right, const Op& op) {
            return make_unary_tensor(right, std::bind1st(op, typename left_tensor_type::value_type::value_type(0)));
          }

          std::shared_ptr<BinaryTiledTensorImpl_> pimpl_;
        }; // class EvalRight

        /// Task function for generating tile evaluation tasks.

        /// The two parameters are given by futures that ensure the child
        /// arguments have completed before spawning tile tasks.
        /// \note: This task cannot return until all other \c for_each() tasks
        /// have completed. get() blocks this task until for_each() is done
        /// while still processing tasks.
        static bool generate_left_tasks(std::shared_ptr<BinaryTiledTensorImpl_> me, bool, bool) {
          madness::Future<bool> done = me->get_world().taskq.for_each(
              madness::Range<typename left_tensor_type::const_iterator>(
                  me->left_.begin(), me->left_.end(), 8), EvalLeft(me));

          // Run other tasks while waiting for for_each() to complete.
          done.get();

          return true;
        }

        /// Task function for generating tile evaluation tasks.

        /// The two parameters are given by futures that ensure the child
        /// arguments have completed before spawning tile tasks.
        /// \note: This task cannot return until all other \c for_each() tasks
        /// have completed. get() blocks this task until for_each() is done
        /// while still processing tasks.
        static bool generate_right_tasks(std::shared_ptr<BinaryTiledTensorImpl_> me, bool, bool) {
          madness::Future<bool> done = me->get_world().taskq.for_each(
              madness::Range<typename right_tensor_type::const_iterator>(
                  me->right_.begin(), me->right_.end(), 8), EvalRight(me));

          // This task cannot return until all other for_each tasks have completed.
          // Tasks are still being processed.
          done.get();

          return true;
        }

        bool eval_shape(bool, bool) {
          if(! is_dense())
            TiledArray::detail::Bitset<>(left_.get_shape() | right_.get_shape()).swap(shape_);
          return true;
        }

        static bool eval_done(bool, bool, bool) {
          return true;
        }

      public:

        /// Construct a unary tiled tensor op

        /// \param arg The argument
        /// \param op The element transform operation
        BinaryTiledTensorImpl(const left_tensor_type& left, const right_tensor_type& right, const Op& op) :
          left_(left), right_(right), op_(op), shape_(0),
          data_(left.get_world(), left.size(), left.get_pmap())
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

        madness::Future<bool> eval(const VariableList& v, std::shared_ptr<BinaryTiledTensorImpl_> me) {
          TA_ASSERT(me.get() == this);
          madness::Future<bool> left_child = left_.eval(v);
          madness::Future<bool> right_child = right_.eval(v);

          madness::Future<bool> shape_done = get_world().taskq.add(
              *this, & BinaryTiledTensorImpl_::eval_shape, left_child, right_child,
              madness::TaskAttributes::hipri());

          madness::Future<bool> left_done = get_world().taskq.add(
              & BinaryTiledTensorImpl_::generate_left_tasks, me, left_child,
              right_child, madness::TaskAttributes::hipri());

          madness::Future<bool> right_done = get_world().taskq.add(
              & BinaryTiledTensorImpl_::generate_right_tasks, me, left_child,
              right_child, madness::TaskAttributes::hipri());

          return get_world().taskq.add(& BinaryTiledTensorImpl_::eval_done, shape_done,
              left_done, right_done, madness::TaskAttributes::hipri());
        }

        /// Tensor tile size array accessor

        /// \return The size array of the tensor tiles
        const range_type& range() const {
          return detail::range_select<typename left_tensor_type::range_type,
              typename right_tensor_type::range_type>::range(left_, right_);
        }

        /// Tensor tile volume accessor

        /// \return The number of tiles in the tensor
        size_type size() const {
          return left_.size();
        }

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
        const std::shared_ptr<pmap_interface>& get_pmap() const { return data_.get_pmap(); }

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
          return trange_select<typename left_tensor_type::trange_type,
            typename right_tensor_type::trange_type>::trange(left_, right_);
        }

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
        const VariableList& vars() const { return left_.vars(); }

        madness::World& get_world() const { return data_.get_world(); }

      private:
        left_tensor_type left_; ///< Left argument
        right_tensor_type right_; ///< Right argument
        Op op_;
        TiledArray::detail::Bitset<> shape_; ///< cache of shape
        storage_type data_; ///< Store temporary data
      }; // class BinaryTiledTensorImpl

    } // namespace


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

    private:
      typedef BinaryTiledTensorImpl<Left, Right, Op> impl_type;

    public:

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      BinaryTiledTensor(const left_tensor_type& left, const right_tensor_type& right, const Op& op) :
        pimpl_(new impl_type(left, right, op),
            madness::make_deferred_deleter<impl_type>(left.get_world()))
      { }

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      BinaryTiledTensor(const BinaryTiledTensor_& other) :
          pimpl_(other.pimpl_)
      { }

      /// Assignment operator

      /// Assignment makes a shallow copy of \c other.
      /// \param other The binary tensor to be copied.
      /// \return A reference to this object
      BinaryTiledTensor_& operator=(const BinaryTiledTensor_& other) {
        pimpl_ = other.pimpl_;
        return *this;
      }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const { pimpl_->eval_to(dest); }

      madness::Future<bool> eval(const VariableList& v) { return pimpl_->eval(v, pimpl_); }

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
      bool is_zero(size_type i) const {  return pimpl_->is_zero(i); }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      const std::shared_ptr<pmap_interface>& get_pmap() const { return pimpl_->get_pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return pimpl_->is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const { return pimpl_->get_shape(); }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const {
        return pimpl_->trange();
      }

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
    }; // class BinaryTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

namespace madness {
  namespace archive {

    template <typename Archive, typename T>
    struct ArchiveStoreImpl;
    template <typename Archive, typename T>
    struct ArchiveLoadImpl;

    template <typename Archive, typename Left, typename Right, typename Op>
    struct ArchiveStoreImpl<Archive, std::shared_ptr<TiledArray::expressions::BinaryTiledTensorImpl<Left, Right, Op> > > {
      static void store(const Archive&, const std::shared_ptr<TiledArray::expressions::BinaryTiledTensorImpl<Left, Right, Op> >&) {
        TA_ASSERT(false);
      }
    };

    template <typename Archive, typename Left, typename Right, typename Op>
    struct ArchiveLoadImpl<Archive, std::shared_ptr<TiledArray::expressions::BinaryTiledTensorImpl<Left, Right, Op> > > {

      static void load(const Archive&, std::shared_ptr<TiledArray::expressions::BinaryTiledTensorImpl<Left, Right, Op> >&) {
        TA_ASSERT(false);
      }
    };
  } // namespace archive
} // namespace madness

#endif // TILEDARRAY_BINARY_TILED_TENSOR_H__INCLUDED
