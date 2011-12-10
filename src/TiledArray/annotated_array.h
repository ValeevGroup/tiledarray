#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <TiledArray/array_base.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/distributed_storage.h>
#include <TiledArray/permute_tensor.h>
#include <world/shared_ptr.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename> class AnnotatedArray;

    template <typename Arg>
    struct TensorTraits<AnnotatedArray<Arg> > {
      typedef typename Arg::range_type range_type;
      typedef typename Arg::trange_type trange_type;
      typedef typename Arg::value_type value_type;
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;
      typedef typename storage_type::iterator iterator; ///< Tensor const iterator
      typedef typename storage_type::const_iterator const_iterator; ///< Tensor const iterator
      typedef typename storage_type::future const_reference;
    }; // struct TensorTraits<AnnotatedArray<Arg> >

    namespace {

      /// Tensor that is composed from an argument tensor

      /// The tensor elements are constructed using a unary transformation
      /// operation.
      /// \tparam A The \c Array type
      /// \tparam Op The Unary transform operator type.
      template <typename A>
      class AnnotatedArrayImpl {
      public:
        typedef AnnotatedArrayImpl<A> AnnotatedArrayImpl_;
        typedef AnnotatedArray<A> AnnotatedArray_;
        typedef A array_type;
        TILEDARRAY_WRITABLE_TILED_TENSOR_INHERIT_TYPEDEF(WritableTiledTensor<AnnotatedArray_>, AnnotatedArray_);
        typedef TiledArray::detail::DistributedStorage<value_type> storage_type; /// The storage type for this object

      private:
        // Not allowed
        AnnotatedArrayImpl_& operator=(const AnnotatedArrayImpl_&);
        AnnotatedArrayImpl(const AnnotatedArrayImpl_& other);

      public:

        /// Tile evaluation task generator

        /// This object is used by the MADNESS \c for_each() to generate evaluation
        /// tasks for the tiles. The resulting future is stored in the distributed
        /// container.
        /// \tparam Perm The permutation type.
        template <typename Perm>
        class Eval {
        public:
          typedef typename array_type::const_iterator iterator; // The input tensor iterator type
          typedef iterator argument_type; // The functor argument type
          typedef bool result_type; // The functor result type

          /// Construct
          Eval(const std::shared_ptr<AnnotatedArrayImpl_>& pimpl, const Perm& p) :
              pimpl_(pimpl), perm_(p)
          { }

          /// Generate an evaluation task for \c it

          /// \param it The tile to be evaluated
          /// \return true
          result_type operator()(argument_type it) const {
            make_task(it, perm_, pimpl_);
            return true;
          }

        private:
          /// Evaluate the tensor permutation

          /// This is the task function used to evaluate a permuted tensor.
          /// \param t The tensor to be permuted
          /// \param perm The permutation to be applied to the tensor
          /// \return An evaluated tensor permutation
          static value_type eval_perm(const typename array_type::value_type& t,
              const Permutation& perm) {
            return make_permute_tensor(t, perm);
          }

          /// Generate the task that will evaluate the input tensor

          /// Generate an evaluation task for \c it and store the result.
          /// \param it An iterator that points to the tensor to be permuted
          /// \param perm The permutation that will be applied to the tensor
          /// \param pimpl A pointer to the AnnotatedArrayImpl object
          static void make_task(const iterator& it, const Permutation& perm,
              const std::shared_ptr<AnnotatedArrayImpl_>& pimpl)
          {
            const size_type i =
                pimpl->trange().tiling().ord(perm ^ pimpl->array().idx(it.index()));
            madness::Future<value_type> t =
                pimpl->get_world().taskq.add(& Eval::eval_perm, *it, perm);

            pimpl->data_.set(i, t);
          }

          /// Generate the task that will evaluate the input tensor

          /// No permutation is needed, so the tile is copied as is. The tile is
          /// in a Future so the copy is shallow.
          /// \param it An iterator that points to the tensor to be permuted
          /// \param perm The permutation that will be applied to the tensor
          /// \param pimpl A pointer to the AnnotatedArrayImpl object
          static void make_task(const iterator& it, const TiledArray::detail::NoPermutation&,
              const std::shared_ptr<AnnotatedArrayImpl_>& pimpl)
          {
            pimpl->data_.set(it.index(), *it);
          }

          Perm perm_; ///< Permutation that will be applied to the array tiles
          std::shared_ptr<AnnotatedArrayImpl_> pimpl_;
        }; // class Eval

        bool perm_structure(const Permutation& perm, const VariableList& v) {
          trange_ = perm ^ trange_;

          // construct the shape
          if(! array_.is_dense()) {
            // Construct the inverse permuted weight and size for this tensor
            typename range_type::size_array ip_weight = (-perm) ^ trange_.tiling().weight();
            const typename array_type::range_type::index& start = array_.range().start();

            // Coordinated iterator for the argument object range
            typename array_type::range_type::const_iterator arg_range_it =
                array_.range().begin();

            // permute the data
            const size_type end = array_.size();
            for(std::size_t i = 0; i < end; ++i, ++arg_range_it)
              if(array_.get_shape()[i])
                shape_.set(TiledArray::detail::calc_ordinal(*arg_range_it, ip_weight, start));
          }

          vars_ = v;

          return true;
        }

        /// Construct a permute tiled tensor op

        /// \param left The left argument
        /// \param right The right argument
        /// \param op The element transform operation
        AnnotatedArrayImpl(const array_type& array, const VariableList& vars) :
            array_(array),
            vars_(vars),
            trange_(array.trange()),
            shape_((array.is_dense() ? 0 : array.size())),
            data_(array.get_world(), array.size(), array.get_pmap())
        { }

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
        bool is_zero(size_type i) const {
          TA_ASSERT(trange_.tiles().includes(i));
          if(array_.is_dense())
            return false;

          return get_shape()[i];
        }

        /// Tensor process map accessor

        /// \return A shared pointer to the process map of this tensor
        const std::shared_ptr<pmap_interface>& get_pmap() const { return data_.get_pmap(); }


        /// Tensor shape accessor

        /// \return A reference to the tensor shape map
        const TiledArray::detail::Bitset<>& get_shape() const {
          TA_ASSERT(! array_.is_dense());
          return shape_;
        }

        /// Tiled range accessor

        /// \return The tiled range of the tensor
        const trange_type& trange() const { return trange_; }

        /// Tile accessor

        /// \param i The tile index
        /// \return Tile \c i
        const_reference operator[](size_type i) const {
          TA_ASSERT(is_zero(i));
          return data_[i];
        }

        /// Array begin iterator

        /// \return A const iterator to the first element of the array.
        iterator begin() { return data_.begin(); }

        /// Array end iterator

        /// \return A const iterator to one past the last element of the array.
        iterator end() { return data_.end(); }
        /// Array begin iterator

        /// \return A const iterator to the first element of the array.
        const_iterator begin() const { return data_.begin(); }

        /// Array end iterator

        /// \return A const iterator to one past the last element of the array.
        const_iterator end() const { return data_.end(); }

        /// Variable annotation for the array.
        const VariableList& vars() const { return vars_; }

        madness::World& get_world() const { return data_.get_world(); }

        array_type& array() { return array_; }

        const array_type& array() const { return array_; }

      private:

        array_type array_; ///< The referenced array
        VariableList vars_; ///< Tensor annotation
        trange_type trange_; ///< Tensor tiled range
        TiledArray::detail::Bitset<> shape_; ///< Tensor shape
        mutable storage_type data_; ///< Tile container
      }; // class PermuteTiledTensor

    } // namespace

    /// Wrapper object that adds annotation to tiled tensor objects.

    /// \tparam A The array object type.
    template <typename A>
    class AnnotatedArray : public WritableTiledTensor<AnnotatedArray<A> > {
    public:

      // Note: This class is a bit awkward. It needs to permute the input array
      // for use in tiled-tensor expressions, but assignments operate on the
      // wrapped array. If this object is being used for assignment (i.e.
      // operator=() is called), do not call eval() to avoid the overhead that
      // is needed for argument expressions.

      typedef AnnotatedArray<A> AnnotatedArray_; ///< This object type
      typedef AnnotatedArrayImpl<A> impl_type; ///< This object type
      TILEDARRAY_WRITABLE_TILED_TENSOR_INHERIT_TYPEDEF(WritableTiledTensor<AnnotatedArray_ >, AnnotatedArray_)
      typedef A array_type; ///< The array type
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type; /// The storage type for this object

      /// Constructor

      /// \param a A const reference to an array_type object
      /// \param v A const reference to a variable list object
      /// \throw std::runtime_error When the dimensions of the array and
      /// variable list are not equal.
      AnnotatedArray(const array_type& a, const VariableList& v) :
          pimpl_(new impl_type(a, v),
              madness::make_deferred_deleter<impl_type>(a.get_world()))
      { }


      AnnotatedArray(const AnnotatedArray_& other) : pimpl_(other.pimpl_) { }

    private:

      /// Evaluate other to this array

      /// This function will construct a new from the \c other expression,
      /// evaluate it to the array and
      template <typename D>
      bool eval_to_this(const D& other, bool) {
        if(other.is_dense())
          array_type(other.get_world(), other.trange(), other.get_map()).swap(pimpl_->array());
        else
          array_type(other.get_world(), other.trange(), other.get_shape(), other.get_map()).swap(pimpl_->array());
        other.eval_to(pimpl_->array());

        return true;
      }

      template <typename D>
      AnnotatedArray_& assign(const D& other) {

        madness::Future<bool> child_eval_done = other.eval(pimpl_->vars());

        madness::Future<bool> done =
            get_world().taskq.add(*this, & AnnotatedArray_::eval_to_this<value_type>,
            other, child_eval_done);

        // Wait until evaluation of the result array structure and tiles has been
        // completed. Tasks are processed by get() until this happens.
        done.get();

        return *this;
      }

      static bool eval_done(bool, bool) { return true; }

    public:

      /// Annotated array assignement

      /// Shallow copy the array of other into this array.
      /// \throw TiledArray::Exception If the variable lists do not match.
      AnnotatedArray_& operator =(const AnnotatedArray_& other) {
        return assign(other);
      }

      template <typename D>
      AnnotatedArray_& operator =(const ReadableTiledTensor<D>& other) {
        return assign(other.derived());
      }


      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(trange() == dest.trange());
        TA_ASSERT(vars() == dest.vars());

        // Add result tiles to dest and wait for all tiles to be added.
        for(const_iterator it = begin(); it != end(); ++it)
          dest.set(it.index(), *it);
      }

      /// Evaluate the tensor

      /// Evaluate this tensor such that the data layout matches \c v . This
      /// may result in permutation and storage of the original.
      /// \param v The target variable list.
      /// \return A future that indicates the tensor evaluation is complete
      madness::Future<bool> eval(const VariableList& v) {
        // No call to array_.eval() is needed because it is initialized before use.

        if(v != pimpl_->vars()) {

          Permutation perm = pimpl_->vars().permutation(v);

          // Task to permute the vars, shape, and trange.
          madness::Future<bool> trange_shape_done = get_world().taskq.add(*pimpl_,
              & impl_type::perm_structure, perm, v, madness::TaskAttributes::hipri());

          // Generate the tile permutation tasks.
          madness::Future<bool> tiles_done = get_world().taskq.for_each(
              madness::Range<const_iterator>(pimpl_->array().begin(),
              pimpl_->array().end(), 8), impl_type::template Eval<Permutation>(pimpl_, perm));

          // return the Future that indicates the initialization is done.
          return get_world().taskq.add(& impl_type::eval_done, tiles_done,
              trange_shape_done, madness::TaskAttributes::hipri());

        }

        // The variables match so no permutation is needed. Just copy the tiles.
        return get_world().taskq.for_each(
            madness::Range<const_iterator>(pimpl_->array().begin(),
            pimpl_->array().end(), 8), impl_type::template Eval<TiledArray::detail::NoPermutation>(pimpl_,
            TiledArray::detail::NoPermutation()));
      }

      /// Tensor tile range object accessor

      /// \return A const reference to the tensor range object
      const range_type& range() const { return trange().tiles(); }

      /// Tensor tile size accessor

      /// \return The number of tiles in the tensor
      size_type size() const { return pimpl_->trange().tiles().volume(); }

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
      bool is_dense() const { return pimpl_->array().is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const TiledArray::detail::Bitset<>& get_shape() const { return pimpl_->get_shape(); }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const { return pimpl_->trange(); }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference operator[](size_type i) const { return pimpl_->operator[](i); }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      iterator begin() { return pimpl_->begin(); }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return pimpl_->begin(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return pimpl_->end(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      iterator end() { return pimpl_->end(); }

      /// Variable annotation for the array.
      const VariableList& vars() const { return pimpl_->vars(); }

      madness::World& get_world() const { return pimpl_->get_world(); }


      void set(size_type i, const value_type& v) {
        return pimpl_->array().set(i, v);
      }

      void set(size_type i, const madness::Future<value_type>& v) {
        return pimpl_->array().set(i, v);
      }

    private:
      std::shared_ptr<impl_type> pimpl_; ///< Distributed data container
    }; // class AnnotatedArray


  } // namespace expressions
} //namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
