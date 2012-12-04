#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <TiledArray/array_base.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/distributed_storage.h>
#include <TiledArray/permute_tensor.h>
#include <TiledArray/blocked_pmap.h>
#include <TiledArray/tensor_expression_impl.h>
#include <world/shared_ptr.h>

namespace TiledArray {

  template <typename, typename> class Array;

  namespace expressions {

    // Forward declaration
    template <typename> class AnnotatedArray;

    template <typename T, typename CS>
    AnnotatedArray<Array<T, CS> > make_annotatied_array(const Array<T, CS>& array, const VariableList& vars) {
      return AnnotatedArray<Array<T, CS> >(const_cast<Array<T, CS>&>(array), vars);
    }

    template <typename T, typename CS>
    AnnotatedArray<Array<T, CS> > make_annotatied_array(const Array<T, CS>& array, const std::string& vars) {
      return AnnotatedArray<Array<T, CS> >(const_cast<Array<T, CS>&>(array), VariableList(vars));
    }

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

    namespace detail {

      /// Tensor that is composed from an argument tensor

      /// The tensor elements are constructed using a unary transformation
      /// operation.
      /// \tparam A The \c Array type
      /// \tparam Op The Unary transform operator type.
      template <typename A>
      class AnnotatedArrayImpl : public TensorExpressionImpl<typename A::trange_type, typename A::value_type> {
      public:
        typedef TensorExpressionImpl<typename A::trange_type, typename A::value_type> TensorExpressionImpl_;
        typedef typename TensorExpressionImpl_::TensorImplBase_ TensorImplBase_;
        typedef AnnotatedArrayImpl<A> AnnotatedArrayImpl_;
        typedef A array_type;

        typedef typename TensorImplBase_::size_type size_type; ///< size type
        typedef typename TensorImplBase_::pmap_interface pmap_interface; ///< The process map interface type
        typedef typename TensorImplBase_::trange_type trange_type; ///< Tiled range type
        typedef typename TensorImplBase_::range_type range_type; ///< Tile range type
        typedef typename TensorImplBase_::value_type value_type; ///< The result value type
        typedef typename TensorImplBase_::storage_type::const_iterator const_iterator; ///< Tensor const iterator
        typedef typename TensorImplBase_::storage_type::future const_reference; /// The storage type for this object

      public:

        /// Construct a permute tiled tensor op

        /// \param left The left argument
        /// \param right The right argument
        /// \param op The element transform operation
        AnnotatedArrayImpl(const array_type& array, const VariableList& vars) :
            TensorExpressionImpl_(array.get_world(), vars, array.trange(), (array.is_dense() ? 0 : array.size())),
            array_(const_cast<array_type&>(array))
        { }

				array_type& array() { return array_; }

        const array_type& array() const { return array_; }

      private:

        /// Function for evaluating this tensor's tiles

        /// This function is run inside a task, and will run after \c eval_children
        /// has completed. It should spwan additional tasks that evaluate the
        /// individule result tiles.
        virtual void eval_tiles() {
          // Make sure all local tiles are present.
          const typename pmap_interface::const_iterator end = TensorImplBase_::pmap()->end();
          typename pmap_interface::const_iterator it = TensorImplBase_::pmap()->begin();
          if(array_.is_dense()) {
            for(; it != end; ++it)
              TensorExpressionImpl_::set(*it, array_.find(*it));
          } else {
            for(; it != end; ++it)
              if(! array_.is_zero(*it))
                TensorExpressionImpl_::set(*it, array_.find(*it));
          }
        }

        /// Function for evaluating child tensors

        /// This function should return true when the child

        /// This function should evaluate all child tensors.
        /// \param vars The variable list for this tensor (may be different from
        /// the variable list used to initialize this tensor).
        /// \param pmap The process map for this tensor
        virtual madness::Future<bool> eval_children(const expressions::VariableList& vars,
            const std::shared_ptr<pmap_interface>& pmap) {
          return array_.eval();
        }

        /// Construct the shape object

        /// \param shape The existing shape object
        virtual void make_shape(TiledArray::detail::Bitset<>& shape) {
          TA_ASSERT(shape.size() == array_.size());
          shape = array_.get_shape();
        }

        array_type& array_; ///< The referenced array
      }; // class PermuteTiledTensor

    } // detail namespace

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
      typedef detail::AnnotatedArrayImpl<A> impl_type; ///< This object type
      typedef WritableTiledTensor<AnnotatedArray_> base;
      typedef typename base::size_type size_type;
      typedef typename base::range_type range_type;
      typedef typename base::eval_type eval_type;
      typedef typename base::pmap_interface pmap_interface;
      typedef typename base::trange_type trange_type;
      typedef typename base::value_type value_type;
      typedef typename base::const_reference const_reference;
      typedef typename base::const_iterator const_iterator;
      typedef typename base::iterator iterator;
      typedef A array_type; ///< The array type
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type; /// The storage type for this object

    private:

      template <typename T>
      AnnotatedArray_& assign(const T& other) {
        TA_ASSERT(pimpl_);

        // Construct new pmap
        std::shared_ptr<TiledArray::Pmap<size_type> >
            pmap(new TiledArray::detail::BlockedPmap(get_world(), size()));

        // Wait until structure of other has been evaluated; tasks are processed
        // by get() until this happens.
        const_cast<T&>(other).eval(pimpl_->vars(), pmap).get();

        // Construct the result arrray
        if(other.is_dense())
          array_type(other.get_world(), other.trange(), other.get_pmap()->clone()).swap(pimpl_->array());
        else
          array_type(other.get_world(), other.trange(), other.get_shape(), other.get_pmap()->clone()).swap(pimpl_->array());

        // Evaluate this
        other.eval_to(pimpl_->array());

        return *this;
      }

      static bool eval_done(bool, bool) { return true; }

    public:

      AnnotatedArray() : pimpl_() { }

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

      /// Annotated array assignement

      /// Shallow copy the array of other into this array.
      /// \throw TiledArray::Exception If the variable lists do not match.
      AnnotatedArray_& operator =(const AnnotatedArray_& other) {
        return assign(other);
      }

      /// Annotated array assignement

      /// Shallow copy the array of other into this array.
      /// \throw TiledArray::Exception If the variable lists do not match.
      template <typename D>
      AnnotatedArray_& operator =(const ReadableTiledTensor<D>& other) {
        return assign(other.derived());
      }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(pimpl_);
        pimpl_->eval_to(dest);
      }

      /// Evaluate the tensor

      /// Evaluate this tensor such that the data layout matches \c v . This
      /// may result in permutation and storage of the original.
      /// \param v The target variable list.
      /// \return A future that indicates the tensor evaluation is complete
      madness::Future<bool> eval(const VariableList& v, const std::shared_ptr<pmap_interface>& pmap) {
        TA_ASSERT(pimpl_);
        return pimpl_->eval(v, pmap);
      }

      /// Tensor tile range object accessor

      /// \return A const reference to the tensor range object
      const range_type& range() const {
        TA_ASSERT(pimpl_);
        return trange().tiles();
      }

      /// Tensor tile size accessor

      /// \return The number of tiles in the tensor
      size_type size() const {
        TA_ASSERT(pimpl_);
        return pimpl_->trange().tiles().volume();
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
        return pimpl_->pmap();
      }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const {
        TA_ASSERT(pimpl_);
        return pimpl_->is_dense();
      }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      TiledArray::detail::Bitset<> get_shape() const {
        TA_ASSERT(pimpl_);
        return pimpl_->shape();
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
      iterator begin() {
        TA_ASSERT(pimpl_);
        return pimpl_->begin();
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

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      iterator end() {
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


      void set(size_type i, const value_type& v) {
        TA_ASSERT(pimpl_);
        return pimpl_->array().set(i, v);
      }

      void set(size_type i, const madness::Future<value_type>& v) {
        TA_ASSERT(pimpl_);
        return pimpl_->array().set(i, v);
      }

      /// Release tensor data

      /// Clear all tensor data from memory. This is equivalent to
      /// \c AnnotatedArray().swap(*this) .
      void release() {
        if(pimpl_) {
          pimpl_->clear();
          pimpl_.reset();
        }
      }

      template <typename Archive>
      void serialize(const Archive&) { TA_EXCEPTION("Serialization not supported."); }

    private:
      std::shared_ptr<impl_type> pimpl_; ///< Distributed data container
    }; // class AnnotatedArray


  } // namespace expressions
} //namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
