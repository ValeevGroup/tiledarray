#ifndef TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED
#define TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED

//#include <TiledArray/annotated_array.h>
#include <TiledArray/array_base.h>
#include <TiledArray/tensor.h>
#include <TiledArray/contraction_tensor.h>
#include <TiledArray/permute_tensor.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/reduce_task.h>
#include <TiledArray/distributed_storage.h>
#include <TiledArray/binary_tensor.h>
#include <TiledArray/array.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename, typename>
    class ContractionTiledTensor;

    template <typename Left, typename Right>
    struct TensorTraits<ContractionTiledTensor<Left, Right> > {
      typedef DynamicTiledRange trange_type;
      typedef typename trange_type::range_type range_type;
      typedef Tensor<typename math::ContractionValue<typename Left::value_type::value_type,
          typename Right::value_type::value_type>::type, range_type> value_type;
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;
      typedef typename storage_type::const_iterator const_iterator; ///< Tensor const iterator
      typedef typename storage_type::future const_reference;
    }; // struct TensorTraits<ContractionTiledTensor<Arg, Op> >

    namespace {
      /// Tensor that is composed from an argument tensor

      /// The tensor elements are constructed using a unary transformation
      /// operation.
      /// \tparam Arg The argument type
      /// \tparam Op The Unary transform operator type.
      template <typename Left, typename Right>
      class ContractionTiledTensorImpl {
      public:
        typedef ContractionTiledTensorImpl<Left, Right> ContractionTiledTensorImpl_;
        typedef ContractionTiledTensor<Left, Right> ContractionTiledTensor_;
        typedef Left left_tensor_type;
        typedef Right right_tensor_type;
        TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<ContractionTiledTensor_>, ContractionTiledTensor_);
        typedef TiledArray::detail::DistributedStorage<value_type> storage_type;

      private:
        // Not allowed
        ContractionTiledTensorImpl(const ContractionTiledTensorImpl_&);
        ContractionTiledTensor_& operator=(const ContractionTiledTensor_&);


        left_tensor_type left_; ///< Left argument
        right_tensor_type right_; ///< Right argument
        std::shared_ptr<math::Contraction> cont_;
        trange_type trange_;
        TiledArray::detail::Bitset<> shape_;
        VariableList vars_;
        storage_type data_;

        template <typename Perm>
        class Eval {
        private:

          typedef Perm perm_type;

          /// Tile evaluation task generator

          /// This object is used by the MADNESS \c for_each() to generate evaluation
          /// tasks for the tiles. The resulting future is stored in the distributed
          /// container.
          /// \tparam Perm The permutation type.
          class EvalImpl {
          private:
            // Not allowed
            EvalImpl(const EvalImpl&);
            EvalImpl& operator=(const EvalImpl&);

            class reduce_op {
            public:
              typedef const value_type& first_argument_type;
              typedef const value_type& second_argument_type;
              typedef value_type result_type;

              result_type operator()(first_argument_type first, second_argument_type second) const {
                return make_binary_tensor(first, second, std::plus<typename value_type::value_type>());
              }
            }; // class reduce_op


          public:

            /// Construct
            EvalImpl(const perm_type& perm, const std::shared_ptr<ContractionTiledTensorImpl_>& pimpl) :
                perm_(perm),
                m_(accumulate(pimpl->left_.range().size().begin(),
                    pimpl->left_.range().size().begin() + pimpl->cont_->left_outer_dim())),
                i_(accumulate(pimpl->left_.range().size().begin() + pimpl->cont_->left_outer_dim(),
                    pimpl->left_.range().size().end())),
                n_(accumulate(pimpl->right_.range().size().begin(),
                    pimpl->right_.range().size().begin() + pimpl->cont_->right_outer_dim())),
                range_(pimpl->range()),
                pimpl_(pimpl),
                left_cache_(pimpl->left().range().volume()),
                right_cache_(pimpl->right().range().volume())
            { }

            /// Generate an evaluation task for \c it

            /// \param i The tile to be evaluated
            /// \return true
            bool operator()(size_type i) const {
              if(pimpl_->is_local(i)) {
                if(! pimpl_->is_zero(i)) {

                  size_type perm_i = pimpl_->range().ord(perm_ ^ range_.idx(i));

                  size_type x = 0; // Row of result matrix
                  size_type y = 0; // Column of result matrix

                  // Calculate the matrix coordinates of i
                  if(range_.order() == TiledArray::detail::decreasing_dimension_order) {
                    // i == x * n + y
                    x = i / n_;
                    y = i % n_;
                  } else {
                    // i == y * m + x
                    x = i % m_;
                    y = i / m_;
                  }

                  // Store the future result
                  // x * i_ == The ordinal index of the first tile in left to be contracted
                  // y * i_ == The ordinal index of the first tile in right to be contracted
                  pimpl_->data_.set(perm_i, dot_product(x * i_, y * i_));

                }
              }

              return true;
            }

          private:

            /// Compute the dot_product tensor tiles

            /// Compute row/column \c a of left with column/row \c b of right.
            /// \param a The row/column of the left argument
            /// \param b The column/row of the right argument
            /// \return A \c madness::Future to the dot product result.
            madness::Future<value_type> dot_product(size_type a, size_type b) const {
              // Construct a reduction object
              TiledArray::detail::ReduceTask<value_type, reduce_op >
                  local_reduce_op(pimpl_->get_world(), reduce_op());

              // Generate tasks that will contract tiles and sum the result
              for(size_type i = 0; i < i_; ++i, ++a, ++b) {
                if(!(pimpl_->left().is_zero(a) || pimpl_->right().is_zero(b))) { // Ignore zero tiles
                  local_reduce_op.add(pimpl_->get_world().taskq.add(& EvalImpl::contract,
                      pimpl_->cont_, left(a), right(b)));
                }
              }

              // This will start the reduction tasks, submit the permute task of
              // the result, and return the resulting future
              return pimpl_->get_world().taskq.add(& EvalImpl::permute,
                  local_reduce_op.submit(), perm_);
            }

            /// Contract two tiles

            /// This function is used as a task function for evaluating contractions.
            /// \param cont The contraction definition
            /// \param l The left tile to contract
            /// \param r The right tile to contract
            /// \return The contracted tile
            static value_type contract(const std::shared_ptr<math::Contraction>& cont,
                const typename left_tensor_type::value_type& l,
                const typename right_tensor_type::value_type& r) {
              return cont->contract_tensor(l, r);
            }

            /// Permute a tile

            /// This function is used as a task function for evaluating permutations.
            /// \param t The tensor to be permuted
            /// \param p The permutation to be applied to \c t
            static value_type permute(const value_type& t, const perm_type& p) {
              return make_permute_tensor(t, p);
            }

            /// Product accumulation

            ///
            /// \tparam InIter The input iterator type
            /// \param first The start of the iterator range to be accumulated
            /// \param first The end of the iterator range to be accumulated
            /// \return The product of each value in the iterator range.
            template <typename InIter>
            static typename std::iterator_traits<InIter>::value_type accumulate(InIter first, InIter last) {
              typename std::iterator_traits<InIter>::value_type result = 1ul;
              for(; first != last; ++first)
                result *= *first;
              return result;
            }

            // Container types for holding cached remote tiles.
            typedef madness::ConcurrentHashMap<size_type,
                madness::Future<typename left_tensor_type::value_type> > left_cache_type;
                  ///< The container type used to cache remote tiles for the left argument
            typedef madness::ConcurrentHashMap<size_type,
                madness::Future<typename right_tensor_type::value_type> > right_cache_type;
                  ///< The container type used to cache remote tiles for the right argument

            /// Get tile \c i from the left argument

            /// \param i The ordinal index of a tile in the left argument
            /// \return A future for tile \c i
            madness::Future<typename left_tensor_type::value_type>
            left(size_type i) const {
              return get_cached_value(i, pimpl_->left(), left_cache_);
            }

            /// Get tile \c i from the right argument

            /// \param i The ordinal index of a tile in the left argument
            /// \return A \c madness::Future for tile \c i
            madness::Future<typename right_tensor_type::value_type>
            right(size_type i) const {
              return get_cached_value(i, pimpl_->right(), right_cache_);
            }

            /// Request A tile from \c arg

            /// If the tile is stored locally, the a copy of the future of the
            /// local tile will be returned. If the tile is remote, \c cache is
            /// checked to see if it has already been requested. If it is found
            /// the cached copy is returned. Otherwise the remote tile is fetched
            /// and the future is stored in the cache.
            /// \tparam Arg The argument type
            /// \tparam Cache The cache container type
            /// \param i The ordinal index of the tile to fetch
            /// \param arg The argument that holds tile \c i
            /// \param cache The container that caches remote tiles that have been
            /// requested
            /// \return A \c madness::Future to tile \c i
            template <typename Arg, typename Cache>
            static madness::Future<typename Arg::value_type>
            get_cached_value(size_type i, const Arg& arg, Cache& cache) {
              // If the tile is stored locally, return the local copy
              if(arg.is_local(i))
                return arg[i];

              // Get the remote tile
              typename Cache::accessor acc;
              if(cache.insert(acc, i))
                acc->second = arg[i];
              return acc->second;
            }

            const perm_type perm_; ///< The permutation applied to the resulting contraction
            const std::size_t m_; ///< The number of rows in the packed result matrix
            const std::size_t i_; ///< The number of tile pairs in a contraction
            const std::size_t n_; ///< The number of columns in the packed result matrix
            const range_type range_; ///< A copy of the original, unpermuted result range
            std::shared_ptr<ContractionTiledTensorImpl_> pimpl_; ///< Pimpl to the contraction tile tensor
            mutable left_cache_type left_cache_; ///< Cache for remote tile requests of right
            mutable right_cache_type right_cache_; ///< Cache for remote tile requests of left
          }; // class EvalImpl

        public:
          typedef size_type argument_type;
          typedef bool result_type;

          Eval(const Perm& perm, const std::shared_ptr<ContractionTiledTensorImpl_>& pimpl) :
              pimpl_(new EvalImpl(perm, pimpl))
          { }

          Eval(const Eval<Perm>& other) :
              pimpl_(other.pimpl_)
          { }

          Eval<Perm>& operator=(const Eval<Perm>& other) {
            pimpl_ = other.pimpl_;
            return *this;
          }

          result_type operator()(argument_type arg) const {
            return (*pimpl_)(arg);
          }

        private:
          std::shared_ptr<EvalImpl> pimpl_;
        }; // class Eval



        bool perm_structure(const Permutation& perm, const VariableList& v, bool, bool) {
          trange_ ^= perm;

          // construct the shape
          if(! is_dense()) {
            typedef TiledArray::detail::Bitset<>::value_type bool_type;

            // Todo: This algorithm is inherently non-scalable and it is probably
            // very slow. Since the shape is replicated, there is no other choice
            // other than iterating over the entire range of the tensor.

            Tensor<bool_type, typename left_tensor_type::range_type>
                left_map(left_.range(), left_.get_shape().begin());
            Tensor<bool_type, typename right_tensor_type::range_type>
                right_map(right_.range(), right_.get_shape().begin());

            Tensor<bool_type, range_type> res =
                make_permute_tensor(
                  cont_->contract_tensor(left_map, right_map),
                perm);

            // This could be merged with the tile task initialization, but this
            // will likely change at some point to be incompatible with that
            // algorithm.

            const size_type n = size();
            for(size_type i = 0; i < n; ++i)
              if(res[i])
                shape_.set(i);
          }

          vars_ = v;

          return true;
        }

        bool structure(bool, bool) {
          // construct the shape
          if(! is_dense()) {
            typedef TiledArray::detail::Bitset<>::value_type bool_type;

            Tensor<bool_type, typename left_tensor_type::range_type>
                left_map(left_.range(), left_.get_shape().begin());
            Tensor<bool_type, typename right_tensor_type::range_type>
                right_map(right_.range(), right_.get_shape().begin());

            Tensor<bool_type, range_type> res =
                  cont_->contract_tensor(left_map, right_map);

            const size_type n = size();
            for(size_type i = 0; i < n; ++i)
              shape_.set(i, res[i]);
          }

          return true;
        }

        template <typename Perm>
        bool generate_tasks(const Eval<Perm>& eval_op, bool) {
          // Todo: This algorithm is inherently non-scalable. It is done this
          // way because other expressions depend on all the tiles being present
          // after eval has finished. But there is currently no way to predict
          // which tiles are local so they can be initialized other than
          // iterating through all elements. In the future I would like to use
          // a lazy global synchronization mechanism that will solve this problem.

          // Divide the result processes among all nodes.
          const size_type n = trange_.tiles().volume();
//          const ProcessID r = get_world().rank();
//          const ProcessID s = get_world().size();
//
//          const size_type x = n / s;
//          const size_type y = n % s;
//
//          // There are s * x + y tiles in the result
//          const size_type first = r * x + (r < y ? r : y);
//          const size_type last = first + x + (r < y ? 1 : 0);

          // Generate the tile permutation tasks.
          madness::Future<bool> tiles_done = get_world().taskq.for_each(
              madness::Range<size_type>(0, n, 8), eval_op);
          return true;
        }


      public:

        static madness::Future<bool> eval_struct(const Permutation& perm, const VariableList& v,
            const std::shared_ptr<ContractionTiledTensorImpl_>& pimpl,
            madness::Future<bool> left_done, madness::Future<bool> right_done)
        {

          return pimpl->get_world().taskq.add(*pimpl,
              & ContractionTiledTensorImpl_::perm_structure, perm, v, left_done, right_done,
              madness::TaskAttributes::hipri());
        }

        static madness::Future<bool> eval_struct(const TiledArray::detail::NoPermutation&,
            const VariableList&,
            const std::shared_ptr<ContractionTiledTensorImpl_>& pimpl,
            madness::Future<bool> left_done, madness::Future<bool> right_done)
        {
          return pimpl->get_world().taskq.add(*pimpl,
              & ContractionTiledTensorImpl_::structure, left_done, right_done,
              madness::TaskAttributes::hipri());
        }

        template <typename Perm>
        static madness::Future<bool>
        generate_tiles(const Perm& perm,
            const std::shared_ptr<ContractionTiledTensorImpl_>& pimpl,
            madness::Future<bool> struct_done) {
          return pimpl->get_world().taskq.add(*pimpl,
              & ContractionTiledTensorImpl_::template generate_tasks<Perm>, Eval<Perm>(perm,
              pimpl), struct_done);
        }

        madness::Future<bool> eval_left() {
          return left_.eval(cont_->left_vars(left_.vars()));
        }

        madness::Future<bool> eval_right() {
          return right_.eval(cont_->right_vars(right_.vars()));
        }

        static bool done(bool, bool) { return true; }


        /// Construct a unary tiled tensor op

        /// \param arg The argument
        /// \param op The element transform operation
        ContractionTiledTensorImpl(const left_tensor_type& left, const right_tensor_type& right) :
          left_(left), right_(right),
          cont_(new math::Contraction(left.vars(), right.vars(), left.range().order())),
          trange_(cont_->contract_trange(left.trange(), right.trange())),
          shape_((left.is_dense() || right.is_dense() ? 0 : trange_.tiles().volume())),
          vars_(cont_->contract_vars(left.vars(), right.vars())),
          data_(left.get_world(), trange_.tiles().volume(), left.get_pmap())
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
        const range_type& range() const { return trange_.tiles(); }

        /// Tensor tile volume accessor

        /// \return The number of tiles in the tensor
        size_type size() const { return trange_.tiles().volume(); }

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
        bool is_dense() const { return left_.is_dense() && right_.is_dense(); }

        /// Tensor shape accessor

        /// \return A reference to the tensor shape map
        const TiledArray::detail::Bitset<>& get_shape() const {
          TA_ASSERT(! is_dense());
          return shape_;
        }

        /// Tiled range accessor

        /// \return The tiled range of the tensor
        const trange_type& trange() const { return trange_; }

        /// Tile accessor

        /// \param i The tile index
        /// \return Tile \c i
        const_reference get_local(size_type i) const {
          TA_ASSERT(left_.is_local(i));
          TA_ASSERT(right_.is_local(i));
          return op_(left_.get_local(i), right_.get_local(i));
        }


        /// Array begin iterator

        /// \return A const iterator to the first element of the array.
        const_iterator begin() const { return data_.begin(); }

        /// Array end iterator

        /// \return A const iterator to one past the last element of the array.
        const_iterator end() const { return data_.end(); }

        /// Variable annotation for the array.
        const VariableList& vars() const { return vars_; }

        madness::World& get_world() const { return data_.get_world(); }

        const left_tensor_type& left() const { return left_; }

        left_tensor_type& left() { return left_; }

        const right_tensor_type& right() const { return right_; }

        right_tensor_type& right() { return right_; }
      }; // class ContractionTiledTensorImpl

    } // namespace

    /// Tensor that is composed from an argument tensor

    /// The tensor elements are constructed using a unary transformation
    /// operation.
    /// \tparam Arg The argument type
    /// \tparam Op The Unary transform operator type.
    template <typename Left, typename Right>
    class ContractionTiledTensor : public ReadableTiledTensor<ContractionTiledTensor<Left, Right> > {
    public:
      typedef ContractionTiledTensor<Left, Right> ContractionTiledTensor_;
      typedef ContractionTiledTensorImpl<Left, Right> impl_type;
      typedef Left left_tensor_type;
      typedef Right right_tensor_type;
      TILEDARRAY_READABLE_TILED_TENSOR_INHERIT_TYPEDEF(ReadableTiledTensor<ContractionTiledTensor_>, ContractionTiledTensor_);
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type;

    private:
      std::shared_ptr<impl_type> pimpl_;

    public:

      /// Construct a unary tiled tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      ContractionTiledTensor(const left_tensor_type& left, const right_tensor_type& right) :
        pimpl_(new impl_type(left, right),
            madness::make_deferred_deleter<impl_type>(left.get_world()))
      { }

      /// Copy constructor

      /// Create a shallow copy of \c other.
      /// \param other The object to be copied
      ContractionTiledTensor(const ContractionTiledTensor_& other) :
        pimpl_(other.pimpl_)
      { }

      /// Assignment operator

      /// Create a shallow copy of \c other.
      /// \param other THe object to be copied
      /// \return A reference to this object
      ContractionTiledTensor_& operator=(const ContractionTiledTensor_& other) {
        pimpl_ = other.pimpl_;
        return *this;
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


      madness::Future<bool> eval(const VariableList& v) {

        madness::Future<bool> left_child = pimpl_->eval_left();
        madness::Future<bool> right_child = pimpl_->eval_right();

        madness::Future<bool> struct_done;

        if(v != pimpl_->vars()) {

          // Get the permutation for the results
          Permutation perm = pimpl_->vars().permutation(v);

          // Generate tile tasks
          // This needs to be done before eval structure.
          madness::Future<bool> tiles_done = impl_type::generate_tiles(perm, pimpl_, struct_done);

          // Task to permute the vars, shape, and trange.
          struct_done.set(impl_type::eval_struct(perm, v, pimpl_, left_child, right_child));

          return tiles_done;

        }

        // This needs to be done before eval structure.
        madness::Future<bool> tiles_done =
            impl_type::generate_tiles(TiledArray::detail::NoPermutation(), pimpl_,
            struct_done);

        // Task to construct the shape the shape.
        struct_done.set(impl_type::eval_struct(TiledArray::detail::NoPermutation(),
            v, pimpl_, left_child, right_child));

        return tiles_done;
      }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const range_type& range() const { return trange().tiles(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const { return range().volume(); }

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
      const TiledArray::detail::Bitset<>& get_shape() const { return pimpl_->get_shape(); }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const { return pimpl_->trange(); }

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
    }; // class ContractionTiledTensor


  }  // namespace expressions
}  // namespace TiledArray

namespace madness {
  namespace archive {

    template <typename Archive, typename T>
    struct ArchiveStoreImpl;
    template <typename Archive, typename T>
    struct ArchiveLoadImpl;

    template <typename Archive>
    struct ArchiveStoreImpl<Archive, std::shared_ptr<TiledArray::math::Contraction> > {
      static void store(const Archive&, const std::shared_ptr<TiledArray::math::Contraction>&) {
        TA_ASSERT(false);
      }
    };

    template <typename Archive>
    struct ArchiveLoadImpl<Archive, std::shared_ptr<TiledArray::math::Contraction> > {

      static void load(const Archive&, std::shared_ptr<TiledArray::math::Contraction>&) {
        TA_ASSERT(false);
      }
    };
  } // namespace archive
} // namespace madness

#endif // TILEDARRAY_CONTRACTION_TILED_TENSOR_H__INCLUDED
