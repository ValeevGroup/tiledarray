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

#ifdef TILEDARRAY_LOG_EVENTS
#include <TiledArray/event_log.h>
#endif // TILEDARRAY_LOG_EVENTS

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename, typename>
    class ContractionTiledTensor;

    template <typename LExp, typename RExp>
    ContractionTiledTensor<LExp, RExp>
    make_contraction_tiled_tensor(const ReadableTiledTensor<LExp>& left, const ReadableTiledTensor<RExp>& right) {
      return ContractionTiledTensor<LExp, RExp>(left.derived(), right.derived());
    }

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

    namespace detail {
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
        typedef ReadableTiledTensor<ContractionTiledTensor_> base;
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
        ContractionTiledTensorImpl(const ContractionTiledTensorImpl_&);
        ContractionTiledTensor_& operator=(const ContractionTiledTensor_&);


        left_tensor_type left_; ///< Left argument
        right_tensor_type right_; ///< Right argument
        std::shared_ptr<math::Contraction> cont_;
        trange_type trange_;
        TiledArray::detail::Bitset<> shape_;
        VariableList vars_;
        storage_type data_;

#ifdef TILEDARRAY_LOG_EVENTS
        static unsigned int log_count_;
        TiledArray::logging::EventLog result_log_;
        TiledArray::logging::EventLog work_log_;
        TiledArray::logging::EventLog request_log_;
        TiledArray::logging::EventLog receive_log_;
        unsigned int log_id_;
#endif

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

            /// Contract and reduce operation

            /// This object handles contraction and reduction of tensor tiles.
            class contract_reduce_op {
            public:
              typedef typename left_tensor_type::value_type first_argument_type; ///< The left tile type
              typedef typename right_tensor_type::value_type second_argument_type; ///< The right tile type
              typedef value_type result_type; ///< The result tile type.

#ifdef TILEDARRAY_LOG_EVENTS
              explicit contract_reduce_op(const std::shared_ptr<math::Contraction>& cont, TiledArray::logging::EventLog& work_log) :
                cont_(cont), work_log_(&work_log)
              { }

              /// Functor copy constructor

              /// Shallow copy of this functor
              /// \param other The functor to be copied
              contract_reduce_op(const contract_reduce_op& other) :
                cont_(other.cont_), work_log_(other.work_log_)
              { }

              /// Functor assignment operator

              /// Shallow copy of this functor
              /// \param other The functor to be copied
              contract_reduce_op& operator=(const contract_reduce_op& other) {
                cont_ = other.cont_;
                work_log_ = other.work_log_;
                return *this;
              }
#else

              /// Construct contract/reduce functor

              /// \param cont Shared pointer to contraction definition object
              explicit contract_reduce_op(const std::shared_ptr<math::Contraction>& cont) :
                cont_(cont)
              { }

              /// Functor copy constructor

              /// Shallow copy of this functor
              /// \param other The functor to be copied
              contract_reduce_op(const contract_reduce_op& other) : cont_(other.cont_) { }

              /// Functor assignment operator

              /// Shallow copy of this functor
              /// \param other The functor to be copied
              contract_reduce_op& operator=(const contract_reduce_op& other) {
                cont_ = other.cont_;
                return *this;
              }
#endif // TILEDARRAY_LOG_EVENTS



              /// Create a result type object

              /// Initialize a result object for subsequent reductions
              result_type operator()() const {
                return result_type();
              }

              /// Reduce two result objects

              /// Add \c arg to \c result .
              /// \param[in,out] result The result object that will be the reduction target
              /// \param[in] arg The argument that will be added to \c result
              void operator()(result_type& result, const result_type& arg) const {
                result += arg;
#ifdef TILEDARRAY_LOG_EVENTS
                work_log_->notify();
#endif
              }


              /// Contract a pair of tiles and add to a target tile

              /// Contracte \c left and \c right and add the result to \c result.
              /// \param[in,out] result The result object that will be the reduction target
              /// \param[in] left The left-hand tile to be contracted
              /// \param[in] right The right-hand tile to be contracted
              void operator()(result_type& result, const first_argument_type& first, const second_argument_type& second) const {
                if(result.empty())
                  result = result_type(cont_->result_range(first.range(), second.range()));
                cont_->contract_tensor(result, first, second);
#ifdef TILEDARRAY_LOG_EVENTS
                work_log_->notify();
#endif
              }

              /// Contract a pair of tiles and add to a target tile

              /// Contracte \c left1 with \c right1 and \c left2 with \c right2 ,
              /// and add the two results.
              /// \param[in] left The first left-hand tile to be contracted
              /// \param[in] right The first right-hand tile to be contracted
              /// \param[in] left The second left-hand tile to be contracted
              /// \param[in] right The second right-hand tile to be contracted
              /// \return A tile that contains the sum of the two contractions.
              result_type operator()(const first_argument_type& first1, const second_argument_type& second1,
                  const first_argument_type& first2, const second_argument_type& second2) const {

#ifdef TILEDARRAY_LOG_EVENTS
                result_type result = cont_->contract_tensor(first1, second1, first2, second2);
                work_log_->notify();
                work_log_->notify();
                return result;
#else
                return cont_->contract_tensor(first1, second1, first2, second2);
#endif
              }

            private:
              std::shared_ptr<math::Contraction> cont_; ///< The contraction definition object pointer
#ifdef TILEDARRAY_LOG_EVENTS
              TiledArray::logging::EventLog* work_log_;
#endif // TILEDARRAY_LOG_EVENTS
            }; // class contract_reduce_op

          public:

            /// Constructor

            /// \param perm The permuation that will be applied to the results.
            /// \param pimpl The implmentation pointer of the ContractionTiledTensor
            EvalImpl(const perm_type& perm, const std::shared_ptr<ContractionTiledTensorImpl_>& pimpl) :
                perm_(perm),
                m_(pimpl->cont_->left_outer(pimpl->left_.range())),
                i_(pimpl->cont_->left_inner(pimpl->left_.range())),
                n_(pimpl->cont_->right_outer(pimpl->right_.range())),
                range_(pimpl->range()),
                pimpl_(pimpl),
                left_cache_(pimpl->left().range().volume()),
                right_cache_(pimpl->right().range().volume())
            { }

            /// Generate tasks to evaluate tile \c i

            /// \param i The tile to be evaluated
            /// \return true
            bool operator()(const size_type i) const {
              if(pimpl_->is_local(i)) {
                if(! pimpl_->is_zero(i)) {
                  const size_type ii = map_ord(i, perm_);

                  size_type x = 0; // Row of result matrix
                  size_type y = 0; // Column of result matrix

                  // Calculate the matrix coordinates of i
                  // ii == x * n + y
                  x = ii / n_;
                  y = ii % n_;

                  // Store the future result
                  // x * i_ == The ordinal index of the first tile in left to be contracted
                  // y * i_ == The ordinal index of the first tile in right to be contracted
#ifdef TILEDARRAY_LOG_EVENTS
                  madness::Future<value_type> result = dot_product(x * i_, y * i_);
                  result.register_callback(&(pimpl_->result_log_));
                  pimpl_->data_.set(i, result);
#else
                  pimpl_->data_.set(i, dot_product(x * i_, y * i_));
#endif // TILEDARRAY_LOG_EVENTS

                }
              }

              return true;
            }

            std::size_t rows() const { return m_; }

            std::size_t cols() const { return n_; }

          private:

            /// Compute the dot_product tensor tiles

            /// Compute row/column \c a of left with column/row \c b of right.
            /// \param a The row/column of the left argument
            /// \param b The column/row of the right argument
            /// \return A \c madness::Future to the dot product result.
            madness::Future<value_type> dot_product(size_type a, size_type b) const {
              // Construct a reduction object
#ifdef TILEDARRAY_LOG_EVENTS
              TiledArray::detail::ReducePairTask<contract_reduce_op>
                  local_reduce_op(pimpl_->get_world(), contract_reduce_op(pimpl_->cont_, pimpl_->work_log_));
#else
              TiledArray::detail::ReducePairTask<contract_reduce_op>
                  local_reduce_op(pimpl_->get_world(), contract_reduce_op(pimpl_->cont_));
#endif // TILEDARRAY_LOG_EVENTS

              // Generate tasks that will contract tiles and sum the result
              for(size_type n = 0; n < i_; ++n, ++a, ++b)
                if(!(pimpl_->left().is_zero(a) || pimpl_->right().is_zero(b))) // Ignore zero tiles
                  local_reduce_op.add(left(a), right(b));

              TA_ASSERT(local_reduce_op.count() != 0ul);
              // This will start the reduction tasks, submit the permute task of
              // the result of the reduction, and return the resulting future
              return make_permute_task(local_reduce_op.submit(), perm_);
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
            static value_type permute(const value_type& t, const Permutation& p) {
              return make_permute_tensor(t, p);
            }

            madness::Future<value_type> make_permute_task(const madness::Future<value_type>& tile, const Permutation& p) const {
              return pimpl_->get_world().taskq.add(& EvalImpl::permute,
                  tile, perm_);
            }

            const madness::Future<value_type>&
            make_permute_task(const madness::Future<value_type>& tile, const TiledArray::detail::NoPermutation&) const {
              return tile;
            }

            size_type map_ord(size_type i, const Permutation& p) const {
              return range_.ord(-perm_ ^ pimpl_->range().idx(i));
            }

            size_type map_ord(size_type i, const TiledArray::detail::NoPermutation&) const {
              return i;
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
            madness::Future<typename Arg::value_type>
            get_cached_value(size_type i, const Arg& arg, Cache& cache) const {
              // If the tile is stored locally, return the local copy
              if(arg.is_local(i))
                return arg[i];

              // Get the remote tile
              typename Cache::accessor acc;
              if(cache.insert(acc, i)) {
                acc->second = arg[i];
#ifdef TILEDARRAY_LOG_EVENTS
                pimpl_->request_log_.notify();
                acc->second.register_callback(& (pimpl_->receive_log_));
#endif // TILEDARRAY_LOG_EVENTS
              }
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

          result_type operator()(typename pmap_interface::const_iterator it) const {
            return (*pimpl_)(*it);
          }

          std::size_t rows() const { return pimpl_->rows(); }

          std::size_t cols() const { return pimpl_->cols(); }

        private:
          std::shared_ptr<EvalImpl> pimpl_;
        }; // class Eval



        bool perm_structure(const Permutation& perm, const VariableList& v) {
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

        bool perm_structure(const TiledArray::detail::NoPermutation&, const VariableList&) {
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
        bool generate_tasks(const Perm& perm, const VariableList& v,
            const std::shared_ptr<ContractionTiledTensorImpl_>& pimpl, bool, bool) {
          Eval<Perm> eval_op(perm, pimpl);
          perm_structure(perm, v);

          // Todo: This algorithm is inherently non-scalable. It is done this
          // way because other expressions depend on all the tiles being present
          // after eval has finished. But there is currently no way to predict
          // which tiles are local so they can be initialized other than
          // iterating through all elements. In the future I would like to use
          // a lazy global synchronization mechanism that will solve this problem.

          // Divide the result processes among all nodes.
          const size_type n = size();
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
          return get_world().taskq.for_each(madness::Range<typename pmap_interface::const_iterator>(data_.get_pmap()->begin(), data_.get_pmap()->end(), eval_op.rows()),
              eval_op).get(); // Wait for for_each() to finish while still processing other tasks
        }

#ifdef TILEDARRAY_LOG_EVENTS
        std::size_t sqr(std::size_t x) {
          return x * x;
        }
#endif // TILEDARRAY_LOG_EVENTS

      public:

        template <typename Perm>
        static madness::Future<bool>
        generate_tiles(const Perm& perm, const VariableList& v,
            const std::shared_ptr<ContractionTiledTensorImpl_>& pimpl) {
          return pimpl->get_world().taskq.add(*pimpl,
              & ContractionTiledTensorImpl_::template generate_tasks<Perm>, perm,
              v, pimpl, pimpl->eval_left(), pimpl->eval_right(),
              madness::TaskAttributes::hipri());
        }

        madness::Future<bool> eval_left() {
          return left_.eval(cont_->left_vars(left_.vars()),
              std::shared_ptr<TiledArray::detail::BlockedPmap>(
              new TiledArray::detail::BlockedPmap(get_world(), left_.size())));
        }

        madness::Future<bool> eval_right() {
          return right_.eval(cont_->right_vars(right_.vars()),
              std::shared_ptr<TiledArray::detail::BlockedPmap>(
              new TiledArray::detail::BlockedPmap(get_world(), right_.size())));
        }

        static bool done(bool, bool) { return true; }


        /// Construct a unary tiled tensor op

        /// \param arg The argument
        /// \param op The element transform operation
        ContractionTiledTensorImpl(const left_tensor_type& left, const right_tensor_type& right) :
          left_(left), right_(right),
          cont_(new math::Contraction(left.vars(), right.vars())),
          trange_(cont_->contract_trange(left.trange(), right.trange())),
          shape_((left.is_dense() || right.is_dense() ? 0 : trange_.tiles().volume())),
          vars_(cont_->contract_vars(left.vars(), right.vars())),
          data_(left.get_world(), trange_.tiles().volume())
#ifdef TILEDARRAY_LOG_EVENTS
          , result_log_("ContractTiledTensor: set result tile", cont_->contract_range(left.range(), right.range()).volume())
          , work_log_("ContractTiledTensor: contract/reduce tile", sqr(cont_->contract_range(left.range(), right.range()).volume()))
          , receive_log_("ContractTiledTensor: receive argument tile", left.range().volume() + right.range().volume())
          , request_log_("ContractTiledTensor: request argument tile", left.range().volume() + right.range().volume())
          , log_id_(log_count_++)
#endif // TILEDARRAY_LOG_EVENTS
        { }

#ifdef TILEDARRAY_LOG_EVENTS
        ~ContractionTiledTensorImpl() {
          std::stringstream ss;
          ss << "contraction" << log_id_ << "-" << get_world().rank() << ".log";
          std::ofstream file(ss.str().c_str());
          TA_ASSERT(! file.fail());
          file << request_log_ << receive_log_ << work_log_ << result_log_;
          file.close();
        }
#endif // TILEDARRAY_LOG_EVENTS


        void set_pmap(const std::shared_ptr<pmap_interface>& pmap) {
          data_.init(pmap);
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
        const VariableList& vars() const { return vars_; }

        madness::World& get_world() const { return data_.get_world(); }

        const left_tensor_type& left() const { return left_; }

        left_tensor_type& left() { return left_; }

        const right_tensor_type& right() const { return right_; }

        right_tensor_type& right() { return right_; }


        /// Clear the tile data

        /// Remove all tiles from the tensor.
        /// \note: Any tiles will remain in memory until the last reference
        /// is destroyed.
        void clear() { data_.clear(); }

      }; // class ContractionTiledTensorImpl


#ifdef TILEDARRAY_LOG_EVENTS
      template <typename Left, typename Right>
      unsigned int ContractionTiledTensorImpl<Left, Right>::log_count_ = 0u;
#endif // TILEDARRAY_LOG_EVENTS

    } // namespace detail

    /// Tensor that is composed from an argument tensor

    /// The tensor elements are constructed using a unary transformation
    /// operation.
    /// \tparam Arg The argument type
    /// \tparam Op The Unary transform operator type.
    template <typename Left, typename Right>
    class ContractionTiledTensor : public ReadableTiledTensor<ContractionTiledTensor<Left, Right> > {
    public:
      typedef ContractionTiledTensor<Left, Right> ContractionTiledTensor_;
      typedef detail::ContractionTiledTensorImpl<Left, Right> impl_type;
      typedef Left left_tensor_type;
      typedef Right right_tensor_type;
      typedef ReadableTiledTensor<ContractionTiledTensor_> base;
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
      std::shared_ptr<impl_type> pimpl_;

    public:

      ContractionTiledTensor() : pimpl_() { }

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
        TA_ASSERT(pimpl_);
        TA_ASSERT(range() == dest.range());

        // Add result tiles to dest
        for(const_iterator it = begin(); it != end(); ++it)
          dest.set(it.index(), *it);
      }


      madness::Future<bool> eval(const VariableList& v, const std::shared_ptr<pmap_interface>& pmap) {
        TA_ASSERT(pimpl_);

        pimpl_->set_pmap(pmap);

        if(v != pimpl_->vars()) {

          // Get the permutation to go from the current variable list to v such
          // that:
          //   v = perm ^ pimpl_->vars()
          Permutation perm = v.permutation(pimpl_->vars());

          // Generate tile tasks
          // This needs to be done before eval structure.
          return impl_type::generate_tiles(perm, v, pimpl_);;

        }

        // This needs to be done before eval structure.
        return impl_type::generate_tiles(TiledArray::detail::NoPermutation(), v, pimpl_);
      }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const range_type& range() const {
        TA_ASSERT(pimpl_);
        return trange().tiles();
      }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const {
        TA_ASSERT(pimpl_);
        return range().volume();
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
      const TiledArray::detail::Bitset<>& get_shape() const {
        TA_ASSERT(pimpl_);
        return pimpl_->get_shape();
      }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const {
        TA_ASSERT(pimpl_);
        return pimpl_->trange();
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


      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      const_reference operator[](size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->operator[](i);
      }

      /// Release tensor data

      /// Clear all tensor data from memory. This is equivalent to
      /// \c ContractionTiledTensor().swap(*this) .
      void release() {
        if(pimpl_) {
          pimpl_->clear();
          pimpl_.reset();
        }
      }

      template <typename Archive>
      void serialize(const Archive&) { TA_ASSERT(false); }

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
