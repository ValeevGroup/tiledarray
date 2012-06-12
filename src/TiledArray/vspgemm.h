#ifndef TILEDARRAY_VSPGEMM_H__INCLUDED
#define TILEDARRAY_VSPGEMM_H__INCLUDED

#include <TiledArray/contraction_algorithm_base.h>

namespace TiledArray {
  namespace detail {

    /// Very Sparse General Matrix Multiplication
    template <typename Impl>
    class VSpGemm : public madness::WorldObject<Summa<Impl> >, public ContractionAlgorithmBase<Impl>, private NO_DEFAULTS {
    protected:
      typedef madness::WorldObject<VSpGemm<Impl> > WorldObject_; ///< Madness world object base class
      typedef ContractionAlgorithmBase<Impl> base;

    public:
      typedef VSpGemm<Impl> VSpGemm_; ///< This object type
      typedef typename base::size_type size_type; ///< size type
      typedef typename base::value_type value_type; ///< The result value type
      typedef typename base::left_tensor_type left_tensor_type; ///< The left tensor type
      typedef typename base::left_value_type left_value_type; /// The left tensor value type
      typedef typename base::right_tensor_type right_tensor_type; ///< The right tensor type
      typedef typename base::right_value_type right_value_type; ///< The right tensor value type
      typedef typename base::pmap_interface pmap_interface; ///< The process map interface type

    private:
      /// The left tensor cache container type
      typedef madness::ConcurrentHashMap<size_type, madness::Future<left_value_type> > left_container;

      /// The right tensor cache container type
      typedef madness::ConcurrentHashMap<size_type, madness::Future<right_value_type> > right_container;

      left_container left_cache_;
      right_container right_cache_;



      /// Contract and reduce operation

      /// This object handles contraction and reduction of tensor tiles.
      class contract_reduce_op {
      public:
        typedef left_value_type first_argument_type; ///< The left tile type
        typedef right_value_type second_argument_type; ///< The right tile type
        typedef value_type result_type; ///< The result tile type.

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
          return cont_->contract_tensor(first1, second1, first2, second2);
        }

      private:
        std::shared_ptr<math::Contraction> cont_; ///< The contraction definition object pointer
      }; // class contract_reduce_op

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
        }
        return acc->second;
      }

      madness::Future<left_value_type> get_left(const size_type i) {
        return get_cached_value(i, base::left(), left_cache_);
      }

      madness::Future<right_value_type> get_right(const size_type i) {
        return get_cached_value(i, base::right(), right_cache_);
      }

      /// Compute result tile for \c i,j

      /// Compute row/column \c a of left with column/row \c b of right.
      /// \param i The row of the result tile to be computed
      /// \param j The column of the result tile to be computed
      /// \return \c madness::None
      madness::Void dot_product(size_type i, size_type j) const {

        // Construct a reduction object
        TiledArray::detail::ReducePairTask<contract_reduce_op>
            local_reduce_op(WorldObject_::get_world(), contract_reduce_op(pimpl_->cont_));

        // Generate tasks that will contract tiles and sum the result
        size_type a = i * base::k_;
        size_type b = j * base::k_;
        const size_type end = a + base::k_;

        // Contract each pair of tiles in the dot product
        for(; a < end; ++a, ++b)
          if(!(base::left().is_zero(a) || base::right().is_zero(b)))
            local_reduce_op.add(get_left(a), get_right(b));

        TA_ASSERT(local_reduce_op.count() != 0ul);
        // This will start the reduction tasks, submit the permute task of
        // the result of the reduction, and return the resulting future
        base::set(i * base::n_ + j, local_reduce_op.submit());

        return madness::None;
      }

    public:
      VSpGemm(const std::shared_ptr<Impl>& pimpl) :
          WorldObject_(pimpl->get_world()),
          base(pimpl),
          left_cache_(base::local_rows_ * base::k_),
          right_cache_(base::local_cols_ * base::k_)
      {
        WorldObject_::process_pending();
      }

      virtual ~VSpGemm() { }

    private:

      /// Cleaup local data for contraction arguments
      class Cleanup {
      private:
        VSpGemm_& owner_;
      public:
        Cleanup(VSpGemm_& owner) : owner_(owner) { }

        void operator()() const {
          owner_.left().clear();
          owner_.right().clear();
        }
      }; // class Cleanup

      virtual void do_eval() {
        // Spawn task for local tile evaluation
        for(size_type i = base::rank_row_; i < base::m_; i += base::proc_rows_) {
          for(size_type j = base::rank_col_; j < base::n_; j += base::proc_cols_) {
            if(! base::is_zero(i * base::n_ + j))
              WorldObject_::task(base::rank_, & VSpGemm_::template dot_product<Perm>, i, j);
          }
        }

        // Cleanup argument data
        lazy_sync(WorldObject_::get_world(), WorldObject_::id(), Cleanup(*this));

        // Cleanup local tile cache
        left_cache_.clear();
        right_cache_.clear();
      }

    }; // class VSpGemm

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_VSPGEMM_H__INCLUDED
