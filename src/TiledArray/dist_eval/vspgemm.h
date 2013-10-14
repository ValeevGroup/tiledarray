/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_VSPGEMM_H__INCLUDED
#define TILEDARRAY_VSPGEMM_H__INCLUDED

#include <TiledArray/contraction_tensor_impl.h>
#include <TiledArray/dist_op/lazy_sync.h>
#include <TiledArray/reduce_task.h>
#include <TiledArray/pmap/hash_pmap.h>

namespace TiledArray {
  namespace expressions {

    /// Very Sparse General Matrix Multiplication
    template <typename Left, typename Right>
    class VSpGemm : public madness::WorldObject<VSpGemm<Left, Right> >, public ContractionTensorImpl<Left, Right> {
    protected:
      typedef madness::WorldObject<VSpGemm<Left, Right> > WorldObject_; ///< Madness world object base class
      typedef ContractionTensorImpl<Left, Right> ContractionTensorImpl_;
      typedef typename ContractionTensorImpl_::TensorExpressionImpl_ DistEvalImpl_;
      typedef typename DistEvalImpl_::TensorImpl_ TensorImpl_;

    public:
      typedef VSpGemm<Left, Right> VSpGemm_; ///< This object type
      typedef typename ContractionTensorImpl_::size_type size_type; ///< size type
      typedef typename ContractionTensorImpl_::value_type value_type; ///< The result value type
      typedef typename ContractionTensorImpl_::left_tensor_type left_tensor_type; ///< The left tensor type
      typedef typename ContractionTensorImpl_::left_value_type left_value_type; /// The left tensor value type
      typedef typename ContractionTensorImpl_::right_tensor_type right_tensor_type; ///< The right tensor type
      typedef typename ContractionTensorImpl_::right_value_type right_value_type; ///< The right tensor value type
      typedef typename ContractionTensorImpl_::pmap_interface pmap_interface; ///< The process map interface type

    private:

      // Constants that define the data layout and sizes
      using ContractionTensorImpl_::rank_; ///< This process's rank
      using ContractionTensorImpl_::size_; ///< Then number of processes
      using ContractionTensorImpl_::m_; ///< Number of element rows in the result and left matrix
      using ContractionTensorImpl_::n_; ///< Number of element columns in the result matrix and rows in the right argument matrix
      using ContractionTensorImpl_::k_; ///< Number of element columns in the left and right argument matrices
      using ContractionTensorImpl_::mk_; ///< Number of elements in left matrix
      using ContractionTensorImpl_::kn_; ///< Number of elements in right matrix
      using ContractionTensorImpl_::proc_cols_; ///< Number of columns in the result process map
      using ContractionTensorImpl_::proc_rows_; ///< Number of rows in the result process map
      using ContractionTensorImpl_::proc_size_; ///< Number of process in the process map. This may be
                         ///< less than the number of processes in world.
      using ContractionTensorImpl_::rank_row_; ///< This node's row in the process map
      using ContractionTensorImpl_::rank_col_; ///< This node's column in the process map
      using ContractionTensorImpl_::local_rows_; ///< The number of local element rows
      using ContractionTensorImpl_::local_cols_; ///< The number of local element columns
      using ContractionTensorImpl_::local_size_; ///< Number of local elements

      /// The left tensor cache container type
      typedef madness::ConcurrentHashMap<size_type, madness::Future<left_value_type> > left_container;

      /// The right tensor cache container type
      typedef madness::ConcurrentHashMap<size_type, madness::Future<right_value_type> > right_container;

      typedef detail::ContractReduceOp<Left, Right> contract_reduce_op;

      left_container left_cache_;
      right_container right_cache_;
      madness::AtomicInt count_;

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
      get_cached_value(const size_type i, const Arg& arg, Cache& cache) const {
        // If the tile is stored locally, return the local copy
        if(arg.is_local(i))
          return arg[i];

        // Get the remote tile
        typename Cache::accessor acc;
        if(cache.insert(acc, i))
          acc->second = arg[i];
        return acc->second;
      }

      madness::Future<left_value_type> get_left(const size_type i) {
        return get_cached_value(i, ContractionTensorImpl_::left(), left_cache_);
      }

      madness::Future<right_value_type> get_right(const size_type i) {
        return get_cached_value(i, ContractionTensorImpl_::right(), right_cache_);
      }

      /// Compute result tile for \c i,j

      /// Compute row/column \c a of left with column/row \c b of right.
      /// \param i The row of the result tile to be computed
      /// \param j The column of the result tile to be computed
      /// \return \c madness::None
      void dot_product(const size_type i, const size_type j) {
        // Construct a reduction object
        TiledArray::detail::ReducePairTask<contract_reduce_op>
            local_reduce_op(WorldObject_::get_world(), contract_reduce_op(*this));

        // Generate tasks that will contract tiles and sum the result
        size_type a = i * k_;
        size_type b = j;
        const size_type end = a + k_;

        // Contract each pair of tiles in the dot product
        for(; a < end; ++a, b += n_)
          if(!(ContractionTensorImpl_::left().is_zero(a) || ContractionTensorImpl_::right().is_zero(b)))
            local_reduce_op.add(get_left(a), get_right(b));

        TA_ASSERT(local_reduce_op.count() != 0ul);
        // This will start the reduction tasks, submit the permute task of
        // the result of the reduction, and return the resulting future
        ContractionTensorImpl_::set(i * n_ + j, local_reduce_op.submit());

      }

    public:
      VSpGemm(const left_tensor_type& left, const right_tensor_type& right) :
          WorldObject_(left.get_world()),
          ContractionTensorImpl_(left, right),
          left_cache_(local_rows_ * k_),
          right_cache_(local_cols_ * k_)
      {
        count_ = local_rows_ * local_cols_;
        WorldObject_::process_pending();
      }

      /// Virtual destructor
      virtual ~VSpGemm() { }

    private:

      /// Cleaup local data for contraction arguments

      /// This object is used by lazy sync to cleanup argument data
      class Cleanup {
      private:
        VSpGemm_* owner_;
      public:
        Cleanup() : owner_(NULL) { }
        Cleanup(const Cleanup& other) : owner_(other.owner_) { }
        Cleanup(VSpGemm_& owner) : owner_(& owner) { }

        Cleanup& operator=(const Cleanup& other) {
          owner_ = other.owner_;
          return *this;
        }

        void operator()() {
          TA_ASSERT(owner_);
          // Release argument data
          owner_->left().release();
          owner_->right().release();
          // Clear cache data
          owner_->left_cache_.clear();
          owner_->right_cache_.clear();
        }
      }; // class Cleanup

    private:

      /// Construct the left argument process map
      virtual std::shared_ptr<pmap_interface> make_left_pmap() const {
        return std::shared_ptr<pmap_interface>(new TiledArray::detail::HashPmap(
            TensorImpl_::get_world(), mk_));
      }

      /// Construct the right argument process map
      virtual std::shared_ptr<pmap_interface> make_right_pmap() const {
        return std::shared_ptr<pmap_interface>(new TiledArray::detail::HashPmap(
            TensorImpl_::get_world(), kn_));
      }

      virtual void eval_tiles() {
        // Spawn task for local tile evaluation
        for(size_type i = rank_row_; i < m_; i += proc_rows_)
          for(size_type j = rank_col_; j < n_; j += proc_cols_) {
            if(! TensorImpl_::is_zero(DistEvalImpl_::perm_index(i * n_ + j))) {
              WorldObject_::task(rank_, & VSpGemm_::dot_product, i, j);
            } else if((count_--) == 1) {
              // Cleanup data for children if this is the last tile
              lazy_sync(WorldObject_::get_world(), WorldObject_::id(), Cleanup(*this));
            }
          }
      }

    }; // class VSpGemm

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_VSPGEMM_H__INCLUDED
