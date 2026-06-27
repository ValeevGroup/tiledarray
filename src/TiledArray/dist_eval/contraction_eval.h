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

#ifndef TILEDARRAY_DIST_EVAL_CONTRACTION_EVAL_H__INCLUDED
#define TILEDARRAY_DIST_EVAL_CONTRACTION_EVAL_H__INCLUDED

#include <atomic>
#include <limits>
#include <map>
#include <mutex>
#include <vector>

#include <TiledArray/config.h>
#include <TiledArray/dist_eval/dist_eval.h>
#include <TiledArray/expressions/contraction_retile.h>
#include <TiledArray/proc_grid.h>
#include <TiledArray/reduce_task.h>
#include <TiledArray/shape.h>
#include <TiledArray/type_traits.h>

#include <TiledArray/tensor/arena_retile.h>
#include <TiledArray/tensor/arena_tensor.h>
#include <TiledArray/tensor/type_traits.h>

// #define TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL 1
// #define TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE 1
// #define TILEDARRAY_ENABLE_SUMMA_TRACE_STEP 1
// #define TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST 1
// #define TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE 1

namespace TiledArray {
namespace detail {

#ifdef TA_STRIDED_DGEMM_COUNT
/// identity anchor: incremented ONLY inside `plan_.active`
/// branches of `Summa`. When a retile plan is inactive this MUST stay 0 --
/// the unit tests assert that, proving the two-trange (fine/coarse) split
/// introduced no active code path on the stock SUMMA path. gop.sum it across
/// ranks for np-correctness.
inline std::atomic<std::size_t> g_summa_plan_active_calls{0};

/// debug hook: total number of FINE U K-tiles gathered+packed per
/// coarse K-cell, accumulated across the active inbound-coarsen gathers in
/// `get_col`/`get_row`. Stays 0 on the inactive (stock SUMMA) path. gop.sum it
/// across ranks for np-correctness. For a single coarse K-cell collapse the
/// per-cell width equals the fine K-tile count (e.g. 4).
inline std::atomic<std::size_t> g_summa_gather_block_count{0};

/// (refine) debug hook: total number of FINE (T) result
/// sub-pages MERGED into a coarser U result tile, accumulated across the
/// active result-axis-refine gather/merge in `finalize_active`. This is the
/// ONLY outbound physical cost and fires ONLY when a RESULT axis is refined
/// (several T result cells map to one U tile). It MUST stay 0 on a pure
/// coarsen/identity config (no result axis refined) -- the unit tests assert
/// that as the witness. gop.sum it across ranks for np-correctness.
inline std::atomic<std::size_t> g_summa_result_merge_count{0};
#endif

/// \brief Distributed contraction evaluator implementation

/// \tparam Left The left-hand argument evaluator type
/// \tparam Right The right-hand argument evaluator type
/// \tparam Op The contraction/reduction operation type
/// \tparam Policy The tensor policy class
/// \note The algorithms in this class assume that the arguments have a two-
/// dimensional cyclic distribution, and that the row phase of the left-hand
/// argument and the column phase of the right-hand argument are equal to
/// the number of rows and columns, respectively, in the \c ProcGrid object
/// passed to the constructor.
template <typename Left, typename Right, typename Op, typename Policy>
class Summa
    : public DistEvalImpl<typename Op::result_type, Policy>,
      public std::enable_shared_from_this<Summa<Left, Right, Op, Policy>> {
 public:
  typedef Summa<Left, Right, Op, Policy> Summa_;  ///< This object type
  typedef DistEvalImpl<typename Op::result_type, Policy>
      DistEvalImpl_;  ///< The base class type
  typedef typename DistEvalImpl_::TensorImpl_
      TensorImpl_;           ///< The base, base class type
  typedef Left left_type;    ///< The left-hand argument type
  typedef Right right_type;  ///< The right-hand argument type
  typedef typename DistEvalImpl_::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename DistEvalImpl_::range_type range_type;      ///< Range type
  typedef typename DistEvalImpl_::shape_type shape_type;      ///< Shape type
  typedef typename DistEvalImpl_::pmap_interface
      pmap_interface;  ///< Process map interface type
  typedef
      typename DistEvalImpl_::trange_type trange_type;    ///< Tiled range type
  typedef typename DistEvalImpl_::value_type value_type;  ///< Tile type
  typedef
      typename DistEvalImpl_::eval_type eval_type;  ///< Tile evaluation type
  typedef Op op_type;  ///< Tile evaluation operator type

 private:
  static ordinal_type max_memory_;  ///< Maximum memory used per node
  static ordinal_type
      max_depth_;  ///< Maximum number of concurrent SUMMA iterations

  // Arguments and operation
  left_type left_;    ///< The left-hand argument
  right_type right_;  /// < The right-hand argument
  op_type op_;  /// < The operation used to evaluate tile-tile contractions

  // Broadcast groups for dense arguments (empty for non-dense arguments)
  madness::Group row_group_;  ///< The row process group for this rank
  madness::Group col_group_;  ///< The column process group for this rank

  // Dimension information
  const ordinal_type k_;      ///< Number of tiles in the inner dimension
                              ///< (COARSE/T count when plan_.active, else U)
  const ordinal_type
      k_fine_;  ///< Number of FINE (U) inner-dimension tiles. Equals k_ on the
                ///< inactive path; on the active inbound-coarsen path the
                ///< operands are stored at the fine count while SUMMA steps
                ///< over the coarse count k_, so get_col/get_row use this to
                ///< index and gather the fine U K-block per coarse cell.
  const ProcGrid proc_grid_;  ///< Process grid for this contraction

  // Batched (fused/Hadamard-index) dimension information. A batched
  // contraction C(h,i,j) = sum_k A(h,i,k) B(h,k,j) is evaluated as nh_
  // independent SUMMA "slabs" sharing one process grid and one task graph:
  // the iteration space is steps s = h*k_ + k, and every tile ordinal is
  // offset by its slab base (h * {left,right,result}_slab_size_). For the
  // ordinary contraction nh_ == 1 and all of this reduces to the unbatched
  // arithmetic.
  const ordinal_type nh_;      ///< Number of fused (Hadamard) slabs
  const ordinal_type nsteps_;  ///< Total SUMMA steps = nh_ * k_
  const ordinal_type
      left_slab_size_;  ///< # of left tiles per slab (= rows*k tiles)
  const ordinal_type
      right_slab_size_;  ///< # of right tiles per slab (= k*cols tiles)
  const ordinal_type
      result_slab_size_;  ///< # of result tiles per slab (= rows*cols tiles)

  // Contraction results
  ReducePairTask<op_type>* reduce_tasks_;  ///< A pointer to the reduction tasks

  // Constants used to iterate over columns and rows of left_ and right_,
  // respectively. N.B. all are *slab-local* (slab h adds
  // h * {left,right}_slab_size_ at the use sites).
  const ordinal_type
      left_start_local_;  ///< The starting point of left column iterator ranges
                          ///< (just add k for specific columns)
  const ordinal_type left_end_;  ///< The end of the left column iterator ranges
                                 ///< within a slab
  const ordinal_type left_stride_;  ///< Stride for left column iterators
  const ordinal_type
      left_stride_local_;            ///< Stride for local left column iterators
  const ordinal_type right_stride_;  ///< Stride for right row iterators
  const ordinal_type
      right_stride_local_;  ///< stride for local right row iterators

  // FINE family. Operand-access / broadcast constants derived from
  // the U=user operand tranges -- the geometry of the tiles SUMMA actually
  // gathers and broadcasts. The members above (left/right *slab_size_,
  // *start_local_, *end_, *stride*) are the COARSE family, derived from the
  // T=target process grid (proc_grid_ / k_ / nh_): the reduce-task / result
  // placement geometry. They coincide today (operand-tile-count ==
  // grid-tile-count) and let them diverge under an active plan_.
  // INVARIANT: when !plan_.active each fine member EQUALS its coarse twin and
  // equals the pre-split single value, byte-for-byte (see ctor init list).
  const ordinal_type
      left_fine_slab_size_;  ///< # of left U-tiles per slab (fine twin of
                             ///< left_slab_size_)
  const ordinal_type
      left_fine_start_local_;       ///< fine twin of left_start_local_
  const ordinal_type left_fine_end_;  ///< fine twin of left_end_
  const ordinal_type
      left_fine_stride_local_;  ///< fine twin of left_stride_local_
  const ordinal_type
      right_fine_slab_size_;  ///< # of right U-tiles per slab (fine twin of
                              ///< right_slab_size_)
  const ordinal_type
      right_fine_stride_local_;  ///< fine twin of right_stride_local_

  // 3-d (h-grouped) grid information. The world's first
  // proc_h_ * proc_h_stride ranks are partitioned into proc_h_ contiguous
  // groups; slab h belongs to group h % proc_h_, and each group runs its
  // own 2-d SUMMA grid (proc_grid_ is this rank's GROUP-LOCAL grid,
  // constructed over the group's rank interval). proc_h_ == 1 is the
  // ordinary shared-grid batched contraction.
  const ordinal_type proc_h_;  ///< Number of slab (h) groups
  const ordinal_type
      proc_h_stride_;              ///< World ranks per slab group (the
                                   ///< group of slab h spans world ranks
                                   ///< [(h % proc_h_) * proc_h_stride_, ...))
  const ordinal_type first_slab_;  ///< This rank's group's first slab (== its
                                   ///< group index), or nh_ if this rank is
                                   ///< in no group (idle for this eval)
  const ordinal_type my_slabs_;    ///< Number of slabs of this rank's group
  const TiledArray::expressions::RetilePlan
      plan_; ///< Two-trange retile plan. Default-constructed
              ///< (inactive) unless the engine threaded a user .retile()
              ///< target. Stored only in this phase; consumed by later phases.
              ///< When inactive the SUMMA behaves exactly as without a retile.

  /// \return the world rank that owns result tile \p i: the within-group
  /// owner (from the group-local process grid) shifted by the world-rank
  /// offset of the group that owns \p i's slab. For proc_h_ == 1 the offset
  /// is 0 and this is the ordinary cyclic owner.
  ProcessID result_tile_owner(const ordinal_type i) const {
    const ordinal_type source_index = DistEvalImpl_::perm_index_to_source(i);
    // owner is independent of slab index *within a group*
    const ordinal_type slab_index = source_index % result_slab_size_;
    const ordinal_type tile_row = slab_index / proc_grid_.cols();
    const ordinal_type tile_col = slab_index % proc_grid_.cols();
    const ordinal_type proc_row = tile_row % proc_grid_.proc_rows();
    const ordinal_type proc_col = tile_col % proc_grid_.proc_cols();
    const ProcessID within_group = proc_row * proc_grid_.proc_cols() + proc_col;
    // shift by the offset of the group that owns this tile's slab
    const ordinal_type slab = source_index / result_slab_size_;
    const ordinal_type group = (proc_h_ > 1ul) ? (slab % proc_h_) : 0ul;
    return ProcessID(group * proc_h_stride_) + within_group;
  }

  /// \return the slab index of SUMMA step \p s
  ordinal_type step_h(const ordinal_type s) const { return s / k_; }
  /// \return the within-slab inner-dimension index of SUMMA step \p s
  ordinal_type step_k(const ordinal_type s) const { return s % k_; }

  /// \return the smallest SUMMA step >= \p s that belongs to one of this
  /// rank's group's slabs, or nsteps_ if there is none
  ordinal_type next_step(ordinal_type s) const {
    if (proc_h_ == 1ul) return std::min(s, nsteps_);
    if (first_slab_ >= nh_) return nsteps_;  // not in any group
    while (s < nsteps_ && (step_h(s) % proc_h_) != first_slab_)
      s = (step_h(s) + 1ul) * k_;  // jump to the start of the next slab
    return std::min(s, nsteps_);
  }

  /// \return this rank's group-local ordinal of slab \p h (which must
  /// belong to this rank's group)
  ordinal_type slab_ord(const ordinal_type h) const {
    return (h - first_slab_) / proc_h_;
  }

  /// \return the number of SUMMA steps of this rank's group's slabs
  ordinal_type my_steps() const { return my_slabs_ * k_; }

  typedef Future<typename right_type::eval_type>
      right_future;  ///< Future to a right-hand argument tile
  typedef Future<typename left_type::eval_type>
      left_future;  ///< Future to a left-hand argument tile
  typedef std::pair<ordinal_type, right_future>
      row_datum;  ///< Datum element type for a right-hand argument row
  typedef std::pair<ordinal_type, left_future>
      col_datum;  ///< Datum element type for a left-hand argument column

  // Active-plan operand-fetch dedupe (review). A lazy-tile operand
  // notifies (set_counter_++) on EVERY get_tile and assumes each U tile is
  // fetched exactly ONCE over the WHOLE contraction. The active refine path
  // gathers the SAME U operand tile from several coarse cells -- WITHIN one
  // SUMMA-K step (several grid rows/cols map to one U tile on a refined result
  // axis) AND ACROSS steps (a refined K axis maps several T K-cells back to one
  // U K-tile, so successive get_col/get_row CALLS re-touch it). A per-call
  // cache fixes only the within-step case; the cross-step case (e.g.
  // refine-K + refine-N) re-arms the operand wait() over-notify deadlock. So
  // the caches are INSTANCE-wide: each distinct U operand ordinal is fetched
  // exactly once for the lifetime of this Summa, and every coarse cell shares
  // the (carved) future. get_col/get_row run as concurrent step tasks, so all
  // access is guarded by active_fetch_mutex_. Used only on the arena-ToT active
  // path (left empty/untouched on the stock path).
  mutable std::mutex active_fetch_mutex_;
  mutable std::map<ordinal_type, left_future> left_fetch_cache_;
  mutable std::map<ordinal_type, right_future> right_fetch_cache_;

  // various tracing/debugging artifacts
  static constexpr const bool trace_tasks =
#ifdef TILEDARRAY_ENABLE_TASK_DEBUG_TRACE
      true
#else
      false
#endif
      ;
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
  mutable std::atomic<ordinal_type>
      left_ntiles_used_;  // # of tiles used from left_
  mutable std::atomic<ordinal_type>
      right_ntiles_used_;  // # of tiles used from right_
  mutable std::atomic<ordinal_type>
      left_ntiles_discarded_;  // # of tiles discarded from left_
  mutable std::atomic<ordinal_type>
      right_ntiles_discarded_;  // # of tiles discarded from right_
#endif

 protected:
  // Import base class functions
  using std::enable_shared_from_this<Summa_>::shared_from_this;

 private:
  // Static variable initialization ----------------------------------------

  /// Initialize max_memory_ limit for SUMMA
  static ordinal_type init_max_memory() {
    const char* max_memory = getenv("TA_SUMMA_MAX_MEMORY");
    if (max_memory) {
      // Convert the string into bytes
      std::stringstream ss(max_memory);
      double memory = 0.0;
      if (ss >> memory) {
        if (memory > 0.0) {
          std::string unit;
          if (ss >> unit) {  // Failure == assume bytes
            if (unit == "KB" || unit == "kB") {
              memory *= 1000.0;
            } else if (unit == "KiB" || unit == "kiB") {
              memory *= 1024.0;
            } else if (unit == "MB") {
              memory *= 1000000.0;
            } else if (unit == "MiB") {
              memory *= 1048576.0;
            } else if (unit == "GB") {
              memory *= 1000000000.0;
            } else if (unit == "GiB") {
              memory *= 1073741824.0;
            }
          }
        }
      }

      memory = std::max(memory, 104857600.0);  // Minimum 100 MiB
      return memory;
    }

    return 0ul;
  }

  static ordinal_type init_max_depth() {
    const char* max_depth = getenv("TA_SUMMA_MAX_DEPTH");
    if (max_depth) return std::stoul(max_depth);
    return 0ul;
  }

  // Process groups --------------------------------------------------------

  /// Process group factory function

  /// This function generates a sparse process group.
  /// \tparam Shape The shape type
  /// \tparam ProcMap The process map operation type
  /// \param shape The shape that will be used to select processes that are
  /// included in the process group
  /// \param process_mask the process mask, if
  ///        \code process_mask[p] == true \endcode,
  ///        process \c p will not be included in the result (p is row/col index
  ///        in this process's row/column)
  /// \param index The first index of the row or column range
  /// \param end The end of the row or column range
  /// \param stride The row or column index stride
  /// \param k The broadcast group index
  /// \param max_proc_h_stride The maximum number of processes in the result
  /// group, which is equal to the number of process in this process row or
  /// column as defined by \c proc_grid_.
  /// \param key_offset The key that will be used to identify the process group
  /// \param proc_map The operator that will convert a process row/column
  /// index into the absolute process index (ProcessID)
  /// \return A sparse process group that includes process in the row or
  /// column of this process as defined by \c proc_grid_.
  template <typename Shape, typename ProcMap>
  madness::Group make_group(const Shape& shape,
                            const std::vector<bool>& process_mask,
                            ordinal_type index, const ordinal_type end,
                            const ordinal_type stride,
                            const ordinal_type max_proc_h_stride,
                            const ordinal_type k, const ordinal_type key_offset,
                            const ProcMap& proc_map) const {
    // Generate the list of processes in rank_row
    std::vector<ProcessID> proc_list(max_proc_h_stride, -1);

    // Flag the root processes of the broadcast, which may not be included
    // by shape.
    ordinal_type p = k % max_proc_h_stride;
    proc_list[p] = proc_map(p);
    ordinal_type count = 1ul;

    // Flag all processes that have non-zero tiles
    for (p = 0ul; (index < end) && (count < max_proc_h_stride);
         index += stride, p = (p + 1u) % max_proc_h_stride) {
      if ((proc_list[p] != -1) || (shape.is_zero(index)) || !process_mask.at(p))
        continue;

      proc_list[p] = proc_map(p);
      ++count;
    }

    // Remove processes from the list that will not be in the group
    for (ordinal_type x = 0ul, p = 0ul; x < count; ++p) {
      if (proc_list[p] == -1) continue;
      proc_list[x++] = proc_list[p];
    }

    // Truncate invalid process id's
    proc_list.resize(count);

    return madness::Group(
        TensorImpl_::world(), proc_list,
        madness::DistributedID(DistEvalImpl_::id(), k + key_offset));
  }

  /// Row process group factory function

  /// \param s The SUMMA step (= slab index * k_ + broadcast group index)
  /// \return A row process group
  madness::Group make_row_group(const ordinal_type s) const {
    const ordinal_type h = step_h(s);
    const ordinal_type k = step_k(s);
    // Construct the sparse broadcast group
    const ordinal_type right_begin_k =
        h * right_slab_size_ + k * proc_grid_.cols();
    const ordinal_type right_end_k = right_begin_k + proc_grid_.cols();
    // make the row mask; using the same mask for all tiles avoids having to
    // compute mask for every tile and use of masked broadcasts
    auto result_row_mask_k = make_row_mask(h, k);

    // return empty group if I am not in this group, otherwise make a group
    // N.B. group key = s + nsteps_ (unique across (h,k) and distinct from the
    // column groups' keys), root flag = k (the within-slab cyclic owner)
    if (result_row_mask_k[proc_grid_.rank_col()])
      return make_group(right_.shape(), result_row_mask_k, right_begin_k,
                        right_end_k, right_stride_, proc_grid_.proc_cols(), k,
                        s - k + nsteps_, [&](const ProcGrid::size_type col) {
                          return proc_grid_.map_col(col);
                        });
    else
      return madness::Group();
  }

  /// Column process group factory function

  /// \param s The SUMMA step (= slab index * k_ + broadcast group index)
  /// \return A column process group
  madness::Group make_col_group(const ordinal_type s) const {
    const ordinal_type h = step_h(s);
    const ordinal_type k = step_k(s);
    // make the column mask; using the same mask for all tiles avoids having to
    // compute mask for every tile and use of masked broadcasts
    auto result_col_mask_k = make_col_mask(h, k);

    // return empty group if I am not in this group, otherwise make a group
    // N.B. group key = s (unique across (h,k)), root flag = k
    if (result_col_mask_k[proc_grid_.rank_row()])
      return make_group(
          left_.shape(), result_col_mask_k, h * left_slab_size_ + k,
          h * left_slab_size_ + left_end_, left_stride_, proc_grid_.proc_rows(),
          k, s - k,
          [&](const ordinal_type row) { return proc_grid_.map_row(row); });
    else
      return madness::Group();
  }

  /// Makes the row result mask

  /// \param k The SUMMA iteration (i.e. contraction tile) index
  /// \return a set object, if \code result[p] == true \endcode the process
  ///         in column \c p of this row has at least 1 result tile for this \c
  ///         k
  std::vector<bool> make_row_mask(const ordinal_type h,
                                  const ordinal_type k) const {
    // "local" A[i][k] (i.e. for all i assigned to my row of processes) will
    // produce C[i][*] for each process in my row of the process grid determine
    // whether there are any nonzero C[i][*] located on that node

    const auto nproc_cols = proc_grid_.proc_cols();
    const auto my_proc_row = proc_grid_.rank_row();

    // result shape
    const auto& result_shape = TensorImpl_::shape();

    // if result is dense, include all processors
    if (result_shape.is_dense()) return std::vector<bool>(nproc_cols, true);

    // initialize the mask
    std::vector<bool> mask(nproc_cols, false);

    // number of tiles in the col dimension of the result
    const auto nj = proc_grid_.cols();
    // number of tiles in contraction dim
    const auto nk = k_;
    // slab bases
    const auto left_base = h * left_slab_size_;
    const auto result_base = h * result_slab_size_;

    // for each i assigned to my row of processes ...
    ordinal_type i_start, i_fence, i_stride;
    std::tie(i_start, i_fence, i_stride) = result_row_range(my_proc_row);
    const auto ik_stride = i_stride * nk;
    for (ordinal_type i = i_start, ik = left_base + i_start * nk + k;
         i < i_fence; i += i_stride, ik += ik_stride) {
      // ... such that A[i][k] exists ...
      if (!left_.shape().is_zero(ik)) {
        // ... the owner of А[i][k] is always in the group ...
        const auto k_proc_col = k % nproc_cols;
        mask[k_proc_col] = true;
        // ... loop over processes in my row ...
        for (ordinal_type proc_col = 0; proc_col != nproc_cols; ++proc_col) {
          // ... that are not the owner of A[i][k] ...
          if (proc_col != k_proc_col) {
            // ... loop over all C[i][j] tiles that belong to this process ...
            ordinal_type j_start, j_fence, j_stride;
            std::tie(j_start, j_fence, j_stride) = result_col_range(proc_col);
            const auto ij_stride = j_stride;
            for (ordinal_type j = j_start, ij = result_base + i * nj + j_start;
                 j < j_fence; j += j_stride, ij += ij_stride) {
              // ... if any such C[i][j] exists, update the mask, and move
              // on to next process
              if (!result_shape.is_zero(
                      DistEvalImpl_::perm_index_to_target(ij))) {
                mask[proc_col] = true;
                break;
              }
            }
          }
        }
      }
    }

    return mask;
  }

  /// Makes the column result mask

  /// \param k The SUMMA iteration (i.e. contraction tile) index
  /// \return a set object, if \code result[p] == true \endcode the process
  ///         in row \c p of this column has at least 1 result tile for this
  ///         \c k
  std::vector<bool> make_col_mask(const ordinal_type h,
                                  const ordinal_type k) const {
    // "local" B[k][j] (i.e. for all j assigned to my column of processes)
    // will produce C[*][j]
    // for each process in my column of the process grid determine whether
    // there are any
    // nonzero C[*][j] located on that node

    const auto nproc_rows = proc_grid_.proc_rows();
    const auto my_proc_col = proc_grid_.rank_col();

    // result shape
    const auto& result_shape = TensorImpl_::shape();

    // if result is dense, include all processors
    if (result_shape.is_dense()) return std::vector<bool>(nproc_rows, true);

    // initialize the mask
    std::vector<bool> mask(nproc_rows, false);

    // number of tiles in col dim of the result
    const auto nj = proc_grid_.cols();
    // slab bases
    const auto right_base = h * right_slab_size_;
    const auto result_base = h * result_slab_size_;

    // for each j assigned to my column of processes ...
    ordinal_type j_start, j_fence, j_stride;
    std::tie(j_start, j_fence, j_stride) = result_col_range(my_proc_col);
    const auto kj_stride = j_stride;
    for (ordinal_type j = j_start, kj = right_base + k * nj + j_start;
         j < j_fence; j += j_stride, kj += kj_stride) {
      // ... such that B[k][j] exists ...
      if (!right_.shape().is_zero(kj)) {
        // ... the owner of B[k][j] is always in the group ...
        auto k_proc_row = k % nproc_rows;
        mask[k_proc_row] = true;
        // ... loop over processes in my col ...
        for (ordinal_type proc_row = 0; proc_row != nproc_rows; ++proc_row) {
          // ... that are not the owner of B[k][j] ...
          if (proc_row != k_proc_row) {
            // ... loop over all C[i][j] tiles that belong to this process
            ordinal_type i_start, i_fence, i_stride;
            std::tie(i_start, i_fence, i_stride) = result_row_range(proc_row);
            const auto ij_stride = i_stride * nj;
            for (ordinal_type i = i_start, ij = result_base + i_start * nj + j;
                 i < i_fence; i += i_stride, ij += ij_stride) {
              // ... if any such C[i][j] exists, update the mask, and move
              // on to next process
              if (!result_shape.is_zero(
                      DistEvalImpl_::perm_index_to_target(ij))) {
                mask[proc_row] = true;
                break;
              }
            }
          }
        }
      }
    }

    return mask;
  }

  /// computes the result row iteration range for a particular processor

  /// \param proc_row the process row in \c this->proc_grid_
  /// \return the {start,fence,stride} tuple which defines the iteration
  ///         range for the row indices of the result tiles residing on
  ///         process in row \c proc_row
  inline std::tuple<ordinal_type, ordinal_type, ordinal_type> result_row_range(
      ordinal_type proc_row) const {
    const ordinal_type start = proc_row;
    const ordinal_type fence = proc_grid_.rows();
    const ordinal_type stride = proc_grid_.proc_rows();
    return std::make_tuple(start, fence, stride);
  }

  /// computes the result column iteration range for a particular processor

  /// \param proc_col the process column in \c this->proc_grid_
  /// \return the {start,fence,stride} tuple which defines the iteration
  ///         range for the column indices of the result tiles residing on
  ///         process in column \c proc_col
  std::tuple<ordinal_type, ordinal_type, ordinal_type> result_col_range(
      ordinal_type proc_col) const {
    const ordinal_type start = proc_col;
    const ordinal_type fence = proc_grid_.cols();
    const ordinal_type stride = proc_grid_.proc_cols();
    return std::make_tuple(start, fence, stride);
  }

  // Broadcast kernels -----------------------------------------------------

  /// Tile conversion task function

  /// \tparam Tile The input tile type
  /// \param tile The input tile
  /// \return The evaluated version of the lazy tile
  template <typename Tile>
  static auto convert_tile(const Tile& tile) {
    TiledArray::Cast<typename eval_trait<Tile>::type, Tile> cast;
    return cast(tile);
  }

  /// Conversion function

  /// This function does nothing since tile is not a lazy tile.
  /// \tparam Arg The type of the argument that holds the input tiles
  /// \param arg The argument that holds the tiles
  /// \param index The tile index of arg
  /// \return \c tile
  template <typename Arg>
  static typename std::enable_if<!is_lazy_tile<typename Arg::value_type>::value,
                                 Future<typename Arg::eval_type>>::type
  get_tile(Arg& arg, const typename Arg::ordinal_type index) {
    return arg.get(index);
  }

  /// Conversion function

  /// This function spawns a task that will convert a lazy tile from the
  /// tile type to the evaluated tile type.
  /// \tparam Arg The type of the argument that holds the input tiles
  /// \param arg The argument that holds the tiles
  /// \param index The tile index of arg
  /// \return A future to the evaluated tile
  template <typename Arg>
  static typename std::enable_if<
      is_lazy_tile<typename Arg::value_type>::value
#ifdef TILEDARRAY_HAS_DEVICE
          && !detail::is_device_tile_v<typename Arg::value_type>
#endif
      ,
      Future<typename Arg::eval_type>>::type
  get_tile(Arg& arg, const typename Arg::ordinal_type index) {
    auto convert_tile_fn =
        &Summa_::template convert_tile<typename Arg::value_type>;
    return arg.world().taskq.add(convert_tile_fn, arg.get(index),
                                 madness::TaskAttributes::hipri());
  }

#ifdef TILEDARRAY_HAS_DEVICE
  /// Conversion function

  /// This function spawns a task that will convert a lazy tile from the
  /// tile type to the evaluated tile type.
  /// \tparam Arg The type of the argument that holds the input tiles
  /// \param arg The argument that holds the tiles
  /// \param index The tile index of arg
  /// \return A future to the evaluated tile
  template <typename Arg>
  static typename std::enable_if<
      is_lazy_tile<typename Arg::value_type>::value &&
          detail::is_device_tile_v<typename Arg::value_type>,
      Future<typename Arg::eval_type>>::type
  get_tile(Arg& arg, const typename Arg::ordinal_type index) {
    auto convert_tile_fn =
        &Summa_::template convert_tile<typename Arg::value_type>;
    return madness::add_device_task(arg.world(), convert_tile_fn,
                                    arg.get(index),
                                    madness::TaskAttributes::hipri());
  }
#endif

  /// Collect non-zero tiles from \c arg

  /// \tparam Arg The argument type
  /// \tparam Datum The vector datum type
  /// \param[in] arg The owner of the input tiles
  /// \param[in] index The index of the first tile to be broadcast
  /// \param[in] end The end of the range of tiles to be broadcast
  /// \param[in] stride The stride between tile indices to be broadcast
  /// \param[out] vec The vector that will hold broadcast tiles
  template <typename Arg, typename Datum>
  void get_vector(Arg& arg, ordinal_type index, const ordinal_type end,
                  const ordinal_type stride, std::vector<Datum>& vec) const {
    TA_ASSERT(vec.size() == 0ul);

    // Iterate over vector of tiles
    if (arg.is_local(index)) {
      for (ordinal_type i = 0ul; index < end; ++i, index += stride) {
        if (arg.shape().is_zero(index)) continue;
        vec.emplace_back(i, get_tile(arg, index));
      }
    } else {
      for (ordinal_type i = 0ul; index < end; ++i, index += stride) {
        if (arg.shape().is_zero(index)) continue;
        vec.emplace_back(i, Future<typename Arg::eval_type>());
      }
    }

    TA_ASSERT(vec.size() > 0ul);
  }

  // -- inbound K-coarsen (active plan) support ---------------------

  /// True iff the evaluated argument tile type is an arena-backed ToT outer
  /// tile (the only tile family the single-page pack `arena_gather_block`
  /// supports). The inbound-coarsen gather is gated on this AND `plan_.active`.
  /// Uses is_tensor_helper (true for any `TA::Tensor<...>`) rather than the
  /// exclusive is_tensor_v (which excludes tensor-of-tensor outers), then
  /// requires the inner cell type to be an `ArenaTensor`.
  template <typename EvalTile>
  static constexpr bool is_arena_tot_v =
      TiledArray::detail::is_tensor_helper<EvalTile>::value &&
      TiledArray::is_arena_tensor_v<typename EvalTile::value_type>;

  // -- result-axis (SUMMA-M/N) coarsen support ---------------------
  //
  // When a SUMMA-M (resp. -N) role axis coarsens, the contraction grid built by
  // the engine is COARSE on that axis (proc_grid_.local_rows()/local_cols()
  // iterate coarse cells), while operands are stored and the result delivered
  // at U. A coarse grid row `mc` covers a contiguous BLOCK of U M-tiles
  // (the Cartesian product of plan_.summaM[a].groups[...] over the M axes); the
  // left operand's U M-block (composed with the U K-block) is gathered+packed
  // into ONE single-page coarse tile so a single tile-GEMM rides the
  // coarse M external (ce_ce ride_on_M) and produces the coarse-M result page,
  // which finalize then carves into the U M-sub-tiles. np=1 only.

  /// Decompose a coarse role-axis grid index \p coarse_idx (row-major over the
  /// role's AxisNest list) into the per-U-axis half-open U-tile ranges
  /// [first,last) it covers, in the role's axis order. Identity/Coarsen yield
  /// the group's range (>= 1 U tile); an empty role yields a single trivial
  /// axis [coarse_idx, coarse_idx+1).
  std::vector<std::pair<ordinal_type, ordinal_type>> coarse_axis_u_ranges(
      const std::vector<TiledArray::expressions::AxisNest>& role,
      ordinal_type coarse_idx) const {
    if (role.empty()) return {{coarse_idx, coarse_idx + 1}};
    const std::size_t nax = role.size();
    // Decompose coarse_idx row-major over the role's per-axis T-tile counts
    // (== groups.size()).
    std::vector<ordinal_type> t_idx(nax);
    ordinal_type rem = coarse_idx;
    for (std::size_t a = nax; a-- > 0;) {
      const ordinal_type ext =
          static_cast<ordinal_type>(role[a].groups.size());
      t_idx[a] = ext ? (rem % ext) : 0;
      rem = ext ? (rem / ext) : rem;
    }
    std::vector<std::pair<ordinal_type, ordinal_type>> out(nax);
    for (std::size_t a = 0; a < nax; ++a) {
      const auto& g = role[a].groups[t_idx[a]];
      out[a] = {static_cast<ordinal_type>(g.first),
                static_cast<ordinal_type>(g.second)};
    }
    return out;
  }

  /// Pack one coarse K-block: a `madness::TaskInterface` that depends
  /// on the variable-count set of FINE (U) operand futures, and on run() packs
  /// them into ONE single-page coarse tile via `arena_gather_block`, setting the
  /// result future. A bespoke task is used (rather than the variadic
  /// `taskq.add`) because madness's `Future<std::vector<Future<T>>>` dependency
  /// holder is non-copyable/non-movable and so cannot be a `TaskFn` argument.
  /// The fine cells share outer (M/N) bounds and partition the K axis, so the
  /// merged outer range is the elementwise min-lobound / max-upbound box.
  template <typename EvalTile>
  class PackBlockTask : public madness::TaskInterface {
   private:
    std::vector<Future<EvalTile>> fine_;  ///< the fine K-block operand futures
    Future<EvalTile> result_;             ///< the packed coarse tile

   public:
    PackBlockTask(std::vector<Future<EvalTile>> fine)
        : madness::TaskInterface(0, "PackBlockTask",
                                 madness::TaskAttributes::hipri()),
          fine_(std::move(fine)),
          result_() {
      // Register each not-yet-ready fine future as a dependency.
      for (auto& f : fine_) {
        if (!f.probe()) {
          madness::DependencyInterface::inc();
          f.register_callback(this);
        }
      }
    }

    const Future<EvalTile>& result() const { return result_; }

    void run(const madness::TaskThreadEnv&) override {
      TA_ASSERT(!fine_.empty());
      std::vector<EvalTile> fine;
      fine.reserve(fine_.size());
      for (auto& f : fine_) fine.push_back(f.get());
      const unsigned int rank = fine.front().range().rank();
      std::vector<std::size_t> lo(rank), up(rank);
      for (unsigned int d = 0; d < rank; ++d) {
        lo[d] = static_cast<std::size_t>(fine.front().range().lobound_data()[d]);
        up[d] = static_cast<std::size_t>(fine.front().range().upbound_data()[d]);
      }
      std::size_t nbatch = 1ul;
      for (const auto& t : fine) {
        if (t.empty()) continue;
        nbatch = static_cast<std::size_t>(t.nbatch());
        const auto* tl = t.range().lobound_data();
        const auto* tu = t.range().upbound_data();
        for (unsigned int d = 0; d < rank; ++d) {
          lo[d] = std::min<std::size_t>(lo[d], static_cast<std::size_t>(tl[d]));
          up[d] = std::max<std::size_t>(up[d], static_cast<std::size_t>(tu[d]));
        }
      }
      const Range coarse_outer(lo, up);
      result_.set(TiledArray::detail::arena_gather_block<EvalTile>(
          fine, coarse_outer, nbatch));
    }
  };  // class PackBlockTask

  /// Gather + pack the FINE (U) operand tiles in \p fine_futs (the contiguous
  /// K-block for one coarse external cell) into ONE single-page coarse tile,
  /// returning a future to it. Only instantiated for arena ToT tiles;
  /// the caller gates on `is_arena_tot_v` so the non-arena branch is never
  /// reached.
  template <typename EvalTile>
  Future<EvalTile> pack_fine_block(
      std::vector<Future<EvalTile>> fine_futs) const {
    TA_ASSERT(!fine_futs.empty());
#ifdef TA_STRIDED_DGEMM_COUNT
    g_summa_gather_block_count.fetch_add(fine_futs.size(),
                                         std::memory_order_relaxed);
#endif
    auto* task = new PackBlockTask<EvalTile>(std::move(fine_futs));
    Future<EvalTile> result = task->result();
    TensorImpl_::world().taskq.add(task);
    return result;
  }

  /// Carve-and-pack one operand cell whose outer footprint is the explicit T
  /// outer box \p t_box (REFINE direction). The fine U operand tiles in
  /// \p fine_futs each cover/overlap \p t_box on the refined axis but are
  /// LARGER than the T cell there; on run() each is view-split (zero-copy,
  /// `arena_carve_block(view=true)`) to its intersection with \p t_box, the
  /// sub-views are gathered into ONE single-page tile at \p t_box, and the
  /// future is set. Refine is the FREE inbound direction: the carve is a
  /// pointer-only sub-view, no physical merge. np=1 only.
  template <typename EvalTile>
  class CarvePackTask : public madness::TaskInterface {
   private:
    std::vector<Future<EvalTile>> fine_;  ///< the covering U operand futures
    Range t_box_;                         ///< the T cell outer box (element)
    Future<EvalTile> result_;             ///< the carved+packed T tile

   public:
    CarvePackTask(std::vector<Future<EvalTile>> fine, Range t_box)
        : madness::TaskInterface(0, "CarvePackTask",
                                 madness::TaskAttributes::hipri()),
          fine_(std::move(fine)),
          t_box_(std::move(t_box)),
          result_() {
      for (auto& f : fine_) {
        if (!f.probe()) {
          madness::DependencyInterface::inc();
          f.register_callback(this);
        }
      }
    }

    const Future<EvalTile>& result() const { return result_; }

    void run(const madness::TaskThreadEnv&) override {
      TA_ASSERT(!fine_.empty());
      std::size_t nbatch = 1ul;
      // Carve each covering U tile to its intersection with the T box. On the
      // refined axis the U tile is larger; on identity axes the intersection
      // equals the U tile's own footprint. The carved sub-views all lie inside
      // t_box, so arena_gather_block lays them down as ONE single-page tile.
      std::vector<EvalTile> subs;
      subs.reserve(fine_.size());
      for (auto& f : fine_) {
        EvalTile u = f.get();
        if (u.empty()) continue;
        nbatch = static_cast<std::size_t>(u.nbatch());
        const Range inter = intersect_range_(u.range(), t_box_);
        if (inter.volume() == 0ul) continue;
        if (inter == u.range()) {
          subs.push_back(u);
        } else {
          // zero-copy outer sub-view of u restricted to the T box.
          auto v = TiledArray::detail::arena_carve_block<EvalTile>(
              u, std::vector<Range>{inter}, /*view=*/true);
          TA_ASSERT(v.size() == 1ul);
          subs.push_back(std::move(v.front()));
        }
      }
      TA_ASSERT(!subs.empty());
      result_.set(TiledArray::detail::arena_gather_block<EvalTile>(subs, t_box_,
                                                                   nbatch));
    }
  };  // class CarvePackTask

  /// Element-wise intersection of two ranges (same coordinate system): per
  /// dim [max(lo), min(up)). A zero-volume box (empty intersection) is
  /// returned as a same-rank lo==up box.
  static Range intersect_range_(const Range& a, const Range& b) {
    const unsigned int rank = a.rank();
    TA_ASSERT(b.rank() == rank);
    std::vector<std::size_t> lo(rank), up(rank);
    for (unsigned int d = 0; d < rank; ++d) {
      const std::size_t alo = static_cast<std::size_t>(a.lobound_data()[d]);
      const std::size_t aup = static_cast<std::size_t>(a.upbound_data()[d]);
      const std::size_t blo = static_cast<std::size_t>(b.lobound_data()[d]);
      const std::size_t bup = static_cast<std::size_t>(b.upbound_data()[d]);
      lo[d] = std::max(alo, blo);
      up[d] = std::min(aup, bup);
      if (up[d] < lo[d]) up[d] = lo[d];
    }
    return Range(lo, up);
  }

  /// Carve+pack the covering U operand tiles \p fine_futs of one coarse cell
  /// into ONE single-page tile at the explicit T outer box \p t_box (REFINE
  /// inbound, free direction). Mirrors `pack_fine_block` but carves to the T
  /// sub-box first. Only instantiated for arena ToT tiles.
  template <typename EvalTile>
  Future<EvalTile> carve_pack_fine_block(std::vector<Future<EvalTile>> fine_futs,
                                         Range t_box) const {
    TA_ASSERT(!fine_futs.empty());
#ifdef TA_STRIDED_DGEMM_COUNT
    g_summa_gather_block_count.fetch_add(fine_futs.size(),
                                         std::memory_order_relaxed);
#endif
    auto* task =
        new CarvePackTask<EvalTile>(std::move(fine_futs), std::move(t_box));
    Future<EvalTile> result = task->result();
    TensorImpl_::world().taskq.add(task);
    return result;
  }

  /// Carve a coarse result page into its covered U result sub-tiles (
  /// free direction) and place each via `set_tile`. A `madness::TaskInterface`
  /// that depends on the coarse result future \p coarse and, on run(), carves
  /// the coarse outer page into the U sub-tiles whose outer ranges are
  /// `u_ranges` (view=true zero-copy sub-views aliasing the coarse arena slab),
  /// then `set_tile(u_ords[i], sub[i])`. Used only when a SUMMA-M/N result axis
  /// coarsens (one coarse cell covers >= 1 U result tile). np=1 only: the carve
  /// is local; `set_tile` to a local owner is a no-send placement.
  template <typename EvalTile>
  class CarveSetTask : public madness::TaskInterface {
   private:
    Summa_* owner_;
    Future<EvalTile> coarse_;
    std::vector<ordinal_type> u_ords_;
    std::vector<Range> u_ranges_;

   public:
    CarveSetTask(Summa_* owner, Future<EvalTile> coarse,
                 std::vector<ordinal_type> u_ords, std::vector<Range> u_ranges)
        : madness::TaskInterface(0, "CarveSetTask",
                                 madness::TaskAttributes::hipri()),
          owner_(owner),
          coarse_(std::move(coarse)),
          u_ords_(std::move(u_ords)),
          u_ranges_(std::move(u_ranges)) {
      if (!coarse_.probe()) {
        madness::DependencyInterface::inc();
        coarse_.register_callback(this);
      }
    }

    void run(const madness::TaskThreadEnv&) override {
      const EvalTile page = coarse_.get();
      // Single covered tile and the page IS that tile (identity result axes):
      // place directly (no carve), matching the stock 1:1 path's value exactly.
      if (u_ranges_.size() == 1ul && page.range() == u_ranges_[0]) {
        owner_->set_tile(u_ords_[0], page);
        return;
      }
      auto subs = TiledArray::detail::arena_carve_block<EvalTile>(
          page, u_ranges_, /*view=*/true);
      TA_ASSERT(subs.size() == u_ords_.size());
      for (std::size_t i = 0; i < subs.size(); ++i)
        owner_->set_tile(u_ords_[i], subs[i]);
    }
  };  // class CarveSetTask

  /// Schedule a CarveSetTask: carve the coarse result page \p coarse into the
  /// U result sub-tiles \p u_ords (their outer ranges taken from the result U
  /// trange) and place each. Only instantiated for arena ToT result tiles.
  template <typename EvalTile>
  void carve_and_set(Future<EvalTile> coarse,
                     const std::vector<ordinal_type>& u_ords) {
    std::vector<Range> u_ranges;
    u_ranges.reserve(u_ords.size());
    for (ordinal_type u : u_ords)
      u_ranges.push_back(this->trange().tile(u));
    auto* task = new CarveSetTask<EvalTile>(this, std::move(coarse), u_ords,
                                            std::move(u_ranges));
    TensorImpl_::world().taskq.add(task);
  }

  /// Merge several FINE (T) result sub-pages into ONE coarser U result tile
  /// (REFINE on a result axis, the outbound physical cost). A
  /// `madness::TaskInterface` that depends on the set of T-cell result pages
  /// \p pages (each a sub-box of the U tile's outer range \p u_range), and on
  /// run() lays them down as ONE single-page U tile via `arena_gather_block`,
  /// then `set_tile(u_ord, merged)`. np=1 only: the pages are locally owned.
  template <typename EvalTile>
  class MergeSetTask : public madness::TaskInterface {
   private:
    Summa_* owner_;
    std::vector<Future<EvalTile>> pages_;  ///< the T result sub-pages
    ordinal_type u_ord_;                   ///< the target U result ordinal
    Range u_range_;                        ///< the U tile's outer range

   public:
    MergeSetTask(Summa_* owner, std::vector<Future<EvalTile>> pages,
                 ordinal_type u_ord, Range u_range)
        : madness::TaskInterface(0, "MergeSetTask",
                                 madness::TaskAttributes::hipri()),
          owner_(owner),
          pages_(std::move(pages)),
          u_ord_(u_ord),
          u_range_(std::move(u_range)) {
      for (auto& p : pages_) {
        if (!p.probe()) {
          madness::DependencyInterface::inc();
          p.register_callback(this);
        }
      }
    }

    void run(const madness::TaskThreadEnv&) override {
      TA_ASSERT(!pages_.empty());
      std::size_t nbatch = 1ul;
      std::vector<EvalTile> subs;
      subs.reserve(pages_.size());
      for (auto& p : pages_) {
        EvalTile t = p.get();
        if (t.empty()) continue;
        nbatch = static_cast<std::size_t>(t.nbatch());
        subs.push_back(std::move(t));
      }
      TA_ASSERT(!subs.empty());
      owner_->set_tile(u_ord_,
                       TiledArray::detail::arena_gather_block<EvalTile>(
                           subs, u_range_, nbatch));
    }
  };  // class MergeSetTask

  /// Schedule a MergeSetTask: gather the T result sub-pages \p pages (covering
  /// disjoint sub-boxes of U result tile \p u_ord) into ONE U tile and place
  /// it. Fires the result-merge counter. Only for arena ToT result tiles.
  template <typename EvalTile>
  void merge_and_set(std::vector<Future<EvalTile>> pages, ordinal_type u_ord) {
#ifdef TA_STRIDED_DGEMM_COUNT
    g_summa_result_merge_count.fetch_add(pages.size(),
                                         std::memory_order_relaxed);
#endif
    Range u_range = this->trange().tile(u_ord);
    auto* task = new MergeSetTask<EvalTile>(this, std::move(pages), u_ord,
                                            std::move(u_range));
    TensorImpl_::world().taskq.add(task);
  }

  /// Collect non-zero tiles from column \c k of slab \c h of \c left_

  /// \param[in] s The SUMMA step (slab * k_ + column index)
  /// \param[out] col The column vector that will hold the tiles
  void get_col(const ordinal_type s, std::vector<col_datum>& col) const {
    using left_eval = typename left_type::eval_type;
    if constexpr (is_arena_tot_v<left_eval>) {
      if (plan_.active) {
        get_col_coarsen(s, col);
        return;
      }
    }
    const ordinal_type base = step_h(s) * left_slab_size_;
    col.reserve(proc_grid_.local_rows());
    get_vector(left_, base + left_start_local_ + step_k(s), base + left_end_,
               left_stride_local_, col);
  }

  /// True iff role \p role has ANY Refine axis (T finer than U). On a refine
  /// axis the operand/result tile must be carved to the exact T sub-box.
  static bool role_has_refine(
      const std::vector<TiledArray::expressions::AxisNest>& role) {
    for (const auto& ax : role)
      if (ax.dir == TiledArray::expressions::NestDir::Refine) return true;
    return false;
  }

  /// Append, to \p lo / \p up, the per-axis element [lo,up) bounds of the T
  /// outer box for role \p role at coarse role-index \p coarse_idx. On a
  /// Refine axis the bound is the T tile's own element range (from the role's
  /// target trange \p target); on Identity/Coarsen it is the union over the
  /// covered U tiles (taken from the operand's U trange \p u_axes -- the
  /// per-axis U TiledRange1s for this role). The two are equal when no axis
  /// refines, so this reduces to the U block box on the coarsen-only path.
  void append_role_t_box(
      const std::vector<TiledArray::expressions::AxisNest>& role,
      const std::vector<TiledRange1>& target,
      const std::vector<TiledRange1>& u_axes, ordinal_type coarse_idx,
      std::vector<std::size_t>& lo, std::vector<std::size_t>& up) const {
    const auto u_ranges = coarse_axis_u_ranges(role, coarse_idx);
    if (role.empty()) {
      // trivial single axis [coarse_idx, coarse_idx+1) -- but for a real
      // operand axis this branch is unused (roles are non-empty here).
      lo.push_back(static_cast<std::size_t>(u_ranges[0].first));
      up.push_back(static_cast<std::size_t>(u_ranges[0].second));
      return;
    }
    const std::size_t nax = role.size();
    // Decompose coarse_idx row-major over per-axis T-tile counts to recover the
    // per-axis T tile index (== coarse_axis_u_ranges' decomposition).
    std::vector<ordinal_type> t_idx(nax);
    ordinal_type rem = coarse_idx;
    for (std::size_t a = nax; a-- > 0;) {
      const ordinal_type ext =
          static_cast<ordinal_type>(role[a].groups.size());
      t_idx[a] = ext ? (rem % ext) : 0;
      rem = ext ? (rem / ext) : rem;
    }
    for (std::size_t a = 0; a < nax; ++a) {
      if (role[a].dir == TiledArray::expressions::NestDir::Refine) {
        TA_ASSERT(a < target.size());
        const auto t_tile = target[a].tile(static_cast<std::size_t>(t_idx[a]));
        lo.push_back(static_cast<std::size_t>(t_tile.first));
        up.push_back(static_cast<std::size_t>(t_tile.second));
      } else {
        // Union over covered U tiles [first_u, last_u): element span from the
        // operand's U axis trange.
        const auto [first_u, last_u] = u_ranges[a];
        const auto e0 = u_axes[a].tile(static_cast<std::size_t>(first_u));
        const auto e1 = u_axes[a].tile(static_cast<std::size_t>(last_u - 1));
        lo.push_back(static_cast<std::size_t>(e0.first));
        up.push_back(static_cast<std::size_t>(e1.second));
      }
    }
  }

  /// Per-role U-axis TiledRange1 lists (the operand U tranges partitioned into
  /// roles), used by append_role_t_box's union branch. Computed from the
  /// operand tranges + the role-axis counts in the plan.
  std::vector<TiledRange1> u_axes_M() const {
    return slice_u_axes(left_.trange(), 0ul, plan_.summaM.size());
  }
  std::vector<TiledRange1> u_axes_K_left() const {
    return slice_u_axes(left_.trange(), plan_.summaM.size(),
                        plan_.summaK.size());
  }
  std::vector<TiledRange1> u_axes_K_right() const {
    return slice_u_axes(right_.trange(), 0ul, plan_.summaK.size());
  }
  std::vector<TiledRange1> u_axes_N() const {
    return slice_u_axes(right_.trange(), plan_.summaK.size(),
                        plan_.summaN.size());
  }
  static std::vector<TiledRange1> slice_u_axes(const trange_type& tr,
                                               std::size_t off,
                                               std::size_t n) {
    std::vector<TiledRange1> out;
    out.reserve(n);
    for (std::size_t a = 0; a < n; ++a) out.push_back(tr.dim(off + a));
    return out;
  }

  /// Active-plan (inbound-coarsen / -refine) left-column gather. For each local
  /// M-row r of the grid, gather the FINE (U) tiles covering the coarse cell
  /// (M-block r, K-cell step_k(s)) and pack into ONE single-page coarse tile
  /// tagged with the coarse-cell-local M-row r. On a Coarsen/Identity
  /// axis the pack box is the union of the covered U tiles; on a Refine axis
  /// the covering U tile is view-split (free, zero-copy) to the exact T
  /// sub-box. np=1 only.
  void get_col_coarsen(const ordinal_type s,
                       std::vector<col_datum>& col) const {
    using left_eval = typename left_type::eval_type;
    // The left U operand's outer dims are [M-axes..., K-axes...].
    const std::size_t nM = plan_.summaM.size();
    const std::size_t nK = plan_.summaK.size();
    const auto kr = coarse_axis_u_ranges(plan_.summaK, step_k(s));
    const bool refine =
        role_has_refine(plan_.summaM) || role_has_refine(plan_.summaK);
    const auto uM = u_axes_M();
    const auto uK = u_axes_K_left();
    const ordinal_type local_rows = proc_grid_.local_rows();
    col.reserve(local_rows);
    // Dedupe U-tile fetches via the INSTANCE-wide cache (see
    // gather_operand_block): a refined M axis maps several rows to the same U
    // tile within this call, and a refined K axis re-touches a U K-tile across
    // SUMMA-K steps -- both must fetch each distinct U tile exactly once.
    for (ordinal_type r = 0ul; r < local_rows; ++r) {
      const auto mr = coarse_axis_u_ranges(plan_.summaM, r);
      std::vector<Future<left_eval>> fine;
      gather_operand_block<left_eval>(left_, /*outer_lead=*/mr, /*outer_tail=*/kr,
                                      nM, nK, fine, left_fetch_cache_,
                                      active_fetch_mutex_);
      if (fine.empty()) continue;
      if (refine) {
        std::vector<std::size_t> lo, up;
        append_role_t_box(plan_.summaM, plan_.targetM, uM, r, lo, up);
        append_role_t_box(plan_.summaK, plan_.targetK, uK, step_k(s), lo, up);
        col.emplace_back(r, carve_pack_fine_block<left_eval>(
                                std::move(fine), Range(lo, up)));
      } else {
        col.emplace_back(r, pack_fine_block<left_eval>(std::move(fine)));
      }
    }
  }

  /// Collect the U sub-tiles of an operand for one coarse cell. \p outer_lead /
  /// \p outer_tail are the per-axis half-open U-tile ranges for the operand's
  /// leading (external) outer axes and trailing (contracted/external) outer
  /// axes; \p n_lead / \p n_tail are their axis counts (the operand's outer rank
  /// is n_lead + n_tail). Enumerates the Cartesian product in operand outer
  /// row-major order, computes each U tile ordinal from the operand's U trange,
  /// and appends non-zero tiles' futures to \p fine.
  ///
  /// \p fetch_cache memoizes `get_tile(arg, ord)` by U-tile ordinal for the
  /// whole Summa instance (NOT just this get_col/get_row call). This is
  /// LOAD-BEARING: a lazy-tile operand (`ArrayEvalImpl`) calls `notify()`
  /// (set_counter_++) on EVERY `get_tile`, and its `task_count_` assumes each U
  /// tile is fetched exactly ONCE (the stock SUMMA contract). On a REFINE axis
  /// the SAME U operand tile is gathered from several coarse cells -- within one
  /// step (several grid rows/cols on a refined result axis) AND across steps (a
  /// refined K axis maps several T K-cells back to one U K-tile). Fetching it
  /// more than once over-increments set_counter_ past task_count_ and deadlocks
  /// the operand's wait(). The instance-wide cache makes each distinct U tile
  /// fetched once; all cells (across all steps) share the (carved) future.
  /// (Coarsen/Identity never share a U tile, so the cache is a no-op there.)
  /// \p cache_mutex guards \p fetch_cache because get_col/get_row run as
  /// concurrent step tasks.
  template <typename EvalTile, typename Arg>
  void gather_operand_block(
      Arg& arg,
      const std::vector<std::pair<ordinal_type, ordinal_type>>& outer_lead,
      const std::vector<std::pair<ordinal_type, ordinal_type>>& outer_tail,
      std::size_t n_lead, std::size_t n_tail,
      std::vector<Future<EvalTile>>& fine,
      std::map<ordinal_type, Future<EvalTile>>& fetch_cache,
      std::mutex& cache_mutex) const {
    const auto& tr = arg.trange();
    const std::size_t rank = n_lead + n_tail;
    TA_ASSERT(tr.tiles_range().rank() == rank);
    std::vector<ordinal_type> lo(rank), hi(rank);
    for (std::size_t a = 0; a < n_lead; ++a) {
      lo[a] = outer_lead[a].first;
      hi[a] = outer_lead[a].second;
    }
    for (std::size_t b = 0; b < n_tail; ++b) {
      lo[n_lead + b] = outer_tail[b].first;
      hi[n_lead + b] = outer_tail[b].second;
    }
    // Row-major Cartesian product over [lo, hi) per axis.
    std::vector<ordinal_type> idx = lo;
    const auto& tiles = tr.tiles_range();
    bool done = false;
    while (!done) {
      const ordinal_type ord = static_cast<ordinal_type>(tiles.ordinal(idx));
      if (!arg.shape().is_zero(ord)) {
        // Find-or-fetch atomically: two threads missing the same ordinal must
        // not both call get_tile (that double-notifies and re-arms the hang).
        // get_tile returns a Future immediately (the tile computes async), so
        // holding the lock across it is cheap.
        Future<EvalTile> fut;
        {
          std::lock_guard<std::mutex> guard(cache_mutex);
          auto it = fetch_cache.find(ord);
          if (it == fetch_cache.end())
            it = fetch_cache.emplace(ord, get_tile(arg, ord)).first;
          fut = it->second;
        }
        fine.emplace_back(std::move(fut));
      }
      // increment idx (row-major, last axis fastest)
      std::size_t a = rank;
      while (a-- > 0) {
        if (++idx[a] < hi[a]) break;
        idx[a] = lo[a];
        if (a == 0) done = true;
      }
      if (rank == 0) done = true;
    }
  }


  /// Collect non-zero tiles from row \c k of slab \c h of \c right_

  /// \param[in] s The SUMMA step (slab * k_ + row index)
  /// \param[out] row The row vector that will hold the tiles
  void get_row(const ordinal_type s, std::vector<row_datum>& row) const {
    using right_eval = typename right_type::eval_type;
    if constexpr (is_arena_tot_v<right_eval>) {
      if (plan_.active) {
        get_row_coarsen(s, row);
        return;
      }
    }
    row.reserve(proc_grid_.local_cols());

    // Compute local iteration limits for row k of slab h of right_.
    ordinal_type begin =
        step_h(s) * right_slab_size_ + step_k(s) * proc_grid_.cols();
    const ordinal_type end = begin + proc_grid_.cols();
    begin += proc_grid_.rank_col();

    get_vector(right_, begin, end, right_stride_local_, row);
  }

  /// Active-plan (inbound-coarsen / -refine) right-row gather. Mirror of
  /// `get_col_coarsen`: for each local N-col c, gather the FINE (U) tiles
  /// covering the coarse cell (K-cell step_k(s), N-block c) and pack into ONE
  /// single-page coarse tile, tagged with the coarse-cell-local N-col
  /// c. On a Coarsen/Identity axis the pack box is the union of the covered U
  /// tiles; on a Refine axis the covering U tile is view-split to the exact T
  /// sub-box. np=1 only.
  void get_row_coarsen(const ordinal_type s,
                       std::vector<row_datum>& row) const {
    using right_eval = typename right_type::eval_type;
    // The canonical SUMMA right operand is laid out [K-axes..., N-axes...]
    // (contracted index leading, external trailing) -- confirmed by stock
    // get_row, which indexes it as kf*cols + n (K-major). So the gather must
    // lead with the U K sub-tiles (coarse K cell step_k(s)) and trail with the
    // U N sub-tiles (plan_.summaN groups for coarse N-col c), packed into ONE
    // single-page coarse tile tagged with `c`. np=1 only.
    const std::size_t nN = plan_.summaN.size();
    const std::size_t nK = plan_.summaK.size();
    const auto kr = coarse_axis_u_ranges(plan_.summaK, step_k(s));
    const bool refine =
        role_has_refine(plan_.summaK) || role_has_refine(plan_.summaN);
    const auto uK = u_axes_K_right();
    const auto uN = u_axes_N();
    const ordinal_type local_cols = proc_grid_.local_cols();
    row.reserve(local_cols);
    // Dedupe U-tile fetches via the INSTANCE-wide cache (see
    // gather_operand_block): a refined N axis maps several cols to the same U
    // tile within this call, and a refined K axis re-touches a U K-tile across
    // SUMMA-K steps -- fetch each distinct U tile once, else the lazy operand's
    // set_counter_ overshoots task_count_ and wait() hangs.
    for (ordinal_type c = 0ul; c < local_cols; ++c) {
      const auto nr = coarse_axis_u_ranges(plan_.summaN, c);
      std::vector<Future<right_eval>> fine;
      gather_operand_block<right_eval>(right_, /*outer_lead=*/kr,
                                       /*outer_tail=*/nr, nK, nN, fine,
                                       right_fetch_cache_, active_fetch_mutex_);
      if (fine.empty()) continue;
      if (refine) {
        std::vector<std::size_t> lo, up;
        append_role_t_box(plan_.summaK, plan_.targetK, uK, step_k(s), lo, up);
        append_role_t_box(plan_.summaN, plan_.targetN, uN, c, lo, up);
        row.emplace_back(c, carve_pack_fine_block<right_eval>(
                                std::move(fine), Range(lo, up)));
      } else {
        row.emplace_back(c, pack_fine_block<right_eval>(std::move(fine)));
      }
    }
  }

  /// Broadcast tiles from \c arg

  /// \param[in] start The index of the first tile to be broadcast
  /// \param[in] stride The stride between tile indices to be broadcast
  /// \param[in] group The process group where the tiles will be broadcast
  /// \param[in] group_root The root process of the broadcast
  /// \param[in] key_offset The broadcast key offset value
  /// \param[out] vec The vector that will hold broadcast tiles
  template <typename Datum>
  void bcast(const ordinal_type start, const ordinal_type stride,
             const madness::Group& group, const ProcessID group_root,
             const ordinal_type key_offset, std::vector<Datum>& vec) const {
    TA_ASSERT(vec.size() != 0ul);
    TA_ASSERT(group.size() > 0);
    TA_ASSERT(group_root < group.size());

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST
    std::stringstream ss;
    ss << "bcast: rank=" << TensorImpl_::world().rank()
       << " root=" << group.world_rank(group_root) << " groupid=("
       << group.id().first << "," << group.id().second
       << ") keyoffset=" << key_offset << " group={ ";
    for (ProcessID group_proc = 0; group_proc < group.size(); ++group_proc)
      ss << group.world_rank(group_proc) << " ";
    ss << "} tiles={ ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST

    // Iterate over tiles to be broadcast
    for (typename std::vector<Datum>::iterator it = vec.begin();
         it != vec.end(); ++it) {
      const ordinal_type index = it->first * stride + start;

      // Broadcast the tile
      const madness::DistributedID key(DistEvalImpl_::id(), index + key_offset);
      TensorImpl_::world().gop.bcast(key, it->second, group_root, group);

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST
      ss << index << " ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST
    }

    TA_ASSERT(vec.size() > 0ul);

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST
    ss << "}\n";
    printf(ss.str().c_str());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST
  }

  // Broadcast specialization for left and right arguments -----------------

  ProcessID get_row_group_root(const ordinal_type k,
                               const madness::Group& row_group) const {
    ProcessID group_root = k % proc_grid_.proc_cols();
    if (!right_.shape().is_dense() &&
        row_group.size() < static_cast<ProcessID>(proc_grid_.proc_cols())) {
      const ProcessID world_root =
          proc_grid_.rank_row() * proc_grid_.proc_cols() + group_root;
      group_root = row_group.rank(world_root);
    }
    return group_root;
  }

  ProcessID get_col_group_root(const ordinal_type k,
                               const madness::Group& col_group) const {
    ProcessID group_root = k % proc_grid_.proc_rows();
    if (!left_.shape().is_dense() &&
        col_group.size() < static_cast<ProcessID>(proc_grid_.proc_rows())) {
      const ProcessID world_root =
          group_root * proc_grid_.proc_cols() + proc_grid_.rank_col();
      group_root = col_group.rank(world_root);
    }
    return group_root;
  }

  /// Broadcast column \c k of slab \c h of \c left_

  /// \param[in] s The SUMMA step (slab * k_ + column index)
  /// \param[out] col The vector that will hold the results of the broadcast
  void bcast_col(const ordinal_type s, std::vector<col_datum>& col,
                 const madness::Group& row_group) const {
    // broadcast if I'm part of the broadcast group
    if (!row_group.empty()) {
      // Broadcast column k of slab h of left_.
      ProcessID group_root = get_row_group_root(step_k(s), row_group);
      bcast(step_h(s) * left_slab_size_ + left_start_local_ + step_k(s),
            left_stride_local_, row_group, group_root, 0ul, col);
    }
  }

  /// Broadcast row \c k of slab \c h of \c right_

  /// \param[in] s The SUMMA step (slab * k_ + row index)
  /// \param[out] row The vector that will hold the results of the broadcast
  void bcast_row(const ordinal_type s, std::vector<row_datum>& row,
                 const madness::Group& col_group) const {
    // broadcast if I'm part of the broadcast group
    if (!col_group.empty()) {
      // Compute the group root process.
      ProcessID group_root = get_col_group_root(step_k(s), col_group);

      // Broadcast row k of slab h of right_.
      bcast(step_h(s) * right_slab_size_ + step_k(s) * proc_grid_.cols() +
                proc_grid_.rank_col(),
            right_stride_local_, col_group, group_root, left_.size(), row);
    }
  }

  void bcast_col_range_task(ordinal_type s, const ordinal_type end) const {
    // Iterate over the skipped steps for which this process column owns the
    // broadcast root (i.e. within-slab k congruent to rank_col mod Pcols)
    const ordinal_type Pcols = proc_grid_.proc_cols();

    for (s = next_step(s); s < end; s = next_step(s + 1ul)) {
      const ordinal_type k = step_k(s);
      if (k % Pcols != static_cast<ordinal_type>(proc_grid_.rank_col()))
        continue;
      const ordinal_type left_base = step_h(s) * left_slab_size_;

      // Compute local iteration limits for column k of slab h of left_.
      ordinal_type index = left_base + left_start_local_ + k;
      const ordinal_type col_end = left_base + left_end_;

      // will create broadcast group only if needed
      bool have_group = false;
      madness::Group row_group;
      ProcessID group_root;
      bool do_broadcast;

      // Search column k of left for non-zero tiles
      for (; index < col_end; index += left_stride_local_) {
        if (left_.shape().is_zero(index)) continue;

        // Construct broadcast group, if needed
        if (!have_group) {
          have_group = true;
          row_group = make_row_group(s);
          // broadcast if I am in this group and this group has others
          do_broadcast = !row_group.empty() && row_group.size() > 1;
          if (do_broadcast) group_root = get_row_group_root(k, row_group);
        }

        if (do_broadcast) {
          // Broadcast the tile
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
          ++left_ntiles_used_;
#endif
          const madness::DistributedID key(DistEvalImpl_::id(), index);
          auto tile = get_tile(left_, index);
          TensorImpl_::world().gop.bcast(key, tile, group_root, row_group);
        } else {
          // Discard the tile
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
          ++left_ntiles_discarded_;
#endif
          left_.discard(index);
        }
      }
    }
  }

  void bcast_row_range_task(ordinal_type s, const ordinal_type end) const {
    // Iterate over the skipped steps for which this process row owns the
    // broadcast root (i.e. within-slab k congruent to rank_row mod Prows)
    const ordinal_type Prows = proc_grid_.proc_rows();

    for (s = next_step(s); s < end; s = next_step(s + 1ul)) {
      const ordinal_type k = step_k(s);
      if (k % Prows != static_cast<ordinal_type>(proc_grid_.rank_row()))
        continue;

      // Compute local iteration limits for row k of slab h of right_.
      ordinal_type index = step_h(s) * right_slab_size_ + k * proc_grid_.cols();
      const ordinal_type row_end = index + proc_grid_.cols();
      index += proc_grid_.rank_col();

      // will create broadcast group only if needed
      bool have_group = false;
      madness::Group col_group;
      ProcessID group_root;
      bool do_broadcast;

      // Search for and broadcast non-zero row
      for (; index < row_end; index += right_stride_local_) {
        if (right_.shape().is_zero(index)) continue;

        // Construct broadcast group
        if (!have_group) {
          have_group = true;
          col_group = make_col_group(s);
          // broadcast if I am in this group and this group has others
          do_broadcast = !col_group.empty() && col_group.size() > 1;
          if (do_broadcast) group_root = get_col_group_root(k, col_group);
        }

        if (do_broadcast) {
          // Broadcast the tile
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
          ++right_ntiles_used_;
#endif
          const madness::DistributedID key(DistEvalImpl_::id(),
                                           index + left_.size());
          auto tile = get_tile(right_, index);
          TensorImpl_::world().gop.bcast(key, tile, group_root, col_group);
        } else {
          // Discard the tile
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
          ++right_ntiles_discarded_;
#endif
          right_.discard(index);
        }
      }
    }
  }

  // Row and column iteration functions ------------------------------------

  /// Find next non-zero row of \c right_ for a sparse shape

  /// Starting at SUMMA step \c s, find the next step whose right-hand row
  /// contains at least one non-zero tile. This search only checks for
  /// non-zero tiles in this processes column.
  /// \param s The first step to search
  /// \return The first step, greater than or equal to \c s with non-zero
  /// tiles, or \c nsteps_ if none is found.
  ordinal_type iterate_row(ordinal_type s) const {
    // Iterate over this rank's group's steps until a non-zero tile is found
    // or the end of the matrix is reached.
    for (s = next_step(s); s < nsteps_; s = next_step(s + 1ul)) {
      // Search for non-zero tiles in row k of slab h of right
      ordinal_type i =
          step_h(s) * right_slab_size_ + step_k(s) * proc_grid_.cols();
      const ordinal_type end = i + proc_grid_.cols();
      i += proc_grid_.rank_col();
      for (; i < end; i += right_stride_local_)
        if (!right_.shape().is_zero(i)) return s;
    }

    return s;
  }

  /// Find the next non-zero column of \c left_ for an arbitrary shape type

  /// Starting at SUMMA step \c s, find the next step whose left-hand column
  /// contains at least one non-zero tile. This search only
  /// checks for non-zero tiles in this process's row.
  /// \param s The first step to test for non-zero tiles
  /// \return The first step, greater than or equal to \c s, that contains
  /// a non-zero tile. If no non-zero tile is not found, return \c nsteps_.
  ordinal_type iterate_col(ordinal_type s) const {
    // Iterate over this rank's group's steps until a non-zero tile is found
    // or the end of the matrix is reached.
    for (s = next_step(s); s < nsteps_; s = next_step(s + 1ul)) {
      // Search column k of slab h for non-zero tiles
      const ordinal_type base = step_h(s) * left_slab_size_;
      for (ordinal_type i = base + left_start_local_ + step_k(s);
           i < base + left_end_; i += left_stride_local_)
        if (!left_.shape().is_zero(i)) return s;
    }

    return s;
  }

  /// Find the next k where the left- and right-hand argument have non-zero
  /// tiles

  /// Search for the next k-th column and row of the left- and right-hand
  /// arguments, respectively, that both contain non-zero tiles. This search
  /// only checks for non-zero tiles in this process's row or column. If a
  /// non-zero, local tile is found that does not contribute to local
  /// contractions, the tiles will be immediately broadcast.
  /// \param k The first row/column to check
  /// \return The next k-th column and row of the left- and right-hand
  /// arguments, respectively, that both have non-zero tiles
  ordinal_type iterate_sparse(const ordinal_type s) const {
    // Initial step for k_col and k_row.
    ordinal_type k_col = iterate_col(s);
    ordinal_type k_row = iterate_row(k_col);

    // Search for a row and column that both have non-zero tiles
    while (k_col != k_row) {
      if (k_col < k_row) {
        k_col = iterate_col(k_row);
      } else {
        k_row = iterate_row(k_col);
      }
    }

    if (s < k_row) {
      // Spawn a task to broadcast any local columns of left that were skipped
      TensorImpl_::world().taskq.add(shared_from_this(),
                                     &Summa_::bcast_col_range_task, s, k_row,
                                     madness::TaskAttributes::hipri());

      // Spawn a task to broadcast any local rows of right that were skipped
      TensorImpl_::world().taskq.add(shared_from_this(),
                                     &Summa_::bcast_row_range_task, s, k_col,
                                     madness::TaskAttributes::hipri());
    }

    return k_col;
  }

  /// Find the next k where the left- and right-hand argument have non-zero
  /// tiles

  /// Search for the next k-th column and row of the left- and right-hand
  /// arguments, respectively, that both contain non-zero tiles. This search
  /// only checks for non-zero tiles in this process's row or column. If a
  /// non-zero, local tile is found that does not contribute to local
  /// contractions, the tiles will be immediately broadcast.
  /// \param k The first row/column to check
  /// \return The next k-th column and row of the left- and right-hand
  /// arguments, respectively, that both have non-zero tiles
  ordinal_type iterate(const ordinal_type k) const {
    return (left_.shape().is_dense() && right_.shape().is_dense()
                ? k
                : iterate_sparse(k));
  }

  // Initialization functions ----------------------------------------------

  /// Initialize reduce tasks and construct broadcast groups
  ordinal_type initialize(const DenseShape&) {
    // if contraction is over zero-volume range just initialize tiles to zero
    if (k_ == 0) {
      ordinal_type tile_count = 0;
      const auto& tiles_range = this->trange().tiles_range();
      for (auto&& tile_idx : tiles_range) {
        auto tile_ord = tiles_range.ordinal(tile_idx);
        if (this->is_local(tile_ord)) {
          this->world().taskq.add([this, tile_ord, tile_idx]() {
            this->set_tile(tile_ord,
                           value_type(this->trange().tile(tile_idx),
                                      typename value_type::value_type{}));
          });
          ++tile_count;
        }
      }
      return tile_count;
    } else {
      // Construct static broadcast groups for dense arguments
      // (key space [0, 2*nsteps_) is reserved for the sparse per-step groups,
      // whose keys h*k_ and h*k_+nsteps_ are disjoint across h-groups; the
      // two static keys are offset PAST that range and made group-unique so
      // that two different groups' single-grid static groups never claim the
      // same DistributedID with inconsistent membership)
      const std::size_t static_key_base = 2ul * nsteps_ + 2ul * first_slab_;
      const madness::DistributedID col_did(DistEvalImpl_::id(),
                                           static_key_base);
      col_group_ = proc_grid_.make_col_group(col_did);
      const madness::DistributedID row_did(DistEvalImpl_::id(),
                                           static_key_base + 1ul);
      row_group_ = proc_grid_.make_row_group(row_did);

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
      std::stringstream ss;
      ss << "init: rank=" << TensorImpl_::world().rank() << "\n    col_group_=("
         << col_did.first << ", " << col_did.second << ") { ";
      for (ProcessID gproc = 0ul; gproc < col_group_.size(); ++gproc)
        ss << col_group_.world_rank(gproc) << " ";
      ss << "}\n    row_group_=(" << row_did.first << ", " << row_did.second
         << ") { ";
      for (ProcessID gproc = 0ul; gproc < row_group_.size(); ++gproc)
        ss << row_group_.world_rank(gproc) << " ";
      ss << "}\n";
      printf(ss.str().c_str());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

      // Allocate memory for the reduce pair tasks (one per local result tile
      // per slab of this rank's group).
      std::allocator<ReducePairTask<op_type>> alloc;
      reduce_tasks_ = alloc.allocate(my_slabs_ * proc_grid_.local_size());

      // Iterate over all local tiles
      const ordinal_type n = my_slabs_ * proc_grid_.local_size();
      for (ordinal_type t = 0ul; t < n; ++t) {
        // Initialize the reduction task
        ReducePairTask<op_type>* MADNESS_RESTRICT const reduce_task =
            reduce_tasks_ + t;
        new (reduce_task) ReducePairTask<op_type>(TensorImpl_::world(), op_
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
                                                  ,
                                                  nullptr, t
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
        );
      }

      // The DistEval task_count_ is the number of result tiles that will be
      // SET (one notify per set_tile). On the active two-trange path each U
      // result tile is SET EXACTLY once -- by a coarsen carve (one cell -> many
      // U), a 1:1 placement, or a refine MERGE (many cells -> one U). The
      // expected set count is therefore the number of DISTINCT covered U result
      // tiles per slab. Summing u_result_ordinals sizes per cell would
      // DOUBLE-count under a result-axis refine (several cells share a U), so
      // dedupe distinct U ordinals. Inactive => 1:1 => equals the stock n.
      if constexpr (is_arena_tot_v<value_type>) {
        if (plan_.active) {
          const ordinal_type local_size = proc_grid_.local_size();
          std::vector<unsigned char> seen(
              this->trange().tiles_range().volume() / nh_, 0u);
          ordinal_type u_count = 0ul;
          for (ordinal_type cell = 0ul; cell < local_size; ++cell)
            for (std::size_t u :
                 plan_.u_result_ordinals(static_cast<std::size_t>(cell)))
              if (!seen[u]) {
                seen[u] = 1u;
                ++u_count;
              }
          return my_slabs_ * u_count;
        }
      }

      return n;
    }
  }

  /// Active two-trange initialize (np=1): allocate COARSE-grid-sized reduce
  /// tasks in the SAME slab-major / coarse-cell order finalize_active consumes.
  /// A coarse cell gets a live reduce task iff ANY covered U result tile is
  /// non-zero (zero-skip on the U result shape, not the coarse grid); else an
  /// empty placeholder task -- preserving the allocation FORM and the count
  /// invariant.
  ///
  /// COUNT INVARIANT: `task_count_` is the number of `set_tile` notifications
  /// finalize_active will issue == the number of DISTINCT non-zero U result
  /// tiles covered (each U tile is SET exactly once, whether by a coarsen carve
  /// (one cell -> many U), a 1:1 placement, or a refine MERGE (many cells ->
  /// one U)). Counting per covered (cell,U) pair would DOUBLE-count under a
  /// result-axis refine (several cells map to the same U), so the count
  /// dedupes distinct U ordinals per slab. np=1 + canonical (no permutation)
  /// only.
  template <typename Shape>
  ordinal_type initialize_active(const Shape& shape) {
    if (DistEvalImpl_::perm_index_to_target(0) != 0)
      TA_EXCEPTION(
          "in-SUMMA two-trange retile: result permutation is not yet supported "
          "on the active (coarsen) path");

    const ordinal_type local_size = proc_grid_.local_size();
    const ordinal_type n_alloc = my_slabs_ * local_size;
    std::allocator<ReducePairTask<op_type>> alloc;
    reduce_tasks_ = alloc.allocate(n_alloc);

    ordinal_type tile_count = 0ul;
    ReducePairTask<op_type>* MADNESS_RESTRICT reduce_task = reduce_tasks_;
    for (ordinal_type h = first_slab_; h < nh_; h += proc_h_) {
      const ordinal_type slab_u_base =
          slab_ord(h) * (this->trange().tiles_range().volume() / nh_);
      // Per-slab set of distinct non-zero U ordinals seen so far (each is SET
      // exactly once). A U covered by several cells (refine-result) increments
      // the count only on first sight.
      std::vector<unsigned char> seen(this->trange().tiles_range().volume(),
                                      0u);
      for (ordinal_type cell = 0ul; cell < local_size; ++cell, ++reduce_task) {
        const std::vector<std::size_t> u_ord_local =
            plan_.u_result_ordinals(static_cast<std::size_t>(cell));
        ordinal_type cell_nonzero = 0ul;
        for (std::size_t u : u_ord_local) {
          const ordinal_type u_ord =
              static_cast<ordinal_type>(u) + slab_u_base;
          if (shape.is_zero(u_ord)) continue;
          ++cell_nonzero;
          if (!seen[u_ord]) {
            seen[u_ord] = 1u;
            ++tile_count;  // distinct non-zero U tile == one set_tile
          }
        }
        if (cell_nonzero > 0ul) {
          new (reduce_task) ReducePairTask<op_type>(TensorImpl_::world(), op_
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
                                                    ,
                                                    nullptr, cell
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
          );
        } else {
          new (reduce_task) ReducePairTask<op_type>();
        }
      }
    }
    return tile_count;
  }

  /// Initialize reduce tasks
  template <typename Shape>
  ordinal_type initialize(const Shape& shape) {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
    std::stringstream ss;
    ss << "    initialize rank=" << TensorImpl_::world().rank() << " tiles={ ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

    // fast return if there is no work to do
    if (k_ == 0) return 0;

    // Active two-trange retile (np=1): reduce tasks are COARSE-grid-sized; the
    // zero-skip evaluates the U result shape per COVERED U tile (a coarse cell
    // is non-zero iff ANY covered U tile is non-zero). Same allocation FORM as
    // the stock path (my_slabs_ * local_size) but the per-cell live/empty
    // decision uses u_result_ordinals so finalize_active's consume count
    // matches exactly. Gated so the inactive loop below is byte-for-byte stock.
    if constexpr (is_arena_tot_v<value_type>) {
      if (plan_.active) return initialize_active(shape);
    }

    // Allocate memory for the reduce pair tasks (one per local result tile
    // per slab of this rank's group).
    std::allocator<ReducePairTask<op_type>> alloc;
    reduce_tasks_ = alloc.allocate(my_slabs_ * proc_grid_.local_size());

    // Initialize iteration variables
    const ordinal_type col_stride =  // The stride to iterate down a column
        proc_grid_.proc_rows() * proc_grid_.cols();
    const ordinal_type row_stride =  // The stride to iterate across a row
        proc_grid_.proc_cols();

    // Iterate over all local tiles, slab by slab (the block-cyclic phase
    // restarts at every slab: within a group, the owner of tile (h,i,j)
    // does not depend on h)
    ordinal_type tile_count = 0ul;
    ReducePairTask<op_type>* MADNESS_RESTRICT reduce_task = reduce_tasks_;
    for (ordinal_type h = first_slab_; h < nh_; h += proc_h_) {
      const ordinal_type slab_base = h * result_slab_size_;
      ordinal_type row_start =
          slab_base + proc_grid_.rank_row() * proc_grid_.cols();
      ordinal_type row_end = row_start + proc_grid_.cols();
      row_start += proc_grid_.rank_col();
      const ordinal_type end = slab_base + result_slab_size_;

      // this loops over result tiles arranged in block-cyclic order
      // index = tile index (row major)
      for (; row_start < end; row_start += col_stride, row_end += col_stride) {
        for (ordinal_type index = row_start; index < row_end;
             index += row_stride, ++reduce_task) {
          // Initialize the reduction task

          // Skip zero tiles
          if (!shape.is_zero(DistEvalImpl_::perm_index_to_target(index))) {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
            ss << index << " ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

            new (reduce_task) ReducePairTask<op_type>(TensorImpl_::world(), op_
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
                                                      ,
                                                      nullptr, index
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
            );
            ++tile_count;
          } else {
            // Construct an empty task to represent zero tiles.
            new (reduce_task) ReducePairTask<op_type>();
          }
        }
      }
    }

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
    ss << "}\n";
    printf(ss.str().c_str());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

    return tile_count;
  }

  ordinal_type initialize() {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
    printf("init: start rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

    const ordinal_type result = initialize(TensorImpl_::shape());

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
    printf("init: finish rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

    return result;
  }

  // Finalize functions ----------------------------------------------------

  /// Set the result tiles, destroy reduce tasks, and destroy broadcast groups
  void finalize(const DenseShape& shape) {
    // Active two-trange retile (np=1): the grid is COARSE; each reduce-task cell
    // covers >= 1 U result tile, reconciled + carved here. Identity reduces to
    // 1:1. Gated so the inactive path below is byte-for-byte the stock loop.
    if constexpr (is_arena_tot_v<value_type>) {
      if (plan_.active) {
        finalize_active(shape);
        return;
      }
    }
    // Initialize iteration variables
    const ordinal_type col_stride =  // The stride to iterate down a column
        proc_grid_.proc_rows() * proc_grid_.cols();
    const ordinal_type row_stride =  // The stride to iterate across a row
        proc_grid_.proc_cols();

    // Iterate over all local tiles, slab by slab
    ReducePairTask<op_type>* reduce_task = reduce_tasks_;
    for (ordinal_type h = first_slab_; h < nh_; h += proc_h_) {
      const ordinal_type slab_base = h * result_slab_size_;
      ordinal_type row_start =
          slab_base + proc_grid_.rank_row() * proc_grid_.cols();
      ordinal_type row_end = row_start + proc_grid_.cols();
      row_start += proc_grid_.rank_col();
      const ordinal_type end = slab_base + result_slab_size_;

      for (; row_start < end; row_start += col_stride, row_end += col_stride) {
        for (ordinal_type index = row_start; index < row_end;
             index += row_stride, ++reduce_task) {
          // Set the result tile
          DistEvalImpl_::set_tile(DistEvalImpl_::perm_index_to_target(index),
                                  reduce_task->submit());

          // Destroy the reduce task
          reduce_task->~ReducePairTask<op_type>();
        }
      }
    }

    // Deallocate the memory for the reduce pair tasks.
    std::allocator<ReducePairTask<op_type>>().deallocate(
        reduce_tasks_, my_slabs_ * proc_grid_.local_size());
  }

  /// Active two-trange finalize (np=1): walk the COARSE reduce-task grid in
  /// slab-major, row-major order (== the initialize() allocation order), submit
  /// each coarse cell's result page, then reconcile the COARSE (T) result grid
  /// to the U result trange. Three reconciliation shapes, by how T nests U on
  /// the RESULT axes (Hadamard ++ M ++ N):
  ///   - COARSEN / IDENTITY: a coarse cell covers >= 1 U result tile (it is the
  ///     SOLE contributor to each); carve the page into the covered U sub-tiles
  ///     (`carve_and_set`, the free view-split). 1:1 identity reduces to a
  ///     direct placement.
  ///   - REFINE on a result axis: several coarse cells (each producing a T
  ///     result sub-page covering a sub-box of one U tile) map to the SAME U
  ///     tile; gather their pages and merge into the single U tile
  /// (`merge_and_set`, the ONLY outbound physical cost -- fires the
  ///     counter).
  /// A HYBRID cell that BOTH covers multiple U tiles AND shares a U tile with
  /// another cell is not implemented and rejected loudly (TA_EXCEPTION).
  ///
  /// COUNT INVARIANT: every reduce task allocated is consumed (submitted +
  /// destroyed) here, and every distinct non-zero U result tile is SET EXACTLY
  /// once (== the count initialize_active returned). Both checked with
  /// TA_EXCEPTION. np=1 + canonical (no permutation) only.
  template <typename Shape>
  void finalize_active(const Shape& shape) {
    // Reject a permutation on the active path (outer-index permutation
    // composition is deferred; canonical order is assumed at np=1).
    if (DistEvalImpl_::perm_index_to_target(0) != 0)
      TA_EXCEPTION(
          "in-SUMMA two-trange retile: result permutation is not yet supported "
          "on the active (coarsen) path");

    const ordinal_type local_size = proc_grid_.local_size();
    const ordinal_type n_alloc = my_slabs_ * local_size;
    const ordinal_type u_vol = this->trange().tiles_range().volume();
    // Track exact-once placement over the U result trange.
    std::vector<unsigned char> written(u_vol, 0u);
    ordinal_type consumed = 0ul;

    ReducePairTask<op_type>* reduce_task = reduce_tasks_;
    for (ordinal_type h = first_slab_; h < nh_; h += proc_h_) {
      const ordinal_type slab_u_base =
          slab_ord(h) * (u_vol / nh_);

      // Pass 1: per cell, submit the page future and collect its covered,
      // non-zero U ordinals (slab-local). Build, per U ordinal, the list of
      // contributing cell pages and the count of cells covering it.
      struct CellInfo {
        Future<value_type> page;
        std::vector<ordinal_type> u_ords;  // covered non-zero U (this slab)
      };
      std::vector<CellInfo> cells(local_size);
      std::vector<ordinal_type> u_cover_count(u_vol, 0ul);
      std::vector<std::vector<Future<value_type>>> u_contrib(u_vol);

      for (ordinal_type cell = 0ul; cell < local_size; ++cell, ++reduce_task) {
        const std::vector<std::size_t> u_ord_local =
            plan_.u_result_ordinals(static_cast<std::size_t>(cell));
        std::vector<ordinal_type> u_ords;
        u_ords.reserve(u_ord_local.size());
        for (std::size_t u : u_ord_local) {
          const ordinal_type u_ord =
              static_cast<ordinal_type>(u) + slab_u_base;
          if (shape.is_zero(u_ord)) continue;
          u_ords.push_back(u_ord);
        }
        Future<value_type> page = reduce_task->submit();
        reduce_task->~ReducePairTask<op_type>();
        ++consumed;
        for (ordinal_type u_ord : u_ords) {
          ++u_cover_count[u_ord];
          u_contrib[u_ord].push_back(page);
        }
        cells[cell].page = std::move(page);
        cells[cell].u_ords = std::move(u_ords);
      }

      // Pass 2a: place each U tile that is MERGED (covered by >= 2 cells).
      // Each contributing cell must cover ONLY this U tile (pure refine on the
      // result axes); a cell that both spans multiple U and shares a U is a
      // hybrid we do not implement.
      for (ordinal_type u_ord = 0ul; u_ord < u_vol; ++u_ord) {
        if (u_cover_count[u_ord] < 2ul) continue;
        merge_and_set<value_type>(std::move(u_contrib[u_ord]), u_ord);
        written[u_ord] = 1u;
      }

      // Pass 2b: place each CARVE cell -- a cell that is the SOLE contributor
      // to every U tile it covers (coarsen / identity). Reject the hybrid.
      for (ordinal_type cell = 0ul; cell < local_size; ++cell) {
        const auto& info = cells[cell];
        if (info.u_ords.empty()) continue;
        // If ANY of this cell's U tiles is shared, this cell participated in a
        // merge above; it must then cover ONLY that one (shared) U tile.
        bool any_shared = false, all_shared = true;
        for (ordinal_type u_ord : info.u_ords) {
          if (u_cover_count[u_ord] >= 2ul)
            any_shared = true;
          else
            all_shared = false;
        }
        if (any_shared) {
          if (!(all_shared && info.u_ords.size() == 1ul))
            TA_EXCEPTION(
                "in-SUMMA two-trange retile: a coarse result cell both spans "
                "multiple U result tiles and shares a U tile with another cell "
                "(coarsen+refine hybrid on the result axes is not implemented)");
          continue;  // handled by the merge pass
        }
        // Pure carve cell: it is the sole contributor to all its U tiles.
        for (ordinal_type u_ord : info.u_ords) {
          if (written[u_ord])
            TA_EXCEPTION(
                "in-SUMMA two-trange retile: a U result tile would be written "
                "more than once (count-invariant violation)");
          written[u_ord] = 1u;
        }
        carve_and_set<value_type>(info.page, info.u_ords);
      }
    }

    if (consumed != n_alloc)
      TA_EXCEPTION(
          "in-SUMMA two-trange retile: reduce-task consume count != allocation "
          "count (count-invariant violation)");

    std::allocator<ReducePairTask<op_type>>().deallocate(reduce_tasks_, n_alloc);
  }

  /// Set the result tiles and destroy reduce tasks
  template <typename Shape>
  void finalize(const Shape& shape) {
    // Active two-trange retile (np=1): coarse grid -> carve into U result tiles.
    if constexpr (is_arena_tot_v<value_type>) {
      if (plan_.active) {
        finalize_active(shape);
        return;
      }
    }
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
    std::stringstream ss;
    ss << "    finalize rank=" << TensorImpl_::world().rank() << " tiles={ ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE

    // Initialize iteration variables
    const ordinal_type col_stride =  // The stride to iterate down a column
        proc_grid_.proc_rows() * proc_grid_.cols();
    const ordinal_type row_stride =  // The stride to iterate across a row
        proc_grid_.proc_cols();

    // Iterate over all local tiles, slab by slab
    ReducePairTask<op_type>* reduce_task = reduce_tasks_;
    for (ordinal_type h = first_slab_; h < nh_; h += proc_h_) {
      const ordinal_type slab_base = h * result_slab_size_;
      ordinal_type row_start =
          slab_base + proc_grid_.rank_row() * proc_grid_.cols();
      ordinal_type row_end = row_start + proc_grid_.cols();
      row_start += proc_grid_.rank_col();
      const ordinal_type end = slab_base + result_slab_size_;

      for (; row_start < end; row_start += col_stride, row_end += col_stride) {
        for (ordinal_type index = row_start; index < row_end;
             index += row_stride, ++reduce_task) {
          // Compute the permuted index
          const ordinal_type perm_index =
              DistEvalImpl_::perm_index_to_target(index);

          // Skip zero tiles
          if (!shape.is_zero(perm_index)) {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
            ss << index << " ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE

            // Set the result tile
            DistEvalImpl_::set_tile(perm_index, reduce_task->submit());
          }

          // Destroy the reduce task
          reduce_task->~ReducePairTask<op_type>();
        }
      }
    }
    // Deallocate the memory for the reduce pair tasks.
    std::allocator<ReducePairTask<op_type>>().deallocate(
        reduce_tasks_, my_slabs_ * proc_grid_.local_size());

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
    ss << "}\n";
    printf(ss.str().c_str());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
  }

  void finalize() {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
    printf("finalize: start rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE

    finalize(TensorImpl_::shape());

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
    printf("finalize: finish rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
  }

  /// SUMMA finalization task

  /// This task will set the tiles and do cleanup.
  class FinalizeTask : public madness::TaskInterface {
   private:
    std::shared_ptr<Summa_> owner_;  ///< The parent object for this task

   public:
    FinalizeTask(const std::shared_ptr<Summa_>& owner, const int ndep)
        : madness::TaskInterface(ndep, madness::TaskAttributes::hipri()),
          owner_(owner) {}

    virtual ~FinalizeTask() {}

    virtual void run(const madness::TaskThreadEnv&) { owner_->finalize(); }

  };  // class FinalizeTask

  // Contraction functions -------------------------------------------------

  /// Schedule local contraction tasks for \c col and \c row tile pairs

  /// Schedule tile contractions for each tile pair of \c row and \c col. A
  /// callback to \c task will be registered with each tile contraction
  /// task.
  /// \param col A column of tiles from the left-hand argument
  /// \param row A row of tiles from the right-hand argument
  /// \param task The task that depends on tile contraction tasks
  void contract(const DenseShape&, const ordinal_type s,
                const std::vector<col_datum>& col,
                const std::vector<row_datum>& row,
                madness::TaskInterface* const task) {
    // The reduce tasks of this group's slab h occupy
    // [slab_ord(h) * local_size, (slab_ord(h)+1) * local_size)
    const ordinal_type slab_offset =
        slab_ord(step_h(s)) * proc_grid_.local_size();

    // Iterate over the row
    for (ordinal_type i = 0ul; i < col.size(); ++i) {
      // Compute the local, result-tile offset
      const ordinal_type reduce_task_offset =
          slab_offset + col[i].first * proc_grid_.local_cols();

      // Iterate over columns
      for (ordinal_type j = 0ul; j < row.size(); ++j) {
        const ordinal_type reduce_task_index =
            reduce_task_offset + row[j].first;

        // Schedule task for contraction pairs
        if (task) task->inc();
        const left_future left = col[i].second;
        const right_future right = row[j].second;
        reduce_tasks_[reduce_task_index].add(left, right, task);
      }
    }
  }

  /// Schedule local contraction tasks for \c col and \c row tile pairs

  /// Schedule tile contractions for each tile pair of \c row and \c col. A
  /// callback to \c task will be registered with each tile contraction
  /// task.
  /// \param col A column of tiles from the left-hand argument
  /// \param row A row of tiles from the right-hand argument
  /// \param task The task that depends on tile contraction tasks
  template <typename Shape>
  void contract(const Shape&, const ordinal_type s,
                const std::vector<col_datum>& col,
                const std::vector<row_datum>& row,
                madness::TaskInterface* const task) {
    // The reduce tasks of this group's slab h occupy
    // [slab_ord(h) * local_size, (slab_ord(h)+1) * local_size)
    const ordinal_type slab_offset =
        slab_ord(step_h(s)) * proc_grid_.local_size();

    // Iterate over the row
    for (ordinal_type i = 0ul; i < col.size(); ++i) {
      // Compute the local, result-tile offset
      const ordinal_type reduce_task_offset =
          slab_offset + col[i].first * proc_grid_.local_cols();

      // Iterate over columns
      for (ordinal_type j = 0ul; j < row.size(); ++j) {
        const ordinal_type reduce_task_index =
            reduce_task_offset + row[j].first;

        // Skip zero tiles
        if (!reduce_tasks_[reduce_task_index]) continue;

        // Schedule task for contraction pairs
        if (task) {
          if (trace_tasks)
            task->inc_debug("destroy(*ReduceObject)");
          else
            task->inc();
        }
        const left_future left = col[i].second;
        const right_future right = row[j].second;
        reduce_tasks_[reduce_task_index].add(left, right, task);
      }
    }
  }

#define TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER
#ifndef TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER
  /// Schedule local contraction tasks for \c col and \c row tile pairs

  /// Schedule tile contractions for each tile pair of \c row and \c col. A
  /// callback to \c task will be registered with each tile contraction
  /// task. This version of contract is used when shape_type is
  /// \c SparseShape. It skips tile contractions that have a negligible
  /// contribution to the result tile.
  /// \tparam T The shape value type
  /// \param k The k step for this contraction set
  /// \param col A column of tiles from the left-hand argument
  /// \param row A row of tiles from the right-hand argument
  /// \param task The task that depends on the tile contraction tasks
  template <typename T>
  typename std::enable_if<std::is_floating_point<T>::value>::type contract(
      const SparseShape<T>&, const ordinal_type k,
      const std::vector<col_datum>& col, const std::vector<row_datum>& row,
      madness::TaskInterface* const task) {
    // Cache row shape data.
    std::vector<typename SparseShape<T>::value_type> row_shape_values;
    row_shape_values.reserve(row.size());
    const ordinal_type row_start =
        k * proc_grid_.cols() + proc_grid_.rank_col();
    for (ordinal_type j = 0ul; j < row.size(); ++j)
      row_shape_values.push_back(
          right_.shape()[row_start + (row[j].first * right_stride_local_)]);

    const ordinal_type col_start = left_start_local_ + k;
    const float threshold_k = TensorImpl_::shape().threshold() /
                              typename SparseShape<T>::value_type(k_);
    // Iterate over the row
    for (ordinal_type i = 0ul; i != col.size(); ++i) {
      // Compute the local, result-tile offset
      const ordinal_type offset = col[i].first * proc_grid_.local_cols();

      // Get the shape data for col_it tile
      const typename SparseShape<T>::value_type col_shape_value =
          left_.shape()[col_start + (col[i].first * left_stride_local_)];

      // Iterate over columns
      for (ordinal_type j = 0ul; j < row.size(); ++j) {
        if ((col_shape_value * row_shape_values[j]) < threshold_k) continue;

        const ordinal_type reduce_task_index = offset + row[j].first;

        // Skip zero tiles
        if (!reduce_tasks_[reduce_task_index]) continue;

        if (task) task->inc();
        reduce_tasks_[reduce_task_index].add(col[i].second, row[j].second,
                                             task);
      }
    }
  }
#endif  // TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER

  void contract(const ordinal_type k, const std::vector<col_datum>& col,
                const std::vector<row_datum>& row,
                madness::TaskInterface* const task) {
    contract(TensorImpl_::shape(), k, col, row, task);
  }

  // SUMMA step task -------------------------------------------------------

  /// SUMMA step task

  /// This task will perform a single SUMMA iteration, and start the next
  /// step task.
  class StepTask : public madness::TaskInterface {
   protected:
    // Member variables
    std::shared_ptr<Summa_> owner_;  ///< The owner of this task
    World& world_;
    std::vector<col_datum> col_{};
    std::vector<row_datum> row_{};
    FinalizeTask* finalize_task_;         ///< The SUMMA finalization task
    StepTask* next_step_task_ = nullptr;  ///< The next SUMMA step task
    StepTask* tail_step_task_ =
        nullptr;  ///< The last SUMMA step task that currently exists

    void get_col(const ordinal_type k) {
      owner_->get_col(k, col_);
      if (trace_tasks)
        this->notify_debug("StepTask::spawn_col");
      else
        this->notify();
    }

    void get_row(const ordinal_type k) {
      owner_->get_row(k, row_);
      if (trace_tasks)
        this->notify_debug("StepTask::spawn_row");
      else
        this->notify();
    }

   public:
    StepTask(const std::shared_ptr<Summa_>& owner, int finalize_ndep)
        :
#ifdef TILEDARRAY_ENABLE_TASK_DEBUG_TRACE
          madness::TaskInterface(0ul, "StepTask 1st ctor",
                                 madness::TaskAttributes::hipri()),
#else
          madness::TaskInterface(0ul, madness::TaskAttributes::hipri()),
#endif
          owner_(owner),
          world_(owner->world()),
          finalize_task_(new FinalizeTask(owner, finalize_ndep)) {
      TA_ASSERT(owner_);
      owner_->world().taskq.add(finalize_task_);
    }

    /// Construct the task for the next step

    /// \param parent The previous SUMMA step task
    /// \param ndep The number of dependencies for this task
    StepTask(StepTask* const parent, const int ndep)
        :
#ifdef TILEDARRAY_ENABLE_TASK_DEBUG_TRACE
          madness::TaskInterface(ndep, "StepTask nth ctor",
                                 madness::TaskAttributes::hipri()),
#else
          madness::TaskInterface(ndep, madness::TaskAttributes::hipri()),
#endif
          owner_(parent->owner_),
          world_(parent->world_),
          finalize_task_(parent->finalize_task_) {
      TA_ASSERT(parent);
      parent->next_step_task_ = this;
    }

    virtual ~StepTask() {}

    void spawn_get_row_col_tasks(const ordinal_type k) {
      // Submit the task to collect column tiles of left for iteration k
      if (trace_tasks)
        madness::DependencyInterface::inc_debug("StepTask::spawn_col");
      else
        madness::DependencyInterface::inc();
      world_.taskq.add(this, &StepTask::get_col, k,
                       madness::TaskAttributes::hipri());

      // Submit the task to collect row tiles of right for iteration k
      if (trace_tasks)
        madness::DependencyInterface::inc_debug("StepTask::spawn_row");
      else
        madness::DependencyInterface::inc();
      world_.taskq.add(this, &StepTask::get_row, k,
                       madness::TaskAttributes::hipri());
    }

    template <typename Derived>
    void make_next_step_tasks(Derived* task, ordinal_type depth) {
      // Set the depth to be no greater than the number of SUMMA steps this
      // rank's group actually executes. In the 2-d (proc_h_ == 1) case this is
      // nsteps_ (my_slabs_ == nh_); in the 3-d (proc_h_ > 1) case my_steps() <
      // nsteps_, and clamping to nsteps_ would pre-spawn surplus step tasks
      // that all resolve to the terminating step (k_ == nsteps_).
      if (depth > owner_->my_steps()) depth = owner_->my_steps();

      // Spawn n=depth step tasks
      for (; depth > 0ul; --depth) {
        // Set dep count of the tail task to 1, it will not start until this
        // task commands
        Derived* const next = new Derived(task, depth == 1 ? 1 : 0);
        task = next;
      }

      // Keep track of the tail ptr
      tail_step_task_ = task;
    }

    template <typename Derived, typename GroupType>
    void run(const ordinal_type k, const GroupType& row_group,
             const GroupType& col_group) {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_STEP
      printf("step:  start rank=%i k=%lu\n", owner_->world().rank(), k);
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_STEP

      if (k < owner_->nsteps_) {
        // Initialize next tail task and submit next task
        TA_ASSERT(next_step_task_);
        next_step_task_->tail_step_task_ = new Derived(
            static_cast<Derived*>(tail_step_task_),
            1);  // <- ndep=1, will control its scheduling by this task
        // submit next step task ... even if it's same as tail_step_task_ it is
        // safe to submit because its ndep > 0 (see
        // StepTask::make_next_step_tasks)
        TA_ASSERT(tail_step_task_->ndep() > 0);
        world_.taskq.add(next_step_task_);
        next_step_task_ = nullptr;

        // Start broadcast of column and row tiles for this step
        world_.taskq.add(owner_, &Summa_::bcast_col, k, col_, row_group,
                         madness::TaskAttributes::hipri());
        world_.taskq.add(owner_, &Summa_::bcast_row, k, row_, col_group,
                         madness::TaskAttributes::hipri());

        // Submit tasks for the contraction of col and row tiles.
        owner_->contract(k, col_, row_, tail_step_task_);

        // Notify task dependencies
        TA_ASSERT(tail_step_task_);
        if (trace_tasks)
          tail_step_task_->notify_debug("StepTask nth ctor");
        else
          tail_step_task_->notify();
        finalize_task_->notify();

      } else if (finalize_task_) {
        // Signal the finalize task so it can run after all non-zero step
        // tasks have completed.
        finalize_task_->notify();

        // Cleanup any remaining step tasks
        StepTask* step_task = next_step_task_;
        while (step_task) {
          StepTask* const next_step_task = step_task->next_step_task_;
          step_task->next_step_task_ = nullptr;
          step_task->finalize_task_ = nullptr;
          world_.taskq.add(step_task);
          step_task = next_step_task;
        }

        if (trace_tasks)
          tail_step_task_->notify_debug("StepTask nth ctor");
        else
          tail_step_task_->notify();
      }

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_STEP
      printf("step: finish rank=%i k=%lu\n", owner_->world().rank(), k);
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_STEP
    }

  };  // class StepTask

  class DenseStepTask : public StepTask {
   protected:
    const ordinal_type k_;
    using StepTask::owner_;

   public:
    DenseStepTask(const std::shared_ptr<Summa_>& owner,
                  const ordinal_type depth)
        : StepTask(owner, owner->my_steps() + 1ul), k_(owner->next_step(0ul)) {
      StepTask::make_next_step_tasks(this, depth);
      if (k_ < owner_->nsteps_) StepTask::spawn_get_row_col_tasks(k_);
    }

    DenseStepTask(DenseStepTask* const parent, const int ndep)
        : StepTask(parent, ndep),
          k_(parent->owner_->next_step(parent->k_ + 1ul)) {
      // Spawn tasks to get k-th row and column tiles
      if (k_ < owner_->nsteps_) StepTask::spawn_get_row_col_tasks(k_);
    }

    virtual ~DenseStepTask() {}

    virtual void run(const madness::TaskThreadEnv&) {
      StepTask::template run<DenseStepTask>(k_, owner_->row_group_,
                                            owner_->col_group_);
    }
  };  // class DenseStepTask

  class SparseStepTask : public StepTask {
   protected:
    Future<ordinal_type> k_{};
    Future<madness::Group> row_group_{};
    Future<madness::Group> col_group_{};
    using StepTask::finalize_task_;
    using StepTask::next_step_task_;
    using StepTask::owner_;
    using StepTask::world_;

   private:
    /// Spawn task to construct process groups and get tiles.
    void iterate_task(ordinal_type k, const ordinal_type offset) {
      // Search for the next non-zero row and column
      k = owner_->iterate_sparse(k + offset);
      k_.set(k);

      if (k < owner_->nsteps_) {
        // NOTE: The order of task submissions is dependent on the order in
        // which we want the tasks to complete.

        // Spawn tasks to get k-th row and column tiles
        StepTask::spawn_get_row_col_tasks(k);

        // Spawn tasks to construct the row and column broadcast group
        row_group_ = world_.taskq.add(owner_, &Summa_::make_row_group, k,
                                      madness::TaskAttributes::hipri());
        col_group_ = world_.taskq.add(owner_, &Summa_::make_col_group, k,
                                      madness::TaskAttributes::hipri());

        // Increment the finalize task dependency counter, which indicates
        // that this task is not the terminating step task.
        TA_ASSERT(finalize_task_);
        finalize_task_->inc();
      }

      if (trace_tasks)
        madness::DependencyInterface::notify_debug("SparseStepTask ctor");
      else
        madness::DependencyInterface::notify();
    }

   public:
    SparseStepTask(const std::shared_ptr<Summa_>& owner, ordinal_type depth)
        : StepTask(owner, 1ul) {
      StepTask::make_next_step_tasks(this, depth);

      // Spawn a task to find the next non-zero iteration
      if (trace_tasks)
        madness::DependencyInterface::inc_debug("SparseStepTask ctor");
      else
        madness::DependencyInterface::inc();
      world_.taskq.add(this, &SparseStepTask::iterate_task, 0ul, 0ul,
                       madness::TaskAttributes::hipri());
    }

    SparseStepTask(SparseStepTask* const parent, const int ndep)
        : StepTask(parent, ndep) {
      if (parent->k_.probe() && (parent->k_.get() >= owner_->nsteps_)) {
        // Avoid running extra tasks if not needed.
        k_.set(parent->k_.get());
        TA_ASSERT(ndep ==
                  1);  // ensure that this does not get executed immediately
      } else {
        // Spawn a task to find the next non-zero iteration
        if (trace_tasks)
          madness::DependencyInterface::inc_debug("SparseStepTask ctor");
        else
          madness::DependencyInterface::inc();
        world_.taskq.add(this, &SparseStepTask::iterate_task, parent->k_, 1ul,
                         madness::TaskAttributes::hipri());
      }
    }

    virtual ~SparseStepTask() {}

    virtual void run(const madness::TaskThreadEnv&) {
      StepTask::template run<SparseStepTask>(k_, row_group_, col_group_);
    }
  };  // class SparseStepTask

 public:
  /// Constructor

  /// \param left The left-hand argument evaluator
  /// \param right The right-hand argument evaluator
  /// \param world The world where the result lives
  /// \param trange The tiled range object for the result
  /// \param shape The tensor shape object for the result
  /// \param pmap The tile-process map for the result
  /// \param perm The permutation that is applied to result tile indices
  /// \param op The tile transform operation
  /// \param k The number of tiles in the inner dimension
  /// \param proc_grid The process grid that defines the layout of the tiles
  ///                  during the contraction evaluation
  /// \param nh The number of fused (Hadamard/batch) slabs; the default (1)
  ///           is the ordinary, unbatched contraction. For nh > 1 the
  ///           arguments and the result carry the fused modes as their
  ///           leading dimensions (left = (h,i,k), right = (h,k,j),
  ///           result = (h,i,j)), each slab is distributed over the same
  ///           2-d process grid (i.e. the owner of a tile is independent of
  ///           h), and the contraction runs as nh independent SUMMA slabs
  ///           sharing one task graph with no inter-slab barriers.
  /// \note The trange, shape, and pmap refer to the final,
  ///       permuted, state for the result, NOT to the result during
  ///       the SUMMA evaluation.
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  Summa(const left_type& left, const right_type& right, World& world,
        const trange_type trange, const shape_type& shape,
        const std::shared_ptr<const pmap_interface>& pmap, const Perm& perm,
        const op_type& op, const ordinal_type k, const ProcGrid& proc_grid,
        const ordinal_type nh = 1ul, const ordinal_type proc_h = 1ul,
        const ordinal_type proc_h_stride = 0ul,
        const TiledArray::expressions::RetilePlan& plan =
            TiledArray::expressions::RetilePlan{},
        const ordinal_type k_fine =
            std::numeric_limits<ordinal_type>::max())
      : DistEvalImpl_(world, trange, shape, pmap, outer(perm)),
        left_(left),
        right_(right),
        op_(op),
        row_group_(),
        col_group_(),
        k_(k),
        k_fine_(k_fine == std::numeric_limits<ordinal_type>::max() ? k
                                                                   : k_fine),
        proc_grid_(proc_grid),
        nh_(nh),
        nsteps_(nh * k),
        left_slab_size_(left.size() / nh),
        right_slab_size_(right.size() / nh),
        result_slab_size_(proc_grid.rows() * proc_grid.cols()),
        proc_h_(proc_h),
        proc_h_stride_(proc_h_stride),
        first_slab_(compute_first_slab(world, nh, proc_h, proc_h_stride)),
        my_slabs_(first_slab_ < nh ? (nh - first_slab_ + proc_h - 1ul) / proc_h
                                   : 0ul),
        plan_(plan),
        reduce_tasks_(NULL),
        left_start_local_(proc_grid_.rank_row() * k),
        left_end_(left.size() / nh),
        left_stride_(k),
        left_stride_local_(proc_grid.proc_rows() * k),
        right_stride_(1ul),
        right_stride_local_(proc_grid.proc_cols()),
        // FINE family. Each fine member is the coarse twin (the
        // operands are tiled at the same count as the grid steps; fine_member()
        // is the identity and leaves the inactive path untouched). They are NO
        // LONGER consulted by the active gather: get_col_coarsen/get_row_coarsen
        // drive the in-step U-block gather off the COARSE proc_grid_ local
        // rows/cols and the plan_'s per-role AxisNest groups (+ target tranges
        // for refine), not these stride/start members. They are retained so the
        // INACTIVE (stock SUMMA) path is byte-for-byte unchanged, and
        // fine_member() still fires the plan-active counter to anchor it.
        left_fine_slab_size_(fine_member(plan, left.size() / nh)),
        left_fine_start_local_(
            fine_member(plan, proc_grid_.rank_row() * k_fine_)),
        left_fine_end_(fine_member(plan, left.size() / nh)),
        left_fine_stride_local_(
            fine_member(plan, proc_grid.proc_rows() * k_fine_)),
        right_fine_slab_size_(fine_member(plan, right.size() / nh)),
        right_fine_stride_local_(
            fine_member(plan, proc_grid.proc_cols()))
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
        ,
        left_ntiles_used_(0),
        right_ntiles_used_(0),
        left_ntiles_discarded_(0),
        right_ntiles_discarded_(0)
#endif
  {
    TA_ASSERT(nh_ > 0);
    TA_ASSERT(left.size() % nh_ == 0);
    TA_ASSERT(right.size() % nh_ == 0);
    TA_ASSERT(proc_h_ > 0);
    TA_ASSERT(proc_h_ == 1ul || proc_h_stride > 0ul);
    TA_ASSERT(proc_h_ <= nh_);
  }

  /// FINE-member binder. Returns the coarse-derived \p coarse_value
  /// unchanged. When the retile \p plan is active this also fires the
  /// identity-anchor counter, so the inactive (stock SUMMA) path is provably
  /// untouched. will branch here on `plan.active` to substitute the
  /// U-operand-derived fine value; this phase keeps fine == coarse so behavior
  /// is byte-for-byte identical regardless of plan state.
  static ordinal_type fine_member(
      const TiledArray::expressions::RetilePlan& plan,
      const ordinal_type coarse_value) {
    if (plan.active) {
#ifdef TA_STRIDED_DGEMM_COUNT
      g_summa_plan_active_calls.fetch_add(1, std::memory_order_relaxed);
#endif
    }
    return coarse_value;
  }

  /// \return this rank's group's first slab (== its group index), or
  /// \p nh if this rank is outside the grouped rank interval
  static ordinal_type compute_first_slab(World& world, const ordinal_type nh,
                                         const ordinal_type proc_h,
                                         const ordinal_type proc_h_stride) {
    if (proc_h == 1ul) return 0ul;
    const auto rank = ordinal_type(world.rank());
    return (rank < proc_h * proc_h_stride) ? (rank / proc_h_stride) : nh;
  }

  virtual ~Summa() {}

  /// Get tile at index \c i

  /// \param i The index of the tile
  /// \return A \c Future to the tile at index i
  /// \throw TiledArray::Exception When tile \c i is owned by a remote node.
  /// \throw TiledArray::Exception When tile \c i a zero tile.
  Future<value_type> get_tile(ordinal_type i) const override {
    TA_ASSERT(TensorImpl_::is_local(i));
    TA_ASSERT(!TensorImpl_::is_zero(i));

    // The process that owns tile i: the within-group cyclic owner shifted by
    // the world-rank offset of the tile's slab group (see
    // result_tile_owner). For proc_h_ == 1 this is the ordinary cyclic
    // owner over the whole world.
    const ProcessID source = result_tile_owner(i);

    const madness::DistributedID key(DistEvalImpl_::id(), i);
    return TensorImpl_::world().gop.template recv<value_type>(source, key);
  }

  /// Discard a tile that is not needed

  /// This function handles the cleanup for tiles that are not needed in
  /// subsequent computation.
  /// \param i The index of the tile
  void discard_tile(ordinal_type i) const override { get_tile(i); }

 private:
  /// Adjust iteration depth based on memory constraints

  /// \param depth The unbounded iteration depth
  /// \param left_sparsity The fraction of zero tiles in the left-hand matrix
  /// \param right_sparsity The fraction of zero tiles in the right-hand matrix
  /// \return The memory bounded iteration depth
  /// \thorw TiledArray::Exception When the memory bounded iteration depth
  /// is less than 1.
  ordinal_type mem_bound_depth(ordinal_type depth, const float left_sparsity,
                               const float right_sparsity) {
    // Check if a memory bound has been set
    const ordinal_type available_memory = max_memory_;
    if (available_memory) {
      // Compute the average memory requirement per iteration of this process
      const std::size_t local_memory_per_iter_left =
          (left_.trange().elements_range().volume() /
           left_.trange().tiles_range().volume()) *
          sizeof(typename numeric_type<typename left_type::eval_type>::type) *
          proc_grid_.local_rows() * (1.0f - left_sparsity);
      const std::size_t local_memory_per_iter_right =
          (right_.trange().elements_range().volume() /
           right_.trange().tiles_range().volume()) *
          sizeof(typename numeric_type<typename right_type::eval_type>::type) *
          proc_grid_.local_cols() * (1.0f - right_sparsity);

      // Compute the maximum number of iterations based on available memory
      const ordinal_type mem_bound_depth =
          ((local_memory_per_iter_left + local_memory_per_iter_right) /
           available_memory);

      // Check if the memory bounded depth is less than the optimal depth
      if (depth > mem_bound_depth) {
        // Adjust the depth based on the available memory
        switch (mem_bound_depth) {
          case 0:
            // When memory bound depth is
            TA_EXCEPTION("Insufficient memory available for SUMMA");
            break;
          case 1:
            if (TensorImpl_::world().rank() == 0)
              printf(
                  "!! WARNING TiledArray: Memory constraints limit the SUMMA "
                  "depth depth to 1.\n"
                  "!! WARNING TiledArray: Performance may be slow.\n");
          default:
            depth = mem_bound_depth;
        }
      }
    }

    return depth;
  }

  /// Evaluate the tiles of this tensor

  /// This function will evaluate the children of this distributed evaluator
  /// and evaluate the tiles for this distributed evaluator. It will block
  /// until the tasks for the children are evaluated (not for the tasks of
  /// this object).
  /// \return The number of tiles that will be set by this process
  int internal_eval() override {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL
    printf("eval: start eval children rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL

    // Start evaluate child tensors
    left_.eval();
    right_.eval();

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL
    printf("eval: finished eval children rank=%i\n",
           TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL

    ordinal_type tile_count = 0ul;
    if (proc_grid_.local_size() > 0ul) {
      tile_count = initialize();

      // depth controls the number of simultaneous SUMMA iterations
      // that are scheduled.

      // The optimal depth is equal to the smallest dimension of the process
      // grid, but no less than 2
      ordinal_type depth =
          std::max(ProcGrid::size_type(2),
                   std::min(proc_grid_.proc_rows(), proc_grid_.proc_cols()));

      // watch out for the corner case: contraction over zero-volume range
      // producing nonzero-volume result ... in that case there is nothing to do
      // the appropriate initialization was performed in the initialize() method
      if (nsteps_ != 0) {
        // Construct the first SUMMA iteration task
        if (TensorImpl_::shape().is_dense()) {
          // We cannot have more iterations than there are SUMMA steps
          if (depth > nsteps_) depth = nsteps_;

          // Modify the number of concurrent iterations based on the available
          // memory.
          depth = mem_bound_depth(depth, 0.0f, 0.0f);

          // Enforce user defined depth bound
          if (max_depth_) depth = std::min(depth, max_depth_);

          TensorImpl_::world().taskq.add(
              new DenseStepTask(shared_from_this(), depth));
        } else {
          // Increase the depth based on the amount of sparsity in an iteration.

          // Get the sparsity fractions for the left- and right-hand arguments.
          const float left_sparsity = left_.shape().sparsity();
          const float right_sparsity = right_.shape().sparsity();

          // Compute the fraction of non-zero result tiles in a single SUMMA
          // iteration.
          const float frac_non_zero = (1.0f - std::min(left_sparsity, 0.9f)) *
                                      (1.0f - std::min(right_sparsity, 0.9f));

          // Compute the new depth based on sparsity of the arguments
          depth = float(depth) * (1.0f - 1.35638f * std::log2(frac_non_zero)) +
                  0.5f;

          // We cannot have more iterations than there are SUMMA steps
          if (depth > nsteps_) depth = nsteps_;

          // Modify the number of concurrent iterations based on the available
          // memory and sparsity of the argument tensors.
          depth = mem_bound_depth(depth, left_sparsity, right_sparsity);

          // Enforce user defined depth bound
          if (max_depth_) depth = std::min(depth, max_depth_);

          TensorImpl_::world().taskq.add(
              new SparseStepTask(shared_from_this(), depth));
        }
      }  // nsteps_ != 0
    }

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL
    printf("eval: start wait children rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL

    // corner case: if left or right are zero-volume no tasks were scheduled, so
    // need to discard all of their tiles manually
    if (left_.range().volume() == 0) {
      for (auto&& tile_idx : right_.range()) {
        auto tile_ord = right_.range().ordinal(tile_idx);
        if (right_.is_local(tile_ord) && !right_.is_zero(tile_ord))
          right_.discard(tile_ord);
      }
    }
    if (right_.range().volume() == 0) {
      for (auto&& tile_idx : left_.range()) {
        auto tile_ord = left_.range().ordinal(tile_idx);
        if (left_.is_local(tile_ord) && !left_.is_zero(tile_ord))
          left_.discard(tile_ord);
      }
    }

    // Wait for child tensors to be evaluated, and process tasks while waiting.
    left_.wait();
    right_.wait();
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
    // values of left_ntiles_used_ etc. are not available until all broadcasts
    // have been completed ...
//    TA_ASSERT(left_.local_nnz() == left_ntiles_used_ +
//    left_ntiles_discarded_); TA_ASSERT(right_.local_nnz() ==
//    right_ntiles_used_ + right_ntiles_discarded_);
//    TA_ASSERT(left_.task_count() >= left_ntiles_used_ +
//    left_ntiles_discarded_); TA_ASSERT(right_.task_count() >=
//    right_ntiles_used_ + right_ntiles_discarded_);
#endif

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL
    printf("eval: finished wait children rank=%i\n",
           TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL

    return tile_count;
  }

};  // class Summa

// Initialize static member variables for Summa

template <typename Left, typename Right, typename Op, typename Policy>
typename Summa<Left, Right, Op, Policy>::ordinal_type
    Summa<Left, Right, Op, Policy>::max_depth_ =
        Summa<Left, Right, Op, Policy>::init_max_depth();

template <typename Left, typename Right, typename Op, typename Policy>
typename Summa<Left, Right, Op, Policy>::ordinal_type
    Summa<Left, Right, Op, Policy>::max_memory_ =
        Summa<Left, Right, Op, Policy>::init_max_memory();
}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_DIST_EVAL_CONTRACTION_EVAL_H__INCLUDED
