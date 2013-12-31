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

#include <TiledArray/dist_eval/dist_eval.h>
#include <TiledArray/proc_grid.h>
#include <TiledArray/reduce_task.h>
#include <TiledArray/tile_op/type_traits.h>

namespace TiledArray {
  namespace detail {

    // Forward declarations
    class SparseShape;

    /// Distributed contraction evaluator implementation

    /// \param Left The left-hand argument evaluator type
    /// \param Right The right-hand argument evaluator type
    /// \param Op The contraction/reduction operation type
    /// \param Policy The tensor policy class
    template <typename Left, typename Right, typename Op, typename Policy>
    class ContractionEvalImpl : public DistEvalImpl<typename Op::result_type, Policy> {
    public:
      typedef ContractionEvalImpl<Left, Right, Op, Policy> ContractionEvalImpl_; ///< This object type
      typedef DistEvalImpl<typename Op::result_type, Policy> DistEvalImpl_; ///< The base class type
      typedef typename DistEvalImpl_::TensorImpl_ TensorImpl_; ///< The base, base class type
      typedef Left left_type; ///< The left-hand argument type
      typedef Right right_type; ///< The right-hand argument type
      typedef typename DistEvalImpl_::size_type size_type; ///< Size type
      typedef typename DistEvalImpl_::range_type range_type; ///< Range type
      typedef typename DistEvalImpl_::shape_type shape_type; ///< Shape type
      typedef typename DistEvalImpl_::pmap_interface pmap_interface; ///< Process map interface type
      typedef typename DistEvalImpl_::trange_type trange_type; ///< Tiled range type
      typedef typename DistEvalImpl_::value_type value_type; ///< Tile type
      typedef typename DistEvalImpl_::eval_type eval_type; ///< Tile evaluation type
      typedef Op op_type; ///< Tile evaluation operator type

    private:
      // Arguments and operation
      left_type left_; ///< The left argument tensor
      right_type right_; /// < The right argument tensor
      op_type op_; /// < The contraction/reduction operation

      // Broadcast groups for dense arguments (empty for non-dense arguments)
      madness::Group row_group_; ///< The row process group for this rank
      madness::Group col_group_; ///< The column process group for this rank

      // Dimension information
      size_type k_; ///< Number of tiles in the inner dimension
      const ProcGrid proc_grid_; ///< Process grid for this contraction

      // Contraction results
      ReducePairTask<op_type>* reduce_tasks_; ///< A pointer to the reduction tasks

    private:

      typedef madness::Future<typename right_type::eval_type> right_future; ///< Future to a right-hand argument tile
      typedef madness::Future<typename left_type::eval_type> left_future; ///< Future to a left-hand argument tile
      typedef std::pair<size_type, right_future> row_datum; ///< Datum element type for a right-hand argument row
      typedef std::pair<size_type, left_future> col_datum; ///< Datum element type for a left-hand argument column

      template <typename T>
      static typename T::eval_type convert_tile(const T& tile) {
        return tile;
      }

      template <typename Arg>
      static typename madness::disable_if<
          TiledArray::math::is_lazy_tile<typename Arg::value_type>,
          madness::Future<typename Arg::value_type> >::type
      move_tile(Arg& arg, size_type index) { return arg.move(index); }

      template <typename Arg>
      typename madness::enable_if<
          TiledArray::math::is_lazy_tile<typename Arg::value_type>,
          madness::Future<typename Arg::eval_type> >::type
      move_tile(Arg& arg, size_type index) const {
        return TensorImpl_::get_world().taskq.add(
            & ContractionEvalImpl_::template convert_tile<typename Arg::value_type>,
            arg.move(index), madness::TaskAttributes::hipri());
      }

      /// Broadcast column \c k of the left-hand argument

      /// \param k The column of \c left_ to be broadcast
      /// \param[out] col The vector that will hold the tiles in column \c k
      void bcast_col(const size_type k, std::vector<col_datum>& col) const {
        TA_ASSERT(col.size() == 0ul);

        // Compute the left-hand argument column first index, end, and stride
        size_type index = proc_grid_.rank_row() * proc_grid_.cols() + k;
        const size_type end = proc_grid_.size();
        const size_type stride = proc_grid_.proc_rows() * proc_grid_.cols();

        // Get the broadcast process group and root process
        madness::Group group;
        if(right_.is_dense()) {
          group = row_group_;
        } else {
          madness::DistributedID did(TensorImpl_::id(), k);
          group = proc_grid_.make_row_group(did, right_.shape(), k, right_.size());
          group.register_group();
        }
        const ProcessID group_root = group.rank(left_.owner(index));

        // Allocate memory for the column
        col.reserve(proc_grid_.local_rows());

        // Broadcast and store non-zero tiles in the k-th column of the left-hand argument.
        for(size_type i = 0ul; index < end; ++i, index += stride) {
          if(! left_.is_zero(index)) {
            // Get column tile
            col.push_back(col_datum(i, move_tile(left_, index)));

            const madness::DistributedID key(TensorImpl_::id(), index);
            TensorImpl_::get_world().gop.bcast(key, col.back().second, group_root, group);
          }
        }

        // Cleanup group
        if(! right_.is_dense())
          group.unregister_group();
      }

      void bcast_row(const size_type k, std::vector<row_datum>& row) const {
        TA_ASSERT(row.size() == 0ul);

        // Compute the left-hand argument column first index, end, and stride
        size_type index = k * proc_grid_.cols() + proc_grid_.rank_col();
        const size_type end = (k + 1) * proc_grid_.cols();
        const size_type stride = proc_grid_.proc_cols();

        // Get the broadcast process group and root
        madness::Group group;
        if(left_.is_dense()) {
          group = col_group_;
        } else {
          madness::DistributedID did(TensorImpl_::id(), k + k_);
          group = proc_grid_.make_col_group(did, left_.shape(), k, left_.size());
          group.register_group();
        }
        const ProcessID group_root = group.rank(right_.owner(index));

        // Allocate memory for the row
        row.reserve(proc_grid_.local_cols());

        // Broadcast and store non-zero tiles in the k-th row of the right-hand argument.
        for(size_type i = 0ul; index < end; ++i, index += stride) {
          if(! right_.is_zero(index)) {
            // Get column tile
            row.push_back(row_datum(i, move_tile(right_, index)));

            const madness::DistributedID key(TensorImpl_::id(), index + left_.size());
            TensorImpl_::get_world().gop.bcast(key, row.back().second, group_root, group);
          }
        }

        // Cleanup group
        if(! left_.is_dense())
          group.unregister_group();
      }

      void bcast_col_task(const size_type k) const {
        std::vector<col_datum> col;
        bcast_col(k, col);
      }

      void bcast_row_task(const size_type k) const {
        std::vector<row_datum> row;
        bcast_row(k, row);
      }

      /// Find the next row

      /// Starting at the k-th row of the right-hand argument, find the next row
      /// that contains at least one non-zero tile. This search only checks for
      /// non-zero tiles in this processes column.
      /// \param k The first row to test for non-zero tiles
      /// \return The first row, greater than or equal to \c k, that contains a
      /// non-zero tile. If no non-zero tile is not found, return \c k_.
      size_type next_k_row(size_type k) const {
        if(right_.is_dense())
          return k;

        // Initialize row end to the start of the k-th row.
        size_type row_end = k * proc_grid_.cols();

        // Iterate over k's until a non-zero tile is found or the end of the
        // matrix is reached.
        for(; k < k_; ++k) {

          // Set the starting and ending point for row k
          size_type i = row_end + proc_grid_.rank_col();
          row_end += proc_grid_.cols();

          // Search row k for non-zero tiles
          for(; i < row_end; i += proc_grid_.proc_cols())
            if(! right_.is_zero(i))
              return k;
        }

        return k;
      }

      /// Find the next column

      /// Starting at the k-th column of the left-hand argument, find the next
      /// column that contains at least one non-zero tile. This search only
      /// checks for non-zero tiles in this process's row.
      /// \param k The first column to test for non-zero tiles
      /// \return The first column, greater than or equal to \c k, that contains
      /// a non-zero tile. If no non-zero tile is not found, return \c k_.
      size_type next_k_col(size_type k) const {
        if(left_.is_dense())
          return k;

        // Iterate over k's until a non-zero tile is found or the end of the
        // matrix is reached.
        const size_type col_start = proc_grid_.rank_row() * k_;
        const size_type col_step = proc_grid_.proc_rows() * k_;
        for(; k < k_; ++k) {

          // Search row k for non-zero tiles
          for(size_type i = col_start + k; i < left_.size(); i += col_step)
            if(! left_.is_zero(i))
              return k;
        }

        return k;
      }

      /// Find the next k where the left- and right-hand argument have non-zero tiles

      /// Search for the next k-th column and row of the left- and right-hand
      /// arguments, respectively, that both contain non-zero tiles. This search
      /// only checks for non-zero tiles in this process's row or column. If a
      /// non-zero, local tile is found that does not contribute to local
      /// contractions, the tiles will be immediately broadcast.
      /// \param k The first row/column to check
      /// \return The next k-th column and row of the left- and right-hand
      /// arguments, respectively, that both have non-zero tiles
      size_type next_k(const std::shared_ptr<ContractionEvalImpl_>& self, const size_type k) const {
        // The if statements below should be optimized away since the left and
        // right dense variables are static constants.
        if(left_.is_dense() && right_.is_dense()) {
          // Left- and right-hand arguments are dense so k is always non-zero
          return k;
        } else {
          // Left and right are sparse so search for the first k column of
          // left and row of right that both have non-zero tiles.
          size_type k_col = next_k_col(k);
          size_type k_row = next_k_row(k);
          while(k_col != k_row) {
            // Check the largest k for non-zero tiles.
            if(k_col < k_row) {
              // If k_col has is local, broadcast the data to other nodes with
              // non-zero rows.
              if(left_.is_local(proc_grid_.rank_row() * k_ + k_col))
                TensorImpl_::get_world().taskq.add(self, & ContractionEvalImpl_::bcast_col_task,
                    k_col, madness::TaskAttributes::hipri());

              // Find the next non-zero column of the left-hand argument
              k_col = next_k_col(k_col + 1ul);
            } else {
              if(right_.is_local(k_row * proc_grid_.cols() + proc_grid_.rank_col()))
                TensorImpl_::get_world().taskq.add(self, & ContractionEvalImpl_::bcast_row_task,
                    k_row, madness::TaskAttributes::hipri());

              k_row = next_k_row(k_row + 1ul);
            }
          }

          return k_col;
        }
      }

      /// Destroy reduce tasks and set the result tiles
      void finalize() {
        // Create allocator
        std::allocator<ReducePairTask<op_type> > alloc;

        // Initialize loop control variables
        size_type offset = 0ul;
        size_type i_start = proc_grid_.rank_row() * proc_grid_.cols() + proc_grid_.rank_col();
        const size_type i_start_step = proc_grid_.proc_rows() * proc_grid_.cols();

        // Iterate over all local rows and columns
        for(; i_start < proc_grid_.size(); i_start += i_start_step) {
          for(size_type col = proc_grid_.rank_col(); col < proc_grid_.cols(); col += proc_grid_.proc_cols(), ++offset) {
            // Compute the index
            const size_type index = i_start + col;

            if(! TensorImpl_::is_zero(index)) {
              // Get the ordinal index and reduce task for tile (r,c).
              ReducePairTask<op_type>* const reduce_task = reduce_tasks_ + offset;

              // Set the result tile
              DistEvalImpl_::set_tile(index, reduce_task->submit());

              // Destroy the the reduce task
              reduce_task->~ReducePairTask<op_type>();
            }
          }

          // Unregister groups if used.
          if(left_.is_dense())
            col_group_.unregister_group();
          if(right_.is_dense())
            row_group_.unregister_group();
        }

        // Deallocate the memory for the reduce pair tasks.
        alloc.deallocate(reduce_tasks_, proc_grid_.local_size());
      }

      /// Schedule local contraction tasks for \c col and \c row tile pairs

      /// Schedule tile contractions for each tile pair of \c row and \c col. A
      /// callback to \c task will be registered with each tile contraction
      /// task.
      /// \param col A column of tiles from the left-hand argument
      /// \param row A row of tiles from the right-hand argument
      /// \param callback The callback that will be invoked after each tile-pair
      /// has been contracted
      template <typename S>
#ifndef TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER
      typename madness::disable_if<std::is_same<S, SparseShape> >::type
#else
      void
#endif // TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER
      contract(const size_type, const std::vector<col_datum>& col,
          const std::vector<row_datum>& row, madness::TaskInterface* const task)
      {
        // Iterate over the row
        for(typename std::vector<col_datum>::const_iterator col_it = col.begin(); col_it != col.end(); ++col_it) {
          // Compute the local, result-tile offset
          const size_type offset = col_it->first * proc_grid_.local_cols();

          for(typename std::vector<row_datum>::const_iterator row_it = row.begin(); row_it != row.end(); ++row_it) {
            task->inc();
            reduce_tasks_[offset + col_it->first].add(col_it->second, row_it->second, task);
          }
        }
      }

#ifndef TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER
      /// Schedule local contraction tasks for \c col and \c row tile pairs

      /// Schedule tile contractions for each tile pair of \c row and \c col. A
      /// callback to \c task will be registered with each tile contraction
      /// task. This version of contract is used when shape_type is
      /// \c SparseShape. It skips tile contractions that have a negligible
      /// contribution to the result tile.
      /// \param k The k step for this contraction set
      /// \param col A column of tiles from the left-hand argument
      /// \param row A row of tiles from the right-hand argument
      /// \param task The task that depends on the tile contraction tasks
      template <typename S>
      typename madness::enable_if<std::is_same<S, SparseShape> >::type
      contract(const size_type k, const std::vector<col_datum>& col,
          const std::vector<row_datum>& row, madness::TaskInterface* const task)
      {

        // Compute the threshold that will be applied to tile pair contractions.
        const float threshold_k = TensorImpl_::shape().threshold() / float(k_);

        /// Compute the base index and strides for tiles of col and row.
        const size_type left_index = proc_grid_.rank_row() * k_ + k;
        const size_type left_stride = proc_grid_.proc_rows() * k_;
        const size_type right_index = k * proc_grid_.cols() + proc_grid_.rank_col();
        const size_type right_stride = proc_grid_.proc_cols();

        // Iterate over the left-hand argument column (rows of the result)
        for(typename std::vector<col_datum>::const_iterator col_it = col.begin(); col_it != col.end(); ++col_it) {
          // Compute the local, result-tile offset for the current result row.
          const size_type result_offset = col_it->first * proc_grid_.local_cols();

          // Get the shape data for col_it tile
          const float col_shape_value =
              left_.shape().data()[left_index + (col_it->first * left_stride)];

          // Iterate over the right-hand argument row (cols of the result)
          for(typename std::vector<row_datum>::const_iterator row_it = row.begin(); row_it != row.end(); ++row_it) {

            const float row_shape_value =
                right_.shape().data()[right_index + (row_it->first * right_stride)];

            if((col_shape_value * row_shape_value) >= threshold_k) { // Filter trivial results
              task->inc();
              reduce_tasks_[result_offset + row_it->first].add(col_it->second, row_it->second, task);
            }
          }
        }
      }
#endif // TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER

      class StepTask : public madness::TaskInterface {
      private:
        class BcastTask;

        // Member variables
        std::shared_ptr<ContractionEvalImpl_> parent_;
        size_type k_;
        StepTask* next_step_task_;


        /// Construct the task for the next step
        StepTask(StepTask* const previous, const int ndep) :
          madness::TaskInterface(ndep, madness::TaskAttributes::hipri()),
          parent_(previous->parent_), k_(0ul), next_step_task_(NULL)
        { }

      public:

        StepTask(const std::shared_ptr<ContractionEvalImpl_>& parent) :
          madness::TaskInterface(madness::TaskAttributes::hipri()),
          parent_(parent), k_(0ul), next_step_task_(new StepTask(this, 0))
        { }

        virtual ~StepTask() { }

        void submit_next(const size_type k) {
          next_step_task_->k_ = k;
          parent_->get_world().taskq.add(next_step_task_);
          next_step_task_ = NULL;
        }

        virtual void run(const madness::TaskThreadEnv&) {
          if(k_ < parent_->k_) {
            // Get the column and row for the next non-zero k iteration
            k_ = parent_->next_k(parent_, k_);

            if(k_ < parent_->k_) {
              // Submit the task for the next iteration
              StepTask* const next_next_step_task =
                  next_step_task_->next_step_task_ = new StepTask(this, 1);
              submit_next(k_ + 1ul);

              // Broadcast row and column
              std::vector<col_datum> col;
              parent_->bcast_col(k_, col);
              std::vector<row_datum> row;
              parent_->bcast_row(k_, row);

              // Submit tasks for the contraction of col and row tiles.
              parent_->template contract<shape_type>(k_, col, row, next_next_step_task);

              next_next_step_task->notify();

            } else {
              parent_->finalize();

              // Submit the tail task to avoid errors.
              submit_next(k_);
            }
          }
        }

      }; // class StepTask

    public:

      ContractionEvalImpl(const left_type& left, const right_type& right,
          madness::World& world, const trange_type trange, const shape_type& shape,
          const std::shared_ptr<pmap_interface>& pmap, const Permutation& perm,
          const op_type& op, const size_type k, const ProcGrid& proc_grid) :
        DistEvalImpl_(world, trange, shape, pmap, perm),
        left_(left), right_(right), op_(op),
        row_group_(), col_group_(),
        k_(k), proc_grid_(proc_grid),
        reduce_tasks_(NULL)
      { }

      virtual ~ContractionEvalImpl() { }

    private:
      /// Function for evaluating this tensor's tiles

      /// This function is run inside a task, and will run after \c eval_children
      /// has completed. It should spawn additional tasks that evaluate the
      /// individual result tiles.
      virtual size_type internal_eval(const std::shared_ptr<DistEvalImpl_>& pimpl) {
        // Convert pimpl to this object type so it can be used in tasks
        std::shared_ptr<ContractionEvalImpl_> self =
            std::static_pointer_cast<ContractionEvalImpl_>(pimpl);

        // Start evaluate child tensors
        left_.eval();
        right_.eval();

        if(proc_grid_.local_size() != 0ul) {
          // Construct static broadcast groups for dense arguments
          if(left_.is_dense()) {
            col_group_ = proc_grid_.make_col_group(madness::DistributedID(TensorImpl_::id(), 0ul));
            col_group_.register_group();
          }
          if(right_.is_dense()) {
            row_group_ = proc_grid_.make_row_group(madness::DistributedID(TensorImpl_::id(), k_));
            row_group_.register_group();
          }

          // Allocate memory for the reduce pair tasks.
          std::allocator<ReducePairTask<op_type> > alloc;
          reduce_tasks_ = alloc.allocate(proc_grid_.local_size());

          // Iterate over all local rows and columns
          size_type offset = 0ul;
          size_type tile_count = 0ul;
          for(size_type row = proc_grid_.rank_row(); row < proc_grid_.rows(); row += proc_grid_.proc_rows()) {
            for(size_type col = proc_grid_.rank_col(); col < proc_grid_.cols(); col += proc_grid_.proc_cols(), ++offset) {

              // Construct non-zero reduce tasks
              if(! TensorImpl_::is_zero(offset)) {
                new(reduce_tasks_ + offset) ReducePairTask<op_type>(TensorImpl_::get_world(), op_);
                ++tile_count;
              }
            }
          }
          // Construct the first SUMMA iteration task
          TensorImpl_::get_world().taskq.add(new StepTask(self));
        }

        // Wait for child tensors to be evaluated, and process tasks while waiting.
        left_.wait();
        right_.wait();

        return tile_count;
      }

    }; // class ContractionEvalImpl


    /// Distributed contraction evaluator factory function

    /// Construct a distributed contraction evaluator, which constructs a new
    /// tensor by applying \c op to tiles of \c left and \c right.
    /// \tparam Tile Tile type of the argument
    /// \tparam Policy The policy type of the argument
    /// \tparam Op The unary tile operation
    /// \param left The left-hand argument
    /// \param right The right-hand argument
    /// \param world The world where the argument will be evaluated
    /// \param shape The shape of the evaluated tensor
    /// \param pmap The process map for the evaluated tensor
    /// \param perm The permutation applied to the tensor
    /// \param op The contraction/reduction tile operation
    template <typename LeftTile, typename RightTile, typename Policy, typename Op>
    DistEval<typename Op::result_type, Policy> make_contract_eval(
        const DistEval<LeftTile, Policy>& left,
        const DistEval<RightTile, Policy>& right,
        madness::World& world,
        const typename DistEval<typename Op::result_type, Policy>::shape_type& shape,
        const std::shared_ptr<typename DistEval<typename Op::result_type, Policy>::pmap_interface>& pmap,
        const Permutation& perm,
        const Op& op)
    {
      TA_ASSERT(left.range().dim() == op.left_rank());
      TA_ASSERT(right.range().dim() == op.right_rank());
      TA_ASSERT((perm.dim() == op.result_rank()) || (perm.dim() == 0u));

      // Define the impl type
      typedef ContractionEvalImpl<DistEval<LeftTile, Policy>, DistEval<RightTile,
          Policy>, Op, Policy> impl_type;

      // Precompute iteration range data
      const unsigned int num_contract_ranks = op.num_contract_ranks();
      const unsigned int left_end = op.left_rank();
      const unsigned int left_middle = left_end - num_contract_ranks;
      const unsigned int right_end = op.right_rank();

      // Construct the trange for the result tensor
      typename impl_type::trange_type::Ranges ranges(op.result_rank());

      // Iterate over the left outer dimensions
      std::size_t M = 1ul, m = 1ul;
      std::size_t pi = 0ul;
      for(unsigned int i = 0ul; i < left_middle; ++i) {
        ranges[(perm.dim() > 0ul ? perm[pi++] : pi++)] = left.trange().data()[i];
        M *= left.range().size()[i];
        m *= left.trange().elements().size()[i];
      }
      // Iterate over the right outer dimensions
      std::size_t N = 1ul, n = 1ul;
      for(std::size_t i = num_contract_ranks; i < right_end; ++i) {
        ranges[(perm.dim() > 0ul ? perm[pi++] : pi++)] = right.trange().data()[i];
        N *= right.range().size()[i];
        n *= right.trange().elements().size()[i];
      }

      // Compute the number of tiles in the inner dimension
      std::size_t K = 1ul;
      for(std::size_t i = left_middle; i < left_end; ++i)
        K *= left.range().size()[i];

      // Construct the result range
      typename impl_type::trange_type trange(ranges.begin(), ranges.end());

      // Construct the process grid
      ProcGrid proc_grid(world, M, N, m, n);

      return DistEval<typename Op::result_type, Policy>(
          std::shared_ptr<typename impl_type::DistEvalImpl_>(new impl_type(left,
              right, world, trange, shape, pmap, perm, op, K, proc_grid)));
    }

  } // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_DIST_EVAL_CONTRACTION_EVAL_H__INCLUDED
