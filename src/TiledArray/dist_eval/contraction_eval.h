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

      // Constant left and right index iterator bounds
      const size_type left_start_local_;
      const size_type left_end_;
      const size_type left_stride_;
      const size_type left_stride_local_;
      const size_type right_stride_;
      const size_type right_stride_local_;


      typedef madness::Future<typename right_type::eval_type> right_future; ///< Future to a right-hand argument tile
      typedef madness::Future<typename left_type::eval_type> left_future; ///< Future to a left-hand argument tile
      typedef std::pair<size_type, right_future> row_datum; ///< Datum element type for a right-hand argument row
      typedef std::pair<size_type, left_future> col_datum; ///< Datum element type for a left-hand argument column


      size_type left_begin(const size_type k) const { return k; }
      size_type left_begin_local(const size_type k) const { return left_start_local_ + k; }

      size_type right_begin(const size_type k) const { return k * proc_grid_.cols(); }
      size_type right_begin_local(const size_type k) const { return k * proc_grid_.cols() + proc_grid_.rank_col(); }
      size_type right_end(const size_type k) const { return (k + 1ul) * proc_grid_.cols(); }


      template <typename T>
      static typename T::eval_type convert_tile(const T& tile) { return tile; }

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


      /// Sparse row group factory function

      /// Construct a row group that includes all processes that will receive/
      /// have non-zero data for iteration \c k.
      /// \param k The row of right that will be used to generate the sparse row group
      /// \return A sparse row group
      madness::Group
      make_sparse_row_group(const size_type k) const {
        // Generate the list of processes in rank_row
        std::vector<ProcessID> proc_list(proc_grid_.proc_cols(), -1);

        // Flag all process that have non-zero tiles
        const size_type p_start = proc_grid_.rank_row() * proc_grid_.proc_cols();
        const size_type row_k_end = right_end(k);
        size_type count = 0ul;
        for(size_type i = right_begin(k), p = 0u; i < row_k_end;
            i += right_stride_, p = (p + 1u) % proc_grid_.proc_cols())
        {
          if(proc_list[p] != -1) continue;
          if(right_.shape().is_zero(i)) continue;

          proc_list[p] = p_start + p;
          ++count;
          if(count == proc_list.size()) break;
        }

        if(count < proc_list.size()) {
          // Convert flags into process numbers
          size_type x = 0ul;
          for(size_type p = 0ul; p < proc_list.size(); ++p) {
            if(proc_list[p] == -1) continue;
            proc_list[x++] = proc_list[p];
          }

          // Truncate invalid process id's
          proc_list.resize(x);
        }


        return madness::Group(TensorImpl_::get_world(), proc_list,
            madness::DistributedID(TensorImpl_::id(), k + k_));
      }

      /// Sparse column group factory function

      /// Construct a column group that includes all processes that will receive/
      /// have non-zero data for iteration \c k.
      /// \param k The column of left that will be used to generate the sparse column group
      /// \return A sparse column group
      /// \note This function assumes that there is at least one process that
      /// contains non-zero data and that this process contains non
      madness::Group
      make_sparse_col_group(const size_type k) const {
        std::vector<ProcessID> proc_list(proc_grid_.proc_rows(), 0);

        // Flag all process that have non-zero tiles
        size_type count = 0ul;
        for(size_type i = left_begin(k), p = 0u;
            i < left_end_; i += left_stride_, p = (p + 1u) % proc_grid_.proc_rows())
        {
          if(proc_list[p] == -1) continue;
          if(left_.shape().is_zero(i)) continue;

          proc_list[p] = p * proc_grid_.proc_cols() + proc_grid_.rank_col();
          ++count;
          if(count == proc_list.size()) break;
        }

        if(count < proc_list.size()) {
          // Convert flags into process numbers
          size_type x = 0ul;
          for(size_type p = 0ul; p < proc_grid_.proc_rows(); ++p) {
            if(proc_list[p] == -1) continue;

            proc_list[x] = proc_list[p];
            ++x;
          }

          // Truncate invalid the process id's
          proc_list.resize(x);
        }

        return madness::Group(TensorImpl_::get_world(), proc_list,
            madness::DistributedID(TensorImpl_::id(), k));
      }

      /// Broadcast column \c k of the left-hand argument

      /// \param k The column of \c left_ to be broadcast
      /// \param[out] col The vector that will hold the tiles in column \c k
      void bcast_col(const size_type k, std::vector<col_datum>& col) const {
        TA_ASSERT(col.size() == 0ul);

        // Compute the left-hand argument column first index, end, and stride
        size_type index = left_begin_local(k);

        // Get the broadcast process group and root process
        madness::Group group;
        if(right_.is_dense()) {
          group = row_group_;
        } else {
          group = make_sparse_row_group(k);
          group.register_group();
        }
        const ProcessID group_root = group.rank(left_.owner(index));

        // Allocate memory for the column
        col.reserve(proc_grid_.local_rows());

        // Broadcast and store non-zero tiles in the k-th column of the left-hand argument.
        for(size_type i = 0ul; index < left_end_; ++i, index += left_stride_local_) {
          if(left_.is_zero(index)) continue;

          // Get column tile
          col.push_back(col_datum(i, (left_.is_local(index) ? move_tile(left_, index) : left_future())));

          const madness::DistributedID key(TensorImpl_::id(), index);
          TensorImpl_::get_world().gop.bcast(key, col.back().second, group_root, group);
        }

        // Cleanup group
        if(! right_.is_dense())
          group.unregister_group();
      }

      void bcast_row(const size_type k, std::vector<row_datum>& row) const {
        TA_ASSERT(row.size() == 0ul);

        // Compute the left-hand argument column first index, end, and stride
        size_type index = right_begin_local(k);
        const size_type end = right_end(k);

        // Get the broadcast process group and root
        madness::Group group;
        if(left_.is_dense()) {
          group = col_group_;
        } else {
          group = make_sparse_col_group(k);
          group.register_group();
        }
        const ProcessID group_root = group.rank(right_.owner(index));

        // Allocate memory for the row
        row.reserve(proc_grid_.local_cols());

        // Broadcast and store non-zero tiles in the k-th row of the right-hand argument.
        for(size_type i = 0ul; index < end; ++i, index += right_stride_local_) {
          if(right_.is_zero(index)) continue;

          // Get row tile
          row.push_back(row_datum(i, (right_.is_local(index) ? move_tile(right_, index) : right_future())));

          const madness::DistributedID key(TensorImpl_::id(), index + left_.size());
          TensorImpl_::get_world().gop.bcast(key, row.back().second, group_root, group);
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
        if(! right_.is_dense()) {
          // Iterate over k's until a non-zero tile is found or the end of the
          // matrix is reached.
          for(; k < k_; ++k) {

            // Set the starting and ending point for row k
            size_type index = right_begin_local(k);
            const size_type end = right_end(k);

            // Search row k for non-zero tiles
            for(; index < end; index += right_stride_local_)
              if(! right_.is_zero(index))
                return k;
          }
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
        if(! left_.is_dense()) {
          // Iterate over k's until a non-zero tile is found or the end of the
          // matrix is reached.
          for(; k < k_; ++k) {
            // Search row k for non-zero tiles
            for(size_type index = left_begin_local(k); index < left_end_; index += left_stride_local_)
              if(! left_.is_zero(index))
                break;
          }
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
        size_type k_col = next_k_col(k);
        size_type k_row = next_k_row(k);
        while(k_col != k_row) {
          // Check the largest k for non-zero tiles.
          if(k_col < k_row) {
            // If k_col has local data, broadcast the data to other nodes with
            // non-zero rows.
            if(left_.is_local(left_start_local_ + k_col))
              TensorImpl_::get_world().taskq.add(self, & ContractionEvalImpl_::bcast_col_task,
                  k_col, madness::TaskAttributes::hipri());

            // Find the next non-zero column of the left-hand argument
            k_col = next_k_col(k_col + 1ul);
          } else {
            // If k_row has local data, broadcast the data to other nodes with
            // non-zero columns.
            if(right_.is_local(k_row * proc_grid_.cols() + proc_grid_.rank_col()))
              TensorImpl_::get_world().taskq.add(self, & ContractionEvalImpl_::bcast_row_task,
                  k_row, madness::TaskAttributes::hipri());

            k_row = next_k_row(k_row + 1ul);
          }
        }

        return k_col;
      }

      /// Destroy reduce tasks and set the result tiles
      void finalize() {

        // Construct inverse permuted weight and start arrays.
        const Permutation inv_perm = -DistEvalImpl_::perm();
        const range_type range = inv_perm ^ TensorImpl_::range();
        const std::vector<size_type> ip_weight = inv_perm ^ TensorImpl_::range().weight();

        // Iterate over all local rows and columns
        ReducePairTask<op_type>* reduce_task = reduce_tasks_;
        for(size_type row = proc_grid_.rank_row(); row < proc_grid_.rows(); row += proc_grid_.proc_rows()) {
          const size_type row_start = row * proc_grid_.cols();
          for(size_type col = proc_grid_.rank_col(); col < proc_grid_.cols(); col += proc_grid_.proc_cols(), ++reduce_task) {

            // Compute convert the working ordinal index to a
            const std::size_t index = DistEvalImpl_::perm_index(row_start + col);

            // Construct non-zero reduce tasks
            if(! TensorImpl_::is_zero(index)) {
              // Set the result tile
              DistEvalImpl_::set_tile(index, reduce_task->submit());

              // Destroy the the reduce task
              reduce_task->~ReducePairTask<op_type>();
            }
          }
        }

        // Unregister groups if used.
        if(left_.is_dense())
          col_group_.unregister_group();
        if(right_.is_dense())
          row_group_.unregister_group();

        // Deallocate the memory for the reduce pair tasks.
        std::allocator<ReducePairTask<op_type> >().deallocate(reduce_tasks_, proc_grid_.local_size());
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
        // Cache row shape data.
        std::vector<float> row_shape_values;
        row_shape_values.reserve(row.size());
        const size_type right_index_base = right_begin_local(k);
        for(typename std::vector<row_datum>::const_iterator row_it = row.begin(); row_it != row.end(); ++row_it)
          row_shape_values.push_back(right_.shape().data()[right_index_base + (row_it->first * right_stride_local_)]);

        // Iterate over the left-hand argument column (rows of the result)
        const size_type left_index_base = left_begin_local(k);
        const float threshold_k = TensorImpl_::shape().threshold() / float(k_);
        for(typename std::vector<col_datum>::const_iterator col_it = col.begin(); col_it != col.end(); ++col_it) {
          // Compute the local, result-tile offset for the current result row.
          const size_type result_offset = col_it->first * proc_grid_.local_cols();

          // Get the shape data for col_it tile
          const float col_shape_value =
              left_.shape().data()[left_index_base + (col_it->first * left_stride_local_)];

          // Iterate over the right-hand argument row (columns of the result)
          for(typename std::vector<row_datum>::const_iterator row_it = row.begin(); row_it != row.end(); ++row_it) {
            // Filter trivial results
            if((col_shape_value * row_shape_values[row_it - row.begin()]) < threshold_k)
              continue;

            task->inc();
            reduce_tasks_[result_offset + row_it->first].add(col_it->second, row_it->second, task);
          }
        }
      }
#endif // TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER

      class FinalizeTask : public madness::TaskInterface {
      private:
        std::shared_ptr<ContractionEvalImpl_> owner_; ///< The parent object for this task

      public:
        FinalizeTask(const std::shared_ptr<ContractionEvalImpl_>& owner) :
          madness::TaskInterface(1, madness::TaskAttributes::hipri()),
          owner_(owner)
        { }

        virtual ~FinalizeTask() { }

        virtual void run(const madness::TaskThreadEnv&) {
          owner_->finalize();
        }

      }; // class FinalizeTask

      class StepTask : public madness::TaskInterface {
      private:
        // Member variables
        std::shared_ptr<ContractionEvalImpl_> owner_;
        size_type k_;
        FinalizeTask* finalize_task_;
        StepTask* next_step_task_;

        /// Construct the task for the next step
        StepTask(StepTask* const previous, const int ndep) :
          madness::TaskInterface(ndep, madness::TaskAttributes::hipri()),
          owner_(previous->owner_),
          k_(0ul),
          finalize_task_(previous->finalize_task_),
          next_step_task_(NULL)
        { }

      public:

        StepTask(const std::shared_ptr<ContractionEvalImpl_>& owner) :
          madness::TaskInterface(madness::TaskAttributes::hipri()),
          owner_(owner),
          k_(0ul),
          finalize_task_(new FinalizeTask(owner)),
          next_step_task_(new StepTask(this, 1))
        {
          owner_->get_world().taskq.add(next_step_task_);
          owner_->get_world().taskq.add(finalize_task_);
        }

        virtual ~StepTask() { }

        // Initialize member variables
        StepTask* initialize(const size_type k) {
          k_ = k;
          StepTask* step_task = NULL;
          if(k < owner_->k_) {
            next_step_task_ = step_task = new StepTask(this, 2);
            owner_->get_world().taskq.add(step_task);
          } else {
            finalize_task_->notify();
          }
          this->notify();
          return step_task;
        }

        virtual void run(const madness::TaskThreadEnv&) {
          // Search for the next k to be processed
          if(k_ < owner_->k_) {

            k_ = owner_->next_k(owner_, k_);

            if(k_ < owner_->k_) {
              finalize_task_->inc();

              // Submit the next step task
              StepTask* const next_next_step_task = next_step_task_->initialize(k_ + 1ul);
              next_step_task_ = NULL;

              // Broadcast row and column
              std::vector<col_datum> col;
              owner_->bcast_col(k_, col);
              std::vector<row_datum> row;
              owner_->bcast_row(k_, row);
              std::cout << "col(" << col.size() << ")  row(" << row.size() << ")\n";

              // Submit tasks for the contraction of col and row tiles.
              owner_->template contract<shape_type>(k_, col, row, next_next_step_task);

              // Notify
              if(next_next_step_task)
                next_next_step_task->notify();
              finalize_task_->notify();


            } else {
              finalize_task_->notify();
              if(next_step_task_) {
                next_step_task_->k_ = std::numeric_limits<size_type>::max();
                next_step_task_->notify();
              }
            }
          }
        }

      }; // class StepTask

    public:

      /// Constructor

      /// \param left The left-hand argument evaluator
      /// \param right The right-hand argument evaluator
      /// \param world The world where this evaluator will live
      /// \param trange The tiled range of the result tensor
      /// \param shape The shape of the result tensor
      /// \param pmap The process map for the result tensor
      /// \param perm The permutation that will be applied to tiles and the
      /// coordinate index after contraction of the result tile
      /// \param op The operation that will be used to contract tile pairs
      /// \param k The number of tiles in the inner dimension
      /// \param proc_grid The process grid that defines the layout of the tiles
      /// during the contraction evaluation
      /// \note The trange, shape, and pmap are assumed to be in the final,
      /// permuted, state for the result.
      ContractionEvalImpl(const left_type& left, const right_type& right,
          madness::World& world, const trange_type trange, const shape_type& shape,
          const std::shared_ptr<pmap_interface>& pmap, const Permutation& perm,
          const op_type& op, const size_type k, const ProcGrid& proc_grid) :
        DistEvalImpl_(world, trange, shape, pmap, perm),
        left_(left), right_(right), op_(op),
        row_group_(), col_group_(),
        k_(k), proc_grid_(proc_grid),
        reduce_tasks_(NULL),
        left_start_local_(proc_grid_.rank_row() * k),
        left_end_(left.size()),
        left_stride_(k),
        left_stride_local_(proc_grid.proc_rows() * k),
        right_stride_(1ul),
        right_stride_local_(proc_grid.proc_cols())
      { }

      virtual ~ContractionEvalImpl() { }

    private:

      /// Evaluate the tiles of this tensor

      /// This function will evaluate the children of this distributed evaluator
      /// and evaluate the tiles for this distributed evaluator. It will block
      /// until the tasks for the children are evaluated (not for the tasks of
      /// this object).
      /// \param pimpl A shared pointer to this object
      /// \return The number of tiles that will be set by this process
      virtual int internal_eval(const std::shared_ptr<DistEvalImpl_>& pimpl) {
        // Convert pimpl to this object type so it can be used in tasks
        std::shared_ptr<ContractionEvalImpl_> self =
            std::static_pointer_cast<ContractionEvalImpl_>(pimpl);

        // Start evaluate child tensors
        left_.eval();
        right_.eval();

        size_type tile_count = 0ul;
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
          ReducePairTask<op_type>* reduce_task = reduce_tasks_;
          for(size_type row = proc_grid_.rank_row(); row < proc_grid_.rows(); row += proc_grid_.proc_rows()) {
            const size_type row_start = row * proc_grid_.cols();
            for(size_type col = proc_grid_.rank_col(); col < proc_grid_.cols(); col += proc_grid_.proc_cols(), ++reduce_task) {
              // Construct non-zero reduce tasks
              if(! TensorImpl_::is_zero(DistEvalImpl_::perm_index(row_start + col))) {
                new(reduce_task) ReducePairTask<op_type>(TensorImpl_::get_world(), op_);
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
    /// \tparam LeftTile Tile type of the left-hand argument
    /// \tparam RightTile Tile type of the right-hand argument
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

      // Construct a vector TiledRange1 objects from the left- and right-hand
      // arguments that will be used to construct the result TiledTange. Also,
      // compute the fused outer dimension sizes, number of tiles and elements,
      // for the contraction.
      typename impl_type::trange_type::Ranges ranges(op.result_rank());
      std::size_t M = 1ul, m = 1ul, N = 1ul, n = 1ul;
      std::size_t pi = 0ul;
      for(unsigned int i = 0ul; i < left_middle; ++i) {
        ranges[(perm.dim() > 0ul ? perm[pi++] : pi++)] = left.trange().data()[i];
        M *= left.range().size()[i];
        m *= left.trange().elements().size()[i];
      }
      for(std::size_t i = num_contract_ranks; i < right_end; ++i) {
        ranges[(perm.dim() > 0ul ? perm[pi++] : pi++)] = right.trange().data()[i];
        N *= right.range().size()[i];
        n *= right.trange().elements().size()[i];
      }

      // Compute the number of tiles in the inner dimension.
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
