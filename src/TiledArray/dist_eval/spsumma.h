/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#ifndef TILEDARRAY_SPSUMMA_H__INCLUDED
#define TILEDARRAY_SPSUMMA_H__INCLUDED

#include <TiledArray/contraction_tensor_impl.h>
#include <TiledArray/reduce_task.h>

namespace TiledArray {
  namespace expressions {

    /// Scalable Universal Matrix Multiplication Algorithm (SUMMA)

    /// This algorithm is used to contract dense tensor. The arguments are
    /// permuted such that the outer and inner indices are fused such that a
    /// standard matrix multiplication algorithm can be used to contract the
    /// tensors. SUMMA is described in:
    /// Van De Geijn, R. A.; Watts, J. Concurrency Practice and Experience 1997, 9, 255-274.
    /// \tparam Left The left-hand-argument type
    /// \tparam Right The right-hand-argument type
    template <typename Left, typename Right>
    class SpSumma : public madness::WorldObject<SpSumma<Left, Right> >, public ContractionTensorImpl<Left, Right> {
    protected:
      typedef madness::WorldObject<SpSumma<Left, Right> > WorldObject_; ///< Madness world object base class
      typedef ContractionTensorImpl<Left, Right> ContractionTensorImpl_;
      typedef typename ContractionTensorImpl_::TensorExpressionImpl_ TensorExpressionImpl_;

      // import functions from world object
      using WorldObject_::task;
      using WorldObject_::get_world;

    public:
      typedef SpSumma<Left, Right> SpSumma_; ///< This object type
      typedef typename ContractionTensorImpl_::size_type size_type; ///< size type
      typedef typename ContractionTensorImpl_::value_type value_type; ///< The result value type
      typedef typename ContractionTensorImpl_::left_tensor_type left_tensor_type; ///< The left tensor type
      typedef typename ContractionTensorImpl_::left_value_type left_value_type; /// The left tensor value type
      typedef typename ContractionTensorImpl_::right_tensor_type right_tensor_type; ///< The right tensor type
      typedef typename ContractionTensorImpl_::right_value_type right_value_type; ///< The right tensor value type
      typedef typename ContractionTensorImpl_::pmap_interface pmap_interface; ///< The process map interface type

    private:
      /// The left tensor cache container type
      typedef madness::ConcurrentHashMap<size_type, madness::Future<left_value_type> > left_container;

      /// The right tensor cache container type
      typedef madness::ConcurrentHashMap<size_type, madness::Future<right_value_type> > right_container;


      /// Contraction and reduction task operation type
      typedef detail::ContractReduceOp<Left, Right> contract_reduce_op;

      /// Contraction and reduction task type
      typedef TiledArray::detail::ReducePairTask<contract_reduce_op> reduce_pair_task;

      /// Datum type for
      typedef std::pair<size_type, madness::Future<right_value_type> > row_datum;
      typedef std::pair<size_type, madness::Future<left_value_type> > col_datum;
      typedef std::pair<size_type, reduce_pair_task> result_datum;


    protected:

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

      std::vector<ProcessID> row_group_; ///< The group of processes included in this node's row
      std::vector<ProcessID> col_group_; ///< The group of processes included in this node's column
      left_container left_cache_; ///< Cache for left bcast tiles
      right_container right_cache_; ///< Cache for right bcast tiles
      std::vector<result_datum> results_; ///< Task object that will contract and reduce tiles

    private:
      // Not allowed
      SpSumma(const SpSumma_&);
      SpSumma_& operator=(const SpSumma_&);

      /// Broadcast a tile to child nodes within group

      /// This function will broadcast tiles from
      template <typename Handler, typename Value>
      void bcast(Handler handler, const size_type i, const Value& value,
          const std::vector<ProcessID>& group, const ProcessID rank, const ProcessID root)
      {
        const ProcessID size = group.size();

        // Get the group child nodes
        ProcessID child0, child1;

        // Renumber processes so root has me=0
        const int me = (rank + size - root) % size;

        // Left child
        child0 = (me << 1) + 1 + root;
        if((child0 >= size) && (child0 < (size + root)))
          child0 -= size;
        if(child0 >= size)
          child0 = -1;

        // Right child
        child1 = (me << 1) + 2 + root;
        if((child1 >= size) && (child1 < (size + root)))
          child1 -= size;
        if(child1 >= size)
          child1 = -1;

        // Send the data to child nodes
        if(child0 != -1)
          task(group[child0], handler, i, value, child0, root);
        if(child1 != -1)
          task(group[child1], handler, i, value, child1, root);
      }

      /// Spawn broadcast task for tile \c i with \c value

      /// If \c value has been set, two remote broadcast tasks will be spawned
      /// on the child nodes. Otherwise a local task is spawned that will
      /// broadcast the tile to the child nodes when the tile has been set.
      /// \tparam Handler The type of the remote task broadcast handler function
      /// \tparam Value The value type of the tile to be broadcast
      /// \param handler The remote task broadcast handler function
      /// \param i The index of the tile being broadcast
      /// \param value The tile value being broadcast
      /// \param group The broadcast group
      /// \param rank The rank of this process in group
      template <typename Handler, typename Value>
      void spawn_bcast_task(Handler handler, const size_type i, const madness::Future<Value>& value,
          const std::vector<ProcessID>& group, const ProcessID rank)
      {
        if(value.probe())
          bcast(handler, i, value, group, rank, rank);
        else
          task(rank_, & SpSumma_::template bcast<Handler, Value>, handler, i, value,
              group, rank, rank);
      }

      /// Task function used for broadcasting tiles along the row

      /// \param i The tile index
      /// \param group_rank The rank of this node within the group
      /// \param group_root The broadcast group root node
      void bcast_row_handler(const size_type i, left_value_type& value,
          const ProcessID group_rank, const ProcessID group_root)
      {
        // Broadcast this task to the next nodes in the tree
        bcast(& SpSumma_::bcast_row_handler, i, value, row_group_, group_rank, group_root);

        // Copy tile into local cache
        typename left_container::const_accessor acc;
        const bool erase_cache = ! left_cache_.insert(acc, i);
        madness::Future<left_value_type> tile = acc->second;

        // If the local future is already present, the cached value is not needed
        if(erase_cache)
          left_cache_.erase(acc);
        else
          acc.release();

        // Set the local future with the broadcast value
        tile.set(madness::move(value)); // Move
      }

      /// Task function used for broadcasting tiles along the column

      /// \param i The tile index
      /// \param group_rank The rank of this node within the group
      /// \param group_root The broadcast group root node
      void bcast_col_handler(const size_type i, right_value_type& value,
          const ProcessID group_rank, const ProcessID group_root)
      {
        // Broadcast this task to the next nodes in the tree
        bcast(& SpSumma_::bcast_col_handler, i, value, col_group_, group_rank, group_root);

        // Copy tile into local cache
        typename right_container::const_accessor acc;
        const bool erase_cache = ! right_cache_.insert(acc, i);
        madness::Future<right_value_type> tile = acc->second;

        // If the local future is already present, the cached value is not needed
        if(erase_cache)
          right_cache_.erase(acc);
        else
          acc.release();

        // Set the local future with the broadcast value
        tile.set(madness::move(value)); // Move
      }

      /// Broadcast task for rows or columns
      class BcastRowColTask : public madness::TaskInterface {
      private:
        SpSumma_* owner_;
        const size_type bcast_k_;
        std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<row_datum> > > results_;

        virtual void get_id(std::pair<void*,unsigned short>& id) const {
            return madness::PoolTaskInterface::make_id(id, *this);
        }

        /// Task function for broadcasting the k-th column of the left tensor argument

        /// This function will construct a task that broadcasts the k-th column of
        /// the left tensor argument and return a vector of futures to the local
        /// elements of the k-th column.  This task must be run on all nodes
        /// for each k.
        /// \param k The column to be broadcast
        std::vector<col_datum> bcast_column() {
          // Construct the result column vector
          std::vector<col_datum> col;
          col.reserve(owner_->local_rows_);

          // Iterate over local rows of the k-th column of the left argument tensor
          size_type i = owner_->rank_row_ * owner_->k_ + bcast_k_;
          const size_type step = owner_->proc_rows_ * owner_->k_;
          const size_type end = owner_->mk_;
          if(owner_->left().is_local(i)) {
            if(! owner_->left().get_pmap()->is_replicated()) {
              for(; i < end; i += step) {
                // Take the tile's local copy and add it to the column vector
                col.push_back(col_datum(i, owner_->left().move(i)));

                // Broadcast the tile to all nodes in the row
                owner_->spawn_bcast_task(& SpSumma_::bcast_row_handler, i,
                    col.back().second, owner_->row_group_, owner_->rank_col_);
              }
            } else {
              for(; i < end; i += step)
                // Take the tile's local copy and add it to the column vector
                col.push_back(col_datum(i, owner_->left().move(i)));
            }
          } else {
            for(; i < end; i += step) {
              // Insert a future into the cache as a placeholder for the broadcast tile.
              typename left_container::const_accessor acc;
              const bool erase_cache = ! owner_->left_cache_.insert(acc, i);
              madness::Future<left_value_type> tile = acc->second;

              // If the local future is already present, the cached value is not needed
              if(erase_cache)
                owner_->left_cache_.erase(acc);
              else
                acc.release();

              // Add tile to column vector
              col.push_back(col_datum(i, tile));
            }
          }

          return col;
        }

        /// Task function for broadcasting the k-th column of the right tensor argument

        /// This function will broadcast and return a vector of futures to the k-th
        /// column of the right tensor argument. Only the tiles that are needed for
        /// local contractions are returned. This task must be run on all nodes
        /// for each k.
        /// \param k The column to be broadcast
        std::vector<row_datum> bcast_row() {
          // Construct the result row vector
          std::vector<row_datum> row;
          row.reserve(owner_->local_cols_);

          // Iterate over local columns of the k-th row of the right argument tensor
          size_type i = bcast_k_ * owner_->n_ + owner_->rank_col_;
          const size_type end = (bcast_k_ + 1) * owner_->n_;
          if(owner_->right().is_local(i)) {
            if(! owner_->right().get_pmap()->is_replicated()) {
              for(; i < end; i += owner_->proc_cols_) {
                // Take the tile's local copy and add it to the row vector
                row.push_back(row_datum(i, owner_->right().move(i)));

                // Broadcast the tile to all nodes in the column
                owner_->spawn_bcast_task(& SpSumma_::bcast_col_handler, i,
                    row.back().second, owner_->col_group_, owner_->rank_row_);
              }
            } else {
              for(; i < end; i += owner_->proc_cols_)
                // Take the tile's local copy and add it to the row vector
                row.push_back(row_datum(i, owner_->right().move(i)));
            }
          } else {
            for(; i < end; i += owner_->proc_cols_) {
              // Insert a future into the cache as a placeholder for the broadcast tile.
              typename right_container::const_accessor acc;
              const bool erase_cache = ! owner_->right_cache_.insert(acc, i);
              madness::Future<right_value_type> tile = acc->second;

              if(erase_cache)
                owner_->right_cache_.erase(acc); // Bcast data has arived, so erase cache
              else
                acc.release();

              // Add tile to row vector
              row.push_back(row_datum(i, tile));

            }
          }

          return row;
        }

      public:
        BcastRowColTask(SpSumma_* owner, size_type k, const int ndep) :
            madness::TaskInterface(ndep, madness::TaskAttributes::hipri()),
            owner_(owner), bcast_k_(k), results_()
        { }

        virtual ~BcastRowColTask() { }

        virtual void run(const madness::TaskThreadEnv&) {
          results_.first.set(bcast_column());
          results_.second.set(bcast_row());
        }

        const std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<row_datum> > >&
        result() const { return results_; }
      }; // class BcastTask

      /// Spawn broadcast tasks for column and row \c k

      /// Spawn two high priority tasks that will broadcast tiles needed for
      /// local tile contractions.
      /// \param k The column and row to broadcast
      /// \return A broad cast task pointer
      BcastRowColTask* bcast_row_and_column(const size_type k, const int ndep = 0) const {
        return (k < k_ ? new BcastRowColTask(const_cast<SpSumma_*>(this), k, ndep) : NULL);
      }

      /// Task function that is created for each iteration of the SUMMA algorithm

      /// This task function spawns the tasks for local contraction tiles,
      /// the next SUMMA iteration (k + 1), broadcast of the k + 2 column of
      /// the left argument tensor, and broadcast of the k + 2 row of the right
      /// argument. The next SUMMA iteration task depends on the results of the
      /// schedule contraction task. The broadcast tasks depend on all of the
      /// individual contraction tasks and the schedule contraction task.
      /// When \c k==k_ , the finalize task is spawned instead, which will assign
      /// the final value to the local tiles.
      /// \param k The SUMMA iteration step, in the range [0,k_].
      /// \param results A vector of futures of shared pointers to result tiles
      /// \param col_k0 The column tiles of the left argument tensor needed for
      /// SUMMA iteration \c k
      /// \param row_k0 The row tiles of the right argument tensor needed for
      /// SUMMA iteration \c k
      /// \param col_row_k1 The column and row tiles for SUMMA iteration \c k+1
      /// for the left and right tensors respectively.
      /// \return madness::None
      void step(const size_type k,
          const std::vector<col_datum>& col_k0, const std::vector<row_datum>& row_k0,
          const std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<row_datum> > >& col_row_k1)
      {
        const bool assign = (k == (k_ - 1));

        BcastRowColTask* task_row_col_k2 = bcast_row_and_column(k + 2, local_size_);

        // Schedule contraction tasks
        typename std::vector<result_datum>::iterator it = results_.begin();
        for(typename std::vector<col_datum>::const_iterator col_it = col_k0.begin(); col_it != col_k0.end(); ++col_it)
          for(typename std::vector<row_datum>::const_iterator row_it = row_k0.begin(); row_it != row_k0.end(); ++row_it, ++it)
            it->second.add(col_it->second, row_it->second, task_row_col_k2);

        // Spawn the task for the next iteration
        if(assign) {
          // Signal the reduce task that all the reduction pairs have been added
          for(it = results_.begin(); it != results_.end(); ++it)
            it->second.submit();
          // Do some memory cleanup
          ContractionTensorImpl_::left().release();
          ContractionTensorImpl_::right().release();
        } else {
          if(task_row_col_k2) {
            task(rank_, & SpSumma_::step, k + 1, col_row_k1.first,
                col_row_k1.second, task_row_col_k2->result(),
                madness::TaskAttributes::hipri());
            get_world().taskq.add(task_row_col_k2);
          } else {
            task(rank_, & SpSumma_::step, k + 1, col_row_k1.first,
                col_row_k1.second, std::make_pair(
                    madness::Future<std::vector<col_datum> >(std::vector<col_datum>()),
                    madness::Future<std::vector<row_datum> >(std::vector<row_datum>())),
                madness::TaskAttributes::hipri());
          }
        }
      }

    public:

      SpSumma(const Left& left, const Right& right) :
          WorldObject_(left.get_world()),
          ContractionTensorImpl_(left, right),
          row_group_(),
          col_group_(),
          left_cache_(local_rows_ * k_),
          right_cache_(local_cols_ * k_)
      {
        if(rank_ < proc_size_) {
          // Fill the row group with all the processes in rank's row
          row_group_.reserve(proc_cols_);
          ProcessID row_first = rank_ - rank_col_;
          const ProcessID row_last = row_first + proc_cols_;
          for(; row_first < row_last; ++row_first)
            row_group_.push_back(row_first);

          // Fill the col group with all the processes in rank's column
          col_group_.reserve(proc_rows_);
          for(ProcessID col_first = rank_col_; col_first < proc_size_; col_first += proc_cols_)
            col_group_.push_back(col_first);
        }

        WorldObject_::process_pending();
      }

      /// Virtual destructor
      virtual ~SpSumma() { }

    private:

      class BcastColTask : public madness::TaskInterface {
      private:
        SpSumma_& owner_; ///< The object that owns this task
        const size_type k_; ///< The column to broadcast
        std::vector<col_datum>& col_; ///< The result column vector
        madness::TaskInterface* parent_; ///< The parent task

      public:
        BcastColTask(SpSumma_& owner, const size_type k, std::vector<col_datum>& col,
            std::vector<row_datum>& row, madness::TaskInterface* parent) :
          owner_(owner), k_(k), col_(col), row_(row), parent_(parent)
        {
        }

        virtual void run(const madness::TaskThreadEnv&) {
          if(parent_)
            parent_->notify();
        }

      }; // class BcastColTask

      class BcastRowTask : public madness::TaskInterface {
      private:
        SpSumma_& owner_; ///< The object that owns this task
        const size_type k_; ///< The row to broadcast
        std::vector<row_datum>& row_; ///< The result row vector
        madness::TaskInterface* parent_; ///< The parent task

      public:

        virtual void run(const madness::TaskThreadEnv&) {
          // Notify the parent task that the result vector has been set.
          if(parent_) {
            parent_->notify();
          }
        }
      }; // class BcastRowTask

      /// SpSumma master scheduling task

      /// This task will schedule all broadcast and computation scheduling tasks.
      class Scheduler : public madness::TaskInterface {
      private:
        SpSumma_& owner_; ///< The object that owns this task
        TiledArray::detail::Bitset<> left_local_mask_;
        TiledArray::detail::Bitset<> col_mask_;
        TiledArray::detail::Bitset<> right_local_mask_;
        TiledArray::detail::Bitset<> row_mask_;

      public:
        Scheduler(SpSumma_& owner, size_type& k, std::vector<col_datum>& col,
            std::vector<row_datum>& row, madness::TaskInterface* parent) :
          owner_(owner), k_(k), col_(col), row_(row), parent_(parent)
        {
          // Initialize left_local_mask_
          size_type i_row = 0;
          for(size_type i = owner_.rank_row_; i < owner_.m_; i += owner_.proc_rows_) {
            i_row = i * owner_.k_;
            for(size_type j = owner_.rank_col_; j < owner_.k_; j += owner_.proc_cols_) {
              left_local_mask_.set(i_row + j);
            }
          }

          // Initialize right_local_mask_
          for(size_type i = owner_.rank_row_; i < owner_.k_; i += owner_.proc_rows_) {
            i_row = i * owner_.k_;
            for(size_type j = owner_.rank_col_; j < owner_.n_; j += owner_.proc_cols_) {
              left_local_mask_.set(i_row + j);
            }
          }

          // Initialize col_mask_
          for(size_type ij = 0; ij < owner_.mk_; ij += onwer_.k_)
            col_mask_.set(ij);

          // Initialize row_mask_
          row_mask_.set_range(0, owner_.n_ - 1);
        }

        virtual void run(const madness::TaskThreadEnv&) {

          if(owner_.left().is_dense()) {
            if(owner_.right().is_dense()) {

            } else {
              TiledArray::detail::Bitset<> right_temp(owner_.right().size());

            }
          } else {
            TiledArray::detail::Bitset<> left_temp(owner_.left().size());
            if(owner_.right().is_dense()) {

            } else {
              TiledArray::detail::Bitset<> right_temp(owner_.right().size());

              for(size_type k = 0; k < owner_.k_; ++k) {
                left_temp = owner_.left().shape();
                left_temp &= col_mask;

                right_temp = owner_.right().shape();
                right_temp &= row_mask;

                // Shift mask for the next iteration
                col_mask <<= 1;
                row_mask <<= owner_.k_;
              }
            }
          }



        }

      }; // class Scheduler

      class Step : public madness::TaskInterface {
      private:
        SpSumma_& owner_;
        size_type k_;
        std::vector<col_datum> col_;
        std::vector<row_datum> row_;

      public:
        Step(SpSumma_& owner, const k current_k) :
          madness::TaskInterface(3),
          owner_(owner), k_(current_k + 1), col_(), row_()
        {
          owner_->get_world().taskq.add(new NextK(owner_, k_, col_, row_, this));
        }

        virtual void run(const madness::TaskThreadEnv&) {

        }
      }; // class Step

      virtual void eval_tiles() {
        if(rank_ < proc_size_) {

          // Start broadcast tasks of column and row for k = 0
          BcastRowColTask* task_col_row_k0 = bcast_row_and_column(0ul);
          std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<row_datum> > >
          col_row_k0 = task_col_row_k0->result();
          get_world().taskq.add(task_col_row_k0);

          // Start broadcast tasks of column and row for k = 1
          BcastRowColTask* task_col_row_k1 = bcast_row_and_column(1ul);
          std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<row_datum> > >
          col_row_k1 = task_col_row_k1->result();
          get_world().taskq.add(task_col_row_k1);

          // Construct a pair reduction object for each local tile
          results_.reserve(local_size_);
          for(size_type i = rank_row_; i < m_; i += proc_rows_)
            for(size_type j = rank_col_; j < n_; j += proc_cols_) {
              const size_type ij = i * n_ + j;
              results_.push_back(result_datum(ij,
                  reduce_pair_task(get_world(), contract_reduce_op(*this))));
              TensorExpressionImpl_::set(ij, results_.back().second.result());
            }

          // Spawn the first step in the algorithm
          task(rank_, & SpSumma_::step, 0ul, col_row_k0.first, col_row_k0.second,
              col_row_k1, madness::TaskAttributes::hipri());
        }
      }

    }; // class SpSumma

  }  // namespace detail
}  // namespace TiledArray


namespace madness {
  namespace archive {

    template <typename Archive, typename T, typename A>
    struct ArchiveLoadImpl<Archive, std::shared_ptr<TiledArray::expressions::Tensor<T,A> > > {
      static inline void load(const Archive& ar, std::shared_ptr<TiledArray::expressions::Tensor<T,A> > &) {
        TA_EXCEPTION("Serialization of shared_ptr not supported.");
      }
    };

    template <typename Archive, typename T, typename A>
    struct ArchiveStoreImpl<Archive,std::shared_ptr<TiledArray::expressions::Tensor<T,A> > > {
      static inline void store(const Archive& ar, const std::shared_ptr<TiledArray::expressions::Tensor<T,A> >&) {
        TA_EXCEPTION("Serialization of shared_ptr not supported.");
      }
    };

  }  // namespace archive
}  // namespace madness

#endif // TILEDARRAY_SPSUMMA_H__INCLUDED

