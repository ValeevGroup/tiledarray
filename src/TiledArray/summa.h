#ifndef TILEDARRAY_SUMMA_H__INCLUDED
#define TILEDARRAY_SUMMA_H__INCLUDED

#include <TiledArray/contraction_tensor_impl.h>
#include <iterator>
#include <world/world.h>
#include <vector>

namespace TiledArray {
  namespace expressions {

    /// Scalable Universal Matrix Multiplication Algorithm (SUMMA)

    /// This algorithm is used to contract dense tensor. The arguments are
    /// permuted such that the outer and inner indices are fused (outer first,
    /// inner second), so the contraction may be evaluated with a standard matrix
    /// multiplication. This algorithm is described in:
    /// Van De Geijn, R. A.; Watts, J. Concurrency Practice and Experience 1997, 9, 255-274.
    /// This
    /// \tparam Left The left-hand-arguement type
    /// \tparam Right The right-hand-argument type
    template <typename Left, typename Right>
    class Summa : public madness::WorldObject<Summa<Left, Right> >, public ContractionTensorImpl<Left, Right> {
    protected:
      typedef madness::WorldObject<Summa<Left, Right> > WorldObject_; ///< Madness world object base class
      typedef ContractionTensorImpl<Left, Right> ContractionTensorImpl_;
      typedef typename ContractionTensorImpl_::TensorExpressionImpl_ TensorExpressionImpl_;

      // import functions from world object
      using WorldObject_::task;
      using WorldObject_::get_world;

    public:
      typedef Summa<Left, Right> Summa_; ///< This object type
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

      /// Datum type for
      typedef std::pair<size_type, madness::Future<right_value_type> > row_datum;
      typedef std::pair<size_type, madness::Future<left_value_type> > col_datum;
      typedef std::shared_ptr<value_type> value_ptr;
      typedef madness::Future<value_ptr> future_value_ptr;
      typedef std::pair<size_type, future_value_ptr > result_datum;

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

    private:
      // Not allowed
      Summa(const Summa_&);
      Summa_& operator=(const Summa_&);

      /// Broadcast a tile to child nodes within group

      /// This function will broadcast tiles from
      template <typename Handler, typename Value>
      madness::Void bcast(Handler handler, const size_type i, const Value& value,
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

        return madness::None;
      }

      /// Spawn broadcast task for tile \c i with \c value

      /// If the future \c value has been set, two remote broadcast tasks will
      /// be spawned on the child nodes. Otherwise a local task is spawned that
      /// will broadcast the tile to the child nodes when the tile has been set.
      /// \tparam Handler The type of the remote task broadcast handler function
      /// \tparam Value The value type of the tile to be broadcast
      /// \param handler The remote task broadcast handler function
      /// \param i The index of the tile being broadcast
      /// \param value The tile value being broadcast
      /// \param group The broadcast group
      /// \param rank The rank of this process in group
      template <typename Handler, typename Value>
      void bcast_task(Handler handler, const size_type i, const madness::Future<Value>& value,
          const std::vector<ProcessID>& group, const ProcessID rank) {
        if(value.probe())
          bcast(handler, i, value, group, rank, rank);
        else
          task(rank_, & Summa_::template bcast<Handler, Value>, handler, i, value,
              group, rank, rank);
      }

      /// Task function used for broadcasting tiles along the row

      /// \param i The tile index
      /// \param group_rank The rank of this node within the group
      /// \param group_root The broadcast group root node
      madness::Void bcast_row_handler(const size_type i, const left_value_type& value,
          const ProcessID group_rank, const ProcessID group_root)
      {
        bcast(& Summa_::bcast_row_handler, i, value, row_group_, group_rank, group_root);

        // Copy tile into local cache
        typename left_container::accessor acc;
        const bool erase_cache = ! left_cache_.insert(acc, i);
        madness::Future<left_value_type> tile = acc->second;
        acc.release();

        // Set the local future with the broadcast value
        tile.set(value); // Move

        // If the local future is already present, the cached value is not needed
        if(erase_cache)
          left_cache_.erase(i);

        return madness::None;
      }

      /// Task function used for broadcasting tiles along the column

      /// \param i The tile index
      /// \param group_rank The rank of this node within the group
      /// \param group_root The broadcast group root node
      madness::Void bcast_col_handler(const size_type i, const right_value_type& value,
          const ProcessID group_rank, const ProcessID group_root)
      {
        bcast(& Summa_::bcast_col_handler, i, value, col_group_, group_rank, group_root);

        // Copy tile into local cache
        typename right_container::accessor acc;
        const bool erase_cache = ! right_cache_.insert(acc, i);
        madness::Future<right_value_type> tile = acc->second;
        acc.release();

        // Set the local future with the broadcast value
        tile.set(value); // Move

        // If the local future is already present, the cached value is not needed
        if(erase_cache)
          right_cache_.erase(i);

        return madness::None;
      }

      /// Task function for broadcasting the k-th column of the left tensor argument

      /// This function will construct a task that broadcasts the k-th column of
      /// the left tensor argument and return a vector of futures to the local
      /// elements of the k-th column.  This task must be run on all nodes
      /// for each k.
      /// \param k The column to be broadcast
      /// \return A vector that contains futures to k-th column tiles
      std::vector<col_datum> bcast_column_task(const size_type k) {
        // Construct the result column vector
        std::vector<col_datum> col;
        col.reserve(local_rows_);

        // Iterate over local rows of the k-th column of the left argument tensor
        size_type i = rank_row_ * k_ + k;
        const size_type step = proc_rows_ * k_;
        if(ContractionTensorImpl_::left().is_local(i)) {
          for(; i < mk_; i += step) {
            // Take the tile's local copy and add it to the column vector
            col.push_back(col_datum(i, ContractionTensorImpl_::left().move(i)));

            // Broadcast the tile to all nodes in the row
            bcast_task(& Summa_::bcast_row_handler, i, col.back().second, row_group_, rank_col_);
          }
        } else {
          for(; i < mk_; i += step) {
            // Insert a future into the cache as a placeholder for the broadcast tile.
            typename left_container::const_accessor acc;
            const bool erase_cache = ! left_cache_.insert(acc, i);
            madness::Future<left_value_type> tile = acc->second;
            acc.release();

            // Add tile to column vector
            col.push_back(col_datum(i, tile));

            // If the local future is already present, the cached value is not needed
            if(erase_cache)
              left_cache_.erase(i);
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
      /// \return A vector that contains futures to k-th column tiles
      std::vector<row_datum> bcast_row_task(const size_type k) {
        // Construct the result row vector
        std::vector<row_datum> row;
        row.reserve(local_cols_);

        // Iterate over local columns of the k-th row of the right argument tensor
        size_type i = k * n_ + rank_col_;
        const size_type end = (k + 1) * n_;
        if(ContractionTensorImpl_::right().is_local(i)) {
          for(; i < end; i += proc_cols_) {
            // Take the tile's local copy and add it to the row vector
            row.push_back(row_datum(i, ContractionTensorImpl_::right().move(i)));

            // Broadcast the tile to all nodes in the column
            bcast_task(& Summa_::bcast_col_handler, i, row.back().second, col_group_, rank_row_);
          }

        } else {
          for(; i < end; i += proc_cols_) {
            // Insert a future into the cache as a placeholder for the broadcast tile.
            typename right_container::const_accessor acc;
            const bool erase_cache = ! right_cache_.insert(acc, i);
            madness::Future<right_value_type> tile = acc->second;
            acc.release();

            // Add tile to row vector
            row.push_back(row_datum(i, tile));

            if(erase_cache)
              right_cache_.erase(i); // Bcast data has arived, so erase cache
          }
        }

        return row;
      }

      /// Contract argument tiles and store the result in the given result tile

      /// This function contracts \c left and \c right , and adds the result to
      /// \c result.
      /// \param result A shared pointer to the result tile
      /// \param left The left tile to be contracted
      /// \param right The right tile to be contracted
      /// \return The shared pointer to the result.
      value_ptr contract(const value_ptr& result, const left_value_type& left,
          const right_value_type& right)
      {
        ContractionTensorImpl_::contract(*result, left, right);
        return result;
      }


      /// Broadcast task for rows or columns

      /// \tparam Func The member function of the owner function that will do the actual broadcasting.
      template <typename Func>
      class BcastTask : public madness::TaskInterface {
      public:
        typedef typename madness::detail::result_of<Func>::type result_type;
      private:
        Summa_* owner_;
        Func func_;
        const size_type k_;
        madness::Future<result_type> results_;

      public:
        BcastTask(Summa_* owner, Func func, size_type k) :
            madness::TaskInterface(madness::TaskAttributes::hipri()),
            owner_(owner), func_(func), k_(k), results_()
        { }

        virtual ~BcastTask() { }

        virtual void run(madness::World&) { results_.set((owner_->*func_)(k_)); }

        void add_dependency(future_value_ptr& f) {
          DependencyInterface::inc();
          f.register_callback(this);
        }

        const madness::Future<result_type>& result() const { return results_; }
      }; // class BcastTask

      /// Spawn broadcast tasks for column and row \c k

      /// Spawn two high priority tasks that will broadcast tiles needed for
      /// local tile contractions.
      /// \param k The column and row to broadcast
      /// \return A \c std::pair of futures that contain vectors of futures for the column and row tiles
      std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<row_datum> > >
      make_bcast_task(const size_type k, std::vector<result_datum>& results) const {
        typedef BcastTask<std::vector<col_datum> (Summa_::*)(size_type)> col_task_type;
        typedef BcastTask<std::vector<row_datum> (Summa_::*)(size_type)> row_task_type;

        // Return empty results if we are at the end of the contraction
        if(k >= k_)
          return std::make_pair(
              madness::Future<std::vector<col_datum> >(std::vector<col_datum>()),
              madness::Future<std::vector<row_datum> >(std::vector<row_datum>()));

        // Construct the row and column broadcast task
        col_task_type* col_task = new col_task_type(const_cast<Summa_*>(this), & Summa_::bcast_column_task, k);
        row_task_type* row_task = new row_task_type(const_cast<Summa_*>(this), & Summa_::bcast_row_task, k);

        // Add callbacks for dependencies.
        if(results.size()) {
          for(typename std::vector<result_datum>::iterator it = results.begin(); it != results.end(); ++it) {
            if(! it->second.probe()) {
              col_task->add_dependency(it->second);
              row_task->add_dependency(it->second);
            }
          }
        }

        // Get the broadcast task results
        std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<row_datum> > >
        bcast_results = std::make_pair(col_task->result(), row_task->result());

        // Spawn the broadcast tasks.
        get_world().taskq.add(col_task);
        get_world().taskq.add(row_task);

        return bcast_results;
      }

      /// Set the value of the result tiles

      /// The final location of the value is tile is determined by the owning
      /// tensor.
      /// \param i The ordinal index of the result tile
      /// \param ptr A shared pointer to the tile value
      madness::Void set_value(const size_type i, const value_ptr& ptr) {
        TensorExpressionImpl_::set(i, *ptr);
        return madness::None;
      }

      /// Task function that is created for each iteration of the SUMMA algorithm

      /// This task function spawns the tasks for local contraction tiles,
      /// the next SUMMA iteration (k + 1), broadcast of the k + 2 column of
      /// the left argument tensor, and broadcast of the k + 2 row of the right
      /// argument. The next SUMMA iteration task depends on the results of the
      /// schedule contraction task. The broadcast tasks depend on all of the
      /// individule contraction tasks and the schedule contraction task.
      /// When \c k==k_ , the finalize task is spawned instead, which will assign
      /// the final value to the local tiles.
      /// \param k The SUMMA iteration step, in the range [0,k_].
      /// \param results A vector of futures of shared pointers to result tiles
      /// \param col_k0 The column tiles of the left argument tensor needed for
      /// SUMMA iterantion \c k
      /// \param row_k0 The row tiles of the right argument tensor needed for
      /// SUMMA iteration \c k
      /// \param col_row_k1 The column and row tiles for SUMMA iteration \c k+1
      /// for the left and right tensors respectively.
      /// \return madness::None
      madness::Void step(const size_type k, const std::vector<result_datum>& results,
          const std::vector<col_datum>& col_k0, const std::vector<row_datum>& row_k0,
          const std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<row_datum> > >& col_row_k1)
      {
        if(k < k_) {
          // Create results vector for the next iteration
          std::vector<result_datum> next_results;
          next_results.reserve(local_size_);

          // Schedule contraction tasks
          typename std::vector<result_datum>::const_iterator it = results.begin();
          for(typename std::vector<col_datum>::const_iterator col_it = col_k0.begin(); col_it != col_k0.end(); ++col_it)
            for(typename std::vector<row_datum>::const_iterator row_it = row_k0.begin(); row_it != row_k0.end(); ++row_it, ++it)
              next_results.push_back(result_datum(it->first, task(rank_,
                  & Summa_::contract, it->second, col_it->second, row_it->second)));

          // Spawn the task for the next iteration
          task(rank_, & Summa_::step, k + 1, next_results, col_row_k1.first,
              col_row_k1.second, make_bcast_task(k + 2,
              const_cast<std::vector<result_datum>&>(results)),
              madness::TaskAttributes::hipri());

        } else {
          /// Spawn tasks that will assign the final value of the tile
          typename std::vector<result_datum>::const_iterator it = results.begin();
          for(size_type i = rank_row_; i < m_; i += proc_rows_)
            for(size_type j = rank_col_; j < n_; j += proc_cols_, ++it) {
              const size_type index = i * n_ + j;
              if(it->second.probe())
                TensorExpressionImpl_::set(index, *(it->second.get()));
              else
                task(rank_, & Summa_::set_value, index, it->second, madness::TaskAttributes::hipri());
            }
        }

        return madness::None;
      }

    public:

      Summa(const Left& left, const Right& right) :
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

          // Fill the col group with all the processes in rank's col
          col_group_.reserve(proc_rows_);
          for(ProcessID col_first = rank_col_; col_first < proc_size_; col_first += proc_cols_)
            col_group_.push_back(col_first);
        }

        WorldObject_::process_pending();
      }

      virtual ~Summa() { }

    private:

      virtual void eval_tiles() {
        if(rank_ < proc_size_) {
          std::vector<result_datum> results;

          // Start broadcast tasks of column and row for k = 0
          std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<row_datum> > >
          col_row_k0 = make_bcast_task(0ul, results);

          // Start broadcast tasks of column and row for k = 1
          std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<row_datum> > >
          col_row_k1 = make_bcast_task(1ul, results);

          // Construct local result tiles
          // The tiles are initially empty, they will be initialized on first use.
          results.reserve(local_size_);
          for(size_type i = rank_row_; i < m_; i += proc_rows_)
            for(size_type j = rank_col_; j < n_; j += proc_cols_)
              results.push_back(result_datum(i * n_ + j,
                  future_value_ptr(value_ptr(new value_type()))));

          // Spawn the first step in the summa algorithm
          task(rank_, & Summa_::step, 0ul, results, col_row_k0.first, col_row_k0.second,
              col_row_k1, madness::TaskAttributes::hipri());
        }
      }

    }; // class Summa

  }  // namespace detail
}  // namespace TiledArray


namespace madness {
  namespace archive {

    template <typename Archive, typename T, typename R, typename A>
    struct ArchiveLoadImpl<Archive, std::shared_ptr<TiledArray::expressions::Tensor<T,R,A> > > {
      static inline void load(const Archive& ar, std::shared_ptr<TiledArray::expressions::Tensor<T,R,A> > &) {
        TA_EXCEPTION("Serialization of shared_ptr not supported.");
      }
    };

    template <typename Archive, typename T, typename R, typename A>
    struct ArchiveStoreImpl<Archive,std::shared_ptr<TiledArray::expressions::Tensor<T,R,A> > > {
      static inline void store(const Archive& ar, const std::shared_ptr<TiledArray::expressions::Tensor<T,R,A> >&) {
        TA_EXCEPTION("Serialization of shared_ptr not supported.");
      }
    };

  }  // namespace archive
}  // namespace madness

#endif // TILEDARRAY_SUMMA_H__INCLUDED

