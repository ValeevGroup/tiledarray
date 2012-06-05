#ifndef TILEDARRAY_SUMMA_H__INCLUDED
#define TILEDARRAY_SUMMA_H__INCLUDED

#include <TiledArray/bcast_group.h>
#include <TiledArray/cyclic_pmap.h>
#include <iterator>
#include <world/world.h>
#include <vector>

namespace TiledArray {
  namespace summa {

    /// Base class for SUMMA

    /// This class handles construction of global iterations. Derived classes
    /// implement specific methods for dense and sparse tensors.
    /// This algorithm evaluates the equation:
    /// \f[
    /// C_{mn} = \sum_k A_{mk} B_{nk}
    /// \f]
    /// The processor phases for \b C and \b A are \c x by \c y
    template <typename Impl>
    class Summa : public madness::WorldObject<Summa<Impl> >, private NO_DEFAULTS {
    protected:
      typedef madness::WorldObject<Summa<Impl> > WorldObject_; ///< Madness world object base class

      // import functions from world object
      using WorldObject_::task;
      using WorldObject_::get_world;

    public:
      typedef Summa<Impl> Summa_; ///< This object type
      typedef typename Impl::size_type size_type; ///< size type
      typedef typename Impl::value_type value_type; ///< The result value type
      typedef typename Impl::left_tensor_type left_tensor_type; ///< The left tensor type
      typedef typename left_tensor_type::value_type left_value_type; /// The left tensor value type
      typedef typename Impl::right_tensor_type right_tensor_type; ///< The right tensor type
      typedef typename right_tensor_type::value_type right_value_type; ///< The right tensor value type
      typedef typename Impl::pmap_interface pmap_interface; ///< The process map interface type

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

    protected:
      const std::shared_ptr<Impl> pimpl_; ///< Shared pointer to the contraction result tensor
      const ProcessID rank_; ///< This process's rank
      const ProcessID size_; ///< Then number of processes
      const size_type m_; ///< Number of element rows in the result and left matrix
      const size_type n_; ///< Number of element columns in the result matrix and rows in the right argument matrix
      const size_type k_; ///< Number of element columns in the left and right argument matrices
      const size_type mk_; ///< Number of elements in left matrix
      const size_type nk_; ///< Number of elements in right matrix
      const size_type proc_cols_; ///< Number of columns in the result process map
      const size_type proc_rows_; ///< Number of rows in the result process map
      const size_type proc_size_; ///< Number of process in the process map. This may be
                         ///< less than the number of processes in world.
      const ProcessID rank_row_; ///< This node's row in the process map
      const ProcessID rank_col_; ///< This node's column in the process map
      const size_type local_rows_; ///< The number of local element rows
      const size_type local_cols_; ///< The number of local element columns
      std::vector<ProcessID> row_group_; ///< The group of processes included in this node's row
      std::vector<ProcessID> col_group_; ///< The group of processes included in this node's column
      left_container left_cache_; ///< Cache for left bcast tiles
      right_container right_cache_; ///< Cache for right bcast tiles

    private:
      // Not allowed
      Summa(const Summa<Impl>&);
      Summa_& operator=(const Summa_&);

      /// Get the child data for teh binary tree

      /// \param[out] child0 The first child of process \c rank
      /// \param[out] child1 The second child of process \c rank
      /// \param rank The rank of this process
      /// \param size The number of nodes in the tree
      /// \param root The root of the tree
      static void binary_tree(ProcessID& child0, ProcessID& child1,
          const ProcessID rank, const ProcessID size, const ProcessID root) {
        // Renumber processes so root has me=0
        int me = (rank + size - root) % size;

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
      }

      /// Task function used for broadcasting tiles along the row

      /// \param i The tile index
      /// \param group_rank The rank of this node within the group
      /// \param group_root The broadcast group root node
      madness::Void bcast_row_handler(const size_type i, const left_value_type& value,
          const ProcessID group_rank, const ProcessID group_root)
      {
        // Get the group child nodes
        ProcessID child0, child1;
        binary_tree(child0, child1, group_rank, proc_cols_, group_root);

        // Send the data to child nodes
        if(child0 != -1)
          task(row_group_[child0], & bcast_row_handler, i, value, child0, group_root);
        if(child1 != -1)
          task(row_group_[child1], & bcast_row_handler, i, value, child1, group_root);

        // Copy tile into local cache
        typename left_container::accessor acc;
        left_cache_.insert(acc, i);
        acc->second.set(value); // move

        return madness::None;
      }

      /// Task function used for broadcasting tiles along the column

      /// \param i The tile index
      /// \param group_rank The rank of this node within the group
      /// \param group_root The broadcast group root node
      madness::Void bcast_col_handler(const size_type i, const right_value_type& value,
          const ProcessID group_rank, const ProcessID group_root)
      {
        // Get the group child nodes
        ProcessID child0, child1;
        binary_tree(child0, child1, group_rank, proc_rows_, group_root);

        // Send the data to child nodes
        if(child0 != -1)
          task(col_group_[child0], & bcast_col_handler, i, value, child0, group_root);
        if(child1 != -1)
          task(col_group_[child1], & bcast_col_handler, i, value, child1, group_root);

        // Copy tile into local cache
        typename right_container::accessor acc;
        right_cache_.insert(acc, i);
        acc->second.set(value); // move

        return madness::None;
      }

      /// Task function for broadcasting the k-th column of the left tensor argument

      /// This function will broadcast and return a vector of futures to the k-th
      /// column of the left tensor argument. Only the tiles that are needed for
      /// local contractions are returned. This task must be run on all nodes
      /// for each k.
      /// \param k The column to be broadcast
      /// \return A vector that contains futures to k-th column tiles
      std::vector<col_datum> bcast_column_task(size_type k) {
        // Construct the result column vector
        std::vector<col_datum> col;
        col.reserve(local_rows_);

        // Iterate over local tiles of the k-th column of the left argument tensor
        const size_type step = k_ * proc_rows_;
        for(; k < mk_; k += step) {
          if(pimpl_->left().is_local(k)) {
            // Broadcast the local tile to all nodes in the row
            col.push_back(col_datum(k, pimpl_->left().move(k)));
            if(col.back().second.probe())
              // The data is ready so broadcast it now
              bcast_row_handler(k, col.back().second.get(), rank_row_, rank_row_);
            else
              // Schedule a task for data that is not ready
              task(rank_, bcast_row_handler, k, col.back().second,
                  rank_row_, rank_row_);
          } else {
            // Insert a future into the cache as a placeholder for the broadcast tile.
            typename left_container::const_accessor acc;
            left_cache_.insert(acc, k);
            col.push_back(col_datum(k, acc->second));
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
      std::vector<row_datum> bcast_row_task(size_type k) {
        // Construct the result row vector
        std::vector<row_datum> row;
        row.reserve(local_cols_);

        // Iterate over the k-th column of the right argument tensor
        const size_type step = k_ * proc_cols_;
        for(; k < nk_; k += step) {
          if(pimpl_->right().is_local(k)) {
            // Broadcast the local tile to all nodes in the row
            row.push_back(row_datum(k, pimpl_->right().move(k)));

            if(row.back().second.probe())
              // The data is ready so broadcast it now
              bcast_col_handler(k, row.back().second.get(), rank_col_, rank_col_);
            else
              // Schedule a task for data that is not ready
              task(rank_, bcast_col_handler, k, row.back().second,
                  rank_col_, rank_col_);
          } else {
            // Insert a future into the cache as a placeholder for the broadcast tile.
            typename right_container::const_accessor acc;
            right_cache_.insert(acc, k);
            row.push_back(row_datum(k, acc->second));
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
      const value_ptr& contract(const value_ptr& result, const left_value_type& left,
          const right_value_type& right)
      {
        pimpl_->contract(*result, left, right);
        return result;
      }


      /// Delayed erase callback object

      /// This callback object removes a tile future from the local cache once
      /// it has been set.
      /// \tparam Cache The container type that holds the cached data
      template <typename Cache>
      class DelayedErase : public madness::CallbackInterface {
        size_type index_; ///< The index of the tile to be reomved
        Cache& cache_; ///< A referece to the tile cache container
      public:
        /// Callback object constructor

        ///< \param index The index of the tile to be reomved
        ///< \param cache A referece to the tile cache container
        DelayedErase(size_type index, Cache& cache) : index_(index), cache_(cache) { }

        /// Virtual destructor
        virtual ~DelayedErase() { }

        /// On callback remove tile from cache and delete this object
        virtual void notify() { cache_.erase(index_); delete this; }
      }; // class DelayedErase

      template <typename Cont, typename Cache>
      void erase_cache(Cont& cont, Cache& cache) {
        for(typename Cont::iterator it = cont.begin(); it != cont.end(); ++it) {
          if(it->second.probe()) {
            // Once the future has been set, we can erase it from cache
            cache.erase(it->first);
          } else {
            // The future has not been set, so we need to create a callback that
            // will erase it from cache once it is set
            DelayedErase<Cache>* delayed_erase = new DelayedErase<Cache>(it->first, cache);
            try {
              it->second.register_callback(delayed_erase);
            } catch(...) {
              delete delayed_erase;
              throw;
            }
          }
        }
      }

      std::vector<future_value_ptr> schedule_contractions(const std::vector<future_value_ptr>& results,
          const std::vector<col_datum>& col, const std::vector<row_datum>& row)
      {
        std::vector<future_value_ptr> next_results;

        // Schedule contraction tasks
        typename std::vector<future_value_ptr>::iterator it = results.begin();
        for(typename std::vector<col_datum>::const_iterator col_it = col.begin(); col_it != col.end(); ++col_it) {
          for(typename std::vector<row_datum>::const_iterator row_it = row.begin(); row_it != row.end(); ++row_it, ++it) {
            next_results.push_back(task(rank_, & contract, *it, col_it->second, row_it->second));
          }
        }

        // Erase row and column from cache
        erase_cache(const_cast<std::vector<col_datum>&>(col), left_cache_);
        erase_cache(const_cast<std::vector<row_datum>&>(row), right_cache_);

        return next_results;
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

        virtual void run(madness::World&) { results_.set(owner_->*func_(k_)); }

        template <typename T>
        void add_dependency(madness::Future<T>& f) {
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
      make_bcast_task(const size_type k, std::vector<future_value_ptr>& results) const {
        typedef BcastTask<std::vector<col_datum> (*)(size_type k)> col_task_type;
        typedef BcastTask<std::vector<row_datum> (*)(size_type k)> row_task_type;

        // Return empty results if we are at the end of the contraction
        if(k >= k_)
          return std::make_pair(
              madness::Future<std::vector<col_datum> >(std::vector<col_datum>()),
              madness::Future<std::vector<row_datum> >(std::vector<row_datum>()));

        // Construct the row and column broadcast task
        col_task_type col_task = new col_task_type(this, & bcast_column_task, k);
        row_task_type row_task = new row_task_type(this, & bcast_row_task, k);

        // Add callbacks for dependencies.
        for(typename std::vector<future_value_ptr>::iterator it = results.begin(); it != results.end(); ++it) {
          if(! it->probe()) {
            col_task->add_dependency(*it);
            row_task->add_dependency(*it);
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
      madness::Void set_value(size_type i, const value_ptr& ptr) const {
        pimpl_->set(i, *ptr);
        return madness::None;
      }

      /// Task function that spawns the tasks for final assignment of the local tiles

      /// This task is spawned by the last SUMMA iteration.
      /// \param results A vector of futures to shared pointers to result tiles
      /// \return madness::None;
      madness::Void finalize(const std::vector<future_value_ptr>& results) const {

        // Allocate local result tiles for contraction
        typename std::vector<future_value_ptr>::const_iterator it = results.begin();
        for(size_type i = rank_row_; i < m_; i += proc_rows_)
          for(size_type j = rank_col_; j < n_; j += proc_cols_, ++it)
            task(rank_, & set_value, i * n_ + j, *it, madness::TaskAttributes::hipri());

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
      madness::Void step(const size_type k, const std::vector<future_value_ptr>& results,
          const std::vector<col_datum>& col_k0, const std::vector<row_datum>& row_k0,
          const std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<row_datum> > >& col_row_k1)
      {
        TA_ASSERT(k <= k_);

        if(k < k_) {
          // Spawn the task that will spawn contraction tasks.
          madness::Future<std::vector<future_value_ptr> > next_results =
              task(rank_, & schedule_contractions, results, col_k0,
              row_k0, madness::TaskAttributes::hipri());

          // Spawn the task for the next iteration
          task(rank_, & step, k + 1, next_results, col_row_k1.first,
              col_row_k1.second, make_bcast_task(k + 2, results),
              madness::TaskAttributes::hipri());
        } else {
          // Spawn the task that will assign the the tile values
          task(rank_, & finalize, results, madness::TaskAttributes::hipri());
        }

        return madness::None;
      }

    public:

      Summa(const std::shared_ptr<Impl>& pimpl, const size_type pipe_size) :
          WorldObject_(pimpl->get_world()),
          pimpl_(pimpl),
          rank_(get_world().rank()),
          size_(get_world().size()),
          n_(pimpl->right_outer()),
          m_(pimpl->left_outer()),
          k_(pimpl->left_inner()),
          mk_(m_ * k_),
          nk_(n_ * k_),
          proc_cols_(std::min(size_ / std::max(std::min<std::size_t>(std::sqrt(size_ * m_ / n_), size_), 1ul), n_)),
          proc_rows_(std::min(size_ / proc_cols_, m_)),
          proc_size_(proc_cols_ * proc_rows_),
          rank_row_((rank_ < proc_size_ ? rank_ / proc_cols_ : -1)),
          rank_col_((rank_ < proc_size_ ? rank_ % proc_cols_ : -1)),
          local_rows_((rank_ < proc_size_ ? (m_ / proc_rows_) + (m_ % proc_rows_ ? 1 : 0) : 0)),
          local_cols_((rank_ < proc_size_ ? (n_ / proc_cols_) + (n_ % proc_cols_ ? 1 : 0) : 0)),
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


      /// Factory function for the left argument process map

      /// \return A shared pointer that contains the left process map
      std::shared_ptr<pmap_interface> make_left_pmap() const {
        return std::shared_ptr<pmap_interface>(new TiledArray::detail::CyclicPmap(
            get_world(), m_, k_, proc_rows_, proc_cols_));
      }

      /// Factory function for the right argument process map

      /// \return A shared pointer that contains the right process map
      std::shared_ptr<pmap_interface> make_righ_pmap() const {
        return std::shared_ptr<pmap_interface>(new TiledArray::detail::CyclicPmap(
            get_world(), n_, k_, proc_cols_, proc_rows_));
      }

      void eval() {
        if(rank_ < proc_size_) {
          std::vector<future_value_ptr> results;

          // Start broadcast tasks of column and row for k[0]
          std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<col_datum> > >
          col_row_k0 = make_bcast_task(0ul, results);

          // Start broadcast tasks of column and row for k[1]
          std::pair<madness::Future<std::vector<col_datum> >, madness::Future<std::vector<col_datum> > >
          col_row_k1 = make_bcast_task(1ul, results);

          // Allocate local result tiles for contraction
          for(size_type i = rank_row_; i < m_; i += proc_rows_)
            for(size_type j = rank_col_; j < n_; j += proc_cols_)
              results.push_back(madness::Future<value_ptr>(
                  value_ptr(new value_type(pimpl_->trange().make_range(i * n_ + j)))));

          // Spawn the first step in the summa algorithm
          task(rank_, & step, 0ul, results, col_row_k0.first, col_row_k0.second,
              col_row_k1, madness::TaskAttributes::hipri());
        }
      }

    }; // class Summa

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_SUMMA_H__INCLUDED

