#ifndef TILEDARRAY_REDUCE_TASK_H__INCLUDED
#define TILEDARRAY_REDUCE_TASK_H__INCLUDED

#include <TiledArray/error.h>
#include <world/make_task.h>
#include <world/worldfut.h>
#include <world/worldrange.h>
#include <list>

namespace TiledArray {
  namespace detail {

    /// Reduction task implementation object

    /// This implementation object is stored in a shared pointer because the
    /// reduction task copies itself
    /// \tparam T The type to be reduced
    /// \tparam Op The reduction operation
    template <typename T, typename Op>
    class ReduceTaskImpl {
    public:
      typedef T value_type; ///< The type to be reduced
      typedef std::list<madness::Future<value_type> > container_type; ///< Container type that holds the object that will be reduced
      typedef madness::Range<typename container_type::iterator> range_type; ///< Iterator range type
      typedef Op op_type; ///< The reduction operation type

      /// Reduction object constructor

      /// \param world The world object
      /// \param op The reduction operation
      ReduceTaskImpl(madness::World& world, const op_type& op) :
          world_(world), list_(), op_(op)
      { }

      /// Add a future to the reduction list

      /// \param f A future of an object to be reduced.
      void add(madness::Future<value_type> f) { list_.push_back(f); }

      /// The number of objects to be reduced

      /// This is here for debugging purposes but it may be useful.
      /// \return The number of elements in the reduction
      std::size_t size() const { return list_.size(); }

      /// Iterator range generator

      /// \param chunk The chunck size used for the reduction
      /// \return A range object of the objects that will be reduced.
      range_type range(const int chunk) const {
        return range_type(list_.begin(), list_.end(), chunk);
      }

      /// Reduce to arguments

      /// \param value1 The first value to be reduced
      /// \param value2 The second value to be reduced
      /// \return The reduced value
      value_type reduce(const value_type& value1, const value_type& value2) const {
        return op_(value1, value2);
      }

      /// World object accessor

      /// \return A reference to the world
      madness::World& get_world() const { return world_; }

    private:
      madness::World& world_; ///< The world
      mutable container_type list_; ///< List of the objects to be reduced
      op_type op_; ///< The reduction operation

    }; // class ReduceTaskImpl

    /// Reduction task  object

    /// This object is used to accumulate objects for reduction, construct
    /// reduction tasks, and perform the reductions.
    /// \tparam T The type to be reduced
    /// \tparam Op The reduction operation
    template <typename T, typename Op>
    class ReduceTask {
    private:
      typedef ReduceTaskImpl<T, Op> impl_type; ///< Implementation type
      typedef ReduceTask<T, Op> ReduceTask_; ///< This object type

    public:
      typedef typename impl_type::value_type value_type; ///< The type to be reduced
      typedef typename impl_type::range_type range_type; ///< Iterator range type
      typedef typename impl_type::op_type op_type; ///< The reduction operation type

      typedef value_type result_type; ///< The functor result type

      /// Reduction object constructor

      /// \param world The world object
      /// \param op The reduction operation
      ReduceTask(madness::World& world, const op_type& op) :
          pimpl_(new impl_type(world, op))
      { }

      /// ReductionTask copy constructor

      /// Create a shallow copy of \c other
      /// \param other The reduction task to be copied
      ReduceTask(const ReduceTask_& other) :
          pimpl_(other.pimpl_)
      { }

      /// ReductionTask assignment operator

      /// Create a shallow copy of \c other
      /// \param other The reduction task to be copied
      /// \return A reference of this object
      ReduceTask_& operator=(const ReduceTask_& other) {
        pimpl_ = other.pimpl_;
        return *this;
      }

      /// Add a future to the reduction list

      /// \param f A future of an object to be reduced.
      void add(const madness::Future<value_type>& f) {
        pimpl_->add(f);
      }

      /// Add an object to the reduction list

      /// \param value A value to be reduced.
      void add(const value_type& value) {
        add(madness::Future<value_type>(value));
      }

      /// The number of objects to be reduced

      /// This is here for debugging purposes but it may be useful.
      /// \return The number of elements in the reduction
      std::size_t size() const { return pimpl_->size(); }

      /// Start the reduction

      /// \return A future to the result
      madness::Future<value_type> operator()() const {
        return make_task(pimpl_->range(8));
      }


      /// Reduce to arguments

      /// \param value1 The first value to be reduced
      /// \param value2 The second value to be reduced
      /// \return The reduced value
      value_type operator()(const value_type& value1, const value_type& value2) const {
        return pimpl_->reduce(value1, value2);
      }

      /// Do a binary tree reduction of \c range

      /// If the range is larger than chunk size (8), then it is split into two
      /// subtasks that may further split the range or evaluate the reduction.
      /// \param range The range to be reduced.
      /// \return The result of the reduction
      madness::Future<value_type> operator()(const range_type& range) const {
        if (range.size() <= range.get_chunksize()) {
          value_type result = value_type();
          for(typename range_type::iterator it = range.begin(); it != range.end(); ++it) {
            TA_ASSERT(it->probe());
            result = (*this)(result, *it);
          }
          return madness::Future<value_type>(result);
        } else {
          range_type left = range;
          range_type right(left, madness::Split());

          madness::Future<value_type> left_red = make_task(left);
          madness::Future<value_type> right_red = make_task(right);
          return pimpl_->get_world().taskq.add(madness::make_task(*this, left_red,
              right_red, madness::TaskAttributes::hipri()));
        }
      }

    private:

      /// Make a reduction task for the given range

      /// \param range The range to be reduced.
      /// \return The result of the reduction task.
      madness::Future<value_type> make_task(range_type range) const {
        madness::TaskFn<ReduceTask, range_type>* task = madness::make_task(*this,
            range, madness::TaskAttributes::hipri());

        if(range.size() <= range.get_chunksize()) {
          for(typename range_type::iterator it = range.begin(); it != range.end(); ++it) {
            if(! (it->probe())) {
                task->inc();
                it->register_callback(task);
            }
          }
        }

        return pimpl_->get_world().taskq.add(task);
      }

      std::shared_ptr<impl_type> pimpl_; ///< The reduction task implementation object

    }; // class ReduceTask

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_REDUCE_TASK_H__INCLUDED
