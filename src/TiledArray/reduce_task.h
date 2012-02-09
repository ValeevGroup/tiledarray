#ifndef TILEDARRAY_REDUCE_TASK_H__INCLUDED
#define TILEDARRAY_REDUCE_TASK_H__INCLUDED

#include <TiledArray/error.h>
#include <world/worldtask.h>
#include <world/make_task.h>

namespace TiledArray {
  namespace detail {

    /// Reduction task implementation object

    /// This is the actual task that is submitted to the task queue. This task
    /// Will generate subtasks that do the reduction operations.
    template <typename T, typename Op>
    class ReduceTaskImpl : public madness::TaskInterface {
    public:
      typedef ReduceTaskImpl<T, Op> ReduceTaskImpl_; ///< This object type
      typedef T value_type; ///< The type of objects that will be reduced
      typedef madness::Future<value_type> future; ///< Future to an object to be reduce

    private:

      // Copy not allowed
      ReduceTaskImpl(const ReduceTaskImpl<T,Op>&);
      ReduceTaskImpl<T,Op> operator=(const ReduceTaskImpl<T,Op>&);

      /// Hold the object that will be reduced.
      class ReduceObject : public madness::CallbackInterface {
      public:

        /// Construct the reduce object holder

        /// \c value may be of type \c value_type (or \c T ),
        /// \c madness::Future<value_type> , or
        /// \c madness::RemoteReference<madness::FutureImpl<value_type>> .
        /// \tparam Value The type of object to be reduced.
        /// \param value The object to be reduced
        /// \param parent The task that owns this reduction object
        template <typename Value>
        ReduceObject(const Value& value, ReduceTaskImpl_* parent) :
            future_(value), parent_(parent)
        {
          TA_ASSERT(parent_);
          if(! future_.probe()) {
            future_.register_callback(this); // The data is not ready; set a callback.
          } else {
            ReduceObject::notify(); // The data is ready; do callback.
          }
        }

        /// Virtual destructor
        virtual ~ReduceObject() { }

        /// Callback function

        /// This function is called when the data is ready to be reduced.
        virtual void notify() {
          TA_ASSERT(parent_);
          TA_ASSERT(future_.probe());
          parent_->ready(this);
        }


        /// Const data accessor

        /// \return A constant reference to the reduction data.
        const value_type& data() const {
          TA_ASSERT(future_.probe());
          return future_.get();
        }

        /// Data accessor

        /// \return A reference to the reduction data.
        value_type& data() {
          TA_ASSERT(future_.probe());
          return future_.get();
        }

      private:
        madness::Future<value_type> future_; ///< The data that will be reduced
        ReduceTaskImpl_* parent_; ///< A pointer to the task that owns this object
      }; // class ReduceObject

    public:

      /// Construct a reduction task

      /// \param world The world object that will handle the tasks.
      /// \param op The reduction operation
      /// \note The dependency counter is initially set to one. \c dec() needs
      /// to be called before the task will run.
      ReduceTaskImpl(madness::World& world, const Op& op) :
        madness::TaskInterface(1, madness::TaskAttributes::HIGHPRIORITY),
        world_(world), op_(op), result_(), ready_(NULL), lock_()
      { }

      virtual ~ReduceTaskImpl() { }

      /// Submit a reduction object for reduction when it is ready.
      void ready(ReduceObject* ro) {
        TA_ASSERT(ro);
        ReduceObject* next = NULL;
        lock_.lock(); // <<< BEGIN CRITICAL SECTION
        if(ready_) {
          // Get the reduce object that is ready
          next = const_cast<ReduceObject*>(ready_);
          ready_ = NULL;
        } else {
          // Store the reduce object until the next one is ready
          ready_ = ro;
          ro = NULL;
        }
        lock_.unlock(); // <<< END CRITICAL SECTION

        // If we have two reduce object ready, add a reduce task to the queue.
        // Otherwise, decrement the dependency counter since another reduction
        // has finished
        if(next)
          world_.taskq.add(*this, & ReduceTaskImpl_::reduce_op, next, ro,
              madness::TaskAttributes::hipri());
        else
          dec();
      }

      /// Add an element to the reduction

      /// The task dependency counter will be incremented for each addition.
      /// \c value may be of type \c value_type (or \c T ), \c madness::Future<value_type> ,
      /// or \c madness::RemoteReference<madness::FutureImpl<value_type>> .
      /// \tparam Value The type of the object that will be reduced
      /// \param value The object that will be reduced
      template <typename Value>
      void add(const Value& value) {
        inc();
        ReduceObject* ro = new ReduceObject(value, this);
        ro = NULL; // Orphan the pointer, but it can take care of itself.
      }

      /// The task result accessor

      /// \return A future to the reduction result.
      const madness::Future<value_type>& result() const { return result_; }

      /// Task run function

      /// This function just sets the result to the reduced value, and performs
      /// cleanup operations.
      virtual void run(madness::World&) {
        lock_.lock();
        ReduceObject* ready = const_cast<ReduceObject*>(ready_);
        ready_ = NULL;
        lock_.unlock();
        TA_ASSERT(ready);
        result_.set(ready->data());
        delete ready;
      }

    private:

      madness::Void reduce_op(ReduceObject* first, const ReduceObject* second) {
        TA_ASSERT(first);
        TA_ASSERT(second);
        TA_ASSERT(first != second);
        first->data() = op_(first->data(), second->data());
        delete second;
        first->notify();
        return madness::None;
      }

      madness::World& world_;
      Op op_;
      madness::Future<value_type> result_;
      volatile ReduceObject* ready_;
      madness::Spinlock lock_;
    }; // class ReduceTaskImpl

    /// Reduction task

    /// This task will reduce an arbitrary number of objects. This task is
    /// optimized for reduction of data that is the result of other tasks or
    /// remote data. Though it can handle data that is not stored in a future,
    /// it may not be the best choice. The objects are reduced as they become
    /// ready, which results in non-deterministic reduction order.
    /// This is theoretically be faster than a simple  binary tree reduction
    /// since the reduction tasks do not have to wait on any specific object to
    /// become ready for reduced. \n
    /// The reduction operation has the following form:
    /// \code
    /// first = op(first, second);
    /// \endcode
    /// where \c op is the reduction operation given to the constructor
    /// \tparam T The object type to be reduced
    /// \tparam Op The reduction operation type
    template <typename T, typename Op>
    class ReduceTask {
    private:
      // Copy not allowed.
      ReduceTask(const ReduceTask<T,Op>&);
      ReduceTask<T,Op>& operator=(const ReduceTask<T,Op>&);
    public:

      typedef T value_type; ///< The type that will be reduced

      ReduceTask(madness::World& world, const Op& op = Op()) :
        world_(world), pimpl_(new ReduceTaskImpl<T, Op>(world, op))
      { }

      /// Destructor

      /// If the reduction has not been submitted or \c destroy() has not been
      /// called, it well be submitted when the the destructor is called.
      ~ReduceTask() {
        if(pimpl_)
          submit();
      }

      /// Add an element to the reduction

      /// \c value may be of type \c value_type (or \c T ), \c madness::Future<value_type> ,
      /// or \c madness::RemoteReference<madness::FutureImpl<value_type>> .
      /// \tparam Value The type of the object that will be reduced
      /// \param value The object that will be reduced
      template <typename Value>
      void add(const Value& value) {
        TA_ASSERT(pimpl_);
        pimpl_->add(value);
      }

      template <typename InIter>
      void add(InIter first, InIter last) {
        TA_ASSERT(pimpl_);
        for(; first != last; ++first)
          pimpl_->add(*first);
      }

      /// Submit the reduction task to the task queue

      /// \return The result of the reduction
      /// \note After submitting the task, objects can no longer be added to the
      /// reduction.
      const madness::Future<value_type>& submit() {
        TA_ASSERT(pimpl_);
        // Get the result before submitting calling dec(), otherwise the task
        // could run and be deleted before we are done here.
        const madness::Future<value_type>& result = pimpl_->result();
        world_.taskq.add(pimpl_);
        pimpl_->dec(); // decrement the fake dependency so the task will run.
        pimpl_ = NULL;

        return result;
      }

      /// Destroy the reduce task without submitting it to the task queue.
      void destroy() {
        delete pimpl_;
      }

    private:
      madness::World& world_; ///< The world that owns the task queue.
      ReduceTaskImpl<T, Op>* pimpl_; ///< The reduction task object.
    }; // class ReduceTask

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_REDUCE_TASK_H__INCLUDED
