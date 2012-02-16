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
      madness::Future<value_type> submit() {
        TA_ASSERT(pimpl_);
        // Get the result before submitting calling dec(), otherwise the task
        // could run and be deleted before we are done here.
        madness::Future<value_type> result = pimpl_->result();
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


    template <typename Op>
    class ReducePairTaskImpl : public madness::TaskInterface {
    public:
      typedef typename Op::result_type result_type;
      typedef typename Op::first_argument_type first_argument_type;
      typedef typename Op::second_argument_type second_argument_type;
      typedef ReducePairTaskImpl<Op> ReducePairTaskImpl_;

    private:

      class ReducePair : public madness::CallbackInterface {
      public:

        template <typename Left, typename Right>
        ReducePair(ReducePairTaskImpl_* parent, const Left& left, const Right& right) :
            parent_(parent), left_(left), right_(right)
        {
          TA_ASSERT(parent_);
          ref_count_ = 0;
          if((register_callback(left_) + register_callback(right_)) == 0u)
            parent_->ready(this);
        }

        virtual ~ReducePair() { }

        virtual void notify() {
          if((--ref_count_) == 0) {
            TA_ASSERT(left_.probe());
            TA_ASSERT(right_.probe());
            parent_->ready(this);
          }
        }

        const first_argument_type& left() const { return left_.get(); }

        const second_argument_type& right() const { return right_.get(); }

      private:

        template <typename T>
        unsigned int register_callback(madness::Future<T>& f) {
          if(! f.probe()) {
            ref_count_++;
            f.register_callback(this);
            return 1u;
          }
          return 0u;
        }

        ReducePairTaskImpl_* parent_;
        madness::AtomicInt ref_count_;
        madness::Future<first_argument_type> left_;
        madness::Future<second_argument_type> right_;
      }; // class ReducePair


      madness::World& world_;
      Op op_;
      std::shared_ptr<result_type> ready_result_;
      ReducePair* ready_pair_;
      madness::Future<result_type> result_;
      madness::Spinlock lock_;

    public:

      ReducePairTaskImpl(madness::World& world, Op op) :
          madness::TaskInterface(1, madness::TaskAttributes::hipri()),
          world_(world), op_(op), ready_result_(new result_type(op())),
          ready_pair_(NULL), result_(), lock_()
      { }

      virtual void run(madness::World&) {
        TA_ASSERT(ready_result_);
        result_.set(*ready_result_);
      }

      template <typename Left, typename Right>
      ReducePair* add(const Left& left, const Right& right) {
        inc();
        return new ReducePair(this, left, right);
      }

      void ready(ReducePair* pair) {
        TA_ASSERT(pair);
        lock_.lock();
        if(ready_result_) {
          std::shared_ptr<result_type> ready_result = ready_result_;
          ready_result_.reset();
          lock_.unlock();
          TA_ASSERT(ready_result);
          world_.taskq.add(*this, & ReducePairTaskImpl::reduce_result_pair,
              ready_result, pair, madness::TaskAttributes::hipri());
        } else if(ready_pair_) {
          ReducePair* ready_pair = ready_pair_;
          ready_pair_ = NULL;
          lock_.unlock();
          TA_ASSERT(ready_pair);
          world_.taskq.add(*this, & ReducePairTaskImpl::reduce_pair_pair,
              pair, ready_pair, madness::TaskAttributes::hipri());
        } else {
          ready_pair_ = pair;
          lock_.unlock();
        }
      }

      const madness::Future<result_type>& result() const { return result_; }

    private:

      void reduce(std::shared_ptr<result_type>& result) {
        while(result) {
          lock_.lock();
          if(ready_pair_) {
            ReducePair* pair = ready_pair_;
            ready_pair_ = NULL;
            lock_.unlock();
            op_(*result, pair->left(), pair->right());
            delete pair;
            this->dec();
          } else if(ready_result_) {
            std::shared_ptr<result_type> arg = ready_result_;
            ready_result_.reset();
            lock_.unlock();
            op_(*result, *arg);
            arg.reset();
          } else {
            ready_result_ = result;
            result.reset();
            lock_.unlock();
          }
        }
      }

      madness::Void reduce_result_pair(std::shared_ptr<result_type> result, const ReducePair* pair) {
        op_(*result, pair->left(), pair->right());
        delete pair;
        reduce(result);
        this->dec();
        return madness::None;
      }

      madness::Void reduce_pair_pair(const ReducePair* pair1, const ReducePair* pair2) {
        std::shared_ptr<result_type> result(new result_type(op_(pair1->left(),
            pair1->right(), pair2->left(), pair2->right())));
        delete pair1;
        delete pair2;
        reduce(result);
        this->dec();
        this->dec();
        return madness::None;
      }
    }; // class ReducePairTask

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
    template <typename Op>
    class ReducePairTask {
    public:

      typedef typename Op::result_type result_type;
      typedef typename Op::first_argument_type first_argument_type;
      typedef typename Op::second_argument_type second_argument_type;
      typedef ReducePairTask<Op> ReducePairTask_;

    private:
      madness::World& world_; ///< The world that owns the task queue.
      ReducePairTaskImpl<Op>* pimpl_; ///< The reduction task object.
      std::size_t count_;

      // Copy not allowed.
      ReducePairTask(const ReducePairTask_&);
      ReducePairTask_& operator=(const ReducePairTask_&);
    public:


      ReducePairTask(madness::World& world, const Op& op = Op()) :
        world_(world), pimpl_(new ReducePairTaskImpl<Op>(world, op)), count_(0ul)
      { }

      /// Destructor

      /// If the reduction has not been submitted or \c destroy() has not been
      /// called, it well be submitted when the the destructor is called.
      ~ReducePairTask() { if(pimpl_) submit(); }

      /// Add an element to the reduction

      /// \c value may be of type \c value_type (or \c T ), \c madness::Future<value_type> ,
      /// or \c madness::RemoteReference<madness::FutureImpl<value_type>> .
      /// \tparam Value The type of the object that will be reduced
      /// \param value The object that will be reduced
      template <typename Left, typename Right>
      void add(const Left& left, const Right& right) {
        TA_ASSERT(pimpl_);
        pimpl_->add(left, right);
        ++count_;
      }

      /// Submit the reduction task to the task queue

      /// \return The result of the reduction
      /// \note After submitting the task, objects can no longer be added to the
      /// reduction.
      madness::Future<result_type> submit() {
        TA_ASSERT(pimpl_);

        madness::Future<result_type> result = pimpl_->result();

        if(count_ == 0ul) {
          pimpl_->run(world_);
          delete pimpl_;
        } else {
          // Get the result before submitting calling dec(), otherwise the task
          // could run and be deleted before we are done here.
          world_.taskq.add(pimpl_);
          pimpl_->dec(); // decrement the fake dependency so the task will run.
        }

        pimpl_ = NULL;
        return result;
      }

    }; // class ReduceTask

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_REDUCE_TASK_H__INCLUDED
