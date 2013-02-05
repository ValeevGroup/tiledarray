#ifndef TILEDARRAY_REDUCE_TASK_H__INCLUDED
#define TILEDARRAY_REDUCE_TASK_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/madness.h>

namespace TiledArray {
  namespace detail {


    template <typename Op>
    class ReduceTaskImpl : public madness::TaskInterface {
    public:
      typedef typename Op::result_type result_type;
      typedef typename Op::argument_type argument_type;
      typedef ReduceTaskImpl<Op> ReduceTaskImpl_;

    private:

      class ReduceObject : public madness::CallbackInterface {
      public:

        template <typename Arg>
        ReduceObject(ReduceTaskImpl_* parent, const Arg& arg) :
            parent_(parent), arg_(arg)
        {
          TA_ASSERT(parent_);
          arg_.register_callback(this);
        }

        virtual ~ReduceObject() { }

        virtual void notify() {
          TA_ASSERT(arg_.probe());
          parent_->ready(this);
        }

        const argument_type& arg() const { return arg_.get(); }

      private:

        ReduceTaskImpl_* parent_;
        madness::Future<argument_type> arg_;
      }; // class ReducePair


      madness::World& world_;
      Op op_;
      std::shared_ptr<result_type> ready_result_;
      volatile ReduceObject* ready_object_;
      madness::Future<result_type> result_;
      madness::Spinlock lock_;

      virtual void get_id(std::pair<void*,unsigned long>& id) const {
          return madness::PoolTaskInterface::make_id(id, *this);
      }

    public:

      ReduceTaskImpl(madness::World& world, Op op) :
          madness::TaskInterface(1, madness::TaskAttributes::hipri()),
          world_(world), op_(op), ready_result_(new result_type(op())),
          ready_object_(NULL), result_(), lock_()
      { }

      virtual ~ReduceTaskImpl() { }

      virtual void run(madness::World&) {
        TA_ASSERT(ready_result_);
        result_.set(*ready_result_);
      }

      template <typename Arg>
      ReduceObject* add(const Arg& arg) {
        inc();
        return new ReduceObject(this, arg);
      }

      void ready(ReduceObject* object) {
        TA_ASSERT(object);
        lock_.lock(); // <<< Begin critical section
        if(ready_result_) {
          std::shared_ptr<result_type> ready_result = ready_result_;
          ready_result_.reset();
          lock_.unlock(); // <<< End critical section
          TA_ASSERT(ready_result);
          world_.taskq.add(this, & ReduceTaskImpl::reduce_result_object,
              ready_result, object, madness::TaskAttributes::hipri());
        } else if(ready_object_) {
          ReduceObject* ready_object = const_cast<ReduceObject*>(ready_object_);
          ready_object_ = NULL;
          lock_.unlock(); // <<< End critical section
          TA_ASSERT(ready_object);
          world_.taskq.add(this, & ReduceTaskImpl::reduce_object_object,
              object, ready_object, madness::TaskAttributes::hipri());
        } else {
          ready_object_ = object;
          lock_.unlock(); // <<< End critical section
        }
      }

      const madness::Future<result_type>& result() const { return result_; }

    private:

      void reduce(std::shared_ptr<result_type>& result) {
        while(result) {
          lock_.lock(); // <<< Begin critical section
          if(ready_object_) {
            ReduceObject* ready_object = const_cast<ReduceObject*>(ready_object_);
            ready_object_ = NULL;
            lock_.unlock(); // <<< End critical section
            op_(*result, ready_object->arg());
            delete ready_object;
            this->dec();
          } else if(ready_result_) {
            std::shared_ptr<result_type> ready_result = ready_result_;
            ready_result_.reset();
            lock_.unlock(); // <<< End critical section
            op_(*result, *ready_result);
            ready_result.reset();
          } else {
            ready_result_ = result;
            result.reset();
            lock_.unlock(); // <<< End critical section
          }
        }
      }

      void reduce_result_object(std::shared_ptr<result_type> result, const ReduceObject* object) {
        op_(*result, object->arg());
        delete object;
        reduce(result);
        this->dec();
      }

      void reduce_object_object(const ReduceObject* object1, const ReduceObject* object2) {
        std::shared_ptr<result_type> result(new result_type(op_()));
        op_(*result, object1->arg(), object2->arg());
        delete object1;
        delete object2;
        reduce(result);
        this->dec();
        this->dec();
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
    class ReduceTask {
    public:

      typedef typename Op::result_type result_type;
      typedef typename Op::argument_type argument_type;
      typedef ReduceTask<Op> ReduceTask_;

    private:
      madness::World& world_; ///< The world that owns the task queue.
      ReduceTaskImpl<Op>* pimpl_; ///< The reduction task object.
      madness::AtomicInt count_;

      // Copy not allowed.
      ReduceTask(const ReduceTask_&);
      ReduceTask_& operator=(const ReduceTask_&);
    public:


      ReduceTask(madness::World& world, const Op& op = Op()) :
        world_(world), pimpl_(new ReduceTaskImpl<Op>(world, op))
      {
        count_ = 0;
      }

      /// Destructor

      /// If the reduction has not been submitted or \c destroy() has not been
      /// called, it well be submitted when the the destructor is called.
      ~ReduceTask() { if(pimpl_) submit(); }

      /// Add an element to the reduction

      /// \c value may be of type \c value_type (or \c T ), \c madness::Future<value_type> ,
      /// or \c madness::RemoteReference<madness::FutureImpl<value_type>> .
      /// \tparam Value The type of the object that will be reduced
      /// \param value The object that will be reduced
      template <typename Arg>
      int add(const Arg& arg) {
        TA_ASSERT(pimpl_);
        pimpl_->add(arg);
        return ++count_;
      }

      int count() const { return count_; }

      /// Submit the reduction task to the task queue

      /// \return The result of the reduction
      /// \note After submitting the task, objects can no longer be added to the
      /// reduction.
      madness::Future<result_type> submit() {
        TA_ASSERT(pimpl_);

        madness::Future<result_type> result = pimpl_->result();

        if(count_ == 0ul) {
          pimpl_->run(world_);
          pimpl_->dec();
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
          ref_count_ = 2;
          left_.register_callback(this);
          right_.register_callback(this);
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

        virtual void get_id(std::pair<void*,unsigned long>& id) const {
            return madness::PoolTaskInterface::make_id(id, *this);
        }

      private:

        ReducePairTaskImpl_* parent_;
        madness::AtomicInt ref_count_;
        madness::Future<first_argument_type> left_;
        madness::Future<second_argument_type> right_;
      }; // class ReducePair


      madness::World& world_;
      Op op_;
      std::shared_ptr<result_type> ready_result_;
      volatile ReducePair* ready_pair_;
      madness::Future<result_type> result_;
      madness::Spinlock lock_;

    public:

      ReducePairTaskImpl(madness::World& world, Op op) :
          madness::TaskInterface(1, madness::TaskAttributes::hipri()),
          world_(world), op_(op), ready_result_(new result_type(op())),
          ready_pair_(NULL), result_(), lock_()
      { }

      virtual ~ReducePairTaskImpl() { }

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
          world_.taskq.add(this, & ReducePairTaskImpl::reduce_result_pair,
              ready_result, pair, madness::TaskAttributes::hipri());
        } else if(ready_pair_) {
          ReducePair* ready_pair = const_cast<ReducePair*>(ready_pair_);
          ready_pair_ = NULL;
          lock_.unlock();
          TA_ASSERT(ready_pair);
          world_.taskq.add(this, & ReducePairTaskImpl::reduce_pair_pair,
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
            ReducePair* pair = const_cast<ReducePair*>(ready_pair_);
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

      void reduce_result_pair(std::shared_ptr<result_type> result, const ReducePair* pair) {
        op_(*result, pair->left(), pair->right());
        delete pair;
        reduce(result);
        this->dec();
      }

      void reduce_pair_pair(const ReducePair* pair1, const ReducePair* pair2) {
        std::shared_ptr<result_type> result(new result_type(op_(pair1->left(),
            pair1->right(), pair2->left(), pair2->right())));
        delete pair1;
        delete pair2;
        reduce(result);
        this->dec();
        this->dec();
      }
    }; // class ReducePairTaskImpl

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

      std::size_t count() const { return count_; }

      /// Submit the reduction task to the task queue

      /// \return The result of the reduction
      /// \note After submitting the task, objects can no longer be added to the
      /// reduction.
      madness::Future<result_type> submit() {
        TA_ASSERT(pimpl_);

        madness::Future<result_type> result = pimpl_->result();

        if(count_ == 0ul) {
          pimpl_->run(world_);
          pimpl_->dec();
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

    }; // class ReducePairTask

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_REDUCE_TASK_H__INCLUDED
