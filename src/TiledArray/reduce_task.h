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
        ReduceObject(ReduceTaskImpl_* parent, const Arg& arg, madness::CallbackInterface* callback) :
            parent_(parent), arg_(arg), callback_(callback)
        {
          TA_ASSERT(parent_);
          arg_.register_callback(this);
        }

        virtual ~ReduceObject() { }

        void do_callback() const {
          if(callback_)
            callback_->notify();
        }

        virtual void notify() {
          TA_ASSERT(arg_.probe());
          parent_->ready(this);
        }

        const argument_type& arg() const { return arg_.get(); }

      private:

        ReduceTaskImpl_* parent_;
        madness::Future<argument_type> arg_;
        madness::CallbackInterface* callback_;
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
      ReduceObject* add(const Arg& arg, madness::CallbackInterface* callback) {
        inc();
        return new ReduceObject(this, arg, callback);
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
            ready_object->do_callback();
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
        object->do_callback();
        delete object;
        reduce(result);
        this->dec();
      }

      void reduce_object_object(const ReduceObject* object1, const ReduceObject* object2) {
        std::shared_ptr<result_type> result(new result_type(op_()));
        op_(*result, object1->arg(), object2->arg());
        object1->do_callback();
        delete object1;
        object2->do_callback();
        delete object2;
        reduce(result);
        this->dec();
        this->dec();
      }
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
      int add(const Arg& arg, madness::CallbackInterface* callback = NULL) {
        TA_ASSERT(pimpl_);
        pimpl_->add(arg, callback);
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
        ReducePair(ReducePairTaskImpl_* parent, const Left& left, const Right& right, madness::CallbackInterface* callback) :
            parent_(parent), left_(left), right_(right), callback_(callback)
        {
          TA_ASSERT(parent_);
          ref_count_ = 2;
          left_.register_callback(this);
          right_.register_callback(this);
        }

        virtual ~ReducePair() { }

        void do_callback() const {
          if(callback_)
            callback_->notify();
        }

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
        madness::CallbackInterface* callback_;
      }; // class ReducePair


      madness::World& world_;
      Op op_;
      std::shared_ptr<result_type> ready_result_;
      volatile ReducePair* ready_pair_;
      madness::Future<result_type> result_;
      madness::Spinlock lock_;
      madness::AtomicInt count_;

    public:

      ReducePairTaskImpl(madness::World& world, Op op) :
          madness::TaskInterface(1, madness::TaskAttributes::hipri()),
          world_(world), op_(op), ready_result_(new result_type(op())),
          ready_pair_(NULL), result_(), lock_(), count_()
      {
        count_ = 0;
      }

      virtual ~ReducePairTaskImpl() { }

      virtual void run(const madness::TaskThreadEnv&) {
        TA_ASSERT(ready_result_);
        result_.set(*ready_result_);
      }

      template <typename Left, typename Right>
      ReducePair* add(const Left& left, const Right& right, madness::CallbackInterface* callback) {
        inc();
        count_++;
        return new ReducePair(this, left, right, callback);
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

      std::size_t count() const { return count_; }

      const madness::Future<result_type>& result() const { return result_; }

      madness::World& get_world() const { return world_; }

    private:

      void reduce(std::shared_ptr<result_type>& result) {
        while(result) {
          lock_.lock();
          if(ready_pair_) {
            ReducePair* pair = const_cast<ReducePair*>(ready_pair_);
            ready_pair_ = NULL;
            lock_.unlock();
            op_(*result, pair->left(), pair->right());
            pair->do_callback();
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
        pair->do_callback();
        delete pair;
        reduce(result);
        this->dec();
      }

      void reduce_pair_pair(const ReducePair* pair1, const ReducePair* pair2) {
        std::shared_ptr<result_type> result(new result_type(op_(pair1->left(),
            pair1->right(), pair2->left(), pair2->right())));
        pair1->do_callback();
        delete pair1;
        pair2->do_callback();
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
    /// where \c op is the reduction operation given to the constructor.
    /// \note There is no need to add this object to the MADNESS task queue. It
    /// will be handled internally by the object. Simply call \c submit() to add
    /// this task to the task queue.
    /// \tparam T The object type to be reduced
    /// \tparam Op The reduction operation type
    template <typename Op>
    class ReducePairTask {
    public:

      typedef typename Op::result_type result_type; ///< The reduction result type
      typedef typename Op::first_argument_type first_argument_type; ///< The left-hand argument type
      typedef typename Op::second_argument_type second_argument_type; ///< The right-hand argument type
      typedef ReducePairTask<Op> ReducePairTask_; ///< This object type

    private:

      std::shared_ptr<ReducePairTaskImpl<Op> > pimpl_; ///< The reduction task object.

      /// Deleter function for the pimpl

      /// This function will either run the task function if the count is zero
      /// or submit the task to MADNESS the task queue. The pointer will
      /// eventually be deleted by the task queue.
      /// \param pimpl The pinter to be disposed of
      static void deleter(ReducePairTaskImpl<Op>* pimpl) {
        if(pimpl->count() == 0ul) {
          pimpl->run(madness::TaskThreadEnv(1,0,0));
          pimpl->dec();
          delete pimpl;
        } else {
          // Get the result before submitting calling dec(), otherwise the task
          // could run and be deleted before we are done here.
          pimpl->get_world().taskq.add(pimpl);
          pimpl->dec(); // decrement the fake dependency so the task will run.
        }
      }

    public:

      /// Task constructor

      /// \param world The world where this task will be run.
      ReducePairTask(madness::World& world, const Op& op = Op()) :
        pimpl_(new ReducePairTaskImpl<Op>(world, op), &deleter)
      { }

      ReducePairTask(const ReducePairTask<Op>& other) :
        pimpl_(other.pimpl_)
      { }

      /// Destructor

      /// If the reduction has not been submitted or \c destroy() has not been
      /// called, it well be submitted when the the destructor is called.
      ~ReducePairTask() { if(pimpl_) submit(); }

      ReducePairTask<Op> operator=(const ReducePairTask<Op>& other) {
        pimpl_ = other.pimpl_;
        return *this;
      }

      /// Add an element to the reduction

      /// \c value may be of type \c value_type (or \c T ), \c madness::Future<value_type> ,
      /// or \c madness::RemoteReference<madness::FutureImpl<value_type>> .
      /// \tparam Value The type of the object that will be reduced
      /// \param value The object that will be reduced
      template <typename Left, typename Right>
      void add(const Left& left, const Right& right, madness::CallbackInterface* callback = NULL) {
        TA_ASSERT(pimpl_);
        pimpl_->add(left, right, callback);
      }

      /// Get a count for the number of pairs added to the reduce task

      /// \warning This is not thread safe.
      std::size_t count() const {
        TA_ASSERT(pimpl_);
        return pimpl_->count();
      }

      /// Task result accessor

      /// \return A future to the result
      madness::Future<result_type> result() const {
        TA_ASSERT(pimpl_);
        return pimpl_->result();
      }

      /// Submit the reduction task to the task queue

      /// \return The result of the reduction
      /// \note After submitting the task, objects can no longer be added to the
      /// reduction.
      madness::Future<result_type> submit() {
        TA_ASSERT(pimpl_);
        madness::Future<result_type> result = pimpl_->result();
        pimpl_.reset();
        return result;
      }

    }; // class ReducePairTask

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_REDUCE_TASK_H__INCLUDED
