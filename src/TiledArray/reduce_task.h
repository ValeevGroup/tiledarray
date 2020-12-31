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

#ifndef TILEDARRAY_REDUCE_TASK_H__INCLUDED
#define TILEDARRAY_REDUCE_TASK_H__INCLUDED

#include <TiledArray/config.h>
#include <TiledArray/error.h>
#include <TiledArray/external/madness.h>

#ifdef TILEDARRAY_HAS_CUDA
#include <TiledArray/cuda/cuda_task_fn.h>
#include <TiledArray/external/cuda.h>
#include <TiledArray/tensor/type_traits.h>
#include <TiledArray/util/time.h>
#endif

namespace TiledArray {
namespace detail {

template <typename T>
struct ArgumentHelper {
  typedef Future<T> type;
};  // struct ArgumentHelper

template <typename T>
struct ArgumentHelper<Future<T> > {
  typedef Future<T> type;
};  // struct ArgumentHelper

template <typename T, typename U>
struct ArgumentHelper<std::pair<Future<T>, Future<U> > > {
  typedef std::pair<Future<T>, Future<U> > type;
};  // struct ArgumentHelper

/// Wrapper that to convert a pair-wise reduction into a standard reduction

/// \tparam opT The pair-wise reduction operation to be reduced
template <typename opT>
class ReducePairOpWrapper {
 public:
  typedef typename opT::result_type result_type;
  ///< The result type of this reduction operation
  typedef typename std::remove_cv<typename std::remove_reference<
      typename opT::first_argument_type>::type>::type first_argument_type;
  ///< The left-hand argument type
  typedef typename std::remove_cv<typename std::remove_reference<
      typename opT::second_argument_type>::type>::type second_argument_type;
  ///< The right-hand argument type
  typedef std::pair<Future<first_argument_type>, Future<second_argument_type> >
      argument_type;
  ///< The combine argument type

 private:
  opT op_;  ///< The pairwise reduction operation

 public:
  /// Default constructor
  ReducePairOpWrapper() : op_() {}

  /// Constructor

  /// \param op The base operation
  ReducePairOpWrapper(const opT& op) : op_(op) {}

  /// Copy constructor

  /// \param other The other operation to be copied
  ReducePairOpWrapper(const ReducePairOpWrapper<opT>& other) : op_(other.op_) {}

  /// Destructor
  ~ReducePairOpWrapper() {}

  /// Copy assignment operator

  /// \param other The other operation to be copied
  /// \return This operation
  ReducePairOpWrapper<opT>& operator=(const ReducePairOpWrapper<opT>& other) {
    op_ = other.op_;
    return *this;
  }

  /// Create an default reduction object
  result_type operator()() const { return op_(); }

  result_type operator()(result_type& temp) const { return op_(temp); }

  /// Reduce two result objects

  /// \param[out] result The object that will hold the result of this reduction
  /// \param[in] arg The result of another reduction operation
  void operator()(result_type& result, const result_type& arg) {
    op_(result, arg);
  }

  /// Reduce an argument pair

  /// \param[out] result The object that will hold the result of this reduction
  /// \param[in] arg The argument pair to be reduced
  void operator()(result_type& result, const argument_type& arg) const {
    op_(result, arg.first, arg.second);
  }

};  // class ReducePairOpWrapper

/// Reduce task

/// This task will reduce an arbitrary number of objects. It is optimized
/// for reduction of data that is the result of other tasks or remote data.
/// Also, it is assumed that individual reduction operations require a
/// substantial amount of work (i.e. your reduction operation should reduce
/// a vector of data, not two numbers). The reduction arguments are reduced
/// as they become ready, which results in non-deterministic reduction
/// order. This is much faster than a simple binary tree reduction since the
/// reduction tasks do not have to wait for specific pairs of data. Though
/// data that is not stored in a future can be used, it may not be the best
/// choice in that case.
///
/// The reduction operation must have the following form:
/// \code
/// struct ReductionOp {
///     // typedefs
///     typedef ... result_type;
///     typedef ... argument_type;
///
///     // Constructors
///     ReductionOp();
///     ReductionOp(const ReductionOp&);
///     ReductionOp& operator=(const ReductionOp&);
///
///     // Reduction functions
///
///     // Make an empty result object
///     result_type operator()() const;
///
///     // Post process the result
///     result_type operator()(const result_type&) const;
///
///     // Reduce two result objects
///     void operator()(result_type&, const result_type&) const;
///
///     // Reduce an argument
///     void operator()(result_type&, const argument_type&) const;
///
/// }; // struct ReductionOp
/// \endcode
///
/// For example, a vector sum function might look like:
///
/// \code
/// struct VectorSum {
///     // typedefs
///     typedef double result_type;
///     typedef std::vector<double> argument_type;
///
///     // Compiler generated constructors and assignment operators are OK here
///
///     // Reduction functions
///
///     // Make an empty result object
///     result_type operator()() const { return 0; }
///
///     // Post process the result (no operation, passthrough)
///     const result_type& operator()(const result_type& result) const {
///       return result;
///     }
///
///     void operator()(result_type& result, const result_type& arg) const {
///         result += arg;
///     }
///
///     /// Reduce an argument pair
///     void operator()(result_type& result, const argument_type& arg) const {
///         for(std::size_t i = 0ul; i < first.size(); ++i)
///             result += arg[i];
///     }
/// }; // struct VectorProduct
/// \endcode
/// \note There is no need to add this object to the MADNESS task queue. It
/// will be handled internally by the object. Simply call \c submit() to add
/// this task to the task queue.
/// \tparam opT The reduction operation type
template <typename opT>
class ReduceTask {
 private:
  typedef typename opT::result_type result_type;
  typedef typename std::remove_const<
      typename std::remove_reference<typename opT::argument_type>::type>::type
      argument_type;

  /// Reduction task implementation

  /// This object is the implementation object and the task object that is
  /// submitted to the task queue. It is also used by other associated task
  /// data sharing.
  class ReduceTaskImpl : public madness::TaskInterface {
   public:
    /// Reduction argument container

    /// This object holds the reduction argument. When the arguments to
    /// this object are ready, it will invoke the parent callback.
    class ReduceObject : public madness::CallbackInterface {
     private:
      ReduceTaskImpl* parent_;  ///< The parent task
      typename ArgumentHelper<argument_type>::type
          arg_;                               ///< The reduction argument
      madness::CallbackInterface* callback_;  ///< Reduction callback
      madness::AtomicInt count_;              ///< Dependency counter

      /// Register a future as a dependency

      /// \tparam T The type of the future
      /// \param f The future that this object depends on
      template <typename T>
      void register_callbacks(Future<T>& f) {
        if (f.probe()) {
          parent_->ready(this);
        } else {
          count_ = 1;
          f.register_callback(this);
        }
      }

      /// Register a pair of futures as dependencies

      /// \tparam T The type of the first future
      /// \tparam U The type of the second future
      /// \param p The pair of futures that this object depends on
      template <typename T, typename U>
      void register_callbacks(std::pair<Future<T>, Future<U> >& p) {
        if (p.first.probe() && p.second.probe()) {
          parent_->ready(this);
        } else {
          count_ = 2;
          p.first.register_callback(this);
          p.second.register_callback(this);
        }
      }

     public:
      /// Constructor

      /// \tparam Arg The argument type
      /// \param parent The owner of this object
      /// \param arg The reduction argument
      /// \param callback The callback to invoke when this argument has been
      /// reduced
      template <typename Arg>
      ReduceObject(ReduceTaskImpl* parent, const Arg& arg,
                   madness::CallbackInterface* callback)
          : parent_(parent), arg_(arg), callback_(callback) {
        TA_ASSERT(parent_);
        register_callbacks(arg_);
      }

      virtual ~ReduceObject() {}

      /// Callback function that is invoked when the argument is ready
      virtual void notify() {
        if ((--count_) == 0) parent_->ready(this);
      }

      /// Argument accessor

      /// \return A const reference to the reduction argument
      const argument_type& arg() const { return arg_; }

      /// Destroy the \c object

      /// This function will invoke the callback and delete object.
      /// \param object The reduce object to be destroyed
      static void destroy(const ReduceObject* object) {
        static constexpr const bool trace_tasks =
#ifdef TILEDARRAY_ENABLE_TASK_DEBUG_TRACE
            true
#else
            false
#endif
            ;
        if (object->callback_) {
          if (trace_tasks)
            object->callback_->notify_debug("destroy(*ReduceObject)");
          else
            object->callback_->notify();
        }
        delete object;
      }

    };  // class ReduceObject

#ifdef TILEDARRAY_HAS_CUDA

    static void CUDART_CB cuda_reduceobject_delete_callback(void* userData) {
      const auto t0 = TiledArray::now();

      std::vector<void*>* objects = static_cast<std::vector<void*>*>(userData);

      /// first pointer is always madness::World*
      madness::World* world = static_cast<madness::World*>((*objects)[0]);

      auto destroy_vector = [](std::vector<void*>* objects) {
        std::size_t n_objects = objects->size();
        /// skip the first pointer since it is always World*
        for (std::size_t i = 1; i < n_objects; i++) {
          // convert void* to ReduceObject*
          ReduceObject* reduce_object =
              static_cast<ReduceObject*>((*objects)[i]);
          // delete ReduceObject
          ReduceObject::destroy(reduce_object);
        }
        /// delete objects pointer
        delete objects;
        //            std::cout << std::to_string(
        //                             TiledArray::get_default_world().rank()) +
        //                             " call 1\n";
      };

      /// use madness task to call the destroy function, since it might call
      /// cuda API
      world->taskq.add(destroy_vector, objects, TaskAttributes::hipri());

      const auto t1 = TiledArray::now();
      TiledArray::detail::cuda_callback_duration_ns<0>() +=
          TiledArray::duration_in_ns(t0, t1);
    }

    static void CUDART_CB cuda_dependency_dec_callback(void* userData) {
      const auto t0 = TiledArray::now();

      std::vector<void*>* objects = static_cast<std::vector<void*>*>(userData);

      for (auto& item : *objects) {
        // convert void* to DependencyInterface
        ReduceTaskImpl* dep = static_cast<ReduceTaskImpl*>(item);
        // call dec
        dep->dec();
      }
      delete objects;
      //          std::cout <<
      //          std::to_string(TiledArray::get_default_world().rank()) +
      //                           " call 2\n";

      const auto t1 = TiledArray::now();
      TiledArray::detail::cuda_callback_duration_ns<1>() +=
          TiledArray::duration_in_ns(t0, t1);
    }

    static void CUDART_CB
    cuda_dependency_dec_reduceobject_delete_callback(void* userData) {
      const auto t0 = TiledArray::now();

      std::vector<void*>* objects = static_cast<std::vector<void*>*>(userData);

      assert(objects->size() == 3);

      /// convert void* to madness::World*
      madness::World* world = static_cast<madness::World*>(objects->at(0));

      // convert void* to DependencyInterface
      ReduceTaskImpl* dep = static_cast<ReduceTaskImpl*>(objects->at(1));
      // call dec
      dep->dec();

      // convert void* to ReduceObject*
      ReduceObject* reduce_object = static_cast<ReduceObject*>(objects->at(2));

      auto destroy = [](ReduceObject* object) {
        ReduceObject::destroy(object);
        //            std::cout << std::to_string(
        //                             TiledArray::get_default_world().rank()) +
        //                             " call 3\n";
      };

      // delete ReduceObject
      world->taskq.add(destroy, reduce_object, TaskAttributes::hipri());

      delete objects;

      const auto t1 = TiledArray::now();
      TiledArray::detail::cuda_callback_duration_ns<2>() +=
          TiledArray::duration_in_ns(t0, t1);
    }

    static void CUDART_CB cuda_readyresult_reset_callback(void* userData) {
      const auto t0 = TiledArray::now();

      std::vector<void*>* objects = static_cast<std::vector<void*>*>(userData);

      /// convert first void* to madness::World*
      madness::World* world = static_cast<madness::World*>((*objects)[0]);

      auto reset = [](std::vector<void*>* objects) {
        // skip the first one since it always be madness::World*
        // convert void* to the correct type
        std::shared_ptr<result_type>* result =
            static_cast<std::shared_ptr<result_type>*>((*objects)[1]);
        // call reset on shared_ptr
        result->reset();
        delete objects;
        //            std::cout <<
        //            std::to_string(TiledArray::get_default_world().rank()) +
        //                         " call 4\n";
      };

      world->taskq.add(reset, objects, TaskAttributes::hipri());

      const auto t1 = TiledArray::now();
      TiledArray::detail::cuda_callback_duration_ns<3>() +=
          TiledArray::duration_in_ns(t0, t1);
    }

#endif
    virtual void get_id(std::pair<void*, unsigned short>& id) const {
      return PoolTaskInterface::make_id(id, *this);
    }

    /// Check for ready reduce arguments and reduce them

    /// This function will check for and reduce data that is ready until
    /// there is no more data to be reduced. Once there is no more data
    /// that is ready to be reduced, result will be placed in the ready
    /// state.
    /// \param result The result object that will be used to reduce
    /// other data
    void reduce(std::shared_ptr<result_type>& result) {
      while (result) {
        lock_.lock();  // <<< Begin critical section
        if (ready_object_) {
          // Get the ready argument
          ReduceObject* ready_object = const_cast<ReduceObject*>(ready_object_);
          ready_object_ = nullptr;
          lock_.unlock();  // <<< End critical section

          // Reduce the argument that was held by ready_object_
          op_(*result, ready_object->arg());

          // cleanup the argument
#ifdef TILEDARRAY_HAS_CUDA
          auto stream_ptr = tls_cudastream_accessor();

          /// non-CUDA op
          if (stream_ptr == nullptr) {
            ReduceObject::destroy(ready_object);
            this->dec();
          } else {
            auto callback_object = new std::vector<void*>(3);
            (*callback_object)[0] = &world_;
            (*callback_object)[1] = this;
            (*callback_object)[2] = ready_object;
            CudaSafeCall(
                cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
            CudaSafeCall(cudaLaunchHostFunc(
                *stream_ptr, cuda_dependency_dec_reduceobject_delete_callback,
                callback_object));
            synchronize_stream(nullptr);
            //                std::cout << std::to_string(world().rank()) + "
            //                add 3\n";
          }
#else
          ReduceObject::destroy(ready_object);
          this->dec();
#endif
        } else if (ready_result_) {
          // Get the ready result
          std::shared_ptr<result_type> ready_result = ready_result_;
          ready_result_.reset();
          lock_.unlock();  // <<< End critical section

          // Reduce the result that was held by ready_result_
          op_(*result, *ready_result);

          // cleanup the result
#ifdef TILEDARRAY_HAS_CUDA
          auto stream_ptr = tls_cudastream_accessor();
          if (stream_ptr == nullptr) {
            ready_result.reset();
          } else {
            auto ready_result_heap =
                new std::shared_ptr<result_type>(ready_result);
            auto callback_object = new std::vector<void*>(2);
            (*callback_object)[0] = &world_;
            (*callback_object)[1] = ready_result_heap;
            CudaSafeCall(
                cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
            CudaSafeCall(cudaLaunchHostFunc(
                *stream_ptr, cuda_readyresult_reset_callback, callback_object));
            synchronize_stream(nullptr);
            //                std::cout << std::to_string(world().rank()) + "
            //                add 4\n";
          }
#else
          ready_result.reset();
#endif
        } else {
          // Nothing is ready, so place result in the ready state.
          ready_result_ = result;
          result.reset();
          lock_.unlock();  // <<< End critical section
        }
      }
    }

    /// Reduce an argument

    /// \param result The target of the reduction
    /// \param object The reduction argument to be reduced
    void reduce_result_object(std::shared_ptr<result_type> result,
                              const ReduceObject* object) {
      // Reduce the argument
      op_(*result, object->arg());

      // Cleanup the argument
#ifdef TILEDARRAY_HAS_CUDA
      auto stream_ptr = tls_cudastream_accessor();
      if (stream_ptr == nullptr) {
        ReduceObject::destroy(object);
      } else {
        auto callback_object = new std::vector<void*>(2);
        (*callback_object)[0] = &world_;
        (*callback_object)[1] = const_cast<ReduceObject*>(object);
        CudaSafeCall(
            cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
        CudaSafeCall(cudaLaunchHostFunc(
            *stream_ptr, cuda_reduceobject_delete_callback, callback_object));
        synchronize_stream(nullptr);
        //            std::cout << std::to_string(world().rank()) + " add 1\n";
      }
#else
      ReduceObject::destroy(object);
#endif
      // Check for more reductions
      reduce(result);

      // Decrement the dependency counter for the argument. This must
      // be done after the reduce call to avoid a race condition.
#ifdef TILEDARRAY_HAS_CUDA
      if (stream_ptr == nullptr) {
        this->dec();
      } else {
        auto callback_object2 = new std::vector<void*>(1);
        (*callback_object2)[0] = this;
        CudaSafeCall(
            cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
        CudaSafeCall(cudaLaunchHostFunc(
            *stream_ptr, cuda_dependency_dec_callback, callback_object2));
        //            std::cout << std::to_string(world().rank()) + " add 2\n";
      }
#else
      this->dec();
#endif
    }

    /// Reduce two reduction arguments
    void reduce_object_object(const ReduceObject* object1,
                              const ReduceObject* object2) {
      // Construct an empty result object
      auto result = std::make_shared<result_type>(op_());

      // Reduce the two arguments
      op_(*result, object1->arg());
      op_(*result, object2->arg());

      // Cleanup arguments
#ifdef TILEDARRAY_HAS_CUDA
      auto stream_ptr = tls_cudastream_accessor();
      if (stream_ptr == nullptr) {
        ReduceObject::destroy(object1);
        ReduceObject::destroy(object2);
      } else {
        auto callback_object1 = new std::vector<void*>(3);
        (*callback_object1)[0] = &world_;
        (*callback_object1)[1] = const_cast<ReduceObject*>(object1);
        (*callback_object1)[2] = const_cast<ReduceObject*>(object2);
        CudaSafeCall(
            cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
        CudaSafeCall(cudaLaunchHostFunc(
            *stream_ptr, cuda_reduceobject_delete_callback, callback_object1));
        synchronize_stream(nullptr);
        //            std::cout << std::to_string(world().rank()) + " add 1\n";
      }
#else
      ReduceObject::destroy(object1);
      ReduceObject::destroy(object2);
#endif

      // Check for more reductions
      reduce(result);

      // Decrement the dependency counter for the two arguments. This
      // must be done after the reduce call to avoid a race condition.
#ifdef TILEDARRAY_HAS_CUDA
      if (stream_ptr == nullptr) {
        this->dec();
        this->dec();
      } else {
        auto callback_object2 = new std::vector<void*>(2);
        (*callback_object2)[0] = this;
        (*callback_object2)[1] = this;
        CudaSafeCall(
            cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
        CudaSafeCall(cudaLaunchHostFunc(
            *stream_ptr, cuda_dependency_dec_callback, callback_object2));
        //            std::cout << std::to_string(world().rank()) + " add 2\n";
      }

#else
      this->dec();
      this->dec();
#endif
    }

#ifdef TILEDARRAY_HAS_CUDA
    template <typename Result = result_type>
    std::enable_if_t<detail::is_cuda_tile_v<Result>, void> internal_run(
        const madness::TaskThreadEnv&) {
      TA_ASSERT(ready_result_);

      auto post_result = madness::add_cuda_task(world_, op_, *ready_result_);
      result_.set(post_result);

      if (callback_) {
        result_.register_callback(callback_);
      }
    }

    template <typename Result = result_type>
    std::enable_if_t<!detail::is_cuda_tile_v<Result>, void>
#else
    void
#endif
    internal_run(const madness::TaskThreadEnv&) {
      TA_ASSERT(ready_result_);
      result_.set(op_(*ready_result_));

      if (callback_) callback_->notify();
    }

    World& world_;  ///< The world that owns this task
    opT op_;        ///< The reduction operation
    std::shared_ptr<result_type>
        ready_result_;  ///< Result object that is ready to be reduced
    volatile ReduceObject*
        ready_object_;  ///< Reduction argument that is ready to be reduced
    Future<result_type> result_;  ///< The result of the reduction task
    madness::Spinlock lock_;      ///< Task lock
    madness::CallbackInterface* callback_;  ///< The completion callback

   public:
    /// Implementation constructor

    /// \param world The world that owns this task
    /// \param op The reduction operation
    /// \param callback The callback that will be invoked when this task
    /// has completed
    ReduceTaskImpl(World& world, opT op, madness::CallbackInterface* callback)
        : madness::TaskInterface(1, TaskAttributes::hipri()),
          world_(world),
          op_(op),
          ready_result_(std::make_shared<result_type>(op())),
          ready_object_(nullptr),
          result_(),
          lock_(),
          callback_(callback) {}

    virtual ~ReduceTaskImpl() {}

    /// Task function
    virtual void run(const madness::TaskThreadEnv& threadEnv) {
      internal_run(threadEnv);
    }

    /// Callback function invoked by \c ReductionObject

    /// This function will place \c object in the ready state. If
    /// another object is already in the ready state, then both objects
    /// are used to spawn a task
    /// \param object The reduction object that is ready to be reduced
    void ready(ReduceObject* object) {
      TA_ASSERT(object);
      lock_.lock();  // <<< Begin critical section
      if (ready_result_) {
        std::shared_ptr<result_type> ready_result = ready_result_;
        ready_result_.reset();
        lock_.unlock();  // <<< End critical section
        TA_ASSERT(ready_result);
        world_.taskq.add(this, &ReduceTaskImpl::reduce_result_object,
                         ready_result, object, TaskAttributes::hipri());
      } else if (ready_object_) {
        ReduceObject* ready_object = const_cast<ReduceObject*>(ready_object_);
        ready_object_ = nullptr;
        lock_.unlock();  // <<< End critical section
        TA_ASSERT(ready_object);
        world_.taskq.add(this, &ReduceTaskImpl::reduce_object_object, object,
                         ready_object, TaskAttributes::hipri());
      } else {
        ready_object_ = object;
        lock_.unlock();  // <<< End critical section
      }
    }

    /// Task result accessor

    /// \return A future that will hold the result of the reduction task
    const Future<result_type>& result() const { return result_; }

    /// World accessor

    /// \return The world that owns this task.
    World& world() const { return world_; }

  };  // class ReduceTaskImpl

  ReduceTaskImpl* pimpl_;  ///< The reduction task object.
  std::size_t count_;      ///< Reduction argument counter

 public:
  /// Default constructor
  ReduceTask() : pimpl_(nullptr), count_(0ul) {}

  /// Constructor

  /// \param world The world that owns this task
  /// \param op The reduction operation [ default = opT() ]
  /// \param callback The callback that will be invoked when this task is
  /// complete
  ReduceTask(World& world, const opT& op = opT(),
             madness::CallbackInterface* callback = nullptr)
      : pimpl_(new ReduceTaskImpl(world, op, callback)), count_(0ul) {}

  /// Move constructor

  /// \param other The object to be moved
  ReduceTask(ReduceTask<opT>&& other) noexcept
      : pimpl_(other.pimpl_), count_(other.count_) {
    other.pimpl_ = nullptr;
    other.count_ = 0ul;
  }

  /// Destructor

  /// If the reduction has not been submitted or \c destroy() has not been
  /// called, it will be submitted when the destructor is called.
  ~ReduceTask() { delete pimpl_; }

  /// Move assignment operator

  /// \param other The object to be moved
  ReduceTask<opT>& operator=(ReduceTask<opT>&& other) noexcept {
    pimpl_ = other.pimpl_;
    count_ = other.count_;
    other.pimpl_ = nullptr;
    other.count_ = 0;
    return *this;
  }

  // Non-copyable
  ReduceTask(const ReduceTask<opT>&) = delete;
  ReduceTask<opT>& operator=(const ReduceTask<opT>&) = delete;

  /// Add an argument to the reduction task

  /// \c arg may be of the argument type of \c opT, a \c Future to the
  /// argument type, or \c RemoteReference<FutureImpl> to the argument
  /// type.
  /// \tparam Arg The argument type
  /// \param arg The argument that will be reduced
  /// \param callback The callback that will be invoked when this argument
  /// pair has been reduced [ default = nullptr ]
  template <typename Arg>
  int add(const Arg& arg, madness::CallbackInterface* callback = nullptr) {
    TA_ASSERT(pimpl_);
    pimpl_->inc();
    new typename ReduceTaskImpl::ReduceObject(pimpl_, arg, callback);
    return ++count_;
  }

  /// Argument count

  /// \return The total number of arguments added to this task
  int count() const { return count_; }

  /// Submit the reduction task to the task queue

  /// \return The result of the reduction
  /// \note Arguments can no longer be added to the reduction after
  /// calling \c submit().
  Future<result_type> submit() {
    TA_ASSERT(pimpl_);

    // Get the result before submitting/running the task, otherwise the
    // task could run and be deleted before we are done here.
    Future<result_type> result = pimpl_->result();

    pimpl_->dec();
    World& world = pimpl_->world();
    world.taskq.add(pimpl_);

    pimpl_ = nullptr;
    return result;
  }

  /// Type conversion operator

  /// \return \c true if the task object is initialized.
  operator bool() const { return pimpl_ != nullptr; }

};  // class ReduceTask

/// Reduce pair task

/// This task will reduce an arbitrary number of pairs of objects. This task
/// is optimized for reduction of data that is the result of other tasks or
/// remote data. Also, it is assumed that individual reduction operations
/// require a substantial amount of work (i.e. your reduction operation
/// should reduce a vector of data, not two numbers). The reduction
/// arguments are reduced as they become ready, which results in non-
/// deterministic reduction order. This is much faster than a simple binary
/// tree reduction since the reduction tasks do not have to wait for
/// specific pairs of data. Though data that is not stored in a future can
/// be used, it may not be the best choice in that case. \n
/// The reduction operation must have the following form:
/// \code
/// struct ReductionOp {
///     // typedefs
///     typedef ... result_type;
///     typedef ... first_argument_type;
///     typedef ... second_argument_type;
///
///     // Constructors
///     ReductionOp();
///     ReductionOp(const ReductionOp&);
///     ReductionOp& operator=(const ReductionOp&);
///
///     // Reduction functions
///
///     // Make an empty result object
///     result_type operator()() const;
///
///     // Post process the result
///     const result_type& operator()(const result_type&) const;
///
///     // Reduce two result objects
///     void operator()(result_type&, const result_type&) const;
///
///     // Reduce an argument pair
///     void operator()(result_type&, const first_argument_type&,
///         const second_argument_type&) const;
///
/// }; // struct ReductionOp
/// \endcode
///
/// For example, a dot product function might look like:
///
/// \code
/// struct DotProduct {
///     // typedefs
///     typedef double result_type;
///     typedef std::vector<double> first_argument_type;
///     tyepdef std::vector<double> second_argument_type;
///
///     // Compiler generated constructors or assignment operator OK here
///
///     // Reduction functions
///
///     // Make an empty result object
///     result_type operator()() const { return 0; }
///
///     // Post process the result (no operation, passthrough)
///     const result_type& operator()(const result_type& result) const {
///       return result;
///     }
///
///     void operator()(result_type& result, const result_type& arg) const {
///         result += arg;
///     }
///
///     /// Reduce an argument pair
///     void operator()(result_type& result,
///             const first_argument_type& first, const second_argument_type&
///             second) const
///     {
///         assert(first.size() == second.size());
///         for(std::size_t i = 0ul; i < first.size(); ++i)
///             result += first[i] * second[i];
///     }
///
/// }; // struct DotProduct
/// \endcode
/// \note There is no need to add this object to the MADNESS task queue. It
/// will be handled internally by the object. Simply call \c submit() to add
/// this task to the task queue.
/// \tparam opT The reduction operation type
template <typename opT>
class ReducePairTask : public ReduceTask<ReducePairOpWrapper<opT> > {
 private:
  typedef ReducePairOpWrapper<opT> op_type;  ///< The reduction operation type
  typedef typename op_type::first_argument_type
      first_argument_type;  ///< The left-hand reduction argument type
  typedef typename op_type::second_argument_type
      second_argument_type;  /// The right-hand reduction argument type
  typedef typename op_type::argument_type
      argument_type;  ///< The pair reduction argument type
  typedef ReduceTask<op_type> ReduceTask_;  ///< The base class

 public:
  /// Default constructor
  ReducePairTask() : ReduceTask_() {}

  /// Constructor

  /// \param world The world that owns this task
  /// \param op The pair reduction operation [ default = opT() ]
  /// \param callback The callback that will be invoked when this task is
  /// complete
  ReducePairTask(World& world, const opT& op = opT(),
                 madness::CallbackInterface* callback = nullptr)
      : ReduceTask_(world, op_type(op), callback) {}

  /// Move constructor

  /// \param other The object to be moved
  ReducePairTask(ReducePairTask<opT>&& other) noexcept
      : ReduceTask_(std::move(other)) {}

  /// Move assignment operator

  /// \param other The object to be moved
  ReducePairTask<opT>& operator=(ReducePairTask<opT>&& other) noexcept {
    ReduceTask_::operator=(std::move(other));
    return *this;
  }

  /// Non-copyable
  ReducePairTask(const ReducePairTask<opT>&) = delete;
  ReducePairTask<opT> operator=(const ReducePairTask<opT>&) = delete;

  /// Add a pair of arguments to the reduction task

  /// \c left and \c right may be of the argument types of \c opT, a
  /// \c Future to the argument types,
  /// \c RemoteReference<FutureImpl> to the argument
  //// types, or any combination of the above.
  /// \tparam L The left-hand object type
  /// \tparam R The right-hand object type
  /// \param left The left-hand argument that will be reduced
  /// \param right The right-hand argument that will be reduced
  /// \param callback The callback that will be invoked when this argument
  /// pair has been reduced [ default = nullptr ]
  template <typename L, typename R>
  void add(const L& left, const R& right,
           madness::CallbackInterface* callback = nullptr) {
    ReduceTask_::add(argument_type(Future<first_argument_type>(left),
                                   Future<second_argument_type>(right)),
                     callback);
  }

};  // class ReducePairTask

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_REDUCE_TASK_H__INCLUDED
