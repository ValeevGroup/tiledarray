//
// Created by Chong Peng on 2019-03-20.
//

#ifndef TILEDARRAY_CUDA_CUDA_TASK_FN_H__INCLUDED
#define TILEDARRAY_CUDA_CUDA_TASK_FN_H__INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <TiledArray/external/cuda.h>
#include <TiledArray/util/time.h>
#include <cuda_runtime.h>
#include <madness/world/taskfn.h>

namespace TiledArray {
namespace detail {

template <int64_t CallabackId>
std::atomic<int64_t>& cuda_callback_duration_ns() {
  static std::atomic<int64_t> value{0};
  return value;
}

inline std::atomic<int64_t>& cuda_taskfn_callback_duration_ns() {
  static std::atomic<int64_t> value{0};
  return value;
}

}  // namespace detail
}  // namespace TiledArray

namespace madness {

///
/// cudaTaskFn class
/// represent a task that calls an async cuda kernel
/// the task must call synchronize_stream function to tell which stream it
/// used
///

template <typename fnT, typename arg1T = void, typename arg2T = void,
          typename arg3T = void, typename arg4T = void, typename arg5T = void,
          typename arg6T = void, typename arg7T = void, typename arg8T = void,
          typename arg9T = void>
struct cudaTaskFn : public TaskInterface {
  static_assert(not(std::is_const<arg1T>::value ||
                    std::is_reference<arg1T>::value),
                "improper instantiation of cudaTaskFn, arg1T cannot be a const "
                "or reference type");
  static_assert(not(std::is_const<arg2T>::value ||
                    std::is_reference<arg2T>::value),
                "improper instantiation of cudaTaskFn, arg2T cannot be a const "
                "or reference type");
  static_assert(not(std::is_const<arg3T>::value ||
                    std::is_reference<arg3T>::value),
                "improper instantiation of cudaTaskFn, arg3T cannot be a const "
                "or reference type");
  static_assert(not(std::is_const<arg4T>::value ||
                    std::is_reference<arg4T>::value),
                "improper instantiation of cudaTaskFn, arg4T cannot be a const "
                "or reference type");
  static_assert(not(std::is_const<arg5T>::value ||
                    std::is_reference<arg5T>::value),
                "improper instantiation of cudaTaskFn, arg5T cannot be a const "
                "or reference type");
  static_assert(not(std::is_const<arg6T>::value ||
                    std::is_reference<arg6T>::value),
                "improper instantiation of cudaTaskFn, arg6T cannot be a const "
                "or reference type");
  static_assert(not(std::is_const<arg7T>::value ||
                    std::is_reference<arg7T>::value),
                "improper instantiation of cudaTaskFn, arg7T cannot be a const "
                "or reference type");
  static_assert(not(std::is_const<arg8T>::value ||
                    std::is_reference<arg8T>::value),
                "improper instantiation of cudaTaskFn, arg8T cannot be a const "
                "or reference type");
  static_assert(not(std::is_const<arg9T>::value ||
                    std::is_reference<arg9T>::value),
                "improper instantiation of cudaTaskFn, arg9T cannot be a const "
                "or reference type");

 private:
  /// This class type
  typedef cudaTaskFn<fnT, arg1T, arg2T, arg3T, arg4T, arg5T, arg6T, arg7T,
                     arg8T, arg9T>
      cudaTaskFn_;

  friend class AsyncTaskInterface;

  /// internal Task structure that wraps the Async cuda function
  struct AsyncTaskInterface : public madness::TaskInterface {
    AsyncTaskInterface(cudaTaskFn_* task, int ndepend = 0,
                       const TaskAttributes attr = TaskAttributes())
        : TaskInterface(ndepend, attr), task_(task) {}

    virtual ~AsyncTaskInterface() = default;

   protected:
    void run(const TaskThreadEnv& env) override {
      // run the async function, the function must call synchronize_stream() to
      // set the stream it used!!
      task_->run_async();

      // get the stream used by async function
      auto stream = TiledArray::tls_cudastream_accessor();

      //      TA_ASSERT(stream != nullptr);

      // WARNING, need to handle NoOp
      if (stream == nullptr) {
        task_->notify();
      } else {
        // TODO should we use cuda callback or cuda events??
        // insert cuda callback
        cudaLaunchHostFunc(*stream, cuda_callback, task_);
        // reset stream to nullptr
        TiledArray::synchronize_stream(nullptr);
      }
    }

   private:
    static void CUDART_CB cuda_callback(void* userData) {
      TA_ASSERT(!madness::is_madness_thread());
      const auto t0 = TiledArray::now();
      // convert void * to AsyncTaskInterface*
      auto* callback = static_cast<cudaTaskFn_*>(userData);
      //      std::stringstream address;
      //      address << (void*) callback;
      //      std::string message = "callback on cudaTaskFn: " + address.str() +
      //        '\n'; std::cout << message;
      callback->notify();
      const auto t1 = TiledArray::now();

      TiledArray::detail::cuda_taskfn_callback_duration_ns() +=
          TiledArray::duration_in_ns(t0, t1);
    }

    cudaTaskFn_* task_;
  };

 public:
  typedef fnT functionT;  ///< The task function type
  typedef typename detail::task_result_type<fnT>::resultT resultT;
  ///< The result type of the function
  typedef typename detail::task_result_type<fnT>::futureT futureT;

  // argument value typedefs

  static const unsigned int arity =
      detail::ArgCount<arg1T, arg2T, arg3T, arg4T, arg5T, arg6T, arg7T, arg8T,
                       arg9T>::value;
  ///< The number of arguments given for the function
  ///< \note This may not match the arity of the function
  ///< if it has default parameter values

 private:
  futureT result_;             ///< The task Future result
  const functionT func_;       ///< The task function
  TaskInterface* async_task_;  ///< The internal AsyncTaskInterface that wraps
  ///< the async cuda function
  futureT async_result_;  ///< the future returned from the async task

  // If the value of the argument is known at the time the
  // Note: The type argNT for argN, where N  is > arity should be void

  typename detail::task_arg<arg1T>::holderT
      arg1_;  ///< Argument 1 that will be given to the function
  typename detail::task_arg<arg2T>::holderT
      arg2_;  ///< Argument 2 that will be given to the function
  typename detail::task_arg<arg3T>::holderT
      arg3_;  ///< Argument 3 that will be given to the function
  typename detail::task_arg<arg4T>::holderT
      arg4_;  ///< Argument 4 that will be given to the function
  typename detail::task_arg<arg5T>::holderT
      arg5_;  ///< Argument 5 that will be given to the function
  typename detail::task_arg<arg6T>::holderT
      arg6_;  ///< Argument 6 that will be given to the function
  typename detail::task_arg<arg7T>::holderT
      arg7_;  ///< Argument 7 that will be given to the function
  typename detail::task_arg<arg8T>::holderT
      arg8_;  ///< Argument 8 that will be given to the function
  typename detail::task_arg<arg9T>::holderT
      arg9_;  ///< Argument 9 that will be given to the function

  template <typename fT>
  static fT& get_func(fT& f) {
    return f;
  }

  template <typename ptrT, typename memfnT, typename resT>
  static memfnT get_func(
      const detail::MemFuncWrapper<ptrT, memfnT, resT>& wrapper) {
    return detail::get_mem_func_ptr(wrapper);
  }

  void get_id(std::pair<void*, unsigned short>& id) const override {
    return make_id(id, get_func(func_));
  }

  /// run the async task function
  void run_async() {
    detail::run_function(async_result_, func_, arg1_, arg2_, arg3_, arg4_,
                         arg5_, arg6_, arg7_, arg8_, arg9_);
  }

  /// Register a non-ready future as a dependency, the dependency goes to the
  /// internal AsyncTaskInterface

  /// \tparam T The type of the future to check
  /// \param fut The future to check
  template <typename T>
  inline void check_dependency(Future<T>& fut) {
    if (!fut.probe()) {
      async_task_->inc();
      fut.register_callback(async_task_);
    }
  }

  /// Register (pointer to) a non-ready future as a dependency, the dependency
  /// goes to the internal AsyncTaskInterface

  /// \tparam T The type of the future to check
  /// \param fut The future to check
  template <typename T>
  inline void check_dependency(Future<T>* fut) {
    if (!fut->probe()) {
      async_task_->inc();
      fut->register_callback(async_task_);
    }
  }

  /// None future arguments are always ready => no op
  template <typename T>
  inline void check_dependency(detail::ArgHolder<std::vector<Future<T>>>& arg) {
    check_dependency(static_cast<std::vector<Future<T>>&>(arg));
  }

  /// None future arguments are always ready => no op
  template <typename T>
  inline void check_dependency(std::vector<Future<T>>& vec) {
    for (typename std::vector<Future<T>>::iterator it = vec.begin();
         it != vec.end(); ++it)
      check_dependency(*it);
  }

  /// Future<void> is always ready => no op
  inline void check_dependency(const std::vector<Future<void>>&) {}

  /// None future arguments are always ready => no op
  template <typename T>
  inline void check_dependency(const detail::ArgHolder<T>&) {}

  /// Future<void> is always ready => no op
  inline void check_dependency(const Future<void>&) {}

  /// Check dependencies and register callbacks where necessary
  void check_dependencies() {
    this->inc();  // the current cudaTaskFn depends on the internal
    // AsyncTaskInterface, dependency = 1
    check_dependency(arg1_);
    check_dependency(arg2_);
    check_dependency(arg3_);
    check_dependency(arg4_);
    check_dependency(arg5_);
    check_dependency(arg6_);
    check_dependency(arg7_);
    check_dependency(arg8_);
    check_dependency(arg9_);
  }

  // Copies are not allowed.
  cudaTaskFn(const cudaTaskFn_&);
  cudaTaskFn_ operator=(cudaTaskFn_&);

 public:
#if MADNESS_TASKQ_VARIADICS

  cudaTaskFn(const futureT& result, functionT func, const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(),
        arg2_(),
        arg3_(),
        arg4_(),
        arg5_(),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 0u);
    check_dependencies();
  }

  template <typename a1T>
  cudaTaskFn(const futureT& result, functionT func, a1T&& a1,
             const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(std::forward<a1T>(a1)),
        arg2_(),
        arg3_(),
        arg4_(),
        arg5_(),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 1u);
    check_dependencies();
  }

  template <typename a1T, typename a2T>
  cudaTaskFn(const futureT& result, functionT func, a1T&& a1, a2T&& a2,
             const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(std::forward<a1T>(a1)),
        arg2_(std::forward<a2T>(a2)),
        arg3_(),
        arg4_(),
        arg5_(),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 2u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T>
  cudaTaskFn(const futureT& result, functionT func, a1T&& a1, a2T&& a2,
             a3T&& a3, const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(std::forward<a1T>(a1)),
        arg2_(std::forward<a2T>(a2)),
        arg3_(std::forward<a3T>(a3)),
        arg4_(),
        arg5_(),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 3u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T>
  cudaTaskFn(const futureT& result, functionT func, a1T&& a1, a2T&& a2,
             a3T&& a3, a4T&& a4, const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(std::forward<a1T>(a1)),
        arg2_(std::forward<a2T>(a2)),
        arg3_(std::forward<a3T>(a3)),
        arg4_(std::forward<a4T>(a4)),
        arg5_(),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 4u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T,
            typename a5T>
  cudaTaskFn(const futureT& result, functionT func, a1T&& a1, a2T&& a2,
             a3T&& a3, a4T&& a4, a5T&& a5, const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(std::forward<a1T>(a1)),
        arg2_(std::forward<a2T>(a2)),
        arg3_(std::forward<a3T>(a3)),
        arg4_(std::forward<a4T>(a4)),
        arg5_(std::forward<a5T>(a5)),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 5u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T,
            typename a5T, typename a6T>
  cudaTaskFn(const futureT& result, functionT func, a1T&& a1, a2T&& a2,
             a3T&& a3, a4T&& a4, a5T&& a5, a6T&& a6, const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(std::forward<a1T>(a1)),
        arg2_(std::forward<a2T>(a2)),
        arg3_(std::forward<a3T>(a3)),
        arg4_(std::forward<a4T>(a4)),
        arg5_(std::forward<a5T>(a5)),
        arg6_(std::forward<a6T>(a6)),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 6u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T,
            typename a5T, typename a6T, typename a7T>
  cudaTaskFn(const futureT& result, functionT func, a1T&& a1, a2T&& a2,
             a3T&& a3, a4T&& a4, a5T&& a5, a6T&& a6, a7T&& a7,
             const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(std::forward<a1T>(a1)),
        arg2_(std::forward<a2T>(a2)),
        arg3_(std::forward<a3T>(a3)),
        arg4_(std::forward<a4T>(a4)),
        arg5_(std::forward<a5T>(a5)),
        arg6_(std::forward<a6T>(a6)),
        arg7_(std::forward<a7T>(a7)),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 7u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T,
            typename a5T, typename a6T, typename a7T, typename a8T>
  cudaTaskFn(const futureT& result, functionT func, a1T&& a1, a2T&& a2,
             a3T&& a3, a4T&& a4, a5T&& a5, a6T&& a6, a7T&& a7, a8T&& a8,
             const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(std::forward<a1T>(a1)),
        arg2_(std::forward<a2T>(a2)),
        arg3_(std::forward<a3T>(a3)),
        arg4_(std::forward<a4T>(a4)),
        arg5_(std::forward<a5T>(a5)),
        arg6_(std::forward<a6T>(a6)),
        arg7_(std::forward<a7T>(a7)),
        arg8_(std::forward<a8T>(a8)),
        arg9_() {
    MADNESS_ASSERT(arity == 8u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T,
            typename a5T, typename a6T, typename a7T, typename a8T,
            typename a9T>
  cudaTaskFn(const futureT& result, functionT func, a1T&& a1, a2T&& a2,
             a3T&& a3, a4T&& a4, a5T&& a5, a6T&& a6, a7T&& a7, a8T&& a8,
             a9T&& a9, const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(std::forward<a1T>(a1)),
        arg2_(std::forward<a2T>(a2)),
        arg3_(std::forward<a3T>(a3)),
        arg4_(std::forward<a4T>(a4)),
        arg5_(std::forward<a5T>(a5)),
        arg6_(std::forward<a6T>(a6)),
        arg7_(std::forward<a7T>(a7)),
        arg8_(std::forward<a8T>(a8)),
        arg9_(std::forward<a9T>(a9)) {
    MADNESS_ASSERT(arity == 9u);
    check_dependencies();
  }

  cudaTaskFn(const futureT& result, functionT func, const TaskAttributes& attr,
             archive::BufferInputArchive& input_arch)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(input_arch),
        arg2_(input_arch),
        arg3_(input_arch),
        arg4_(input_arch),
        arg5_(input_arch),
        arg6_(input_arch),
        arg7_(input_arch),
        arg8_(input_arch),
        arg9_(input_arch) {
    check_dependencies();
  }
#else   // MADNESS_TASKQ_VARIADICS
  cudaTaskFn(const futureT& result, functionT func, const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(),
        arg2_(),
        arg3_(),
        arg4_(),
        arg5_(),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 0u);
    check_dependencies();
  }

  template <typename a1T>
  cudaTaskFn(const futureT& result, functionT func, const a1T& a1,
             const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(a1),
        arg2_(),
        arg3_(),
        arg4_(),
        arg5_(),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 1u);
    check_dependencies();
  }

  template <typename a1T, typename a2T>
  cudaTaskFn(const futureT& result, functionT func, const a1T& a1,
             const a2T& a2, const TaskAttributes& attr = TaskAttributes())
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(a1),
        arg2_(a2),
        arg3_(),
        arg4_(),
        arg5_(),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 2u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T>
  cudaTaskFn(const futureT& result, functionT func, const a1T& a1,
             const a2T& a2, const a3T& a3, const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(a1),
        arg2_(a2),
        arg3_(a3),
        arg4_(),
        arg5_(),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 3u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T>
  cudaTaskFn(const futureT& result, functionT func, const a1T& a1,
             const a2T& a2, const a3T& a3, const a4T& a4,
             const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(a1),
        arg2_(a2),
        arg3_(a3),
        arg4_(a4),
        arg5_(),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 4u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T,
            typename a5T>
  cudaTaskFn(const futureT& result, functionT func, const a1T& a1,
             const a2T& a2, const a3T& a3, const a4T& a4, const a5T& a5,
             const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(a1),
        arg2_(a2),
        arg3_(a3),
        arg4_(a4),
        arg5_(a5),
        arg6_(),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 5u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T,
            typename a5T, typename a6T>
  cudaTaskFn(const futureT& result, functionT func, const a1T& a1,
             const a2T& a2, const a3T& a3, const a4T& a4, const a5T& a5,
             const a6T& a6, const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(a1),
        arg2_(a2),
        arg3_(a3),
        arg4_(a4),
        arg5_(a5),
        arg6_(a6),
        arg7_(),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 6u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T,
            typename a5T, typename a6T, typename a7T>
  cudaTaskFn(const futureT& result, functionT func, const a1T& a1,
             const a2T& a2, const a3T& a3, const a4T& a4, const a5T& a5,
             const a6T& a6, const a7T& a7, const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(a1),
        arg2_(a2),
        arg3_(a3),
        arg4_(a4),
        arg5_(a5),
        arg6_(a6),
        arg7_(a7),
        arg8_(),
        arg9_() {
    MADNESS_ASSERT(arity == 7u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T,
            typename a5T, typename a6T, typename a7T, typename a8T>
  cudaTaskFn(const futureT& result, functionT func, const a1T& a1,
             const a2T& a2, const a3T& a3, const a4T& a4, const a5T& a5,
             const a6T& a6, const a7T& a7, const a8T& a8,
             const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(a1),
        arg2_(a2),
        arg3_(a3),
        arg4_(a4),
        arg5_(a5),
        arg6_(a6),
        arg7_(a7),
        arg8_(a8),
        arg9_() {
    MADNESS_ASSERT(arity == 8u);
    check_dependencies();
  }

  template <typename a1T, typename a2T, typename a3T, typename a4T,
            typename a5T, typename a6T, typename a7T, typename a8T,
            typename a9T>
  cudaTaskFn(const futureT& result, functionT func, const a1T& a1,
             const a2T& a2, const a3T& a3, const a4T& a4, const a5T& a5,
             const a6T& a6, const a7T& a7, const a8T& a8, const a9T& a9,
             const TaskAttributes& attr)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(a1),
        arg2_(a2),
        arg3_(a3),
        arg4_(a4),
        arg5_(a5),
        arg6_(a6),
        arg7_(a7),
        arg8_(a8),
        arg9_(a9) {
    MADNESS_ASSERT(arity == 9u);
    check_dependencies();
  }

  cudaTaskFn(const futureT& result, functionT func, const TaskAttributes& attr,
             archive::BufferInputArchive& input_arch)
      : TaskInterface(attr),
        result_(result),
        func_(func),
        async_task_(new AsyncTaskInterface(this)),
        async_result_(),
        arg1_(input_arch),
        arg2_(input_arch),
        arg3_(input_arch),
        arg4_(input_arch),
        arg5_(input_arch),
        arg6_(input_arch),
        arg7_(input_arch),
        arg8_(input_arch),
        arg9_(input_arch) {
    check_dependencies();
  }
#endif  // MADNESS_TASKQ_VARIADICS

  // no need to delete async_task_, as it will be deleted by the TaskQueue
  virtual ~cudaTaskFn() = default;

  const futureT& result() const { return result_; }

  TaskInterface* async_task() { return async_task_; }

#ifdef HAVE_INTEL_TBB
  virtual tbb::task* execute() {
    result_.set(std::move(async_result_));
    return nullptr;
  }
#else
 protected:
  /// when this cudaTaskFn gets run, it means the AsyncTaskInterface is done
  /// set the result with async_result_, which is finished
  void run(const TaskThreadEnv& env) override {
    result_.set(std::move(async_result_));
  }
#endif  // HAVE_INTEL_TBB

};  // class cudaTaskFn

/// add a cudaTaskFn object to World
/// \tparam fnT A function pointer or functor
/// \tparam a1T Type of argument 1.
/// \tparam a2T Type of argument 2.
/// \tparam a3T Type of argument 3.
/// \tparam a4T Type of argument 4.
/// \tparam a5T Type of argument 5.
/// \tparam a6T Type of argument 6.
/// \tparam a7T Type of argument 7.
/// \tparam a8T Type of argument 8.
/// \tparam a9T Type of argument 9.
/// \param[in] t Description needed.
/// \return Description needed.
template <typename fnT, typename a1T, typename a2T, typename a3T, typename a4T,
          typename a5T, typename a6T, typename a7T, typename a8T, typename a9T>
typename cudaTaskFn<fnT, a1T, a2T, a3T, a4T, a5T, a6T, a7T, a8T, a9T>::futureT
add_cuda_taskfn(
    madness::World& world,
    cudaTaskFn<fnT, a1T, a2T, a3T, a4T, a5T, a6T, a7T, a8T, a9T>* t) {
  typename cudaTaskFn<fnT, a1T, a2T, a3T, a4T, a5T, a6T, a7T, a8T, a9T>::futureT
      res(t->result());
  // add the internal async task in cuda task as well
  world.taskq.add(t->async_task());
  // add the cuda task
  world.taskq.add(static_cast<TaskInterface*>(t));
  return res;
}

/// enabled if last argument is not TaskAttributes
/// \tparam fnT  A function pointer or functor
/// \tparam argsT types of all argument
/// \return A future to the result
template <
    typename fnT, typename... argsT,
    typename = std::enable_if_t<!meta::taskattr_is_last_arg<argsT...>::value>>
typename detail::function_enabler<fnT(future_to_ref_t<argsT>...)>::type
add_cuda_task(madness::World& world, fnT&& fn, argsT&&... args) {
  /// type of cudaTaskFn object
  using taskT =
      cudaTaskFn<std::decay_t<fnT>,
                 std::remove_const_t<std::remove_reference_t<argsT>>...>;

  return add_cuda_taskfn(
      world, new taskT(typename taskT::futureT(), std::forward<fnT>(fn),
                       std::forward<argsT>(args)..., TaskAttributes()));
}

/// enabled if last argument is  TaskAttributes
/// \tparam fnT  A function pointer or functor
/// \tparam argsT types of all argument
/// \return A future to the result
template <
    typename fnT, typename... argsT,
    typename = std::enable_if_t<meta::taskattr_is_last_arg<argsT...>::value>>
typename meta::drop_last_arg_and_apply_callable<
    detail::function_enabler, fnT, future_to_ref_t<argsT>...>::type::type
add_cuda_task(madness::World& world, fnT&& fn, argsT&&... args) {
  /// type of cudaTaskFn object
  using taskT = typename meta::drop_last_arg_and_apply<
      cudaTaskFn, std::decay_t<fnT>,
      std::remove_const_t<std::remove_reference_t<argsT>>...>::type;

  return add_cuda_taskfn(
      world, new taskT(typename taskT::futureT(), std::forward<fnT>(fn),
                       std::forward<argsT>(args)...));
}

///
/// \tparam objT    the object type
/// \tparam memfnT  the member function type
/// \tparam argsT   variadic template for arguments
/// \return A future to the result
template <typename objT, typename memfnT, typename... argsT>
typename detail::memfunc_enabler<objT, memfnT>::type add_cuda_task(
    madness::World& world, objT&& obj, memfnT memfn, argsT&&... args) {
  return add_cuda_task(world,
                       detail::wrap_mem_fn(std::forward<objT>(obj), memfn),
                       std::forward<argsT>(args)...);
}

}  // namespace madness

#endif  // TILDARRAY_HAS_CUDA
#endif  // TILEDARRAY_CUDA_CUDA_TASK_FN_H__INCLUDED
