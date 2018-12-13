//
// Created by Chong Peng on 11/14/18.
//

#include <tiledarray.h>
#include <TiledArray/tensor/cuda/btas_um_tensor.h>


using value_type = double;
using tensor_type = TA::btasUMTensorVarray<value_type>;
using tile_type = TA::Tile<tensor_type>;


/// verify the elements in tile is equal to value
void verify(const tile_type& tile, value_type value){
//  const auto size = tile.size();
  std::cout << "verify" << std::endl;
  for(auto& num : tile){
    if(num != value){
      std::cout << "Error: " << num << " " << value << std::endl;
    }
    break;
  }

}


namespace madness {

template<typename fnT, typename arg1T = void, typename arg2T = void,
        typename arg3T = void, typename arg4T = void, typename arg5T = void,
        typename arg6T = void, typename arg7T = void, typename arg8T = void,
        typename arg9T = void>
struct cudaTaskFn : public madness::TaskFn<fnT, arg1T, arg2T, arg3T, arg4T, arg5T, arg6T, arg7T, arg8T, arg9T> {

  friend class AsyncTaskInterface;

  using TaskFn_ = madness::TaskFn<fnT, arg1T, arg2T, arg3T, arg4T, arg5T, arg6T, arg7T,
          arg8T, arg9T>;

  using cudaTaskFn_ = madness::cudaTaskFn<fnT, arg1T, arg2T, arg3T, arg4T, arg5T, arg6T, arg7T,
          arg8T, arg9T>;

  using typename TaskFn_::functionT;
  using typename TaskFn_::resultT;
  using typename TaskFn_::futureT;

  using TaskFn_::arity;

  using TaskFn_::result_;
  using TaskFn_::func_;
  using TaskFn_::arg1_;
  using TaskFn_::arg2_;
  using TaskFn_::arg3_;
  using TaskFn_::arg4_;
  using TaskFn_::arg5_;
  using TaskFn_::arg6_;
  using TaskFn_::arg7_;
  using TaskFn_::arg8_;
  using TaskFn_::arg9_;

  cudaTaskFn(const futureT &result, functionT func, const TaskAttributes &attr) :
          TaskFn_(result, func, attr), async_task_(new AsyncTaskInterface(this)), async_result_() {
//    TA_ASSERT(arity == 0u);
    this->inc();
    this->check_dependencies();
  }


  template<typename a1T>
  cudaTaskFn(const futureT &result, functionT func, a1T &&a1,
             const TaskAttributes &attr) : TaskFn_(result, func, a1, attr), async_task_(new AsyncTaskInterface(this)),
                                           async_result_() {
//    TA_ASSERT(arity == 1u);
    this->inc();
    this->check_dependencies();
  }

  template<typename a1T, typename a2T>
  cudaTaskFn(const futureT &result, functionT func, a1T &&a1, a2T &&a2,
             const TaskAttributes &attr) : TaskFn_(result, func, a1, a2, attr),
                                           async_task_(new AsyncTaskInterface(this)), async_result_() {
//    TA_ASSERT(arity == 2u);
    this->inc();
    this->check_dependencies();
    std::cout << "task: " << this->ndep() << " async_task: " << async_task_->ndep() << std::endl;
  }

  virtual ~cudaTaskFn() = default;

  void run_async() {
    std::cout << "run asyn task" << std::endl;
    detail::run_function(async_result_, func_, arg1_, arg2_, arg3_, arg4_, arg5_, arg6_, arg7_, arg8_, arg9_);
  }

private:

  // the argument dependency goes to async_task
  template<typename T>
  void check_dependency(Future<T> &fut) {
    std::cout << "check dependency cudaTaskFn" << std::endl;
    if (!fut.probe()) {
      async_task_->inc();
      fut.register_callback(async_task_);
    }
  }

  // the argument dependency goes to async_task
  template<typename T>
  void check_dependency(Future<T> *fut) {
    std::cout << "check dependency cudaTaskFn" << std::endl;
    if (!fut->probe()) {
      async_task_->inc();
      fut->register_callback(async_task_);
    }
  }

  // not only register final callback
  // but also submit the asyn task to taskq
  void register_submit_callback() override {
    this->get_world()->taskq.add(async_task_);
    std::cout << "add async task to taskq" << std::endl;
    TaskInterface::register_submit_callback();
  }

protected:
#ifdef HAVE_INTEL_TBB
  tbb::task* execute() override {
    std::cout << "run sync task" << std::endl;
    result_.set(async_result_);
    return nullptr;
  }
#else

  // when this starts to run means the output is ready
  void run(const TaskThreadEnv &env) override {
    std::cout << "run sync task" << std::endl;
    result_.set(async_result_);
  }

#endif

private:


  struct AsyncTaskInterface : public madness::TaskInterface {

    AsyncTaskInterface(cudaTaskFn_ *task, int ndpend = 0, const TaskAttributes attr = TaskAttributes()) : TaskInterface(
            ndpend, attr), task_(task) {}

  protected:

    void run(const TaskThreadEnv &env) override {

      // run the async function, the function must call synchronize_stream() to set the stream it used!!
      task_->run_async();

      // get the stream used by async function
      const cudaStream_t *stream = TiledArray::tls_cudastream_accessor();

      TA_ASSERT(stream != nullptr);

      std::cout << "insert callback to stream: " << *stream << std::endl;

      // insert cuda callback
      cudaStreamAddCallback(*stream, cuda_callback, task_, 0);
    }

  private:

    static void CUDART_CB cuda_callback(cudaStream_t stream, cudaError_t status,
                                        void *userData) {
      // convert void * to AsyncTaskInterface*
      AsyncTaskInterface* callback = static_cast<AsyncTaskInterface*>(userData);
      callback->notify();
    }

    cudaTaskFn_ *task_;

  };


  TaskInterface *async_task_; //

  futureT async_result_; // the future returned from the async task

};



} // end of namespace madness


int try_main(int argc, char**argv) {
  // Initialize runtime
  TiledArray::World &world = TiledArray::initialize(argc, argv);


  {
    const std::size_t M = 1000;
    const std::size_t N = 1000;

    TiledArray::Range range {M,N};

    tile_type tensor(range, 1.0);

    const double scale_factor = 2.0;

    // function pointer to the scale function to call
    tile_type (*scale_fn) (const tile_type&, const double) = &TiledArray::scale<tensor_type, double, nullptr>;

    madness::Future<tile_type> result;


    auto* scale_taskfn = new madness::cudaTaskFn<decltype(scale_fn),tile_type,double> (result, scale_fn, tensor, scale_factor, madness::TaskAttributes());


    auto scale_future = world.taskq.add(scale_taskfn);

    // this should start until scale_taskfn is finished
    world.taskq.add(verify, scale_future, scale_factor);

  }

  world.gop.fence();
//  auto tensor2 = tensor;
  // finalize runtime
  TiledArray::finalize();
  return 0;
}

int main(int argc, char *argv[]) {
  try {
    try_main(argc, argv);
  } catch (thrust::system::detail::bad_alloc &ex) {
    std::cout << ex.what() << std::endl;

    size_t free_mem, total_mem;
    auto result = cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "CUDA memory stats: {total,free} = {" << total_mem << ","
              << free_mem << "}" << std::endl;
  } catch (...) {
    std::cerr << "unknown exception" << std::endl;
  }

  return 0;
}