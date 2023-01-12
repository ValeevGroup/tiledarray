//
// Created by Chong Peng on 11/14/18.
//

#include <TiledArray/gpu/btas_um_tensor.h>
#include <TiledArray/gpu/gpu_task_fn.h>
#include <tiledarray.h>

using value_type = double;
using tensor_type = TA::btasUMTensorVarray<value_type>;
using tile_type = TA::Tile<tensor_type>;

/// verify the elements in tile is equal to value
void verify(const tile_type& tile, value_type value, std::size_t index) {
  //  const auto size = tile.size();
  std::string message = "verify Tensor: " + std::to_string(index) + '\n';
  std::cout << message;
  for (auto& num : tile) {
    if (num != value) {
      std::string error("Error: " + std::to_string(num) + " " +
                        std::to_string(value) +
                        " Tensor: " + std::to_string(index) + "\n");
      std::cout << error;
    }
    break;
  }
}

tile_type scale(const tile_type& arg, value_type a, const cudaStream_t* stream,
                std::size_t index) {
  CudaSafeCall(
      cudaSetDevice(TiledArray::cudaEnv::instance()->current_cuda_device_id()));
  /// make result Tensor
  using Storage = typename tile_type::tensor_type::storage_type;
  Storage result_storage;
  auto result_range = arg.range();
  make_device_storage(result_storage, arg.size(), *stream);

  typename tile_type::tensor_type result(std::move(result_range),
                                         std::move(result_storage));

  /// copy the original Tensor
  const auto& handle = TiledArray::cuBLASHandlePool::handle();
  CublasSafeCall(cublasSetStream(handle, *stream));

  CublasSafeCall(TiledArray::cublasCopy(handle, result.size(), arg.data(), 1,
                                        device_data(result.storage()), 1));

  CublasSafeCall(TiledArray::cublasScal(handle, result.size(), &a,
                                        device_data(result.storage()), 1));

  //  cudaStreamSynchronize(stream);

  TiledArray::synchronize_stream(stream);

  //  std::stringstream stream_str;
  //  stream_str << *stream;
  //  std::string message = "run scale on Tensor: " + std::to_string(index) +  "
  //  on stream: " + stream_str.str() +'\n'; std::cout << message;
  return tile_type(std::move(result));
}

void process_task(madness::World* world,
                  const std::vector<cudaStream_t>* streams, std::size_t ntask) {
  const std::size_t iter = 50;
  const std::size_t M = 1000;
  const std::size_t N = 1000;

  std::size_t n_stream = streams->size();

  for (std::size_t i = 0; i < iter; i++) {
    auto& stream = (*streams)[i % n_stream];

    TiledArray::Range range{M, N};

    tile_type tensor(range, 1.0);

    const double scale_factor = 2.0;

    // function pointer to the scale function to call
    tile_type (*scale_fn)(const tile_type&, double, const cudaStream_t*,
                          std::size_t) = &::scale;

    madness::Future<tile_type> scale_future = madness::add_cuda_task(
        *world, ::scale, tensor, scale_factor, &stream, ntask * iter + i);

    /// this should start until scale_taskfn is finished
    world->taskq.add(verify, scale_future, scale_factor, ntask * iter + i);
  }
}

int try_main(int argc, char** argv) {
  auto& world = TiledArray::get_default_world();

  const std::size_t n_stream = 5;
  const std::size_t n_tasks = 5;

  std::vector<cudaStream_t> streams(n_stream);
  for (auto& stream : streams) {
    // create the streams
    CudaSafeCall(cudaStreamCreate(&stream));
    //    std::cout << "stream: " << stream << "\n";
  }

  // add process_task to different tasks/threads
  for (auto i = 0; i < n_tasks; i++) {
    world.taskq.add(process_task, &world, &streams, i);
  }

  world.gop.fence();

  for (auto& stream : streams) {
    // create the streams
    cudaStreamDestroy(stream);
  }
  return 0;
}

int main(int argc, char* argv[]) {
  TiledArray::World& world = TA_SCOPED_INITIALIZE(argc, argv);
  try {
    // Initialize runtime
    try_main(argc, argv);
  } catch (thrust::system::detail::bad_alloc& ex) {
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
