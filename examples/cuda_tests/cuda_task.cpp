//
// Created by Chong Peng on 11/14/18.
//

#include <tiledarray.h>
#include <TiledArray/cuda/btas_um_tensor.h>
#include <TiledArray/cuda/cudaTaskFn.h>

using value_type = double;
using tensor_type = TA::btasUMTensorVarray<value_type>;
using tile_type = TA::Tile<tensor_type>;

/// verify the elements in tile is equal to value
void verify(const tile_type& tile, value_type value) {
  //  const auto size = tile.size();
  std::cout << "verify" << std::endl;
  for (auto& num : tile) {
    if (num != value) {
      std::cout << "Error: " << num << " " << value << std::endl;
    }
    break;
  }
}

tile_type scale(const tile_type& arg, value_type a,
                const cudaStream_t& stream) {
  /// make result Tensor
  using Storage = typename tile_type::tensor_type::storage_type;
  Storage result_storage;
  auto result_range = arg.range();
  make_device_storage(result_storage, arg.size(), stream);

  typename tile_type::tensor_type result(std::move(result_range),
                                         std::move(result_storage));

  /// copy the original Tensor
  const auto& handle = TiledArray::cuBLASHandlePool::handle();
  CublasSafeCall(cublasSetStream(handle, stream));

  CublasSafeCall(TiledArray::cublasCopy(handle, result.size(), arg.data(), 1,
                                        device_data(result.storage()), 1));

  /// scale the Tensor
  CublasSafeCall(TiledArray::cublasScal(handle, result.size(), &a,
                                        device_data(result.storage()), 1));

  TiledArray::synchronize_stream(&stream);

  return tile_type(result);
}

int try_main(int argc, char** argv) {
  // Initialize runtime
  TiledArray::World& world = TiledArray::initialize(argc, argv);

  const std::size_t n_stream = 2;
  const std::size_t iter = 20;
  const std::size_t M = 1000;
  const std::size_t N = 1000;

  std::vector<cudaStream_t> streams(2);
  for (auto& stream : streams) {
    // create the streams
    cudaStreamCreate(&stream);
  }

  for (std::size_t i = 0; i < iter; i++) {
    auto& stream = streams[i % n_stream];

    TiledArray::Range range{M, N};

    tile_type tensor(range, 1.0);

    const double scale_factor = 2.0;

    // function pointer to the scale function to call
    tile_type (*scale_fn)(const tile_type&, double,
                            const cudaStream_t&) = &::scale;

    madness::Future<tile_type> result;

    auto* scale_taskfn = new madness::cudaTaskFn<decltype(scale_fn), tile_type,
                                                 double, cudaStream_t>(
        result, scale_fn, tensor, scale_factor, stream,
        madness::TaskAttributes());

    /// add the cudaTaskFn
    auto scale_future = scale_taskfn->result();
    world.taskq.add(static_cast<madness::TaskInterface*>(scale_taskfn));

    /// this should start until scale_taskfn is finished
    world.taskq.add(verify, scale_future, scale_factor);
  }

  world.gop.fence();
  TiledArray::finalize();

  for (auto& stream : streams) {
    // create the streams
    cudaStreamDestroy(stream);
  }
  return 0;
}

int main(int argc, char* argv[]) {
  try {
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