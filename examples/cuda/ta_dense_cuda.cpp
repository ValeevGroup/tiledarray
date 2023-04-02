/*
 * This file is a part of TiledArray.
 * Copyright (C) 2018  Virginia Tech
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

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <madness/config.h>

// clang-format off

#include <tiledarray.h>
#include <TiledArray/cuda/btas_um_tensor.h>
#include "TiledArray/cuda/cpu_cuda_vector.h"
#include <TiledArray/external/btas.h>
// clang-format on

#include <cuda_profiler_api.h>

namespace TiledArray {

///
/// cuda gemm interface function on left*right
///

template <typename T, typename Range>
btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>> gemm(
    const btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>> &left,
    const btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>> &right,
    T factor, const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(left, right, factor, gemm_helper);
}

///
/// cuda gemm interface function on result = left*right
///

template <typename T, typename Range>
void gemm(btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>> &result,
          const btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>> &left,
          const btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>> &right,
          T factor, const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(result, left, right, factor, gemm_helper);
}

///
/// cuda axpy interface function
///

template <typename T, typename Range>
void add_to(btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>> &result,
            const btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>> &arg) {
  btas_tensor_add_to_cuda_impl(result, arg, T(1.0));
}

///
/// cuda dot interface function
///

template <typename T, typename Range>
typename btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>>::value_type
squared_norm(
    const btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>> &arg) {
  return btas_tensor_squared_norm_cuda_impl(arg);
}

template <typename T, typename Range>
typename btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>>::value_type
norm(const btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>> &arg) {
  return std::sqrt(squared_norm(arg));
}

/// to host for CPU GPU Array
template <typename T, typename Range, typename Policy>
void to_host(
    TiledArray::DistArray<TiledArray::Tile<btas::Tensor<
                              T, Range, TiledArray::cpu_cuda_vector<T>>>,
                          Policy> &cpu_cuda_array) {
  auto to_host =
      [](TiledArray::Tile<
          btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>>> &tile) {
        auto &stream = detail::get_stream_based_on_range(tile.range());

        // do norm on GPU
        auto tile_norm = norm(tile.tensor());

        TiledArray::to_execution_space<TiledArray::ExecutionSpace::CPU>(
            tile.tensor().storage(), stream);

        return tile_norm;
      };

  foreach_inplace(cpu_cuda_array, to_host);
  cpu_cuda_array.world().gop.fence();
  cudaDeviceSynchronize();
};

/// to device for CPU GPU array
template <typename T, typename Range, typename Policy>
void to_device(
    TiledArray::DistArray<TiledArray::Tile<btas::Tensor<
                              T, Range, TiledArray::cpu_cuda_vector<T>>>,
                          Policy> &cpu_gpu_array) {
  auto to_device =
      [](TiledArray::Tile<
          btas::Tensor<T, Range, TiledArray::cpu_cuda_vector<T>>> &tile) {
        auto &stream = detail::get_stream_based_on_range(tile.range());

        TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
            tile.tensor().storage(), stream);

        return norm(tile.tensor());
      };

  foreach_inplace(cpu_gpu_array, to_device);
  cpu_gpu_array.world().gop.fence();
  cudaDeviceSynchronize();
};

}  // namespace TiledArray

template <typename Storage>
void do_main_body(TiledArray::World &world, const long Nm, const long Bm,
                  const long Nn, const long Bn, const long Nk, const long Bk,
                  const long nrepeat) {
  using T = TiledArray::detail::numeric_t<Storage>;
  using RT = TiledArray::detail::scalar_t<Storage>;
  constexpr auto complex_T = TiledArray::detail::is_complex_v<T>;

  const std::size_t Tm = Nm / Bm;
  const std::size_t Tn = Nn / Bn;
  const std::size_t Tk = Nk / Bk;

  const std::int64_t nflops =
      (complex_T ? 8 : 2)  // 1 multiply takes 6/1 flops for complex/real
                           // 1 add takes 2/1 flops for complex/real
      * static_cast<std::int64_t>(Nn) * static_cast<std::int64_t>(Nm) *
      static_cast<std::int64_t>(Nk);

  if (world.rank() == 0)
    std::cout << "TiledArray: dense matrix multiply test...\n"
              << "Number of nodes     = " << world.size()
              << "\nSize of A         = " << Nm << "x" << Nk << " ("
              << double(Nm * Nk * sizeof(T)) / 1.0e9 << " GB)"
              << "\nSize of A block   = " << Bm << "x" << Bk
              << "\nSize of B         = " << Nk << "x" << Nn << " ("
              << double(Nk * Nn * sizeof(T)) / 1.0e9 << " GB)"
              << "\nSize of B block   = " << Bk << "x" << Bn
              << "\nSize of C         = " << Nm << "x" << Nn << " ("
              << double(Nm * Nn * sizeof(T)) / 1.0e9 << " GB)"
              << "\nSize of C block   = " << Bm << "x" << Bn
              << "\n# of blocks of C  = " << Tm * Tn
              << "\nAverage # of blocks of C/node = "
              << double(Tm * Tn) / double(world.size()) << "\n";

  // Construct TiledRange
  std::vector<unsigned int> blocking_m;
  blocking_m.reserve(Tm + 1);
  for (long i = 0l; i <= Nm; i += Bm) blocking_m.push_back(i);

  std::vector<unsigned int> blocking_n;
  blocking_n.reserve(Tn + 1);
  for (long i = 0l; i <= Nn; i += Bn) blocking_n.push_back(i);

  std::vector<unsigned int> blocking_k;
  blocking_k.reserve(Tk + 1);
  for (long i = 0l; i <= Nk; i += Bk) blocking_k.push_back(i);

  // Structure of c
  std::vector<TiledArray::TiledRange1> blocking_C;
  blocking_C.reserve(2);
  blocking_C.push_back(
      TiledArray::TiledRange1(blocking_m.begin(), blocking_m.end()));
  blocking_C.push_back(
      TiledArray::TiledRange1(blocking_n.begin(), blocking_n.end()));

  // Structure of a
  std::vector<TiledArray::TiledRange1> blocking_A;
  blocking_A.reserve(2);
  blocking_A.push_back(
      TiledArray::TiledRange1(blocking_m.begin(), blocking_m.end()));
  blocking_A.push_back(
      TiledArray::TiledRange1(blocking_k.begin(), blocking_k.end()));

  // Structure of b
  std::vector<TiledArray::TiledRange1> blocking_B;
  blocking_B.reserve(2);
  blocking_B.push_back(
      TiledArray::TiledRange1(blocking_k.begin(), blocking_k.end()));
  blocking_B.push_back(
      TiledArray::TiledRange1(blocking_n.begin(), blocking_n.end()));

  TiledArray::TiledRange  // TRange for c
      trange_c(blocking_C.begin(), blocking_C.end());

  TiledArray::TiledRange  // TRange for a
      trange_a(blocking_A.begin(), blocking_A.end());

  TiledArray::TiledRange  // TRange for b
      trange_b(blocking_B.begin(), blocking_B.end());

  using CUDATile = btas::Tensor<T, TA::Range, Storage>;
  using CUDAMatrix = TA::DistArray<TA::Tile<CUDATile>>;
  using TAMatrix = TA::DistArray<TA::Tensor<T>>;

  CUDAMatrix c(world, trange_c);
  auto val_a = 0.03;
  auto val_b = 0.02;

  {
    // Construct and initialize arrays

    TAMatrix a_host(world, trange_a);
    TAMatrix b_host(world, trange_b);

    a_host.fill(val_a);
    b_host.fill(val_b);
    CUDAMatrix a = TA::ta_tensor_to_um_tensor<TA::Tile<CUDATile>>(a_host);
    CUDAMatrix b = TA::ta_tensor_to_um_tensor<TA::Tile<CUDATile>>(b_host);

    world.gop.fence();

    //    TA::to_device(a);
    //    TA::to_device(b);

    //    c("m,n") = a("m,k") * b("k,n");

    // start profiler
    cudaProfilerStart();

    double total_time = 0.0;
    double total_gflop_rate = 0.0;

    // Do matrix multiplication
    for (int i = 0; i < nrepeat; ++i) {
      double iter_time_start = madness::wall_time();
      //      c("m,n") = a("m,k") * b("k,n") + a("m,n") - b("m,n");
      c("m,n") = a("m,k") * b("k,n");
      c.world().gop.fence();  // fence since GEMM can return early
      double iter_time_stop = madness::wall_time();
      const double iter_time = iter_time_stop - iter_time_start;
      total_time += iter_time;
      const double gflop_rate = double(nflops) / (iter_time * 1.e9);
      total_gflop_rate += gflop_rate;
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1 << " wall time: " << iter_time
                  << "\n";
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1 << "   time=" << time
                  << "   GFLOPS=" << gflop_rate << "\n";
    }
    // Stop clock
    const double wall_time_stop = madness::wall_time();

    // stop profiler
    cudaProfilerStop();

    if (world.rank() == 0)
      std::cout << "Average wall time   = " << total_time / double(nrepeat)
                << " sec\nAverage GFLOPS      = "
                << total_gflop_rate / double(nrepeat) << "\n";
  }

  double threshold = std::numeric_limits<RT>::epsilon();
  auto dot_length = Nk;
  //  auto result = dot_length * val_a * val_b + val_a - val_b;
  T result;
  if constexpr (complex_T) {
    result = T(dot_length * val_a * val_b, 0.);
  } else
    result = dot_length * val_a * val_b;

  auto verify = [&world, &threshold, &result,
                 &dot_length](TA::Tile<CUDATile> &tile) {
    auto n_elements = tile.size();
    for (std::size_t i = 0; i < n_elements; i++) {
      double abs_err = std::abs(tile[i] - result);
      //      double abs_val = fabs(tile[i]);
      double rel_err = abs_err / std::abs(result) / dot_length;
      if (rel_err > threshold) {
        auto to_string = [](const auto &v) {
          constexpr bool complex_T =
              TiledArray::detail::is_complex_v<std::decay_t<decltype(v)>>;
          if constexpr (complex_T) {
            std::string result;
            result = "{" + std::to_string(v.real()) + "," +
                     std::to_string(v.imag()) + "}";
            return result;
          } else
            return std::to_string(v);
        };
        std::cout << "Node: " << world.rank() << " Tile: " << tile.range()
                  << " id: " << i
                  << std::string(" gpu: " + to_string(tile[i]) +
                                 " cpu: " + to_string(result) + "\n");
        break;
      }
    }
  };

  for (auto iter = c.begin(); iter != c.end(); iter++) {
    world.taskq.add(verify, c.find(iter.index()));
  }

  world.gop.fence();

  if (world.rank() == 0) {
    std::cout << "Verification Passed" << std::endl;
  }
}

int try_main(int argc, char **argv) {
  // Initialize runtime
  TiledArray::World &world = TA_SCOPED_INITIALIZE(argc, argv);

  // Get command line arguments
  if (argc < 6) {
    std::cout << "multiplies A(Nm,Nk) * B(Nk,Nn), with dimensions m, n, and k "
                 "blocked by Bm, Bn, and Bk, respectively"
              << std::endl
              << "Usage: " << argv[0]
              << " Nm Bm Nn Bn Nk Bk [# of repetitions = 5] [scalar = double] "
                 "[storage type = cuda_um_btas_varray]\n";
    return 0;
  }
  const long Nm = atol(argv[1]);
  const long Bm = atol(argv[2]);
  const long Nn = atol(argv[3]);
  const long Bn = atol(argv[4]);
  const long Nk = atol(argv[5]);
  const long Bk = atol(argv[6]);
  if (Nm <= 0 || Nn <= 0 || Nk <= 0) {
    std::cerr << "Error: dimensions must be greater than zero.\n";
    return 1;
  }
  if (Bm <= 0 || Bn <= 0 || Bk <= 0) {
    std::cerr << "Error: block sizes must be greater than zero.\n";
    return 1;
  }
  if ((Nm % Bm) != 0ul || Nn % Bn != 0ul || Nk % Bk != 0ul) {
    std::cerr
        << "Error: dimension size must be evenly divisible by block size.\n";
    return 1;
  }
  const long nrepeat = (argc >= 8 ? atol(argv[7]) : 5);
  if (nrepeat <= 0) {
    std::cerr << "Error: number of repetitions must be greater than zero.\n";
    return 1;
  }

  const std::string scalar_type_str = (argc >= 9 ? argv[8] : "double");
  if (scalar_type_str != "double" && scalar_type_str != "float" &&
      scalar_type_str != "zdouble" && scalar_type_str != "zfloat") {
    std::cerr << "Error: invalid real type " << scalar_type_str << ".\n";
    std::cerr << "       valid real types are \"double\", \"float\", "
                 "\"zdouble\", and \"zfloat\".\n";
    return 1;
  }

  const auto storage_type =
      (argc >= 10) ? std::string(argv[9]) : std::string{"cuda_um_btas_varray"};

  if (storage_type != "cuda_um_btas_varray" &&
      storage_type != "cuda_um_thrust_vector" &&
      storage_type != "cpu_cuda_vector") {
    std::cerr << "Error: invalid storage type: " << storage_type
              << "\n Valid option includes: cuda_um_vector or "
                 "cuda_um_btas_varray or cuda_um_thrust_vector "
                 "or cpu_cuda_vector. \n";
  }
  std::cout << "Storage type: " << storage_type << "<" << scalar_type_str << ">"
            << std::endl;
  //  auto to_bool = [](const std::string &str) {
  //    return (str == "true" || str == "True" || str == "TRUE" || str == "1" ||
  //            str == "yes" || str == "Yes" || str == "YES");
  //  };

  int driverVersion, runtimeVersion;
  auto error = cudaDriverGetVersion(&driverVersion);
  if (error != cudaSuccess) {
    std::cout << "error(cudaDriverGetVersion) = " << error << std::endl;
  }
  error = cudaRuntimeGetVersion(&runtimeVersion);
  if (error != cudaSuccess) {
    std::cout << "error(cudaRuntimeGetVersion) = " << error << std::endl;
  }
  std::cout << "CUDA {driver,runtime} versions = " << driverVersion << ","
            << runtimeVersion << std::endl;

  {  // print device properties
    int num_cuda_devices = TA::cudaEnv::instance()->num_cuda_devices();

    if (num_cuda_devices <= 0) {
      throw std::runtime_error("No CUDA-Enabled GPUs Found!\n");
    }

    int cuda_device_id = TA::cudaEnv::instance()->current_cuda_device_id();

    int mpi_size = world.size();
    int mpi_rank = world.rank();

    for (int i = 0; i < mpi_size; i++) {
      if (i == mpi_rank) {
        std::cout << "CUDA Device Information for MPI Process Rank: "
                  << mpi_rank << std::endl;
        cudaDeviceProp prop;
        auto error = cudaGetDeviceProperties(&prop, cuda_device_id);
        if (error != cudaSuccess) {
          std::cout << "error(cudaGetDeviceProperties) = " << error
                    << std::endl;
        }
        std::cout << "Device #" << cuda_device_id << ": " << prop.name
                  << std::endl
                  << "  managedMemory = " << prop.managedMemory << std::endl
                  << "  singleToDoublePrecisionPerfRatio = "
                  << prop.singleToDoublePrecisionPerfRatio << std::endl;
        int result;
        error = cudaDeviceGetAttribute(&result, cudaDevAttrUnifiedAddressing,
                                       cuda_device_id);
        std::cout << "  attrUnifiedAddressing = " << result << std::endl;
        error = cudaDeviceGetAttribute(
            &result, cudaDevAttrConcurrentManagedAccess, cuda_device_id);
        std::cout << "  attrConcurrentManagedAccess = " << result << std::endl;
        error = cudaSetDevice(cuda_device_id);
        if (error != cudaSuccess) {
          std::cout << "error(cudaSetDevice) = " << error << std::endl;
        }
        size_t free_mem, total_mem;
        error = cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "  {total,free} memory = {" << total_mem << "," << free_mem
                  << "}" << std::endl;
      }
      world.gop.fence();
    }
  }  // print device properties

  //  if (storage_type == "cpu_cuda_vector") {
  //    if (scalar_type_str == "double")
  //      do_main_body<TiledArray::cpu_cuda_vector<double>>(world, Nm, Bm, Nn,
  //      Bn,
  //                                                        Nk, Bk, nrepeat);
  //    else
  //      do_main_body<TiledArray::cpu_cuda_vector<float>>(world, Nm, Bm, Nn,
  //      Bn,
  //                                                       Nk, Bk, nrepeat);
  //  } else if (storage_type == "cuda_um_btas_varray") {
  if (storage_type == "cuda_um_btas_varray") {
    if (scalar_type_str == "double")
      do_main_body<TiledArray::cuda_um_btas_varray<double>>(
          world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
    else if (scalar_type_str == "float")
      do_main_body<TiledArray::cuda_um_btas_varray<float>>(world, Nm, Bm, Nn,
                                                           Bn, Nk, Bk, nrepeat);
    else if (scalar_type_str == "zdouble")
      do_main_body<TiledArray::cuda_um_btas_varray<std::complex<double>>>(
          world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
    else if (scalar_type_str == "zfloat")
      do_main_body<TiledArray::cuda_um_btas_varray<std::complex<float>>>(
          world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
    else {
      abort();  // unreachable
    }
  }
  // else if (storage_type == "cuda_um_thrust_vector") {
  //    if (scalar_type_str == "double")
  //      do_main_body<TiledArray::cuda_um_thrust_vector<double>>(
  //          world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
  //    else
  //      do_main_body<TiledArray::cuda_um_thrust_vector<float>>(
  //          world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
  //  }
  else {
    throw std::runtime_error("Invalid storage type!\n");
  }

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
  } catch (std::exception &ex) {
    std::cout << ex.what() << std::endl;
  } catch (...) {
    std::cerr << "unknown exception" << std::endl;
  }

  return 0;
}
