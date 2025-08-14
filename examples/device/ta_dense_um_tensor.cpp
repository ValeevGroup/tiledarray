/*
 * This file is a part of TiledArray.
 * Copyright (C) 2025  Virginia Tech
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

// clang-format off
#include <tiledarray.h>
#include <TiledArray/device/um_tensor.h>
// clang-format on

#ifdef TILEDARRAY_HAS_CUDA
#include <cuda_profiler_api.h>
#endif  // TILEDARRAY_HAS_CUDA

template <typename T>
void do_main_body(TiledArray::World& world, const long Nm, const long Bm,
                  const long Nn, const long Bn, const long Nk, const long Bk,
                  const long nrepeat) {
  using RT = TiledArray::detail::scalar_t<T>;
  constexpr auto complex_T = TiledArray::detail::is_complex_v<T>;

  const std::int64_t nflops =
      (complex_T ? 8 : 2)  // 1 multiply takes 6/1 flops for complex/real
                           // 1 add takes 2/1 flops for complex/real
      * static_cast<std::int64_t>(Nn) * static_cast<std::int64_t>(Nm) *
      static_cast<std::int64_t>(Nk);

  // Construct TiledRange
  std::vector<unsigned int> blocking_m;
  for (long i = 0l; i <= Nm; i += Bm) blocking_m.push_back(i);
  const std::size_t Tm = blocking_m.size() - 1;

  std::vector<unsigned int> blocking_n;
  for (long i = 0l; i <= Nn; i += Bn) blocking_n.push_back(i);
  const std::size_t Tn = blocking_n.size() - 1;

  std::vector<unsigned int> blocking_k;
  for (long i = 0l; i <= Nk; i += Bk) blocking_k.push_back(i);
  const std::size_t Tk = blocking_k.size();

  if (world.rank() == 0)
    std::cout << "TiledArray: UMTensor dense matrix multiply test...\n"
              << "Number of nodes     = " << world.size()
              << "\nSize of A         = " << Nm << "x" << Nk << " ("
              << double(Nm * Nk * sizeof(T)) / 1.0e9 << " GB)"
              << "\nSize of (largest) A block   = " << Bm << "x" << Bk
              << "\nSize of B         = " << Nk << "x" << Nn << " ("
              << double(Nk * Nn * sizeof(T)) / 1.0e9 << " GB)"
              << "\nSize of (largest) B block   = " << Bk << "x" << Bn
              << "\nSize of C         = " << Nm << "x" << Nn << " ("
              << double(Nm * Nn * sizeof(T)) / 1.0e9 << " GB)"
              << "\nSize of (largest) C block   = " << Bm << "x" << Bn
              << "\n# of blocks of C  = " << Tm * Tn
              << "\nAverage # of blocks of C/node = "
              << double(Tm * Tn) / double(world.size()) << "\n";

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

  TiledArray::TiledRange trange_c(blocking_C.begin(), blocking_C.end());

  TiledArray::TiledRange trange_a(blocking_A.begin(), blocking_A.end());

  TiledArray::TiledRange trange_b(blocking_B.begin(), blocking_B.end());

  using DeviceTile = TA::UMTensor<T>;
  using DeviceMatrix = TA::DistArray<TA::Tile<DeviceTile>>;
  using HostTensor = TA::Tensor<T>; // Should this be a PinnedTile ? Or is it okay because we call to_device on Tiles anyway?
  using HostMatrix = TA::DistArray<HostTensor>;

  DeviceMatrix c(world, trange_c);
  auto val_a = 0.03;
  auto val_b = 0.02;

  {
    // Construct and initialize arrays on host first
    HostMatrix a_host(world, trange_a);
    HostMatrix b_host(world, trange_b);

    a_host.fill(val_a);
    b_host.fill(val_b);

    // Convert to UMTensor arrays
    DeviceMatrix a(world, trange_a);
    DeviceMatrix b(world, trange_b);

    // Copy data from host to device tensors
    // TODO: Wrap this into a reusable function
    for (auto it = a_host.begin(); it != a_host.end(); ++it) {
      const auto& index = it.index();
      const auto& host_tile_ref = *it;
      const auto& host_tile =
          host_tile_ref.get();  // Get actual tensor from reference

      DeviceTile device_tile(host_tile.range());

      std::copy(host_tile.data(), host_tile.data() + host_tile.size(),
                device_tile.data());
      TiledArray::detail::to_device(device_tile);

      a.set(index, TA::Tile<DeviceTile>(std::move(device_tile)));
    }

    for (auto it = b_host.begin(); it != b_host.end(); ++it) {
      const auto& index = it.index();
      const auto& host_tile_ref = *it;
      const auto& host_tile =
          host_tile_ref.get();  // Get actual tensor from reference
      DeviceTile device_tile(host_tile.range());

      std::copy(host_tile.data(), host_tile.data() + host_tile.size(),
                device_tile.data());

      TiledArray::detail::to_device(device_tile);

      b.set(index, TA::Tile<DeviceTile>(std::move(device_tile)));
    }

    world.gop.fence();

#ifdef TILEDARRAY_HAS_CUDA
    // start profiler
    cudaProfilerStart();
#endif  // TILEDARRAY_HAS_CUDA

    double total_time = 0.0;
    double total_gflop_rate = 0.0;

    // Do matrix multiplication
    for (int i = 0; i < nrepeat; ++i) {
      double iter_time_start = madness::wall_time();
      c("m,n") = a("m,k") * b("k,n");
      c.world().gop.fence();  // fence since GEMM can return early
      double iter_time_stop = madness::wall_time();
      const double iter_time = iter_time_stop - iter_time_start;
      total_time += iter_time;
      const double gflop_rate = double(nflops) / (iter_time * 1.e9);
      total_gflop_rate += gflop_rate;
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1 << " wall time: " << iter_time
                  << " sec\n";
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1 << "   GFLOPS=" << gflop_rate
                  << "\n";
    }

#ifdef TILEDARRAY_HAS_CUDA
    // stop profiler
    cudaProfilerStop();
#endif  // TILEDARRAY_HAS_CUDA

    if (world.rank() == 0)
      std::cout << "Average wall time   = " << total_time / double(nrepeat)
                << " sec\nAverage GFLOPS      = "
                << total_gflop_rate / double(nrepeat) << "\n";
  }

  double threshold = std::numeric_limits<RT>::epsilon();
  auto dot_length = Nk;
  T result;
  if constexpr (complex_T) {
    result = T(dot_length * val_a * val_b, 0.);
  } else
    result = dot_length * val_a * val_b;

  auto verify = [&world, &threshold, &result,
                 &dot_length](TA::Tile<DeviceTile>& tile) {
    auto& um_tensor = tile.tensor();
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::Host>(
        um_tensor, TiledArray::device::stream_for(um_tensor.range()));
    TiledArray::device::sync_madness_task_with(
        TiledArray::device::stream_for(um_tensor.range()));

    auto n_elements = tile.size();
    for (std::size_t i = 0; i < n_elements; i++) {
      double abs_err = std::abs(tile[i] - result);
      double rel_err = abs_err / std::abs(result) / dot_length;
      if (rel_err > threshold) {
        auto to_string = [](const auto& v) {
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

int try_main(int argc, char** argv) {
  // Initialize runtime
  TiledArray::World& world = TA_SCOPED_INITIALIZE(argc, argv);

  // Get command line arguments
  if (argc < 6) {
    std::cout
        << "multiplies A(Nm,Nk) * B(Nk,Nn), with dimensions m, n, and k "
           "blocked by Bm, Bn, and Bk, respectively"
        << std::endl
        << "Usage: " << argv[0]
        << " Nm Bm Nn Bn Nk Bk [# of repetitions = 5] [scalar = double]\n";
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

  std::cout << "Using TA::UMTensor<" << scalar_type_str << ">" << std::endl;

  int driverVersion, runtimeVersion;
  auto error = TiledArray::device::driverVersion(&driverVersion);
  if (error != TiledArray::device::Success) {
    std::cout << "error(DriverGetVersion) = " << error << std::endl;
  }
  error = TiledArray::device::runtimeVersion(&runtimeVersion);
  if (error != TiledArray::device::Success) {
    std::cout << "error(RuntimeGetVersion) = " << error << std::endl;
  }
  std::cout << "device {driver,runtime} versions = " << driverVersion << ","
            << runtimeVersion << std::endl;

  {  // print device properties
    int num_devices = TA::deviceEnv::instance()->num_visible_devices();

    if (num_devices <= 0) {
      throw std::runtime_error("No GPUs Found!\n");
    }

    const int device_id = TA::deviceEnv::instance()->current_device_id();

    int mpi_size = world.size();
    int mpi_rank = world.rank();

    for (int i = 0; i < mpi_size; i++) {
      if (i == mpi_rank) {
        std::cout << "Device Information for MPI Process Rank: " << mpi_rank
                  << std::endl;
        TiledArray::device::deviceProp_t prop;
        auto error = TiledArray::device::getDeviceProperties(&prop, device_id);
        if (error != TiledArray::device::Success) {
          std::cout << "error(GetDeviceProperties) = " << error << std::endl;
        }
        std::cout << "Device #" << device_id << ": " << prop.name << std::endl
                  << "  managedMemory = " << prop.managedMemory << std::endl;
        int result;
        error = TiledArray::device::deviceGetAttribute(
            &result, TiledArray::device::DevAttrUnifiedAddressing, device_id);
        std::cout << "  attrUnifiedAddressing = " << result << std::endl;
        error = TiledArray::device::deviceGetAttribute(
            &result, TiledArray::device::DevAttrConcurrentManagedAccess,
            device_id);
        std::cout << "  attrConcurrentManagedAccess = " << result << std::endl;
        error = TiledArray::device::setDevice(device_id);
        if (error != TiledArray::device::Success) {
          std::cout << "error(device::setDevice) = " << error << std::endl;
        }
        size_t free_mem, total_mem;
        error = TiledArray::device::memGetInfo(&free_mem, &total_mem);
        std::cout << "  {total,free} memory = {" << total_mem << "," << free_mem
                  << "}" << std::endl;
      }
      world.gop.fence();
    }
  }  // print device properties

  if (scalar_type_str == "double")
    do_main_body<double>(world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
  else if (scalar_type_str == "float")
    do_main_body<float>(world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
  else if (scalar_type_str == "zdouble")
    do_main_body<std::complex<double>>(world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
  else if (scalar_type_str == "zfloat")
    do_main_body<std::complex<float>>(world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
  else {
    abort();  // unreachable
  }

  return 0;
}

int main(int argc, char* argv[]) {
  try {
    try_main(argc, argv);
  } catch (std::exception& ex) {
    std::cout << ex.what() << std::endl;

    size_t free_mem, total_mem;
    auto result = TiledArray::device::memGetInfo(&free_mem, &total_mem);
    std::cout << "device memory stats: {total,free} = {" << total_mem << ","
              << free_mem << "}" << std::endl;
  } catch (...) {
    std::cerr << "unknown exception" << std::endl;
  }

  return 0;
}
