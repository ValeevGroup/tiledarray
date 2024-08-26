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

// clang-format off

#include <tiledarray.h>
#include <TiledArray/device/btas_um_tensor.h>
#include <TiledArray/external/btas.h>
// clang-format on

#ifdef TILEDARRAY_HAS_CUDA
#include <cuda_profiler_api.h>
#endif  // TILEDARRAY_HAS_CUDA

template <typename Storage>
void do_main_body(TiledArray::World &world, const long Nm, const long Bm,
                  const long Nn, const long Bn, const long Nk, const long Bk,
                  const long nrepeat) {
  using T = TiledArray::detail::numeric_t<Storage>;
  using RT = TiledArray::detail::scalar_t<Storage>;
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
    std::cout << "TiledArray: dense matrix multiply test...\n"
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

  TiledArray::TiledRange  // TRange for c
      trange_c(blocking_C.begin(), blocking_C.end());

  TiledArray::TiledRange  // TRange for a
      trange_a(blocking_A.begin(), blocking_A.end());

  TiledArray::TiledRange  // TRange for b
      trange_b(blocking_B.begin(), blocking_B.end());

  using DeviceTile = btas::Tensor<T, TA::Range, Storage>;
  using DeviceMatrix = TA::DistArray<TA::Tile<DeviceTile>>;
  using PinnedTile =
      btas::Tensor<T, TA::Range,
                   ::btas::varray<typename Storage::value_type,
                                  TiledArray::device_pinned_allocator<T>>>;
  using PinnedMatrix = TA::DistArray<TA::Tile<PinnedTile>>;
  // using TAMatrix = TA::DistArray<TA::Tensor<T>>;

  DeviceMatrix c(world, trange_c);
  auto val_a = 0.03;
  auto val_b = 0.02;

  {
    // Construct and initialize arrays

    PinnedMatrix a_host(world, trange_a);
    PinnedMatrix b_host(world, trange_b);

    a_host.fill(val_a);
    b_host.fill(val_b);
    DeviceMatrix a = TA::ta_tensor_to_um_tensor<TA::Tile<DeviceTile>>(a_host);
    DeviceMatrix b = TA::ta_tensor_to_um_tensor<TA::Tile<DeviceTile>>(b_host);

    world.gop.fence();

    //    TA::to_device(a);
    //    TA::to_device(b);

    //    c("m,n") = a("m,k") * b("k,n");

#ifdef TILEDARRAY_HAS_CUDA
    // start profiler
    cudaProfilerStart();
#endif  // TILEDARRAY_HAS_CUDA

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
  //  auto result = dot_length * val_a * val_b + val_a - val_b;
  T result;
  if constexpr (complex_T) {
    result = T(dot_length * val_a * val_b, 0.);
  } else
    result = dot_length * val_a * val_b;

  auto verify = [&world, &threshold, &result,
                 &dot_length](TA::Tile<DeviceTile> &tile) {
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
                 "[storage type = device_um_btas_varray]\n";
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

  const auto storage_type = (argc >= 10) ? std::string(argv[9])
                                         : std::string{"device_um_btas_varray"};

  if (storage_type != "device_um_btas_varray") {
    std::cerr << "Error: invalid storage type: " << storage_type
              << "\n Valid option includes: "
                 "device_um_btas_varray \n";
  }
  std::cout << "Storage type: " << storage_type << "<" << scalar_type_str << ">"
            << std::endl;
  //  auto to_bool = [](const std::string &str) {
  //    return (str == "true" || str == "True" || str == "TRUE" || str == "1" ||
  //            str == "yes" || str == "Yes" || str == "YES");
  //  };

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

  if (storage_type == "device_um_btas_varray") {
    if (scalar_type_str == "double")
      do_main_body<TiledArray::device_um_btas_varray<double>>(
          world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
    else if (scalar_type_str == "float")
      do_main_body<TiledArray::device_um_btas_varray<float>>(
          world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
    else if (scalar_type_str == "zdouble")
      do_main_body<TiledArray::device_um_btas_varray<std::complex<double>>>(
          world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
    else if (scalar_type_str == "zfloat")
      do_main_body<TiledArray::device_um_btas_varray<std::complex<float>>>(
          world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
    else {
      abort();  // unreachable
    }
  } else {
    throw std::runtime_error("Invalid storage type!\n");
  }

  return 0;
}

int main(int argc, char *argv[]) {
  try {
    try_main(argc, argv);
  } catch (std::exception &ex) {
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
