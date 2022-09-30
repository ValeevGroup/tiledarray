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

template <typename Tile>
void do_main_body(TiledArray::World &world, const long Nm, const long Bm,
                  const long Nn, const long Bn, const long nrepeat) {
  const std::size_t Tm = Nm / Bm;
  const std::size_t Tn = Nn / Bn;

  if (world.rank() == 0)
    std::cout << "TiledArray: dense matrix vector test...\n"
              << "Number of nodes     = " << world.size()
              << "\nSize of Matrix         = " << Nm << "x" << Nn << " ("
              << double(Nm * Nn * sizeof(double)) / 1.0e9 << " GB)"
              << "\nSize of Block   = " << Bm << "x" << Bn
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

  // Structure of Matrix
  std::vector<TiledArray::TiledRange1> blocking;
  blocking.reserve(2);
  blocking.push_back(
      TiledArray::TiledRange1(blocking_m.begin(), blocking_m.end()));
  blocking.push_back(
      TiledArray::TiledRange1(blocking_n.begin(), blocking_n.end()));

  TiledArray::TiledRange  // TRange
      trange(blocking.begin(), blocking.end());

  using value_type = typename Tile::value_type;
  using TArray = TA::DistArray<Tile, TA::DensePolicy>;

  TArray c(world, trange);
  value_type val_a = 0.03;
  value_type val_b = 0.02;

  {
    if (world.rank() == 0) {
      std::cout << "\nAdd test: a(m,n) + b(m,n)\n";
    }

    TArray a(world, trange);
    TArray b(world, trange);

    a.fill(val_a);
    b.fill(val_b);

    // Start clock
    const double wall_time_start = madness::wall_time();

    // Do
    for (int i = 0; i < nrepeat; ++i) {
      double iter_time_start = madness::wall_time();
      c("m,n") = a("m,n") + b("m,n");
      double iter_time_stop = madness::wall_time();
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1
                  << " wall time: " << (iter_time_stop - iter_time_start)
                  << "\n";
    }
    // Stop clock
    const double wall_time_stop = madness::wall_time();

    if (world.rank() == 0)
      std::cout << "Average wall time   = "
                << (wall_time_stop - wall_time_start) / double(nrepeat)
                << " sec\nAverage GFLOPS      = "
                << double(nrepeat) * double(Nn * Nm) /
                       (wall_time_stop - wall_time_start) / 1.0e9
                << "\n";
  }

  {
    if (world.rank() == 0) {
      std::cout << "\nAdd scale test: 2*a(m,n) + 2*b(m,n)\n";
    }

    TArray a(world, trange);
    TArray b(world, trange);

    a.fill(val_a);
    b.fill(val_b);

    // Start clock
    const double wall_time_start = madness::wall_time();

    // Do
    for (int i = 0; i < nrepeat; ++i) {
      double iter_time_start = madness::wall_time();
      c("m,n") = 2 * a("m,n") + 2 * b("m,n");
      double iter_time_stop = madness::wall_time();
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1
                  << " wall time: " << (iter_time_stop - iter_time_start)
                  << "\n";
    }
    // Stop clock
    const double wall_time_stop = madness::wall_time();

    if (world.rank() == 0)
      std::cout << "Average wall time   = "
                << (wall_time_stop - wall_time_start) / double(nrepeat)
                << " sec\nAverage GFLOPS      = "
                << double(nrepeat) * 3 * double(Nn * Nm) /
                       (wall_time_stop - wall_time_start) / 1.0e9
                << "\n";
  }

  {
    if (world.rank() == 0) {
      std::cout << "\nAdd permute test: 2*a(m,n) + 2*b(n,m)\n";
    }

    TArray a(world, trange);
    TArray b(world, trange);

    a.fill(val_a);
    b.fill(val_b);

    // Start clock
    const double wall_time_start = madness::wall_time();

    // Do
    for (int i = 0; i < nrepeat; ++i) {
      double iter_time_start = madness::wall_time();
      c("m,n") = 2 * a("m,n") + 2 * b("n,m");
      double iter_time_stop = madness::wall_time();
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1
                  << " wall time: " << (iter_time_stop - iter_time_start)
                  << "\n";
    }
    // Stop clock
    const double wall_time_stop = madness::wall_time();

    if (world.rank() == 0)
      std::cout << "Average wall time   = "
                << (wall_time_stop - wall_time_start) / double(nrepeat)
                << " sec\nAverage GFLOPS      = "
                << double(nrepeat) * 3 * double(Nn * Nm) /
                       (wall_time_stop - wall_time_start) / 1.0e9
                << "\n";
  }

  {
    if (world.rank() == 0) {
      std::cout << "\nScale add test: 5*(2*a(m,n) + 3*b(m,n))\n";
    }

    TArray a(world, trange);
    TArray b(world, trange);

    a.fill(val_a);
    b.fill(val_b);

    // Start clock
    const double wall_time_start = madness::wall_time();

    // Do
    for (int i = 0; i < nrepeat; ++i) {
      double iter_time_start = madness::wall_time();
      c("m,n") = 5 * (2 * a("m,n") + 3 * b("m,n"));
      double iter_time_stop = madness::wall_time();
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1
                  << " wall time: " << (iter_time_stop - iter_time_start)
                  << "\n";
    }
    // Stop clock
    const double wall_time_stop = madness::wall_time();

    if (world.rank() == 0)
      std::cout << "Average wall time   = "
                << (wall_time_stop - wall_time_start) / double(nrepeat)
                << " sec\nAverage GFLOPS      = "
                << double(nrepeat) * 4 * double(Nn * Nm) /
                       (wall_time_stop - wall_time_start) / 1.0e9
                << "\n";
  }

  {
    if (world.rank() == 0) {
      std::cout << "\nScale add permute test: 5*(2*a(m,n) + 3*b(n,m))\n";
    }

    TArray a(world, trange);
    TArray b(world, trange);

    a.fill(val_a);
    b.fill(val_b);

    // Start clock
    const double wall_time_start = madness::wall_time();

    // Do
    for (int i = 0; i < nrepeat; ++i) {
      double iter_time_start = madness::wall_time();
      c("m,n") = 5 * (2 * a("m,n") + 3 * b("n,m"));
      double iter_time_stop = madness::wall_time();
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1
                  << " wall time: " << (iter_time_stop - iter_time_start)
                  << "\n";
    }
    // Stop clock
    const double wall_time_stop = madness::wall_time();

    if (world.rank() == 0)
      std::cout << "Average wall time   = "
                << (wall_time_stop - wall_time_start) / double(nrepeat)
                << " sec\nAverage GFLOPS      = "
                << double(nrepeat) * 4 * double(Nn * Nm) /
                       (wall_time_stop - wall_time_start) / 1.0e9
                << "\n";
  }
}

template <typename T>
using cudaTile = TiledArray::Tile<TiledArray::btasUMTensorVarray<T>>;

int try_main(int argc, char **argv) {
  // Initialize runtime
  auto &world = TA_SCOPED_INITIALIZE(argc, argv);

  // Get command line arguments
  if (argc < 4) {
    std::cout
        << "vector operations on A(Nm,Nn) and B(Nm,Nn), with dimensions m, n"
           "blocked by Bm, Bn respectively"
        << std::endl
        << "Usage: " << argv[0]
        << " Nm Bm Nn Bn [# of repetitions = 5] [real = double] \n";
    return 0;
  }
  const long Nm = atol(argv[1]);
  const long Bm = atol(argv[2]);
  const long Nn = atol(argv[3]);
  const long Bn = atol(argv[4]);
  if (Nm <= 0 || Nn <= 0) {
    std::cerr << "Error: dimensions must be greater than zero.\n";
    return 1;
  }
  if (Bm <= 0 || Bn <= 0) {
    std::cerr << "Error: block sizes must be greater than zero.\n";
    return 1;
  }
  if ((Nm % Bm) != 0ul || Nn % Bn != 0ul) {
    std::cerr
        << "Error: dimension size must be evenly divisible by block size.\n";
    return 1;
  }
  const long nrepeat = (argc >= 6 ? atol(argv[5]) : 5);
  if (nrepeat <= 0) {
    std::cerr << "Error: number of repetitions must be greater than zero.\n";
    return 1;
  }

  const auto real_type_str =
      (argc >= 7) ? std::string(argv[6]) : std::string("double");

  if (real_type_str != "float" && real_type_str != "double") {
    std::cerr << "Error: invalid real type: " << real_type_str
              << "\n Valid option includes: float or "
                 "double. \n";
  }

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

  if (real_type_str == "double") {
    if (world.rank() == 0) {
      std::cout << "\n GPU vector operations. \n\n";
    }
    do_main_body<cudaTile<double>>(world, Nm, Bm, Nn, Bn, nrepeat);

    if (world.rank() == 0) {
      std::cout << "\n CPU vector operations. \n\n";
    }
    do_main_body<TiledArray::Tensor<double>>(world, Nm, Bm, Nn, Bn, nrepeat);

  } else {
    if (world.rank() == 0) {
      std::cout << "\n GPU vector operations. \n\n";
    }
    do_main_body<cudaTile<float>>(world, Nm, Bm, Nn, Bn, nrepeat);

    if (world.rank() == 0) {
      std::cout << "\n CPU vector operations. \n\n";
    }
    do_main_body<TiledArray::Tensor<float>>(world, Nm, Bm, Nn, Bn, nrepeat);
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
