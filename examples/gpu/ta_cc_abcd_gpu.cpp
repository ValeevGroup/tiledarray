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

#include <TiledArray/gpu/btas_um_tensor.h>
#include <TiledArray/version.h>
#include <tiledarray.h>
#include <iostream>

bool to_bool(const char* str) {
  if (not strcmp(str, "0") || not strcmp(str, "no") || not strcmp(str, "false"))
    return false;
  if (not strcmp(str, "1") || not strcmp(str, "yes") || not strcmp(str, "true"))
    return true;
  throw std::runtime_error("unrecognized string specification of bool");
}

// makes tiles of fluctuating sizes
// if n = average tile size
// this will produce tiles of these sizes: n+1, n-1, n+2, n-2, etc.
// the last tile absorbs the remainder
std::vector<unsigned int> make_tiling(unsigned int range_size,
                                      unsigned int ntiles) {
  const auto average_tile_size = range_size / ntiles;
  TA_ASSERT(average_tile_size > ntiles);
  std::vector<unsigned int> result(ntiles + 1);
  result[0] = 0;
  for (long t = 0; t != ntiles - 1; ++t) {
    result[t + 1] =
        result[t] + average_tile_size + ((t % 2 == 0) ? (t + 1) : (-t));
  }
  result[ntiles] = range_size;
  return result;
}

template <typename Tile, typename Policy>
void rand_fill_array(TA::DistArray<Tile, Policy>& array);

template <typename T>
void cc_abcd(madness::World& world, const TA::TiledRange1& trange_occ,
             const TA::TiledRange1& trange_uocc, long repeat);

int main(int argc, char** argv) {
  int rc = 0;

  try {
    // Initialize runtime
    TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);

    // Get command line arguments
    if (argc < 5) {
      std::cout << "Mocks t2(i,a,j,b) * v(a,b,c,d) term in CC amplitude eqs"
                << std::endl
                << "Usage: " << argv[0]
                << " occ_size occ_nblocks uocc_size "
                   "uocc_nblocks [repetitions] float/double"
                << std::endl;
      return 0;
    }
    const long n_occ = atol(argv[1]);
    const long nblk_occ = atol(argv[2]);
    const long n_uocc = atol(argv[3]);
    const long nblk_uocc = atol(argv[4]);
    if (n_occ <= 0) {
      std::cerr << "Error: occ_size must be greater than zero.\n";
      return 1;
    }
    if (nblk_occ <= 0) {
      std::cerr << "Error: occ_nblocks must be greater than zero.\n";
      return 1;
    }
    if (n_uocc <= 0) {
      std::cerr << "Error: uocc_size must be greater than zero.\n";
      return 1;
    }
    if (nblk_uocc <= 0) {
      std::cerr << "Error: uocc_nblocks must be greater than zero.\n";
      return 1;
    }
    if ((n_occ < nblk_occ) != 0ul) {
      std::cerr << "Error: occ_size must be greater than occ_nblocks.\n";
      return 1;
    }
    if ((n_uocc < nblk_uocc) != 0ul) {
      std::cerr << "Error: uocc_size must be greater than uocc_nblocks.\n";
      return 1;
    }
    const long repeat = (argc >= 6 ? atol(argv[5]) : 5);
    if (repeat <= 0) {
      std::cerr << "Error: number of repetitions must be greater than zero.\n";
      return 1;
    }
    const auto real_type_str =
        (argc >= 7) ? std::string(argv[6]) : std::string("double");

    if (!(real_type_str == "double" || real_type_str == "float")) {
      std::cerr << "Error: unrecognized floating point precision type, it is "
                   "either float or double.\n";
      return 1;
    }

    if (world.rank() == 0) {
      std::cout << "TiledArray: CC T2.V term test..."
                << "\nGit description: " << TiledArray::git_description()
                << "\nNumber of nodes     = " << world.size()
                << "\nocc size            = " << n_occ
                << "\nocc nblocks         = " << nblk_occ
                << "\nuocc size           = " << n_uocc
                << "\nuocc nblocks        = " << nblk_uocc
                << "\nprecision           = " << real_type_str;
    }

    // Construct TiledRange1's
    std::vector<unsigned int> tiling_occ = make_tiling(n_occ, nblk_occ);
    std::vector<unsigned int> tiling_uocc = make_tiling(n_uocc, nblk_uocc);
    auto trange_occ = TA::TiledRange1(tiling_occ.begin(), tiling_occ.end());
    auto trange_uocc = TA::TiledRange1(tiling_uocc.begin(), tiling_uocc.end());

    if (real_type_str == "double") {
      cc_abcd<double>(world, trange_occ, trange_uocc, repeat);
    } else {
      cc_abcd<float>(world, trange_occ, trange_uocc, repeat);
    }
  } catch (TA::Exception& e) {
    std::cerr << "!! TiledArray exception: " << e.what() << "\n";
    rc = 1;
  } catch (madness::MadnessException& e) {
    std::cerr << "!! MADNESS exception: " << e.what() << "\n";
    rc = 1;
  } catch (SafeMPI::Exception& e) {
    std::cerr << "!! SafeMPI exception: " << e.what() << "\n";
    rc = 1;
  } catch (std::exception& e) {
    std::cerr << "!! std exception: " << e.what() << "\n";
    rc = 1;
  } catch (...) {
    std::cerr << "!! exception: unknown exception\n";
    rc = 1;
  }

  return rc;
}

template <typename T>
void cc_abcd(TA::World& world, const TA::TiledRange1& trange_occ,
             const TA::TiledRange1& trange_uocc, long repeat) {
  double to_gb = 1000000000.0;
  auto n_occ = trange_occ.extent();
  auto n_uocc = trange_uocc.extent();
  if (world.rank() == 0) {
    std::cout << "\nOOVV memory         = "
              << n_occ * n_occ * n_uocc * n_uocc * sizeof(T) / to_gb << "GB"
              << "\nVVVV memory         = "
              << n_uocc * n_uocc * n_uocc * n_uocc * sizeof(T) / to_gb << "GB"
              << "\n";
  }

  TA::TiledRange trange_oovv(
      {trange_occ, trange_occ, trange_uocc, trange_uocc});
  TA::TiledRange trange_vvvv(
      {trange_uocc, trange_uocc, trange_uocc, trange_uocc});

  //  const bool do_validate = false;  // set to true if need to validate the
  //  result

  const auto complex_T = TA::detail::is_complex<T>::value;
  const double flops_per_fma =
      (complex_T ? 8 : 2);  // 1 multiply takes 6/1 flops for complex/real
                            // 1 add takes 2/1 flops for complex/real
  const double n_gflop = flops_per_fma * std::pow(n_occ, 2) *
                         std::pow(n_uocc, 4) / std::pow(1024., 3);

  using CUDATile =
      btas::Tensor<T, TA::Range, TiledArray::cuda_um_btas_varray<T>>;
  using CUDAMatrix = TA::DistArray<TA::Tile<CUDATile>>;

  // Construct tensors
  CUDAMatrix t2(world, trange_oovv);
  CUDAMatrix v(world, trange_vvvv);
  CUDAMatrix t2_v;
  // To validate, fill input tensors with random data, otherwise just with 1s
  //  if (do_validate) {
  //    rand_fill_array(t2);
  //    rand_fill_array(v);
  //  } else {
  t2.fill_local(0.2);
  v.fill_local(0.3);
  //  }

  // Start clock
  world.gop.fence();
  if (world.rank() == 0) {
    std::cout << "Starting iterations: "
              << "\n";
  }

  double total_time = 0.0;
  double total_gflop_rate = 0.0;

  // Do matrix multiplication
  for (int i = 0; i < repeat; ++i) {
    const double start = madness::wall_time();

    // this is how the user would express this contraction
    t2_v("i,j,a,b") = t2("i,j,c,d") * v("a,b,c,d");

    const double stop = madness::wall_time();
    const double time = stop - start;
    const double gflop_rate = n_gflop / time;

    // exclude iteration 1
    if (i != 0) {
      total_time += time;
      total_gflop_rate += gflop_rate;
    }
    if (world.rank() == 0)
      std::cout << "Iteration " << i + 1 << "   time=" << time
                << "   GFLOPS=" << gflop_rate << "\n";
  }

  // Print results
  if (world.rank() == 0)
    std::cout << "Average wall time   = "
              << total_time / static_cast<double>(repeat - 1)
              << " sec\nAverage GFLOPS      = "
              << total_gflop_rate / static_cast<double>(repeat - 1) << "\n";

  double threshold = std::numeric_limits<T>::epsilon();
  auto dot_length = n_uocc * n_uocc;
  auto result = dot_length * 0.2 * 0.3;

  auto verify = [&world, &threshold, &result,
                 &dot_length](const TA::Tile<CUDATile>& tile) {
    auto n_elements = tile.size();
    for (std::size_t i = 0; i < n_elements; i++) {
      double abs_err = fabs(tile[i] - result);
      //      double abs_val = fabs(tile[i]);
      double rel_err = abs_err / result / dot_length;
      if (rel_err > threshold) {
        std::cout << "Node: " << world.rank() << " Tile: " << tile.range()
                  << " id: " << i
                  << std::string(" gpu: " + std::to_string(tile[i]) +
                                 " cpu: " + std::to_string(result) + "\n");
        break;
      }
    }
  };

  for (auto iter = t2_v.begin(); iter != t2_v.end(); iter++) {
    world.taskq.add(verify, t2_v.find(iter.index()));
  }

  world.gop.fence();

  if (world.rank() == 0) {
    std::cout << "Verification Passed" << std::endl;
  }
}

template <typename Tile, typename Policy>
void rand_fill_array(TA::DistArray<Tile, Policy>& array) {
  auto& world = array.world();
  // Iterate over local, non-zero tiles
  for (auto it : array) {
    // Construct a new tile with random data
    typename TA::DistArray<Tile, Policy>::value_type tile(
        array.trange().make_tile_range(it.index()));
    for (auto& tile_it : tile) tile_it = world.drand();

    // Set array tile
    it = tile;
  }
}
