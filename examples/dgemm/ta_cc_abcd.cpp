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

// low-level evaluation of tv(i1,i2,v3,v4) = t(o1,o2,v1,v2) * v(v1,v2,v3,v4)
// as an explicit SUMMA
// can be extended to handle permutations, etc.
template <typename Tile, typename Policy>
void tensor_contract_444(TA::DistArray<Tile, Policy>& tv,
                         const TA::DistArray<Tile, Policy>& t,
                         const TA::DistArray<Tile, Policy>& v);

template <typename Tile, typename Policy>
void rand_fill_array(TA::DistArray<Tile, Policy>& array);

template <typename T>
void cc_abcd(madness::World& world, const TA::TiledRange1& trange_occ,
             const TA::TiledRange1& trange_uocc, long repeat);

int main(int argc, char** argv) {
  int rc = 0;

  try {
    // Initialize runtime
    TA::World& world = TA::initialize(argc, argv);

    // Get command line arguments
    if (argc < 5) {
      std::cout << "Mocks t2(i,a,j,b) * v(a,b,c,d) term in CC amplitude eqs"
                << std::endl
                << "Usage: " << argv[0]
                << " occ_size occ_nblocks uocc_size "
                   "uocc_nblocks [repetitions] [use_complex]"
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
    const bool use_complex = (argc >= 7 ? to_bool(argv[6]) : false);

    if (world.rank() == 0)
      std::cout << "TiledArray: CC T2.V term test..."
                << "\nGit HASH: " << TILEDARRAY_REVISION
                << "\nNumber of nodes     = " << world.size()
                << "\nocc size            = " << n_occ
                << "\nocc nblocks         = " << nblk_occ
                << "\nuocc size           = " << n_uocc
                << "\nuocc nblocks        = " << nblk_uocc
                << "\nComplex             = "
                << (use_complex ? "true" : "false") << "\n";

    // Construct TiledRange1's
    std::vector<unsigned int> tiling_occ = make_tiling(n_occ, nblk_occ);
    std::vector<unsigned int> tiling_uocc = make_tiling(n_uocc, nblk_uocc);
    auto trange_occ = TA::TiledRange1(tiling_occ.begin(), tiling_occ.end());
    auto trange_uocc = TA::TiledRange1(tiling_uocc.begin(), tiling_uocc.end());

    if (use_complex)
      cc_abcd<std::complex<double>>(world, trange_occ, trange_uocc, repeat);
    else
      cc_abcd<double>(world, trange_occ, trange_uocc, repeat);

    TA::finalize();

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
  TA::TiledRange trange_oovv(
      {trange_occ, trange_occ, trange_uocc, trange_uocc});
  TA::TiledRange trange_vvvv(
      {trange_uocc, trange_uocc, trange_uocc, trange_uocc});

  const bool do_validate = false;  // set to true if need to validate the result
  auto n_occ = trange_occ.extent();
  auto n_uocc = trange_uocc.extent();

  const auto complex_T = TA::detail::is_complex<T>::value;
  const double flops_per_fma =
      (complex_T ? 8 : 2);  // 1 multiply takes 6/1 flops for complex/real
                            // 1 add takes 2/1 flops for complex/real
  const double n_gflop = flops_per_fma * std::pow(n_occ, 2) *
                         std::pow(n_uocc, 4) / std::pow(1024., 3);

  // Construct tensors
  TA::TArrayD t2(world, trange_oovv);
  TA::TArrayD v(world, trange_vvvv);
  TA::TArrayD t2_v;
  // To validate, fill input tensors with random data, otherwise just with 1s
  if (do_validate) {
    rand_fill_array(t2);
    rand_fill_array(v);
  } else {
    t2.fill_local(1.0);
    v.fill_local(1.0);
  }

  // Start clock
  world.gop.fence();
  if (world.rank() == 0)
    std::cout << "Starting iterations: "
              << "\n";

  double total_time = 0.0;
  double total_gflop_rate = 0.0;

  // Do matrix multiplication
  for (int i = 0; i < repeat; ++i) {
    const double start = madness::wall_time();

    // this is how the user would express this contraction
    if (false) t2_v("i,j,a,b") = t2("i,j,c,d") * v("a,b,c,d");

    // this demonstrates to the PaRSEC team what happens under the hood of the
    // expression above
    if (true) {
      tensor_contract_444(t2_v, t2, v);

      // to validate replace: false -> true
      if (do_validate) {
        // obtain reference result using the high-level DSL
        TA::TArrayD t2_v_ref;
        t2_v_ref("i,j,a,b") = t2("i,j,c,d") * v("c,d,a,b");
        TA::TArrayD error;
        error("i,j,a,b") = t2_v_ref("i,j,a,b") - t2_v("i,j,a,b");
        std::cout << "Validating the result (ignore the timings/performance!): "
                     "||ref_result - result||_2^2 = "
                  << error("i,j,a,b").squared_norm().get() << std::endl;
      }
    }

    const double stop = madness::wall_time();
    const double time = stop - start;
    total_time += time;
    const double gflop_rate = n_gflop / time;
    total_gflop_rate += gflop_rate;
    if (world.rank() == 0)
      std::cout << "Iteration " << i + 1 << "   time=" << time
                << "   GFLOPS=" << gflop_rate << "\n";
  }

  // Print results
  if (world.rank() == 0)
    std::cout << "Average wall time   = "
              << total_time / static_cast<double>(repeat)
              << " sec\nAverage GFLOPS      = "
              << total_gflop_rate / static_cast<double>(repeat) << "\n";
}

template <typename LeftTile, typename RightTile, typename Policy, typename Op>
TA::detail::DistEval<typename Op::result_type, Policy> make_contract_eval(
    const TA::detail::DistEval<LeftTile, Policy>& left,
    const TA::detail::DistEval<RightTile, Policy>& right, madness::World& world,
    const typename TA::detail::DistEval<typename Op::result_type,
                                        Policy>::shape_type& shape,
    const std::shared_ptr<typename TA::detail::DistEval<
        typename Op::result_type, Policy>::pmap_interface>& pmap,
    const TA::Permutation& perm, const Op& op) {
  TA_ASSERT(left.range().rank() == op.left_rank());
  TA_ASSERT(right.range().rank() == op.right_rank());
  TA_ASSERT((perm.size() == op.result_rank()) || !perm);

  // Define the impl type
  typedef TA::detail::Summa<TA::detail::DistEval<LeftTile, Policy>,
                            TA::detail::DistEval<RightTile, Policy>, Op, Policy>
      impl_type;

  // Precompute iteration range data
  const unsigned int num_contract_ranks = op.num_contract_ranks();
  const unsigned int left_end = op.left_rank();
  const unsigned int left_middle = left_end - num_contract_ranks;
  const unsigned int right_end = op.right_rank();

  // Construct a vector TiledRange1 objects from the left- and right-hand
  // arguments that will be used to construct the result TiledRange. Also,
  // compute the fused outer dimension sizes, number of tiles and elements,
  // for the contraction.
  typename impl_type::trange_type::Ranges ranges(op.result_rank());
  std::size_t M = 1ul, m = 1ul, N = 1ul, n = 1ul;
  std::size_t pi = 0ul;
  for (unsigned int i = 0ul; i < left_middle; ++i) {
    ranges[(perm ? perm[pi++] : pi++)] = left.trange().data()[i];
    M *= left.range().extent(i);
    m *= left.trange().elements_range().extent(i);
  }
  for (std::size_t i = num_contract_ranks; i < right_end; ++i) {
    ranges[(perm ? perm[pi++] : pi++)] = right.trange().data()[i];
    N *= right.range().extent(i);
    n *= right.trange().elements_range().extent(i);
  }

  // Compute the number of tiles in the inner dimension.
  std::size_t K = 1ul;
  for (std::size_t i = left_middle; i < left_end; ++i)
    K *= left.range().extent(i);

  // Construct the result range
  typename impl_type::trange_type trange(ranges.begin(), ranges.end());

  // Construct the process grid
  TA::detail::ProcGrid proc_grid(world, M, N, m, n);

  return TA::detail::DistEval<typename Op::result_type, Policy>(
      std::shared_ptr<impl_type>(new impl_type(
          left, right, world, trange, shape, pmap, perm, op, K, proc_grid)));
}

template <typename Tile, typename Policy, typename Op>
static TA::detail::DistEval<
    TA::detail::LazyArrayTile<typename TA::DistArray<Tile, Policy>::value_type,
                              Op>,
    Policy>
make_array_eval(
    const TA::DistArray<Tile, Policy>& array, madness::World& world,
    const typename TA::detail::DistEval<Tile, Policy>::shape_type& shape,
    const std::shared_ptr<
        typename TA::detail::DistEval<Tile, Policy>::pmap_interface>& pmap,
    const TA::Permutation& perm, const Op& op) {
  typedef TA::detail::ArrayEvalImpl<TA::DistArray<Tile, Policy>, Op, Policy>
      impl_type;
  return TA::detail::DistEval<
      TA::detail::LazyArrayTile<
          typename TA::DistArray<Tile, Policy>::value_type, Op>,
      Policy>(std::shared_ptr<impl_type>(new impl_type(
      array, world, (perm ? perm * array.trange() : array.trange()), shape,
      pmap, perm, op)));
}

template <typename Tile>
TA::detail::ContractReduce<Tile, Tile, Tile, typename Tile::value_type>
make_contract(const unsigned int result_rank, const unsigned int left_rank,
              const unsigned int right_rank,
              const TA::Permutation& perm = TA::Permutation()) {
  return TA::detail::ContractReduce<Tile, Tile, Tile,
                                    typename Tile::value_type>(
      TiledArray::math::blas::Op::NoTrans, TiledArray::math::blas::Op::NoTrans,
      1, result_rank, left_rank, right_rank, perm);
}

template <typename Tile>
static TA::detail::UnaryWrapper<TA::detail::Noop<Tile, Tile, true>>
make_array_noop(const TA::Permutation& perm = TA::Permutation()) {
  return TA::detail::UnaryWrapper<TA::detail::Noop<Tile, Tile, true>>(
      TA::detail::Noop<Tile, Tile, true>(), perm);
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

template <typename Tile, typename Policy>
void tensor_contract_444(TA::DistArray<Tile, Policy>& tv,
                         const TA::DistArray<Tile, Policy>& t,
                         const TA::DistArray<Tile, Policy>& v) {
  // for convenience, obtain the tiled ranges for the two kinds of dimensions
  // used to define t, v, and tv
  auto trange_occ = t.trange().dim(0);   // the first dimension of t is occ
  auto trange_uocc = v.trange().dim(0);  // every dimension of v is uocc
  auto ntiles_occ = trange_occ.tile_extent();
  auto ntiles_uocc = trange_uocc.tile_extent();
  auto n_occ = trange_occ.extent();
  auto n_uocc = trange_occ.extent();

  // compute the 2-d grid of processors for the SUMMA
  // note that the result is (occ occ|uocc uocc), hence the row dimension is occ
  // x occ, etc.
  auto& world = t.world();
  auto nrowtiles = ntiles_occ * ntiles_occ;
  auto ncoltiles = ntiles_uocc * ntiles_uocc;
  auto ninttiles =
      ntiles_uocc * ntiles_uocc;  // contraction is over uocc x uocc
  auto nrows = n_occ * n_occ;
  auto ncols = n_uocc * n_uocc;
  TA::detail::ProcGrid proc_grid(world, nrowtiles, ncoltiles, nrows, ncols);
  std::shared_ptr<TA::Pmap> pmap;
  auto t_eval = make_array_eval(t, t.world(), TA::DenseShape(),
                                proc_grid.make_row_phase_pmap(ninttiles),
                                TA::Permutation(), make_array_noop<Tile>());
  auto v_eval = make_array_eval(v, v.world(), TA::DenseShape(),
                                proc_grid.make_col_phase_pmap(ninttiles),
                                TA::Permutation(), make_array_noop<Tile>());

  //
  // make the result metadata
  //

  // result shape
  TA::TiledRange trange_tv({trange_occ, trange_occ, trange_uocc, trange_uocc});
  //
  pmap.reset(
      new TA::detail::BlockedPmap(world, trange_tv.tiles_range().volume()));
  // 'contract' object is of type
  // PaRSEC's PTG object will do the job here:
  // 1. it will use t_eval and v_eval's Futures as input
  // 2. there will be a dummy output ArrayEval, its Futures will be set by the
  // PTG
  auto contract =
      make_contract_eval(t_eval, v_eval, world, TA::DenseShape(), pmap,
                         TA::Permutation(), make_contract<Tile>(4u, 4u, 4u));

  // eval() just schedules the Summa task and proceeds
  // in expressions evaluation is lazy ... you could just use contract tiles
  // immediately to compose further (in principle even before eval()!)
  contract.eval();

  // since the intent of this function is to return result as a named DistArray
  // migrate contract's futures to tv here

  // Create a temporary result array
  TA::DistArray<Tile, Policy> result(contract.world(), contract.trange(),
                                     contract.shape(), contract.pmap());

  // Move the data from dist_eval into the result array. There is no
  // communication in this step.
  for (const auto index : *contract.pmap()) {
    if (!contract.is_zero(index)) result.set(index, contract.get(index));
  }

  // uncomment this to block until Summa is complete .. but no need to wait
  // contract.wait();

  // Swap the new array with the result array object.
  result.swap(tv);
}
