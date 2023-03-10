/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Karl Pierce
 *  Department of Chemistry, Virginia Tech
 *
 *  cp.cpp
 *  June 7, 2022
 *
 */

#include "compute_trange1.h"
#include "range_fixture.h"

#include "tiledarray.h"
#include "unit_test_config.h"

#include <libgen.h>
#include <iomanip>
#include "TiledArray/math/solvers/cp.h"
#include "TiledArray/math/solvers/cp/btas_cp.h"

constexpr std::int64_t rank_tile_size = 10;
constexpr bool verbose = false;

using namespace TiledArray;

struct CPFixture : public TiledRangeFixture {
  CPFixture()
      : shape_tr(make_random_sparseshape(tr)),
        a_sparse(*GlobalFixture::world, tr, shape_tr) {
    a_sparse.fill_random<HostExecutor::Thread>();
    a_sparse.truncate();
  }

  /// make a shape with approximate half dense and half sparse
  static SparseShape<float> make_random_sparseshape(const TiledRange& tr) {
    std::size_t n = tr.tiles_range().volume();
    Tensor<float> norms(tr.tiles_range(), 0.0);

    // make sure all mpi gets the same shape
    if (GlobalFixture::world->rank() == 0) {
      for (std::size_t i = 0; i < n; i++) {
        norms[i] = GlobalFixture::world->drand() > 0.5 ? 0.0 : 1.0;
      }
    }

    GlobalFixture::world->gop.broadcast_serializable(norms, 0);

    return SparseShape<float>(norms, tr);
  }

  static TensorF tensori_to_tensorf(const TensorI& tensori) {
    std::size_t n = tensori.size();
    TensorF tensorf(tensori.range());

    for (std::size_t i = 0ul; i < n; i++) {
      tensorf[i] = float(tensori[i]);
    }
    return tensorf;
  }

  static TensorI tensorf_to_tensori(const TensorF& tensorf) {
    std::size_t n = tensorf.size();
    TensorI tensori(tensorf.range());
    for (std::size_t i = 0ul; i < n; i++) {
      tensori[i] = int(tensorf[i]);
    }
    return tensori;
  }

  ~CPFixture() { GlobalFixture::world->gop.fence(); }

  SparseShape<float> shape_tr;
  TArrayI a_dense;
  TSpArrayI a_sparse;
};

template <typename TArrayT>
TArrayT compute_cp(const TArrayT& T, size_t cp_rank, bool verbose = false) {
  CP_ALS<typename TArrayT::value_type, typename TArrayT::policy_type> CPD(T);
  CPD.compute_rank(cp_rank, rank_tile_size, false, 1e-3, verbose);
  return CPD.reconstruct();
}

enum class Fill { Random, Constant };

template <typename TArrayT, Fill fill = Fill::Constant,
          typename... OptionalArgs>
TArrayT make(const TiledRange& tr, OptionalArgs&&... opt_args) {
  TArrayT result(*GlobalFixture::world, tr, opt_args...);
  switch (fill) {
    case Fill::Constant:
      result.fill(1);
      break;
    case Fill::Random:
      // make sure randomness is deterministic
      result.template fill_random<HostExecutor::Thread>();
      break;
  }
  return result;
}

BOOST_FIXTURE_TEST_SUITE(cp_suite, CPFixture)

const auto target_rel_error = std::sqrt(std::numeric_limits<double>::epsilon());

BOOST_AUTO_TEST_CASE(btas_cp_als) {
  TiledArray::srand(1);

  // Make a tiled range with block size of 1
  TiledArray::TiledRange tr3, tr4, tr5;
  {
    TiledRange1 tr1_mode0 = compute_trange1(11, 1),
                tr1_mode1 = compute_trange1(7, 2),
                tr1_mode2 = compute_trange1(15, 4),
                tr1_mode3 = compute_trange1(8, 7);
    tr3 = TiledRange({tr1_mode0, tr1_mode1, tr1_mode2});
    tr4 = TiledRange({tr1_mode0, tr1_mode1, tr1_mode2, tr1_mode3});
    tr5 = TiledRange({tr1_mode0, tr1_mode0, tr1_mode1, tr1_mode2, tr1_mode3});
  }

  // Dense

  // order-3 rank-1 test
  {
    std::vector<TArrayD> factors;
    auto b_dense = make<TArrayD>(tr3);
    size_t cp_rank = 1;
    factors = math::cp::btas::cp_als(*GlobalFixture::world, b_dense, cp_rank,
                                     compute_trange1(cp_rank, rank_tile_size),
                                     0, 1e-3, verbose);
    auto b_cp = cp_reconstruct(factors);
    TArrayD diff;
    diff("a,b,c") = b_dense("a,b,c") - b_cp("a,b,c");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-4 rank-1 test
  {
    std::vector<TArrayD> factors;
    auto b_dense = make<TArrayD>(tr4);
    size_t cp_rank = 1;
    factors = math::cp::btas::cp_als(*GlobalFixture::world, b_dense, cp_rank,
                                     compute_trange1(cp_rank, rank_tile_size),
                                     0, 1e-3, verbose);

    auto b_cp = cp_reconstruct(factors);
    TArrayD diff;
    diff("a,b,c,d") = b_dense("a,b,c,d") - b_cp("a,b,c,d");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-5 rank-1 test
  {
    std::vector<TArrayD> factors;
    auto b_dense = make<TArrayD>(tr5);
    size_t cp_rank = 1;
    factors = math::cp::btas::cp_als(*GlobalFixture::world, b_dense, cp_rank,
                                     compute_trange1(cp_rank, rank_tile_size),
                                     0, 1e-3, verbose);

    auto b_cp = cp_reconstruct(factors);
    TArrayD diff;
    diff("a,b,c,d,e") = b_dense("a,b,c,d,e") - b_cp("a,b,c,d,e");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < target_rel_error);
    BOOST_CHECK(accurate);
  }

  // sparse

  // order-3 rank-N test
  {
    std::vector<TSpArrayD> factors;
    auto b_sparse = make<TSpArrayD, Fill::Random>(tr3);
    size_t cp_rank = 77;
    factors = math::cp::btas::cp_als(*GlobalFixture::world, b_sparse, cp_rank,
                                     compute_trange1(cp_rank, rank_tile_size),
                                     0, 1e-3);

    auto b_cp = cp_reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c") = b_sparse("a,b,c") - b_cp("a,b,c");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-4 rank-1 test
  {
    std::vector<TSpArrayD> factors;
    auto b_sparse = make<TSpArrayD>(tr4);
    size_t cp_rank = 1;
    factors = math::cp::btas::cp_als(*GlobalFixture::world, b_sparse, cp_rank,
                                     compute_trange1(cp_rank, rank_tile_size),
                                     0, 1e-3, verbose);

    auto b_cp = cp_reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c,d") = b_sparse("a,b,c,d") - b_cp("a,b,c,d");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-5 test
  {
    std::vector<TSpArrayD> factors;
    auto b_sparse = make<TSpArrayD>(tr5);
    size_t cp_rank = 1;
    factors = math::cp::btas::cp_als(*GlobalFixture::world, b_sparse, cp_rank,
                                     compute_trange1(cp_rank, rank_tile_size),
                                     0, 1e-3, verbose);

    auto b_cp = cp_reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c,d,e") = b_sparse("a,b,c,d,e") - b_cp("a,b,c,d,e");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < 1e-9);
    BOOST_CHECK(accurate);
  }
}

BOOST_AUTO_TEST_CASE(btas_cp_rals) {
  TiledArray::srand(1);

  // Make a tiled range with block size of 1
  TiledArray::TiledRange tr3, tr4, tr5;
  {
    TiledRange1 tr1_mode0 = compute_trange1(11, 1),
                tr1_mode1 = compute_trange1(7, 2),
                tr1_mode2 = compute_trange1(15, 4),
                tr1_mode3 = compute_trange1(8, 7);
    tr3 = TiledRange({tr1_mode0, tr1_mode1, tr1_mode2});
    tr4 = TiledRange({tr1_mode0, tr1_mode1, tr1_mode2, tr1_mode3});
    tr5 = TiledRange({tr1_mode0, tr1_mode0, tr1_mode1, tr1_mode2, tr1_mode3});
  }

  // Dense test
  // order-3 test
  {
    std::vector<TArrayD> factors;
    auto b_dense = make<TArrayD>(tr3);
    size_t cp_rank = 1;
    factors = math::cp::btas::cp_rals(*GlobalFixture::world, b_dense, cp_rank,
                                      compute_trange1(cp_rank, rank_tile_size),
                                      0, 1e-3, verbose);

    auto b_cp = cp_reconstruct(factors);
    TArrayD diff;
    diff("a,b,c") = b_dense("a,b,c") - b_cp("a,b,c");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-4 test
  {
    std::vector<TArrayD> factors;
    auto b_dense = make<TArrayD>(tr4);
    size_t cp_rank = 1;
    factors = math::cp::btas::cp_rals(*GlobalFixture::world, b_dense, cp_rank,
                                      compute_trange1(cp_rank, rank_tile_size),
                                      0, 1e-3, verbose);

    auto b_cp = cp_reconstruct(factors);
    TArrayD diff;
    diff("a,b,c,d") = b_dense("a,b,c,d") - b_cp("a,b,c,d");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-5 test
  {
    std::vector<TArrayD> factors;
    auto b_dense = make<TArrayD>(tr5);
    size_t cp_rank = 1;
    factors = math::cp::btas::cp_rals(*GlobalFixture::world, b_dense, cp_rank,
                                      compute_trange1(cp_rank, rank_tile_size),
                                      0, 1e-3, verbose);

    auto b_cp = cp_reconstruct(factors);
    TArrayD diff;
    diff("a,b,c,d,e") = b_dense("a,b,c,d,e") - b_cp("a,b,c,d,e");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < target_rel_error);
    BOOST_CHECK(accurate);
  }

  // sparse test
  {
    std::vector<TSpArrayD> factors;
    auto b_sparse = make<TSpArrayD, Fill::Random>(tr3);
    size_t cp_rank = 77;
    factors = math::cp::btas::cp_rals(*GlobalFixture::world, b_sparse, cp_rank,
                                      compute_trange1(cp_rank, rank_tile_size),
                                      0, 1e-3, verbose);

    auto b_cp = cp_reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c") = b_sparse("a,b,c") - b_cp("a,b,c");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-4 test
  {
    std::vector<TSpArrayD> factors;
    auto b_sparse = make<TSpArrayD>(tr4);
    size_t cp_rank = 1;
    factors = math::cp::btas::cp_rals(*GlobalFixture::world, b_sparse, cp_rank,
                                      compute_trange1(cp_rank, rank_tile_size),
                                      0, 1e-3, verbose);

    auto b_cp = cp_reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c,d") = b_sparse("a,b,c,d") - b_cp("a,b,c,d");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-5 test
  {
    std::vector<TSpArrayD> factors;
    auto b_sparse = make<TSpArrayD>(tr5);
    size_t cp_rank = 1;
    factors = math::cp::btas::cp_als(*GlobalFixture::world, b_sparse, cp_rank,
                                     compute_trange1(cp_rank, rank_tile_size),
                                     0, 1e-3, verbose);

    auto b_cp = cp_reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c,d,e") = b_sparse("a,b,c,d,e") - b_cp("a,b,c,d,e");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < 1e-9);
    BOOST_CHECK(accurate);
  }
}

BOOST_AUTO_TEST_CASE(ta_cp_als) {
  TiledArray::srand(1);

  // Make a tiled range with block size of 1
  TiledArray::TiledRange tr3, tr4, tr5;
  {
    TiledRange1 tr1_mode0 = compute_trange1(11, 1),
                tr1_mode1 = compute_trange1(7, 2),
                tr1_mode2 = compute_trange1(15, 4),
                tr1_mode3 = compute_trange1(8, 7);
    tr3 = TiledRange({tr1_mode0, tr1_mode1, tr1_mode2});
    tr4 = TiledRange({tr1_mode0, tr1_mode1, tr1_mode2, tr1_mode3});
    tr5 = TiledRange({tr1_mode0, tr1_mode0, tr1_mode1, tr1_mode2, tr1_mode3});
  }

  // Dense test
  // order-3 test
  {
    auto b_dense = make<TArrayD>(tr3);
    size_t cp_rank = 1;
    auto b_cp = compute_cp(b_dense, cp_rank, verbose);
    TArrayD diff;
    diff("a,b,c") = b_dense("a,b,c") - b_cp("a,b,c");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-4 test
  {
    auto b_dense = make<TArrayD>(tr4);
    size_t cp_rank = 1;
    auto b_cp = compute_cp(b_dense, cp_rank, verbose);

    TArrayD diff;
    diff("a,b,c,d") = b_dense("a,b,c,d") - b_cp("a,b,c,d");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-5 test
  {
    auto b_dense = make<TArrayD>(tr5);
    size_t cp_rank = 1;
    auto b_cp = compute_cp(b_dense, cp_rank, verbose);

    TArrayD diff;
    diff("a,b,c,d,e") = b_dense("a,b,c,d,e") - b_cp("a,b,c,d,e");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < target_rel_error);
    BOOST_CHECK(accurate);
  }

  // sparse test
  // order-3 test
  {
    auto b_sparse = make<TSpArrayD, Fill::Random>(tr3);
    // std::cout << "b_sparse = " <<
    // array_to_eigen_tensor<Eigen::Tensor<double,3>>(b_sparse) << std::endl;
    size_t cp_rank = 77;
    auto b_cp = compute_cp(b_sparse, cp_rank, verbose);
    TSpArrayD diff;
    diff("a,b,c") = b_sparse("a,b,c") - b_cp("a,b,c");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-4 test
  {
    auto b_sparse = make<TSpArrayD>(tr4);
    size_t cp_rank = 1;
    auto b_cp = compute_cp(b_sparse, cp_rank, verbose);
    TSpArrayD diff;
    diff("a,b,c,d") = b_sparse("a,b,c,d") - b_cp("a,b,c,d");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < target_rel_error);
    BOOST_CHECK(accurate);
  }
  // order-5 test
  {
    auto b_sparse = make<TSpArrayD>(tr5);
    double cp_rank = 1;
    auto b_cp = compute_cp(b_sparse, cp_rank, verbose);
    TSpArrayD diff;
    diff("a,b,c,d,e") = b_sparse("a,b,c,d,e") - b_cp("a,b,c,d,e");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < target_rel_error);
    BOOST_CHECK(accurate);
  }
}

BOOST_AUTO_TEST_SUITE_END()
