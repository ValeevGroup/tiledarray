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

#include "range_fixture.h"
#include "compute_trange1.h"

#include "tiledarray.h"
#include "unit_test_config.h"

#include "TiledArray/cp/btas_cp.h"
#include "TiledArray/cp/cp_reconstruct.h"
#include <libgen.h>
#include <iomanip>

const std::string __dirname = dirname(strdup(__FILE__));

using namespace TiledArray;

struct CPFixture : public TiledRangeFixture {
  CPFixture()
      : shape_tr(make_random_sparseshape(tr)),
        a_sparse(*GlobalFixture::world, tr, shape_tr) {
    random_fill(a_sparse);
    a_sparse.truncate();
  }

  template <typename Tile, typename Policy>
  static void random_fill(DistArray<Tile, Policy>& array) {
    typename DistArray<Tile, Policy>::pmap_interface::const_iterator it =
        array.pmap()->begin();
    typename DistArray<Tile, Policy>::pmap_interface::const_iterator end =
        array.pmap()->end();
    for (; it != end; ++it) {
      if (!array.is_zero(*it))
        array.set(*it, array.world().taskq.add(
                           &CPFixture::template make_rand_tile<Tile>,
                           array.trange().make_tile_range(*it)));
    }
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

  // Fill a tile with random data
  template <typename Tile>
  static Tile make_rand_tile(const typename Tile::range_type& r) {
    Tile tile(r);
    for (std::size_t i = 0ul; i < tile.size(); ++i) set_random(tile[i]);
    return tile;
  }

  template <typename Tile>
  static double init_rand_tile(Tile& tile, const typename Tile::range_type& r) {
    tile = Tile(r);
    for (std::size_t i = 0ul; i < tile.size(); ++i) set_random(tile[i]);
    double result;
    norm(tile, result);
    return result;
  }

  template <typename Tile>
  static double init_unit_tile(Tile& tile, const typename Tile::range_type& r) {
    tile = Tile(r);
    std::mt19937 generator(3);
    std::uniform_real_distribution distribution(-1.0, 1.0);
    for (std::size_t i = 0ul; i < tile.size(); ++i) tile[i] = 1;
    double result;
    norm(tile, result);
    return result;
  }

  template <typename T>
  static void set_random(T& t) {
    t = GlobalFixture::world->rand() % 101;
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

//TiledArray::TiledRange1 compute_trange1(std::size_t range_size,
//                                        std::size_t target_block_size) {
//  if (range_size > 0) {
//    std::size_t nblocks =
//        (range_size + target_block_size - 1) / target_block_size;
//    auto dv = std::div((int) (range_size + nblocks - 1), (int) nblocks);
//    auto avg_block_size = dv.quot - 1, num_avg_plus_one = dv.rem + 1;
//    std::vector<std::size_t> hashmarks;
//    hashmarks.reserve(nblocks + 1);
//    auto block_counter = 0;
//    for(auto i = 0; i < num_avg_plus_one; ++i, block_counter += avg_block_size + 1){
//      hashmarks.push_back(block_counter);
//    }
//    for (auto i = num_avg_plus_one; i < nblocks; ++i, block_counter+= avg_block_size) {
//      hashmarks.push_back(block_counter);
//    }
//    hashmarks.push_back(range_size);
//    return TA::TiledRange1(hashmarks.begin(), hashmarks.end());
//  } else
//    return TA::TiledRange1{};
//}

BOOST_FIXTURE_TEST_SUITE(cp_suite, CPFixture)

BOOST_AUTO_TEST_CASE(btas_cp_als){
  // get local world
  const auto rank = (*GlobalFixture::world).rank();
  const auto size = (*GlobalFixture::world).size();

  madness::World* tmp_ptr;
  std::shared_ptr<madness::World> world_ptr;

  if (size > 1) {
    SafeMPI::Group group =
        (*GlobalFixture::world).mpi.comm().Get_group().Incl(1, &rank);
    SafeMPI::Intracomm comm = (*GlobalFixture::world).mpi.comm().Create(group);
    world_ptr = std::make_shared<madness::World>(comm);
    tmp_ptr = world_ptr.get();
  } else {
    tmp_ptr = &(*GlobalFixture::world);
  }
  auto& this_world = *tmp_ptr;
  (*GlobalFixture::world).gop.fence();

  // Make a tiled range with block size of 1
  TiledArray::TiledRange tr3, tr4, tr5;
  {
    TiledRange1 tr1_mode0 = compute_trange1(11, 1),
                    tr1_mode1 = compute_trange1(7, 2),
                    tr1_mode2 = compute_trange1(15, 4),
                    tr1_mode3 = compute_trange1(8,7);
    tr3 = TiledRange({tr1_mode0, tr1_mode1, tr1_mode2});
    tr4 = TiledRange({tr1_mode0, tr1_mode1, tr1_mode2, tr1_mode3});
    tr5 = TiledRange({tr1_mode0, tr1_mode0, tr1_mode1, tr1_mode2, tr1_mode3});
  }

  // Dense test
  // order-3 test
  {
    std::vector<TArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_dense = make_array<TArrayD>(*GlobalFixture::world, tr3,
                                       &this->init_unit_tile<TensorD>);
    double cp_rank = 1;
    factors = cp::btas_cp_als(*GlobalFixture::world, b_dense, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);
    auto b_cp = cp::reconstruct(factors);
    TArrayD diff;
    diff("a,b,c") = b_dense("a,b,c") - b_cp("a,b,c");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < 1e-10);
    if(!accurate) std::cout << "The error in order-3 dense is : " << TA::norm2(diff) / TA::norm2(b_dense) << std::endl;
    BOOST_CHECK(accurate);
  }
  // order-4 test
  {
    std::vector<TArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_dense = make_array<TArrayD>(*GlobalFixture::world, tr4,
                                       &this->init_unit_tile<TensorD>);
    double cp_rank = 1;
    factors = cp::btas_cp_als(*GlobalFixture::world, b_dense, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);

    auto b_cp = cp::reconstruct(factors);
    TArrayD diff;
    diff("a,b,c,d") = b_dense("a,b,c,d") - b_cp("a,b,c,d");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < 1e-10);
    if(!accurate) std::cout << "The error in order-4 dense is : " << TA::norm2(diff) / TA::norm2(b_dense) << std::endl;
    BOOST_CHECK(accurate);
  }
  // order-5 test
  {
    std::vector<TArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_dense = make_array<TArrayD>(*GlobalFixture::world, tr5,
                                       &this->init_unit_tile<TensorD>);
    double cp_rank = 1;
    factors = cp::btas_cp_als(*GlobalFixture::world, b_dense, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);

    auto b_cp = cp::reconstruct(factors);
    TArrayD diff;
    diff("a,b,c,d,e") = b_dense("a,b,c,d,e") - b_cp("a,b,c,d,e");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < 1e-10);
    if(!accurate) std::cout << "The error in order-5 dense is : " << TA::norm2(diff) / TA::norm2(b_dense) << std::endl;
    BOOST_CHECK(accurate);
  }

  // sparse test
  {
    std::vector<TSpArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_sparse = make_array<TSpArrayD>(*GlobalFixture::world, tr3,
                                              &this->init_rand_tile<TensorD>);
    double cp_rank = 100;
    factors = cp::btas_cp_als(*GlobalFixture::world, b_sparse, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);

    auto b_cp = cp::reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c") = b_sparse("a,b,c") - b_cp("a,b,c");
    bool accurate = ( TA::norm2(diff) / TA::norm2(b_sparse) < 1e-10);
    if(!accurate) std::cout << "The error in order-3 sparse is : " << TA::norm2(diff) / TA::norm2(b_sparse) << std::endl;
    BOOST_CHECK(accurate);
  }
  // order-4 test
  {
    std::vector<TSpArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_sparse = make_array<TSpArrayD>(*GlobalFixture::world, tr4,
                                       &this->init_unit_tile<TensorD>);
    double cp_rank = 100;
    factors = cp::btas_cp_als(*GlobalFixture::world, b_sparse, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);

    auto b_cp = cp::reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c,d") = b_sparse("a,b,c,d") - b_cp("a,b,c,d");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < 1e-10);
    if(!accurate) std::cout << "The error in order-4 sparse is : " << TA::norm2(diff) / TA::norm2(b_sparse) << std::endl;
    BOOST_CHECK(accurate);
  }
  // order-5 test
  {
    std::vector<TSpArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_sparse = make_array<TSpArrayD>(*GlobalFixture::world, tr5,
                                       &this->init_unit_tile<TensorD>);
    double cp_rank = 105;
    factors = cp::btas_cp_als(*GlobalFixture::world, b_sparse, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);

    auto b_cp = cp::reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c,d,e") = b_sparse("a,b,c,d,e") - b_cp("a,b,c,d,e");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < 1e-10);
    if(!accurate) std::cout << "The error in order-5 sparse is : " << TA::norm2(diff) / TA::norm2(b_sparse) << std::endl;
    BOOST_CHECK(accurate);
  }
}

BOOST_AUTO_TEST_CASE(btas_cp_rals){
  // get local world
  const auto rank = (*GlobalFixture::world).rank();
  const auto size = (*GlobalFixture::world).size();

  madness::World* tmp_ptr;
  std::shared_ptr<madness::World> world_ptr;

  if (size > 1) {
    SafeMPI::Group group =
        (*GlobalFixture::world).mpi.comm().Get_group().Incl(1, &rank);
    SafeMPI::Intracomm comm = (*GlobalFixture::world).mpi.comm().Create(group);
    world_ptr = std::make_shared<madness::World>(comm);
    tmp_ptr = world_ptr.get();
  } else {
    tmp_ptr = &(*GlobalFixture::world);
  }
  auto& this_world = *tmp_ptr;
  (*GlobalFixture::world).gop.fence();

  // Make a tiled range with block size of 1
  TiledArray::TiledRange tr3, tr4, tr5;
  {
    TiledRange1 tr1_mode0 = compute_trange1(11, 1),
                tr1_mode1 = compute_trange1(7, 2),
                tr1_mode2 = compute_trange1(15, 4),
                tr1_mode3 = compute_trange1(8,7);
    tr3 = TiledRange({tr1_mode0, tr1_mode1, tr1_mode2});
    tr4 = TiledRange({tr1_mode0, tr1_mode1, tr1_mode2, tr1_mode3});
    tr5 = TiledRange({tr1_mode0, tr1_mode0, tr1_mode1, tr1_mode2, tr1_mode3});
  }

  // Dense test
  // order-3 test
  {
    std::vector<TArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_dense = make_array<TArrayD>(*GlobalFixture::world, tr3,
                                       &this->init_unit_tile<TensorD>);
    double cp_rank = 1;
    factors = cp::btas_cp_rals(*GlobalFixture::world, b_dense, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);

    auto b_cp = cp::reconstruct(factors);
    TArrayD diff;
    diff("a,b,c") = b_dense("a,b,c") - b_cp("a,b,c");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < 1e-10);
    BOOST_CHECK(accurate);
  }
  // order-4 test
  {
    std::vector<TArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_dense = make_array<TArrayD>(*GlobalFixture::world, tr4,
                                       &this->init_unit_tile<TensorD>);
    double cp_rank = 1;
    factors = cp::btas_cp_rals(*GlobalFixture::world, b_dense, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);

    auto b_cp = cp::reconstruct(factors);
    TArrayD diff;
    diff("a,b,c,d") = b_dense("a,b,c,d") - b_cp("a,b,c,d");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < 1e-10);
    BOOST_CHECK(accurate);
  }
  // order-5 test
  {
    std::vector<TArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_dense = make_array<TArrayD>(*GlobalFixture::world, tr5,
                                       &this->init_unit_tile<TensorD>);
    double cp_rank = 1;
    factors = cp::btas_cp_rals(*GlobalFixture::world, b_dense, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);

    auto b_cp = cp::reconstruct(factors);
    TArrayD diff;
    diff("a,b,c,d,e") = b_dense("a,b,c,d,e") - b_cp("a,b,c,d,e");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_dense) < 1e-10);
    BOOST_CHECK(accurate);
  }

  // sparse test
  {
    std::vector<TSpArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_sparse = make_array<TSpArrayD>(*GlobalFixture::world, tr3,
                                          &this->init_rand_tile<TensorD>);
    double cp_rank = 100;
    factors = cp::btas_cp_rals(*GlobalFixture::world, b_sparse, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);

    auto b_cp = cp::reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c") = b_sparse("a,b,c") - b_cp("a,b,c");
    bool accurate = ( TA::norm2(diff) / TA::norm2(b_sparse) < 1e-10);
    BOOST_CHECK(accurate);
  }
  // order-4 test
  {
    std::vector<TSpArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_sparse = make_array<TSpArrayD>(*GlobalFixture::world, tr4,
                                          &this->init_unit_tile<TensorD>);
    double cp_rank = 100;
    factors = cp::btas_cp_rals(*GlobalFixture::world, b_sparse, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);

    auto b_cp = cp::reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c,d") = b_sparse("a,b,c,d") - b_cp("a,b,c,d");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < 1e-10);
    BOOST_CHECK(accurate);
  }
  // order-5 test
  {
    std::vector<TSpArrayD> factors;
    // Make an sparse array with tiled range from above.
    auto b_sparse = make_array<TSpArrayD>(*GlobalFixture::world, tr5,
                                          &this->init_unit_tile<TensorD>);
    double cp_rank = 105;
    factors = cp::btas_cp_als(*GlobalFixture::world, b_sparse, cp_rank,
                              compute_trange1(cp_rank, 80),
                              0, 1e-3, false);

    auto b_cp = cp::reconstruct(factors);
    TSpArrayD diff;
    diff("a,b,c,d,e") = b_sparse("a,b,c,d,e") - b_cp("a,b,c,d,e");
    bool accurate = (TA::norm2(diff) / TA::norm2(b_sparse) < 1e-10);
    BOOST_CHECK(accurate);
  }
}

BOOST_AUTO_TEST_SUITE_END()
