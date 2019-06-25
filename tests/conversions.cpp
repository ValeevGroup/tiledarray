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
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *
 *  conversions.cpp
 *  May 8, 2018
 *
 */

#include "range_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include "TiledArray/conversions/vector_of_arrays.h"

using namespace TiledArray;

struct ConversionsFixture : public TiledRangeFixture {
  ConversionsFixture()
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
        array.set(
            *it,
            array.world().taskq.add(
                &ConversionsFixture::template make_rand_tile<Tile>,
                array.trange().make_tile_range(*it)));
    }
  }

  /// make a shape with approximate half dense and half sparse
  static SparseShape<float> make_random_sparseshape(const TiledRange& tr) {
    std::size_t n = tr.tiles_range().volume();
    Tensor<float> norms(tr.tiles_range(), 0.0);

    // make sure all mpi gets the same shape
    if(GlobalFixture::world->rank() == 0){
      for (std::size_t i = 0; i < n; i++) {
        norms[i] = GlobalFixture::world->drand() > 0.5 ? 0.0 : 1.0;
      }
    }

    GlobalFixture::world->gop.broadcast_serializable(norms, 0);

    return SparseShape<float>(norms, tr);
  }

  // Fill a tile with random data
  template <typename Tile>
  static Tile make_rand_tile(
      const typename Tile::range_type& r) {
    Tile tile(r);
    for (std::size_t i = 0ul; i < tile.size(); ++i) set_random(tile[i]);
    return tile;
  }

  template <typename Tile>
  static double init_rand_tile(
          Tile& tile, const typename Tile::range_type& r) {
    tile =  Tile(r);
    for (std::size_t i = 0ul; i < tile.size(); ++i) set_random(tile[i]);
    return tile.norm();
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

  ~ConversionsFixture() { GlobalFixture::world->gop.fence(); }

  SparseShape<float> shape_tr;
  TArrayI a_dense;
  TSpArrayI a_sparse;
};

TiledArray::TiledRange1 compute_trange1(std::size_t range_size,
                                std::size_t block_size) {
  if (range_size > 0) {
    std::vector<std::size_t> blocks;
    blocks.push_back(0);
    for (std::size_t i = block_size; i < range_size; i += block_size) {
      blocks.push_back(i);
    }
    blocks.push_back(range_size);
    return TA::TiledRange1(blocks.begin(), blocks.end());
  }
  else
    return TA::TiledRange1{};
}

BOOST_FIXTURE_TEST_SUITE(conversions_suite, ConversionsFixture)

BOOST_AUTO_TEST_CASE(policy_conversions) {
  GlobalFixture::world->gop.fence();
  // convert sparse to dense
//  std::cout << a_sparse.shape() << std::endl;
  BOOST_CHECK_NO_THROW(a_dense = to_dense(a_sparse));
  // convert dense back to sparse
  TSpArrayI b_sparse;
  BOOST_CHECK_NO_THROW(b_sparse = to_sparse(a_dense));
//  std::cout << b_sparse.shape() << std::endl;

  BOOST_CHECK_EQUAL(a_sparse.shape().data(), b_sparse.shape().data());

  // check correctness
  for (std::size_t i = 0; i < a_sparse.size(); i++) {
    if (!a_sparse.is_zero(i)) {
      TSpArrayI::value_type a_tile = a_sparse.find(i).get();
      TSpArrayI::value_type b_tile = b_sparse.find(i).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        BOOST_CHECK_EQUAL(a_tile[j], b_tile[j]);
    } else {
      BOOST_CHECK(b_sparse.is_zero(i));
    }
  }

}

BOOST_AUTO_TEST_CASE(tile_element_conversions) {
  // convert int to float
  TSpArrayF a_f_sparse;
  BOOST_CHECK_NO_THROW(
      a_f_sparse = to_new_tile_type(a_sparse, &this->tensori_to_tensorf));

  // convert float to int
  TSpArrayI b_sparse;
  BOOST_CHECK_NO_THROW(
      b_sparse = to_new_tile_type(a_f_sparse, &this->tensorf_to_tensori));

  // check correctness
  for (std::size_t i = 0; i < a_sparse.size(); i++) {
    if (!a_sparse.is_zero(i)) {
      TSpArrayI::value_type a_tile = a_sparse.find(i).get();
      TSpArrayI::value_type b_tile = b_sparse.find(i).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        BOOST_CHECK_EQUAL(a_tile[j], b_tile[j]);
    } else {
      BOOST_CHECK(b_sparse.is_zero(i));
    }
  }
}

BOOST_AUTO_TEST_CASE(make_array_test) {
  // make dense array
  BOOST_CHECK_NO_THROW(auto b_dense =
                           make_array<TArrayI>(*GlobalFixture::world, this->tr,
                                               &this->init_rand_tile<TensorI>));

  // make sparse array
  BOOST_CHECK_NO_THROW(
      auto b_sparse = make_array<TSpArrayI>(*GlobalFixture::world, this->tr,
                                            &this->init_rand_tile<TensorI>));
}

BOOST_AUTO_TEST_CASE(vector_of_arrays_unit_blocking){

  // Make a tiled range with block size of 1
  TiledArray::TiledRange tr;
  TiledArray::TiledRange tr_split;
  {
    TA::TiledRange1 tr1_mode0 = compute_trange1(10, 1);
    TA::TiledRange1 tr1_mode1 = compute_trange1(7, 1);
    tr = TiledArray::TiledRange({tr1_mode0,tr1_mode1});
    tr_split = TiledArray::TiledRange({tr1_mode1});
  }

  // Make an array with tiled range from above.
  auto b_dense = make_array<TArrayI>(*GlobalFixture::world, tr,
          &this->init_rand_tile<TensorI>);

  // Grab number of tiles in fused mode
  auto text = b_dense.trange().tiles_range().extent_data();
  auto num_mode0_tiles = text[0];
  auto num_mode1_tiles = text[1];

  // Convert dense array to vector of arrays
  std::vector<TArrayI> b_dense_vector;
  for(int i = 0; i < 10; ++i){
    b_dense_vector.push_back(
            TiledArray::subarray_from_fused_array(b_dense,
                                                  i, tr_split));
  }

  //convert vector of arrays back into dense array
  auto b_dense_fused = TiledArray::fuse_vector_of_arrays(b_dense_vector);

  // Check to see if the fused and original arrays are the same
  for (std::size_t i = 0; i < num_mode0_tiles; ++i) {
    for (std::size_t j = 0; j < num_mode1_tiles; ++j) {
      auto tile_orig = b_dense.find({i,j}).get();
      auto tile_fused = b_dense_fused.find({i,j}).get();

      auto lo = tile_orig.range().lobound_data();
      auto up = tile_orig.range().upbound_data();
      for (auto k = lo[0]; k < up[0]; ++k) {
        for (auto l = lo[1]; l < up[1]; ++l) {
          BOOST_CHECK_EQUAL(tile_orig(k,l), tile_fused(k,l));
        }
      }
    }
  }

  // Make an sparse array with tiled range from above.
  auto b_sparse = make_array<TSpArrayI>(*GlobalFixture::world, tr,
                                     &this->init_rand_tile<TensorI>);

  // Grab number of tiles in fused mode
  text = b_dense.trange().tiles_range().extent_data();
  num_mode0_tiles = text[0];
  num_mode1_tiles = text[1];

  // Convert sparse array to vector of arrays
  std::vector<TSpArrayI> b_sparse_vector;
  for(int i = 0; i < 10; ++i){
    b_sparse_vector.push_back(
            TiledArray::subarray_from_fused_array(b_sparse,
                                                  i, tr_split));
  }

  // convert vector of arrays back into sparse array
  auto b_sparse_fused = TiledArray::fuse_vector_of_arrays(b_sparse_vector);

  // Check to see if the fused and original arrays are the same
  for (std::size_t i = 0; i < num_mode0_tiles; ++i) {
    for (std::size_t j = 0; j < num_mode1_tiles; ++j) {
      auto tile_orig = b_dense.find({i,j}).get();
      auto tile_fused = b_dense_fused.find({i,j}).get();

      auto lo = tile_orig.range().lobound_data();
      auto up = tile_orig.range().upbound_data();
      for (auto k = lo[0]; k < up[0]; ++k) {
        for (auto l = lo[1]; l < up[1]; ++l) {
          BOOST_CHECK_EQUAL(tile_orig(k,l), tile_fused(k,l));
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(vector_of_arrays_non_unit_blocking){

    // Make a tiled range with block size of 1
    TiledArray::TiledRange tr;
    TiledArray::TiledRange tr_split;
    {
      TA::TiledRange1 tr1_mode0 = compute_trange1(10, 2);
      TA::TiledRange1 tr1_mode1 = compute_trange1(7, 3);
      tr = TiledArray::TiledRange({tr1_mode0,tr1_mode1});
      tr_split = TiledArray::TiledRange({tr1_mode1});
    }

    // Make an array with tiled range from above.
    auto b_dense = make_array<TArrayI>(*GlobalFixture::world, tr,
                                       &this->init_rand_tile<TensorI>);

    // Grab number of tiles in fused mode
    auto text = b_dense.trange().tiles_range().extent_data();
    auto num_mode0_tiles = text[0];
    auto num_mode1_tiles = text[1];

    // Convert dense array to vector of arrays
    std::vector<TArrayI> b_dense_vector;
    for(int i = 0; i < 10; ++i){
      b_dense_vector.push_back(
              TiledArray::subarray_from_fused_array(b_dense,
                                                    i, tr_split));
    }

    //convert vector of arrays back into dense array
    auto b_dense_fused = TiledArray::fuse_vector_of_arrays(b_dense_vector,2);

    // Check to see if the fused and original arrays are the same
    for (std::size_t i = 0; i < num_mode0_tiles; ++i) {
      for (std::size_t j = 0; j < num_mode1_tiles; ++j) {
        auto tile_orig = b_dense.find({i,j}).get();
        auto tile_fused = b_dense_fused.find({i,j}).get();

        auto lo = tile_orig.range().lobound_data();
        auto up = tile_orig.range().upbound_data();
        for (auto k = lo[0]; k < up[0]; ++k) {
          for (auto l = lo[1]; l < up[1]; ++l) {
            BOOST_CHECK_EQUAL(tile_orig(k,l), tile_fused(k,l));
          }
        }
      }
    }

    // Make an sparse array with tiled range from above.
    auto b_sparse = make_array<TSpArrayI>(*GlobalFixture::world, tr,
                                          &this->init_rand_tile<TensorI>);

    // Grab number of tiles in fused mode
    text = b_dense.trange().tiles_range().extent_data();
    num_mode0_tiles = text[0];
    num_mode1_tiles = text[1];

    // Convert sparse array to vector of arrays
    std::vector<TSpArrayI> b_sparse_vector;
    for(int i = 0; i < 10; ++i){
      b_sparse_vector.push_back(
              TiledArray::subarray_from_fused_array(b_sparse,
                                                    i, tr_split));
    }

    // convert vector of arrays back into sparse array
    auto b_sparse_fused = TiledArray::fuse_vector_of_arrays(b_sparse_vector,2);

    // Check to see if the fused and original arrays are the same
    for (std::size_t i = 0; i < num_mode0_tiles; ++i) {
      for (std::size_t j = 0; j < num_mode1_tiles; ++j) {
        auto tile_orig = b_dense.find({i,j}).get();
        auto tile_fused = b_dense_fused.find({i,j}).get();

        auto lo = tile_orig.range().lobound_data();
        auto up = tile_orig.range().upbound_data();
        for (auto k = lo[0]; k < up[0]; ++k) {
          for (auto l = lo[1]; l < up[1]; ++l) {
            BOOST_CHECK_EQUAL(tile_orig(k,l), tile_fused(k,l));
          }
        }
      }
    }
}

BOOST_AUTO_TEST_SUITE_END()
