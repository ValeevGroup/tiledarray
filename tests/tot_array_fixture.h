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

#ifndef TILEDARRAY_TEST_TOT_ARRAY_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_TOT_ARRAY_FIXTURE_H__INCLUDED
#include "tiledarray.h"
#include "unit_test_config.h"
#ifdef TILEDARRAY_HAS_BTAS
#include <TiledArray/external/btas.h>
#endif

/* Notes:
 *
 * This test suite currently does not test:
 * - wait_for_lazy_cleanup (either overload)
 * - id() (documentation suggests it's not part of the public API)
 * - elements (it's deprecated)
 * - get_world (it's deprecated)
 * - get_pmap (it's deprecated)
 * - register_set_notifier
 */

using namespace TiledArray;

// These are all of the template parameters we are going to test over
using test_params =
    boost::mpl::list<std::tuple<int, Tensor<Tensor<int>>>,
                     std::tuple<float, Tensor<Tensor<float>>>,
                     std::tuple<double, Tensor<Tensor<double>>>
#ifdef TILEDARRAY_HAS_BTAS
                     ,
                     std::tuple<int, Tensor<btas::Tensor<int, Range>>>,
                     std::tuple<float, Tensor<btas::Tensor<float, Range>>>,
                     std::tuple<double, Tensor<btas::Tensor<double, Range>>>
//    ,std::tuple<int, btas::Tensor<btas::Tensor<int, Range>, Range>>,
//    std::tuple<float, btas::Tensor<btas::Tensor<float, Range>, Range>>,
//    std::tuple<double, btas::Tensor<btas::Tensor<double, Range>, Range>>
//    ,std::tuple<int, Tile<btas::Tensor<btas::Tensor<int, Range>, Range>>>,
//    std::tuple<float, Tile<btas::Tensor<btas::Tensor<float, Range>, Range>>>,
//    std::tuple<double, Tile<btas::Tensor<btas::Tensor<double, Range>, Range>>>
#endif
                     >;

// These typedefs unpack the unit test template parameter
//{
template <typename TupleElementType>
using scalar_type = std::tuple_element_t<0, TupleElementType>;

template <typename TupleElementType>
using tile_type = std::tuple_element_t<1, TupleElementType>;

template <typename>
using policy_type = DensePolicy;
//}

// The type of a DistArray consistent with the unit test template parameter
template <typename TupleElementType>
using tensor_type =
    DistArray<tile_type<TupleElementType>, policy_type<TupleElementType>>;

// Type of the object storing the tiling
template <typename TupleElementType>
using trange_type = typename tensor_type<TupleElementType>::trange_type;

// Type of the inner tile
template <typename TupleElementType>
using inner_type = typename tile_type<TupleElementType>::value_type;

// Type of an input archive
using input_archive_type = madness::archive::BinaryFstreamInputArchive;

// Type of an output archive
using output_archive_type = madness::archive::BinaryFstreamOutputArchive;

/*
 *
 * When generating arrays containing tensors of tensors (ToT) we adopt simple
 * algorithms for initialing the values. These algorithms map the outer indices
 * to the values of the inner tensor in such a way that the inner tensors will
 * have differing extents (they must all have the same rank).
 */
struct ToTArrayFixture {
  ToTArrayFixture() : m_world(*GlobalFixture::world) {}
  ~ToTArrayFixture() { GlobalFixture::world->gop.fence(); }

  /* This function returns an std::vector of tiled ranges. The ranges are for a
   * tensor of rank 1 and cover the following scenarios:
   *
   * - A single element in a single tile
   * - Multiple elements in a single tile
   * - Multiple tiles with a single element each
   * - Multiple tiles, one with a single element and one with multiple elements,
   * - Multiple tiles, each with two elements
   */
  template <typename TupleElementType>
  auto vector_tiled_ranges() {
    using trange_type = trange_type<TupleElementType>;
    return std::vector<trange_type>{
        trange_type{{0, 1}}, trange_type{{0, 2}}, trange_type{{0, 1, 2}},
        trange_type{{0, 1, 3}}, trange_type{{0, 2, 4}}};
  }

  /* This function returns an std::vector of tiled ranges. The ranges are for a
   * tensor of rank 2 and cover the following scenarios:
   * - Single tile
   *   - single element
   *   - multiple elements on row/column but single element on column/row
   *   - multiple elements on rows and columns
   * - multiple tiles on rows/columns, but single tile on column/rows
   *   - single element row/column and multiple element column/row
   *   - Multiple elements in both rows and columns
   * - multiple tiles on rows and columns
   */
  template <typename TupleElementType>
  auto matrix_tiled_ranges() {
    using trange_type = trange_type<TupleElementType>;
    return std::vector<trange_type>{
        trange_type{{0, 1}, {0, 1}},      trange_type{{0, 2}, {0, 1}},
        trange_type{{0, 2}, {0, 2}},      trange_type{{0, 1}, {0, 2}},
        trange_type{{0, 1}, {0, 1, 2}},   trange_type{{0, 1, 2}, {0, 1}},
        trange_type{{0, 2}, {0, 1, 2}},   trange_type{{0, 1, 2}, {0, 2}},
        trange_type{{0, 1, 2}, {0, 1, 2}}};
  }

  template <typename TupleElementType, typename Index>
  auto inner_vector_tile(Index&& idx) {
    auto sum = std::accumulate(idx.begin(), idx.end(), 0);
    inner_type<TupleElementType> elem(Range(sum + 1));
    std::iota(elem.begin(), elem.end(), 1);
    return elem;
  }

  template <typename TupleElementType, typename Index>
  auto inner_matrix_tile(Index&& idx) {
    unsigned int row_max = idx[0] + 1;
    unsigned int col_max = std::accumulate(idx.begin(), idx.end(), 0) + 1;
    unsigned int zero = 0;
    inner_type<TupleElementType> elem(Range({zero, zero}, {row_max, col_max}));
    std::iota(elem.begin(), elem.end(), 1);
    return elem;
  }

  template <typename TupleElementType>
  auto tensor_of_vector(const TiledRange& tr) {
    return make_array<tensor_type<TupleElementType>>(
        m_world, tr, [this](tile_type<TupleElementType>& tile, const Range& r) {
          tile_type<TupleElementType> new_tile(r);
          for (auto idx : r) {
            new_tile(idx) = inner_vector_tile<TupleElementType>(idx);
          }
          tile = new_tile;
        });
  }

  template <typename TupleElementType>
  auto tensor_of_matrix(const TiledRange& tr) {
    return make_array<tensor_type<TupleElementType>>(
        m_world, tr, [this](tile_type<TupleElementType>& tile, const Range& r) {
          tile_type<TupleElementType> new_tile(r);
          for (auto idx : r) {
            new_tile(idx) = inner_matrix_tile<TupleElementType>(idx);
          }
          tile = new_tile;
        });
  }

  /* The majority of the unit tests can be written in such a way that all they
   * need is the tiled range, the rank of the inner tensors, and the resulting
   * array. The tiled range plus the inner rank is more or less the correct
   * answer, since the ToTArrayFixture maps it to the values of the array, and
   * the tensor is the thing to test. This function creates a std::vector of
   * tiled range, inner rank, tensor tuples spanning:
   *
   * - vector of vectors
   * - vector of matrices
   * - matrix of vectors
   * - matrix of matrices
   *
   * The unit tests simply loop over the vector testing for all these scenarios
   * with a few lines of code.
   */
  template <typename TupleElementType>
  auto run_all() {
    using trange_type = trange_type<TupleElementType>;
    using tensor_type = tensor_type<TupleElementType>;
    std::vector<std::tuple<trange_type, int, tensor_type>> rv;

    // Make vector of vector
    for (auto tr : vector_tiled_ranges<TupleElementType>())
      rv.push_back(
          std::make_tuple(tr, 1, tensor_of_vector<TupleElementType>(tr)));

    // Make vector of matrix
    for (auto tr : vector_tiled_ranges<TupleElementType>())
      rv.push_back(
          std::make_tuple(tr, 2, tensor_of_matrix<TupleElementType>(tr)));

    // Make matrix of vector
    for (auto tr : matrix_tiled_ranges<TupleElementType>())
      rv.push_back(
          std::make_tuple(tr, 1, tensor_of_vector<TupleElementType>(tr)));

    // Make matrix of matrix
    for (auto tr : matrix_tiled_ranges<TupleElementType>())
      rv.push_back(
          std::make_tuple(tr, 2, tensor_of_matrix<TupleElementType>(tr)));

    // Make sure all the tensors are actually made
    m_world.gop.fence();
    return rv;
  }

  /* This function tests for exact equality between DistArray instances. By
   * exact equality we mean:
   * - Same type
   * - Either both are initialized or both are not initialized
   * - Same MPI context
   * - Same shape
   * - Same distribution
   * - Same tiling
   * - Components are bit-wise equal (i.e., 3.1400000000 != 3.1400000001)
   *
   * TODO: pmap comparisons
   */
  template <typename LHSTileType, typename LHSPolicy, typename RHSTileType,
            typename RHSPolicy>
  static bool are_equal(const DistArray<LHSTileType, LHSPolicy>& lhs,
                        const DistArray<RHSTileType, RHSPolicy>& rhs) {
    // Same type
    if constexpr (!std::is_same_v<decltype(lhs), decltype(rhs)>) {
      return false;
    } else {
      // Are initialized?
      if (lhs.is_initialized() != rhs.is_initialized()) return false;
      if (!lhs.is_initialized()) return true;  // both are default constructed

      // Same world instance?
      if (&lhs.world() != &rhs.world()) return false;

      // Same shape?
      if (lhs.shape() != rhs.shape()) return false;

      // Same pmap?
      // if(*lhs.pmap() != *rhs.pmap()) return false;

      // Same tiling?
      if (lhs.trange() != rhs.trange()) return false;

      // Same components? Here we make all ranks check all tiles
      bool are_same = true;
      for (auto idx : lhs.range()) {
        const auto& lhs_tot = lhs.find(idx).get();
        const auto& rhs_tot = rhs.find(idx).get();
        if (lhs_tot != rhs_tot) {
          are_same = false;
          break;
        }
      }
      lhs.world().gop.fence();
      return are_same;
    }
  }
  // The world to use for the test suite
  madness::World& m_world;

};  // TotArrayFixture
#endif
