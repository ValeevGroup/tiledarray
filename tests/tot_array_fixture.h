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
#include <TiledArray/conversions/btas.h>
#include <TiledArray/external/btas.h>
#include <btas/generic/contract.h>
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

enum class ShapeComp { True, False };

template <typename TensorT,
          std::enable_if_t<TA::detail::is_tensor_v<TensorT>, bool> = true>
auto random_tensor(TA::Range const& rng) {
  using Ix1 = typename TA::Range::index1_type;
  using Num = typename TensorT::numeric_type;
  TensorT result{rng};
  using NumericT = typename TensorT::numeric_type;
  for (auto const& ix : rng) {
    result(ix) =
        static_cast<Num>(std::accumulate(ix.begin(), ix.end(), Ix1{0}));
  }
  //  std::generate(/*std::execution::par, */
  //                result.begin(), result.end(),
  //                TA::detail::MakeRandom<char>::generate_value);
  return result;
}

//
// note: all the inner tensors (elements of the outer tensor)
//       have the same @c inner_rng
//
template <
    typename TensorT,
    std::enable_if_t<TA::detail::is_tensor_of_tensor_v<TensorT>, bool> = true>
auto random_tensor(TA::Range const& outer_rng, TA::Range const& inner_rng) {
  using InnerTensorT = typename TensorT::value_type;
  using Num = typename TensorT::numeric_type;
  using Ix1 = typename TA::Range::index1_type;
  TensorT result{outer_rng};

  for (auto const& ix : outer_rng) {
    auto inner = random_tensor<InnerTensorT>(inner_rng);
    auto plus = std::accumulate(ix.begin(), ix.end(), Ix1{0});
    inner.add_to(static_cast<Num>(plus));
    result(ix) = inner;
  }

  //  std::generate(/*std::execution::par,*/
  //                result.begin(), result.end(), [inner_rng]() {
  //                  return random_tensor<InnerTensorT>(inner_rng);
  //                });

  return result;
}

///
/// \tparam Array The type of DistArray to be generated. Cannot be cv-qualified
/// or reference type.
/// \tparam Args TA::Range type for inner tensor if the tile type of the result
/// is a tensor-of-tensor.
/// \param trange The TiledRange of the result DistArray.
/// \param args Either exactly one TA::Range type when the tile type of Array is
/// tensor-of-tensor or nothing.
/// \return Returns a DistArray of type Array whose elements are randomly
/// generated.
/// @note:
/// - Although DistArrays with Sparse policy can be generated all of their
///   tiles are initialized with random values -- technically the returned value
///   is dense.
/// - In case of arrays with tensor-of-tensor tiles, all the inner tensors have
///   the same rank and the same extent of corresponding modes.
///
template <
    typename Array, typename... Args,
    typename =
        std::void_t<typename Array::value_type, typename Array::policy_type>,
    std::enable_if_t<TA::detail::is_nested_tensor_v<typename Array::value_type>,
                     bool> = true>
auto random_array(TA::TiledRange const& trange, Args const&... args) {
  static_assert(
      (sizeof...(Args) == 0 &&
       TA::detail::is_tensor_v<typename Array::value_type>) ||
      (sizeof...(Args) == 1) &&
          (TA::detail::is_tensor_of_tensor_v<typename Array::value_type>));

  if constexpr (sizeof...(Args) == 1)
    static_assert(std::is_convertible_v<Args..., TA::Range>);

  using TensorT = typename Array::value_type;
  using PolicyT = typename Array::policy_type;

  auto make_tile_meta = [](auto&&... args) {
    return [=](TensorT& tile, TA::Range const& rng) {
      tile = random_tensor<TensorT>(rng, args...);
      if constexpr (std::is_same_v<TA::SparsePolicy, PolicyT>)
        return tile.norm();
    };
  };

  return TA::make_array<Array>(TA::get_default_world(), trange,
                               make_tile_meta(args...));
}

///
/// Succinctly call TA::detail::tensor_contract
///
/// \tparam T TA::Tensor type.
/// \param einsum_annot Example annot: 'ik,kj->ij', when @c A is annotated by
/// 'i' and 'k' for its two modes, and @c B is annotated by 'k' and 'j' for the
/// same. The result tensor is rank-2 as well and its modes are annotated by 'i'
/// and 'j'.
/// \return Tensor contraction result.
///
template <typename T, std::enable_if_t<TA::detail::is_tensor_v<T>, bool> = true>
auto tensor_contract(std::string const& einsum_annot, T const& A, T const& B) {
  using ::Einsum::string::split2;
  auto [ab, aC] = split2(einsum_annot, "->");
  auto [aA, aB] = split2(ab, ",");

  return TA::detail::tensor_contract(A, aA, B, aB, aC);
}

#ifdef TILEDARRAY_HAS_BTAS

template <typename T, typename = std::enable_if_t<TA::detail::is_tensor_v<T>>>
auto tensor_to_btas_tensor(T const& ta_tensor) {
  using value_type = typename T::value_type;
  using range_type = typename T::range_type;

  btas::Tensor<value_type, range_type> result{ta_tensor.range()};
  TA::tensor_to_btas_subtensor(ta_tensor, result);
  return result;
}

template <typename NumericT, typename RangeT, typename... Ts,
          typename = std::enable_if_t<std::is_convertible_v<RangeT, TA::Range>>>
auto btas_tensor_to_tensor(
    btas::Tensor<NumericT, RangeT, Ts...> const& btas_tensor) {
  TA::Tensor<NumericT> result{TA::Range(btas_tensor.range())};
  TA::btas_subtensor_to_tensor(btas_tensor, result);
  return result;
}

///
/// @c einsum_annot pattern example: 'ik,kj->ij'. See tensor_contract function.
///
template <typename T, std::enable_if_t<TA::detail::is_tensor_v<T>, bool> = true>
auto tensor_contract_btas(std::string const& einsum_annot, T const& A,
                          T const& B) {
  using ::Einsum::string::split2;
  auto [ab, aC] = split2(einsum_annot, "->");
  auto [aA, aB] = split2(ab, ",");

  using NumericT = typename T::numeric_type;

  struct {
    btas::Tensor<NumericT, TA::Range> A, B, C;
  } btas_tensor{tensor_to_btas_tensor(A), tensor_to_btas_tensor(B), {}};

  btas::contract(NumericT{1}, btas_tensor.A, aA, btas_tensor.B, aB, NumericT{0},
                 btas_tensor.C, aC);

  return btas_tensor_to_tensor(btas_tensor.C);
}

///
/// \tparam T TA::Tensor type
/// \param einsum_annot see tensor_contract_mult
/// \return True when TA::detail::tensor_contract and btas::contract result the
///         result. Performs bitwise comparison.
///
template <typename T, typename = std::enable_if_t<TA::detail::is_tensor_v<T>>>
auto tensor_contract_equal(std::string const& einsum_annot, T const& A,
                           T const& B) {
  T result_ta = tensor_contract(einsum_annot, A, B);
  T result_btas = tensor_contract_btas(einsum_annot, A, B);
  return result_ta == result_btas;
}

#endif

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
   * - Same shape (unless the template parameter ShapeCmp is set false)
   * - Same distribution
   * - Same tiling
   * - Components are bit-wise equal (i.e., 3.1400000000 != 3.1400000001)
   *
   * TODO: pmap comparisons
   */
  template <ShapeComp ShapeCompFlag = ShapeComp::True, typename LHSTileType,
            typename LHSPolicy, typename RHSTileType, typename RHSPolicy>
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
      if constexpr (ShapeCompFlag == ShapeComp::True)
        if (lhs.shape() != rhs.shape()) return false;

      // Same pmap?
      // if(*lhs.pmap() != *rhs.pmap()) return false;

      // Same tiling?
      if (lhs.trange() != rhs.trange()) return false;

      // Same components? Here we make all ranks check all tiles
      bool are_same = true;
      for (auto idx : lhs.tiles_range()) {
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
    abort();  // unreachable
  }
  // The world to use for the test suite
  madness::World& m_world;

};  // TotArrayFixture
#endif
