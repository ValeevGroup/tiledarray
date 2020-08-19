/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 */

#ifndef TILEDARRAY_INITIALIZER_LIST_UTILS_H__INCLUDED
#define TILEDARRAY_INITIALIZER_LIST_UTILS_H__INCLUDED
#include <TiledArray/tiled_range.h>
#include <TiledArray/tiled_range1.h>
#include <TiledArray/type_traits.h>
#include <algorithm>         // copy
#include <array>             // array
#include <initializer_list>  // initializer_list
#include <type_traits>       // decay, false_type, true_type

/** @file util/initializer_list.h
 *
 *  util/initializer_list.h contains routines and template meta-programming
 *  utilities for manipulating std::initializer_list instances.
 */

namespace TiledArray {

//------------------------------------------------------------------------------
// InitializerListRank Struct
//------------------------------------------------------------------------------

/** @brief Primary template for determining how many nested
 *         std::initializer_list's are in a type.
 *
 *  This is the primary template for determining how many`std::initializer_list`
 *  are in @p T. It is selected when @p T is **NOT** an `std::initializer_list`
 *  and will contain a static constexpr member `value` of type @p SizeType equal
 *  to 0.
 *
 *  @tparam T The type we are analyzing for its
 *          std::initializer_list-nested-ness
 *  @tparam SizeType the type to use for the `value` member. Defaults to
 *          `std::size_t`.
 */
template <typename T, typename SizeType = std::size_t>
struct InitializerListRank : std::integral_constant<SizeType, 0> {};

/** @brief Helper variable for retrieving the degree of nesting for an
 *         std::initializer_list.
 *
 *  This helper variable creates a global variable which contains the value
 *  of InitializerListRank<T, SizeType> and is intended to be used as a (more)
 *  convenient means of retrieving the value.
 *
 *  @tparam T The type we are analyzing for its
 *          std::initializer_list-nested-ness
 *  @tparam SizeType the type to use for the `value` member. Defaults to
 *          `std::size_t`.
 */
template <typename T, typename SizeType = std::size_t>
constexpr auto initializer_list_rank_v =
    InitializerListRank<T, SizeType>::value;

/** @brief Specialization of InitializerListRank used when the template type
 *         parameter is a std::initializer_list type
 *
 *  This specialization is selected when InitializerListRank is parameterized
 *  with an `std::initializer_list` and will contain a static constexpr member
 *  `value` of type @p SizeType equal to 1 plus the number of
 *  `std::initializer_list`s in @p T.
 *
 *  @tparam T The type we are analyzing for its
 *          std::initializer_list-nested-ness
 *  @tparam SizeType the type to use for the `value` member. Defaults to
 *          `std::size_t`.
 */
template <typename T, typename SizeType>
struct InitializerListRank<std::initializer_list<T>, SizeType>
    : std::integral_constant<SizeType,
                             initializer_list_rank_v<T, SizeType> + 1> {};

//------------------------------------------------------------------------------
// tiled_range_from_il free function
//------------------------------------------------------------------------------

/** @brief Creates a TiledRange for the provided initializer list
 *
 *  Tensors which are constructed with initializer lists are assumed to be small
 *  enough that the data should reside in a single tile. This function will
 *  recurse through @p il and create a TiledRange instance such that each rank
 *  is tiled from `[0, n_i)` where `n_i` is the length of the `i`-th nested
 *  `std::initializer_list` in @p il.
 *
 * @tparam T Expected to be the type of a tensor element (*i.e.* float, double,
 *           *etc.*) or a (possibly nested) `std::initializer_list` of tensor
 *           elements.
 * @tparam U The type of the container which will hold the TiledRange1 instances
 *           for each level of nesting in @p T. @p U must satisfy the concept of
 *           a random-access container. @p U defaults to
 *           `std::array<TiledRange1, N>` where `N` is the degree of nesting of
 *           @p il.
 *
 * @param[in] il The state we intend to initialize the tensor to.
 * @param[in] shape A pre-allocated buffer that will be used to hold the
 *                  TiledRange1 instances for each `std::initializer_list` as
 *                  this function recurses. The default value is an `std::array`
 *                  of default constructed  TiledRange1 instances, which should
 *                  suffice for most purposes.
 *
 * @return A TiledRange instance consistent with treating @p il as a tensor with
 *         a single tile.
 *
 * @throw TiledArray::Exception if @p il contains no elements. If an exception
 *                              is raised this way @p il and @p shape are
 *                              guaranteed to be in the same state (strong throw
 *                              guarantee).
 */
template <typename T,
          typename U =
              std::array<TiledRange1, initializer_list_rank_v<std::decay_t<T>>>>
auto tiled_range_from_il(T&& il, U shape = {}) {
  using clean_type = std::decay_t<T>;
  constexpr auto ranks_left = initializer_list_rank_v<clean_type>;

  if constexpr (ranks_left == 0) {  // Scalar or end of recursion
    return TiledRange(shape.begin(), shape.end());
  } else {
    // The length of this initializer_list
    const auto length = il.size();
    TA_ASSERT(length > 0);

    // This nesting level = (total-nestings) - (nestings-left)
    const auto this_rank = shape.size() - ranks_left;
    shape[this_rank] = TiledRange1(0, length);

    // verify that each sub-IL (if a list) has same length
    const auto first_sub_il_it = il.begin();
    if constexpr (detail::is_initializer_list_v<
                      std::decay_t<decltype(*first_sub_il_it)>>) {
      auto sub_il_it = il.begin();
      const size_t sub_il_length = sub_il_it->size();
      for (++sub_il_it; sub_il_it != il.end(); ++sub_il_it) {
        TA_ASSERT(sub_il_it->size() == sub_il_length);
      }
    }

    return tiled_range_from_il(*first_sub_il_it, std::move(shape));
  }
}

//------------------------------------------------------------------------------
// flatten_il free function
//------------------------------------------------------------------------------

/** @brief Flattens the contents of a (possibly nested) initializer_list into
 *         the provided buffer.
 *
 *  This function is used to flatten a (possibly nested) `std::initializer_list`
 *  into a buffer provided by the user. The flattening occurs by iterating over
 *  the layers of the `std::initializer_list` in a depth-first manner. As the
 *  initializer_list is flattened the data is copied into the container
 *  associated with @p out_itr. It is assumed that the container associated with
 *  @p out_itr is already allocated or that @p out_itr will internally allocate
 *  the memory on-the-fly (*e.g.* `std::back_insert_iterator`).
 *
 *  This function works with empty `std::initializer_list` instances (you will
 *  get back @p out_itr unchanged and the corresponding container is unchanged)
 *  as well as single tensor elements (*i.e.*, initializing a scalar); in the
 *  latter case the buffer corresponding to @p out_itr must contain room for at
 *  least one element as the element will be copied to the buffer.
 *
 *  @tparam T Expected to be the type of a tensor element (*i.e.* float, double,
 *            *etc.*) or a (possibly nested) `std::initializer_list` of tensor
 *            elements.
 *  @tparam OutputItr The type of an iterator which can be used to fill a
 *                    container. It must satisfy the concept of Output Iterator.
 *
 *  @param[in] il The `std::initializer_list` we are flattening.
 *  @param[in] out_itr An iterator pointing to the first element where data
 *                     should be copied to. Memory in the destination container
 *                     is assumed to be pre-allocated otherwise @
 *
 *  @return @p out_itr pointing to just past the last element inserted by this
 *          function.
 *
 *  @throw TiledArray::Exception If the provided `std::initializer_list` is not
 *                               rectangular (*e.g.*, attempting to initialize
 *                               a matrix with the value `{{1, 2}, {3, 4, 5}}`).
 *                               If an exception is thrown @p il and @p out_itr
 *                               are in their original state (strong throw
 *                               guarantee).
 */
template <typename T, typename OutputItr>
auto flatten_il(T&& il, OutputItr out_itr) {
  constexpr auto ranks_left = initializer_list_rank_v<std::decay_t<T>>;

  // We were given a scalar, just copy its value
  // (input of std::initializer_list ends recursion on ranks_left == 1)
  if constexpr (ranks_left == 0) {
    *out_itr = il;
    ++out_itr;
  }
  // We were given a vector or we have recursed to the most nested
  // initializer_list, either way copy the contents to the buffer
  else if constexpr (ranks_left == 1) {
    out_itr = std::copy(il.begin(), il.end(), out_itr);
  }
  // The initializer list is at least a matrix, so recurse over sub-lists
  else {
    const auto length = il.begin()->size();
    for (auto&& x : il) {
      TA_ASSERT(x.size() == length);  // sub-lists must be the same size
      out_itr = flatten_il(x, out_itr);
    }
  }
  return out_itr;
}

//------------------------------------------------------------------------------
// get_elem_from_il free function
//------------------------------------------------------------------------------

/** @brief Retrieves the specified element from an initializer_list
 *
 *  Given an initializer_list with @f$N@f$ nestings, @p il, and an @f$N@f$
 *  element index, @p idx, this function will return the element which is offset
 *  `idx[i]` along the @f$i@f$-th mode of @p il.
 *
 * @tparam T The type of the container holding the index. Assumed to be a random
 *           access container whose elements are of an integral type.
 * @tparam U Assumed to be a scalar type (*e.g.* float, double, *etc.*) or a
 *           (possibly nested) `std::initializer_list` of scalar types.
 *
 * @param[in] idx The desired element's offsets along each mode.
 * @param[in] il The initializer list we are retrieving the value from.
 * @param[in] depth Used internally to keep track of how many levels of
 *                  recursion have occurred. Defaults to 0 and should not be
 *                  modified.
 * @return The requested element.
 *
 * @throws TiledArray::Exception if the number of elements in @p idx does not
 *                               equal the nesting of @p il. Strong throw
 *                               guarantee.
 * @throws TiledArray::Exception if the offset along a mode is greater than the
 *                               length of the mode. Strong throw guarantee.
 */
template <typename T, typename U>
auto get_elem_from_il(T idx, U&& il, std::size_t depth = 0) {
  constexpr auto nestings_left = initializer_list_rank_v<std::decay_t<U>>;
  TA_ASSERT(idx.size() == nestings_left + depth);
  if constexpr (nestings_left == 0) {  // Handle scalars
    return il;
  } else {
    // Make sure the current nesting is long enough
    TA_ASSERT(il.size() > static_cast<std::size_t>(idx[depth]));
    auto itr = il.begin() + idx[depth];
    if constexpr (nestings_left == 1) {
      return *itr;
    } else {
      return get_elem_from_il(std::forward<T>(idx), *itr, depth + 1);
    }
  }
}

//------------------------------------------------------------------------------
// array_from_il free function
//------------------------------------------------------------------------------

/** @brief Converts an `std::initializer_list` into a tiled array.
 *
 *  This function encapsulates the process of turning an `std::initializer_list`
 *  into a TiledArray array. The resulting tensor will have a tiling consistent
 *  with the provided TiledRange, @p trange.
 *
 *  @note This function will raise a static assertion if you try to construct a
 *        rank 0 tensor (*i.e.*, you pass in a single element and not an
 *        `std::initializer_list`).
 *
 * @tparam ArrayType The type of the array we are creating. Expected to be
 *                   have an API akin to that of DistArray
 * @tparam T The type of the provided `std::initializer_list`.
 *
 * @param[in] world The context in which the resulting tensor will live.
 * @param[in] trange The tiling for the resulting tensor.
 * @param[in] il The initializer_list containing the initial state of the
 *               tensor. @p il is assumed to be non-empty and in row-major form.
 *               The nesting of @p il will be used to determine the rank of the
 *               resulting tensor.
 *
 * @return A newly created instance of type @p ArrayType whose state is derived
 *         from @p il and exists in the @p world context.
 *
 * @throw TiledArray::Exception if @p il contains no elements. If an exception
 *                              is raised @p world, @p trange, and @p il are
 *                              unchanged (strong throw guarantee).
 * @throw TiledArray::Exception If the provided `std::initializer_list` is not
 *                              rectangular (*e.g.*, attempting to initialize
 *                              a matrix with the value `{{1, 2}, {3, 4, 5}}`).
 *                              If an exception is raised @p world, @p trange,
 *                              and @p il are unchanged.
 */
template <typename ArrayType, typename T>
auto array_from_il(World& world, const TiledRange& trange, T&& il) {
  using tile_type = typename ArrayType::value_type;

  static_assert(initializer_list_rank_v<std::decay_t<T>> > 0,
                "value initializing rank 0 tensors is not supported");

  ArrayType rv(world, trange);

  for (auto itr = rv.begin(); itr != rv.end(); ++itr) {
    auto range = rv.trange().make_tile_range(itr.index());
    tile_type tile(range);
    for (auto idx : range) {
      tile(idx) = get_elem_from_il(idx, il);
    }
    *itr = tile;
  }
  return rv;
}

/** @brief Converts an `std::initializer_list` into a single tile array.
 *
 *  This function encapsulates the process of turning an `std::initializer_list`
 *  into a TiledArray array. The resulting tensor will consistent of a single
 *  tile which holds all of the values.
 *
 *  @note This function will raise a static assertion if you try to construct a
 *        rank 0 tensor (*i.e.*, you pass in a single element and not an
 *        `std::initializer_list`).
 *
 * @tparam ArrayType The type of the array we are creating. Expected to be
 *                   have an API akin to that of DistArray
 * @tparam T The type of the provided `std::initializer_list`.
 *
 * @param[in] world The context in which the resulting tensor will live.
 * @param[in] il The initializer_list containing the initial state of the
 *               tensor. @p il is assumed to be non-empty and in row-major form.
 *               The nesting of @p il will be used to determine the rank of the
 *               resulting tensor.
 *
 * @return A newly created instance of type @p ArrayType whose state is derived
 *         from @p il and exists in the @p world context.
 *
 * @throw TiledArray::Exception if @p il contains no elements. If an exception
 *                              is raised @p world and @p il are unchanged
 *                              (strong throw guarantee).
 * @throw TiledArray::Exception If the provided `std::initializer_list` is not
 *                              rectangular (*e.g.*, attempting to initialize
 *                              a matrix with the value `{{1, 2}, {3, 4, 5}}`).
 *                              If an exception is raised @p world and @p il are
 *                              unchanged.
 */
template <typename ArrayType, typename T>
auto array_from_il(World& world, T&& il) {
  auto trange = tiled_range_from_il(il);
  return array_from_il<ArrayType, T>(world, std::move(trange),
                                     std::forward<T>(il));
}

namespace detail {

// Typedef of an initializer list for a vector
template <typename T>
using vector_il = std::initializer_list<T>;

// Typedef of an initializer list for a matrix
template <typename T>
using matrix_il = std::initializer_list<vector_il<T>>;

// Typedef of an il for a rank 3 tensor
template <typename T>
using tensor3_il = std::initializer_list<matrix_il<T>>;

// Typedef of an il for a rank 4 tensor
template <typename T>
using tensor4_il = std::initializer_list<tensor3_il<T>>;

// Typedef of an il for a rank 5 tensor
template <typename T>
using tensor5_il = std::initializer_list<tensor4_il<T>>;

// Typedef of an il for a rank 6 tensor
template <typename T>
using tensor6_il = std::initializer_list<tensor5_il<T>>;

}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_INITIALIZER_LIST_UTILS_H__INCLUDED
