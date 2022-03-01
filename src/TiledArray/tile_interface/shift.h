/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2016  Virginia Tech
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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  shift.h
 *  Jan 10, 2016
 *
 */

#ifndef TILEDARRAY_TILE_INTERFACE_SHIFT_H__INCLUDED
#define TILEDARRAY_TILE_INTERFACE_SHIFT_H__INCLUDED

#include "../type_traits.h"
#include "cast.h"

namespace TiledArray {

/// Shift the range of \c arg

/// \tparam Arg The tile argument type
/// \tparam Index An integral range type
/// \param arg The tile argument to be shifted
/// \param range_shift The offset to be applied to the argument range
/// \return A copy of the tile with a new range
template <typename Arg, typename Index,
          typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
inline auto shift(const Arg& arg, const Index& range_shift) {
  return arg.shift(range_shift);
}

/// Shift the range of \c arg

/// \tparam Arg The tile argument type
/// \tparam Index An integral type
/// \param arg The tile argument to be shifted
/// \param range_shift The offset to be applied to the argument range
/// \return A copy of the tile with a new range
template <typename Arg, typename Index,
          typename = std::enable_if_t<std::is_integral_v<Index>>>
inline auto shift(const Arg& arg,
                  const std::initializer_list<Index>& range_shift) {
  return arg.shift(range_shift);
}

/// Shift the range of \c arg in place

/// \tparam Arg The tile argument type
/// \tparam Index An integral range type
/// \param arg The tile argument to be shifted
/// \param range_shift The offset to be applied to the argument range
/// \return A copy of the tile with a new range
template <typename Arg, typename Index,
          typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
inline auto shift_to(Arg& arg, const Index& range_shift) {
  return arg.shift_to(range_shift);
}

/// Shift the range of \c arg in place

/// \tparam Arg The tile argument type
/// \tparam Index An integral type
/// \param arg The tile argument to be shifted
/// \param range_shift The offset to be applied to the argument range
/// \return A copy of the tile with a new range
template <typename Arg, typename Index,
          typename = std::enable_if_t<std::is_integral_v<Index>>>
inline auto shift_to(Arg& arg,
                     const std::initializer_list<Index>& range_shift) {
  return arg.shift_to(range_shift);
}

namespace tile_interface {

using TiledArray::shift;
using TiledArray::shift_to;

template <typename T>
using result_of_shift_t = typename std::decay<decltype(shift(
    std::declval<T>(), std::declval<container::svector<long>>()))>::type;

template <typename T>
using result_of_shift_to_t = typename std::decay<decltype(shift_to(
    std::declval<T>(), std::declval<container::svector<long>>()))>::type;

template <typename Result, typename Arg, typename Enabler = void>
class Shift {
 public:
  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type

  template <typename Index>
  result_type operator()(const argument_type& arg,
                         const Index& range_shift) const {
    return shift(arg, range_shift);
  }
};

template <typename Result, typename Arg>
class Shift<Result, Arg,
            typename std::enable_if<
                !std::is_same<Result, result_of_shift_t<Arg>>::value>::type>
    : public TiledArray::Cast<Result, result_of_shift_t<Arg>> {
 private:
  typedef TiledArray::Cast<Result, result_of_shift_t<Arg>> Cast_;

 public:
  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type

  template <typename Index>
  result_type operator()(const argument_type& arg,
                         const Index& range_shift) const {
    return Cast_::operator()(shift(arg, range_shift));
  }
};

template <typename Result, typename Arg, typename Enabler = void>
class ShiftTo {
 public:
  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type

  template <typename Index>
  result_type operator()(argument_type& arg, const Index& range_shift) const {
    return shift_to(arg, range_shift);
  }
};

template <typename Result, typename Arg>
class ShiftTo<Result, Arg,
              typename std::enable_if<!std::is_same<
                  Result, result_of_shift_to_t<Arg>>::value>::type>
    : public TiledArray::Cast<Result, result_of_shift_to_t<Arg>> {
 private:
  typedef TiledArray::Cast<Result, result_of_shift_to_t<Arg>> Cast_;

 public:
  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type

  template <typename Index>
  result_type operator()(argument_type& arg, const Index& range_shift) const {
    return Cast_::operator()(shift_to(arg, range_shift));
  }
};

template <typename Arg, typename Enabler = void>
struct shift_trait {
  typedef Arg type;
};

template <typename Arg>
struct shift_trait<Arg, typename std::enable_if<TiledArray::detail::is_type<
                            result_of_shift_t<Arg>>::value>::type> {
  typedef result_of_shift_t<Arg> type;
};

}  // namespace tile_interface

/// Shift the range of tile

/// This operation creates a deep copy of a tile and shifts the lower and
/// upper bounds of the range.
/// \tparam Result The result tile type
/// \tparam Argument The argument tile type
template <typename Result, typename Arg>
class Shift : public TiledArray::tile_interface::Shift<Result, Arg> {};

/// Shift the range of tile in place

/// This operation shifts the range of a tile without copying or otherwise
/// modifying the tile data.
/// \tparam Result The result tile type
/// \tparam Argument The argument tile type
template <typename Result, typename Arg>
class ShiftTo : public TiledArray::tile_interface::ShiftTo<Result, Arg> {};

}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_INTERFACE_SHIFT_H__INCLUDED
