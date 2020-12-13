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
 *  permute.h
 *  Jan 10, 2016
 *
 */

#ifndef TILEDARRAY_TILE_INTERFACE_PERMUTE_H__INCLUDED
#define TILEDARRAY_TILE_INTERFACE_PERMUTE_H__INCLUDED

#include "../tensor/type_traits.h"
#include "../tile_interface/cast.h"
#include "../type_traits.h"

namespace TiledArray {

/// Create a permuted copy of \c arg

/// \tparam Arg The tile argument type
/// \tparam Perm A permutation type
/// \param arg The tile argument to be permuted
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ arg</tt>
template <
    typename Arg, typename Perm,
    typename = std::enable_if_t<detail::is_permutation_v<Perm> &&
                                detail::has_member_function_permute_anyreturn_v<
                                    const Arg, const Perm&>>>
inline auto permute(const Arg& arg, const Perm& perm) {
  return arg.permute(perm);
}

template <typename>
struct permute_trait;

namespace tile_interface {

using TiledArray::permute;

template <typename T>
using result_of_permute_t = typename std::decay<decltype(
    permute(std::declval<T>(), std::declval<Permutation>()))>::type;

template <typename Tile, typename Enabler = void>
struct permute_trait {
  typedef Tile type;
};

template <typename Arg>
struct permute_trait<Arg, typename std::enable_if<TiledArray::detail::is_type<
                              result_of_permute_t<Arg>>::value>::type> {
  typedef result_of_permute_t<Arg> type;
};

template <typename Result, typename Arg, typename Enabler = void>
class Permute {
 public:
  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type

  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_type operator()(const argument_type& arg, const Perm& perm) const {
    using TiledArray::permute;
    if constexpr (detail::is_bipartite_permutable_v<argument_type>) {
      return permute(arg, perm);
    } else {
      TA_ASSERT(inner_size(perm));
      return permute(arg, outer(perm));
    }
  }
};

template <typename Result, typename Arg>
class Permute<Result, Arg,
              typename std::enable_if<
                  !std::is_same<Result, result_of_permute_t<Arg>>::value>::type>
    : public TiledArray::Cast<Result, result_of_permute_t<Arg>> {
 private:
  typedef TiledArray::Cast<Result, result_of_permute_t<Arg>> Cast_;

 public:
  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type

  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_type operator()(const argument_type& arg, const Perm& perm) const {
    using TiledArray::permute;
    if constexpr (detail::is_bipartite_permutable_v<argument_type>) {
      return Cast_::operator()(permute(arg, perm));
    } else {
      TA_ASSERT(inner_size(perm));
      return Cast_::operator()(permute(arg, outer(perm)));
    }
  }
};

}  // namespace tile_interface

/// Permute trait

/// This class defines the default return type for a permutation operation.
/// The default return type is defined by the `permute()` function, if it
/// exists, otherwise the return type is assumed to be the same as `Arg` type.
/// \tparam Arg The argument tile type
template <typename Arg>
struct permute_trait : public TiledArray::tile_interface::permute_trait<Arg> {};

/// Permute a tile

/// This operation creates a permuted copy of a tile.
/// \tparam Result The result tile type
/// \tparam Argument The argument tile type
template <typename Result, typename Arg>
class Permute : public TiledArray::tile_interface::Permute<Result, Arg> {};

}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_INTERFACE_PERMUTE_H__INCLUDED
