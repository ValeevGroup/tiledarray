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
 *  clone.h
 *  Jan 10, 2016
 *
 */

#ifndef TILEDARRAY_TILE_INTERFACE_CLONE_H__INCLUDED
#define TILEDARRAY_TILE_INTERFACE_CLONE_H__INCLUDED

#include "../tile_interface/cast.h"
#include "../type_traits.h"

namespace TiledArray {

GENERATE_HAS_MEMBER(clone);

/// Create a copy of \c arg

/// \tparam Arg The tile argument type
/// \param arg The tile argument to be permuted
/// \return A (deep) copy of \c arg
template <typename Arg,
          typename = std::enable_if_t<has_member_clone<Arg>::value>>
inline auto clone(const Arg& arg) {
  return arg.clone();
}

namespace tile_interface {

using TiledArray::clone;

template <typename T>
using result_of_clone_t =
    typename std::decay<decltype(clone(std::declval<T>()))>::type;

/// Internal clone trait

/// This trait class is used to determine the default output type for tile
/// clone operations. This version of `clone_trait` is used for tile types
/// where a `clone` is NOT function defined.
/// \tparam Arg The argument type to be cloned
template <typename Arg, typename Enabler = void>
struct clone_trait {
  typedef Arg type;
};

/// Internal clone trait

/// This trait class is used to determine the default output type for tile
/// clone operations. This version of `clone_trait` is used for tile types
/// where a `clone` is function defined.
/// \tparam Arg The argument type to be cloned
template <typename Arg>
struct clone_trait<Arg, typename std::enable_if<TiledArray::detail::is_type<
                            result_of_clone_t<Arg> >::value>::type> {
  typedef result_of_clone_t<Arg> type;
};

// Internal tile clone operation

/// Clone a tile using the clone function. Here we rely on ADL to select the
/// correct clone function.
/// \tparam Result The result tile type
/// \tparam Argument The argument tile type
template <typename Result, typename Arg, typename Enabler = void>
class Clone {
 public:
  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type

  result_type operator()(const argument_type& arg) const {
    using TiledArray::clone;
    return clone(arg);
  }
};

// Internal tile clone and cast operation

/// Clone is implemented using a cast operation instead of the `clone`
/// function.
/// \tparam Result The result tile type
/// \tparam Argument The argument tile type
template <typename Result, typename Arg>
class Clone<Result, Arg,
            typename std::enable_if<!std::is_same<
                Result, typename clone_trait<Arg>::type>::value>::type>
    : public TiledArray::Cast<Result, Arg> {};

}  // namespace tile_interface

/// Clone trait

/// This class defines the default return type for a tile clone operation.
/// The default return type is defined by the return type of the `clone()`
/// function, if it exists. Otherwise it is assumed to be `Arg` type.
/// Users may override this trait by providing a (partial) specialization for
/// this class.
/// \tparam Arg The argument tile type
template <typename Arg>
struct clone_trait : public TiledArray::tile_interface::clone_trait<Arg> {};

/// Create a deep copy of a tile

/// This operation creates a deep copy of a tile. The copy operation may
/// optionally perform a clone or cast operation. If the `Result` and `Arg`
/// types are the same, the argument tile is copied using the `clone()`
/// operation is performed, otherwise the argument tile is cast to the
/// `Result` type using the `TiledArray::Cast<Result, Arg>` functor.
/// \tparam Result The result tile type
/// \tparam Argument The argument tile type
template <typename Result, typename Arg>
class Clone : public TiledArray::tile_interface::Clone<Result, Arg> {};

}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_INTERFACE_CLONE_H__INCLUDED
