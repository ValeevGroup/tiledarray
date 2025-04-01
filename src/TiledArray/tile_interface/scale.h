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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  scale.h
 *  Jan 11, 2016
 *
 */

#ifndef TILEDARRAY_SRC_TILEDARRAY_TILE_INTERFACE_SCALE_H__INCLUDED
#define TILEDARRAY_SRC_TILEDARRAY_TILE_INTERFACE_SCALE_H__INCLUDED

#include "../permutation.h"
#include "../type_traits.h"
#include "cast.h"

namespace TiledArray {

/// Scalar the tile argument

/// \tparam Arg A tile type
/// \tparam Scalar A numeric type (i.e. TiledArray::detail::is_numeric_v<Scalar>
/// is true) \param arg The left-hand argument to be scaled \param factor The
/// scaling factor \return A tile that is equal to <tt>arg * factor</tt>
template <
    typename Arg, typename Scalar,
    std::enable_if_t<
        TiledArray::detail::is_numeric_v<Scalar> &&
        !TiledArray::detail::is_array_v<Arg> &&
        std::is_void_v<std::void_t<decltype(std::declval<const Arg&>().scale(
            std::declval<Scalar>()))>>>* = nullptr>
inline auto scale(const Arg& arg, const Scalar factor) {
  return arg.scale(factor);
}

/// Scale and permute tile argument

/// \tparam Arg The tile argument type
/// \tparam Scalar A scalar type
/// \param arg The left-hand argument to be scaled
/// \param factor The scaling factor
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (arg * factor)</tt>
template <
    typename Arg, typename Scalar, typename Perm,
    std::enable_if_t<
        TiledArray::detail::is_numeric_v<Scalar> &&
        TiledArray::detail::is_permutation_v<Perm> &&
        std::is_void_v<std::void_t<decltype(std::declval<const Arg&>().scale(
            std::declval<Scalar>(), std::declval<const Perm&>()))>>>* = nullptr>
inline auto scale(const Arg& arg, const Scalar factor, const Perm& perm) {
  return arg.scale(factor, perm);
}

/// Scale to the result tile

/// \tparam Result The result tile type
/// \tparam Scalar A scalar type
/// \param result The result tile to be scaled
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>result *= factor</tt>
template <typename Result, typename Scalar,
          std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar> &&
                           detail::has_member_function_scale_to_anyreturn_v<
                               Result&&, Scalar>>* = nullptr>
inline decltype(auto) scale_to(Result&& result, const Scalar factor) {
  static_assert(!std::is_const_v<std::remove_reference_t<Result>>,
                "TA::scale_to(result,factor): result cannot be const");
  return std::forward<Result>(result).scale_to(factor);
}

namespace tile_interface {

template <typename... T>
using result_of_scale_t = decltype(scale(std::declval<T>()...));

template <typename... T>
using result_of_scale_to_t = decltype(scale_to(std::declval<T>()...));

template <typename Arg, typename Scalar, typename Enabler = void>
struct scale_trait {
  typedef Arg type;
};

template <typename Arg, typename Scalar>
struct scale_trait<Arg, Scalar,
                   typename std::enable_if<TiledArray::detail::is_type<
                       result_of_scale_t<Arg, Scalar>>::value>::type> {
  typedef result_of_scale_t<Arg> type;
};

template <typename Result, typename Arg, typename Scalar,
          typename Enabler = void>
class Scale {
 public:
  static_assert(TiledArray::detail::is_numeric_v<Scalar>,
                "Cannot scale tiles by a non-scalar type");

  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type
  typedef Scalar scalar_type;  ///< Scaling factor type

  result_type operator()(const argument_type& arg,
                         const scalar_type factor) const {
    using TiledArray::scale;
    return scale(arg, factor);
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(const argument_type& arg, const scalar_type factor,
                         const Perm& perm) const {
    using TiledArray::scale;
    return scale(arg, factor, perm);
  }
};

template <typename Result, typename Arg, typename Scalar>
class Scale<Result, Arg, Scalar,
            typename std::enable_if<!std::is_same<
                Result, result_of_scale_t<Arg, Scalar>>::value>::type> {
 public:
  static_assert(TiledArray::detail::is_numeric_v<Scalar>,
                "Cannot scale tiles by a non-scalar type");

  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type
  typedef Scalar scalar_type;  ///< The scaling factor type

  result_type operator()(const argument_type& arg,
                         const scalar_type factor) const {
    using TiledArray::scale;
    TiledArray::Cast<Result, result_of_scale_t<Arg, Scalar>> cast;
    return cast(scale(arg, factor));
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(const argument_type& arg, const scalar_type factor,
                         const Perm& perm) const {
    using TiledArray::scale;
    TiledArray::Cast<Result, result_of_scale_t<Arg, Scalar, Perm>> cast;
    return cast(scale(arg, factor, perm));
  }
};

template <typename Result, typename Arg, typename Scalar,
          typename Enabler = void>
class ScaleTo {
 public:
  static_assert(TiledArray::detail::is_numeric_v<Scalar>,
                "Cannot scale tiles by a non-scalar type");

  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type
  typedef Scalar scalar_type;  ///< Scaling factor type

  result_type operator()(argument_type& arg, const scalar_type factor) const {
    using TiledArray::scale_to;
    return scale_to(arg, factor);
  }
};

template <typename Result, typename Arg, typename Scalar>
class ScaleTo<Result, Arg, Scalar,
              typename std::enable_if<!std::is_same<
                  Result, result_of_scale_t<Arg, Scalar>>::value>::type> {
 public:
  static_assert(TiledArray::detail::is_numeric_v<Scalar>,
                "Cannot scale tiles by a non-scalar type");

  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type
  typedef Scalar scalar_type;  ///< The scaling factor type

  result_type operator()(const argument_type& arg,
                         const scalar_type factor) const {
    using TiledArray::scale_to;
    TiledArray::Cast<Result, result_of_scale_to_t<Arg, Scalar>> cast;
    return cast(scale_to(arg, factor));
  }
};

}  // namespace tile_interface

/// Scale trait

/// This class defines the default return type for a permutation operation.
/// The default return type is defined by the `permute()` function, if it
/// exists, otherwise the return type is assumed to be the same as `Arg` type.
/// \tparam Arg The argument tile type
template <typename Arg, typename Scalar>
struct scale_trait
    : public TiledArray::tile_interface::scale_trait<Arg, Scalar> {};

/// Scale tile

/// This operation creates a scaled copy of a tile.
/// \tparam Result The result tile type
/// \tparam Argument The argument tile type
/// \tparam Scalar The scaling factor type
template <typename Result, typename Arg, typename Scalar>
class Scale : public TiledArray::tile_interface::Scale<Result, Arg, Scalar> {};

}  // namespace TiledArray

#endif  // TILEDARRAY_SRC_TILEDARRAY_TILE_INTERFACE_SCALE_H__INCLUDED
