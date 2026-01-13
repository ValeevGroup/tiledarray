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
 *  add.h
 *  Jan 11, 2016
 *
 */

#ifndef TILEDARRAY_TILE_INTERFACE_ADD_H__INCLUDED
#define TILEDARRAY_TILE_INTERFACE_ADD_H__INCLUDED

#include "../tile_interface/cast.h"
#include "../tile_interface/permute.h"
#include "../type_traits.h"

namespace TiledArray {

/// Add tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \param left The left-hand argument to be added
/// \param right The right-hand argument to be added
/// \return A tile that is equal to <tt>(left + right)</tt>
template <typename Left, typename Right,
          typename = std::enable_if_t<
              detail::has_member_function_add_anyreturn_v<Left&&, Right&&> ||
              detail::has_member_function_add_anyreturn_v<Right&&, Left&&>>>
inline decltype(auto) add(Left&& left, Right&& right) {
  constexpr auto left_right =
      (detail::has_member_function_add_anyreturn_v<Left&&, Right&&> &&
       !std::is_reference_v<Left>) ||
      (!detail::has_member_function_add_anyreturn_v<Right&&, Left&&>);
  if constexpr (left_right)
    return std::forward<Left>(left).add(std::forward<Right>(right));
  else
    return std::forward<Right>(right).add(std::forward<Left>(left));
}

/// Add and scale tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar A scalar type
/// \param left The left-hand argument to be added
/// \param right The right-hand argument to be added
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(left + right) * factor</tt>
template <
    typename Left, typename Right, typename Scalar,
    typename = std::enable_if_t<detail::is_numeric_v<Scalar> &&
                                (detail::has_member_function_add_anyreturn_v<
                                     Left&&, Right&&, const Scalar> ||
                                 detail::has_member_function_add_anyreturn_v<
                                     Right&&, Left&&, const Scalar>)>>
inline decltype(auto) add(Left&& left, Right&& right, const Scalar factor) {
  constexpr auto left_right =
      (detail::has_member_function_add_anyreturn_v<Left&&, Right&&,
                                                   const Scalar> &&
       !std::is_reference_v<Left>) ||
      (!detail::has_member_function_add_anyreturn_v<Right&&, Left&&,
                                                    const Scalar>);
  if constexpr (left_right)
    return std::forward<Left>(left).add(std::forward<Right>(right), factor);
  else
    return std::forward<Right>(right).add(std::forward<Left>(left), factor);
}

/// Add and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \param left The left-hand argument to be added
/// \param right The right-hand argument to be added
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (left + right)</tt>
template <
    typename Left, typename Right, typename Perm,
    typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm> &&
                                (detail::has_member_function_add_anyreturn_v<
                                     Left&&, Right&&, const Perm&> ||
                                 detail::has_member_function_add_anyreturn_v<
                                     Right&&, Left&&, const Perm&>)>>
inline decltype(auto) add(Left&& left, Right&& right, const Perm& perm) {
  constexpr auto left_right =
      (detail::has_member_function_add_anyreturn_v<Left&&, Right&&,
                                                   const Perm&> &&
       !std::is_reference_v<Left>) ||
      (!detail::has_member_function_add_anyreturn_v<Right&&, Left&&,
                                                    const Perm&>);
  if constexpr (left_right)
    return std::forward<Left>(left).add(std::forward<Right>(right), perm);
  else
    return std::forward<Right>(right).add(std::forward<Left>(left), perm);
}

/// Add, scale, and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar A scalar type
/// \param left The left-hand argument to be added
/// \param right The right-hand argument to be added
/// \param factor The scaling factor
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (left + right) * factor</tt>
template <typename Left, typename Right, typename Scalar, typename Perm,
          typename = std::enable_if_t<
              detail::is_numeric_v<Scalar> && detail::is_permutation_v<Perm> &&
              (detail::has_member_function_add_anyreturn_v<
                   Left&&, Right&&, const Scalar, const Perm&> ||
               detail::has_member_function_add_anyreturn_v<
                   Right&&, Left&&, const Scalar, const Perm&>)>>
inline decltype(auto) add(Left&& left, Right&& right, const Scalar factor,
                          const Perm& perm) {
  constexpr auto left_right =
      (detail::has_member_function_add_anyreturn_v<Left&&, Right&&,
                                                   const Scalar, const Perm&> &&
       !std::is_reference_v<Left>) ||
      (!detail::has_member_function_add_anyreturn_v<Right&&, Left&&,
                                                    const Scalar, const Perm&>);
  if constexpr (left_right)
    return std::forward<Left>(left).add(std::forward<Right>(right), factor,
                                        perm);
  else
    return std::forward<Right>(right).add(std::forward<Left>(left), factor,
                                          perm);
}

/// Add to the result tile

/// \tparam Result The result tile type
/// \tparam Arg The argument tile type
/// \param result The result tile
/// \param arg The argument to be added to the result
/// \return A tile that is equal to <tt>result[i] += arg[i]</tt>
template <
    typename Result, typename Arg,
    typename = std::enable_if_t<
        detail::has_member_function_add_to_anyreturn_v<Result&&, const Arg&>>>
inline decltype(auto) add_to(Result&& result, const Arg& arg) {
  return std::forward<Result>(result).add_to(arg);
}

/// Add and scale to the result tile

/// \tparam Result The result tile type
/// \tparam Arg The argument tile type
/// \tparam Scalar A scalar type
/// \param result The result tile
/// \param arg The argument to be added to \c result
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(result[i] += arg[i]) *= factor</tt>
template <typename Result, typename Arg, typename Scalar,
          typename std::enable_if<
              detail::is_numeric_v<Scalar> &&
              detail::has_member_function_add_to_anyreturn_v<
                  Result&&, const Arg&, const Scalar>>::type* = nullptr>
inline decltype(auto) add_to(Result&& result, const Arg& arg,
                             const Scalar factor) {
  return std::forward<Result>(result).add_to(arg, factor);
}

namespace tile_interface {

using TiledArray::add;
using TiledArray::add_to;

template <typename... T>
using result_of_add_t = decltype(add(std::declval<T>()...));

template <typename... T>
using result_of_add_to_t = decltype(add_to(std::declval<T>()...));

//    template <typename Left, typename Right, typename Enabler = void>
//    struct add_trait {
//      typedef Left type;
//    };
//
//    template <typename Left, typename Right>
//    struct add_trait<Left, Right,
//        typename std::enable_if<
//            TiledArray::detail::is_type<result_of_add_t<Left, Right> >::value
//        >::type>
//    {
//      typedef result_of_add_t<Left, Right> type;
//    };
//
//    template <typename Left, typename Right, typename Enabler = void>
//    struct scal_add_trait :
//        public scal_trait<typename add_trait<Left, Right>::type, Scalar>
//    { };
//
//    template <typename Left, typename Right, typename Scalar>
//    struct scal_add_trait<Left, Right,
//        typename std::enable_if<
//            TiledArray::detail::is_type<result_of_add_t<Left, Right, Scalar>
//            >::value
//        >::type>
//    {
//      typedef result_of_add_t<Left, Right, Scalar> type;
//    };

template <typename Result, typename Left, typename Right,
          typename Enabler = void>
class Add {
 public:
  typedef Result result_type;  ///< Result tile type
  typedef Left left_type;      ///< Left-hand argument tile type
  typedef Right right_type;    ///< Right-hand argument tile type

  result_type operator()(const left_type& left, const right_type& right) const {
    using TiledArray::add;
    return add(left, right);
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(const left_type& left, const right_type& right,
                         const Perm& perm) const {
    using TiledArray::add;
    return add(left, right, perm);
  }
};

template <typename Result, typename Left, typename Right>
class Add<Result, Left, Right,
          typename std::enable_if<!(
              std::is_same<Result, result_of_add_t<Left, Right>>::value &&
              std::is_same<Result,
                           result_of_add_t<Left, Right, BipartitePermutation>>::
                  value)>::type> {
 public:
  typedef Result result_type;  ///< Result tile type
  typedef Left left_type;      ///< Left-hand argument tile type
  typedef Right right_type;    ///< Right-hand argument tile type

  result_type operator()(const left_type& left, const right_type& right) const {
    using TiledArray::add;
    TiledArray::Cast<Result, result_of_add_t<Left, Right>> cast;
    return cast(add(left, right));
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(const left_type& left, const right_type& right,
                         const Perm& perm) const {
    using TiledArray::add;
    TiledArray::Cast<Result, result_of_add_t<Left, Right, Perm>> cast;
    return cast(add(left, right, perm));
  }
};

template <typename Result, typename Left, typename Right, typename Scalar,
          typename Enabler = void>
class ScalAdd {
 public:
  static_assert(TiledArray::detail::is_numeric_v<Scalar>,
                "Cannot scale tiles by a non-scalar type");

  typedef Result result_type;  ///< Result tile type
  typedef Left left_type;      ///< Left-hand argument tile type
  typedef Right right_type;    ///< Right-hand argument tile type
  typedef Scalar scalar_type;  ///< The scaling factor type

  result_type operator()(const left_type& left, const right_type& right,
                         const scalar_type factor) const {
    using TiledArray::add;
    return add(left, right, factor);
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(const left_type& left, const right_type& right,
                         const scalar_type factor, const Perm& perm) const {
    using TiledArray::add;
    return add(left, right, factor, perm);
  }
};

template <typename Result, typename Left, typename Right, typename Scalar>
class ScalAdd<
    Result, Left, Right, Scalar,
    typename std::enable_if<!(
        std::is_same<Result, result_of_add_t<Left, Right, Scalar>>::value &&
        std::is_same<Result, result_of_add_t<Left, Right, Scalar,
                                             BipartitePermutation>>::value)>::
        type> {
 public:
  static_assert(TiledArray::detail::is_numeric_v<Scalar>,
                "Cannot scale tiles by a non-scalar type");

  typedef Result result_type;  ///< Result tile type
  typedef Left left_type;      ///< Left-hand argument tile type
  typedef Right right_type;    ///< Right-hand argument tile type
  typedef Scalar scalar_type;  ///< The scaling factor type

  result_type operator()(const left_type& left, const right_type& right,
                         const scalar_type factor) const {
    using TiledArray::add;
    TiledArray::Cast<Result, result_of_add_t<Left, Right, Scalar>> cast;
    return cast(add(left, right, factor));
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(const left_type& left, const right_type& right,
                         const scalar_type factor, const Perm& perm) const {
    using TiledArray::add;
    TiledArray::Cast<Result, result_of_add_t<Left, Right, Scalar, Perm>> cast;
    return cast(add(left, right, factor, perm));
  }
};

template <typename Result, typename Left, typename Right,
          typename Enabler = void>
class AddTo {
 public:
  typedef Result result_type;  ///< Result tile type
  typedef Left left_type;      ///< Left-hand argument tile type
  typedef Right right_type;    ///< Right-hand argument tile type

  result_type& operator()(left_type& left, const right_type& right) const {
    using TiledArray::add_to;
    return add_to(left, right);
  }
  result_type&& operator()(left_type&& left, const right_type& right) const {
    using TiledArray::add_to;
    return add_to(std::move(left), right);
  }
};

template <typename Result, typename Left, typename Right>
class AddTo<Result, Left, Right,
            typename std::enable_if<!std::is_same<
                Result, result_of_add_to_t<Left, Right>>::value>::type> {
 public:
  typedef Result result_type;  ///< Result tile type
  typedef Left left_type;      ///< Left-hand argument tile type
  typedef Right right_type;    ///< Right-hand argument tile type

  result_type operator()(left_type& left, const right_type& right) const {
    using TiledArray::add_to;
    TiledArray::Cast<Result, result_of_add_to_t<Left, Right>> cast;
    add_to(left, right);
    return cast(left);
  }
};

template <typename Result, typename Left, typename Right, typename Scalar,
          typename Enabler = void>
class ScalAddTo {
 public:
  static_assert(TiledArray::detail::is_numeric_v<Scalar>,
                "Cannot scale tiles by a non-scalar type");

  typedef Result result_type;  ///< Result tile type
  typedef Left left_type;      ///< Left-hand argument tile type
  typedef Right right_type;    ///< Right-hand argument tile type
  typedef Scalar scalar_type;  ///< The scaling factor type

  result_type& operator()(left_type& left, const right_type& right) const {
    using TiledArray::add_to;
    return add_to(left, right);
  }
};

template <typename Result, typename Left, typename Right, typename Scalar>
class ScalAddTo<
    Result, Left, Right, Scalar,
    typename std::enable_if<!std::is_same<
        Result, result_of_add_to_t<Left, Right, Scalar>>::value>::type> {
 public:
  static_assert(TiledArray::detail::is_numeric_v<Scalar>,
                "Cannot scale tiles by a non-scalar type");

  typedef Result result_type;  ///< Result tile type
  typedef Left left_type;      ///< Left-hand argument tile type
  typedef Right right_type;    ///< Right-hand argument tile type
  typedef Scalar scalar_type;  ///< The scaling factor type

  result_type operator()(left_type& left, const right_type& right,
                         const scalar_type factor) const {
    using TiledArray::add_to;
    TiledArray::Cast<Result, result_of_add_to_t<Left, Right, Scalar>> cast;
    add_to(left, right);
    return cast(left);
  }
};

}  // namespace tile_interface

/// Add tile operation

/// This operation adds two tiles of type `Left` and `Right` to create a tile
/// of type `Result`.
/// \tparam Result The result tile type
/// \tparam Left The left-hand argument tile type
/// \tparam Right The right-hand argument tile type
template <typename Result, typename Left, typename Right>
class Add : public TiledArray::tile_interface::Add<Result, Left, Right> {};

/// Add tile operation

/// This operation adds two tiles of type `Left` and `Right` to create a tile
/// of type `Result`.
/// \tparam Result The result tile type
/// \tparam Left The left-hand argument tile type
/// \tparam Right The right-hand argument tile type
/// \tparam Scalar The scaling factor type
template <typename Result, typename Left, typename Right, typename Scalar>
class ScalAdd
    : public TiledArray::tile_interface::ScalAdd<Result, Left, Right, Scalar> {
};

/// Add-to tile operation

/// This operation adds a `Right` tile type to a `Left` tile of type.
/// \tparam Result The result tile type
/// \tparam Left The left-hand argument tile type
/// \tparam Right The right-hand argument tile type
template <typename Result, typename Left, typename Right>
class AddTo : public TiledArray::tile_interface::AddTo<Result, Left, Right> {};

/// Add-to and scale tile operation

/// This operation adds a `Right` tile type to a `Left` tile of type.
/// \tparam Result The result tile type
/// \tparam Left The left-hand argument tile type
/// \tparam Right The right-hand argument tile type
/// \tparam Scalar The scaling factor type
template <typename Result, typename Left, typename Right, typename Scalar>
class ScalAddTo : public TiledArray::tile_interface::ScalAddTo<Result, Left,
                                                               Right, Scalar> {
};

}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_INTERFACE_ADD_H__INCLUDED
