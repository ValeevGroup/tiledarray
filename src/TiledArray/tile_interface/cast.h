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
 *  cast.h
 *  Jan 10, 2016
 *
 */

#ifndef TILEDARRAY_TILE_INTERFACE_CAST_H__INCLUDED
#define TILEDARRAY_TILE_INTERFACE_CAST_H__INCLUDED

#include "../meta.h"
#include "../type_traits.h"

namespace TiledArray {

template <typename Result, typename Arg, typename Enabler = void>
class Cast;

namespace tile_interface {

template <typename Result, typename Arg, typename Enabler>
class Cast;

/// Internal cast (aka conversion) implementation

/// This class is used to define internal tile cast operations. Users may
/// specialize the `TiledArray::Cast` class.
/// \tparam Result The output tile type
/// \tparam Arg The input tile type
/// \tparam Enabler Enabler type used to select (partial) specializations
/// \note the base implementation is invoked when Arg is a lazy tile (see
/// TiledArray::is_lazy_tile)
///       that evaluates to Result.
template <typename Result, typename Arg>
class Cast<Result, Arg,
           std::enable_if_t<detail::has_conversion_operator_v<
                                Arg, madness::Future<Result>> ||
                            detail::is_convertible_v<Arg, Result>>> {
 public:
  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type
  static const constexpr bool is_nonblocking =
      detail::has_conversion_operator_v<argument_type,
                                        madness::Future<result_type>>;

  // basic type sanity checks
  static_assert(std::is_same<result_type, std::decay_t<result_type>>::value,
                "Cast<Result,Arg>: Result must be a non-const non-ref type");
  static_assert(std::is_same<argument_type, std::decay_t<argument_type>>::value,
                "Cast<Result,Arg>: Arg must be a non-const non-ref type");
  static_assert(detail::has_conversion_operator_v<
                    argument_type, madness::Future<result_type>> ||
                    detail::is_convertible_v<argument_type, result_type>,
                "Cast<Result,Arg> does not know how to construct Result or "
                "Future<Result> from Arg");

 private:
  template <typename Result_, typename Arg_,
            typename = std::enable_if_t<
                detail::is_convertible<std::decay_t<Arg_>, Result_>::value>>
  static auto invoker(Arg_&& arg) {
    auto exec = [](Arg_&& arg) {
      return static_cast<Result_>(std::forward<Arg_>(arg));
    };
    return TiledArray::meta::invoke(exec, arg);
  }
  template <typename Result_, typename Arg_>
  static auto invoker(
      Arg_&& arg,
      std::enable_if_t<
          !madness::is_future<Result_>::value &&
          !detail::is_convertible<std::decay_t<Arg_>, Result_>::value &&
          detail::has_conversion_operator_v<
              std::decay_t<Arg_>, madness::Future<Result_>>>* = nullptr) {
    auto exec = [](Arg_&& arg) {
      return static_cast<madness::Future<Result_>>(std::forward<Arg_>(arg));
    };
    return TiledArray::meta::invoke(exec, std::forward<Arg_>(arg));
  }

 public:
  /// this converts an Arg object to a Result object
  /// \note get the argument by universal ref as a step towards moving
  /// conversions
  template <typename Arg_,
            typename = std::enable_if_t<std::is_same<
                argument_type, madness::remove_fcvr_t<Arg_>>::value>>
  auto operator()(Arg_&& arg) const {
    return this->invoker<result_type>(arg);
  }

};  // class Cast

/// Internal cast implementation

/// This class is used to define internal tile cast operations. This
/// specialization handles casting of lazy tiles to a type other than the
/// evaluation type.
/// \tparam Result The output tile type
/// \tparam Arg The input tile type
/// \note this specialization is invoked when Arg is a lazy tile (see
/// TiledArray::is_lazy_tile)
///       that evaluates to something other than Result.
template <typename Result, typename Arg>
class Cast<Result, Arg,
           typename std::enable_if<
               is_lazy_tile<Arg>::value &&
               !std::is_same<Result, typename TiledArray::eval_trait<
                                         Arg>::type>::value>::type> {
 private:
  typedef typename TiledArray::eval_trait<Arg>::type
      eval_type;  ///< Lazy tile evaluation type
  typedef TiledArray::Cast<eval_type, Arg>
      arg_to_eval_caster_type;  ///< this converts Arg to eval_type
  typedef TiledArray::Cast<Result, eval_type>
      eval_to_result_caster_type;  ///< this converts eval_type to Result
 public:
  typedef Result result_type;  ///< Result tile type
  typedef Arg argument_type;   ///< Argument tile type

  /// Tile cast operation

  /// Cast `arg` to a `result_type` tile. This potentially involves 2 steps:
  /// - from `argument_type` to `eval_t<argument_type>`
  /// - from `eval_t<argument_type>` to `result_tile`
  /// These casts are nonblocking, if needed.
  /// \param arg The tile to be cast
  /// \return A cast copy of `arg`
  /// \note get the argument by universal ref as a step towards moving
  /// conversions
  template <typename Arg_, typename = std::enable_if_t<std::is_same<
                               argument_type, std::decay_t<Arg_>>::value>>
  auto operator()(Arg_&& arg) const {
    arg_to_eval_caster_type cast_to_eval;
    eval_to_result_caster_type cast_to_result;
    return meta::invoke(cast_to_result, meta::invoke(cast_to_eval, arg));
  }

};  // class Cast

}  // namespace tile_interface

/// Tile cast operation

/// This class is used to define tile cast operations. Users may specialize
/// this class for arbitrary tile type conversion operations.
/// \tparam Result The output tile type
/// \tparam Arg The input tile type
template <typename Result, typename Arg, typename Enabler>
class Cast : public TiledArray::tile_interface::Cast<Result, Arg, Enabler> {};

/// Invokes TiledArray::Cast to cast/convert the argument to type Result.
/// The operation may be nonblocking, if needed. The cast may involve zero, one,
/// or more conversions, depending on the implementation of Cast<>, and the
/// properties of types Arg and Result.
template <typename Arg, typename Result = typename TiledArray::eval_trait<
                            madness::remove_fcvr_t<Arg>>::type>
auto invoke_cast(Arg&& arg) {
  Cast<Result, std::decay_t<Arg>> cast;
  return TiledArray::meta::invoke(cast, std::forward<Arg>(arg));
}

}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_INTERFACE_CAST_H__INCLUDED
