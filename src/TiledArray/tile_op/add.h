/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  May 7, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_ADD_H__INCLUDED
#define TILEDARRAY_TILE_OP_ADD_H__INCLUDED

#include <TiledArray/tile_op/tile_interface.h>
#include <TiledArray/tile_op/clone.h>
#include <TiledArray/tile_op/permute.h>
#include <TiledArray/zero_tensor.h>

namespace TiledArray {

  /// Tile addition operation

  /// This addition operation will add the content two tiles, and accepts an
  /// optional permute argument.
  /// \tparam Left The left-hand argument type
  /// \tparam Right The right-hand argument type
  /// \tparam LeftConsumable If \c true , will try to consume the left-hand argument when necessary.
  /// \tparam RightConsumable If \c true , will try to consume the right-hand argument when necessary.
  /// \note Input tiles can be consumed only if their type matches the result type.
  template <typename Left, typename Right, bool LeftConsumable,
      bool RightConsumable>
  class Add {
  public:

    typedef Add<Left, Right, LeftConsumable, RightConsumable> Add_;
    typedef Left left_type; ///< Left-hand argument base type
    typedef Right right_type; ///< Right-hand argument base type
    typedef decltype(add(std::declval<left_type>(), std::declval<right_type>()))
        result_type; ///< result type is uniquely determined by return of freestanding add(Left,Right) function

    /// indicates whether it is *possible* to consume the left tile
    static constexpr bool left_is_consumable =
        LeftConsumable && std::is_same<result_type, left_type>::value;
    /// indicates whether it is *possible* to consume the right tile
    static constexpr bool right_is_consumable =
        RightConsumable && std::is_same<result_type, right_type>::value;

  private:

    // Permuting tile evaluation function
    // These operations cannot consume the argument tile since this operation
    // requires temporary storage space.

    static result_type eval(const left_type& first, const right_type& second,
        const Permutation& perm)
    {
      using TiledArray::add;
      return add(first, second, perm);
    }

    static result_type eval(ZeroTensor, const right_type& second,
        const Permutation& perm)
    {
      Permute<result_type,right_type> permute;
      return permute(second, perm);
    }

    static result_type eval(const left_type& first, ZeroTensor,
        const Permutation& perm)
    {
      Permute<result_type,left_type> permute;
      return permute(first, perm);
    }

    // Non-permuting tile evaluation functions
    // The compiler will select the correct functions based on the consumability
    // of the arguments.

    template <bool LC, bool RC,
        typename std::enable_if<!(LC || RC)>::type* = nullptr>
    static result_type eval(const left_type& first, const right_type& second) {
      using TiledArray::add;
      return add(first, second);
    }

    template <bool LC, bool RC,
        typename std::enable_if<LC>::type* = nullptr>
    static result_type eval(left_type& first, const right_type& second) {
      using TiledArray::add_to;
      return add_to(first, second);
    }

    template <bool LC, bool RC,
        typename std::enable_if<!LC && RC>::type* = nullptr>
    static result_type eval(const left_type& first, right_type& second) {
      using TiledArray::add_to;
      return add_to(second, first);
    }

    template <bool LC, bool RC,
        typename std::enable_if<!RC>::type* = nullptr>
    static result_type eval(const ZeroTensor&, const right_type& second) {
      TiledArray::Clone<result_type,right_type> clone;
      return clone(second);
    }

    template <bool LC, bool RC,
        typename std::enable_if<RC>::type* = nullptr>
    static result_type eval(const ZeroTensor&, right_type& second) {
      return second;
    }

    template <bool LC, bool RC,
        typename std::enable_if<!LC>::type* = nullptr>
    static result_type eval(const left_type& first, const ZeroTensor&) {
      TiledArray::Clone<result_type,left_type> clone;
      return clone(first);
    }

    template <bool LC, bool RC,
        typename std::enable_if<LC>::type* = nullptr>
    static result_type eval(left_type& first, const ZeroTensor&) {
      return first;
    }

  public:

    /// Add-and-permute operator

    /// Compute the sum of two tiles and permute the result. One of the argument
    /// tiles may be replaced with `ZeroTensor` argument, in which case the
    /// argument's element values are assumed to be `0`.
    /// \tparam L The left-hand tile argument type
    /// \tparam R The right-hand tile argument type
    /// \param left The left-hand tile argument
    /// \param right The right-hand tile argument
    /// \param perm The permutation applied to the result tile
    /// \return The permuted and scaled sum of `left` and `right`.
    template <typename L, typename R>
    result_type operator()(L&& left, R&& right, const Permutation& perm) const {
      return eval(std::forward<L>(left), std::forward<R>(right), perm);
    }

    /// Add operator

    /// Compute the sum of two tiles. One of the argument tiles may be replaced
    /// with `ZeroTensor` argument, in which case the argument's element values
    /// are assumed to be `0`.
    /// \tparam L The left-hand tile argument type
    /// \tparam R The right-hand tile argument type
    /// \param left The left-hand tile argument
    /// \param right The right-hand tile argument
    /// \return The scaled sum of `left` and `right`.
    template <typename L, typename R>
    result_type operator()(L&& left, R&& right) const {
      return Add_::template eval<left_is_consumable, right_is_consumable>(
          std::forward<L>(left), std::forward<R>(right));
    }

    /// Add right to left

    /// Add the right tile to the left. The right tile may be replaced with
    /// `ZeroTensor` argument, in which case the argument's element values are
    /// assumed to be `0`.
    /// \tparam R The right-hand tile argument type
    /// \param left The left-hand tile argument
    /// \param right The right-hand tile argument
    /// \return The sum of `left` and `right`.
    template <typename R>
    result_type consume_left(left_type& left, R&& right) const {
      constexpr auto can_consume = is_consumable_tile<left_type>::value && std::is_same<result_type,left_type>::value;
      return Add_::template eval<can_consume,false>(left, std::forward<R>(right));
    }

    /// Add left to right

    /// Add the left tile to the right. The left tile may be replaced with
    /// `ZeroTensor` argument, in which case the argument's element values are
    /// assumed to be `0`.
    /// \tparam L The left-hand tile argument type
    /// \param left The left-hand tile argument
    /// \param right The right-hand tile argument
    /// \return The sum of `left` and `right`.
    template <typename L>
    result_type consume_right(L&& left, right_type& right) const {
      constexpr auto can_consume = is_consumable_tile<right_type>::value && std::is_same<result_type,right_type>::value;
      return Add_::template eval<false,can_consume>(std::forward<L>(left), right);
    }

  }; // class Add

  /// Tile scale-addition operation

  /// This addition operation will add the content two tiles and apply a
  /// permutation to the result tensor. If no permutation is given or the
  /// permutation is null, then the result is not permuted.
  /// \tparam Left The left-hand argument type
  /// \tparam Right The right-hand argument type
  /// \tparam Scalar The scaling factor type
  /// \tparam LeftConsumable A flag that is \c true when the left-hand
  /// argument is consumable.
  /// \tparam RightConsumable A flag that is \c true when the right-hand
  /// argument is consumable.
  template <typename Left, typename Right, typename Scalar, bool LeftConsumable,
      bool RightConsumable>
  class ScalAdd {
  public:

    typedef ScalAdd<Left, Right, Scalar, LeftConsumable, RightConsumable> ScalAdd_;
    typedef Left left_type; ///< Left-hand argument base type
    typedef Right right_type; ///< Right-hand argument base type
    typedef Scalar scalar_type; ///< Scaling factor type
//    typedef Left result_type;
    typedef decltype(add(std::declval<left_type>(), std::declval<right_type>(),
        std::declval<scalar_type>())) result_type;

    static constexpr bool left_is_consumable =
        LeftConsumable && std::is_same<result_type, left_type>::value;
    static constexpr bool right_is_consumable =
        RightConsumable && std::is_same<result_type, right_type>::value;

  private:

    scalar_type factor_;

    // Permuting tile evaluation function
    // These operations cannot consume the argument tile since this operation
    // requires temporary storage space.

    result_type eval(const left_type& first, const right_type& second,
        const Permutation& perm) const
    {
      using TiledArray::add;
      return add(first, second, factor_, perm);
    }

    result_type eval(ZeroTensor, const right_type& second,
        const Permutation& perm) const
    {
      using TiledArray::scale;
      return scale(second, factor_, perm);
    }

    result_type eval(const left_type& first, ZeroTensor,
        const Permutation& perm) const
    {
      using TiledArray::scale;
      return scale(first, factor_, perm);
    }

    // Non-permuting tile evaluation functions
    // The compiler will select the correct functions based on the consumability
    // of the arguments.

    template <bool LC, bool RC,
        typename std::enable_if<!(LC || RC)>::type* = nullptr>
    result_type eval(const left_type& first, const right_type& second) const {
      using TiledArray::add;
      return add(first, second, factor_);
    }

    template <bool LC, bool RC,
        typename std::enable_if<LC>::type* = nullptr>
    result_type eval(left_type& first, const right_type& second) const {
      using TiledArray::add_to;
      return add_to(first, second, factor_);
    }

    template <bool LC, bool RC,
        typename std::enable_if<!LC && RC>::type* = nullptr>
    result_type eval(const left_type& first, right_type& second) const {
      using TiledArray::add_to;
      return add_to(second, first, factor_);
    }

    template <bool LC, bool RC,
        typename std::enable_if<!RC>::type* = nullptr>
    result_type eval(const ZeroTensor&, const right_type& second) const {
      using TiledArray::scale;
      return scale(second, factor_);
    }

    template <bool LC, bool RC,
        typename std::enable_if<RC>::type* = nullptr>
    result_type eval(const ZeroTensor&, right_type& second) const {
      using TiledArray::scale_to;
      return scale_to(second, factor_);
    }

    template <bool LC, bool RC,
        typename std::enable_if<!LC>::type* = nullptr>
    result_type eval(const left_type& first, const ZeroTensor&) const {
      using TiledArray::scale;
      return scale(first, factor_);
    }

    template <bool LC, bool RC,
        typename std::enable_if<LC>::type* = nullptr>
    result_type eval(left_type& first, const ZeroTensor&) const {
      using TiledArray::scale_to;
      return scale_to(first, factor_);
    }

  public:

    // Compiler generated functions
    ScalAdd(const ScalAdd_&) = default;
    ScalAdd(ScalAdd_&&) = default;
    ~ScalAdd() = default;
    ScalAdd_& operator=(const ScalAdd_&) = default;
    ScalAdd_& operator=(ScalAdd_&&) = default;

    /// Constructor

    /// \param factor The scaling factor applied to result tiles
    explicit ScalAdd(const Scalar factor) : factor_(factor) { }

    /// Scale-add-and-permute operator

    /// Compute the scaled sum of two tiles and permute the result. One of the
    /// argument tiles may be replaced with `ZeroTensor` argument, in which
    /// case the argument's element values are assumed to be `0`.
    /// \tparam L The left-hand tile argument type
    /// \tparam R The right-hand tile argument type
    /// \param left The left-hand tile argument
    /// \param right The right-hand tile argument
    /// \param perm The permutation applied to the result tile
    /// \return The permuted and scaled sum of `left` and `right`.
    template <typename L, typename R>
    result_type operator()(L&& left, R&& right, const Permutation& perm) const {
      return eval(std::forward<L>(left), std::forward<R>(right), perm);
    }

    /// Scale-and-add operator

    /// Compute the scaled sum of two tiles. One of the argument tiles may be
    /// replaced with `ZeroTensor` argument, in which case the argument's
    /// element values are assumed to be `0`.
    /// \tparam L The left-hand tile argument type
    /// \tparam R The right-hand tile argument type
    /// \param left The left-hand tile argument
    /// \param right The right-hand tile argument
    /// \return The scaled sum of `left` and `right`.
    template <typename L, typename R>
    result_type operator()(L&& left, R&& right) const {
      return ScalAdd_::template eval<left_is_consumable,
          right_is_consumable>(std::forward<L>(left), std::forward<R>(right));
    }

    /// Add right to left and scale the result

    /// Add the right tile to the left. The right tile may be replaced with
    /// `ZeroTensor` argument, in which case the argument's element values are
    /// assumed to be `0`.
    /// \tparam R The right-hand tile argument type
    /// \param left The left-hand tile argument
    /// \param right The right-hand tile argument
    /// \return The sum of `left` and `right`.
    template <typename R>
    result_type consume_left(left_type& left, R&& right) const {
      return ScalAdd_::template eval<is_consumable_tile<left_type>::value,
          false>(left, std::forward<R>(right));
    }

    /// Add left to right and scale the result

    /// Add the left tile to the right, and scale the resulting left tile. The
    /// left tile may be replaced with `ZeroTensor` argument, in which case the
    /// argument's element values are assumed to be `0`.
    /// \tparam L The left-hand tile argument type
    /// \param left The left-hand tile argument
    /// \param right The right-hand tile argument
    /// \return The sum of `left` and `right`.
    template <typename L>
    result_type consume_right(L&& left, right_type& right) const {
      return ScalAdd_::template eval<false,
          is_consumable_tile<right_type>::value>(std::forward<L>(left), right);
    }

  }; // class ScalAdd


  // contains code for mixed-type expressions with result type enforcement
  namespace experimental {

    /// Tile addition operation

    /// This addition operation will add the content two tiles, and accepts an
    /// optional permute argument.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    /// \tparam LeftConsumable If \c true , will try to consume the left-hand argument when necessary.
    /// \tparam RightConsumable If \c true , will try to consume the right-hand argument when necessary.
    /// \note Input tiles can be consumed only if their type matches the result type.
    template <typename Result, typename Left, typename Right, bool LeftConsumable,
        bool RightConsumable>
    class Add {
    public:

      typedef Add<Result, Left, Right, LeftConsumable, RightConsumable> Add_;
      typedef Left left_type; ///< Left-hand argument base type
      typedef Right right_type; ///< Right-hand argument base type
      typedef Result result_type; ///< Result type

      /// indicates whether it is *possible* to consume the left tile
      static constexpr bool left_is_consumable =
          LeftConsumable && std::is_same<result_type, left_type>::value;
      /// indicates whether it is *possible* to consume the right tile
      static constexpr bool right_is_consumable =
          RightConsumable && std::is_same<result_type, right_type>::value;

    private:

      // Permuting tile evaluation function
      // These operations cannot consume the argument tile since this operation
      // requires temporary storage space.

      static result_type eval(const left_type& first, const right_type& second,
          const Permutation& perm)
      {
        using TiledArray::add;
        typedef decltype(add(std::declval<left_type>(), std::declval<right_type>(), std::declval<Permutation>()))
            add_result_type;
        Cast<result_type, add_result_type> cast;
        return cast(add(first, second, perm));
      }

      static result_type eval(ZeroTensor, const right_type& second,
          const Permutation& perm)
      {
        Permute<result_type,right_type> permute;
        return permute(second, perm);
      }

      static result_type eval(const left_type& first, ZeroTensor,
          const Permutation& perm)
      {
        Permute<result_type,left_type> permute;
        return permute(first, perm);
      }

      // Non-permuting tile evaluation functions
      // The compiler will select the correct functions based on the consumability
      // of the arguments.

      template <bool LC, bool RC,
          typename std::enable_if<!(LC || RC)>::type* = nullptr>
      static result_type eval(const left_type& first, const right_type& second) {
        using TiledArray::add;
        typedef decltype(add(std::declval<left_type>(), std::declval<right_type>()))
            add_result_type;
        Cast<result_type, add_result_type> cast;
        return cast(add(first, second));
      }

      template <bool LC, bool RC,
          typename std::enable_if<LC>::type* = nullptr>
      static result_type eval(left_type& first, const right_type& second) {
        using TiledArray::add_to;
        return add_to(first, second);
      }

      template <bool LC, bool RC,
          typename std::enable_if<!LC && RC>::type* = nullptr>
      static result_type eval(const left_type& first, right_type& second) {
        using TiledArray::add_to;
        return add_to(second, first);
      }

      template <bool LC, bool RC,
          typename std::enable_if<!RC>::type* = nullptr>
      static result_type eval(const ZeroTensor&, const right_type& second) {
        TiledArray::Clone<result_type,right_type> clone;
        return clone(second);
      }

      template <bool LC, bool RC,
          typename std::enable_if<RC>::type* = nullptr>
      static result_type eval(const ZeroTensor&, right_type& second) {
        return second;
      }

      template <bool LC, bool RC,
          typename std::enable_if<!LC>::type* = nullptr>
      static result_type eval(const left_type& first, const ZeroTensor&) {
        TiledArray::Clone<result_type,left_type> clone;
        return clone(first);
      }

      template <bool LC, bool RC,
          typename std::enable_if<LC>::type* = nullptr>
      static result_type eval(left_type& first, const ZeroTensor&) {
        return first;
      }

    public:

      /// Add-and-permute operator

      /// Compute the sum of two tiles and permute the result. One of the argument
      /// tiles may be replaced with `ZeroTensor` argument, in which case the
      /// argument's element values are assumed to be `0`.
      /// \tparam L The left-hand tile argument type
      /// \tparam R The right-hand tile argument type
      /// \param left The left-hand tile argument
      /// \param right The right-hand tile argument
      /// \param perm The permutation applied to the result tile
      /// \return The permuted and scaled sum of `left` and `right`.
      template <typename L, typename R>
      result_type operator()(L&& left, R&& right, const Permutation& perm) const {
        return eval(std::forward<L>(left), std::forward<R>(right), perm);
      }

      /// Add operator

      /// Compute the sum of two tiles. One of the argument tiles may be replaced
      /// with `ZeroTensor` argument, in which case the argument's element values
      /// are assumed to be `0`.
      /// \tparam L The left-hand tile argument type
      /// \tparam R The right-hand tile argument type
      /// \param left The left-hand tile argument
      /// \param right The right-hand tile argument
      /// \return The scaled sum of `left` and `right`.
      template <typename L, typename R>
      result_type operator()(L&& left, R&& right) const {
        return Add_::template eval<left_is_consumable, right_is_consumable>(
            std::forward<L>(left), std::forward<R>(right));
      }

      /// Add right to left

      /// Add the right tile to the left. The right tile may be replaced with
      /// `ZeroTensor` argument, in which case the argument's element values are
      /// assumed to be `0`.
      /// \tparam R The right-hand tile argument type
      /// \param left The left-hand tile argument
      /// \param right The right-hand tile argument
      /// \return The sum of `left` and `right`.
      template <typename R>
      result_type consume_left(left_type& left, R&& right) const {
        constexpr auto can_consume = is_consumable_tile<left_type>::value && std::is_same<result_type,left_type>::value;
        return Add_::template eval<can_consume,false>(left, std::forward<R>(right));
      }

      /// Add left to right

      /// Add the left tile to the right. The left tile may be replaced with
      /// `ZeroTensor` argument, in which case the argument's element values are
      /// assumed to be `0`.
      /// \tparam L The left-hand tile argument type
      /// \param left The left-hand tile argument
      /// \param right The right-hand tile argument
      /// \return The sum of `left` and `right`.
      template <typename L>
      result_type consume_right(L&& left, right_type& right) const {
        constexpr auto can_consume = is_consumable_tile<right_type>::value && std::is_same<result_type,right_type>::value;
        return Add_::template eval<false,can_consume>(std::forward<L>(left), right);
      }

    }; // class Add
  }

} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_ADD_H__INCLUDED
