/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  Justusu Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  tile_interface.h
 *  Sep 29, 2014
 *
 */

#ifndef TILEDARRAY_NONINTRUSIVE_API_TENSOR_H__INCLUDED
#define TILEDARRAY_NONINTRUSIVE_API_TENSOR_H__INCLUDED

#include <TiledArray/type_traits.h>
#include <TiledArray/tile_op/eval_trait.h>

namespace TiledArray {

  // Forward declaration
  class Permutation;

  namespace math {

    // Forward declaration
    class GemmHelper;

  }  // namespace math

  /**
   * \defgroup NonIntrusiveTileInterface Non-intrusive tile interface
   * @{
   */



  // Clone operations ----------------------------------------------------------

  /// Create a copy of \c arg

  /// \tparam Arg The tile argument type
  /// \param arg The tile argument to be permuted
  /// \return A (deep) copy of \c arg
  template <typename Arg>
  inline Arg clone(const Arg& arg) {
    return arg.clone();
  }


  // Empty operations ----------------------------------------------------------

  /// Check that \c arg is empty (no data)

  /// \tparam Arg The tile argument type
  /// \param arg The tile argument to be permuted
  /// \return \c true if \c arg is empty, otherwise \c false.
  template <typename Arg>
  inline bool empty(const Arg& arg) {
    return arg.empty();
  }

  // Shift operations ----------------------------------------------------------

  /// Shift the range of \c arg

  /// \tparam Arg The tile argument type
  /// \tparam Index An array type
  /// \param arg The tile argument to be shifted
  /// \param range_shift The offset to be applied to the argument range
  /// \return A copy of the tile with a new range
  template <typename Arg, typename Index>
  inline auto shift(const Arg& arg, const Index& range_shift) ->
      decltype(arg.shift(range_shift))
  { return arg.shift(range_shift); }

  /// Shift the range of \c arg in place

  /// \tparam Arg The tile argument type
  /// \tparam Index An array type
  /// \param arg The tile argument to be shifted
  /// \param range_shift The offset to be applied to the argument range
  /// \return A copy of the tile with a new range
  template <typename Arg, typename Index>
  inline auto shift_to(Arg& arg, const Index& range_shift) ->
      decltype(arg.shift_to(range_shift))
  { return arg.shift_to(range_shift); }

  // Permutation operations ----------------------------------------------------

  /// Create a permuted copy of \c arg

  /// \tparam Arg The tile argument type
  /// \param arg The tile argument to be permuted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ arg</tt>
  template <typename Arg>
  inline auto permute(const Arg& arg, const Permutation& perm) ->
      decltype(arg.permute(perm))
  { return arg.permute(perm); }


  // Addition operations -------------------------------------------------------

  /// Add tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \return A tile that is equal to <tt>(left + right)</tt>
  template <typename Left, typename Right>
  inline auto add(const Left& left, const Right& right) ->
      decltype(left.add(right))
  { return left.add(right); }

  /// Add and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \tparam Scalar A scalar type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left + right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto add(const Left& left, const Right& right, const Scalar factor) ->
      decltype(left.add(right, factor))
  { return left.add(right, factor); }

  /// Add and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left + right)</tt>
  template <typename Left, typename Right>
  inline auto add(const Left& left, const Right& right, const Permutation& perm) ->
      decltype(left.add(right, perm))
  { return left.add(right, perm); }

  /// Add, scale, and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \tparam Scalar A scalar type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left + right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto add(const Left& left, const Right& right, const Scalar factor,
      const Permutation& perm) ->
      decltype(left.add(right, factor, perm))
  { return left.add(right, factor, perm); }

  /// Add a constant scalar to tile argument

  /// \tparam Arg The tile argument type
  /// \tparam Scalar A scalar type
  /// \param arg The left-hand argument to be added
  /// \param value The constant scalar to be added
  /// \return A tile that is equal to <tt>arg + value</tt>
  template <typename Arg, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto
  add(const Arg& arg, const Scalar value) -> decltype(arg.add(value))
  { return arg.add(value); }

  /// Add a constant scalar and permute tile argument

  /// \tparam Arg The tile argument type
  /// \tparam Scalar A scalar type
  /// \param arg The left-hand argument to be added
  /// \param value The constant scalar value to be added
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg + value)</tt>
  template <typename Arg, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto
  add(const Arg& arg, const Scalar value, const Permutation& perm) ->
      decltype(arg.add(value,perm))
  { return arg.add(value,perm); }

  /// Add to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be added to the result
  /// \return A tile that is equal to <tt>result[i] += arg[i]</tt>
  template <typename Result, typename Arg>
  inline Result& add_to(Result& result, const Arg& arg)
  { return result.add_to(arg); }

  /// Add and scale to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \tparam Scalar A scalar type
  /// \param result The result tile
  /// \param arg The argument to be added to \c result
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(result[i] += arg[i]) *= factor</tt>
  template <typename Result, typename Arg, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline Result& add_to(Result& result, const Arg& arg, const Scalar factor)
  { return result.add_to(arg, factor); }

  /// Add constant scalar to the result tile

  /// \tparam Result The result tile type
  /// \tparam Scalar A scalar type
  /// \param result The result tile
  /// \param value The constant scalar to be added to \c result
  /// \return A tile that is equal to <tt>(result[i] += arg[i]) *= factor</tt>
  template <typename Result, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline Result& add_to(Result& result, const Scalar value) {
    return result.add_to(value);
  }


  // Subtraction ---------------------------------------------------------------

  /// Subtract tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \return A tile that is equal to <tt>(left - right)</tt>
  template <typename Left, typename Right>
  inline auto subt(const Left& left, const Right& right) ->
      decltype(left.subt(right))
  { return left.subt(right); }

  /// Subtract and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \tparam Scalar A scalar type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left - right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto subt(const Left& left, const Right& right, const Scalar factor) ->
      decltype(left.subt(right, factor))
  { return left.subt(right, factor); }

  /// Subtract and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left - right)</tt>
  template <typename Left, typename Right>
  inline auto
  subt(const Left& left, const Right& right, const Permutation& perm) ->
      decltype(left.subt(right, perm))
  { return left.subt(right, perm); }

  /// Subtract, scale, and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \tparam Scalar A scalar type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left - right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto subt(const Left& left, const Right& right, const Scalar factor,
      const Permutation& perm) ->
      decltype(left.subt(right, factor, perm))
  { return left.subt(right, factor, perm); }

  /// Subtract a scalar constant from the tile argument

  /// \tparam Arg The tile argument type
  /// \tparam Scalar A scalar type
  /// \param arg The left-hand argument to be subtracted
  /// \param value The constant scalar to be subtracted
  /// \return A tile that is equal to <tt>arg - value</tt>
  template <typename Arg, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto subt(const Arg& arg, const Scalar value) ->
      decltype(arg.subt(value))
  { return arg.subt(value); }

  /// Subtract a constant scalar and permute tile argument

  /// \tparam Arg The tile argument type
  /// \tparam Scalar A scalar type
  /// \param arg The left-hand argument to be subtracted
  /// \param value The constant scalar value to be subtracted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg - value)</tt>
  template <typename Arg, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto
  subt(const Arg& arg, const Scalar value, const Permutation& perm) ->
      decltype(arg.subt(value,perm))
  { return arg.subt(value,perm); }

  /// Subtract from the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be subtracted from the result
  /// \return A tile that is equal to <tt>result[i] -= arg[i]</tt>
  template <typename Result, typename Arg>
  inline Result& subt_to(Result& result, const Arg& arg)
  { return result.subt_to(arg); }

  /// Subtract and scale from the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \tparam Scalar A scalar type
  /// \param result The result tile
  /// \param arg The argument to be subtracted from \c result
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
  template <typename Result, typename Arg, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline Result& subt_to(Result& result, const Arg& arg, const Scalar factor)
  { return result.subt_to(arg, factor); }

  /// Subtract constant scalar from the result tile

  /// \tparam Result The result tile type
  /// \tparam Scalar A scalar type
  /// \param result The result tile
  /// \param value The constant scalar to be subtracted from \c result
  /// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
  template <typename Result, typename Scalar,
      enable_if_t<detail::is_numeric<Scalar>::value>* = nullptr>
  inline Result& subt_to(Result& result, const Scalar value)
  { return result.subt_to(value); }


  // Multiplication operations -------------------------------------------------


  /// Multiplication tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \return A tile that is equal to <tt>(left * right)</tt>
  template <typename Left, typename Right>
  inline auto mult(const Left& left, const Right& right) ->
      decltype(left.mult(right))
  { return left.mult(right); }

  /// Multiplication and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \tparam Scalar A scalar type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left * right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      enable_if_t<TiledArray::detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto mult(const Left& left, const Right& right, const Scalar factor) ->
      decltype(left.mult(right, factor))
  { return left.mult(right, factor); }

  /// Multiplication and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left * right)</tt>
  template <typename Left, typename Right>
  inline auto
  mult(const Left& left, const Right& right, const Permutation& perm) ->
      decltype(left.mult(right, perm))
  { return left.mult(right, perm); }

  /// Multiplication, scale, and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \tparam Scalar A scalar type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left * right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      enable_if_t<TiledArray::detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto mult(const Left& left, const Right& right, const Scalar factor,
      const Permutation& perm) ->
      decltype(left.mult(right, factor, perm))
  { return left.mult(right, factor, perm); }

  /// Multiply to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile  to be multiplied
  /// \param arg The argument to be multiplied by the result
  /// \return A tile that is equal to <tt>result *= arg</tt>
  template <typename Result, typename Arg>
  inline Result& mult_to(Result& result, const Arg& arg)
  { return result.mult_to(arg); }

  /// Multiply and scale to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \tparam Scalar A scalar type
  /// \param result The result tile to be multiplied
  /// \param arg The argument to be multiplied by \c result
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(result *= arg) *= factor</tt>
  template <typename Result, typename Arg, typename Scalar,
      enable_if_t<TiledArray::detail::is_numeric<Scalar>::value>* = nullptr>
  inline Result& mult_to(Result& result, const Arg& arg,
      const Scalar factor)
  { return result.mult_to(arg, factor); }


  // Scaling operations --------------------------------------------------------

  /// Scalar the tile argument

  /// \tparam Arg The tile argument type
  /// \tparam Scalar A scalar type
  /// \param arg The left-hand argument to be scaled
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>arg * factor</tt>
  template <typename Arg, typename Scalar,
      enable_if_t<TiledArray::detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto scale(const Arg& arg, const Scalar factor) ->
      decltype(arg.scale(factor))
  { return arg.scale(factor); }

  /// Scale and permute tile argument

  /// \tparam Arg The tile argument type
  /// \tparam Scalar A scalar type
  /// \param arg The left-hand argument to be scaled
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg * factor)</tt>
  template <typename Arg, typename Scalar,
      enable_if_t<TiledArray::detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto scale(const Arg& arg, const Scalar factor, const Permutation& perm) ->
      decltype(arg.scale(factor, perm))
  { return arg.scale(factor, perm); }

  /// Scale to the result tile

  /// \tparam Result The result tile type
  /// \tparam Scalar A scalar type
  /// \param result The result tile to be scaled
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>result *= factor</tt>
  template <typename Result, typename Scalar,
      enable_if_t<TiledArray::detail::is_numeric<Scalar>::value>* = nullptr>
  inline Result& scale_to(Result& result, const Scalar factor)
  { return result.scale_to(factor); }


  // Negation operations -------------------------------------------------------

  /// Negate the tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be negated
  /// \return A tile that is equal to <tt>-arg</tt>
  template <typename Arg>
  inline auto neg(const Arg& arg) -> decltype(arg.neg())
  { return arg.neg(); }

  /// Negate and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be negated
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ -arg</tt>
  template <typename Arg>
  inline auto neg(const Arg& arg, const Permutation& perm) ->
      decltype(arg.neg(perm))
  { return arg.neg(perm); }

  /// Multiplication constant scalar to a tile

  /// \tparam Result The result tile type
  /// \param result The result tile to be negated
  /// \return A tile that is equal to <tt>result = -result</tt>
  template <typename Result>
  inline Result& neg_to(Result& result)
  { return result.neg_to(); }


  // Contraction operations ----------------------------------------------------


  /// Contract and scale tile arguments

  /// The contraction is done via a GEMM operation with fused indices as defined
  /// by \c gemm_config.
  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \tparam Scalar A scalar type
  /// \param left The left-hand argument to be contracted
  /// \param right The right-hand argument to be contracted
  /// \param factor The scaling factor
  /// \param gemm_config A helper object used to simplify gemm operations
  /// \return A tile that is equal to <tt>(left * right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      enable_if_t<TiledArray::detail::is_numeric<Scalar>::value>* = nullptr>
  inline auto gemm(const Left& left, const Right& right, const Scalar factor,
      const math::GemmHelper& gemm_config) ->
      decltype(left.gemm(right, factor, gemm_config))
  { return left.gemm(right, factor, gemm_config); }

  /// Contract and scale tile arguments to the result tile

  /// The contraction is done via a GEMM operation with fused indices as defined
  /// by \c gemm_config.
  /// \tparam Result The result tile type
  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \tparam Scalar A scalar type
  /// \param result The contracted result
  /// \param left The left-hand argument to be contracted
  /// \param right The right-hand argument to be contracted
  /// \param factor The scaling factor
  /// \param gemm_config A helper object used to simplify gemm operations
  /// \return A tile that is equal to <tt>result = (left * right) * factor</tt>
  template <typename Result, typename Left, typename Right, typename Scalar,
      enable_if_t<TiledArray::detail::is_numeric<Scalar>::value>* = nullptr>
  inline Result& gemm(Result& result, const Left& left, const Right& right,
            const Scalar factor, const math::GemmHelper& gemm_config)
  {
    return result.gemm(left, right, factor, gemm_config);
  }


  // Reduction operations ------------------------------------------------------

  /// Sum the hyper-diagonal elements a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be summed
  /// \return The sum of the hyper-diagonal elements of \c arg
  template <typename Arg>
  inline auto trace(const Arg& arg) -> decltype(arg.trace())
  { return arg.trace(); }

  /// Sum the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be summed
  /// \return A scalar that is equal to <tt>sum_i arg[i]</tt>
  template <typename Arg>
  inline auto sum(const Arg& arg) -> decltype(arg.sum())
  { return arg.sum(); }

  /// Multiply the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied
  /// \return A scalar that is equal to <tt>prod_i arg[i]</tt>
  template <typename Arg>
  inline auto product(const Arg& arg) -> decltype(arg.product())
  { return arg.product(); }

  /// Squared vector 2-norm of the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied and summed
  /// \return The sum of the squared elements of \c arg
  /// \return A scalar that is equal to <tt>sum_i arg[i] * arg[i]</tt>
  template <typename Arg>
  inline auto squared_norm(const Arg& arg) -> decltype(arg.squared_norm())
  { return arg.squared_norm(); }

  /// Vector 2-norm of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied and summed
  /// \return A scalar that is equal to <tt>sqrt(sum_i arg[i] * arg[i])</tt>
  template <typename Arg>
  inline auto norm(const Arg& arg) -> decltype(arg.norm())
  { return arg.norm(); }

  /// Maximum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the maximum
  /// \return A scalar that is equal to <tt>max(arg)</tt>
  template <typename Arg>
  inline auto max(const Arg& arg) -> decltype(arg.max())
  { return arg.max(); }

  /// Minimum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the minimum
  /// \return A scalar that is equal to <tt>min(arg)</tt>
  template <typename Arg>
  inline auto min(const Arg& arg) -> decltype(arg.min())
  { return arg.min(); }

  /// Absolute maximum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the maximum
  /// \return A scalar that is equal to <tt>abs(max(arg))</tt>
  template <typename Arg>
  inline auto abs_max(const Arg& arg) -> decltype(arg.abs_max())
  { return arg.abs_max(); }

  /// Absolute mainimum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the minimum
  /// \return A scalar that is equal to <tt>abs(min(arg))</tt>
  template <typename Arg>
  inline auto abs_min(const Arg& arg) -> decltype(arg.abs_min())
  { return arg.abs_min(); }

  /// Vector dot product of a tile

  /// \tparam Left The left-hand argument type
  /// \tparam Right The right-hand argument type
  /// \param left The left-hand argument tile to be contracted
  /// \param right The right-hand argument tile to be contracted
  /// \return A scalar that is equal to <tt>sum_i left[i] * right[i]</tt>
  template <typename Left, typename Right>
  inline auto dot(const Left& left, const Right& right) -> decltype(left.dot(right))
  { return left.dot(right); }

  /** @}*/

} // namespace TiledArray

#endif /* TILEDARRAY_NONINTRUSIVE_API_TENSOR_H__INCLUDED */
