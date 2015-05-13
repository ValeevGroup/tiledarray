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


  // The following functions define the non-intrusive interface used to apply
  // math operations to Array tiles. Users are expected to proved the default
  // implementation for these.

  /// Type traits for tiles

  /// This class allows users to override/specify the type traits of tiles.
  /// \tparam T Tensor type
  template <typename T>
  struct TileTrait {
    /// Tensor type that will store tile data
    typedef T tensor_type;
    /// Element type of the tile
    typedef typename T::value_type value_type;
    /// Range type used to represent the upper and lower bounds of the tensor data
    typedef typename T::range_type range_type;
    /// Size type used to represent the size and offsets of the tensor data
    typedef typename T::size_type size_type;
    /// Element reference type
    typedef typename T::reference reference;
    /// Element const reference type
    typedef typename T::const_reference const_reference;
    /// Element pointer type
    typedef typename T::pointer pointer;
    /// Element const pointer type
    typedef typename T::const_pointer const_pointer;
    /// Difference type used to represent the difference between two elements
    typedef typename T::difference_type difference_type;
    /// Element iterator type
    typedef typename T::iterator iterator;
    /// Element const iterator type
    typedef typename T::const_iterator const_iterator;
    /// The scalar type of the tensor type
    typedef typename TiledArray::detail::scalar_type<T>::type numeric_type;

  }; // struct TileTrait


  // Tensor range accessor -----------------------------------------------------

  /// Tile data pointer accessor

  /// Access the data pointer of tile
  /// \tparam Arg The tile type
  /// \param arg The tile argument
  /// \return A pointer to the data of \c arg
  template <typename Arg>
  inline const typename TileTrait<Arg>::range_type& range(Arg& arg) {
    return arg.range();
  }


  // Tile element accessors ----------------------------------------------------

  /// Array like accessor

  /// Access an element of \c arg for the given ordinal offset \c index.
  /// \tparam Arg The tensor type
  /// \param arg The tensor argument
  /// \param index The ordinal index of the tensor element
  /// \return A reference to the element at offset \c index of \c arg
  template <typename Arg>
  inline typename TileTrait<Arg>::reference
  array(Arg& arg, const typename TileTrait<Arg>::size_type index) {
    return arg[index];
  }


  /// Array like accessor

  /// Access an element of \c arg for the given ordinal offset \c index.
  /// \tparam Arg The tile type
  /// \param arg The tile argument
  /// \param index The ordinal index of the element
  /// \return A reference to the element at offset \c index of \c arg
  template <typename Arg, typename Index>
  inline typename TileTrait<Arg>::reference
  array(const Arg& arg, const typename TileTrait<Arg>::size_type index) {
    return arg[index];
  }


  /// Element accessor

  /// Access an element of \c arg at given a coordinate \c indices.
  /// \tparam Arg The tile type
  /// \tparam Indices The coordinate index types
  /// \param arg The tile argument
  /// \param indices The ordinal index of the element
  /// \return A reference to the element at offset \c index of \c arg
  template <typename Arg, typename... Indices>
  inline typename TileTrait<Arg>::reference
  element(Arg& arg, const Indices... indices) {
    return arg(indices...);
  }

  /// Element accessor

  /// Access an element of \c arg at given a coordinate \c indices.
  /// \tparam Arg The tile type
  /// \tparam Indices The coordinate index types
  /// \param arg The tile argument
  /// \param indices The ordinal index of the element
  /// \return A const reference to the element at offset \c index of \c arg
  template <typename Arg, typename... Indices>
  inline typename TileTrait<Arg>::const_reference
  element(const Arg& arg, const Indices... indices) {
    return arg(indices...);
  }

  // Data pointer accessor -----------------------------------------------------

  /// Tile data pointer accessor

  /// Access the data pointer of tile
  /// \tparam Arg The tile type
  /// \param arg The tile argument
  /// \return A pointer to the data of \c arg
  template <typename Arg>
  inline typename TileTrait<Arg>::pointer data(Arg& arg) {
    return arg.data();
  }

  /// Tile data const pointer accessor

  /// Access the data pointer of tile
  /// \tparam Arg The tile type
  /// \param arg The tile argument
  /// \return A pointer to the data of \c arg
  template <typename Arg>
  inline typename TileTrait<Arg>::const_pointer data(const Arg& arg) {
    return arg.data();
  }

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


  // Permutation operations ----------------------------------------------------

  /// Create a permuted copy of \c arg

  /// \tparam Arg The tile argument type
  /// \param arg The tile argument to be permuted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ arg</tt>
  template <typename Arg>
  inline Arg permute(const Arg& arg, const Permutation& perm) {
    return arg.permute(perm);
  }


  // Addition operations -------------------------------------------------------

  /// Add tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \return A tile that is equal to <tt>(left + right)</tt>
  template <typename Left, typename Right>
  inline Left add(const Left& left, const Right& right) {
    return left.add(right);
  }

  /// Add and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left + right) * factor</tt>
  template <typename Left, typename Right>
  inline Left add(const Left& left, const Right& right,
      const typename TileTrait<Left>::numeric_type factor)
  {
    return left.add(right, factor);
  }

  /// Add and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left + right)</tt>
  template <typename Left, typename Right>
  inline Left add(const Left& left, const Right& right, const Permutation& perm) {
    return left.add(right, perm);
  }

  /// Add, scale, and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left + right) * factor</tt>
  template <typename Left, typename Right>
  inline Left add(const Left& left, const Right& right,
      const typename TileTrait<Left>::numeric_type factor, const Permutation& perm)
  {
    return left.add(right, factor, perm);
  }

  /// Add a constant scalar to tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be added
  /// \param value The constant scalar to be added
  /// \return A tile that is equal to <tt>arg + value</tt>
  template <typename Arg>
  inline Arg add(const Arg& arg, const typename TileTrait<Arg>::numeric_type value) {
    return arg.add(value);
  }

  /// Add a constant scalar and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be added
  /// \param value The constant scalar value to be added
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg + value)</tt>
  template <typename Arg>
  inline Arg add(const Arg& arg, const typename TileTrait<Arg>::numeric_type value,
      const Permutation& perm)
  {
    return arg.add(value,perm);
  }

  /// Add to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be added to the result
  /// \return A tile that is equal to <tt>result[i] += arg[i]</tt>
  template <typename Result, typename Arg>
  inline Result& add_to(Result& result, const Arg& arg) {
    return result.add_to(arg);
  }

  /// Add and scale to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be added to \c result
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(result[i] += arg[i]) *= factor</tt>
  template <typename Result, typename Arg>
  inline Result& add_to(Result& result, const Arg& arg,
      const typename TileTrait<Result>::numeric_type factor)
  {
    return result.add_to(arg, factor);
  }

  /// Add constant scalar to the result tile

  /// \tparam Result The result tile type
  /// \param result The result tile
  /// \param value The constant scalar to be added to \c result
  /// \return A tile that is equal to <tt>(result[i] += arg[i]) *= factor</tt>
  template <typename Result>
  inline Result& add_to(Result& result, const typename TileTrait<Result>::numeric_type value) {
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
  inline Left subt(const Left& left, const Right& right) {
    return left.subt(right);
  }

  /// Subtract and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left - right) * factor</tt>
  template <typename Left, typename Right>
  inline Left subt(const Left& left, const Right& right,
      const typename TileTrait<Left>::numeric_type factor)
  {
    return left.subt(right, factor);
  }

  /// Subtract and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left - right)</tt>
  template <typename Left, typename Right>
  inline Left subt(const Left& left, const Right& right, const Permutation& perm) {
    return left.subt(right, perm);
  }

  /// Subtract, scale, and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left - right) * factor</tt>
  template <typename Left, typename Right>
  inline Left subt(const Left& left, const Right& right,
      const typename TileTrait<Left>::numeric_type factor, const Permutation& perm)
  {
    return left.subt(right, factor, perm);
  }

  /// Subtract a scalar constant from the tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be subtracted
  /// \param value The constant scalar to be subtracted
  /// \return A tile that is equal to <tt>arg - value</tt>
  template <typename Arg>
  inline Arg subt(const Arg& arg, const typename TileTrait<Arg>::numeric_type value) {
    return arg.subt(value);
  }

  /// Subtract a constant scalar and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be subtracted
  /// \param value The constant scalar value to be subtracted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg - value)</tt>
  template <typename Arg>
  inline Arg subt(const Arg& arg, const typename TileTrait<Arg>::numeric_type value,
      const Permutation& perm)
  {
    return arg.subt(value,perm);
  }

  /// Subtract from the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be subtracted from the result
  /// \return A tile that is equal to <tt>result[i] -= arg[i]</tt>
  template <typename Result, typename Arg>
  inline Result& subt_to(Result& result, const Arg& arg) {
    return result.subt_to(arg);
  }

  /// Subtract and scale from the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be subtracted from \c result
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
  template <typename Result, typename Arg>
  inline Result& subt_to(Result& result, const Arg& arg,
      const typename TileTrait<Result>::numeric_type factor)
  {
    return result.subt_to(arg, factor);
  }

  /// Subtract constant scalar from the result tile

  /// \tparam Result The result tile type
  /// \param result The result tile
  /// \param value The constant scalar to be subtracted from \c result
  /// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
  template <typename Result>
  inline Result& subt_to(Result& result, const typename TileTrait<Result>::numeric_type value) {
    return result.subt_to(value);
  }


  // Multiplication operations -------------------------------------------------


  /// Multiplication tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \return A tile that is equal to <tt>(left * right)</tt>
  template <typename Left, typename Right>
  inline Left mult(const Left& left, const Right& right) {
    return left.mult(right);
  }

  /// Multiplication and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left * right) * factor</tt>
  template <typename Left, typename Right>
  inline Left mult(const Left& left, const Right& right,
      const typename TileTrait<Left>::numeric_type factor)
  {
    return left.mult(right, factor);
  }

  /// Multiplication and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left * right)</tt>
  template <typename Left, typename Right>
  inline Left mult(const Left& left, const Right& right, const Permutation& perm) {
    return left.mult(right, perm);
  }

  /// Multiplication, scale, and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left * right) * factor</tt>
  template <typename Left, typename Right>
  inline Left mult(const Left& left, const Right& right,
      const typename TileTrait<Left>::numeric_type factor, const Permutation& perm)
  {
    return left.mult(right, factor, perm);
  }

  /// Multiply to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile  to be multiplied
  /// \param arg The argument to be multiplied by the result
  /// \return A tile that is equal to <tt>result *= arg</tt>
  template <typename Result, typename Arg>
  inline Result& mult_to(Result& result, const Arg& arg) {
    return result.mult_to(arg);
  }

  /// Multiply and scale to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile to be multiplied
  /// \param arg The argument to be multiplied by \c result
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(result *= arg) *= factor</tt>
  template <typename Result, typename Arg>
  inline Result& mult_to(Result& result, const Arg& arg,
      const typename TileTrait<Result>::numeric_type factor)
  {
    return result.mult_to(arg, factor);
  }


  // Scaling operations --------------------------------------------------------

  /// Scalar the tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be scaled
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>arg * factor</tt>
  template <typename Arg>
  inline Arg scale(const Arg& arg, const typename TileTrait<Arg>::numeric_type factor) {
    return arg.scale(factor);
  }

  /// Scale and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be scaled
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg * factor)</tt>
  template <typename Arg>
  inline Arg scale(const Arg& arg, const typename TileTrait<Arg>::numeric_type factor,
      const Permutation& perm)
  {
    return arg.scale(factor, perm);
  }

  /// Scale to the result tile

  /// \tparam Result The result tile type
  /// \param result The result tile to be scaled
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>result *= factor</tt>
  template <typename Result>
  inline Result& scale_to(Result& result, const typename TileTrait<Result>::numeric_type factor) {
    return result.scale_to(factor);
  }


  // Negation operations -------------------------------------------------------

  /// Negate the tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be negated
  /// \return A tile that is equal to <tt>-arg</tt>
  template <typename Arg>
  inline Arg neg(const Arg& arg) {
    return arg.neg();
  }

  /// Negate and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be negated
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ -arg</tt>
  template <typename Arg>
  inline Arg neg(const Arg& arg, const Permutation& perm) {
    return arg.neg(perm);
  }

  /// Multiplication constant scalar to a tile

  /// \tparam Result The result tile type
  /// \param result The result tile to be negated
  /// \return A tile that is equal to <tt>result = -result</tt>
  template <typename Result>
  inline Result& neg_to(Result& result) {
    return result.neg_to();
  }


  // Contraction operations ----------------------------------------------------


  /// Contract and scale tile arguments

  /// The contraction is done via a GEMM operation with fused indices as defined
  /// by \c gemm_config.
  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be contracted
  /// \param right The right-hand argument to be contracted
  /// \param factor The scaling factor
  /// \param gemm_config A helper object used to simplify gemm operations
  /// \return A tile that is equal to <tt>(left * right) * factor</tt>
  template <typename Left, typename Right>
  inline Left gemm(const Left& left, const Right& right,
      const typename TileTrait<Left>::numeric_type factor, const math::GemmHelper& gemm_config)
  {
    return left.gemm(right, factor, gemm_config);
  }

  /// Contract and scale tile arguments to the result tile

  /// The contraction is done via a GEMM operation with fused indices as defined
  /// by \c gemm_config.
  /// \tparam Result The result tile type
  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param result The contracted result
  /// \param left The left-hand argument to be contracted
  /// \param right The right-hand argument to be contracted
  /// \param factor The scaling factor
  /// \param gemm_config A helper object used to simplify gemm operations
  /// \return A tile that is equal to <tt>result = (left * right) * factor</tt>
  template <typename Result, typename Left, typename Right>
  inline Result& gemm(Result& result, const Left& left, const Right& right,
            const typename TileTrait<Result>::numeric_type factor, const math::GemmHelper& gemm_config)
  {
    return result.gemm(left, right, factor, gemm_config);
  }


  // Reduction operations ------------------------------------------------------

  /// Sum the hyper-diagonal elements a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be summed
  /// \return The sum of the hyper-diagonal elements of \c arg
  template <typename Arg>
  inline typename TileTrait<Arg>::numeric_type trace(const Arg& arg) {
    return arg.trace();
  }

  /// Sum the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be summed
  /// \return A scalar that is equal to <tt>sum_i arg[i]</tt>
  template <typename Arg>
  inline typename TileTrait<Arg>::numeric_type sum(const Arg& arg) {
    return arg.sum();
  }

  /// Multiply the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied
  /// \return A scalar that is equal to <tt>prod_i arg[i]</tt>
  template <typename Arg>
  inline typename TileTrait<Arg>::numeric_type product(const Arg& arg) {
    return arg.product();
  }

  /// Squared vector 2-norm of the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied and summed
  /// \return The sum of the squared elements of \c arg
  /// \return A scalar that is equal to <tt>sum_i arg[i] * arg[i]</tt>
  template <typename Arg>
  inline typename TileTrait<Arg>::numeric_type squared_norm(const Arg& arg) {
    return arg.squared_norm();
  }

  /// Vector 2-norm of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied and summed
  /// \return A scalar that is equal to <tt>sqrt(sum_i arg[i] * arg[i])</tt>
  template <typename Arg>
  inline typename TileTrait<Arg>::numeric_type norm(const Arg& arg) {
    return arg.norm();
  }

  /// Maximum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the maximum
  /// \return A scalar that is equal to <tt>max(arg)</tt>
  template <typename Arg>
  inline typename TileTrait<Arg>::numeric_type max(const Arg& arg) {
    return arg.max();
  }

  /// Minimum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the minimum
  /// \return A scalar that is equal to <tt>min(arg)</tt>
  template <typename Arg>
  inline typename TileTrait<Arg>::numeric_type min(const Arg& arg) {
    return arg.min();
  }

  /// Absolute maximum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the maximum
  /// \return A scalar that is equal to <tt>abs(max(arg))</tt>
  template <typename Arg>
  inline typename TileTrait<Arg>::numeric_type abs_max(const Arg& arg) {
    return arg.abs_max();
  }

  /// Absolute mainimum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the minimum
  /// \return A scalar that is equal to <tt>abs(min(arg))</tt>
  template <typename Arg>
  inline typename TileTrait<Arg>::numeric_type abs_min(const Arg& arg) {
    return arg.abs_min();
  }

  /// Vector dot product of a tile

  /// \tparam Left The left-hand argument type
  /// \tparam Right The right-hand argument type
  /// \param left The left-hand argument tile to be contracted
  /// \param right The right-hand argument tile to be contracted
  /// \return A scalar that is equal to <tt>sum_i left[i] * right[i]</tt>
  template <typename Left, typename Right>
  inline typename TileTrait<Left>::numeric_type dot(const Left& left, const Right& right) {
    return left.dot(right);
  }

  /** @}*/

} // namespace TiledArray

#endif /* TILEDARRAY_NONINTRUSIVE_API_TENSOR_H__INCLUDED */
