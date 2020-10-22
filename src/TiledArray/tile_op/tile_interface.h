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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  tile_interface.h
 *  Sep 29, 2014
 *
 */

#ifndef TILEDARRAY_NONINTRUSIVE_API_TENSOR_H__INCLUDED
#define TILEDARRAY_NONINTRUSIVE_API_TENSOR_H__INCLUDED

#include <TiledArray/tensor/type_traits.h>
#include <TiledArray/type_traits.h>
#include <iterator>
#include <vector>

namespace TiledArray {

// Forward declaration
namespace math {
class GemmHelper;
}  // namespace math
namespace detail {
template <typename, typename>
class LazyArrayTile;
}  // namespace detail

/**
 * \defgroup NonIntrusiveTileInterface Non-intrusive tile interface
 * @{
 *
 * \page NonIntrusiveTileInterfaceDetails
 *
 * \tableofcontents
 *
 * \section TileInterfaceIntroduction Introduction
 *
 * To use a user defined tile types in TiledArray expressions, users must
 * define a set of interface function that define basic arithmetic and query
 * operations. It is not necessary to define all operations, only those that
 * are required for the algebraic tensor operations used in your application.
 * However, more than one function may be necessary for a given expression
 * operator. Each function has an intrusive and non-intrusive interface that
 * may be used to implement the required functionality. Below is a
 * description of each interface function, the intrusive and non-intrusive
 * function signatures of the function, and, in some cases, a reference
 * implementation. The reference implementation assumes that the tensor type
 * has a \c range() member for convenience, but this is not necessary. Tensor
 * implementations may have any arbitrary range interface function(s).
 *
 * \subsection TileInterfaceMinimumTileRequirements Minimum Tile Requirements
 *
 * The minimum requirements for user defined tile types are:
 *
 * \li An accessible copy constructor
 * \li An accessible destructor
 * \li Must be a shallow copy object
 *
 * TiledArray assumes tiles are shallow copy objects for efficiency when
 * coping objects and to avoid unnecessary replication of data. A shallow
 * copy object is an object that only copies a pointer (and updates a
 * reference counter) instead of explicitly copying all elements of the
 * tile. If your tile object is not a shallow copy object, you can use the
 * \c TiledArray::Tile class to wrap the object.
 *
 * \subsection TileInterfaceConvenstion Interface and Naming Convention
 *
 * The naming convention used for interface functions is:
 *
 * \li \c xxxx - Base \c xxxx operation that creates a new tile object from
 *     the input arguments.
 * \li \c xxxx_to - Base \c xxxx operation that modifies the first argument or
 *     calling object in-place (without constructing an new object).
 *
 * where \c xxxx represent the arithmetic/mutating operation performed by the
 * interface function, which includes:
 *
 * \li \c add
 * \li \c subt
 * \li \c mult
 * \li \c scal
 * \li \c gemm
 * \li \c neg
 * \li \c shift
 *
 * There are multiple overloaded version of these function, which combine
 * scaling, permuting, or scaling and permuting operations with the base
 * arithmetic/mutating operation.
 *
 * In the following sections, \c TensorType is used to represent an arbitrary
 * user-defined tile type that meets the minimum requirements specified above.
 * \c ScalarType is used to represent an built-in scalar data type (e.g.
 * \c int, \c float, etc.). The example code below is used for demonstration
 * purposes and may not be the an optimal solution.
 *
 * \section TileRequiredInterface Required Functions
 *
 * The minimum set of functions required are:
 *
 * \li \c empty
 * \li \c clone
 * \li \c permute
 * \li \c scale
 *
 * These functions are necessary for all tile operations.
 *
 * \subsection TileInterfaceEmpty Empty
 *
 * The empty function checks that the tile object has been initialized and is
 * usable in arithmetic operations. It returns \c true if the tile has not
 * been initialized, otherwise \c false. It is possible for empty to always
 * return \c false, if the tile type does not support default construction
 * (or uninitialized) objects.
 *
 * Non-intrusive interface:
 *
 * \code
 * bool empty(const TensorType& tensor);
 * \endcode
 *
 * Intrusive interface:
 *
 * \code
 * bool TensorType::empty();
 * \endcode
 *
 * \subsection TileInterfaceClone Clone
 *
 * The clone function creates a "deep" copy of a tensor, i.e. all data
 * elements of the tile are explicitly copied to a new object.
 *
 * Non-intrusive interface:
 *
 * \code
 * TensorType clone(const TensorType& tensor);
 * \endcode
 *
 * Intrusive interface:
 *
 * \code
 * TensorType TensorType::clone();
 * \endcode
 *
 * \subsection TileInterfacePermute Permute
 *
 * The permute function reorders the data of an input tile by swapping the
 * indices in the output tile. For example, the transpose of a matrix is the
 * equivalent of permutation P(1,0), where indices 0 and 1 are swapped. The
 * implementation of permute must support arbitrary permutations up to the
 * highest rank supported by the tile type.
 *
 * Non-intrusive interface:
 *
 * \code
 * TensorType permute(const TensorType& arg, const TiledArray::Permutation&
 * perm); \endcode
 *
 * Intrusive interface:
 *
 * \code
 * TensorType TensorType::permute(const TiledArray::Permutation& perm);
 * \endcode
 *
 * Example:
 *
 * The following code constructs a permuted copy of the argument. The example
 * code assumes \c TensorType has a \c range() member function that returns a
 * \c TiledArray::Range object. However, this is an implementation detail that
 * is not necessary.
 *
 * \code
 * TensorType
 * permute(const TensorType& arg, const TiledArray::Permutation& perm) {
 *   // Get tile boundaries
 *   const auto lobound = arg.range().lobound();
 *   const auto upbound = arg.range().upbound();
 *
 *   // Construct the result tile with a permuted range
 *   TensorType result(perm * arg.range());
 *
 *   // Iterate over tile elements
 *   for(auto it = arg.range().begin(); it < arg.range().end(); ++it) {
 *     // Get the coordinate index of the argument and the result element
 *     const auto index = it.index();
 *     const auto perm_index = perm * it.index();
 *
 *     result(perm_index) = arg(index);
 *   }
 *
 *   return result;
 * }
 * \endcode
 *
 * \subsection TileInterfaceScale Scale
 *
 * The scale interface consists of two function groups \c scale and
 * \c scale_to.
 *
 * Non-intrusive interface:
 *
 * \code
 * TensorType scale(const TensorType& arg, const ScalarType factor); // (1)
 * TensorType scale(const TensorType& arg, const ScalarType factor,  // (2)
 *     const TiledArray::Permutation& perm);
 * TensorType& scale_to(TensorType& arg, const ScalarType factor);   // (3)
 * \endcode
 *
 * Intrusive interface:
 *
 * \code
 * TensorType TensorType::scale(const ScalarType factor);       // (1)
 * TensorType TensorType::scale(const ScalarType factor,        // (2)
 *     const TiledArray::Permutation& perm);
 * TensorType& TensorType::scale_to(const ScalarType factor);   // (3)
 * \endcode
 *
 * Function (1) creates a copy of the argument tensor that is scaled by
 * \c factor, (2) creates a copy of the argument tensor that is scaled by
 * \c factor and permuted by \c perm, and (3) scales the argument tensor
 * in-place (without creating a copy).
 *
 * Example:
 *
 * \code
 * TensorType
 * scale(const TensorType& arg, const ScalarType factor) {
 *
 *   // Construct the result tile
 *   TensorType result(arg.range());
 *   std::transform(arg.begin(), arg.end(), result.begin(),
 *       [] (const TensorType& value) { return value * factor; });
 *
 *   return result;
 * }
 * \endcode
 * \section TileAdditionInterface Tile Addition Interface
 *
 * The tile addition interface include several functions, which are required
 * for to implement simple addition operations.
 *
 */

// Empty operations ----------------------------------------------------------

// to check that `arg` is empty (no data) just use std::empty

using std::empty;

// Subtraction ---------------------------------------------------------------

/// Subtract tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \param left The left-hand argument to be subtracted
/// \param right The right-hand argument to be subtracted
/// \return A tile that is equal to <tt>(left - right)</tt>
template <typename Left, typename Right>
inline auto subt(const Left& left, const Right& right) {
  return left.subt(right);
}

/// Subtract and scale tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar A scalar type
/// \param left The left-hand argument to be subtracted
/// \param right The right-hand argument to be subtracted
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(left - right) * factor</tt>
template <
    typename Left, typename Right, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline auto subt(const Left& left, const Right& right, const Scalar factor) {
  return left.subt(right, factor);
}

/// Subtract and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \param left The left-hand argument to be subtracted
/// \param right The right-hand argument to be subtracted
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (left - right)</tt>
template <typename Left, typename Right, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline auto subt(const Left& left, const Right& right, const Perm& perm) {
  return left.subt(right, perm);
}

/// Subtract, scale, and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar A scalar type
/// \param left The left-hand argument to be subtracted
/// \param right The right-hand argument to be subtracted
/// \param factor The scaling factor
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (left - right) * factor</tt>
template <
    typename Left, typename Right, typename Scalar, typename Perm,
    typename std::enable_if<detail::is_numeric_v<Scalar> &&
                            detail::is_permutation_v<Perm>>::type* = nullptr>
inline auto subt(const Left& left, const Right& right, const Scalar factor,
                 const Perm& perm) {
  return left.subt(right, factor, perm);
}

/// Subtract a scalar constant from the tile argument

/// \tparam Arg The tile argument type
/// \tparam Scalar A scalar type
/// \param arg The left-hand argument to be subtracted
/// \param value The constant scalar to be subtracted
/// \return A tile that is equal to <tt>arg - value</tt>
template <
    typename Arg, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline auto subt(const Arg& arg, const Scalar value) {
  return arg.subt(value);
}

/// Subtract a constant scalar and permute tile argument

/// \tparam Arg The tile argument type
/// \tparam Scalar A scalar type
/// \param arg The left-hand argument to be subtracted
/// \param value The constant scalar value to be subtracted
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (arg - value)</tt>
template <
    typename Arg, typename Scalar, typename Perm,
    typename std::enable_if<detail::is_numeric_v<Scalar> &&
                            detail::is_permutation_v<Perm>>::type* = nullptr>
inline auto subt(const Arg& arg, const Scalar value, const Perm& perm) {
  return arg.subt(value, perm);
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
/// \tparam Scalar A scalar type
/// \param result The result tile
/// \param arg The argument to be subtracted from \c result
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
template <
    typename Result, typename Arg, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline Result& subt_to(Result& result, const Arg& arg, const Scalar factor) {
  return result.subt_to(arg, factor);
}

/// Subtract constant scalar from the result tile

/// \tparam Result The result tile type
/// \tparam Scalar A scalar type
/// \param result The result tile
/// \param value The constant scalar to be subtracted from \c result
/// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
template <
    typename Result, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline Result& subt_to(Result& result, const Scalar value) {
  return result.subt_to(value);
}

template <typename... T>
using result_of_subt_t = decltype(subt(std::declval<T>()...));

template <typename... T>
using result_of_subt_to_t = decltype(subt_to(std::declval<T>()...));

// Multiplication operations -------------------------------------------------

/// Multiplication tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \param left The left-hand argument to be multiplied
/// \param right The right-hand argument to be multiplied
/// \return A tile that is equal to <tt>(left * right)</tt>
template <typename Left, typename Right>
inline auto mult(const Left& left, const Right& right) {
  return left.mult(right);
}

/// Multiplication and scale tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar A scalar type
/// \param left The left-hand argument to be multiplied
/// \param right The right-hand argument to be multiplied
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(left * right) * factor</tt>
template <typename Left, typename Right, typename Scalar,
          std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>* = nullptr>
inline auto mult(const Left& left, const Right& right, const Scalar factor) {
  return left.mult(right, factor);
}

/// Multiplication and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \param left The left-hand argument to be multiplied
/// \param right The right-hand argument to be multiplied
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (left * right)</tt>
template <
    typename Left, typename Right, typename Perm,
    typename = std::enable_if_t<detail::is_permutation_v<Perm> &&
                                detail::has_member_function_mult_anyreturn_v<
                                    const Left, const Right&, const Perm&>>>
inline auto mult(const Left& left, const Right& right, const Perm& perm) {
  return left.mult(right, perm);
}

/// Multiplication, scale, and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar A scalar type
/// \param left The left-hand argument to be multiplied
/// \param right The right-hand argument to be multiplied
/// \param factor The scaling factor
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (left * right) * factor</tt>
template <typename Left, typename Right, typename Scalar, typename Perm,
          std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar> &&
                           detail::is_permutation_v<Perm>>* = nullptr>
inline auto mult(const Left& left, const Right& right, const Scalar factor,
                 const Perm& perm) {
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
/// \tparam Scalar A scalar type
/// \param result The result tile to be multiplied
/// \param arg The argument to be multiplied by \c result
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(result *= arg) *= factor</tt>
template <typename Result, typename Arg, typename Scalar,
          std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>* = nullptr>
inline Result& mult_to(Result& result, const Arg& arg, const Scalar factor) {
  return result.mult_to(arg, factor);
}

template <typename... T>
using result_of_mult_t = decltype(mult(std::declval<T>()...));

template <typename... T>
using result_of_mult_to_t = decltype(mult_to(std::declval<T>()...));

// Scaling operations --------------------------------------------------------

// see tile_interface/scale.h

// Negation operations -------------------------------------------------------

/// Negate the tile argument

/// \tparam Arg The tile argument type
/// \param arg The argument to be negated
/// \return A tile that is equal to <tt>-arg</tt>
template <typename Arg>
inline auto neg(const Arg& arg) {
  return arg.neg();
}

/// Negate and permute tile argument

/// \tparam Arg The tile argument type
/// \param arg The argument to be negated
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ -arg</tt>
template <typename Arg, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline auto neg(const Arg& arg, const Perm& perm) {
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

template <typename... T>
using result_of_neg_t = decltype(neg(std::declval<T>()...));

template <typename... T>
using result_of_neg_to_t = decltype(neg_to(std::declval<T>()...));

// Complex conjugate operations ---------------------------------------------

/// Create a complex conjugated copy of a tile

/// \tparam Arg The tile argument type
/// \param arg The tile to be conjugated
/// \return A complex conjugated copy of `arg`
template <typename Arg>
inline auto conj(const Arg& arg) {
  return arg.conj();
}

/// Create a complex conjugated and scaled copy of a tile

/// \tparam Arg The tile argument type
/// \tparam Scalar A scalar type
/// \param arg The tile to be conjugated
/// \param factor The scaling factor
/// \return A complex conjugated and scaled copy of `arg`
template <typename Arg, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar>>::type* = nullptr>
inline auto conj(const Arg& arg, const Scalar factor) {
  return arg.conj(factor);
}

/// Create a complex conjugated and permuted copy of a tile

/// \tparam Arg The tile argument type
/// \param arg The tile to be conjugated
/// \param perm The permutation to be applied to `arg`
/// \return A complex conjugated and permuted copy of `arg`
template <typename Arg, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline auto conj(const Arg& arg, const Perm& perm) {
  return arg.conj(perm);
}

/// Create a complex conjugated, scaled, and permuted copy of a tile

/// \tparam Arg The tile argument type
/// \tparam Scalar A scalar type
/// \param arg The argument to be conjugated
/// \param factor The scaling factor
/// \param perm The permutation to be applied to `arg`
/// \return A complex conjugated, scaled, and permuted copy of `arg`
template <
    typename Arg, typename Scalar, typename Perm,
    typename std::enable_if<TiledArray::detail::is_numeric_v<Scalar> &&
                            detail::is_permutation_v<Perm>>::type* = nullptr>
inline auto conj(const Arg& arg, const Scalar factor, const Perm& perm) {
  return arg.conj(factor, perm);
}

/// In-place complex conjugate a tile

/// \tparam Result The tile type
/// \param result The tile to be conjugated
/// \return A reference to `result`
template <typename Result>
inline Result& conj_to(Result& result) {
  return result.conj_to();
}

/// In-place complex conjugate and scale a tile

/// \tparam Result The tile type
/// \tparam Scalar A scalar type
/// \param result The tile to be conjugated
/// \param factor The scaling factor
/// \return A reference to `result`
template <typename Result, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar>>::type* = nullptr>
inline Result& conj_to(Result& result, const Scalar factor) {
  return result.conj_to(factor);
}

template <typename... T>
using result_of_conj_t = decltype(conj(std::declval<T>()...));

template <typename... T>
using result_of_conj_to_t = decltype(conj_to(std::declval<T>()...));

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
          std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>* = nullptr>
inline auto gemm(const Left& left, const Right& right, const Scalar factor,
                 const math::GemmHelper& gemm_config) {
  return left.gemm(right, factor, gemm_config);
}

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
          std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>* = nullptr>
inline Result& gemm(Result& result, const Left& left, const Right& right,
                    const Scalar factor, const math::GemmHelper& gemm_config) {
  return result.gemm(left, right, factor, gemm_config);
}

template <typename... T>
using result_of_gemm_t = decltype(gemm(std::declval<T>()...));

// Reduction operations ------------------------------------------------------

/// Sum the hyper-diagonal elements a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to be summed
/// \return The sum of the hyper-diagonal elements of \c arg
// template <typename Arg>
// inline auto trace(const Arg& arg) {
//  return arg.trace();
//}

/// Sum the elements of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to be summed
/// \return A scalar that is equal to <tt>sum_i arg[i]</tt>
template <typename Arg>
inline auto sum(const Arg& arg) {
  return arg.sum();
}

/// Multiply the elements of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to be multiplied
/// \return A scalar that is equal to <tt>prod_i arg[i]</tt>
template <typename Arg>
inline auto product(const Arg& arg) {
  return arg.product();
}

/// Squared vector 2-norm of the elements of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to be multiplied and summed
/// \return The sum of the squared elements of \c arg
/// \return A scalar that is equal to <tt>sum_i arg[i] * arg[i]</tt>
template <typename Arg>
inline auto squared_norm(const Arg& arg) {
  return arg.squared_norm();
}

/// Vector 2-norm of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to be multiplied and summed
/// \return A scalar that is equal to <tt>sqrt(sum_i arg[i] * arg[i])</tt>
template <typename Arg>
inline auto norm(const Arg& arg) {
  return arg.norm();
}

/// Vector 2-norm of a tile

/// \tparam Arg The tile argument type
/// \tparam ResultType The result type
/// \param arg The argument to be multiplied and summed
/// \return A scalar that is equal to <tt>sqrt(sum_i arg[i] * arg[i])</tt>
template <typename Arg, typename ResultType>
inline void norm(const Arg& arg, ResultType& result) {
  result = arg.template norm<ResultType>();
}

/// Maximum element of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to find the maximum
/// \return A scalar that is equal to <tt>max(arg)</tt>
template <typename Arg>
inline auto max(const Arg& arg) {
  return arg.max();
}

/// Minimum element of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to find the minimum
/// \return A scalar that is equal to <tt>min(arg)</tt>
template <typename Arg>
inline auto min(const Arg& arg) {
  return arg.min();
}

/// Absolute maximum element of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to find the maximum
/// \return A scalar that is equal to <tt>abs(max(arg))</tt>
template <typename Arg>
inline auto abs_max(const Arg& arg) {
  return arg.abs_max();
}

/// Absolute mainimum element of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to find the minimum
/// \return A scalar that is equal to <tt>abs(min(arg))</tt>
template <typename Arg>
inline auto abs_min(const Arg& arg) {
  return arg.abs_min();
}

/// Vector dot product of two tiles

/// \tparam Left The left-hand argument tile type
/// \tparam Right The right-hand argument tile type
/// \param left The left-hand argument tile
/// \param right The right-hand argument tile
/// \return A scalar that is equal to <tt>sum_i left[i] * right[i]</tt>
template <typename Left, typename Right>
inline auto dot(const Left& left, const Right& right) {
  return left.dot(right);
}

/// Vector inner product of two tiles

/// \tparam Left The left-hand argument tile type
/// \tparam Right The right-hand argument tile type
/// \param left The left-hand argument tile
/// \param right The right-hand argument tile
/// \return A scalar that is equal to <tt>sum_i conj(left[i]) * right[i]</tt>
template <typename Left, typename Right>
inline auto inner_product(const Left& left, const Right& right) {
  return left.inner_product(right);
}

// template <typename T>
// using result_of_trace_t = decltype(mult(std::declval<T>()));

template <typename T>
using result_of_sum_t = decltype(sum(std::declval<T>()));

template <typename T>
using result_of_product_t = decltype(product(std::declval<T>()));

template <typename T>
using result_of_squared_norm_t = decltype(squared_norm(std::declval<T>()));

template <typename T, typename ResultType = T>
using result_of_norm_t =
    decltype(norm(std::declval<T>(), std::declval<ResultType&>()));

template <typename T>
using result_of_max_t = decltype(max(std::declval<T>()));

template <typename T>
using result_of_min_t = decltype(min(std::declval<T>()));

template <typename T>
using result_of_abs_max_t = decltype(abs_max(std::declval<T>()));

template <typename T>
using result_of_abs_min_t = decltype(abs_min(std::declval<T>()));

template <typename L, typename R>
using result_of_dot_t = decltype(dot(std::declval<L>(), std::declval<R>()));

/** @}*/

}  // namespace TiledArray

#endif /* TILEDARRAY_NONINTRUSIVE_API_TENSOR_H__INCLUDED */
