/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 */

#ifndef TILEDARRAY_TILE_H__INCLUDED
#define TILEDARRAY_TILE_H__INCLUDED

#include <TiledArray/tile_op/tile_interface.h>
#include <memory>

// Forward declaration of MADNESS archive type traits
namespace madness {
  namespace archive {

    template <typename> struct is_output_archive;
    template <typename> struct is_input_archive;

  }  // namespace archive
}  // namespace madness

namespace TiledArray {

  /// An N-dimensional shallow copy wrapper for tile objects

  /// \c Tile represents a slice of an Array. The rank of the tile slice is the
  /// same as the owning \c Array object.
  /// \tparam T The tensor type used to represent tile data
  template <typename T>
  class Tile {
  public:
    /// This object type
    typedef Tile<T> Tile_;
    /// Tensor type used to represent tile data
    typedef typename TileTrait<T>::tensor_type tensor_type;
    /// size type used to represent the size and offsets of the tensor data
    typedef typename TileTrait<T>::size_type size_type;
    /// Tensor element type
    typedef typename TileTrait<T>::value_type value_type;
    /// Element reference type
    typedef typename TileTrait<T>::reference reference;
    /// Element reference type
    typedef typename TileTrait<T>::const_reference const_reference;
    /// Element pointer type
    typedef typename TileTrait<T>::pointer pointer;
    /// Element const pointer type
    typedef typename TileTrait<T>::const_pointer const_pointer;
    /// Difference type
    typedef typename TileTrait<T>::difference_type difference_type;
    /// Element iterator type
    typedef typename TileTrait<T>::iterator iterator;
    /// Element const iterator type
    typedef typename TileTrait<T>::const_iterator const_iterator;
    /// The scalar type of the tensor type
    typedef typename TileTrait<T>::numeric_type numeric_type;

  private:

    std::shared_ptr<tensor_type> pimpl_;

  public:

    // Constructors and destructor ---------------------------------------------

    Tile() = default;
    Tile(const Tile_&) = default;
    Tile(Tile_&&) = default;

    explicit Tile(const tensor_type& tensor) :
      pimpl_(std::make_shared<tensor_type>(tensor))
    { }

    explicit Tile(tensor_type&& tensor) :
      pimpl_(std::make_shared<tensor_type>(std::move(tensor)))
    { }

    ~Tile() = default;

    // Assignment operators ----------------------------------------------------

    Tile_& operator=(Tile_&&) = default;
    Tile_& operator=(const Tile_&) = default;

    Tile_& operator=(const tensor_type& tensor) {
      *pimpl_ = tensor;
      return *this;
    }

    Tile_& operator=(tensor_type&& tensor) {
      *pimpl_ = std::move(tensor);
      return *this;
    }


    // Tile accessor -----------------------------------------------------------

    tensor_type& tensor() { return pimpl_->tensor_; }

    const tensor_type& tensor() const { return pimpl_->tensor_; }


    // Iterator accessor -------------------------------------------------------


    /// Iterator factory

    /// \return An iterator to the first data element
    iterator begin() { return std::begin(pimpl_->tensor_); }

    /// Iterator factory

    /// \return A const iterator to the first data element
    const_iterator begin() const { return std::begin(pimpl_->tensor_); }

    /// Iterator factory

    /// \return An iterator to the last data element
    iterator end() { return std::end(pimpl_->tensor_); }

    /// Iterator factory

    /// \return A const iterator to the last data element
    const_iterator end() const { return std::end(pimpl_->tensor_); }


    // Serialization -----------------------------------------------------------

    template <typename Archive,
        enable_if_t<madness::archive::is_output_archive<Archive>::value>* = nullptr>
    void serialize(Archive &ar) const {
      // Serialize data for empty tile check
      bool empty = !static_cast<bool>(pimpl_);
      ar & empty;
      if (!empty) {
        // Serialize tile data
        ar & pimpl_->tensor_;
      }
    }

    template <typename Archive,
        enable_if_t<madness::archive::is_input_archive<Archive>::value>* = nullptr>
    void serialize(Archive &ar) {
      // Check for empty tile
      bool empty = false;
      ar & empty;

      if (!empty) {
        // Deserialize tile data
        tensor_type tensor;
        ar & tensor;

        // construct a new pimpl
        pimpl_ = std::make_shared<T>(std::move(tensor));
      } else {
        // Set pimpl to an empty tile
        pimpl_.reset();
      }
    }

  }; // class Tile

  // The following functions define the non-intrusive interface used to apply
  // math operations to Tiles. These functions in turn use the non-intrusive
  // interface functions to evaluate tiles.


  // Clone operations ----------------------------------------------------------

  /// Create a copy of \c arg

  /// \tparam Arg The tile argument type
  /// \param arg The tile argument to be permuted
  /// \return A (deep) copy of \c arg
  template <typename Arg>
  inline Tile<Arg> clone(const Tile<Arg>& arg) {
    return Tile<Arg>(clone(arg.tensor()));
  }


  // Empty operations ----------------------------------------------------------

  /// Check that \c arg is empty (no data)

  /// \tparam Arg The tile argument type
  /// \param arg The tile argument to be permuted
  /// \return \c true if \c arg is empty, otherwise \c false.
  template <typename Arg>
  inline bool empty(const Tile<Arg>& arg) {
    return empty(arg.tensor());
  }


  // Permutation operations ----------------------------------------------------

  /// Create a permuted copy of \c arg

  /// \tparam Arg The tile argument type
  /// \param arg The tile argument to be permuted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ arg</tt>
  template <typename Arg>
  inline Tile<Arg> permute(const Tile<Arg>& arg, const Permutation& perm) {
    return Tile<Arg>(permute(arg.tensor(), perm));
  }


  // Addition operations -------------------------------------------------------

  /// Add tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \return A tile that is equal to <tt>(left + right)</tt>
  template <typename Left, typename Right>
  inline Tile<Left> add(const Tile<Left>& left, const Tile<Right>& right) {
    return Tile<Left>(add(left.tensor(), right.tensor()));
  }

  /// Add and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left + right) * factor</tt>
  template <typename Left, typename Right>
  inline Tile<Left> add(const Tile<Left>& left, const Tile<Right>& right,
      const typename Tile<Left>::numeric_type factor)
  {
    return Tile<Left>(add(left.tensor(), right.tensor(), factor));
  }

  /// Add and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left + right)</tt>
  template <typename Left, typename Right>
  inline Tile<Left> add(const Tile<Left>& left, const Tile<Right>& right,
      const Permutation& perm)
  {
    return Tile<Left>(add(left.tensor(), right.tensor(), perm));
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
  inline Tile<Left> add(const Tile<Left>& left, const Tile<Right>& right,
      const typename Tile<Left>::numeric_type factor, const Permutation& perm)
  {
    return Tile<Left>(add(left.tensor(), right.tensor(), factor, perm));
  }

  /// Add a constant scalar to tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be added
  /// \param value The constant scalar to be added
  /// \return A tile that is equal to <tt>arg + value</tt>
  template <typename Arg>
  inline Tile<Arg> add(const Tile<Arg>& arg,
      const typename Tile<Arg>::numeric_type value)
  {
    return Tile<Arg>(add(arg.tensor(), value));
  }

  /// Add a constant scalar and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be added
  /// \param value The constant scalar value to be added
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg + value)</tt>
  template <typename Arg>
  inline Tile<Arg> add(const Tile<Arg>& arg,
      const typename Tile<Arg>::numeric_type value, const Permutation& perm)
  {
    return Tile<Arg>(add(arg.tensor(), value, perm));
  }

  /// Add to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be added to the result
  /// \return A tile that is equal to <tt>result[i] += arg[i]</tt>
  template <typename Result, typename Arg>
  inline Tile<Result>& add_to(Tile<Result>& result, const Tile<Arg>& arg) {
    return Tile<Result>(add_to(result.tensor(), arg.tensor()));
  }

  /// Add and scale to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be added to \c result
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(result[i] += arg[i]) *= factor</tt>
  template <typename Result, typename Arg>
  inline Tile<Result>& add_to(Tile<Result>& result, const Tile<Arg>& arg,
      const typename Tile<Result>::numeric_type factor)
  {
    return Tile<Arg>(add_to(result.tensor(), arg.tensor(), factor));
  }

  /// Add constant scalar to the result tile

  /// \tparam Result The result tile type
  /// \param result The result tile
  /// \param value The constant scalar to be added to \c result
  /// \return A tile that is equal to <tt>(result[i] += arg[i]) *= factor</tt>
  template <typename Result>
  inline Tile<Result>& add_to(Tile<Result>& result,
      const typename Tile<Result>::numeric_type value)
  {
    return Tile<Result>(add_to(result.tensor(), value));
  }


  // Subtraction ---------------------------------------------------------------

  /// Subtract tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \return A tile that is equal to <tt>(left - right)</tt>
  template <typename Left, typename Right>
  inline Tile<Left> subt(const Tile<Left>& left, const Tile<Right>& right) {
    return Tile<Left>(subt(left.tensor(), right.tensor()));
  }

  /// Subtract and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left - right) * factor</tt>
  template <typename Left, typename Right>
  inline Tile<Left> subt(const Tile<Left>& left, const Tile<Right>& right,
      const typename Tile<Left>::numeric_type factor)
  {
    return Tile<Left>(subt(left.tensor(), right.tensor(), factor));
  }

  /// Subtract and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left - right)</tt>
  template <typename Left, typename Right>
  inline Tile<Left> subt(const Tile<Left>& left, const Tile<Right>& right,
      const Permutation& perm)
  {
    return Tile<Left>(subt(left.tensor(), right.tensor(), perm));
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
  inline Tile<Left> subt(const Tile<Left>& left, const Tile<Right>& right,
      const typename Tile<Left>::numeric_type factor, const Permutation& perm)
  {
    return Tile<Left>(subt(left.tensor(), right.tensor(), factor, perm));
  }

  /// Subtract a scalar constant from the tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be subtracted
  /// \param value The constant scalar to be subtracted
  /// \return A tile that is equal to <tt>arg - value</tt>
  template <typename Arg>
  inline Tile<Arg> subt(const Tile<Arg>& arg,
      const typename Tile<Arg>::numeric_type value)
  {
    return Tile<Arg>(subt(arg.tensor(), value));
  }

  /// Subtract a constant scalar and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be subtracted
  /// \param value The constant scalar value to be subtracted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg - value)</tt>
  template <typename Arg>
  inline Tile<Arg> subt(const Tile<Arg>& arg,
      const typename Tile<Arg>::numeric_type value, const Permutation& perm)
  {
    return Tile<Arg>(subt(arg.tensor(), value, perm));
  }

  /// Subtract from the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be subtracted from the result
  /// \return A tile that is equal to <tt>result[i] -= arg[i]</tt>
  template <typename Result, typename Arg>
  inline Tile<Result>& subt_to(Tile<Result>& result, const Tile<Arg>& arg) {
    return Tile<Result>(subt_to(result.tensor(), arg.tensor()));
  }

  /// Subtract and scale from the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be subtracted from \c result
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
  template <typename Result, typename Arg>
  inline Tile<Result>& subt_to(Tile<Result>& result, const Tile<Arg>& arg,
      const typename Tile<Result>::numeric_type factor)
  {
    return Tile<Result>(subt_to(result.tensor(), arg.tensor(), factor));
  }

  /// Subtract constant scalar from the result tile

  /// \tparam Result The result tile type
  /// \param result The result tile
  /// \param value The constant scalar to be subtracted from \c result
  /// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
  template <typename Result>
  inline Tile<Result>& subt_to(Tile<Result>& result,
      const typename Tile<Result>::numeric_type value)
  {
    return Tile<Result>(subt_to(result.tensor(), value));
  }


  // Multiplication operations -------------------------------------------------


  /// Multiplication tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \return A tile that is equal to <tt>(left * right)</tt>
  template <typename Left, typename Right>
  inline Tile<Left> mult(const Tile<Left>& left, const Tile<Right>& right) {
    return Tile<Left>(mult(left.tensor(), right.tensor()));
  }

  /// Multiplication and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left * right) * factor</tt>
  template <typename Left, typename Right>
  inline Tile<Left> mult(const Tile<Left>& left, const Tile<Right>& right,
      const typename Tile<Left>::numeric_type factor)
  {
    return Tile<Left>(mult(left.tensor(), right.tensor(), factor));
  }

  /// Multiplication and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left * right)</tt>
  template <typename Left, typename Right>
  inline Tile<Left> mult(const Tile<Left>& left, const Tile<Right>& right,
      const Permutation& perm)
  {
    return Tile<Left>(mult(left.tensor(), right.tensor(), perm));
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
  inline Tile<Left> mult(const Tile<Left>& left, const Tile<Right>& right,
      const typename Tile<Left>::numeric_type factor, const Permutation& perm)
  {
    return Tile<Left>(mult(left.tensor(), right.tensor(), factor, perm));
  }

  /// Multiply to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile  to be multiplied
  /// \param arg The argument to be multiplied by the result
  /// \return A tile that is equal to <tt>result *= arg</tt>
  template <typename Result, typename Arg>
  inline Tile<Result>& mult_to(Tile<Result>& result, const Tile<Arg>& arg) {
    return Tile<Result>(mult_to(result.tensor(), arg.tensor()));
  }

  /// Multiply and scale to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile to be multiplied
  /// \param arg The argument to be multiplied by \c result
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(result *= arg) *= factor</tt>
  template <typename Result, typename Arg>
  inline Tile<Result>& mult_to(Tile<Result>& result, const Tile<Arg>& arg,
      const typename Tile<Result>::numeric_type factor)
  {
    return Tile<Arg>(mult_to(result.tensor(), arg.tensor(), factor));
  }


  // Scaling operations --------------------------------------------------------

  /// Scalar the tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be scaled
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>arg * factor</tt>
  template <typename Arg>
  inline Tile<Arg> scale(const Tile<Arg>& arg,
      const typename Tile<Arg>::numeric_type factor)
  {
    return Tile<Arg>(scale(arg.tensor(), factor));
  }

  /// Scale and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be scaled
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg * factor)</tt>
  template <typename Arg>
  inline Tile<Arg> scale(const Tile<Arg>& arg,
      const typename Tile<Arg>::numeric_type factor, const Permutation& perm)
  {
    return Tile<Arg>(scale(arg.tensor(), factor, perm));
  }

  /// Scale to the result tile

  /// \tparam Result The result tile type
  /// \param result The result tile to be scaled
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>result *= factor</tt>
  template <typename Result>
  inline Tile<Result>& scale_to(Tile<Result>& result,
      const typename Tile<Result>::numeric_type factor)
  {
    return Tile<Result>(scale_to(result.tensor(), factor));
  }


  // Negation operations -------------------------------------------------------

  /// Negate the tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be negated
  /// \return A tile that is equal to <tt>-arg</tt>
  template <typename Arg>
  inline Tile<Arg> neg(const Tile<Arg>& arg) {
    return Tile<Arg>(neg(arg.tensor()));
  }

  /// Negate and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be negated
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ -arg</tt>
  template <typename Arg>
  inline Tile<Arg> neg(const Tile<Arg>& arg, const Permutation& perm) {
    return Tile<Arg>(neg(arg.tensor(), perm));
  }

  /// Multiplication constant scalar to a tile

  /// \tparam Result The result tile type
  /// \param result The result tile to be negated
  /// \return A tile that is equal to <tt>result = -result</tt>
  template <typename Result>
  inline Tile<Result>& neg_to(Tile<Result>& result) {
    return Tile<Result>(neg_to(result.tensor()));
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
  inline Tile<Left> gemm(const Tile<Left>& left, const Tile<Right>& right,
      const typename Tile<Left>::numeric_type factor,
      const math::GemmHelper& gemm_config)
  {
    return Tile<Left>(gemm(left.tensor(), right.tensor(), factor, gemm_config));
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
  inline Tile<Result>& gemm(Tile<Result>& result, const Tile<Left>& left,
      const Tile<Right>& right, const typename Tile<Result>::numeric_type factor,
      const math::GemmHelper& gemm_config)
  {
    return Tile<Result>(gemm(result.tensor(), left.tensor(), right.result(), factor,
        gemm_config));
  }


  // Reduction operations ------------------------------------------------------

  /// Sum the hyper-diagonal elements a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be summed
  /// \return The sum of the hyper-diagonal elements of \c arg
  template <typename Arg>
  inline typename Tile<Arg>::numeric_type trace(const Tile<Arg>& arg) {
    return trace(arg.tensor());
  }

  /// Sum the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be summed
  /// \return A scalar that is equal to <tt>sum_i arg[i]</tt>
  template <typename Arg>
  inline typename Tile<Arg>::numeric_type sum(const Tile<Arg>& arg) {
    return sum(arg.tensor());
  }

  /// Multiply the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied
  /// \return A scalar that is equal to <tt>prod_i arg[i]</tt>
  template <typename Arg>
  inline typename Tile<Arg>::numeric_type product(const Tile<Arg>& arg) {
    return product(arg.tensor());
  }

  /// Squared vector 2-norm of the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied and summed
  /// \return The sum of the squared elements of \c arg
  /// \return A scalar that is equal to <tt>sum_i arg[i] * arg[i]</tt>
  template <typename Arg>
  inline typename Tile<Arg>::numeric_type squared_norm(const Tile<Arg>& arg) {
    return squared_norm(arg.tensor());
  }

  /// Vector 2-norm of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied and summed
  /// \return A scalar that is equal to <tt>sqrt(sum_i arg[i] * arg[i])</tt>
  template <typename Arg>
  inline typename Tile<Arg>::numeric_type norm(const Tile<Arg>& arg) {
    return norm(arg.tensor());
  }

  /// Maximum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the maximum
  /// \return A scalar that is equal to <tt>max(arg)</tt>
  template <typename Arg>
  inline typename Tile<Arg>::numeric_type max(const Tile<Arg>& arg) {
    return max(arg.tensor());
  }

  /// Minimum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the minimum
  /// \return A scalar that is equal to <tt>min(arg)</tt>
  template <typename Arg>
  inline typename Tile<Arg>::numeric_type min(const Tile<Arg>& arg) {
    return min(arg.tensor());
  }

  /// Absolute maximum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the maximum
  /// \return A scalar that is equal to <tt>abs(max(arg))</tt>
  template <typename Arg>
  inline typename Tile<Arg>::numeric_type abs_max(const Tile<Arg>& arg) {
    return abs_max(arg.tensor());
  }

  /// Absolute mainimum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the minimum
  /// \return A scalar that is equal to <tt>abs(min(arg))</tt>
  template <typename Arg>
  inline typename Tile<Arg>::numeric_type
  abs_min(const Tile<Arg>& arg) {
    return abs_min(arg.tensor());
  }

  /// Vector dot product of a tile

  /// \tparam Left The left-hand argument type
  /// \tparam Right The right-hand argument type
  /// \param left The left-hand argument tile to be contracted
  /// \param right The right-hand argument tile to be contracted
  /// \return A scalar that is equal to <tt>sum_i left[i] * right[i]</tt>
  template <typename Left, typename Right>
  inline typename Tile<Left>::numeric_type
  dot(const Tile<Left>& left, const Tile<Right>& right) {
    return dot(left.tensor(), right.tensor());
  }



  template <typename Left, typename Right>
  inline Tile<Left> operator+(const Tile<Left>&left, const Tile<Right>& right) {
    return add(left, right);
  }

  template <typename Left, typename Right>
  inline Tile<Left> operator-(const Tile<Left>& left, const Tile<Right>& right) {
    return subt(left, right);
  }

  template <typename Left, typename Right>
  Tile<Left> operator*(const Tile<Left>& left, const Tile<Right>& right) {
    return mult(left, right);
  }

  template <typename Left, typename Right,
      enable_if_t<TiledArray::detail::is_numeric<Right>::value>* = nullptr>
  Tile<Left> operator*(const Tile<Left>& left, const Right right) {
    return scale(left, right);
  }

  template <typename Left, typename Right,
      enable_if_t<TiledArray::detail::is_numeric<Left>::value>* = nullptr>
  Tile<Right> operator*(const Left left, const Tile<Right>& right) {
    return scale(right, left);
  }

  template <typename Arg>
  inline Tile<Arg> operator-(const Tile<Arg>& arg) {
    return neg(arg);
  }

  template <typename Arg>
  inline Tile<Arg> operator*(const Permutation& perm, Tile<Arg> const arg) {
    return permute(arg, perm);
  }

  template <typename T>
  inline std::ostream &operator<<(std::ostream &os, const Tile<T>& tile) {
    os << tile.range() << ": " << tile.tensor() << "\n";
    return os;
  }

}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_H__INCLUDED
