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


  /**
   * \defgroup TileInterface Tile interface for user defined tensor types
   * @{
   */

  /// An N-dimensional shallow copy wrapper for tile objects

  /// \c Tile represents a block of an \c Array. The rank of the tile block is
  /// the same as the owning \c Array object. In order for a user defined tensor
  /// object to be used in TiledArray expressions, users must also define the
  /// following functions:
  /// \li \c add
  /// \li \c add_to
  /// \li \c subt
  /// \li \c subt_to
  /// \li \c mult
  /// \li \c mult_to
  /// \li \c scal
  /// \li \c scal_to
  /// \li \c gemm
  /// \li \c neg
  /// \li \c permute
  /// \li \c empty
  /// \li \c shift
  /// \li \c shift_to
  /// \li \c trance
  /// \li \c sum
  /// \li \c product
  /// \li \c squared_norm
  /// \li \c norm
  /// \li \c min
  /// \li \c max
  /// \li \c abs_min
  /// \li \c abs_max
  /// \li \c dot
  /// as for the intrusive or non-instrusive interface. See the
  /// \ref NonIntrusiveTileInterface "non-intrusive tile interface"
  /// documentation for more details.
  /// \tparam T The tensor type used to represent tile data
  template <typename T>
  class Tile {
  public:
    /// This object type
    typedef Tile<T> Tile_;
    /// Tensor type used to represent tile data
    typedef T tensor_type;

  private:

    std::shared_ptr<tensor_type> pimpl_;

  public:

    // Constructors and destructor ---------------------------------------------

    Tile() = default;
    Tile(const Tile_&) = default;
    Tile(Tile_&&) = default;

    /// Forwarding ctor

    /// To simplify construction, Tile provides ctors that all forward their args
    /// to T. To avoid clashing with copy and move ctors need conditional instantiation --
    /// e.g. see http://ericniebler.com/2013/08/07/universal-references-and-the-copy-constructo/
    /// NB For Arg that can be converted to Tile also use the copy/move ctors.
    template <typename Arg,
      typename = typename std::enable_if<
        not detail::is_same_or_derived<Tile_,Arg>::value &&
        not std::is_convertible<Arg,Tile_>::value
      >::type
    >
    explicit Tile(Arg&& arg) :
      pimpl_(std::make_shared<tensor_type>(std::forward<Arg>(arg)))
    { }

    template <typename Arg1, typename Arg2, typename ... Args>
    Tile(Arg1&& arg1, Arg2&& arg2, Args&&... args) :
      pimpl_(std::make_shared<tensor_type>(std::forward<Arg1>(arg1),
                                           std::forward<Arg2>(arg2),
                                           std::forward<Args>(args)...))
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


    // State accessor ----------------------------------------------------------

    bool empty() const {
      return not bool(pimpl_);
    }

    // Tile accessor -----------------------------------------------------------

    tensor_type& tensor() { return *pimpl_; }

    const tensor_type& tensor() const { return *pimpl_; }


    // Iterator accessor -------------------------------------------------------

    /// Iterator factory

    /// \return An iterator to the first data element
    decltype(auto) begin()
    { return std::begin(tensor()); }

    /// Iterator factory

    /// \return A const iterator to the first data element
    decltype(auto) begin() const
    { return std::begin(tensor()); }

    /// Iterator factory

    /// \return An iterator to the last data element
    decltype(auto) end()
    { return std::end(tensor()); }

    /// Iterator factory

    /// \return A const iterator to the last data element
    decltype(auto) end() const
    { return std::end(tensor()); }


    // Dimension information accessors -----------------------------------------

    /// Size accessors

    /// \return The number of elements in the tensor
    decltype(auto) size() const
    { return tensor().size(); }

    /// Range accessor

    /// \return An object describes the upper and lower bounds of  the tensor data
    decltype(auto) range() const
    { return tensor().range(); }


    // Element accessors -------------------------------------------------------

    /// Const element accessor via subscript operator

    /// \param i The ordinal index of the element to be returned
    /// \return The i-th element of the tensor
    decltype(auto) operator[](std::size_t i) const
    { return tensor()[i]; }

    /// Element accessor via subscript operator

    /// \param i The ordinal index of the element to be returned
    /// \return The i-th element of the tensor
    decltype(auto) operator[](std::size_t i)
    { return tensor()[i]; }

    /// Const element accessor via parentheses operator

    /// \tparam I The set of coordinate index types (integral types)
    /// \param i The set of coordinate indices of the tile element
    /// \return The element of the tensor at the coordinate (i...)
    template <typename... I>
    decltype(auto) operator()(const I... i) const
    { return tensor()(i...); }

    /// Element accessor via parentheses operator

    /// \tparam I The set of coordinate index types (integral types)
    /// \param i The set of coordinate indices of the tile element
    /// \return The element of the tensor at the coordinate (i...)
    template <typename... I>
    decltype(auto) operator()(const I... i)
    { return tensor()(i...); }


    // Serialization -----------------------------------------------------------

    template <typename Archive,
        typename std::enable_if<madness::archive::is_output_archive<Archive>::value>::type* = nullptr>
    void serialize(Archive &ar) const {
      // Serialize data for empty tile check
      bool empty = !static_cast<bool>(pimpl_);
      ar & empty;
      if (!empty) {
        // Serialize tile data
        ar & *pimpl_;
      }
    }

    template <typename Archive,
        typename std::enable_if<madness::archive::is_input_archive<Archive>::value>::type* = nullptr>
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

  namespace detail {

    /// Factory function for tiles

    /// \tparam T A tensor type
    /// \param t A tensor object
    /// \return A tile that wraps a copy of t.
    template <typename T>
    Tile<T> make_tile(T&& t) { return Tile<T>(std::forward<T>(t)); }

  }  // namespace detail


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
    return arg.empty() || empty(arg.tensor());
  }


  // Permutation operations ----------------------------------------------------

  /// Create a permuted copy of \c arg

  /// \tparam Arg The tile argument type
  /// \param arg The tile argument to be permuted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ arg</tt>
  template <typename Arg>
  inline decltype(auto) permute(const Tile<Arg>& arg, const Permutation& perm)
  {
    return Tile<Arg>(permute(arg.tensor(), perm));
  }


  // Shift operations ----------------------------------------------------------

  /// Shift the range of \c arg

  /// \tparam Arg The tensor argument type
  /// \tparam Index An array type
  /// \param arg The tile argument to be shifted
  /// \param range_shift The offset to be applied to the argument range
  /// \return A copy of the tile with a new range
  template <typename Arg, typename Index>
  inline decltype(auto) shift(const Tile<Arg>& arg, const Index& range_shift)
  { return detail::make_tile(shift(arg.tensor(), range_shift)); }

  /// Shift the range of \c arg in place

  /// \tparam Arg The tensor argument type
  /// \tparam Index An array type
  /// \param arg The tile argument to be shifted
  /// \param range_shift The offset to be applied to the argument range
  /// \return A copy of the tile with a new range
  template <typename Arg, typename Index>
  inline Tile<Arg>& shift_to(Tile<Arg>& arg, const Index& range_shift) {
    shift_to(arg.tensor(), range_shift);
    return arg;
  }


  // Addition operations -------------------------------------------------------

  /// Add tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \return A tile that is equal to <tt>(left + right)</tt>
  template <typename Left, typename Right>
  inline decltype(auto) add(const Tile<Left>& left, const Tile<Right>& right)
  { return detail::make_tile(add(left.tensor(), right.tensor())); }

  /// Add and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \tparam Scalar A scalar type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left + right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto) add(const Tile<Left>& left, const Tile<Right>& right, const Scalar factor)
  { return detail::make_tile(add(left.tensor(), right.tensor(), factor)); }

  /// Add and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be added
  /// \param right The right-hand argument to be added
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm * (left + right)</tt>
  template <typename Left, typename Right>
  inline decltype(auto) add(const Tile<Left>& left, const Tile<Right>& right, const Permutation& perm)
  { return detail::make_tile(add(left.tensor(), right.tensor(), perm)); }

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
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto) add(const Tile<Left>& left, const Tile<Right>& right,
      const Scalar factor, const Permutation& perm)
  { return detail::make_tile(add(left.tensor(), right.tensor(), factor, perm)); }

  /// Add a constant scalar to tile argument

  /// \tparam Arg The tile argument type
  /// \tparam Scalar A scalar type
  /// \param arg The left-hand argument to be added
  /// \param value The constant scalar to be added
  /// \return A tile that is equal to <tt>arg + value</tt>
  template <typename Arg, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto) add(const Tile<Arg>& arg, const Scalar value)
  { return detail::make_tile(add(arg.tensor(), value)); }

  /// Add a constant scalar and permute tile argument

  /// \tparam Arg The tile argument type
  /// \tparam Scalar A scalar type
  /// \param arg The left-hand argument to be added
  /// \param value The constant scalar value to be added
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg + value)</tt>
  template <typename Arg, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto)
  add(const Tile<Arg>& arg, const Scalar value, const Permutation& perm)
  { return detail::make_tile(add(arg.tensor(), value, perm)); }

  /// Add to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be added to the result
  /// \return A tile that is equal to <tt>result[i] += arg[i]</tt>
  template <typename Result, typename Arg>
  inline Tile<Result>& add_to(Tile<Result>& result, const Tile<Arg>& arg) {
    add_to(result.tensor(), arg.tensor());
    return result;
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
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline Tile<Result>&
  add_to(Tile<Result>& result, const Tile<Arg>& arg, const Scalar factor) {
    add_to(result.tensor(), arg.tensor(), factor);
    return result;
  }

  /// Add constant scalar to the result tile

  /// \tparam Result The result tile type
  /// \tparam Scalar A scalar type
  /// \param result The result tile
  /// \param value The constant scalar to be added to \c result
  /// \return A tile that is equal to <tt>(result[i] += arg[i]) *= factor</tt>
  template <typename Result, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline Tile<Result>& add_to(Tile<Result>& result, const Scalar value) {
    add_to(result.tensor(), value);
    return result;
  }


  // Subtraction ---------------------------------------------------------------

  /// Subtract tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \return A tile that is equal to <tt>(left - right)</tt>
  template <typename Left, typename Right>
  inline decltype(auto)
  subt(const Tile<Left>& left, const Tile<Right>& right)
  { return detail::make_tile(subt(left.tensor(), right.tensor())); }

  /// Subtract and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left - right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto)
  subt(const Tile<Left>& left, const Tile<Right>& right, const Scalar factor)
  { return detail::make_tile(subt(left.tensor(), right.tensor(), factor)); }

  /// Subtract and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left - right)</tt>
  template <typename Left, typename Right>
  inline decltype(auto)
  subt(const Tile<Left>& left, const Tile<Right>& right, const Permutation& perm)
  { return detail::make_tile(subt(left.tensor(), right.tensor(), perm)); }

  /// Subtract, scale, and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be subtracted
  /// \param right The right-hand argument to be subtracted
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left - right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto)
  subt(const Tile<Left>& left, const Tile<Right>& right, const Scalar factor,
      const Permutation& perm)
  { return detail::make_tile(subt(left.tensor(), right.tensor(), factor, perm)); }

  /// Subtract a scalar constant from the tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be subtracted
  /// \param value The constant scalar to be subtracted
  /// \return A tile that is equal to <tt>arg - value</tt>
  template <typename Arg, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto)
  subt(const Tile<Arg>& arg, const Scalar value)
  { return detail::make_tile(subt(arg.tensor(), value)); }

  /// Subtract a constant scalar and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be subtracted
  /// \param value The constant scalar value to be subtracted
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg - value)</tt>
  template <typename Arg, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto)
  subt(const Tile<Arg>& arg, const Scalar value, const Permutation& perm)
  { return detail::make_tile(subt(arg.tensor(), value, perm)); }

  /// Subtract from the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be subtracted from the result
  /// \return A tile that is equal to <tt>result[i] -= arg[i]</tt>
  template <typename Result, typename Arg>
  inline Tile<Result>& subt_to(Tile<Result>& result, const Tile<Arg>& arg) {
    subt_to(result.tensor(), arg.tensor());
    return result;
  }

  /// Subtract and scale from the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile
  /// \param arg The argument to be subtracted from \c result
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
  template <typename Result, typename Arg, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline Tile<Result>&
  subt_to(Tile<Result>& result, const Tile<Arg>& arg, const Scalar factor) {
    subt_to(result.tensor(), arg.tensor(), factor);
    return result;
  }

  /// Subtract constant scalar from the result tile

  /// \tparam Result The result tile type
  /// \param result The result tile
  /// \param value The constant scalar to be subtracted from \c result
  /// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
  template <typename Result, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline Tile<Result>& subt_to(Tile<Result>& result, const Scalar value) {
    subt_to(result.tensor(), value);
    return result;
  }


  // Multiplication operations -------------------------------------------------


  /// Multiplication tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \return A tile that is equal to <tt>(left * right)</tt>
  template <typename Left, typename Right>
  inline decltype(auto) mult(const Tile<Left>& left, const Tile<Right>& right)
  { return detail::make_tile(mult(left.tensor(), right.tensor())); }

  /// Multiplication and scale tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(left * right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto)
  mult(const Tile<Left>& left, const Tile<Right>& right,  const Scalar factor)
  { return detail::make_tile(mult(left.tensor(), right.tensor(), factor)); }

  /// Multiplication and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left * right)</tt>
  template <typename Left, typename Right>
  inline decltype(auto)
  mult(const Tile<Left>& left, const Tile<Right>& right, const Permutation& perm)
  { return detail::make_tile(mult(left.tensor(), right.tensor(), perm)); }

  /// Multiplication, scale, and permute tile arguments

  /// \tparam Left The left-hand tile type
  /// \tparam Right The right-hand tile type
  /// \param left The left-hand argument to be multiplied
  /// \param right The right-hand argument to be multiplied
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (left * right) * factor</tt>
  template <typename Left, typename Right, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto)
  mult(const Tile<Left>& left, const Tile<Right>& right, const Scalar factor,
      const Permutation& perm)
  { return detail::make_tile(mult(left.tensor(), right.tensor(), factor, perm)); }

  /// Multiply to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile  to be multiplied
  /// \param arg The argument to be multiplied by the result
  /// \return A tile that is equal to <tt>result *= arg</tt>
  template <typename Result, typename Arg>
  inline Tile<Result>& mult_to(Tile<Result>& result, const Tile<Arg>& arg) {
    mult_to(result.tensor(), arg.tensor());
    return result;
  }

  /// Multiply and scale to the result tile

  /// \tparam Result The result tile type
  /// \tparam Arg The argument tile type
  /// \param result The result tile to be multiplied
  /// \param arg The argument to be multiplied by \c result
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>(result *= arg) *= factor</tt>
  template <typename Result, typename Arg, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline Tile<Result>& mult_to(Tile<Result>& result, const Tile<Arg>& arg,
      const Scalar factor)
  {
    mult_to(result.tensor(), arg.tensor(), factor);
    return result;
  }


  // Scaling operations --------------------------------------------------------

  /// Scalar the tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be scaled
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>arg * factor</tt>
  template <typename Arg, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto) scale(const Tile<Arg>& arg, const Scalar factor)
  { return detail::make_tile(scale(arg.tensor(), factor)); }

  /// Scale and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The left-hand argument to be scaled
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ (arg * factor)</tt>
  template <typename Arg, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto)
  scale(const Tile<Arg>& arg, const Scalar factor, const Permutation& perm)
  { return detail::make_tile(scale(arg.tensor(), factor, perm)); }

  /// Scale to the result tile

  /// \tparam Result The result tile type
  /// \param result The result tile to be scaled
  /// \param factor The scaling factor
  /// \return A tile that is equal to <tt>result *= factor</tt>
  template <typename Result, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline Tile<Result>& scale_to(Tile<Result>& result, const Scalar factor) {
    scale_to(result.tensor(), factor);
    return result;
  }


  // Negation operations -------------------------------------------------------

  /// Negate the tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be negated
  /// \return A tile that is equal to <tt>-arg</tt>
  template <typename Arg>
  inline decltype(auto) neg(const Tile<Arg>& arg)
  { return detail::make_tile(neg(arg.tensor())); }

  /// Negate and permute tile argument

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be negated
  /// \param perm The permutation to be applied to the result
  /// \return A tile that is equal to <tt>perm ^ -arg</tt>
  template <typename Arg>
  inline decltype(auto) neg(const Tile<Arg>& arg, const Permutation& perm)
  { return detail::make_tile(neg(arg.tensor(), perm)); }

  /// Multiplication constant scalar to a tile

  /// \tparam Result The result tile type
  /// \param result The result tile to be negated
  /// \return A tile that is equal to <tt>result = -result</tt>
  template <typename Result>
  inline Tile<Result>& neg_to(Tile<Result>& result) {
    neg_to(result.tensor());
    return result;
  }


  // Complex conjugate operations ---------------------------------------------

  /// Create a complex conjugated copy of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The tile to be conjugated
  /// \return A complex conjugated copy of `arg`
  template <typename Arg>
  inline decltype(auto) conj(const Tile<Arg>& arg)
  { return detail::make_tile(conj(arg.tensor())); }

  /// Create a complex conjugated and scaled copy of a tile

  /// \tparam Arg The tile argument type
  /// \tparam Scalar A scalar type
  /// \param arg The tile to be conjugated
  /// \param factor The scaling factor
  /// \return A complex conjugated and scaled copy of `arg`
  template <typename Arg, typename Scalar,
      typename std::enable_if<
          TiledArray::detail::is_numeric<Scalar>::value
      >::type* = nullptr>
  inline decltype(auto) conj(const Tile<Arg>& arg, const Scalar factor)
  { return detail::make_tile(conj(arg.tensor(), factor)); }

  /// Create a complex conjugated and permuted copy of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The tile to be conjugated
  /// \param perm The permutation to be applied to `arg`
  /// \return A complex conjugated and permuted copy of `arg`
  template <typename Arg>
  inline decltype(auto) conj(const Tile<Arg>& arg, const Permutation& perm)
  { return detail::make_tile(conj(arg.tensor(), perm)); }

  /// Create a complex conjugated, scaled, and permuted copy of a tile

  /// \tparam Arg The tile argument type
  /// \tparam Scalar A scalar type
  /// \param arg The argument to be conjugated
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to `arg`
  /// \return A complex conjugated, scaled, and permuted copy of `arg`
  template <typename Arg, typename Scalar,
      typename std::enable_if<
          TiledArray::detail::is_numeric<Scalar>::value
      >::type* = nullptr>
  inline decltype(auto) conj(const Tile<Arg>& arg, const Scalar factor, const Permutation& perm)
  { return detail::make_tile(conj(arg.tensor(), factor, perm)); }

  /// In-place complex conjugate a tile

  /// \tparam Result The tile type
  /// \param result The tile to be conjugated
  /// \return A reference to `result`
  template <typename Result>
  inline Result& conj_to(Tile<Result>& result) {
    conj_to(result.tensor());
    return result;
  }

  /// In-place complex conjugate and scale a tile

  /// \tparam Result The tile type
  /// \tparam Scalar A scalar type
  /// \param result The tile to be conjugated
  /// \param factor The scaling factor
  /// \return A reference to `result`
  template <typename Result, typename Scalar,
      typename std::enable_if<
          TiledArray::detail::is_numeric<Scalar>::value
      >::type* = nullptr>
  inline Result& conj_to(Tile<Result>& result, const Scalar factor) {
    conj_to(result.tensor(), factor);
    return result;
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
  template <typename Left, typename Right, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline decltype(auto) gemm(const Tile<Left>& left, const Tile<Right>& right,
      const Scalar factor, const math::GemmHelper& gemm_config)
  { return detail::make_tile(gemm(left.tensor(), right.tensor(), factor, gemm_config)); }

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
  template <typename Result, typename Left, typename Right, typename Scalar,
      typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
  inline Tile<Result>& gemm(Tile<Result>& result, const Tile<Left>& left,
      const Tile<Right>& right, const Scalar factor,
      const math::GemmHelper& gemm_config)
  {
    gemm(result.tensor(), left.tensor(), right.tensor(), factor, gemm_config);
    return result;
  }


  // Reduction operations ------------------------------------------------------

  /// Sum the hyper-diagonal elements a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be summed
  /// \return The sum of the hyper-diagonal elements of \c arg
  template <typename Arg>
  inline decltype(auto) trace(const Tile<Arg>& arg)
  { return trace(arg.tensor()); }

  /// Sum the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be summed
  /// \return A scalar that is equal to <tt>sum_i arg[i]</tt>
  template <typename Arg>
  inline decltype(auto) sum(const Tile<Arg>& arg)
  { return sum(arg.tensor()); }

  /// Multiply the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied
  /// \return A scalar that is equal to <tt>prod_i arg[i]</tt>
  template <typename Arg>
  inline decltype(auto) product(const Tile<Arg>& arg)
  { return product(arg.tensor()); }

  /// Squared vector 2-norm of the elements of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied and summed
  /// \return The sum of the squared elements of \c arg
  /// \return A scalar that is equal to <tt>sum_i arg[i] * arg[i]</tt>
  template <typename Arg>
  inline decltype(auto) squared_norm(const Tile<Arg>& arg)
  { return squared_norm(arg.tensor()); }

  /// Vector 2-norm of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to be multiplied and summed
  /// \return A scalar that is equal to <tt>sqrt(sum_i arg[i] * arg[i])</tt>
  template <typename Arg>
  inline decltype(auto) norm(const Tile<Arg>& arg)
  { return norm(arg.tensor()); }

  /// Maximum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the maximum
  /// \return A scalar that is equal to <tt>max(arg)</tt>
  template <typename Arg>
  inline decltype(auto) max(const Tile<Arg>& arg)
  { return max(arg.tensor()); }

  /// Minimum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the minimum
  /// \return A scalar that is equal to <tt>min(arg)</tt>
  template <typename Arg>
  inline decltype(auto) min(const Tile<Arg>& arg)
  { return min(arg.tensor()); }

  /// Absolute maximum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the maximum
  /// \return A scalar that is equal to <tt>abs(max(arg))</tt>
  template <typename Arg>
  inline decltype(auto) abs_max(const Tile<Arg>& arg)
  { return abs_max(arg.tensor()); }

  /// Absolute mainimum element of a tile

  /// \tparam Arg The tile argument type
  /// \param arg The argument to find the minimum
  /// \return A scalar that is equal to <tt>abs(min(arg))</tt>
  template <typename Arg>
  inline decltype(auto) abs_min(const Tile<Arg>& arg)
  { return abs_min(arg.tensor()); }

  /// Vector dot product of a tile

  /// \tparam Left The left-hand argument type
  /// \tparam Right The right-hand argument type
  /// \param left The left-hand argument tile to be contracted
  /// \param right The right-hand argument tile to be contracted
  /// \return A scalar that is equal to <tt>sum_i left[i] * right[i]</tt>
  template <typename Left, typename Right>
  inline decltype(auto) dot(const Tile<Left>& left, const Tile<Right>& right)
  { return dot(left.tensor(), right.tensor()); }

  // Tile arithmetic operators -------------------------------------------------


  /// Add tiles operator

  /// \tparam Left The left-hand tensor type
  /// \tparam Right The right-hand tensor type
  /// \param left The left-hand tile
  /// \param right The right-hand tile
  /// \return The sum of the \c left and \c right tiles
  template <typename Left, typename Right>
  inline decltype(auto) operator+(const Tile<Left>& left, const Tile<Right>& right)
  { return add(left, right); }

  /// In-place add tile operator

  /// Add the elements of the \c right tile to that of the \c left tile.
  /// \tparam Left The left-hand tensor type
  /// \tparam Right The right-hand tensor type
  /// \param left The left-hand tile
  /// \param right The right-hand tile
  /// \return The \c left tile, <tt>left[i] += right[i]</tt>
  template <typename Left, typename Right>
  inline Tile<Left>& operator+=(Tile<Left>& left, const Tile<Right>& right)
  { return add_to(left, right); }

  /// Subtract tiles operator

  /// \tparam Left The left-hand tensor type
  /// \tparam Right The right-hand tensor type
  /// \param left The left-hand tile
  /// \param right The right-hand tile
  /// \return The difference of the \c left and \c right tiles
  template <typename Left, typename Right>
  inline decltype(auto) operator-(const Tile<Left>& left, const Tile<Right>& right)
  { return subt(left, right); }

  /// In-place subtract tile operator

  /// Subtract the elements of the \c right tile from that of the \c left tile.
  /// \tparam Left The left-hand tensor type
  /// \tparam Right The right-hand tensor type
  /// \param left The left-hand tile
  /// \param right The right-hand tile
  /// \return The \c left tile, <tt>left[i] -= right[i]</tt>
  template <typename Left, typename Right>
  inline Tile<Left>& operator-=(Tile<Left>& left, const Tile<Right>& right)
  { return subt_to(left, right); }

  /// Product tiles operator

  /// \tparam Left The left-hand tensor type
  /// \tparam Right The right-hand tensor type
  /// \param left The left-hand tile
  /// \param right The right-hand tile
  /// \return The product of the \c left and \c right tiles
  template <typename Left, typename Right>
  inline decltype(auto) operator*(const Tile<Left>& left, const Tile<Right>& right)
  { return mult(left, right); }

  /// Scale tile operator

  /// \tparam Left The left-hand tensor type
  /// \tparam Right The right-hand scalar type
  /// \param left The left-hand tile
  /// \param right The right-hand scaling factor
  /// \return The \c left tile scaled by \c right
  template <typename Left, typename Right,
      typename std::enable_if<detail::is_numeric<Right>::value>::type* = nullptr>
  inline decltype(auto) operator*(const Tile<Left>& left, const Right right)
  { return scale(left, right); }

  /// Scale tile operator

  /// \tparam Left The left-hand scalar type
  /// \tparam Right The right-hand scalar type
  /// \param left The left-hand scaling factor
  /// \param right The right-hand tile
  /// \return The \c right tile scaled by \c left
  template <typename Left, typename Right,
      typename std::enable_if<TiledArray::detail::is_numeric<Left>::value>::type* = nullptr>
  inline decltype(auto) operator*(const Left left, const Tile<Right>& right)
  { return scale(right, left); }

  /// In-place product tile operator

  /// Multiply the elements of the \c right tile by that of the \c left tile.
  /// \tparam Left The left-hand tensor type
  /// \tparam Right The right-hand tensor type
  /// \param left The left-hand tile
  /// \param right The right-hand tile
  /// \return The \c left tile, <tt>left[i] *= right[i]</tt>
  template <typename Left, typename Right>
  inline Tile<Left>& operator*=(Tile<Left>& left, const Tile<Right>& right)
  { return mult_to(left, right); }

  /// Negate tile operator

  /// \tparam Arg The argument tensor type
  /// \param arg The argument tile
  /// \return A negated copy of \c arg
  template <typename Arg>
  inline decltype(auto) operator-(const Tile<Arg>& arg)
  { return neg(arg); }

  /// Permute tile operator

  /// \tparam Arg The argument tensor type
  /// \param perm The permutation to be applied to \c arg
  /// \param arg The argument tile
  /// \return A permuted copy of \c arg
  template <typename Arg>
  inline decltype(auto) operator*(const Permutation& perm, Tile<Arg> const arg)
  { return permute(arg, perm); }

  /// Tile output stream operator

  /// \tparam T The tensor type
  /// \param os The output stream
  /// \param tile The tile to be printted
  /// \return The modified output stream
  template <typename T>
  inline std::ostream &operator<<(std::ostream &os, const Tile<T>& tile) {
    os << tile.tensor();
    return os;
  }

  /** @}*/

}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_H__INCLUDED
