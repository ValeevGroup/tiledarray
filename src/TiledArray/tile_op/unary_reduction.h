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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  unary_reduction.h
 *  Apr 2, 2014
 *
 */

#ifndef TILEDARRAY_UNARY_REDUCTION_H__INCLUDED
#define TILEDARRAY_UNARY_REDUCTION_H__INCLUDED

namespace TiledArray {
  namespace math {

    /// Tile sum reduction

    /// This reduction operation is used to sum all elements of a tile.
    /// \tparam Tile The tile type
    template <typename Tile>
    class SumReduction {
    public:
      // typedefs
      typedef typename Tile::numeric_type result_type;
      typedef Tile argument_type;

      // Reduction functions

      // Make an empty result object
      result_type operator()() const { return result_type(0); }

      // Post process the result
      const result_type& operator()(const result_type& result) const { return result; }

      // Reduce two result objects
      void operator()(result_type& result, const result_type& arg) const {
        result += arg;
      }

      // Reduce an argument
      void operator()(result_type& result, const argument_type& arg) const {
        result += arg.sum();
      }
    }; // class SumReduction

    /// Tile product reduction

    /// This reduction operation is used to multiply all elements of a tile.
    /// \tparam Tile The tile type
    template <typename Tile>
    class ProductReduction {
    public:
      // typedefs
      typedef typename Tile::numeric_type result_type;
      typedef Tile argument_type;

      // Reduction functions

      // Make an empty result object
      result_type operator()() const { return result_type(1); }

      // Post process the result
      const result_type& operator()(const result_type& result) const { return result; }

      // Reduce two result objects
      void operator()(result_type& result, const result_type& arg) const {
        result *= arg;
      }

      // Reduce an argument
      void operator()(result_type& result, const argument_type& arg) const {
        result *= arg.product();
      }

    }; // class ProductReduction

    /// Tile trace reduction

    /// This reduction operation is used to sum the hyper-diagonal elements of a
    /// tile.
    /// \tparam Tile The tile type
    template <typename Tile>
    class TraceReduction {
    public:
      // typedefs
      typedef typename Tile::value_type result_type;
      typedef Tile argument_type;

      // Reduction functions

      // Make an empty result object
      result_type operator()() const { return result_type(0); }

      // Post process the result
      const result_type& operator()(const result_type& result) const { return result; }

      // Reduce two result objects
      void operator()(result_type& result, const result_type& arg) const {
        result += arg;
      }

      // Reduce an argument
      void operator()(result_type& result, const argument_type& arg) const {
        result += arg.trace();
      }

    }; // class TraceReduction

    /// Squared norm tile reduction

    /// This reduction operation is used to sum the square of all elements of a
    /// tile.
    /// \tparam Tile The tile type
    template <typename Tile>
    class SquaredNormReduction {
    public:
      // typedefs
      typedef typename Tile::numeric_type result_type;
      typedef Tile argument_type;

      // Reduction functions

      // Make an empty result object
      result_type operator()() const { return result_type(0); }

      // Post process the result
      const result_type& operator()(const result_type& result) const { return result; }

      // Reduce two result objects
      void operator()(result_type& result, const result_type& arg) const {
        result += arg;
      }

      // Reduce an argument
      void operator()(result_type& result, const argument_type& arg) const {
        result += arg.squared_norm();
      }

    }; // class SquaredNormReduction

    /// Minimum tile reduction

    /// This reduction operation is used to find the minimum value of tiles.
    /// \tparam Tile The tile type
    template <typename Tile>
    class MinReduction {
    public:
      // typedefs
      typedef typename Tile::numeric_type result_type;
      typedef Tile argument_type;

      // Reduction functions

      // Make an empty result object
      result_type operator()() const { return std::numeric_limits<result_type>::max(); }

      // Post process the result
      const result_type& operator()(const result_type& result) const { return result; }

      // Reduce two result objects
      void operator()(result_type& result, const result_type& arg) const {
        result = std::min(result, arg);
      }

      // Reduce an argument
      void operator()(result_type& result, const argument_type& arg) const {
        result = std::min(result, arg.min());
      }

    }; // class MinReduction

    /// Maximum tile reduction

    /// This reduction operation is used to find the maximum value of tiles.
    /// \tparam Tile The tile type
    template <typename Tile>
    class MaxReduction {
    public:
      // typedefs
      typedef typename Tile::numeric_type result_type;
      typedef Tile argument_type;

      // Reduction functions

      // Make an empty result object
      result_type operator()() const { return std::numeric_limits<result_type>::min(); }

      // Post process the result
      const result_type& operator()(const result_type& result) const { return result; }

      // Reduce two result objects
      void operator()(result_type& result, const result_type& arg) const {
        result = std::max(result, arg);
      }

      // Reduce an argument
      void operator()(result_type& result, const argument_type& arg) const {
        result = std::max(result, arg.max());
      }

    }; // class MaxReduction

    /// Minimum tile reduction

    /// This reduction operation is used to find the absolute minimum value of
    /// tiles.
    /// \tparam Tile The tile type
    template <typename Tile>
    class AbsMinReduction {
    public:
      // typedefs
      typedef typename Tile::numeric_type result_type;
      typedef Tile argument_type;

      // Reduction functions

      // Make an empty result object
      result_type operator()() const { return std::numeric_limits<result_type>::max(); }

      // Post process the result
      const result_type& operator()(const result_type& result) const { return result; }

      // Reduce two result objects
      void operator()(result_type& result, const result_type& arg) const {
        result = std::min(result, arg);
      }

      // Reduce an argument
      void operator()(result_type& result, const argument_type& arg) const {
        result = std::min(result, arg.min());
      }

    }; // class AbsMinReduction

    /// Maximum tile reduction

    /// This reduction operation is used to find the absolute maximum value of
    /// tiles.
    /// \tparam Tile The tile type
    template <typename Tile>
    class AbsMaxReduction {
    public:
      // typedefs
      typedef typename Tile::numeric_type result_type;
      typedef Tile argument_type;

      // Reduction functions

      // Make an empty result object
      result_type operator()() const { return result_type(0); }

      // Post process the result
      const result_type& operator()(const result_type& result) const { return result; }

      // Reduce two result objects
      void operator()(result_type& result, const result_type& arg) const {
        result = std::max(result, arg);
      }

      // Reduce an argument
      void operator()(result_type& result, const argument_type& arg) const {
        result = std::max(result, arg.abs_max());
      }

    }; // class AbsMaxReduction

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_UNARY_REDUCTION_H__INCLUDED
