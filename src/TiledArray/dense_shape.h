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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  dense_shape.h
 *  Jul 9, 2013
 *
 */

#ifndef TILEDARRAY_DENSE_SHAPE_H__INCLUDED
#define TILEDARRAY_DENSE_SHAPE_H__INCLUDED

#include <TiledArray/permutation.h>
#include <TiledArray/madness.h>
#include <TiledArray/type_traits.h>

namespace madness {
  class World;
} // namespace

namespace TiledArray {

  // Forward declarations
  namespace expressions {
    class VariableList;
  }  // namespace expressions

  /// Dense shape of an array

  /// Since all tiles are present in dense arrays, this shape has no data and
  /// and all checks return their logical result. The hope is that the compiler
  /// will optimize branches that use these checks.
  class DenseShape {
  public:
    // There is no data in DenseShape so the compiler generated constructors,
    // assignment operator, and destructor are OK.


    /// Collective initialization of a shape

    /// No operation since there is no data.
    static void collective_init(madness::World&) { }

    /// Check that a tile is zero

    /// \tparam Index The type of the index
    /// \return false
    template <typename Index>
    static bool is_zero(const Index&) { return false; }

    /// Check density

    /// \return true
    static bool is_dense() { return true; }
  }; // class DenseShape


  // Shape math operations

  /// Permute shape operator
  template <typename>
  class ShapeNoop;

  template <>
  class ShapeNoop<DenseShape> {
    typedef DenseShape result_type; ///< Operation result type

    result_type operator()(const Permutation&, const DenseShape&) const {
      return result_type();
    }
  }; // class ShapePermute<DenseShape>

  /// Add shapes
  template <typename, typename>
  class ShapeAdd;

  template <>
  class ShapeAdd<DenseShape, DenseShape> {
    typedef DenseShape result_type;  ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left && right; }

    /// Add shape operator
    result_type operator()(const Permutation&, const DenseShape&, const DenseShape&) const {
      return result_type();
    }
  }; // class ShapeAdd<DenseShape, DenseShape>

  /// Add and scale shapes
  template <typename, typename>
  class ShapeScalAdd;

  /// Add and scale dense shapes
  template <>
  class ShapeScalAdd<DenseShape, DenseShape> {
    typedef DenseShape result_type;  ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left && right; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation&, const DenseShape&, const DenseShape&, const N) const {
      return result_type();
    }
  }; // class ShapeScalAdd<DenseShape, DenseShape>

  /// Subtract shapes
  template <typename, typename>
  class ShapeSubt;

  template <>
  class ShapeSubt<DenseShape, DenseShape> {
    typedef DenseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left && right; }

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation&, const DenseShape&, const DenseShape&) const {
      return result_type();
    }
  }; // class ShapeSubt<DenseShape, DenseShape>

  /// Subtract and scale shapes
  template <typename, typename>
  class ShapeScalSubt;

  template <>
  class ShapeScalSubt<DenseShape, DenseShape> {
    typedef DenseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left && right; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation&, const DenseShape&, const DenseShape&, const N) const {
      return result_type();
    }
  }; // class ShapeScalSubt<DenseShape, DenseShape>

  /// Multiply shape
  template <typename, typename>
  class ShapeMult;

  template <>
  class ShapeMult<DenseShape, DenseShape> {
    typedef DenseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left || right; }

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation&, const DenseShape&, const DenseShape&) const {
      return result_type();
    }
  }; // class ShapeMult<DenseShape, DenseShape>


  /// Multiply and scale shapes
  template <typename, typename>
  class ShapeScalMult;

  /// Multiply and scale dense shape
  template <>
  class ShapeScalMult<DenseShape, DenseShape> {
    typedef DenseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left || right; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation&, const DenseShape&, const DenseShape&, const N) const {
      return result_type();
    }
  }; // class ShapeScalMult<DenseShape, DenseShape>

  /// Contract shapes
  template <typename, typename>
  class ShapeCont;

  /// Contract dense shape
  template <>
  class ShapeCont<DenseShape, DenseShape> {
    typedef DenseShape result_type;

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation&, const DenseShape&, const DenseShape&) const {
      return result_type();
    }
  }; // class ShapeCont<DenseShape, DenseShape>

  /// Contract and scale shapes
  template <typename, typename>
  class ShapeScalCont;

  /// Contract and scale dense shape
  template <>
  class ShapeScalCont<DenseShape, DenseShape> {
    typedef DenseShape result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation&, const DenseShape&, const DenseShape&, const N) const {
      return result_type();
    }
  }; // ShapeScalCont<DenseShape, DenseShape>

  /// Scale shape
  template <typename>
  class ShapeScale;

  /// Scale dense shape
  template <>
  class ShapeScale<DenseShape> {
    typedef DenseShape result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation&, const DenseShape&, const N) const {
      return result_type();
    }
  }; // class ShapeScale<DenseShape>

  /// Negate shape
  template <typename>
  class ShapeNeg;

  /// Negate dense shape
  template <>
  class ShapeNeg<DenseShape> {
    typedef DenseShape result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation&, const DenseShape&) const {
      return result_type();
    }
  }; // class ShapeNeg<DenseShape>

} // namespace TiledArray

#endif // TILEDARRAY_DENSE_SHAPE_H__INCLUDED
