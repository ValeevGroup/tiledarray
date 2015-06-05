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
 *  dense_shape.h
 *  Jul 9, 2013
 *
 */

#ifndef TILEDARRAY_DENSE_SHAPE_H__INCLUDED
#define TILEDARRAY_DENSE_SHAPE_H__INCLUDED

namespace madness {
  class World;
} // namespace

namespace TiledArray {

  // Forward declarations
  namespace expressions {
    class VariableList;
  }  // namespace expressions
  namespace math {
    class GemmHelper;
  } // namespace math
  class Range;
  class Permutation;
  using madness::World;


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
    static void collective_init(World&) { }

    /// Validate shape range

    /// \return \c true when range matches the range of this shape
    static constexpr bool validate(const Range&) { return true; }

    /// Check that a tile is zero

    /// \tparam Index The type of the index
    /// \return false
    template <typename Index>
    static constexpr bool is_zero(const Index&) { return false; }

    /// Check density

    /// \return true
    static constexpr bool is_dense() { return true; }


    /// Sparsity fraction

    /// \return The fraction of tiles that are zero tiles.
    static constexpr float sparsity() { return 0.0f; }


    /// Check if the shape is empty (uninitialized)

    /// \return Always \c false
    static constexpr bool empty() { return false; }


    template <typename Index>
    static DenseShape block(const Index&, const Index&) const { return DenseShape(); }

    static DenseShape perm(const Permutation&) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape scale(const Scalar) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape scale(const Scalar, const Permutation&) { return DenseShape(); }

    static DenseShape add(const DenseShape&) { return DenseShape(); }

    static DenseShape add(const DenseShape&, const Permutation&) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape add(const DenseShape&, const Scalar) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape add(const DenseShape&, const Scalar, const Permutation&) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape add(const Scalar) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape add(const Scalar, const Permutation&) { return DenseShape(); }

    static DenseShape subt(const DenseShape&) { return DenseShape(); }

    static DenseShape subt(const DenseShape&, const Permutation&) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape subt(const DenseShape&, const Scalar) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape subt(const DenseShape&, const Scalar, const Permutation&) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape subt(const Scalar) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape subt(const Scalar, const Permutation&) { return DenseShape(); }

    static DenseShape mult(const DenseShape&) { return DenseShape(); }

    static DenseShape mult(const DenseShape&, const Permutation&) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape mult(const DenseShape&, const Scalar) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape mult(const DenseShape&, const Scalar, const Permutation&) { return DenseShape(); }

    template <typename Scalar>
    static DenseShape gemm(const DenseShape&, const Scalar, const math::GemmHelper&)
    { return DenseShape(); }

    template <typename Scalar>
    static DenseShape gemm(const DenseShape&, const Scalar, const math::GemmHelper&, const Permutation&)
    { return DenseShape(); }

  }; // class DenseShape

} // namespace TiledArray

#endif // TILEDARRAY_DENSE_SHAPE_H__INCLUDED
