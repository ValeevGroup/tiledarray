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

#include <TiledArray/config.h>
#include <TiledArray/type_traits.h>
#include <cstdint>

namespace madness {
class World;
}  // namespace madness

namespace TiledArray {

// Forward declarations
namespace expressions {
class VariableList;
}  // namespace expressions
namespace math {
class GemmHelper;
}  // namespace math
class Range;
class Permutation;
class TiledRange;
using madness::World;

/// Dense shape of an array

/// Since all tiles are present in dense arrays, this shape has no data and
/// and all checks return their logical result. The hope is that the compiler
/// will optimize branches that use these checks.
class DenseShape {
 public:
  using index1_type = TA_1INDEX_TYPE;
  using value_type = float;

  // There is no data in DenseShape so the compiler generated constructors,
  // assignment operator, and destructor are OK.

  DenseShape() = default;
  DenseShape(const DenseShape&) = default;
  DenseShape(DenseShape&&) = default;
  DenseShape& operator=(const DenseShape&) = default;
  DenseShape& operator=(DenseShape&&) = default;
  ~DenseShape() = default;

  // Several no-op constructors are needed to make it interoperable with
  // SparseShape
  template <typename Real>
  DenseShape(Real&&, const TiledRange&) {}

  /// Collective initialization of a shape

  /// No operation since there is no data.
  static void collective_init(World&) {}

  /// Validate shape range

  /// \return \c true when range matches the range of this shape
  static constexpr bool validate(const Range&) { return true; }

  /// Check that a tile is zero

  /// \tparam Index The type of the index
  /// \return false
  template <typename Index>
  static constexpr bool is_zero(const Index&) {
    return false;
  }

  /// Check density

  /// \return true
  static constexpr bool is_dense() { return true; }

  /// Sparsity fraction

  /// \return The fraction of tiles that are zero tiles.
  static constexpr float sparsity() { return 0.0f; }

  /// Threshold accessor

  /// \return The current threshold
  static value_type threshold() { return threshold_; }

  /// Set threshold to \c thresh

  /// \param thresh The new threshold
  static void threshold(const value_type thresh) { threshold_ = thresh; }

  /// Check if the shape is empty (uninitialized)

  /// \return Always \c false
  static constexpr bool empty() { return false; }

  DenseShape mask(const DenseShape&) const { return DenseShape{}; };

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  static DenseShape update_block(const Index1&, const Index2&,
                                 const DenseShape&) {
    return DenseShape();
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  static DenseShape update_block(const std::initializer_list<Index1>&,
                                 const std::initializer_list<Index2>&,
                                 const DenseShape&) {
    return DenseShape();
  }

  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  DenseShape update_block(const PairRange& bounds,
                          const DenseShape& other) const {
    return DenseShape();
  }

  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  static DenseShape update_block(
      const std::initializer_list<std::initializer_list<Index>>&,
      const DenseShape&) {
    return DenseShape();
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  static DenseShape block(const Index1&, const Index2&) {
    return DenseShape();
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  static DenseShape block(const std::initializer_list<Index1>&,
                          const std::initializer_list<Index2>&) {
    return DenseShape();
  }

  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  DenseShape block(const PairRange& bounds) const {
    return DenseShape();
  }

  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  static DenseShape block(
      const std::initializer_list<std::initializer_list<Index>>&) {
    return DenseShape();
  }

  template <typename Index1, typename Index2, typename Scalar,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2> &&
                                        detail::is_numeric_v<Scalar>>>
  static DenseShape block(const Index1&, const Index2&, const Scalar) {
    return DenseShape();
  }

  template <typename Index1, typename Index2, typename Scalar,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2> &&
                                        detail::is_numeric_v<Scalar>>>
  static DenseShape block(const std::initializer_list<Index1>&,
                          const std::initializer_list<Index2>&, const Scalar) {
    return DenseShape();
  }

  template <typename PairRange, typename Scalar,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange> &&
                                        detail::is_numeric_v<Scalar>>>
  DenseShape block(const PairRange& bounds, const Scalar) const {
    return DenseShape();
  }

  template <typename Index, typename Scalar,
            typename = std::enable_if_t<std::is_integral_v<Index> &&
                                        detail::is_numeric_v<Scalar>>>
  static DenseShape block(
      const std::initializer_list<std::initializer_list<Index>>&,
      const Scalar) {
    return DenseShape();
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  static DenseShape block(const Index1&, const Index2&, const Permutation&) {
    return DenseShape();
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  static DenseShape block(const std::initializer_list<Index1>&,
                          const std::initializer_list<Index2>&,
                          const Permutation&) {
    return DenseShape();
  }

  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  DenseShape block(const PairRange& bounds, const Permutation&) const {
    return DenseShape();
  }

  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  static DenseShape block(
      const std::initializer_list<std::initializer_list<Index>>&,
      const Permutation&) {
    return DenseShape();
  }

  template <typename Index1, typename Index2, typename Scalar,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2> &&
                                        detail::is_numeric_v<Scalar>>>
  static DenseShape block(const Index1&, const Index2&, const Scalar,
                          const Permutation&) {
    return DenseShape();
  }

  template <typename Index1, typename Index2, typename Scalar,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2> &&
                                        detail::is_numeric_v<Scalar>>>
  static DenseShape block(const std::initializer_list<Index1>&,
                          const std::initializer_list<Index2>&, const Scalar,
                          const Permutation&) {
    return DenseShape();
  }

  template <typename PairRange, typename Scalar,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange> &&
                                        detail::is_numeric_v<Scalar>>>
  DenseShape block(const PairRange& bounds, const Scalar,
                   const Permutation&) const {
    return DenseShape();
  }

  template <typename Index, typename Scalar,
            typename = std::enable_if_t<std::is_integral_v<Index> &&
                                        detail::is_numeric_v<Scalar>>>
  static DenseShape block(
      const std::initializer_list<std::initializer_list<Index>>&, const Scalar,
      const Permutation&) {
    return DenseShape();
  }

  static DenseShape perm(const Permutation&) { return DenseShape(); }

  template <typename Scalar>
  static DenseShape scale(const Scalar) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape scale(const Scalar, const Permutation&) {
    return DenseShape();
  }

  static DenseShape add(const DenseShape&) { return DenseShape(); }

  static DenseShape add(const DenseShape&, const Permutation&) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape add(const DenseShape&, const Scalar) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape add(const DenseShape&, const Scalar, const Permutation&) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape add(const Scalar) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape add(const Scalar, const Permutation&) {
    return DenseShape();
  }

  static DenseShape subt(const DenseShape&) { return DenseShape(); }

  static DenseShape subt(const DenseShape&, const Permutation&) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape subt(const DenseShape&, const Scalar) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape subt(const DenseShape&, const Scalar, const Permutation&) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape subt(const Scalar) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape subt(const Scalar, const Permutation&) {
    return DenseShape();
  }

  static DenseShape mult(const DenseShape&) { return DenseShape(); }

  static DenseShape mult(const DenseShape&, const Permutation&) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape mult(const DenseShape&, const Scalar) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape mult(const DenseShape&, const Scalar, const Permutation&) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape gemm(const DenseShape&, const Scalar,
                         const math::GemmHelper&) {
    return DenseShape();
  }

  template <typename Scalar>
  static DenseShape gemm(const DenseShape&, const Scalar,
                         const math::GemmHelper&, const Permutation&) {
    return DenseShape();
  }

  template <typename Archive>
  void serialize(const Archive& ar) const {}
 private:
  inline static value_type threshold_ = std::numeric_limits<value_type>::epsilon();
};  // class DenseShape



constexpr inline bool operator==(const DenseShape& a, const DenseShape& b) {
  return true;
}

constexpr inline bool operator!=(const DenseShape& a, const DenseShape& b) {
  return !(a == b);
}

constexpr inline bool is_replicated(World& world, const DenseShape& t) {
  return true;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_DENSE_SHAPE_H__INCLUDED
