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
 *  sparse_shape.h
 *  Jul 9, 2013
 *
 */

#ifndef TILEDARRAY_SPARSE_SHAPE_H__INCLUDED
#define TILEDARRAY_SPARSE_SHAPE_H__INCLUDED

#include <TiledArray/madness.h>
#include <TiledArray/tensor.h>
#include <TiledArray/dense_shape.h>
#include <TiledArray/tile_op/noop.h>
#include <TiledArray/tile_op/add.h>
#include <TiledArray/tile_op/scal_add.h>
#include <TiledArray/tile_op/subt.h>
#include <TiledArray/tile_op/scal_subt.h>
#include <TiledArray/tile_op/mult.h>
#include <TiledArray/tile_op/scal_mult.h>
#include <TiledArray/tile_op/scal.h>
#include <TiledArray/tile_op/neg.h>


namespace TiledArray {

  /// Arbitrary sparse shape
  class SparseShape {
  private:
    Tensor<float> data_; ///< Tile magnitude data
    float threshold_; ///< The zero threshold

  public:

    /// Default constructor
    SparseShape() : data_(), threshold_(0.0) { }

    /// Constructor

    /// \param tensor The tile magnitude data
    /// \param threshold The zero threshold
    SparseShape(const Tensor<float>& tensor, float threshold) :
      data_(tensor), threshold_(std::abs(threshold))
    { }


    /// Collective initialization of a shape

    /// No operation since there is no data.
    void collective_init(madness::World& world) {
      world.gop.sum(data_.data(), data_.size());
    }

    /// Check that a tile is zero

    /// \tparam Index The type of the index
    /// \return false
    template <typename Index>
    bool is_zero(const Index& i) { return (std::abs(data_[i]) < threshold_); }

    /// Check density

    /// \return true
    static bool is_dense() { return false; }

    /// Threshold accessor

    /// \return The current threshold
    float threshold() const { return threshold_; }

    /// Set threshold to \c thresh

    /// \param thresh The new threshold
    void threshold(const float thresh) { threshold_ = thresh; }

    /// Shape data accessor

    /// \return A const reference to the shape data
    const Tensor<float>& data() const { return data_; }

    /// Shape data accessor

    /// \return A const reference to the shape data
    Tensor<float>& data() { return data_; }
  }; // class SparseShape

  /// Permute sparse shape
  template <>
  class ShapeNoop<SparseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \param perm The permutation to be applied to \c arg
    /// \param arg The sparse shape to be permuted
    /// \return The permuted sparse shape
    result_type operator()(const Permutation& perm, const SparseShape& arg) const {
      math::Noop<Tensor<float>, Tensor<float>, false> op(perm);
      return SparseShape(op(arg.data()), arg.threshold());
    }
  }; // class ShapePermute<SparseShape>

  /// Add sparse shapes
  template <>
  class ShapeAdd<SparseShape, SparseShape> {
  public:
    typedef SparseShape result_type;  ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left && right; }

    /// Shape evaluation operator

    /// \param left Left-hand shape
    /// \param right Right-hand shape
    /// \return The sum of \c left and \c right shapes
    result_type operator()(const Permutation& perm, const SparseShape& left, const SparseShape& right) const {
      math::Add<Tensor<float>, Tensor<float>, Tensor<float>, false, false> op(perm);
      return SparseShape(op(left.data(), right.data()),
          left.threshold() + right.threshold());
    }
  }; // class ShapeAdd<SparseShape, SparseShape>

  /// Add a dense shape to a sparse shape
  template <>
  class ShapeAdd<DenseShape, SparseShape> {
  public:
    typedef DenseShape result_type;  ///< Operation result type

    /// Result tile is zero test

    /// \return \c false (never zero)
    bool operator()(bool, bool) const { return false; }

    /// Shape evaluation operator

    /// \return A dense shape
    result_type operator()(const Permutation&, const DenseShape&, const SparseShape&) const {
      return result_type();
    }
  }; // class ShapeAdd<DenseShape, SparseShape>

  /// Add sparse shape to a dense shape
  template <>
  class ShapeAdd<SparseShape, DenseShape> {
  public:
    typedef DenseShape result_type;  ///< Operation result type

    /// Result tile is zero test

    /// \return \c false (never zero)
    bool operator()(bool, bool) const { return false; }

    /// Shape evaluation operator
    result_type operator()(const Permutation&, const SparseShape&, const DenseShape&) const {
      return result_type();
    }
  }; // class ShapeAdd<SparseShape, DenseShape>

  /// Add and scale sparse shapes
  template <>
  class ShapeScalAdd<SparseShape, SparseShape> {
  public:
    typedef SparseShape result_type;  ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left && right; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \param left The left-hand shape
    /// \param right The right-hand shape
    /// \return The sum of \c left and \c right shapes
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation& perm, const SparseShape& left, const SparseShape& right, const N factor) {
      math::ScalAdd<Tensor<float>, Tensor<float>, Tensor<float>, false, false>
          op(perm, factor);
      return SparseShape(op(left.data(), right.data()),
          (left.threshold() + right.threshold()) * factor);
    }

  }; // class ShapeScalAdd<DenseShape, DenseShape>

  /// Add and scale a dense shape to a sparse shape
  template <>
  class ShapeScalAdd<DenseShape, SparseShape> {
  public:
    typedef DenseShape result_type;  ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return false; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation&, const DenseShape&, const SparseShape&, const N) const {
      return result_type();
    }
  }; // class ShapeScalAdd<DenseShape, SparseShape>

  /// Add and scale a sparse shape to a dense shape
  template <>
  class ShapeScalAdd<SparseShape, DenseShape> {
  public:
    typedef DenseShape result_type;  ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return false; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation&, const SparseShape&, const DenseShape&, const N) const {
      return result_type();
    }
  }; // class ShapeScalAdd<SparseShape, DenseShape>

  /// Subtract sparse shapes
  template <>
  class ShapeSubt<SparseShape, SparseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left && right; }

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation& perm, const SparseShape& left, const SparseShape& right) const {
      math::Subt<Tensor<float>, Tensor<float>, Tensor<float>, false, false>
          op(perm);
      return SparseShape(op(left.data(), right.data()),
          left.threshold() - right.threshold());
    }
  }; // class ShapeSubt<SparseShape, SparseShape>

  /// Subtract a dense shape from a sparse shape
  template <>
  class ShapeSubt<DenseShape, SparseShape> {
  public:
    typedef DenseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \return \c false (never zero)
    bool operator()(bool, bool) const { return false; }

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation&, const DenseShape&, const SparseShape&) const {
      return result_type();
    }
  }; // class ShapeSubt<DenseShape, SparseShape>

  /// Subtract sparse shape from a dense shape
  template <>
  class ShapeSubt<SparseShape, DenseShape> {
  public:
    typedef DenseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \return \c false (never zero)
    bool operator()(bool, bool) const { return false; }

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation&, const SparseShape&, const DenseShape&) const {
      return result_type();
    }
  }; // class ShapeSubt<SparseShape, DenseShape>

  /// Subtract and scale sparse shapes
  template <>
  class ShapeScalSubt<SparseShape, SparseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left && right; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \param perm The permutation that will be applied to the result shape
    /// \param left The left-hand argument shape
    /// \param right The right-hand argument shape
    /// \param factor The scaling factor that will be applied to the result shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation& perm, const SparseShape& left, const SparseShape& right, const N factor) const {
      math::ScalSubt<Tensor<float>, Tensor<float>, Tensor<float>, false, false>
          op(perm, factor);
      return SparseShape(op(left.data(), right.data()),
          (left.threshold() - right.threshold()) * factor);
    }
  }; // class ShapeScalSubt<SparseShape, SparseShape>

  /// Subtract and scale sparse shapes
  template <>
  class ShapeScalSubt<DenseShape, SparseShape> {
  public:
    typedef DenseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool, bool) const { return false; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation&, const DenseShape&, const SparseShape&, const N) const {
      return DenseShape();
    }
  }; // class ShapeScalSubt<DenseShape, DenseShape>

  /// Subtract and scale sparse shapes
  template <>
  class ShapeScalSubt<SparseShape, DenseShape> {
  public:
    typedef DenseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool, bool) const { return false; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation&, const SparseShape&, const DenseShape&, const N) const {
      return DenseShape();
    }
  }; // class ShapeScalSubt<SparseShape, DenseShape>

  /// Multiply sparse shapes
  template <>
  class ShapeMult<SparseShape, SparseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left || right; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \param perm The permutation that will be applied to the result shape
    /// \param left The left-hand argument shape
    /// \param right The right-hand argument shape
    /// \return The result sparse shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation& perm, const SparseShape& left, const SparseShape& right) const {
      math::ScalMult<Tensor<float>, Tensor<float>, Tensor<float>, false, false>
          op(perm);
      return SparseShape(op(left.data(), right.data()),
          left.threshold() * right.threshold());
    }
  }; // class ShapeMult<SparseShape, SparseShape>

  /// Multiply a dense shape by a sparse shape
  template <>
  class ShapeMult<DenseShape, SparseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left || right; }

    /// Shape evaluation operator

    /// \return The result dense shape
    inline result_type operator()(const Permutation& perm, const DenseShape&, const SparseShape& right) const {
      // Note: Here it is assume that dense shape values and threshold are equal
      // to one (1).
      math::Noop<Tensor<float>, Tensor<float>, false> op(perm);
      return SparseShape(op(right.data()), right.threshold());
    }
  }; // class ShapeMult<DenseShape, SparseShape>

  /// Multiply a sparse shape by a dense shape
  template <>
  class ShapeMult<SparseShape, DenseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left || right; }

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation& perm, const SparseShape& left, const DenseShape&) const {
      // Note: Here it is assume that dense shape values and threshold are equal
      // to one (1).
      math::Noop<Tensor<float>, Tensor<float>, false> op(perm);
      return SparseShape(op(left.data()), left.threshold());
    }
  }; // class ShapeMult<SparseShape, DenseShape>

  /// Multiply and scale sparse shape
  template <>
  class ShapeScalMult<SparseShape, SparseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

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
    operator()(const Permutation& perm, const SparseShape& left, const SparseShape& right, const N factor) const {
      math::ScalMult<Tensor<float>, Tensor<float>, Tensor<float>, false, false>
          op(perm, factor);
      return SparseShape(op(left.data(), right.data()),
          left.threshold() * right.threshold() * factor);
    }
  }; // class ShapeScalMult<SparseShape, SparseShape>

  /// Multiply and scale dense shape
  template <>
  class ShapeScalMult<DenseShape, SparseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left || right; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    inline typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation& perm, const DenseShape&, const SparseShape& right, const N factor) const {
      math::Scal<Tensor<float>, Tensor<float>, false> op(perm, factor);
      return SparseShape(op(right.data()), right.threshold() * factor);
    }
  }; // class ShapeScalMult<DenseShape, SparseShape>

  /// Multiply and scale dense shape
  template <>
  class ShapeScalMult<SparseShape, DenseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left || right; }

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    inline typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation& perm, const SparseShape& left, const DenseShape&, const N factor) const {
      math::Scal<Tensor<float>, Tensor<float>, false> op(perm, factor);
      return SparseShape(op(left.data()), left.threshold() * factor);
    }
  }; // class ShapeScalMult<SparseShape, DenseShape>

  /// Contract dense shape
  template <>
  class ShapeCont<SparseShape, SparseShape> {
  public:
    typedef SparseShape result_type;

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation& perm, const std::size_t m,
        const std::size_t n, const std::size_t k, const SparseShape& left,
        const SparseShape& right, const Range& result_range) const
    {
      Tensor<float> result(result_range, 0.0);
      math::gemm(m, n, k, 1.0, left.data().data(), right.data().data(), result.data());

      if(perm.dim() > 1u)
        result = perm ^ result;

      return SparseShape(result, left.threshold() * right.threshold());
    }
  }; // class ShapeCont<SparseShape, SparseShape>

  /// Contract dense shape
  template <>
  class ShapeCont<DenseShape, SparseShape> {
  public:
    typedef SparseShape result_type;

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation& perm, const std::size_t m,
        const std::size_t n, const std::size_t k, const DenseShape&,
        const SparseShape& right, const Range& result_range) const
    {
      Tensor<float> result(result_range);

      math::eigen_map(result.data(), m, n).rowwise() =
          math::eigen_map(right.data().data(), k, n).colwise().sum();

      if(perm.dim() > 1u)
        result = perm ^ result;

      return SparseShape(result, right.threshold());
    }
  }; // class ShapeCont<DenseShape, DenseShape>

  /// Contract dense shape
  template <>
  class ShapeCont<SparseShape, DenseShape> {
  public:
    typedef SparseShape result_type;

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation& perm, const std::size_t m,
        const std::size_t n, const std::size_t k, const SparseShape& left,
        const DenseShape&, const Range& result_range) const
    {
      Tensor<float> result(result_range);

      math::eigen_map(result.data(), m, n).colwise() =
          math::eigen_map(left.data().data(), m, k).rowwise().sum();

      if(perm.dim() > 1u)
        result = perm ^ result;

      return SparseShape(result, left.threshold());
    }
  }; // class ShapeCont<SparseShape, DenseShape>

  /// Contract and scale sparse shapes
  template <>
  class ShapeScalCont<SparseShape, SparseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation& perm, const std::size_t m,
        const std::size_t n, const std::size_t k, const SparseShape& left,
        const SparseShape& right, const Range& result_range, const N factor) const
    {
      Tensor<float> result(result_range, 0.0);
      math::gemm(m, n, k, factor, left.data().data(), right.data().data(), result.data());

      if(perm.dim() > 1u)
        result = perm ^ result;

      return SparseShape(result, left.threshold() * right.threshold() * factor);
    }
  }; // ShapeScalCont<SparseShape, SparseShape>

  /// Contract and scale sparse shapes
  template <>
  class ShapeScalCont<DenseShape, SparseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation& perm, const std::size_t m,
        const std::size_t n, const std::size_t k, const DenseShape&,
        const SparseShape& right, const Range& result_range, const N factor) const
    {
      Tensor<float> result(result_range);

      math::eigen_map(result.data(), m, n).rowwise() =
          math::eigen_map(right.data().data(), k, n).colwise().sum() * factor;

      if(perm.dim() > 1u)
        result = perm ^ result;

      return SparseShape(result, right.threshold());
    }
  }; // ShapeScalCont<SparseShape, SparseShape>

  /// Contract and scale sparse shapes
  template <>
  class ShapeScalCont<SparseShape, DenseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation& perm, const std::size_t m,
        const std::size_t n, const std::size_t k, const SparseShape& left,
        const DenseShape&, const Range& result_range, const N factor) const
    {
      Tensor<float> result(result_range);

      math::eigen_map(result.data(), m, n).colwise() =
          math::eigen_map(left.data().data(), m, k).rowwise().sum() * factor;

      if(perm.dim() > 1u)
        result = perm ^ result;

      return SparseShape(result, left.threshold());
    }
  }; // ShapeScalCont<SparseShape, SparseShape>

  /// Scale sparse shape
  template <>
  class ShapeScale<SparseShape> {
  public:
    typedef DenseShape result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation& perm, const SparseShape& arg, const N factor) const {
      math::Scal<Tensor<float>, Tensor<float>, false> op(perm, factor);
      return SparseShape(op(arg.data()), arg.threshold() * factor);
    }
  }; // class ShapeScale<SparseShape>

  /// Negate sparse shape
  template <>
  class ShapeNeg<SparseShape> {
  public:
    typedef SparseShape result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \param arg The argument shape
    /// \return The result dense shape
    result_type operator()(const Permutation& perm, const SparseShape& arg) const {
      math::Neg<Tensor<float>, Tensor<float>, false> op(perm);
      return SparseShape(op(arg.data()), arg.threshold());
    }
  }; // class ShapeNeg<SparseShape>

} // namespace TiledArray

#endif // TILEDARRAY_SPASE_SHAPE_H__INCLUDED
