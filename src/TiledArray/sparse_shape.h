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
#include <TiledArray/math/blas.h>
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

  /// \tparam T The sparse element value type
  template <typename T>
  class SparseShape {
  public:
    typedef T value_type;

  private:
    Tensor<value_type> data_; ///< Tile magnitude data
    value_type threshold_; ///< The zero threshold

  public:

    /// Default constructor
    SparseShape() : data_(), threshold_(0) { }

    /// Constructor

    /// \param tensor The tile magnitude data
    /// \param threshold The zero threshold
    SparseShape(const Tensor<T>& tensor, const value_type threshold) :
      data_(tensor), threshold_(std::abs(threshold))
    { }

    /// Collective constructor

    /// After initializing local data, share data by calling \c collective_init .
    /// \param world The world where the shape will live
    /// \param tensor The tile magnitude data
    /// \param threshold The zero threshold
    SparseShape(madness::World& world, const Tensor<value_type>& tensor, const value_type threshold) :
      data_(tensor), threshold_(std::abs(threshold))
    {
      collective_init(world);
    }


    /// Collective initialization shape

    /// Share data on each node with all other nodes. The data is shared using
    /// a collective, sum-reduction algorithm.
    /// \param world The world where the shape will live
    void collective_init(madness::World& world) {
      world.gop.sum(data_.data(), data_.size());
    }

    /// Validate shape range

    /// \return \c true when range matches the range of this shape
    bool validate(const Range& range) const { return (range == data_.range()); }

    /// Check that a tile is zero

    /// \tparam Index The type of the index
    /// \return false
    template <typename Index>
    bool is_zero(const Index& i) const { return (std::abs(data_[i]) < threshold_); }

    /// Check density

    /// \return true
    static bool is_dense() { return false; }

    /// Threshold accessor

    /// \return The current threshold
    value_type threshold() const { return threshold_; }

    /// Set threshold to \c thresh

    /// \param thresh The new threshold
    void threshold(const value_type thresh) { threshold_ = thresh; }

    /// Shape data accessor

    /// \return A const reference to the shape data
    const Tensor<value_type>& data() const { return data_; }

  }; // class SparseShape

  /// Permute sparse shape

  /// \tparam T The sparse element value type
  template <typename T>
  class ShapeNoop<SparseShape<T> > {
  public:
    typedef SparseShape<T> result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \param perm The permutation to be applied to \c arg
    /// \param arg The sparse shape to be permuted
    /// \return The permuted sparse shape
    result_type operator()(const Permutation& perm, const SparseShape<T>& arg) const {
      math::Noop<Tensor<T>, Tensor<T>, false> op(perm);
      return SparseShape<T>(op(arg.data()), arg.threshold());
    }
  }; // class ShapePermute<SparseShape<T> >

  /// Add sparse shapes

  /// \tparam T The sparse element value type
  template <typename T>
  class ShapeAdd<SparseShape<T>, SparseShape<T> > {
  public:
    typedef SparseShape<T> result_type;  ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left && right; }

    /// Shape evaluation operator

    /// \param left Left-hand shape
    /// \param right Right-hand shape
    /// \return The sum of \c left and \c right shapes
    result_type operator()(const Permutation& perm, const SparseShape<T>& left, const SparseShape<T>& right) const {
      math::Add<Tensor<T>, Tensor<T>, Tensor<T>, false, false> op(perm);
      return SparseShape<T>(op(left.data(), right.data()),
          left.threshold() + right.threshold());
    }
  }; // class ShapeAdd<SparseShape<T>, SparseShape<T> >

  /// Add and scale sparse shapes

  /// \tparam T The sparse element value type
  template <typename T>
  class ShapeScalAdd<SparseShape<T>, SparseShape<T> > {
  public:
    typedef SparseShape<T> result_type;  ///< Operation result type

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
    operator()(const Permutation& perm, const SparseShape<T>& left, const SparseShape<T>& right, const N factor) {
      math::ScalAdd<Tensor<T>, Tensor<T>, Tensor<T>, false, false>
          op(perm, factor);
      return SparseShape<T>(op(left.data(), right.data()),
          (left.threshold() + right.threshold()) * factor);
    }

  }; // class ShapeScalAdd<DenseShape, DenseShape>

  /// Subtract sparse shapes

  /// \tparam T The sparse element value type
  template <typename T>
  class ShapeSubt<SparseShape<T>, SparseShape<T> > {
  public:
    typedef SparseShape<T> result_type; ///< Operation result type

    /// Result tile is zero test

    /// \param left Is zero result for left-hand tile
    /// \param right Is zero result for right-hand tile
    /// \return \c true When the result is zero, otherwise \c false
    bool operator()(bool left, bool right) const { return left && right; }

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation& perm, const SparseShape<T>& left, const SparseShape<T>& right) const {
      math::Subt<Tensor<T>, Tensor<T>, Tensor<T>, false, false>
          op(perm);
      return SparseShape<T>(op(left.data(), right.data()),
          left.threshold() - right.threshold());
    }
  }; // class ShapeSubt<SparseShape<T>, SparseShape<T> >

  /// Subtract and scale sparse shapes

  /// \tparam T The sparse element value type
  template <typename T>
  class ShapeScalSubt<SparseShape<T>, SparseShape<T> > {
  public:
    typedef SparseShape<T> result_type; ///< Operation result type

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
    operator()(const Permutation& perm, const SparseShape<T>& left, const SparseShape<T>& right, const N factor) const {
      math::ScalSubt<Tensor<T>, Tensor<T>, Tensor<T>, false, false>
          op(perm, factor);
      return SparseShape<T>(op(left.data(), right.data()),
          (left.threshold() - right.threshold()) * factor);
    }
  }; // class ShapeScalSubt<SparseShape<T>, SparseShape<T> >

  /// Multiply sparse shapes

  /// \tparam T The sparse element value type
  template <typename T>
  class ShapeMult<SparseShape<T>, SparseShape<T> > {
  public:
    typedef SparseShape<T> result_type; ///< Operation result type

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
    operator()(const Permutation& perm, const SparseShape<T>& left, const SparseShape<T>& right) const {
      math::ScalMult<Tensor<T>, Tensor<T>, Tensor<T>, false, false>
          op(perm);
      return SparseShape<T>(op(left.data(), right.data()),
          left.threshold() * right.threshold());
    }
  }; // class ShapeMult<SparseShape<T>, SparseShape<T> >

  /// Multiply and scale sparse shape

  /// \tparam T The sparse element value type
  template <typename T>
  class ShapeScalMult<SparseShape<T>, SparseShape<T> > {
  public:
    typedef SparseShape<T> result_type; ///< Operation result type

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
    operator()(const Permutation& perm, const SparseShape<T>& left, const SparseShape<T>& right, const N factor) const {
      math::ScalMult<Tensor<T>, Tensor<T>, Tensor<T>, false, false>
          op(perm, factor);
      return SparseShape<T>(op(left.data(), right.data()),
          left.threshold() * right.threshold() * factor);
    }
  }; // class ShapeScalMult<SparseShape<T>, SparseShape<T> >

  /// Contract dense shape

  /// \tparam T The sparse element value type
  template <typename T>
  class ShapeCont<SparseShape<T>, SparseShape<T> > {
  public:
    typedef SparseShape<T> result_type;

    /// Shape evaluation operator

    /// \return The result dense shape
    result_type operator()(const Permutation& perm, const std::size_t m,
        const std::size_t n, const std::size_t k, const SparseShape<T>& left,
        const SparseShape<T>& right, const Range& result_range) const
    {
      Tensor<T> result(result_range, 0);
      math::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans, m, n, k, 1,
          left.data().data(), right.data().data(), 1, result.data());

      if(perm.dim() > 1u)
        result = perm ^ result;

      return SparseShape<T>(result, left.threshold() * right.threshold());
    }
  }; // class ShapeCont<SparseShape<T>, SparseShape<T> >

  /// Contract and scale sparse shapes

  /// \tparam T The sparse element value type
  template <typename T>
  class ShapeScalCont<SparseShape<T>, SparseShape<T> > {
  public:
    typedef SparseShape<T> result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation& perm, const std::size_t m,
        const std::size_t n, const std::size_t k, const SparseShape<T>& left,
        const SparseShape<T>& right, const Range& result_range, const N factor) const
    {
      Tensor<T> result(result_range, 0.0);
      math::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans, m, n, k, factor,
          left.data().data(), right.data().data(), 1.0, result.data());

      if(perm.dim() > 1u)
        result = perm ^ result;

      return SparseShape<T>(result, left.threshold() * right.threshold() * factor);
    }
  }; // ShapeScalCont<SparseShape<T>, SparseShape<T> >

  /// Scale sparse shape

  /// \tparam T The sparse element value type
  template <typename T>
  class ShapeScale<SparseShape<T> > {
  public:
    typedef DenseShape result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \tparam N Numeric scalar type
    /// \return The result dense shape
    template <typename N>
    typename madness::enable_if<detail::is_numeric<N>, result_type>::type
    operator()(const Permutation& perm, const SparseShape<T>& arg, const N factor) const {
      math::Scal<Tensor<T>, Tensor<T>, false> op(perm, factor);
      return SparseShape<T>(op(arg.data()), arg.threshold() * factor);
    }
  }; // class ShapeScale<SparseShape<T> >

  /// Negate sparse shape

  /// \tparam T The sparse element value type
  template <typename T>
  class ShapeNeg<SparseShape<T> > {
  public:
    typedef SparseShape<T> result_type; ///< Operation result type

    /// Shape evaluation operator

    /// \param arg The argument shape
    /// \return The result dense shape
    result_type operator()(const Permutation& perm, const SparseShape<T>& arg) const {
      math::Neg<Tensor<T>, Tensor<T>, false> op(perm);
      return SparseShape<T>(op(arg.data()), arg.threshold());
    }
  }; // class ShapeNeg<SparseShape<T> >

} // namespace TiledArray

#endif // TILEDARRAY_SPASE_SHAPE_H__INCLUDED
