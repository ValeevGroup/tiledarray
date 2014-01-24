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
#include <TiledArray/tile_op/contract_reduce.h>


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

    /// Permute shape

    /// \param perm The permutation to be applied
    /// \return A new, permuted shape
    SparseShape<T> perm(const Permutation& perm) const {
      return SparseShape<T>(data_.permute(perm), threshold_);
    }

    template <typename Index>
    value_type operator[](const Index& index) const { return std::abs(data_[index]); }

    /// Data accessor

    /// \return A reference to the \c Tensor object that stores shape data
    const Tensor<value_type>& data() const { return data_; }

    /// Scale shape

    /// \param factor The scaling factor
    /// \return A new, scaled shape
    SparseShape<T> scale(const value_type factor) const {
      return SparseShape<T>(data_.scale(factor), threshold_ * factor);
    }

    SparseShape<T> scale(const value_type factor, const Permutation& perm) const {
      return SparseShape<T>(data_.scale(factor, perm), threshold_ * factor);
    }

    SparseShape<T> add(const SparseShape<T>& other) const {
      return SparseShape<T>(data_.add(other.data_), threshold_ + other.threshold_);
    }

    SparseShape<T> add(const SparseShape<T>& other, const Permutation& perm) const {
      return SparseShape<T>(data_.add(other.data_, perm), threshold_ + other.threshold_);
    }

    SparseShape<T> add(const SparseShape<T>& other, const value_type factor) const {
      return SparseShape<T>(data_.add(other.data_, factor),
          (threshold_ + other.threshold_) * factor);
    }

    SparseShape<T> add(const SparseShape<T>& other, const value_type factor,
        const Permutation& perm) const
    {
      return SparseShape<T>(data_.add(other.data_, factor, perm),
          (threshold_ + other.threshold_) * factor);
    }

    SparseShape<T> add(const value_type value) {
      return SparseShape<T>(data_.add(value), threshold_ + value);
    }

    SparseShape<T> add(const value_type value, const Permutation& perm) const {
      return SparseShape<T>(data_.add(value, perm), threshold_ + value);
    }

    SparseShape<T> subt(const SparseShape<T>& other) const {
      return SparseShape<T>(data_.subt(other.data_), threshold_ - other.threshold_);
    }

    SparseShape<T> subt(const SparseShape<T>& other, const Permutation& perm) const {
      return SparseShape<T>(data_.subt(other.data_, perm), threshold_ - other.threshold_);
    }

    SparseShape<T> subt(const SparseShape<T>& other, const value_type factor) const {
      return SparseShape<T>(data_.subt(other.data_, factor),
          (threshold_ - other.threshold_) * factor);
    }

    SparseShape<T> subt(const SparseShape<T>& other, const value_type factor,
        const Permutation& perm) const
    {
      return SparseShape<T>(data_.subt(other.data_, factor, perm),
          (threshold_ - other.threshold_) * factor);
    }

    SparseShape<T> subt(const value_type value) const {
      return SparseShape<T>(data_.unary(AbsSubtConst(value)), threshold_ + value);
    }

    SparseShape<T> subt(const value_type value, const Permutation& perm) const {
      return SparseShape<T>(data_.unary(AbsSubtConst(value), perm), threshold_ + value);
    }

    SparseShape<T> mult(const SparseShape<T>& other) const {
      return SparseShape<T>(data_.mult(other.data_), threshold_ * other.threshold_);
    }

    SparseShape<T> mult(const SparseShape<T>& other, const Permutation& perm) const {
      return SparseShape<T>(data_.mult(other.data_, perm), threshold_ * other.threshold_);
    }

    SparseShape<T> mult(const SparseShape<T>& other, const value_type factor) const {
      return SparseShape<T>(data_.mult(other.data_, factor),
          (threshold_ * other.threshold_) * factor);
    }

    SparseShape<T> mult(const SparseShape<T>& other, const value_type factor,
        const Permutation& perm) const
    {
      return SparseShape<T>(data_.mult(other.data_, factor, perm),
          (threshold_ * other.threshold_) * factor);
    }

    SparseShape<T> gemm(const SparseShape<T>& other, const value_type factor,
        const math::GemmHelper& gemm_helper) const
    {
      integer m, n, k;
      gemm_helper.compute_matrix_sizes(m, n, k, data_.range(), other.data_.range());
      return SparseShape<T>(data_.gemm(other.data_, factor, gemm_helper),
          (threshold_ * other.threshold_) * value_type(k));
    }

    SparseShape<T> gemm(const SparseShape<T>& other, const value_type factor,
        const math::GemmHelper& gemm_helper, const Permutation& perm) const
    {
      integer m, n, k;
      gemm_helper.compute_matrix_sizes(m, n, k, data_.range(), other.data_.range());
      return SparseShape<T>(data_.gemm(other.data_, factor, gemm_helper).perm(perm),
          (threshold_ * other.threshold_) * value_type(k));
    }

  }; // class SparseShape

} // namespace TiledArray

#endif // TILEDARRAY_SPASE_SHAPE_H__INCLUDED
