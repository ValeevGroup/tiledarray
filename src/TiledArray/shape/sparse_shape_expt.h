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
 *  Justus Calvin and Edward Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  sparse_shape_expt.h
 *  Jul 9, 2013
 *
 */

#ifndef TILEDARRAY_SHAPE_SPARSE_SHAPE_EXPT_H__INCLUDED
#define TILEDARRAY_SHAPE_SPARSE_SHAPE_EXPT_H__INCLUDED

#include <TiledArray/tensor.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/tensor/shift_wrapper.h>
#include <TiledArray/tensor/tensor_interface.h>

namespace TiledArray {

  namespace experimental {

  /// Sparse shape describes which tiles of Array are numerically zero.

  /// Shape plays two roles: keeping track of (non)zero tiles and
  /// estimating shapes of results of arithmetic expressions.
  ///
  /// Compared to TiledArray::SparseShape,
  /// which defined as nonzero blocks whose <em>per-element</em> Frobenius norms
  /// exceeded a threshold, this Shape defines as nonzero blocks with Frobenius norm
  /// greater than the threshold.
  ///
  /// \tparam T The sparse element value type; can be a real floating-point type or bool.
  template <typename T>
  class SparseShape {
  public:
    typedef SparseShape<T> SparseShape_; ///< This object type
    typedef T value_type; ///< The norm value type
    typedef typename Tensor<value_type>::size_type size_type; ///< Size type

  private:

    // T must be a real floating-point type OR bool
    static_assert(std::is_floating_point<T>::value || std::is_same<T,bool>::value,
        "SparseShape template type T must be a floating-point type or bool");

    Tensor<value_type> tile_norms_; ///< Tile magnitude data
    size_type zero_tile_count_; ///< Number of zero tiles
    value_type threshold_; ///< The zero threshold
    static value_type default_threshold_; ///< The zero threshold

    SparseShape(const Tensor<T>& tile_norms,
                const size_type zero_tile_count) :
      tile_norms_(tile_norms),
      zero_tile_count_(zero_tile_count)
    { }

    /// Screens out blocks with norm < threshold_
    void truncate() {
      if (std::is_same<T,bool>::value) // nothing to do if bool-based shape
        return;

      const value_type threshold = threshold_;
      const unsigned int dim = tile_norms_.range().rank();
      size_type zero_tile_count = 0ul;

      auto truncate_op = [threshold, &zero_tile_count] (value_type& norm) {
        TA_ASSERT(norm >= value_type(0));
        if(norm < threshold) {
          norm = value_type(0);
          ++zero_tile_count;
        }
      };

      math::inplace_vector_op(truncate_op, tile_norms_.size(), tile_norms_.data());

      zero_tile_count_ = zero_tile_count;
    }

  public:

    /// Default constructor

    /// Construct a shape with no data.
    SparseShape() : tile_norms_(), zero_tile_count_(0ul) { }

    /// Constructor

    /// This constructor will normalize the tile norm, where the normalization
    /// constant for each tile is the inverse of the number of elements in the
    /// tile.
    /// \param tile_norms The Frobenius norm of tiles
    SparseShape(const Tensor<value_type>& tile_norms) :
      tile_norms_(tile_norms.clone()),
      zero_tile_count_(0ul)
    {
      TA_ASSERT(! tile_norms_.empty());

      truncate();
    }

    /// Collective constructor

    /// This constructor will sum the tile_norms data across all processes (via
    /// an all reduce). After the norms have been summed, the shape will be truncated
    /// \param world The world where the shape will live
    /// \param tile_norms The Frobenius norm of tiles
    /// \param trange The tiled range of the tensor
    SparseShape(World& world, const Tensor<value_type>& tile_norms) :
      tile_norms_(tile_norms.clone()),
      zero_tile_count_(0ul)
    {
      TA_ASSERT(! tile_norms_.empty());

      // Do global initialization of norm data
      world.gop.sum(tile_norms_.data(), tile_norms_.size());

      truncate();
    }

    /// Copy constructor

    /// Shallow copy of \c other.
    /// \param other The other shape object to be copied
    SparseShape(const SparseShape<T>& other) :
      tile_norms_(other.tile_norms_),
      zero_tile_count_(other.zero_tile_count_)
    { }

    /// Copy assignment operator

    /// Shallow copy of \c other.
    /// \param other The other shape object to be copied
    /// \return A reference to this object.
    SparseShape<T>& operator=(const SparseShape<T>& other) {
      tile_norms_ = other.tile_norms_;
      zero_tile_count_ = other.zero_tile_count_;
      return *this;
    }

    /// Validate shape range

    /// \return \c true when range matches the range of this shape
    bool validate(const Range& range) const {
      if(tile_norms_.empty())
        return false;
      return (range == tile_norms_.range());
    }

    /// Check if tile is numerically zero

    /// \tparam Index The type of the index
    /// \return false
    template <typename Index>
    bool is_zero(const Index& i) const {
      TA_ASSERT(! tile_norms_.empty());
      return tile_norms_[i] < threshold_;
    }

    /// Check density

    /// \return true
    static constexpr bool is_dense() { return false; }

    /// Sparsity of the shape

    /// \return The fraction of tiles that are zero.
    float sparsity() const {
      TA_ASSERT(! tile_norms_.empty());
      return float(zero_tile_count_) / float(tile_norms_.size());
    }

    /// Threshold accessor

    /// \return The current threshold
    value_type threshold() const { return threshold_; }

    /// Default threshold accessor

    /// \return The current default threshold
    static value_type default_threshold() { return default_threshold_; }

    /// Set default threshold to \c thresh

    /// \param thresh The new default threshold
    static void default_threshold(const value_type thresh) { default_threshold_ = thresh; }

    /// Tile norm accessor

    /// \tparam Index The index type
    /// \param index The index of the tile norm to retrieve
    /// \return The norm of the tile at \c index
    template <typename Index>
    value_type operator[](const Index& index) const {
      TA_ASSERT(! tile_norms_.empty());
      return tile_norms_[index];
    }

    /// Data accessor

    /// \return A reference to the \c Tensor object that stores shape data
    const Tensor<value_type>& data() const { return tile_norms_; }

    /// Initialization check

    /// \return \c true when this shape has been initialized.
    bool empty() const { return tile_norms_.empty(); }

    /// Update sub-block of shape

    /// Update a sub-block shape information with another shape object.
    /// \tparam Index The bound index type
    /// \param lower_bound The lower bound of the sub-block to be updated
    /// \param upper_bound The upper bound of the sub-block to be updated
    /// \param other The shape that will be used to update the sub-block
    /// \return A new sparse shape object where the specified sub-block contains the data
    /// result_tile_norms of \c other.
    template <typename Index>
    SparseShape update_block(const Index& lower_bound, const Index& upper_bound,
                             const SparseShape& other)
    {
      Tensor<value_type> result_tile_norms = tile_norms_.clone();

      auto result_tile_norms_blk = result_tile_norms.block(lower_bound, upper_bound);
      const value_type threshold = threshold_;
      size_type zero_tile_count = zero_tile_count_;
      result_tile_norms_blk.inplace_binary(other.tile_norms_,
          [threshold,&zero_tile_count] (value_type& l, const value_type r) {
            // Update the zero tile count for the result
            if((l < threshold) && (r >= threshold))
              ++zero_tile_count;
            else if((l >= threshold) && (r < threshold))
              --zero_tile_count;

            // Update the tile norm value
            l = r;
          });

      return SparseShape_(result_tile_norms, zero_tile_count);
    }

  public:
    /// Create a copy of a sub-block of the shape

    /// \tparam Index The upper and lower bound array type
    /// \param lower_bound The lower bound of the sub-block
    /// \param upper_bound The upper bound of the sub-block
    template <typename Index>
    SparseShape block(const Index& lower_bound, const Index& upper_bound) const {
      // Copy the data from arg to result
      const value_type threshold = threshold_;
      size_type zero_tile_count = 0ul;
      auto copy_op = [threshold,&zero_tile_count] (value_type& restrict result,
          const value_type arg)
      {
        result = arg;
        if(arg < threshold)
          ++zero_tile_count;
      };


      // Construct the result norms tensor
      TensorConstView<value_type> block_view =
          tile_norms_.block(lower_bound, upper_bound);
      Tensor<value_type> result_norms((Range(block_view.range().extent())));
      result_norms.inplace_binary(shift(block_view), copy_op);

      return SparseShape(result_norms, zero_tile_count);
    }


    /// Create a scaled sub-block of the shape

    /// \tparam Index The upper and lower bound array type
    /// \param lower_bound The lower bound of the sub-block
    /// \param upper_bound The upper bound of the sub-block
    template <typename Index>
    SparseShape block(const Index& lower_bound, const Index& upper_bound,
        const value_type factor) const
    {
      // Copy the data from arg to result
      const value_type threshold = threshold_;
      size_type zero_tile_count = 0ul;
      auto copy_op = [factor,threshold,&zero_tile_count] (value_type& restrict result,
              const value_type arg)
      {
        result = arg * factor;
        if(result < threshold) {
          ++zero_tile_count;
          result = value_type(0);
        }
      };

      // Construct the result norms tensor
      TensorConstView<value_type> block_view =
          tile_norms_.block(lower_bound, upper_bound);
      Tensor<value_type> result_norms((Range(block_view.range().extent())));
      result_norms.inplace_binary(shift(block_view), copy_op);

      return SparseShape(result_norms, zero_tile_count);
    }

    /// Create a copy of a sub-block of the shape

    /// \param lower_bound The lower bound of the sub-block
    /// \param upper_bound The upper bound of the sub-block
    template <typename Index>
    SparseShape block(const Index& lower_bound, const Index& upper_bound,
        const Permutation& perm) const
    {
      return block(lower_bound, upper_bound).perm(perm);
    }


    /// Create a copy of a sub-block of the shape

    /// \param lower_bound The lower bound of the sub-block
    /// \param upper_bound The upper bound of the sub-block
    template <typename Index>
    SparseShape block(const Index& lower_bound, const Index& upper_bound,
        const value_type factor, const Permutation& perm) const
    {
      return block(lower_bound, upper_bound, factor).perm(perm);
    }

    /// Create a permuted shape of this shape

    /// \param perm The permutation to be applied
    /// \return A new, permuted shape
    SparseShape_ perm(const Permutation& perm) const {
      return SparseShape_(tile_norms_.permute(perm),
          zero_tile_count_);
    }

    /// Scale shape

    /// Construct a new scaled shape as:
    /// \f[
    /// {(\rm{result})}_{ij...} = |(\rm{factor})| (\rm{this})_{ij...}
    /// \f]
    /// \param factor The scaling factor
    /// \return A new, scaled shape
    SparseShape_ scale(const value_type factor) const {
      TA_ASSERT(! tile_norms_.empty());
      const value_type threshold = threshold_;
      const value_type abs_factor = std::abs(factor);
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count, abs_factor] (value_type value) {
        value *= abs_factor;
        if(value < threshold) {
          value = value_type(0);
          ++zero_tile_count;
        }
        return value;
      };

      Tensor<value_type> result_tile_norms = tile_norms_.unary(op);

      return SparseShape_(result_tile_norms, zero_tile_count);
    }

    /// Scale and permute shape

    /// Compute a new scaled shape is computed as:
    /// \f[
    /// {(\rm{result})}_{ji...} = \rm{perm}(j,i) |(\rm{factor})| (\rm{this})_{ij...}
    /// \f]
    /// \param factor The scaling factor
    /// \param perm The permutation that will be applied to this tensor.
    /// \return A new, scaled-and-permuted shape
    SparseShape_ scale(const value_type factor, const Permutation& perm) const {
      TA_ASSERT(! tile_norms_.empty());
      const value_type threshold = threshold_;
      const value_type abs_factor = std::abs(factor);
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count, abs_factor] (value_type value) {
        value *= abs_factor;
        if(value < threshold) {
          value = value_type(0);
          ++zero_tile_count;
        }
        return value;
      };

      Tensor<value_type> result_tile_norms = tile_norms_.unary(op, perm);

      return SparseShape_(result_tile_norms,
                          zero_tile_count);
    }

    /// Add shapes

    /// Construct a new sum of shapes as:
    /// \f[
    /// {(\rm{result})}_{ij...} = (\rm{this})_{ij...} + (\rm{other})_{ij...}
    /// \f]
    /// \param other The shape to be added to this shape
    /// \return A sum of shapes
    SparseShape_ add(const SparseShape_& other) const {
      TA_ASSERT(! tile_norms_.empty());
      const value_type threshold = threshold_;
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count] (value_type left,
          const value_type right)
      {
        left += right;
        if(left < threshold) {
          left = value_type(0);
          ++zero_tile_count;
        }
        return left;
      };

      Tensor<value_type> result_tile_norms =
          tile_norms_.binary(other.tile_norms_, op);

      return SparseShape_(result_tile_norms, zero_tile_count);
    }

    /// Add and permute shapes

    /// Construct a new sum of shapes as:
    /// \f[
    /// {(\rm{result})}_{ji...} = \rm{perm}(i,j) (\rm{this})_{ij...} + (\rm{other})_{ij...}
    /// \f]
    /// \param other The shape to be added to this shape
    /// \param perm The permutation that is applied to the result
    /// \return A new, scaled shape
    SparseShape_ add(const SparseShape_& other, const Permutation& perm) const {
      TA_ASSERT(! tile_norms_.empty());
      const value_type threshold = threshold_;
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count] (value_type left,
          const value_type right)
      {
        left += right;
        if(left < threshold) {
          left = value_type(0);
          ++zero_tile_count;
        }
        return left;
      };

      Tensor<value_type> result_tile_norms =
          tile_norms_.binary(other.tile_norms_, op, perm);

      return SparseShape_(result_tile_norms,
                          zero_tile_count);
    }

    /// Add and scale shapes

    /// Construct a new sum of shapes as:
    /// \f[
    /// {(\rm{result})}_{ij...} = |(\rm{factor})| ((\rm{this})_{ij...} + (\rm{other})_{ij...})
    /// \f]
    /// \param other The shape to be added to this shape
    /// \param factor The scaling factor
    /// \return A scaled sum of shapes
    SparseShape_ add(const SparseShape_& other, value_type factor) const {
      TA_ASSERT(! tile_norms_.empty());
      const value_type threshold = threshold_;
      const value_type abs_factor = std::abs(factor);
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count, abs_factor] (value_type left,
          const value_type right)
      {
        left += right;
        left *= abs_factor;
        if(left < threshold) {
          left = value_type(0);
          ++zero_tile_count;
        }
        return left;
      };

      Tensor<value_type> result_tile_norms =
          tile_norms_.binary(other.tile_norms_, op);

      return SparseShape_(result_tile_norms, zero_tile_count);
    }

    /// Add, scale, and permute shapes

    /// Construct a new sum of shapes as:
    /// \f[
    /// {(\rm{result})}_{ij...} = |(\rm{factor})| ((\rm{this})_{ij...} + (\rm{other})_{ij...})
    /// \f]
    /// \param other The shape to be added to this shape
    /// \param factor The scaling factor
    /// \param perm The permutation that is applied to the result
    /// \return A scaled and permuted sum of shapes
    SparseShape_ add(const SparseShape_& other, const value_type factor,
        const Permutation& perm) const
    {
      TA_ASSERT(! tile_norms_.empty());
      const value_type threshold = threshold_;
      const value_type abs_factor = std::abs(factor);
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count, abs_factor]
                 (value_type left, const value_type right)
      {
        left += right;
        left *= abs_factor;
        if(left < threshold) {
          left = value_type(0);
          ++zero_tile_count;
        }
        return left;
      };

      Tensor<value_type> result_tile_norms =
          tile_norms_.binary(other.tile_norms_, op, perm);

      return SparseShape_(result_tile_norms,
          zero_tile_count);
    }

    SparseShape_ subt(const SparseShape_& other) const {
      return add(other);
    }

    SparseShape_ subt(const SparseShape_& other, const Permutation& perm) const {
      return add(other, perm);
    }

    SparseShape_ subt(const SparseShape_& other, const value_type factor) const {
      return add(other, factor);
    }

    SparseShape_ subt(const SparseShape_& other, const value_type factor,
        const Permutation& perm) const
    {
      return add(other, factor, perm);
    }

  public:

    SparseShape_ mult(const SparseShape_& other) const {
      TA_ASSERT(! tile_norms_.empty());
      auto result_tile_norms = tile_norms_.mult(other.tile_norms_);
      return SparseShape_(result_tile_norms);
    }

    SparseShape_ mult(const SparseShape_& other, const Permutation& perm) const {
      // TODO: Optimize this function so that the tensor arithmetic and
      // scale_by_size operations are performed in one step instead of two.

      TA_ASSERT(! tile_norms_.empty());
      auto result_tile_norms = tile_norms_.mult(other.tile_norms_, perm);
      return SparseShape_(result_tile_norms);
    }

    SparseShape_ mult(const SparseShape_& other, const value_type factor) const {
      // TODO: Optimize this function so that the tensor arithmetic and
      // scale_by_size operations are performed in one step instead of two.

      TA_ASSERT(! tile_norms_.empty());
      auto result_tile_norms = tile_norms_.mult(other.tile_norms_, std::abs(factor));
      return SparseShape_(result_tile_norms);
    }

    SparseShape_ mult(const SparseShape_& other, const value_type factor,
        const Permutation& perm) const
    {
      TA_ASSERT(! tile_norms_.empty());
      auto result_tile_norms = tile_norms_.mult(other.tile_norms_, std::abs(factor), perm);
      return SparseShape_(result_tile_norms);
    }

    SparseShape_ gemm(const SparseShape_& other, value_type factor,
        const math::GemmHelper& gemm_helper) const
    {
      TA_ASSERT(! tile_norms_.empty());

      factor = std::abs(factor);
      const value_type threshold = threshold_;
      size_type zero_tile_count = 0ul;
      integer M = 0, N = 0, K = 0;
      gemm_helper.compute_matrix_sizes(M, N, K, tile_norms_.range(), other.tile_norms_.range());

      // Compute the number of inner ranks
      const unsigned int k_rank = gemm_helper.left_inner_end() - gemm_helper.left_inner_begin();

      assert(false); // not yet implemented
    }

    SparseShape_ gemm(const SparseShape_& other, const value_type factor,
                      const math::GemmHelper& gemm_helper, const Permutation& perm) const
    {
      return gemm(other, factor, gemm_helper).perm(perm);
    }

  }; // class SparseShape

  // Static member initialization
  template <typename T>
  typename SparseShape<T>::value_type SparseShape<T>::default_threshold_ = std::numeric_limits<T>::epsilon();

  } // namespace experimental
} // namespace TiledArray

#endif // TILEDARRAY_SHAPE_SPARSE_SHAPE_EXPT_H__INCLUDED
