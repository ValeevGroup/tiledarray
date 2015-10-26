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

#include <TiledArray/tensor.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/val_array.h>
#include <TiledArray/tensor/shift_wrapper.h>
#include <TiledArray/tensor/tensor_interface.h>

namespace TiledArray {

  /// Arbitrary sparse shape

  /// Sparse shape uses a \c Tensor of Frobenius norms to estimate the magnitude
  /// of the data contained in tiles of an Array object. Because tiles may have
  /// an arbitrary size, the norm data is normalized, internally, by dividing
  /// the norms by the number of elements in each tile.
  /// \f[
  /// {\rm{shape}}_{ij...} = \frac{\|A_{ij...}\|}{N_i N_j ...}
  /// \f]
  /// where \f$ij...\f$ are tile indices, \f$\|A_{ij}\|\f$ is norm of tile
  /// \f$ij...\f$, and \f$N_i N_j ...\f$ is the product of tile \f$ij...\f$ in
  /// each dimension.
  /// \tparam T The sparse element value type
  template <typename T>
  class SparseShape {
  public:
    typedef SparseShape<T> SparseShape_; ///< This object type
    typedef T value_type; ///< The norm value type
    typedef typename Tensor<value_type>::size_type size_type; ///< Size type

  private:

    // T must be a numeric type
    static_assert(std::is_floating_point<T>::value,
        "SparseShape template type T must be a floating point type");

    // Internal typedefs
    typedef detail::ValArray<value_type> vector_type;

    Tensor<value_type> tile_norms_; ///< Tile magnitude data
    std::shared_ptr<vector_type> size_vectors_; ///< Tile volume data
    size_type zero_tile_count_; ///< Number of zero tiles
    static value_type threshold_; ///< The zero threshold

    template <typename Op>
    static vector_type
    recursive_outer_product(const vector_type* const size_vectors,
        const unsigned int dim, const Op& op)
    {
      vector_type result;

      if(dim == 1u) {
        // Construct a modified copy of size_vector[0]
        result = op(*size_vectors);
      } else {
        // Compute split the range and compute the outer products
        const unsigned int middle = (dim >> 1u) + (dim & 1u);
        const vector_type left = recursive_outer_product(size_vectors, middle, op);
        const vector_type right = recursive_outer_product(size_vectors + middle, dim - middle, op);

        // Compute the outer product of left and right

        result = vector_type(left.size() * right.size());
        result.outer_fill(left, right,
            [] (const value_type left, const value_type right) { return left * right; });
      }

      return result;
    }


    /// Normalize tile norms

    /// This function will divide each norm by the number of elements in the
    /// tile. If the normalized norm is less than threshold, the value is set to
    /// zero.
    void normalize() {
      const value_type threshold = threshold_;
      const unsigned int dim = tile_norms_.range().rank();
      const vector_type* restrict const size_vectors = size_vectors_.get();
      size_type zero_tile_count = 0ul;

      if(dim == 1u) {
        auto normalize_op = [threshold, &zero_tile_count] (value_type& norm, const value_type size) {
          TA_ASSERT(norm >= value_type(0));
          norm /= size;
          if(norm < threshold) {
            norm = value_type(0);
            ++zero_tile_count;
          }
        };

        // This is the easy case where the data is a vector and can be
        // normalized directly.
        math::inplace_vector_op(normalize_op, size_vectors[0].size(), tile_norms_.data(),
            size_vectors[0].data());

      } else {
        // Here the normalization constants are computed and multiplied by the
        // norm data using a recursive, outer-product algorithm. This is done to
        // minimize temporary memory requirements, memory bandwidth, and work.

        auto inv_vec_op = [] (const vector_type& size_vector) {
          return vector_type(size_vector,
              [] (const value_type size) { return value_type(1) / size; });
        };

        // Compute the left and right outer products
        const unsigned int middle = (dim >> 1u) + (dim & 1u);
        const vector_type left = recursive_outer_product(size_vectors, middle, inv_vec_op);
        const vector_type right = recursive_outer_product(size_vectors + middle, dim - middle, inv_vec_op);

        auto normalize_op = [threshold, &zero_tile_count] (value_type& norm,
            const value_type x, const value_type y)
        {
          TA_ASSERT(norm >= value_type(0));
          norm *= x * y;
          if(norm < threshold) {
            norm = value_type(0);
            ++zero_tile_count;
          }
        };

        math::outer(left.size(), right.size(), left.data(), right.data(),
            tile_norms_.data(), normalize_op);
      }

      zero_tile_count_ = zero_tile_count;
    }

    static std::shared_ptr<vector_type>
    initialize_size_vectors(const TiledRange& trange) {
      // Allocate memory for size vectors
      const unsigned int dim = trange.tiles().rank();
      std::shared_ptr<vector_type> size_vectors(new vector_type[dim],
          std::default_delete<vector_type[]>());

      // Initialize the size vectors
      for(unsigned int i = 0ul; i != dim; ++i) {
        const size_type n = trange.data()[i].tiles().second - trange.data()[i].tiles().first;

        size_vectors.get()[i] = vector_type(n, & (* trange.data()[i].begin()),
            [] (const TiledRange1::range_type& tile)
            { return value_type(tile.second - tile.first); });
      }

      return size_vectors;
    }

    std::shared_ptr<vector_type> perm_size_vectors(const Permutation& perm) const {
      const unsigned int n = tile_norms_.range().rank();

      // Allocate memory for the contracted size vectors
      std::shared_ptr<vector_type> result_size_vectors(new vector_type[n],
          std::default_delete<vector_type[]>());

      // Initialize the size vectors
      for(unsigned int i = 0u; i < n; ++i) {
        const unsigned int perm_i = perm[i];
        result_size_vectors.get()[perm_i] = size_vectors_.get()[i];
      }

      return result_size_vectors;
    }

    SparseShape(const Tensor<T>& tile_norms, const std::shared_ptr<vector_type>& size_vectors,
        const size_type zero_tile_count) :
      tile_norms_(tile_norms), size_vectors_(size_vectors),
      zero_tile_count_(zero_tile_count)
    { }

  public:

    /// Default constructor

    /// Construct a shape with no data.
    SparseShape() : tile_norms_(), size_vectors_(), zero_tile_count_(0ul) { }

    /// Constructor

    /// This constructor will normalize the tile norm, where the normalization
    /// constant for each tile is the inverse of the number of elements in the
    /// tile.
    /// \param tile_norms The Frobenius norm of tiles
    /// \param trange The tiled range of the tensor
    SparseShape(const Tensor<value_type>& tile_norms, const TiledRange& trange) :
      tile_norms_(tile_norms.clone()), size_vectors_(initialize_size_vectors(trange)),
      zero_tile_count_(0ul)
    {
      TA_ASSERT(! tile_norms_.empty());
      TA_ASSERT(tile_norms_.range() == trange.tiles());

      normalize();
    }

    /// Collective constructor

    /// This constructor will sum the tile_norms data across all processes (via
    /// an all reduce). After the norms have been summed, it will be normalized.
    /// The normalization constant for each tile is the inverse of the number of
    /// elements in the tile.
    /// \param world The world where the shape will live
    /// \param tile_norms The Frobenius norm of tiles
    /// \param trange The tiled range of the tensor
    SparseShape(World& world, const Tensor<value_type>& tile_norms,
        const TiledRange& trange) :
      tile_norms_(tile_norms.clone()), size_vectors_(initialize_size_vectors(trange)),
      zero_tile_count_(0ul)
    {
      TA_ASSERT(! tile_norms_.empty());
      TA_ASSERT(tile_norms_.range() == trange.tiles());

      // Do global initialization of norm data
      world.gop.sum(tile_norms_.data(), tile_norms_.size());

      normalize();
    }

    /// Copy constructor

    /// Shallow copy of \c other.
    /// \param other The other shape object to be copied
    SparseShape(const SparseShape<T>& other) :
      tile_norms_(other.tile_norms_), size_vectors_(other.size_vectors_),
      zero_tile_count_(other.zero_tile_count_)
    { }

    /// Copy assignment operator

    /// Shallow copy of \c other.
    /// \param other The other shape object to be copied
    /// \return A reference to this object.
    SparseShape<T>& operator=(const SparseShape<T>& other) {
      tile_norms_ = other.tile_norms_;
      size_vectors_ = other.size_vectors_;
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
    static value_type threshold() { return threshold_; }

    /// Set threshold to \c thresh

    /// \param thresh The new threshold
    static void threshold(const value_type thresh) { threshold_ = thresh; }

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

      return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count);
    }

  private:

    /// Create a copy of a sub-block of the shape

    /// \tparam Index The upper and lower bound array type
    /// \param lower_bound The lower bound of the sub-block
    /// \param upper_bound The upper bound of the sub-block
    template <typename Index>
    std::shared_ptr<vector_type>
    block_range(const Index& lower_bound, const Index& upper_bound) const {
      TA_ASSERT(detail::size(lower_bound) == tile_norms_.range().rank());
      TA_ASSERT(detail::size(upper_bound) == tile_norms_.range().rank());

      // Get the number dimensions of the the shape
      const auto rank = detail::size(lower_bound);
      const auto* restrict const lower = detail::data(lower_bound);
      const auto* restrict const upper = detail::data(upper_bound);

      std::shared_ptr<vector_type> size_vectors(new vector_type[rank],
          std::default_delete<vector_type[]>());

      for(auto i = 0ul; i < rank; ++i) {
        // Get the new range size
        const auto lower_i = lower[i];
        const auto upper_i = upper[i];
        const auto extent_i = upper_i - lower_i;

        // Check that the input indices are in range
        TA_ASSERT(lower_i < upper_i);
        TA_ASSERT(upper_i <= tile_norms_.range().upbound_data()[i]);

        // Compute the trange data for the result shape
        size_vectors.get()[i] = vector_type(extent_i,
            size_vectors_.get()[i].data() + lower_i,
            [=] (const size_type j) { return j - lower_i; });
      }

      return size_vectors;
    }

  public:
    /// Create a copy of a sub-block of the shape

    /// \tparam Index The upper and lower bound array type
    /// \param lower_bound The lower bound of the sub-block
    /// \param upper_bound The upper bound of the sub-block
    template <typename Index>
    SparseShape block(const Index& lower_bound, const Index& upper_bound) const {
      std::shared_ptr<vector_type> size_vectors =
          block_range(lower_bound, upper_bound);

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

      return SparseShape(result_norms, size_vectors, zero_tile_count);
    }


    /// Create a scaled sub-block of the shape

    /// \tparam Index The upper and lower bound array type
    /// \param lower_bound The lower bound of the sub-block
    /// \param upper_bound The upper bound of the sub-block
    template <typename Index>
    SparseShape block(const Index& lower_bound, const Index& upper_bound,
        const value_type factor) const
    {
      std::shared_ptr<vector_type> size_vectors =
          block_range(lower_bound, upper_bound);

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

      return SparseShape(result_norms, size_vectors, zero_tile_count);
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
      return SparseShape_(tile_norms_.permute(perm), perm_size_vectors(perm),
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

      return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count);
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

      return SparseShape_(result_tile_norms, perm_size_vectors(perm),
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

      return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count);
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

      return SparseShape_(result_tile_norms, perm_size_vectors(perm),
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

      return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count);
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

      return SparseShape_(result_tile_norms, perm_size_vectors(perm),
          zero_tile_count);
    }

    SparseShape_ add(value_type value) const {
      TA_ASSERT(! tile_norms_.empty());
      const value_type threshold = threshold_;
      size_type zero_tile_count = 0ul;

      Tensor<T> result_tile_norms(tile_norms_.range());

      value = std::abs(value);
      const unsigned int dim = tile_norms_.range().rank();
      const vector_type* restrict const size_vectors = size_vectors_.get();

      if(dim == 1u) {
        auto add_const_op = [threshold, &zero_tile_count, value] (value_type norm,
            const value_type size)
        {
          norm += value / std::sqrt(size);
          if(norm < threshold) {
            norm = 0;
            ++zero_tile_count;
          }
          return norm;
        };

        // This is the easy case where the data is a vector and can be
        // normalized directly.
        math::vector_op(add_const_op, size_vectors[0].size(), result_tile_norms.data(),
            tile_norms_.data(), size_vectors[0].data());

      } else {
        // Here the normalization constants are computed and multiplied by the
        // norm data using a recursive, outer algorithm. This is done to
        // minimize temporary memory requirements, memory bandwidth, and work.

        auto inv_sqrt_vec_op = [] (const vector_type size_vector) {
          return vector_type(size_vector,
              [] (const value_type size) { return value_type(1) / std::sqrt(size); });
        };

        // Compute the left and right outer products
        const unsigned int middle = (dim >> 1u) + (dim & 1u);
        const vector_type left = recursive_outer_product(size_vectors, middle, inv_sqrt_vec_op);
        const vector_type right = recursive_outer_product(size_vectors + middle, dim - middle, inv_sqrt_vec_op);

        math::outer_fill(left.size(), right.size(), left.data(), right.data(),
            tile_norms_.data(), result_tile_norms.data(),
            [threshold, &zero_tile_count, value] (value_type& norm,
                const value_type x, const value_type y)
            {
              norm += value * x * y;
              if(norm < threshold) {
                norm = value_type(0);
                ++zero_tile_count;
              }
            });
      }

      return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count);
    }

    SparseShape_ add(const value_type value, const Permutation& perm) const {
      // TODO: Optimize this function so it does the permutation at the same
      // time as the addition.
      return add(value).perm(perm);
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

    SparseShape_ subt(const value_type value) const {
      return add(value);
    }

    SparseShape_ subt(const value_type value, const Permutation& perm) const {
      return add(value, perm);
    }

  private:

    static size_type scale_by_size(Tensor<T>& tile_norms,
        const vector_type* restrict const size_vectors)
    {
      const unsigned int dim = tile_norms.range().rank();
      const value_type threshold = threshold_;
      size_type zero_tile_count = 0ul;

      if(dim == 1u) {
        // This is the easy case where the data is a vector and can be
        // normalized directly.
        math::inplace_vector_op(
            [threshold, &zero_tile_count] (value_type& norm, const value_type size) {
              norm *= size;
              if(norm < threshold) {
                norm = value_type(0);
                ++zero_tile_count;
              }
            },
            size_vectors[0].size(), tile_norms.data(), size_vectors[0].data());
      } else {
        // Here the normalization constants are computed and multiplied by the
        // norm data using a recursive, outer algorithm. This is done to
        // minimize temporary memory requirements, memory bandwidth, and work.

        auto noop = [](const vector_type& size_vector) -> const vector_type& {
          return size_vector;
        };

        // Compute the left and right outer products
        const unsigned int middle = (dim >> 1u) + (dim & 1u);
        const vector_type left = recursive_outer_product(size_vectors, middle, noop);
        const vector_type right = recursive_outer_product(size_vectors + middle, dim - middle, noop);

        math::outer(left.size(), right.size(), left.data(), right.data(), tile_norms.data(),
            [threshold, &zero_tile_count] (value_type& norm, const value_type x,
                const value_type y)
            {
              norm *= x * y;
              if(norm < threshold) {
                norm = value_type(0);
                ++zero_tile_count;
              }
            });
      }

      return zero_tile_count;
    }

  public:

    SparseShape_ mult(const SparseShape_& other) const {
      // TODO: Optimize this function so that the tensor arithmetic and
      // scale_by_size operations are performed in one step instead of two.

      TA_ASSERT(! tile_norms_.empty());
      Tensor<T> result_tile_norms = tile_norms_.mult(other.tile_norms_);
      const size_type zero_tile_count =
          scale_by_size(result_tile_norms, size_vectors_.get());

      return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count);
    }

    SparseShape_ mult(const SparseShape_& other, const Permutation& perm) const {
      // TODO: Optimize this function so that the tensor arithmetic and
      // scale_by_size operations are performed in one step instead of two.

      TA_ASSERT(! tile_norms_.empty());
      Tensor<T> result_tile_norms = tile_norms_.mult(other.tile_norms_, perm);
      std::shared_ptr<vector_type> result_size_vector = perm_size_vectors(perm);
      const size_type zero_tile_count =
                scale_by_size(result_tile_norms, result_size_vector.get());

      return SparseShape_(result_tile_norms, result_size_vector, zero_tile_count);
    }

    SparseShape_ mult(const SparseShape_& other, const value_type factor) const {
      // TODO: Optimize this function so that the tensor arithmetic and
      // scale_by_size operations are performed in one step instead of two.

      TA_ASSERT(! tile_norms_.empty());
      Tensor<T> result_tile_norms = tile_norms_.mult(other.tile_norms_, std::abs(factor));
      const size_type zero_tile_count =
          scale_by_size(result_tile_norms, size_vectors_.get());

      return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count);
    }

    SparseShape_ mult(const SparseShape_& other, const value_type factor,
        const Permutation& perm) const
    {
      // TODO: Optimize this function so that the tensor arithmetic and
      // scale_by_size operations are performed in one step instead of two.

      TA_ASSERT(! tile_norms_.empty());
      Tensor<T> result_tile_norms = tile_norms_.mult(other.tile_norms_, std::abs(factor), perm);
      std::shared_ptr<vector_type> result_size_vector = perm_size_vectors(perm);
      const size_type zero_tile_count =
          scale_by_size(result_tile_norms, result_size_vector.get());

      return SparseShape_(result_tile_norms, result_size_vector, zero_tile_count);
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

      // Allocate memory for the contracted size vectors
      std::shared_ptr<vector_type> result_size_vectors(new vector_type[gemm_helper.result_rank()],
          std::default_delete<vector_type[]>());

      // Initialize the result size vectors
      unsigned int x = 0ul;
      for(unsigned int i = gemm_helper.left_outer_begin(); i < gemm_helper.left_outer_end(); ++i, ++x)
        result_size_vectors.get()[x] = size_vectors_.get()[i];
      for(unsigned int i = gemm_helper.right_outer_begin(); i < gemm_helper.right_outer_end(); ++i, ++x)
        result_size_vectors.get()[x] = other.size_vectors_.get()[i];

      // Compute the number of inner ranks
      const unsigned int k_rank = gemm_helper.left_inner_end() - gemm_helper.left_inner_begin();

      // Construct the result norm tensor
      Tensor<value_type> result_norms(gemm_helper.make_result_range<typename Tensor<T>::range_type>(
          tile_norms_.range(), other.tile_norms_.range()), 0);

      if(k_rank > 0u) {

        // Compute size vector
        const vector_type k_sizes =
            recursive_outer_product(size_vectors_.get() + gemm_helper.left_inner_begin(),
                k_rank, [] (const vector_type& size_vector) -> const vector_type&
                { return size_vector; });

        // TODO: Make this faster. It can be done without using temporaries
        // for the arguments, but requires a custom matrix multiply.

        Tensor<value_type> left(tile_norms_.range());
        const size_type mk = M * K;
        auto left_op = [] (const value_type left, const value_type right)
            { return left * right; };
        for(size_type i = 0ul; i < mk; i += K)
          math::vector_op(left_op, K, left.data() + i,
              tile_norms_.data() + i, k_sizes.data());

        Tensor<value_type> right(other.tile_norms_.range());
        for(integer i = 0ul, k = 0; k < K; i += N, ++k) {
          const value_type factor = k_sizes[k];
          auto right_op = [=] (const value_type arg) { return arg * factor; };
          math::vector_op(right_op, N, right.data() + i, other.tile_norms_.data() + i);
        }

        result_norms = left.gemm(right, factor, gemm_helper);

        // Hard zero tiles that are below the zero threshold.
        result_norms.inplace_unary(
            [threshold, &zero_tile_count] (value_type& value) {
              if(value < threshold) {
                value = value_type(0);
                ++zero_tile_count;
              }
            });

      } else {

        // This is an outer product, so the inputs can be used directly
        math::outer_fill(M, N, tile_norms_.data(), other.tile_norms_.data(), result_norms.data(),
            [threshold, &zero_tile_count, factor] (const value_type left,
                const value_type right)
            {
              value_type norm = left * right * factor;
              if(norm < threshold) {
                norm = value_type(0);
                ++zero_tile_count;
              }
              return norm;
            });
      }

      return SparseShape_(result_norms, result_size_vectors, zero_tile_count);
    }

    SparseShape_ gemm(const SparseShape_& other, const value_type factor,
        const math::GemmHelper& gemm_helper, const Permutation& perm) const
    {
      return gemm(other, factor, gemm_helper).perm(perm);
    }

  }; // class SparseShape

  // Static member initialization
  template <typename T>
  typename SparseShape<T>::value_type SparseShape<T>::threshold_ = std::numeric_limits<T>::epsilon();

} // namespace TiledArray

#endif // TILEDARRAY_SPASE_SHAPE_H__INCLUDED
