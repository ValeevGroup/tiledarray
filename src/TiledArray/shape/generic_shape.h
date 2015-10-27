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
 *  Justus Calvin and Edward Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  sparse_shape.h
 *  Oct 25, 2015
 *
 */

#ifndef TILEDARRAY_SHAPE_GENERIC_SHAPE_H__INCLUDED
#define TILEDARRAY_SHAPE_GENERIC_SHAPE_H__INCLUDED

#include <TiledArray/tensor.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/tensor/shift_wrapper.h>
#include <TiledArray/tensor/tensor_interface.h>
#include <TiledArray/utility/variant.h>

namespace TiledArray {

  /// A generic shape class that can describe dense, sparse(real) and sparse(bool)
  /// tensors.

  /// Shape is a runtime variant of DenseShape and SparseShape classes, with some additional
  /// simplifications.
  ///
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
  template <typename Real = float>
  class GenericShape {
  public:
    typedef GenericShape<Real> GenericShape_; ///< This object type
    typedef Real real_type; ///< The norm value type
    typedef typename Tensor<real_type>::size_type size_type; ///< Size type

  private:

    // Real must be a numeric type
    static_assert(std::is_floating_point<Real>::value,
        "GenericShape template type Real must be a floating point type");

    // type of GenericShape is a runtime parameter
    enum shape_type {uninitialized, dense, sparse_real, sparse_bool};
    shape_type shape_type_;        ///< Type of this shape (dense, sparse_real, sparse_bool)

    // applicable to any sparse
//    union tile_norm_type {
//        Tensor<real_type> _real; ///< Tile norms
//        Tensor<bool> _bool; ///< Tile exists/not
//
//        tile_norm_type() : _bool() {}
//        tile_norm_type(const Tensor<real_type>& real_norms) : _real(real_norms) {}
//        tile_norm_type(const Tensor<bool>& bool_norms) : _bool(bool_norms) {}
//        ~tile_norm_type() {}
//        void delete_real() { _real.~Tensor<real_type>(); }
//        void delete_bool() { _real.~Tensor<bool>(); }
//    };
    typedef TiledArray::detail::Variant<Tensor<real_type>, Tensor<bool>> tile_norm_type;
    tile_norm_type tile_norms_;
    size_type zero_tile_count_; ///< Number of zero tiles
    // applicable to sparse-real
    real_type threshold_; ///< Real norm threshold used to construct this
    static real_type default_threshold_; ///< Real norm threshold used to construct new sparse-real shapes

    GenericShape(const Tensor<Real>& tile_norms,
                 const size_type zero_tile_count) :
      shape_type_(sparse_real), tile_norms_{tile_norms},
      zero_tile_count_(zero_tile_count), threshold_(default_threshold_)
    { }

  public:

    /// Default constructor

    /// Construct a shape with no data.
    GenericShape() : shape_type_(uninitialized),
                     zero_tile_count_(0ul),
                     threshold_(default_threshold_) { }

    /// Constructor

    /// This constructor will normalize the tile norm, where the normalization
    /// constant for each tile is the inverse of the number of elements in the
    /// tile.
    /// \param tile_norms The Frobenius norm of tiles
    /// \param trange The tiled range of the tensor
    GenericShape(const Tensor<real_type>& tile_norms, const TiledRange& trange) :
      shape_type_(sparse_real),
      tile_norms_(tile_norms.clone()),
      zero_tile_count_(0ul),
      threshold_(default_threshold_)
    {
      TA_ASSERT(! tile_norms.empty());
      TA_ASSERT(tile_norms.range() == trange.tiles());
    }

    /// Collective constructor

    /// This constructor will sum the tile_norms data across all processes (via
    /// an all reduce). After the norms have been summed, it will be normalized.
    /// The normalization constant for each tile is the inverse of the number of
    /// elements in the tile.
    /// \param world The world where the shape will live
    /// \param tile_norms The Frobenius norm of tiles
    /// \param trange The tiled range of the tensor
    GenericShape(World& world, const Tensor<real_type>& tile_norms,
                 const TiledRange& trange) :
      shape_type_(sparse_real), tile_norms_(tile_norms.clone()),
      zero_tile_count_(0ul), threshold_(default_threshold_)
    {
      TA_ASSERT(! tile_norms.empty());
      TA_ASSERT(tile_norms.range() == trange.tiles());

      // Do global initialization of norm data
      world.gop.sum(tile_norms_.data(), tile_norms_.size());
    }

    /// Copy constructor

    /// Shallow copy of \c other.
    /// \param other The other shape object to be copied
    GenericShape(const GenericShape<Real>& other) :
      shape_type_(other.shape_type_), tile_norms_(other.tile_norms_),
      zero_tile_count_(other.zero_tile_count_), threshold_(default_threshold_)
    { }

    /// Copy assignment operator

    /// Shallow copy of \c other.
    /// \param other The other shape object to be copied
    /// \return A reference to this object.
    GenericShape<Real>& operator=(const GenericShape<Real>& other) {
      shape_type_ = other.shape_type_;
      tile_norms_ = other.tile_norms_;
      zero_tile_count_ = other.zero_tile_count_;
      threshold_ = other.threshold_;
      return *this;
    }

    /// Validate shape range

    /// \return \c true when range matches the range of this shape
    bool validate(const Range& range) const {
      if (shape_type_ == uninitialized) return false;
      if (shape_type_ == dense) return true;
      if (shape_type_ == sparse_real)
        return range == tile_norms_.template as<Tensor<Real>>().range();
      if (shape_type_ == sparse_bool)
        return range == tile_norms_.template as<Tensor<bool>>().range();
      assert(false); // unreachanble
      return false;
    }

    /// Check if tile is numerically zero

    /// \tparam Index The type of the index
    /// \return false
    template <typename Index>
    bool is_zero(const Index& i) const {
      TA_ASSERT(not empty());
      if (shape_type_ == dense) return false;
      if (shape_type_ == sparse_real)
        return tile_norms_.template as<Tensor<Real>>()[i] < threshold_;
      if (shape_type_ == sparse_bool)
        return tile_norms_.template as<Tensor<bool>>()[i] == false;
      assert(false); // unreachanble
      return false;
    }

    /// Initialization check

    /// \return \c true when this shape has been initialized.
    bool empty() const { return shape_type_ == uninitialized; }

    /// Dense predicate

    /// \return true
    bool is_dense() const { return shape_type_ == dense; }

#if 0
    /// Sparsity of the shape

    /// \return The fraction of tiles that are zero.
    float sparsity() const {
      TA_ASSERT(! tile_norms_.empty());
      return float(zero_tile_count_) / float(tile_norms_.size());
    }
#endif

    /// Threshold accessor

    /// \return The current threshold
    real_type threshold() const { return threshold_; }

    /// Set threshold to \c thresh

    /// \param thresh The new threshold
    void threshold(const real_type thresh) { threshold_ = thresh; }

    /// Global threshold accessor

    /// \param The current threshold
    static real_type default_threshold() { return default_threshold_; }

    /// Set global threshold to \c gthresh

    /// \param gthresh The new global threshold
    static void default_threshold(const real_type gthresh) { default_threshold_ = gthresh; }

#if 0
    /// Tile norm accessor

    /// \tparam Index The index type
    /// \param index The index of the tile norm to retrieve
    /// \return The norm of the tile at \c index
    template <typename Index>
    real_type operator[](const Index& index) const {
      TA_ASSERT(! tile_norms_.empty());
      return tile_norms_[index];
    }

    /// Data accessor

    /// \return A reference to the \c Tensor object that stores shape data
    const Tensor<real_type>& data() const { return tile_norms_; }

    /// Update sub-block of shape

    /// Update a sub-block shape information with another shape object.
    /// \tparam Index The bound index type
    /// \param lower_bound The lower bound of the sub-block to be updated
    /// \param upper_bound The upper bound of the sub-block to be updated
    /// \param other The shape that will be used to update the sub-block
    /// \return A new sparse shape object where the specified sub-block contains the data
    /// result_tile_norms of \c other.
    template <typename Index>
    GenericShape update_block(const Index& lower_bound, const Index& upper_bound,
        const GenericShape& other)
    {
      Tensor<real_type> result_tile_norms = tile_norms_.clone();

      auto result_tile_norms_blk = result_tile_norms.block(lower_bound, upper_bound);
      const real_type threshold = threshold_;
      size_type zero_tile_count = zero_tile_count_;
      result_tile_norms_blk.inplace_binary(other.tile_norms_,
          [threshold,&zero_tile_count] (real_type& l, const real_type r) {
            // Update the zero tile count for the result
            if((l < threshold) && (r >= threshold))
              ++zero_tile_count;
            else if((l >= threshold) && (r < threshold))
              --zero_tile_count;

            // Update the tile norm value
            l = r;
          });

      return GenericShape_(result_tile_norms, size_vectors_, zero_tile_count);
    }
#endif

  private:

#if 0
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
#endif

  public:

#if 0
    /// Create a copy of a sub-block of the shape

    /// \tparam Index The upper and lower bound array type
    /// \param lower_bound The lower bound of the sub-block
    /// \param upper_bound The upper bound of the sub-block
    template <typename Index>
    GenericShape block(const Index& lower_bound, const Index& upper_bound) const {
      std::shared_ptr<vector_type> size_vectors =
          block_range(lower_bound, upper_bound);

      // Copy the data from arg to result
      const real_type threshold = threshold_;
      size_type zero_tile_count = 0ul;
      auto copy_op = [threshold,&zero_tile_count] (real_type& restrict result,
          const real_type arg)
      {
        result = arg;
        if(arg < threshold)
          ++zero_tile_count;
      };


      // Construct the result norms tensor
      TensorConstView<real_type> block_view =
          tile_norms_.block(lower_bound, upper_bound);
      Tensor<real_type> result_norms((Range(block_view.range().extent())));
      result_norms.inplace_binary(shift(block_view), copy_op);

      return GenericShape(result_norms, size_vectors, zero_tile_count);
    }


    /// Create a scaled sub-block of the shape

    /// \tparam Index The upper and lower bound array type
    /// \param lower_bound The lower bound of the sub-block
    /// \param upper_bound The upper bound of the sub-block
    template <typename Index>
    GenericShape block(const Index& lower_bound, const Index& upper_bound,
        const real_type factor) const
    {
      std::shared_ptr<vector_type> size_vectors =
          block_range(lower_bound, upper_bound);

      // Copy the data from arg to result
      const real_type threshold = threshold_;
      size_type zero_tile_count = 0ul;
      auto copy_op = [factor,threshold,&zero_tile_count] (real_type& restrict result,
              const real_type arg)
      {
        result = arg * factor;
        if(result < threshold) {
          ++zero_tile_count;
          result = real_type(0);
        }
      };

      // Construct the result norms tensor
      TensorConstView<real_type> block_view =
          tile_norms_.block(lower_bound, upper_bound);
      Tensor<real_type> result_norms((Range(block_view.range().extent())));
      result_norms.inplace_binary(shift(block_view), copy_op);

      return GenericShape(result_norms, size_vectors, zero_tile_count);
    }

    /// Create a copy of a sub-block of the shape

    /// \param lower_bound The lower bound of the sub-block
    /// \param upper_bound The upper bound of the sub-block
    template <typename Index>
    GenericShape block(const Index& lower_bound, const Index& upper_bound,
        const Permutation& perm) const
    {
      return block(lower_bound, upper_bound).perm(perm);
    }


    /// Create a copy of a sub-block of the shape

    /// \param lower_bound The lower bound of the sub-block
    /// \param upper_bound The upper bound of the sub-block
    template <typename Index>
    GenericShape block(const Index& lower_bound, const Index& upper_bound,
        const real_type factor, const Permutation& perm) const
    {
      return block(lower_bound, upper_bound, factor).perm(perm);
    }
#endif

    /// Create a permuted shape of this shape

    /// \param perm The permutation to be applied
    /// \return A new, permuted shape
    GenericShape_ perm(const Permutation& perm) const {
      return GenericShape_(shape_type_,
                           tile_norms_.permute(perm),
                           zero_tile_count_);
    }

#if 0
    /// Scale shape

    /// Construct a new scaled shape as:
    /// \f[
    /// {(\rm{result})}_{ij...} = |(\rm{factor})| (\rm{this})_{ij...}
    /// \f]
    /// \param factor The scaling factor
    /// \return A new, scaled shape
    GenericShape_ scale(const real_type factor) const {
      TA_ASSERT(! tile_norms_.empty());
      const real_type threshold = threshold_;
      const real_type abs_factor = std::abs(factor);
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count, abs_factor] (real_type value) {
        value *= abs_factor;
        if(value < threshold) {
          value = real_type(0);
          ++zero_tile_count;
        }
        return value;
      };

      Tensor<real_type> result_tile_norms = tile_norms_.unary(op);

      return GenericShape_(result_tile_norms, size_vectors_, zero_tile_count);
    }

    /// Scale and permute shape

    /// Compute a new scaled shape is computed as:
    /// \f[
    /// {(\rm{result})}_{ji...} = \rm{perm}(j,i) |(\rm{factor})| (\rm{this})_{ij...}
    /// \f]
    /// \param factor The scaling factor
    /// \param perm The permutation that will be applied to this tensor.
    /// \return A new, scaled-and-permuted shape
    GenericShape_ scale(const real_type factor, const Permutation& perm) const {
      TA_ASSERT(! tile_norms_.empty());
      const real_type threshold = threshold_;
      const real_type abs_factor = std::abs(factor);
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count, abs_factor] (real_type value) {
        value *= abs_factor;
        if(value < threshold) {
          value = real_type(0);
          ++zero_tile_count;
        }
        return value;
      };

      Tensor<real_type> result_tile_norms = tile_norms_.unary(op, perm);

      return GenericShape_(result_tile_norms, perm_size_vectors(perm),
          zero_tile_count);
    }

    /// Add shapes

    /// Construct a new sum of shapes as:
    /// \f[
    /// {(\rm{result})}_{ij...} = (\rm{this})_{ij...} + (\rm{other})_{ij...}
    /// \f]
    /// \param other The shape to be added to this shape
    /// \return A sum of shapes
    GenericShape_ add(const GenericShape_& other) const {
      TA_ASSERT(! tile_norms_.empty());
      const real_type threshold = threshold_;
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count] (real_type left,
          const real_type right)
      {
        left += right;
        if(left < threshold) {
          left = real_type(0);
          ++zero_tile_count;
        }
        return left;
      };

      Tensor<real_type> result_tile_norms =
          tile_norms_.binary(other.tile_norms_, op);

      return GenericShape_(result_tile_norms, size_vectors_, zero_tile_count);
    }

    /// Add and permute shapes

    /// Construct a new sum of shapes as:
    /// \f[
    /// {(\rm{result})}_{ji...} = \rm{perm}(i,j) (\rm{this})_{ij...} + (\rm{other})_{ij...}
    /// \f]
    /// \param other The shape to be added to this shape
    /// \param perm The permutation that is applied to the result
    /// \return A new, scaled shape
    GenericShape_ add(const GenericShape_& other, const Permutation& perm) const {
      TA_ASSERT(! tile_norms_.empty());
      const real_type threshold = threshold_;
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count] (real_type left,
          const real_type right)
      {
        left += right;
        if(left < threshold) {
          left = real_type(0);
          ++zero_tile_count;
        }
        return left;
      };

      Tensor<real_type> result_tile_norms =
          tile_norms_.binary(other.tile_norms_, op, perm);

      return GenericShape_(result_tile_norms, perm_size_vectors(perm),
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
    GenericShape_ add(const GenericShape_& other, real_type factor) const {
      TA_ASSERT(! tile_norms_.empty());
      const real_type threshold = threshold_;
      const real_type abs_factor = std::abs(factor);
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count, abs_factor] (real_type left,
          const real_type right)
      {
        left += right;
        left *= abs_factor;
        if(left < threshold) {
          left = real_type(0);
          ++zero_tile_count;
        }
        return left;
      };

      Tensor<real_type> result_tile_norms =
          tile_norms_.binary(other.tile_norms_, op);

      return GenericShape_(result_tile_norms, size_vectors_, zero_tile_count);
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
    GenericShape_ add(const GenericShape_& other, const real_type factor,
        const Permutation& perm) const
    {
      TA_ASSERT(! tile_norms_.empty());
      const real_type threshold = threshold_;
      const real_type abs_factor = std::abs(factor);
      size_type zero_tile_count = 0ul;
      auto op = [threshold, &zero_tile_count, abs_factor]
                 (real_type left, const real_type right)
      {
        left += right;
        left *= abs_factor;
        if(left < threshold) {
          left = real_type(0);
          ++zero_tile_count;
        }
        return left;
      };

      Tensor<real_type> result_tile_norms =
          tile_norms_.binary(other.tile_norms_, op, perm);

      return GenericShape_(result_tile_norms, perm_size_vectors(perm),
          zero_tile_count);
    }

    GenericShape_ add(real_type value) const {
      TA_ASSERT(! tile_norms_.empty());
      const real_type threshold = threshold_;
      size_type zero_tile_count = 0ul;

      Tensor<Real> result_tile_norms(tile_norms_.range());

      value = std::abs(value);
      const unsigned int dim = tile_norms_.range().rank();
      const vector_type* restrict const size_vectors = size_vectors_.get();

      if(dim == 1u) {
        auto add_const_op = [threshold, &zero_tile_count, value] (real_type norm,
            const real_type size)
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
              [] (const real_type size) { return real_type(1) / std::sqrt(size); });
        };

        // Compute the left and right outer products
        const unsigned int middle = (dim >> 1u) + (dim & 1u);
        const vector_type left = recursive_outer_product(size_vectors, middle, inv_sqrt_vec_op);
        const vector_type right = recursive_outer_product(size_vectors + middle, dim - middle, inv_sqrt_vec_op);

        math::outer_fill(left.size(), right.size(), left.data(), right.data(),
            tile_norms_.data(), result_tile_norms.data(),
            [threshold, &zero_tile_count, value] (real_type& norm,
                const real_type x, const real_type y)
            {
              norm += value * x * y;
              if(norm < threshold) {
                norm = real_type(0);
                ++zero_tile_count;
              }
            });
      }

      return GenericShape_(result_tile_norms, size_vectors_, zero_tile_count);
    }

    GenericShape_ add(const real_type value, const Permutation& perm) const {
      // TODO: Optimize this function so it does the permutation at the same
      // time as the addition.
      return add(value).perm(perm);
    }

    GenericShape_ subt(const GenericShape_& other) const {
      return add(other);
    }

    GenericShape_ subt(const GenericShape_& other, const Permutation& perm) const {
      return add(other, perm);
    }

    GenericShape_ subt(const GenericShape_& other, const real_type factor) const {
      return add(other, factor);
    }

    GenericShape_ subt(const GenericShape_& other, const real_type factor,
        const Permutation& perm) const
    {
      return add(other, factor, perm);
    }

    GenericShape_ subt(const real_type value) const {
      return add(value);
    }

    GenericShape_ subt(const real_type value, const Permutation& perm) const {
      return add(value, perm);
    }
#endif

  private:

#if 0
    static size_type scale_by_size(Tensor<Real>& tile_norms,
        const vector_type* restrict const size_vectors)
    {
      const unsigned int dim = tile_norms.range().rank();
      const real_type threshold = threshold_;
      size_type zero_tile_count = 0ul;

      if(dim == 1u) {
        // This is the easy case where the data is a vector and can be
        // normalized directly.
        math::inplace_vector_op(
            [threshold, &zero_tile_count] (real_type& norm, const real_type size) {
              norm *= size;
              if(norm < threshold) {
                norm = real_type(0);
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
            [threshold, &zero_tile_count] (real_type& norm, const real_type x,
                const real_type y)
            {
              norm *= x * y;
              if(norm < threshold) {
                norm = real_type(0);
                ++zero_tile_count;
              }
            });
      }

      return zero_tile_count;
    }
#endif

  public:

#if 0
    GenericShape_ mult(const GenericShape_& other) const {
      // TODO: Optimize this function so that the tensor arithmetic and
      // scale_by_size operations are performed in one step instead of two.

      TA_ASSERT(! tile_norms_.empty());
      Tensor<Real> result_tile_norms = tile_norms_.mult(other.tile_norms_);
      const size_type zero_tile_count =
          scale_by_size(result_tile_norms, size_vectors_.get());

      return GenericShape_(result_tile_norms, size_vectors_, zero_tile_count);
    }

    GenericShape_ mult(const GenericShape_& other, const Permutation& perm) const {
      // TODO: Optimize this function so that the tensor arithmetic and
      // scale_by_size operations are performed in one step instead of two.

      TA_ASSERT(! tile_norms_.empty());
      Tensor<Real> result_tile_norms = tile_norms_.mult(other.tile_norms_, perm);
      std::shared_ptr<vector_type> result_size_vector = perm_size_vectors(perm);
      const size_type zero_tile_count =
                scale_by_size(result_tile_norms, result_size_vector.get());

      return GenericShape_(result_tile_norms, result_size_vector, zero_tile_count);
    }

    GenericShape_ mult(const GenericShape_& other, const real_type factor) const {
      // TODO: Optimize this function so that the tensor arithmetic and
      // scale_by_size operations are performed in one step instead of two.

      TA_ASSERT(! tile_norms_.empty());
      Tensor<Real> result_tile_norms = tile_norms_.mult(other.tile_norms_, std::abs(factor));
      const size_type zero_tile_count =
          scale_by_size(result_tile_norms, size_vectors_.get());

      return GenericShape_(result_tile_norms, size_vectors_, zero_tile_count);
    }

    GenericShape_ mult(const GenericShape_& other, const real_type factor,
        const Permutation& perm) const
    {
      // TODO: Optimize this function so that the tensor arithmetic and
      // scale_by_size operations are performed in one step instead of two.

      TA_ASSERT(! tile_norms_.empty());
      Tensor<Real> result_tile_norms = tile_norms_.mult(other.tile_norms_, std::abs(factor), perm);
      std::shared_ptr<vector_type> result_size_vector = perm_size_vectors(perm);
      const size_type zero_tile_count =
          scale_by_size(result_tile_norms, result_size_vector.get());

      return GenericShape_(result_tile_norms, result_size_vector, zero_tile_count);
    }

    GenericShape_ gemm(const GenericShape_& other, real_type factor,
        const math::GemmHelper& gemm_helper) const
    {
      TA_ASSERT(! tile_norms_.empty());

      factor = std::abs(factor);
      const real_type threshold = threshold_;
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
      Tensor<real_type> result_norms(gemm_helper.make_result_range<typename Tensor<Real>::range_type>(
          tile_norms_.range(), other.tile_norms_.range()), 0);

      if(k_rank > 0u) {

        // Compute size vector
        const vector_type k_sizes =
            recursive_outer_product(size_vectors_.get() + gemm_helper.left_inner_begin(),
                k_rank, [] (const vector_type& size_vector) -> const vector_type&
                { return size_vector; });

        // TODO: Make this faster. It can be done without using temporaries
        // for the arguments, but requires a custom matrix multiply.

        Tensor<real_type> left(tile_norms_.range());
        const size_type mk = M * K;
        auto left_op = [] (const real_type left, const real_type right)
            { return left * right; };
        for(size_type i = 0ul; i < mk; i += K)
          math::vector_op(left_op, K, left.data() + i,
              tile_norms_.data() + i, k_sizes.data());

        Tensor<real_type> right(other.tile_norms_.range());
        for(integer i = 0ul, k = 0; k < K; i += N, ++k) {
          const real_type factor = k_sizes[k];
          auto right_op = [=] (const real_type arg) { return arg * factor; };
          math::vector_op(right_op, N, right.data() + i, other.tile_norms_.data() + i);
        }

        result_norms = left.gemm(right, factor, gemm_helper);

        // Hard zero tiles that are below the zero threshold.
        result_norms.inplace_unary(
            [threshold, &zero_tile_count] (real_type& value) {
              if(value < threshold) {
                value = real_type(0);
                ++zero_tile_count;
              }
            });

      } else {

        // This is an outer product, so the inputs can be used directly
        math::outer_fill(M, N, tile_norms_.data(), other.tile_norms_.data(), result_norms.data(),
            [threshold, &zero_tile_count, factor] (const real_type left,
                const real_type right)
            {
              real_type norm = left * right * factor;
              if(norm < threshold) {
                norm = real_type(0);
                ++zero_tile_count;
              }
              return norm;
            });
      }

      return GenericShape_(result_norms, result_size_vectors, zero_tile_count);
    }

    GenericShape_ gemm(const GenericShape_& other, const real_type factor,
        const math::GemmHelper& gemm_helper, const Permutation& perm) const
    {
      return gemm(other, factor, gemm_helper).perm(perm);
    }
#endif

  }; // class GenericShape

  // Static member initialization
  template <typename Real>
  typename GenericShape<Real>::real_type GenericShape<Real>::default_threshold_ = std::numeric_limits<Real>::epsilon();

} // namespace TiledArray

#endif // TILEDARRAY_SHAPE_GENERIC_SHAPE_H__INCLUDED
