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
#include <TiledArray/tiled_range.h>
#include <TiledArray/val_array.h>

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
    TA_STATIC_ASSERT(std::is_arithmetic<T>::value);

    // Internal typedefs
    typedef detail::ValArray<value_type> vector_type;

    Tensor<value_type> tile_norms_; ///< Tile magnitude data
    std::shared_ptr<vector_type> size_vectors_; ///< Tile volume data
    static value_type threshold_; ///< The zero threshold

    class Size {
    public:
      TILEDARRAY_FORCE_INLINE value_type
      operator()(const TiledRange1::range_type& tile) const {
        return value_type(tile.second - tile.first);
      }
    };

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
        result.outer_fill(left, right, math::Multiplies<value_type, value_type, value_type>());
      }

      return result;
    }

    /// Scale a number and set it to zero if less threshold
    class NormalizeAndZero {
      value_type threshold_;
    public:
      typedef void result_type;

      NormalizeAndZero() : threshold_(SparseShape::threshold_) { }

      void operator()(value_type& restrict norm, const value_type size) const {
        norm /= size;
        if(norm < threshold_)
          norm = 0;
      }

      vector_type operator()(const vector_type& size_vector) const {
        return vector_type(size_vector, *this);
      }

      value_type operator()(const value_type size) const {
        return value_type(1) / size;
      }

      void operator()(value_type& restrict norm, const value_type x, const value_type y) const {
        norm *= x * y;
        if(norm < threshold_)
          norm = 0;
      }
    }; // class NormalizeAndZero


    /// Normalize tile norms

    /// This function will divide each norm by the number of elements in the
    /// tile. If the normalized norm is less than threshold, the value is set to
    /// zero.
    void normalize() {
      const NormalizeAndZero op;
      const unsigned int dim = tile_norms_.range().dim();
      const vector_type* restrict const size_vectors = size_vectors_.get();

      if(dim == 1u) {
        // This is the easy case where the data is a vector and can be
        // normalized directly.
        math::binary_vector_op(size_vectors[0].size(), size_vectors[0].data(),
            tile_norms_.data(), op);

      } else {
        // Here the normalization constants are computed and multiplied by the
        // norm data using a recursive, outer-product algorithm. This is done to
        // minimize temporary memory requirements, memory bandwidth, and work.

        // Compute the left and right outer products
        const unsigned int middle = (dim >> 1u) + (dim & 1u);
        const vector_type left = recursive_outer_product(size_vectors, middle, op);
        const vector_type right = recursive_outer_product(size_vectors + middle, dim - middle, op);

        math::outer(left.size(), right.size(), left.data(), right.data(),
            tile_norms_.data(), op);
      }
    }

    static std::shared_ptr<vector_type>
    initialize_size_vectors(const TiledRange& trange) {
      // Allocate memory for size vectors
      const unsigned int dim = trange.tiles().dim();
      std::shared_ptr<vector_type> size_vectors(new vector_type[dim],
          madness::detail::CheckedArrayDeleter<vector_type>());

      // Initialize the size vectors
      for(unsigned int i = 0ul; i != dim; ++i) {
        const size_type n = trange.data()[i].tiles().second - trange.data()[i].tiles().first;

        size_vectors.get()[i] = vector_type(n, & (* trange.data()[i].begin()), Size());
      }

      return size_vectors;
    }

    std::shared_ptr<vector_type> perm_size_vectors(const Permutation& perm) const {
      const unsigned int n = tile_norms_.range().dim();

      // Allocate memory for the contracted size vectors
      std::shared_ptr<vector_type> result_size_vectors(new vector_type[n],
          madness::detail::CheckedArrayDeleter<vector_type>());

      // Initialize the size vectors
      for(unsigned int i = 0u; i < n; ++i) {
        const unsigned int perm_i = perm[i];
        result_size_vectors.get()[perm_i] = size_vectors_.get()[i];
      }

      return result_size_vectors;
    }

    SparseShape(const Tensor<T>& tile_norms, const std::shared_ptr<vector_type>& size_vectors) :
      tile_norms_(tile_norms), size_vectors_(size_vectors)
    { }

  public:

    /// Constructor

    /// This constructor will normalize the tile norm, where the normalization
    /// constant for each tile is the inverse of the number of elements in the
    /// tile.
    /// \param tile_norms The Frobenius norm of tiles
    /// \param trange The tiled range of the tensor
    SparseShape(const Tensor<T>& tile_norms, const TiledRange& trange) :
      tile_norms_(tile_norms.clone()),
      size_vectors_(initialize_size_vectors(trange))
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
    SparseShape(madness::World& world, const Tensor<value_type>& tile_norms, const TiledRange& trange) :
      tile_norms_(tile_norms.clone()),
      size_vectors_(initialize_size_vectors(trange))
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
      tile_norms_(other.tile_norms_), size_vectors_(other.size_vectors_)
    { }

    /// Copy assignment operator

    /// Shallow copy of \c other.
    /// \param other The other shape object to be copied
    /// \return A reference to this object.
    SparseShape<T>& operator=(const SparseShape<T>& other) {
      tile_norms_ = other.tile_norms_;
      size_vectors_ = other.size_vectors_;
      return *this;
    }

    /// Validate shape range

    /// \return \c true when range matches the range of this shape
    bool validate(const Range& range) const { return (range == tile_norms_.range()); }

    /// Check that a tile is zero

    /// \tparam Index The type of the index
    /// \return false
    template <typename Index>
    bool is_zero(const Index& i) const { return tile_norms_[i] < threshold_; }

    /// Check density

    /// \return true
    static bool is_dense() { return false; }

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
    value_type operator[](const Index& index) const { return tile_norms_[index]; }

    /// Create a permuted shape of this shape

    /// \param perm The permutation to be applied
    /// \return A new, permuted shape
    SparseShape_ perm(const Permutation& perm) const {
      return SparseShape_(tile_norms_.permute(perm), perm_size_vectors(perm));
    }

    /// Data accessor

    /// \return A reference to the \c Tensor object that stores shape data
    const Tensor<value_type>& data() const { return tile_norms_; }

    /// Scale shape

    /// Construct a new scaled shape as:
    /// \f[
    /// {(\rm{result})}_{ij...} = |(\rm{factor})| (\rm{this})_{ij...}
    /// \f]
    /// \param factor The scaling factor
    /// \return A new, scaled shape
    SparseShape_ scale(const value_type factor) const {
      return SparseShape_(tile_norms_.scale(std::abs(factor)), size_vectors_);
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
      return SparseShape_(tile_norms_.scale(std::abs(factor), perm), size_vectors_);
    }

    /// Add shapes

    /// Construct a new sum of shapes as:
    /// \f[
    /// {(\rm{result})}_{ij...} = (\rm{this})_{ij...} + (\rm{other})_{ij...}
    /// \f]
    /// \param other The shape to be added to this shape
    /// \return A sum of shapes
    SparseShape_ add(const SparseShape_& other) const {
      return SparseShape_(tile_norms_.add(other.tile_norms_), size_vectors_);
    }

    /// Add and permute shapes

    /// Construct a new sum of shapes as:
    /// \f[
    /// {(\rm{result})}_{ji...} = \rm{perm}(i,j) (\rm{this})_{ij...} + (\rm{other})_{ij...}
    /// \f]
    /// \param factor The scaling factor
    /// \param perm The permutation that is applied to the result
    /// \return A new, scaled shape
    SparseShape_ add(const SparseShape_& other, const Permutation& perm) const {
      return SparseShape_(tile_norms_.add(other.tile_norms_, perm), size_vectors_);
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
      return SparseShape_(tile_norms_.add(other, std::abs(factor)), size_vectors_);
    }

    /// Add, scale, and permute shapes

    /// Construct a new sum of shapes as:
    /// \f[
    /// {(\rm{result})}_{ij...} = |(\rm{factor})| ((\rm{this})_{ij...} + (\rm{other})_{ij...})
    /// \f]
    /// \param other The shape to be added to this shape
    /// \param factor The scaling factor
    /// \return A scaled and permuted sum of shapes
    SparseShape_ add(const SparseShape_& other, const value_type factor,
        const Permutation& perm) const
    {
      return SparseShape_(tile_norms_.add(other.tile_norms_, std::abs(factor),
          perm), size_vectors_);
    }

  private:

    class ConstTensorNorm {
      const value_type value_;
      const value_type threshold_;

    public:
      typedef void result_type;

      ConstTensorNorm(const value_type value) :
        value_(value), threshold_(SparseShape::threshold_)
      { }

      TILEDARRAY_FORCE_INLINE value_type
      operator()(value_type norm, const value_type size) const {
        norm += value_ / std::sqrt(size);
        if(norm < threshold_)
          norm = 0;
        return norm;
      }

      TILEDARRAY_FORCE_INLINE vector_type operator()(const vector_type size_vector) const {
        return vector_type(size_vector, *this);
      }

      TILEDARRAY_FORCE_INLINE value_type operator()(const value_type size) const {
        return value_type(1) / std::sqrt(size);
      }

      TILEDARRAY_FORCE_INLINE void
      operator()(value_type& restrict norm, const value_type x, const value_type y) const {
        norm += value_ * x * y;
        if(norm < threshold_)
          norm = 0;
      }
    }; // class ConstTensorNorm

  public:

    SparseShape_ add(const value_type value) {
      Tensor<T> result_tile_norms(tile_norms_.range());

      const ConstTensorNorm op(value);
      const unsigned int dim = tile_norms_.range().dim();
      const vector_type* restrict const size_vectors = size_vectors_.get();

      if(dim == 1u) {
        // This is the easy case where the data is a vector and can be
        // normalized directly.
        math::binary_vector_op(size_vectors[0].size(), tile_norms_.data(),
            size_vectors[0].data(), result_tile_norms.data(), op);

      } else {
        // Here the normalization constants are computed and multiplied by the
        // norm data using a recursive, outer algorithm. This is done to
        // minimize temporary memory requirements, memory bandwidth, and work.

        // Compute the left and right outer products
        const unsigned int middle = (dim >> 1u) + (dim & 1u);
        const vector_type left = recursive_outer_product(size_vectors, middle, op);
        const vector_type right = recursive_outer_product(size_vectors + middle, dim - middle, op);

        math::outer_fill(left.size(), right.size(), left.data(), right.data(),
            tile_norms_.data(), result_tile_norms.data(), op);
      }


      return SparseShape_(result_tile_norms, size_vectors_);
    }

    SparseShape_ add(const value_type value, const Permutation& perm) const {
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

    SparseShape_ mult(const SparseShape_& other) const {
      return SparseShape_(tile_norms_.mult(other.tile_norms_));
    }

    SparseShape_ mult(const SparseShape_& other, const Permutation& perm) const {
      return SparseShape_(tile_norms_.mult(other.tile_norms_, perm));
    }

    SparseShape_ mult(const SparseShape_& other, const value_type factor) const {
      return SparseShape_(tile_norms_.mult(other.tile_norms_, std::abs(factor)));
    }

    SparseShape_ mult(const SparseShape_& other, const value_type factor,
        const Permutation& perm) const
    {
      return SparseShape_(tile_norms_.mult(other.tile_norms_, std::abs(factor), perm));
    }

  private:

    class GemmArgReduce {
    public:
      typedef void result_type;

      void operator()(value_type& restrict result, const value_type inner_size, value_type norm) const {
        norm *= inner_size;
        result += norm * norm;
      }

      void operator()(value_type& restrict arg) const { arg = std::sqrt(arg); }

    }; // class GemmLeftReduce

    class Product {
    public:
      typedef value_type result_type;

      const vector_type& operator()(const vector_type& size_vector) const {
        return size_vector;
      }

      value_type operator()(const value_type left, const value_type right) const {
        return left * right;
      }
    };

    class MultAndZero {
      value_type threshold_;
    public:
      typedef value_type result_type;

      MultAndZero() : threshold_(SparseShape::threshold_) { }

      TILEDARRAY_FORCE_INLINE value_type
      operator()(const value_type left, const value_type right) const {
        value_type norm = left * right;
        if(norm < threshold_)
          norm = 0;
        return norm;
      }
    }; // class NormalizeAndZero

  public:

    SparseShape_ gemm(const SparseShape_& other, const value_type factor,
        const math::GemmHelper& gemm_helper) const
    {

      integer m, n, k;
      gemm_helper.compute_matrix_sizes(m, n, k, tile_norms_.range(), other.tile_norms_.range());

      // Allocate memory for the contracted size vectors
      std::shared_ptr<vector_type> result_size_vectors(new vector_type[gemm_helper.result_rank()],
          madness::detail::CheckedArrayDeleter<vector_type>());

      // Initialize the size vectors
      unsigned int x = 0ul;
      for(unsigned int i = gemm_helper.left_outer_begin(); i < gemm_helper.left_outer_end(); ++i, ++x)
        result_size_vectors.get()[x] = size_vectors_.get()[i];
      for(unsigned int i = gemm_helper.right_outer_begin(); i < gemm_helper.right_outer_end(); ++i, ++x)
        result_size_vectors.get()[x] = other.size_vectors_.get()[i];

      // Compute the number of inner ranks
      const unsigned int k_rank = gemm_helper.left_inner_end() - gemm_helper.left_inner_begin();

      Tensor<T> result_norms(gemm_helper.make_result_range<typename Tensor<T>::range_type>(
          tile_norms_.range(), other.tile_norms_.range()));

      if(k_rank > 0) {

        // Compute k size vector
        const vector_type k_size_vector =
            recursive_outer_product(size_vectors_.get() + gemm_helper.left_inner_begin(),
                k_rank, Product());

        vector_type left(m, value_type(0));
        left.row_reduce(k, tile_norms_.data(), k_size_vector.data(), GemmArgReduce());

        vector_type right(n, value_type(0));
        right.col_reduce(n, other.tile_norms_.data(), k_size_vector.data(), GemmArgReduce());

        math::outer_fill(m, n, left.data(), right.data(), result_norms.data(),
            MultAndZero());
      } else {

        // This is an outer product, so the inputs can be used directly
        math::outer_fill(m, n, tile_norms_.data(), other.tile_norms_.data(), result_norms.data(),
            MultAndZero());
      }

      return SparseShape_(result_norms, result_size_vectors);
    }

    SparseShape_ gemm(const SparseShape_& other, const value_type factor,
        const math::GemmHelper& gemm_helper, const Permutation& perm) const
    {
      return gemm(other, factor, gemm_helper).perm(perm);
    }

  }; // class SparseShape

  // Static member initialization
  template <typename T>
  typename SparseShape<T>::value_type SparseShape<T>::threshold_ = 0;

} // namespace TiledArray

#endif // TILEDARRAY_SPASE_SHAPE_H__INCLUDED
