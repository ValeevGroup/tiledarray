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

namespace TiledArray {

  /// Arbitrary sparse shape

  /// \tparam T The sparse element value type
  template <typename T>
  class SparseShape {
  public:
    typedef T value_type;

  private:
    Tensor<value_type> tile_norms_; ///< Tile magnitude data
    std::vector<std::vector<value_type> > tile_volumes_; ///< Tile volume data
    static value_type threshold_; ///< The zero threshold

  public:

    /// Default constructor
    SparseShape() : tile_norms_() { }

    /// Constructor

    /// \param tile_norms The tile magnitude data
    SparseShape(const Tensor<T>& tile_norms) :
      tile_norms_(tile_norms)
    {
      TA_ASSERT(! tile_norms.empty());
    }

    /// Collective constructor

    /// After initializing local data, share data by calling \c collective_init .
    /// \param world The world where the shape will live
    /// \param tensor The tile magnitude data
    /// \param threshold The zero threshold
    SparseShape(madness::World& world, const Tensor<value_type>& tile_norms) :
      tile_norms_(tile_norms)
    {
      TA_ASSERT(! tile_norms.empty());
      collective_init(world);
    }


    /// Collective initialization shape

    /// Share data on each node with all other nodes. The data is shared using
    /// a collective, sum-reduction algorithm.
    /// \param world The world where the shape will live
    void collective_init(madness::World& world) {
      world.gop.sum(tile_norms_.data(), tile_norms_.size());
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

  private:

    typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_type;
    typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1, Eigen::RowMajor> vector_type;

    static vector_type make_inv_size_vector(const TiledRange1& trange1) {
      // Get the number of tiles
      const TiledRange1::range_type tiles = trange1.tiles();

      // Construct the result vector
      vector_type result(tiles.second - tiles.first);

      // Compute and store the inverse of the tile sizes
      const value_type one(1);
      for(TiledRange::size_type t = tiles.first; t < tiles.second; ++t) {
        const TiledRange1::range_type tile = trange1.tile(t);
        result[t - tiles.first] = one / value_type(tile.second - tile.first);
      }

      return result;
    }

    template <typename V>
    static void recursive_outer_product(V& result, const vector_type* inv_size_vectors, const unsigned int n) {
      switch(n) {
        case 2u:
        {
          Eigen::Map<const matrix_type, Eigen::AutoAlign>
              result_map(result.data(), inv_size_vectors[0].size(), inv_size_vectors[1].size());
          result_map.no_alias() = inv_size_vectors[0] * inv_size_vectors[1].transpose();
          break;
        }
        case 3u:
        {
          const matrix_type high = inv_size_vectors[1] * inv_size_vectors[2].transpose();

          Eigen::Map<const vector_type, Eigen::AutoAlign> high_map(high.data(), high.size());
          Eigen::Map<const matrix_type, Eigen::AutoAlign>
              result_map(result.data(), inv_size_vectors[0].size(), high_map.size());

          result_map.no_alias() = inv_size_vectors[0] * high.transpose();
          break;
        }
        case 4u:
        {
          const matrix_type low = inv_size_vectors[0] * inv_size_vectors[1].transpose();
          const matrix_type high = inv_size_vectors[2] * inv_size_vectors[3].transpose();

          // Remap low and high to vectors
          Eigen::Map<const vector_type, Eigen::AutoAlign> low_map(low.data(), low.size());
          Eigen::Map<const vector_type, Eigen::AutoAlign> high_map(high.data(), high.size());
          Eigen::Map<matrix_type, Eigen::AutoAlign>
              result_map(result.data(), low_map.size(), high_map.size());

          result_map.no_alias() = low_map * high_map.transpose();
          break;
        }
        default:
        {
          // Divide the range
          const std::size_t two = 2ul;
          const std::size_t low_n = n / two;
          const std::size_t high_n = low_n + n % two;

          // Compute the low outer product
          typename vector_type::Index low_size = 1u;
          for(unsigned int i = 0ul; i < low_n; ++i)
            low_size *= inv_size_vectors[i].size();
          vector_type low(low_size);
          outer_product(low, inv_size_vectors, low_n);

          // Compute the high outer product
          typename vector_type::Index high_size = 1u;
          for(unsigned int i = 0ul; i < low_n; ++i)
            high_size *= inv_size_vectors[i].size();
          vector_type high(high_size);
          outer_product(high, inv_size_vectors + low_n, high_n);

          // Construct the matrix that will hold the result of the outer product
          Eigen::Map<const matrix_type, Eigen::AutoAlign>
              result_map(result.data(), inv_size_vectors[0].size(), high.size());

          result_map.no_alias() = low * high.transpose();

          break;
        }
      }
    }

  public:

    /// Normalize tile norms
    static void normalize(Tensor<value_type>& norms, const TiledRange& trange) {
      TA_ASSERT(norms.range() == trange.tiles());

      const unsigned int dim = trange.tiles().dim();

      if(dim == 1u) {
        // Normalize norms directly
        const TiledRange1::range_type tiles = trange.data()[0].tiles();
        for(TiledRange::size_type t = tiles.first; t < tiles.second; ++t) {
          const TiledRange1::range_type tile = trange.data()[0].tile(t);
          norms[t - tiles.first] /= value_type(tile.second - tile.first);
        }
      } else {

        // Get the inverse size vectors
        std::vector<vector_type> inv_vec_sizes;
        inv_vec_sizes.reserve(dim);
        for(unsigned int i = 0ul; i < dim; ++i)
          inv_vec_sizes.push_back(make_inv_size_vector(trange.data()[i]));


        switch(dim) {
          case 2u:
          {
            Eigen::Map<matrix_type, Eigen::AutoAlign>
                norms_map(norms.data(), inv_vec_sizes[0].size(), inv_vec_sizes[1].size());

            norms_map.no_alias() *= inv_vec_sizes[0] * inv_vec_sizes[1].transpose();
            break;
          }

          case 3u:
          {
            vector_type high(inv_vec_sizes[1].size() * inv_vec_sizes[2].size());
            recursive_outer_product(high, &inv_vec_sizes[1], 2ul);

            Eigen::Map<matrix_type, Eigen::AutoAlign>
                norms_map(norms.data(), inv_vec_sizes[0].size(), high.size());

            norms_map.no_alias() *= inv_vec_sizes[0] * high.transpose();
            break;
          }
          default:
          {
            // Split the range
            const unsigned int two = 2ul;
            const unsigned int low_n = dim / two;
            const unsigned int high_n = low_n + dim % two;

            // Compute the size of the low and high vectors.
            typename vector_type::Index low_size = 1, high_size = 1;
            unsigned int i = 0u;
            for(; i < low_n; ++i)
              low_size *= inv_vec_sizes[i].size();
            for(; i < dim; ++i)
              high_size *= inv_vec_sizes[i].size();

            // Compute the low and high vectors
            vector_type low(low_size);
            recursive_outer_product(low, &inv_vec_sizes[1], low_n);
            vector_type high(high_size);
            recursive_outer_product(high, &inv_vec_sizes[1], high_n);

            Eigen::Map<matrix_type, Eigen::AutoAlign>
                norms_map(norms.data(), low.size(), high.size());


            norms_map.no_alias() *= low * high.transpose();
          }
        }
      }
    }

    /// Permute shape

    /// \param perm The permutation to be applied
    /// \return A new, permuted shape
    SparseShape<T> perm(const Permutation& perm) const {
      return SparseShape<T>(tile_norms_.permute(perm));
    }

    /// Data accessor

    /// \return A reference to the \c Tensor object that stores shape data
    const Tensor<value_type>& data() const { return tile_norms_; }

    /// Scale shape

    /// \param factor The scaling factor
    /// \return A new, scaled shape
    SparseShape<T> scale(const value_type factor) const {
      return SparseShape<T>(tile_norms_.scale(factor));
    }

    SparseShape<T> scale(const value_type factor, const Permutation& perm) const {
      return SparseShape<T>(tile_norms_.scale(std::abs(factor), perm));
    }

    SparseShape<T> add(const SparseShape<T>& other) const {
      return SparseShape<T>(tile_norms_.add(other.tile_norms_));
    }

    SparseShape<T> add(const SparseShape<T>& other, const Permutation& perm) const {
      return SparseShape<T>(tile_norms_.add(other.tile_norms_, perm));
    }

    SparseShape<T> add(const SparseShape<T>& other, const value_type factor) const {
      return SparseShape<T>(tile_norms_.add(other.tile_norms_, std::abs(factor)));
    }

    SparseShape<T> add(const SparseShape<T>& other, const value_type factor,
        const Permutation& perm) const
    {
      return SparseShape<T>(tile_norms_.add(other.tile_norms_, std::abs(factor), perm));
    }

    SparseShape<T> add(const value_type value) {
      return SparseShape<T>(tile_norms_.add(std::abs(value)));
    }

    SparseShape<T> add(const value_type value, const Permutation& perm) const {
      return SparseShape<T>(tile_norms_.add(std::abs(value), perm));
    }

    SparseShape<T> subt(const SparseShape<T>& other) const {
      return SparseShape<T>(tile_norms_.add(other.tile_norms_));
    }

    SparseShape<T> subt(const SparseShape<T>& other, const Permutation& perm) const {
      return SparseShape<T>(tile_norms_.add(other.tile_norms_, perm));
    }

    SparseShape<T> subt(const SparseShape<T>& other, const value_type factor) const {
      return SparseShape<T>(tile_norms_.add(other.tile_norms_, std::abs(factor)));
    }

    SparseShape<T> subt(const SparseShape<T>& other, const value_type factor,
        const Permutation& perm) const
    {
      return SparseShape<T>(tile_norms_.add(other.tile_norms_, std::abs(factor), perm));
    }

    SparseShape<T> subt(const value_type value) const {
      return SparseShape<T>(tile_norms_.add(std::abs(value)), threshold_ + value);
    }

    SparseShape<T> subt(const value_type value, const Permutation& perm) const {
      return SparseShape<T>(tile_norms_.add(std::abs(value), perm));
    }

    SparseShape<T> mult(const SparseShape<T>& other) const {
      return SparseShape<T>(tile_norms_.mult(other.data_));
    }

    SparseShape<T> mult(const SparseShape<T>& other, const Permutation& perm) const {
      return SparseShape<T>(tile_norms_.mult(other.data_, perm));
    }

    SparseShape<T> mult(const SparseShape<T>& other, const value_type factor) const {
      return SparseShape<T>(tile_norms_.mult(other.data_, std::abs(factor)));
    }

    SparseShape<T> mult(const SparseShape<T>& other, const value_type factor,
        const Permutation& perm) const
    {
      return SparseShape<T>(tile_norms_.mult(other.data_, std::abs(factor), perm));
    }

    SparseShape<T> gemm(const SparseShape<T>& other, const value_type factor,
        const math::GemmHelper& gemm_helper, const TiledRange& trange) const
    {
      typedef TiledRange::size_type size_type;

      integer m, n, k;
      gemm_helper.compute_matrix_sizes(m, n, k, tile_norms_.range(), other.data_.range());
      return SparseShape<T>(tile_norms_.gemm(other.tile_norms_, factor, gemm_helper));
    }

    SparseShape<T> gemm(const SparseShape<T>& other, const value_type factor,
        const math::GemmHelper& gemm_helper, const Permutation& perm) const
    {
      integer m, n, k;
      gemm_helper.compute_matrix_sizes(m, n, k, tile_norms_.range(), other.tile_norms_.range());
      return SparseShape<T>(tile_norms_.gemm(other.tile_norms_, factor, gemm_helper).perm(perm));
    }

  }; // class SparseShape

  // Static member initialization
  template <typename T>
  typename SparseShape<T>::value_type SparseShape<T>::threshold_ = 0;

} // namespace TiledArray

#endif // TILEDARRAY_SPASE_SHAPE_H__INCLUDED
