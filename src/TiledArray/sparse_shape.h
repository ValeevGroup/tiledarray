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

#include <TiledArray/fwd.h>

#include <TiledArray/tensor.h>
#include <TiledArray/tensor/shift_wrapper.h>
#include <TiledArray/tensor/tensor_interface.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/val_array.h>
#include <typeinfo>

namespace TiledArray {

/// Frobenius-norm-based sparse shape

/// Sparse shape uses a \c Tensor of Frobenius norms to describe the magnitude
/// of the data contained in tiles of an Array object. Because tiles may have
/// an arbitrary size, screening of tiles uses *scaled* (per-element) Frobenius
/// norms obtained by dividing the tile's Frobenius norm by the tile volume
/// (=its number of elements):
/// \f[
/// {\rm{shape}}_{ij...} = \frac{\|A_{ij...}\|}{N_i N_j ...}
/// \f]
/// where \f$ij...\f$ are tile indices, \f$\|A_{ij}\|\f$ is norm of tile
/// \f$ij...\f$, and \f$N_i N_j ...\f$ is the product of tile \f$ij...\f$ in
/// each dimension. Note that such scaled Frobenius norms no longer have the
/// properties of the Frobenius norms such as the submiltiplicativity.
///
/// All constructors will zero out tiles whose scaled norms are below the
/// threshold. The screening threshold is accessed via
/// SparseShape:::threshold() ; it is the global, but not immutable.
/// Thus it is possible to screen each operation separately, by changing the
/// screening threshold between each operation.
/// \warning If tile's scaled norm is below threshold, its scaled norm is set to
///          to zero and thus lost forever. E.g.
///          \c shape.scale(1e-10).scale(1e10) does not in general
///          equal \c shape , whereas \c shape.scale(1e10).scale(1e-10)
///          does.
///
/// \internal Thus the norms are stored in scaled form, and must be unscaled
///           to obtain Frobenius norms, e.g. for estimating the shapes of
///           arithmetic operation results.
/// \tparam T The sparse element value type
/// \note Scaling operations, such as SparseShape<T>::scale ,
/// SparseShape<T>::gemm , etc.
///       accept generic scaling factors; internally (modulus of) the scaling
///       factor is first converted to T, then used (see
///       SparseShape<T>::to_abs_factor).
template <typename T>
class SparseShape {
 public:
  typedef SparseShape<T> SparseShape_;  ///< This object type
  typedef T value_type;                 ///< The norm value type
  using index1_type = TA_1INDEX_TYPE;
  static_assert(TiledArray::detail::is_scalar_v<T>,
                "SparseShape<T> only supports scalar numeric types for T");
  typedef typename Tensor<value_type>::size_type size_type;  ///< Size type

 private:
  // T must be a numeric type
  static_assert(std::is_floating_point<T>::value,
                "SparseShape template type T must be a floating point type");

  // Internal typedefs
  typedef detail::ValArray<value_type> vector_type;

  Tensor<value_type> tile_norms_;  ///< scaled Tile norms
  mutable std::unique_ptr<Tensor<value_type>> tile_norms_unscaled_ =
      nullptr;  ///< unscaled Tile norms (memoized)
  std::shared_ptr<vector_type>
      size_vectors_;  ///< Tile size information; size_vectors_.get()[d][i]
                      ///< reports the size of i-th tile in dimension d
  size_type zero_tile_count_;    ///< Number of zero tiles
  static value_type threshold_;  ///< The current default threshold
  value_type my_threshold_ =
      threshold_;  ///< The threshold used to initialize this

  template <typename Op>
  static vector_type recursive_outer_product(
      const vector_type* const size_vectors, const unsigned int dim,
      const Op& op) {
    vector_type result;

    if (dim == 1u) {
      // Construct a modified copy of size_vector[0]
      result = op(*size_vectors);
    } else {
      // Compute split the range and compute the outer products
      const unsigned int middle = (dim >> 1u) + (dim & 1u);
      const vector_type left =
          recursive_outer_product(size_vectors, middle, op);
      const vector_type right =
          recursive_outer_product(size_vectors + middle, dim - middle, op);

      // Compute the outer product of left and right

      result = vector_type(left.size() * right.size());
      result.outer_fill(left, right,
                        [](const value_type left, const value_type right) {
                          return left * right;
                        });
    }

    return result;
  }

  enum class ScaleBy { Volume, InverseVolume };

  /// scales the contents of \c tile_norms by the corresponding tile's (inverse)
  /// volume

  /// \tparam ScaleBy_ defines the scaling factor: tile's volume, if
  /// ScaleBy::Volume, or tile's inverse volume, if ScaleBy::InverseVolume .
  /// \tparam Screen if true, will Screen the resulting contents of tile_norms
  /// \return the number of zero tiles if \c Screen is true, 0 otherwise.
  /// \note \c Screen=true can be useful even in ScaleBy_==ScaleBy::Volume ,
  ///       e.g. in SparseShape::mult()
  ///       where product of the scaled
  ///       norms of 2 tiles needs to be converted to the scaled norm of
  ///       the product tile by multiplying by its volume; the result needs
  ///       to be screened.
  template <ScaleBy ScaleBy_, bool Screen = true>
  static size_type scale_tile_norms(
      Tensor<T>& tile_norms,
      const vector_type* MADNESS_RESTRICT const size_vectors,
      const value_type threshold = threshold_) {
    const unsigned int dim = tile_norms.range().rank();
    madness::AtomicInt zero_tile_count;
    zero_tile_count = 0;

    if (dim == 1u) {
      // This is the easy case where the data is a vector and can be
      // normalized directly.
      math::inplace_vector_op(
          [threshold, &zero_tile_count](value_type& norm,
                                        const value_type size) {
            if (ScaleBy_ == ScaleBy::Volume)
              norm *= size;
            else
              norm /= size;
            if (Screen && norm < threshold) {
              norm = value_type(0);
              ++zero_tile_count;
            }
          },
          size_vectors[0].size(), tile_norms.data(), size_vectors[0].data());
    } else {
      // Here the normalization constants are computed and multiplied by the
      // norm data using a recursive, outer algorithm. This is done to
      // minimize temporary memory requirements, memory bandwidth, and work.

      /// for scaling by volume
      auto noop = [](const vector_type& size_vector) -> const vector_type& {
        return size_vector;
      };
      /// for scaling by inverse volume
      auto inv_vec_op = [](const vector_type& size_vector) {
        return vector_type(size_vector, [](const value_type size) {
          return value_type(1) / size;
        });
      };

      // Compute the left and right outer products
      const unsigned int middle = (dim >> 1u) + (dim & 1u);
      const vector_type left =
          ScaleBy_ == ScaleBy::Volume
              ? recursive_outer_product(size_vectors, middle, noop)
              : recursive_outer_product(size_vectors, middle, inv_vec_op);
      const vector_type right =
          ScaleBy_ == ScaleBy::Volume
              ? recursive_outer_product(size_vectors + middle, dim - middle,
                                        noop)
              : recursive_outer_product(size_vectors + middle, dim - middle,
                                        inv_vec_op);

      math::outer(
          left.size(), right.size(), left.data(), right.data(),
          tile_norms.data(),
          [threshold, &zero_tile_count](value_type& norm, const value_type x,
                                        const value_type y) {
            norm *= x * y;
            if (Screen && norm < threshold) {
              norm = value_type(0);
              ++zero_tile_count;
            }
          });
    }

    return Screen ? zero_tile_count : 0;
  }

  static std::shared_ptr<vector_type> initialize_size_vectors(
      const TiledRange& trange) {
    // Allocate memory for size vectors
    const unsigned int dim = trange.tiles_range().rank();
    std::shared_ptr<vector_type> size_vectors(
        new vector_type[dim], std::default_delete<vector_type[]>());

    // Initialize the size vectors
    for (unsigned int i = 0ul; i != dim; ++i) {
      const size_type n = trange.data()[i].tiles_range().second -
                          trange.data()[i].tiles_range().first;

      size_vectors.get()[i] =
          vector_type(n, &(*trange.data()[i].begin()),
                      [](const TiledRange1::range_type& tile) {
                        return value_type(tile.second - tile.first);
                      });
    }

    return size_vectors;
  }

  std::shared_ptr<vector_type> perm_size_vectors(
      const Permutation& perm) const {
    const unsigned int n = tile_norms_.range().rank();

    // Allocate memory for the contracted size vectors
    std::shared_ptr<vector_type> result_size_vectors(
        new vector_type[n], std::default_delete<vector_type[]>());

    // Initialize the size vectors
    for (unsigned int i = 0u; i < n; ++i) {
      const unsigned int perm_i = perm[i];
      result_size_vectors.get()[perm_i] = size_vectors_.get()[i];
    }

    return result_size_vectors;
  }

  /// @brief screens out zero tiles by zeroing out the norms of tiles below
  ///        `this->init_threshold()`
  /// @return the number of zero tiles
  auto screen_out_zero_tiles() {
    decltype(zero_tile_count_) zero_tile_count = 0;
    for (auto& n : tile_norms_) {
      if (n < my_threshold_) {
        n = 0;
        ++zero_tile_count;
      }
    }
    return zero_tile_count;
  }

  SparseShape(const Tensor<T>& tile_norms,
              const std::shared_ptr<vector_type>& size_vectors,
              const size_type zero_tile_count,
              const value_type my_threshold = threshold_)
      : tile_norms_(tile_norms),
        size_vectors_(size_vectors),
        zero_tile_count_(zero_tile_count),
        my_threshold_(my_threshold) {}

 public:
  /// Default constructor

  /// Construct a shape with no data.
  SparseShape() : tile_norms_(), size_vectors_(), zero_tile_count_(0ul) {}

  /// "Dense" Constructor

  /// This constructor set the tile norms to the same value.
  /// \param tile_norm the value of the (per-element) norm for every tile
  /// \param trange The tiled range of the tensor
  /// \note this ctor *does not* scale tile norms
  /// \note if @c tile_norm is less than the threshold then all tile norms are
  /// set to zero
  SparseShape(const value_type& tile_norm, const TiledRange& trange)
      : tile_norms_(trange.tiles_range(),
                    (tile_norm < threshold_ ? 0 : tile_norm)),
        size_vectors_(initialize_size_vectors(trange)),
        zero_tile_count_(tile_norm < threshold_ ? trange.tiles_range().area()
                                                : 0ul) {}

  /// Constructs a SparseShape from a functor returning norm values

  /// \tparam Op callable of signature `value_type(const Range::index&)`
  /// \param tile_norm_op a functor that returns Frobenius norms of tiles
  /// \param trange The tiled range of the tensor
  /// \param do_not_scale if true, assume that the tile norms in \c tile_norms
  /// are already scaled
  template <
      typename Op,
      typename = std::enable_if_t<std::is_invocable_r_v<
          value_type, std::remove_reference_t<Op>, const Range::index_type&>>>
  SparseShape(Op&& tile_norm_op, const TiledRange& trange,
              bool do_not_scale = false)
      : tile_norms_(trange.tiles_range(), std::forward<Op>(tile_norm_op)),
        size_vectors_(initialize_size_vectors(trange)),
        zero_tile_count_(0ul) {
    TA_ASSERT(!tile_norms_.empty());
    TA_ASSERT(tile_norms_.range() == trange.tiles_range());

    if (!do_not_scale) {
      zero_tile_count_ = scale_tile_norms<ScaleBy::InverseVolume>(
          tile_norms_, size_vectors_.get());
    } else {
      zero_tile_count_ = screen_out_zero_tiles();
    }
  }

  /// Constructor from a tensor of (scaled/unscaled) norm values

  /// \param tile_norms The Frobenius norm of tiles
  /// \param trange The tiled range of the tensor
  /// \param do_not_scale if true, assume that the tile norms in \c tile_norms
  /// are already scaled
  SparseShape(const Tensor<value_type>& tile_norms, const TiledRange& trange,
              bool do_not_scale = false)
      : tile_norms_(tile_norms.clone()),
        size_vectors_(initialize_size_vectors(trange)),
        zero_tile_count_(0ul) {
    TA_ASSERT(!tile_norms_.empty());
    TA_ASSERT(tile_norms_.range() == trange.tiles_range());

    if (!do_not_scale) {
      zero_tile_count_ = scale_tile_norms<ScaleBy::InverseVolume>(
          tile_norms_, size_vectors_.get());
    } else {
      zero_tile_count_ = screen_out_zero_tiles();
    }
  }

  /// "Sparse" constructor

  /// This constructor uses tile norms given as a sparse tensor,
  /// represented as a sequence of {index,value_type} data.
  /// The tile norms are scaled by the inverse of the corresponding tile's
  /// volumes.
  /// \tparam SparseNormSequence the sequence of
  ///         `std::pair<index,value_type>` objects,
  ///         where `index` is a directly-addressable sequence indices.
  /// \param tile_norms The Frobenius norm of tiles
  /// \param trange The tiled range of the tensor
  /// \param do_not_scale if true, assume that the tile norms in \c tile_norms
  /// are already scaled
  template <typename SparseNormSequence,
            typename = std::enable_if_t<
                TiledArray::detail::has_member_function_begin_anyreturn<
                    std::decay_t<SparseNormSequence>>::value &&
                TiledArray::detail::has_member_function_end_anyreturn<
                    std::decay_t<SparseNormSequence>>::value>>
  SparseShape(const SparseNormSequence& tile_norms, const TiledRange& trange,
              bool do_not_scale = false)
      : tile_norms_(trange.tiles_range(), value_type(0)),
        size_vectors_(initialize_size_vectors(trange)),
        zero_tile_count_(trange.tiles_range().volume()) {
    const auto dim = tile_norms_.range().rank();
    for (const auto& pair_idx_norm : tile_norms) {
      auto compute_tile_volume = [dim, this, pair_idx_norm]() -> uint64_t {
        uint64_t tile_volume = 1;
        for (size_t d = 0; d != dim; ++d)
          tile_volume *= size_vectors_.get()[d].at(pair_idx_norm.first[d]);
        return tile_volume;
      };
      auto norm_per_element =
          do_not_scale ? pair_idx_norm.second
                       : (pair_idx_norm.second / compute_tile_volume());
      if (norm_per_element >= my_threshold_) {
        tile_norms_[pair_idx_norm.first] = norm_per_element;
        --zero_tile_count_;
      }
    }
  }

  /// Collective "dense" constructor

  /// This constructor uses tile norms given as a dense tensor.
  /// The tile norms are max-reduced across all processes (via
  /// an all reduce).
  /// Next, the norms are scaled by the inverse of the corresponding tile's
  /// volumes.
  /// \param world The world where the shape will live
  /// \param tile_norms The Frobenius norm of tiles by default;
  ///        expected to contain nonzeros for this rank's subset of tiles,
  ///        or be replicated.
  /// \param trange The tiled range of the tensor
  /// \param do_not_scale if true, assume that the tile norms in \c tile_norms
  /// are already scaled
  SparseShape(World& world, const Tensor<value_type>& tile_norms,
              const TiledRange& trange, bool do_not_scale = false)
      : tile_norms_(tile_norms.clone()),
        size_vectors_(initialize_size_vectors(trange)),
        zero_tile_count_(0ul) {
    TA_ASSERT(!tile_norms_.empty());
    TA_ASSERT(tile_norms_.range() == trange.tiles_range());

    // reduce norm data from all processors
    world.gop.max(tile_norms_.data(), tile_norms_.size());

    if (!do_not_scale) {
      zero_tile_count_ = scale_tile_norms<ScaleBy::InverseVolume>(
          tile_norms_, size_vectors_.get());
    } else {
      zero_tile_count_ = screen_out_zero_tiles();
    }
  }

  /// Collective "sparse" constructor

  /// This constructor uses tile norms given as a sparse tensor,
  /// represented as a sequence of {index,value_type} data.
  /// The tile norms are scaled to per-element norms by dividing each
  /// norm by the tile's volume.
  /// Lastly, the norms are max-reduced across all processors.
  /// \tparam SparseNormSequence the sequence of \c std::pair<index,value_type>
  /// objects,
  ///         where \c index is a directly-addressable sequence of integers.
  /// \param world The world where the shape will live
  /// \param tile_norms The Frobenius norm of tiles; expected to contain
  /// nonzeros
  ///        for this rank's subset of tiles, or be replicated.
  /// \param trange The tiled range of the tensor
  template <typename SparseNormSequence>
  SparseShape(World& world, const SparseNormSequence& tile_norms,
              const TiledRange& trange)
      : SparseShape(tile_norms, trange) {
    world.gop.max(tile_norms_.data(), tile_norms_.size());
    zero_tile_count_ = screen_out_zero_tiles();
  }

  /// Copy constructor

  /// Shallow copy of \c other.
  /// \param other The other shape object to be copied
  SparseShape(const SparseShape<T>& other)
      : tile_norms_(other.tile_norms_),
        tile_norms_unscaled_(
            other.tile_norms_unscaled_
                ? std::make_unique<decltype(tile_norms_)>(
                      other.tile_norms_unscaled_.get()->clone())
                : nullptr),
        size_vectors_(other.size_vectors_),
        zero_tile_count_(other.zero_tile_count_),
        my_threshold_(other.my_threshold_) {}

  /// Copy assignment operator

  /// Shallow copy of \c other.
  /// \param other The other shape object to be copied
  /// \return A reference to this object.
  SparseShape<T>& operator=(const SparseShape<T>& other) {
    tile_norms_ = other.tile_norms_;
    tile_norms_unscaled_ = other.tile_norms_unscaled_
                               ? std::make_unique<decltype(tile_norms_)>(
                                     other.tile_norms_unscaled_.get()->clone())
                               : nullptr;
    size_vectors_ = other.size_vectors_;
    zero_tile_count_ = other.zero_tile_count_;
    my_threshold_ = other.my_threshold_;
    return *this;
  }

  /// Validate shape range

  /// \return \c true when range matches the range of this shape
  bool validate(const Range& range) const {
    if (tile_norms_.empty()) return false;
    return (range == tile_norms_.range());
  }

  /// Check that a tile is zero

  /// \tparam Ordinal an integer type
  /// \param ord the ordinal index
  /// \return true if tile at position \p ord is zero
  template <typename Ordinal>
  std::enable_if_t<std::is_integral_v<Ordinal>, bool> is_zero(
      const Ordinal& ord) const {
    TA_ASSERT(!tile_norms_.empty());
    return tile_norms_.at_ordinal(ord) < my_threshold_;
  }

  /// Check that a tile is zero

  /// \tparam Index a sized integral range type
  /// \param i the index
  /// \return true if tile at position \p i is zero
  template <typename Index, typename = std::enable_if_t<
                                detail::is_integral_sized_range_v<Index>>>
  bool is_zero(const Index& i) const {
    TA_ASSERT(!tile_norms_.empty());
    return tile_norms_[i] < my_threshold_;
  }

  /// Check that a tile is zero

  /// \tparam Integer an integer type
  /// \param i the index
  /// \return true if tile at position \p i is zero
  template <typename Integer>
  std::enable_if_t<std::is_integral_v<Integer>, bool> is_zero(
      const std::initializer_list<Integer>& i) const {
    return this->is_zero<std::initializer_list<Integer>>(i);
  }

  /// Check density

  /// \return true
  static constexpr bool is_dense() { return false; }

  /// Sparsity of the shape

  /// \return The fraction of tiles that are zero. Always returns 0 if
  /// `this->data().size()` is zero.
  float sparsity() const {
    TA_ASSERT(!tile_norms_.empty());
    return tile_norms_.size() != 0
               ? float(zero_tile_count_) / float(tile_norms_.size())
               : 0.f;
  }

  // clang-format off
  /// Default threshold accessor

  /// Default threshold is used to initialize new shapes resulting from
  /// _binary_ (not unary) expressions involving SparseShape;
  /// \note This threshold is also used to screen binary arithmetic expressions involving DistArray objects
  /// \return The current default threshold
  // clang-format on
  static value_type threshold() { return threshold_; }

  // clang-format off
  /// Set default threshold to \c thresh

  /// \warning it is only safe to change default threshold after a fence;
  /// this can be done as e.g. `world.gop.serial_invoke([new_thresh] { SparseShape<T>::threshold(new_thresh); });`
  /// \param thresh The new default threshold
  // clang-format on
  static void threshold(const value_type thresh) { threshold_ = thresh; }

  /// Accesses the threshold used to initialize this tile

  /// \return The threshold used to initialize this
  value_type init_threshold() const { return my_threshold_; }

  /// Tile norm accessor

  /// \tparam Index The index type
  /// \param index The index of the tile norm to retrieve
  /// \return The (scaled) norm of the tile at \c index
  template <typename Index>
  value_type operator[](const Index& index) const {
    TA_ASSERT(!tile_norms_.empty());
    return tile_norms_[index];
  }

  /// Transform the norm tensor with an operation

  /// \tparam Op a functor such that `Op(const Tensor<T>&)` is convertible to
  /// Tensor<T>
  /// \param op functor used to transform norms
  /// \return a transformed shape; i.e., `result.data() == op(this->data())`
  /// \post `result->init_threshold() == SparseShape::threshold()`
  /// \note  Since the input tile norms have already been scaled the output
  ///  norms will be identically scaled, e.g. when Op is an identity operation
  ///  the output
  /// SparseShape data will have the same values as this.
  template <typename Op>
  SparseShape_ transform(Op&& op) const {
    Tensor<T> new_norms = op(tile_norms_);
    madness::AtomicInt zero_tile_count;
    zero_tile_count = 0;

    const value_type threshold = my_threshold_;
    auto apply_threshold = [threshold, &zero_tile_count](value_type& norm) {
      TA_ASSERT(norm >= value_type(0));
      if (norm < threshold) {
        norm = value_type(0);
        ++zero_tile_count;
      }
    };

    math::inplace_vector_op(apply_threshold, new_norms.range().volume(),
                            new_norms.data());

    return SparseShape_(std::move(new_norms), size_vectors_, zero_tile_count,
                        my_threshold_);
  }

  /// Data accessor

  /// \return A const reference to the \c Tensor object that stores the scaled
  /// (per-element) Frobenius norms of tiles
  const Tensor<value_type>& data() const { return tile_norms_; }

  /// Data accessor

  /// \return A const reference to the \c Tensor object that stores the
  /// Frobenius norms of tiles
  const Tensor<value_type>& tile_norms() const {
    if (tile_norms_unscaled_ == nullptr) {
      tile_norms_unscaled_ =
          std::make_unique<decltype(tile_norms_)>(tile_norms_.clone());
      [[maybe_unused]] auto should_be_zero =
          scale_tile_norms<ScaleBy::Volume, false>(
              *tile_norms_unscaled_, size_vectors_.get(), my_threshold_);
      TA_ASSERT(should_be_zero == 0);
    }
    return *(tile_norms_unscaled_.get());
  }

  /// Initialization check

  /// \return \c true when this shape has been initialized.
  bool empty() const { return tile_norms_.empty(); }

  /// Check if the shape is initialized (i.e. !empty())

  /// \return true if the shape is non-default initialized and its data is
  /// nonnull
  explicit operator bool() const { return !empty(); }

  /// \return number of nonzero tiles
  std::size_t nnz() const {
    TA_ASSERT(!empty());
    return tile_norms_.size() - zero_tile_count_;
  }

  /// Compute union of two shapes

  /// \param mask_shape The input shape used to mask the output; if
  /// `mask_shape.is_zero(i)` then `result[i]` will be zero \return A shape that
  /// is masked by the mask \post `result.init_threshold() ==
  /// this->init_threshold()`
  SparseShape_ mask(const SparseShape_& mask_shape) const {
    TA_ASSERT(!tile_norms_.empty());
    TA_ASSERT(!mask_shape.empty());
    TA_ASSERT(tile_norms_.range() == mask_shape.tile_norms_.range());

    madness::AtomicInt zero_tile_count;
    zero_tile_count = zero_tile_count_;
    auto op = [this_threshold = this->init_threshold(),
               mask_threshold = mask_shape.init_threshold(),
               &zero_tile_count](value_type left, const value_type right) {
      if (left >= this_threshold && right < mask_threshold) {
        left = value_type(0);
        ++zero_tile_count;
      }

      return left;
    };

    Tensor<value_type> result_tile_norms =
        tile_norms_.binary(mask_shape.tile_norms_, op);

    return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count,
                        my_threshold_);
  }

  // clang-format off
  /// Creates a copy of this with a sub-block updated with contents of another shape

  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param lower_bound The lower bound of the sub-block to be updated
  /// \param upper_bound The upper bound of the sub-block to be updated
  /// \param other The shape that will be used to update the sub-block
  /// \return A new sparse shape object where the sub-block defined by \p lower_bound and \p upper_bound contains
  /// the data result_tile_norms of \c other ; note that result constructed using the same threshold as used to construct this
  /// \post `result.init_threshold() == this->init_threshold()`
  // clang-format on
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  SparseShape update_block(const Index1& lower_bound, const Index2& upper_bound,
                           const SparseShape& other) const {
    Tensor<value_type> result_tile_norms = tile_norms_.clone();

    auto result_tile_norms_blk =
        result_tile_norms.block(lower_bound, upper_bound);
    const value_type threshold = my_threshold_;
    madness::AtomicInt zero_tile_count;
    zero_tile_count = zero_tile_count_;
    result_tile_norms_blk.inplace_binary(
        other.tile_norms_,
        [threshold, &zero_tile_count](value_type& l, const value_type r) {
          // Update the zero tile count for the result
          if ((l < threshold) && (r >= threshold))
            ++zero_tile_count;
          else if ((l >= threshold) && (r < threshold))
            --zero_tile_count;

          // Update the tile norm value
          l = r;
        });

    return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count,
                        my_threshold_);
  }

  // clang-format off
  /// Creates a copy of this with a sub-block updated with contents of another shape

  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \param lower_bound The lower bound of the sub-block to be updated
  /// \param upper_bound The upper bound of the sub-block to be updated
  /// \param other The shape that will be used to update the sub-block
  /// \return A new sparse shape object where the sub-block defined by \p lower_bound and \p upper_bound contains
  /// the data result_tile_norms of \c other.
  // clang-format on
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  SparseShape update_block(const std::initializer_list<Index1>& lower_bound,
                           const std::initializer_list<Index2>& upper_bound,
                           const SparseShape& other) const {
    return update_block<std::initializer_list<Index1>,
                        std::initializer_list<Index2>>(lower_bound, upper_bound,
                                                       other);
  }

  // clang-format off
  /// Creates a copy of this with a sub-block updated with contents of another shape

  /// \tparam PairRange Type representing a range of generalized pairs (see TiledArray::detail::is_gpair_v )
  /// \param bounds The {lower,upper} bounds of the sub-block
  /// \param other The shape that will be used to update the sub-block
  /// \return A new sparse shape object where the sub-block defined by \p lower_bound and \p upper_bound contains
  /// the data result_tile_norms of \c other; note that result constructed using the same threshold as used to construct this
  /// \post `result.init_threshold() == this->init_threshold()`
  // clang-format on
  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  SparseShape update_block(const PairRange& bounds,
                           const SparseShape& other) const {
    Tensor<value_type> result_tile_norms = tile_norms_.clone();

    auto result_tile_norms_blk = result_tile_norms.block(bounds);
    const value_type threshold = my_threshold_;
    madness::AtomicInt zero_tile_count;
    zero_tile_count = zero_tile_count_;
    result_tile_norms_blk.inplace_binary(
        other.tile_norms_,
        [threshold, &zero_tile_count](value_type& l, const value_type r) {
          // Update the zero tile count for the result
          if ((l < threshold) && (r >= threshold))
            ++zero_tile_count;
          else if ((l >= threshold) && (r < threshold))
            --zero_tile_count;

          // Update the tile norm value
          l = r;
        });

    return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count,
                        my_threshold_);
  }

  // clang-format off
  /// Creates a copy of this with a sub-block updated with contents of another shape

  /// \tparam Index An integral type
  /// \param bounds The {lower,upper} bounds of the sub-block
  /// \param other The shape that will be used to update the sub-block
  /// \return A new sparse shape object where the sub-block defined by \p lower_bound and \p upper_bound contains
  /// the data result_tile_norms of \c other.
  // clang-format on
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  SparseShape update_block(
      const std::initializer_list<std::initializer_list<Index>>& bounds,
      const SparseShape& other) const {
    return update_block<std::initializer_list<std::initializer_list<Index>>>(
        bounds, other);
  }

  /// Bitwise comparison

  /// \param other a SparseShape object
  /// \return true if this object and @c other object are bitwise identical
  inline bool operator==(const SparseShape<T>& other) const {
    bool equal = this->zero_tile_count_ == other.zero_tile_count_;
    if (equal) {
      const unsigned int dim = tile_norms_.range().rank();
      for (unsigned d = 0; d != dim && equal; ++d) {
        equal =
            equal && (size_vectors_.get()[d] == other.size_vectors_.get()[d]);
      }
      if (equal) {
        equal = equal && (tile_norms_ == other.tile_norms_);
      }
    }
    return equal;
  }

  /// Bitwise comparison
  /// \param other a SparseShape object
  /// \return true if this object and @c other object are bitwise NOT identical
  inline bool operator!=(const SparseShape<T>& other) const {
    return !(*this == other);
  }

 private:
  /// Create a copy of a sub-block of the shape

  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param lower_bound The lower bound of the sub-block
  /// \param upper_bound The upper bound of the sub-block
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  std::shared_ptr<vector_type> block_range(const Index1& lower_bound,
                                           const Index2& upper_bound) const {
    // Get the number dimensions of the shape
    const auto rank = tile_norms_.range().rank();
    std::shared_ptr<vector_type> size_vectors(
        new vector_type[rank], std::default_delete<vector_type[]>());

    unsigned int d = 0;
    using std::begin;
    using std::end;
    auto lower_it = begin(lower_bound);
    auto upper_it = begin(upper_bound);
    const auto lower_end = end(lower_bound);
    const auto upper_end = end(upper_bound);
    for (; lower_it != lower_end && upper_it != upper_end;
         ++d, ++lower_it, ++upper_it) {
      // Get the new range size
      const auto lower_d = *lower_it;
      const auto upper_d = *upper_it;
      const auto extent_d = upper_d - lower_d;

      // Check that the input indices are in range
      TA_ASSERT(lower_d >= tile_norms_.range().lobound(d));
      TA_ASSERT(lower_d <= upper_d);
      TA_ASSERT(upper_d <= tile_norms_.range().upbound(d));

      // Construct the size vector for rank i
      size_vectors.get()[d] =
          vector_type(extent_d, size_vectors_.get()[d].data() + lower_d);
    }
    TA_ASSERT(d == rank);

    return size_vectors;
  }

  /// Create a copy of a sub-block of the shape

  /// \tparam PairRange Type representing a range of generalized pairs (see
  /// TiledArray::detail::is_gpair_v ) \param bounds The {lower,upper} bounds of
  /// the sub-block
  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  std::shared_ptr<vector_type> block_range(const PairRange& bounds) const {
    // Get the number dimensions of the shape
    const auto rank = tile_norms_.range().rank();
    std::shared_ptr<vector_type> size_vectors(
        new vector_type[rank], std::default_delete<vector_type[]>());

    unsigned int d = 0;
    for (auto&& bound_d : bounds) {
      // Get the new range size
      const auto lower_d = detail::at(bound_d, 0);
      const auto upper_d = detail::at(bound_d, 1);
      const auto extent_d = upper_d - lower_d;

      // Check that the input indices are in range
      TA_ASSERT(lower_d >= tile_norms_.range().lobound(d));
      TA_ASSERT(lower_d <= upper_d);
      TA_ASSERT(upper_d <= tile_norms_.range().upbound(d));

      // Construct the size vector for rank i
      size_vectors.get()[d] =
          vector_type(extent_d, size_vectors_.get()[d].data() + lower_d);

      ++d;
    }
    TA_ASSERT(d == rank);

    return size_vectors;
  }

  /// makes a transformed subblock of the shape
  template <typename Op>
  static SparseShape_ make_block(
      const std::shared_ptr<vector_type>& size_vectors,
      const TensorConstView<value_type>& block_view, const Op& op,
      const value_type threshold = threshold_) {
    // Copy the data from arg to result
    madness::AtomicInt zero_tile_count;
    zero_tile_count = 0;
    auto copy_op = [threshold, &zero_tile_count, &op](
                       value_type& MADNESS_RESTRICT result,
                       const value_type arg) {
      result = op(arg);
      if (result < threshold) {
        ++zero_tile_count;
        result = value_type(0);
      }
    };

    // Construct the result norms tensor
    Tensor<value_type> result_norms(Range(block_view.range().extent()));
    result_norms.inplace_binary(shift(block_view), copy_op);

    return SparseShape(result_norms, size_vectors, zero_tile_count, threshold);
  }

 public:
  /// Create a copy of a sub-block of the shape

  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param lower_bound The lower bound of the sub-block
  /// \param upper_bound The upper bound of the sub-block
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  SparseShape block(const Index1& lower_bound,
                    const Index2& upper_bound) const {
    return make_block(
        block_range(lower_bound, upper_bound),
        tile_norms_.block(lower_bound, upper_bound),
        [](auto&& arg) { return arg; }, my_threshold_);
  }

  /// Create a copy of a sub-block of the shape

  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \param lower_bound The lower bound of the sub-block
  /// \param upper_bound The upper bound of the sub-block
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  SparseShape block(const std::initializer_list<Index1>& lower_bound,
                    const std::initializer_list<Index2>& upper_bound) const {
    return this
        ->block<std::initializer_list<Index1>, std::initializer_list<Index2>>(
            lower_bound, upper_bound);
  }

  /// Create a copy of a sub-block of the shape

  /// \tparam PairRange Type representing a range of generalized pairs (see
  /// TiledArray::detail::is_gpair_v ) \param bounds The {lower,upper} bounds of
  /// the sub-block
  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  SparseShape block(const PairRange& bounds) const {
    return make_block(
        block_range(bounds), tile_norms_.block(bounds),
        [](auto&& arg) { return arg; }, my_threshold_);
  }

  /// Create a copy of a sub-block of the shape

  /// \tparam Index An integral type
  /// \param bounds A range of {lower,upper} bounds for each dimension
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  SparseShape block(
      const std::initializer_list<std::initializer_list<Index>>& bounds) const {
    return make_block(
        block_range(bounds), tile_norms_.block(bounds),
        [](auto&& arg) { return arg; }, my_threshold_);
  }

  /// Create a scaled sub-block of the shape

  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \tparam Scalar A numeric type
  /// \note expression abs(Scalar) must be well defined (by default, std::abs
  /// will be used)
  /// \param lower_bound The lower bound of the sub-block
  /// \param upper_bound The upper bound of the sub-block
  /// \param factor the scaling factor
  template <typename Index1, typename Index2, typename Scalar,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2> &&
                                        detail::is_numeric_v<Scalar>>>
  SparseShape block(const Index1& lower_bound, const Index2& upper_bound,
                    const Scalar factor) const {
    const value_type abs_factor = to_abs_factor(factor);
    return make_block(
        block_range(lower_bound, upper_bound),
        tile_norms_.block(lower_bound, upper_bound),
        [&abs_factor](auto&& arg) { return abs_factor * arg; }, my_threshold_);
  }

  /// Create a scaled sub-block of the shape

  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \tparam Scalar A numeric type
  /// \note expression abs(Scalar) must be well defined (by default, std::abs
  /// will be used)
  /// \param lower_bound The lower bound of the sub-block
  /// \param upper_bound The upper bound of the sub-block
  template <typename Index1, typename Index2, typename Scalar,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2> &&
                                        detail::is_numeric_v<Scalar>>>
  SparseShape block(const std::initializer_list<Index1>& lower_bound,
                    const std::initializer_list<Index2>& upper_bound,
                    const Scalar factor) const {
    return this->block<std::initializer_list<Index1>,
                       std::initializer_list<Index2>, Scalar>(
        lower_bound, upper_bound, factor);
  }

  /// Create a scaled sub-block of the shape

  /// \tparam PairRange Type representing a range of generalized pairs (see
  /// TiledArray::detail::is_gpair_v ) \tparam Scalar A numeric type \note
  /// expression abs(Scalar) must be well defined (by default, std::abs will be
  /// used) \param bounds A range of {lower,upper} bounds for each dimension
  /// \param factor the scaling factor
  template <typename PairRange, typename Scalar,
            typename = std::enable_if_t<detail::is_numeric_v<Scalar> &&
                                        detail::is_gpair_range_v<PairRange>>>
  SparseShape block(const PairRange& bounds, const Scalar factor) const {
    const value_type abs_factor = to_abs_factor(factor);
    return make_block(
        block_range(bounds), tile_norms_.block(bounds),
        [&abs_factor](auto&& arg) { return abs_factor * arg; }, my_threshold_);
  }

  /// Create a scaled sub-block of the shape

  /// \tparam Index An integral type
  /// \tparam Scalar A numeric type
  /// \note expression abs(Scalar) must be well defined (by default, std::abs
  /// will be used)
  /// \param bounds A range of {lower,upper} bounds for each dimension
  /// \param factor the scaling factor
  template <typename Index, typename Scalar,
            typename = std::enable_if_t<detail::is_numeric_v<Scalar> &&
                                        std::is_integral_v<Index>>>
  SparseShape block(
      const std::initializer_list<std::initializer_list<Index>>& bounds,
      const Scalar factor) const {
    const value_type abs_factor = to_abs_factor(factor);
    return make_block(
        block_range(bounds), tile_norms_.block(bounds),
        [&abs_factor](auto&& arg) { return abs_factor * arg; }, my_threshold_);
  }

  /// Create a permuted sub-block of the shape

  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param lower_bound The lower bound of the sub-block
  /// \param upper_bound The upper bound of the sub-block
  /// \param perm permutation to apply
  /// \note permutation is not fused into construction
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  SparseShape block(const Index1& lower_bound, const Index2& upper_bound,
                    const Permutation& perm) const {
    return block(lower_bound, upper_bound).perm(perm);
  }

  /// Create a permuted sub-block of the shape

  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \param lower_bound The lower bound of the sub-block
  /// \param upper_bound The upper bound of the sub-block
  /// \param perm permutation to apply
  /// \note permutation is not fused into construction
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  SparseShape block(const std::initializer_list<Index1>& lower_bound,
                    const std::initializer_list<Index2>& upper_bound,
                    const Permutation& perm) const {
    return block(lower_bound, upper_bound).perm(perm);
  }

  /// Create a permuted sub-block of the shape

  /// \tparam PairRange Type representing a range of generalized pairs (see
  /// TiledArray::detail::is_gpair_v ) \param bounds A range of {lower,upper}
  /// bounds for each dimension \param perm permutation to apply \note
  /// permutation is not fused into construction
  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  SparseShape block(const PairRange& bounds, const Permutation& perm) const {
    return block(bounds).perm(perm);
  }

  /// Create a permuted sub-block of the shape

  /// \tparam Index An integral type
  /// \param bounds A range of {lower,upper} bounds for each dimension
  /// \param perm permutation to apply
  /// \note permutation is not fused into construction
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  SparseShape block(
      const std::initializer_list<std::initializer_list<Index>>& bounds,
      const Permutation& perm) const {
    return block(bounds).perm(perm);
  }

  /// Create a permuted scaled sub-block of the shape

  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \tparam Scalar A numeric type
  /// \note expression abs(Scalar) must be well defined (by default, std::abs
  /// will be used)
  /// \param lower_bound The lower bound of the sub-block
  /// \param upper_bound The upper bound of the sub-block
  /// \param factor the scaling factor
  /// \param perm permutation to apply
  /// \note permutation is not fused into construction
  template <typename Index1, typename Index2, typename Scalar,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2> &&
                                        detail::is_numeric_v<Scalar>>>
  SparseShape block(const Index1& lower_bound, const Index2& upper_bound,
                    const Scalar factor, const Permutation& perm) const {
    return block(lower_bound, upper_bound, factor).perm(perm);
  }

  /// Create a permuted scaled sub-block of the shape

  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \tparam Scalar A numeric type
  /// \note expression abs(Scalar) must be well defined (by default, std::abs
  /// will be used)
  /// \param lower_bound The lower bound of the sub-block
  /// \param upper_bound The upper bound of the sub-block
  /// \param factor the scaling factor
  /// \param perm permutation to apply
  /// \note permutation is not fused into construction
  template <typename Index1, typename Index2, typename Scalar,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2> &&
                                        detail::is_numeric_v<Scalar>>>
  SparseShape block(const std::initializer_list<Index1>& lower_bound,
                    const std::initializer_list<Index2>& upper_bound,
                    const Scalar factor, const Permutation& perm) const {
    return block(lower_bound, upper_bound, factor).perm(perm);
  }

  /// Create a permuted scaled sub-block of the shape

  /// \tparam PairRange Type representing a range of generalized pairs (see
  /// TiledArray::detail::is_gpair_v ) \tparam Scalar A numeric type \note
  /// expression abs(Scalar) must be well defined (by default, std::abs will be
  /// used) \param bounds A range of {lower,upper} bounds for each dimension
  /// \param factor the scaling factor
  /// \param perm permutation to apply
  /// \note permutation is not fused into construction
  template <typename PairRange, typename Scalar,
            typename = std::enable_if_t<detail::is_numeric_v<Scalar> &&
                                        detail::is_gpair_range_v<PairRange>>>
  SparseShape block(const PairRange& bounds, const Scalar factor,
                    const Permutation& perm) const {
    const value_type abs_factor = to_abs_factor(factor);
    return make_block(
               block_range(bounds), tile_norms_.block(bounds),
               [&abs_factor](auto&& arg) { return abs_factor * arg; },
               my_threshold_)
        .perm(perm);
  }

  /// Create a permuted scaled sub-block of the shape

  /// \tparam Index An integral type
  /// \tparam Scalar A numeric type
  /// \note expression abs(Scalar) must be well defined (by default, std::abs
  /// will be used)
  /// \param bounds A range of {lower,upper} bounds for each dimension
  /// \param factor the scaling factor
  /// \param perm permutation to apply
  /// \note permutation is not fused into construction
  template <typename Index, typename Scalar,
            typename = std::enable_if_t<detail::is_numeric_v<Scalar> &&
                                        std::is_integral_v<Index>>>
  SparseShape block(
      const std::initializer_list<std::initializer_list<Index>>& bounds,
      const Scalar factor, const Permutation& perm) const {
    const value_type abs_factor = to_abs_factor(factor);
    return make_block(
               block_range(bounds), tile_norms_.block(bounds),
               [&abs_factor](auto&& arg) { return abs_factor * arg; },
               my_threshold_)
        .perm(perm);
  }

  /// Create a permuted shape of this shape

  /// \param perm The permutation to be applied
  /// \return A new, permuted shape using the same threshold as this object
  SparseShape_ perm(const Permutation& perm) const {
    return SparseShape_(tile_norms_.permute(perm), perm_size_vectors(perm),
                        zero_tile_count_, my_threshold_);
  }

  // clang-format off
  /// Scale shape

  /// Construct a new scaled shape as:
  /// \f[
  /// {(\rm{result})}_{ij...} = |(\rm{factor})| (\rm{this})_{ij...}
  /// \f]
  /// \tparam Scalar A numeric type
  /// \note expression abs(Scalar) must be well defined (by default, std::abs
  /// will be used)
  /// \param factor The scaling factor
  /// \return A new, scaled shape initialized using `this->init_threshold()`
  /// \post `result.init_threshold() == this->init_threshold()`
  // clang-format on
  template <typename Scalar,
            typename = std::enable_if_t<detail::is_numeric_v<Scalar>>>
  SparseShape_ scale(const Scalar factor) const {
    TA_ASSERT(!tile_norms_.empty());
    const value_type threshold = my_threshold_;
    const value_type abs_factor = to_abs_factor(factor);
    madness::AtomicInt zero_tile_count;
    zero_tile_count = 0;
    auto op = [threshold, &zero_tile_count, abs_factor](value_type value) {
      value *= abs_factor;
      if (value < threshold) {
        value = value_type(0);
        ++zero_tile_count;
      }
      return value;
    };

    Tensor<value_type> result_tile_norms = tile_norms_.unary(op);

    return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count,
                        my_threshold_);
  }

  // clang-format off
  /// Scale and permute shape

  /// Compute a new scaled shape is computed as:
  /// \f[
  /// {(\rm{result})}_{ji...} = \rm{perm}(j,i) |(\rm{factor})|
  /// (\rm{this})_{ij...}
  /// \f]
  /// \tparam Factor The scaling factor type
  /// \note expression abs(Factor) must be well defined (by default, std::abs will be used)
  /// \param factor The scaling factor
  /// \param perm The permutation that will be applied to this tensor.
  /// \return A new, scaled-and-permuted shape initialized using `this->init_threshold()`
  /// \post `result.init_threshold() == this->init_threshold()`
  // clang-format on
  template <typename Factor>
  SparseShape_ scale(const Factor factor, const Permutation& perm) const {
    TA_ASSERT(!tile_norms_.empty());
    const value_type threshold = my_threshold_;
    const value_type abs_factor = to_abs_factor(factor);
    madness::AtomicInt zero_tile_count;
    zero_tile_count = 0;
    auto op = [threshold, &zero_tile_count, abs_factor](value_type value) {
      value *= abs_factor;
      if (value < threshold) {
        value = value_type(0);
        ++zero_tile_count;
      }
      return value;
    };

    Tensor<value_type> result_tile_norms = tile_norms_.unary(op, perm);

    return SparseShape_(result_tile_norms, perm_size_vectors(perm),
                        zero_tile_count, my_threshold_);
  }

  /// Add shapes

  /// Construct a new sum of shapes as:
  /// \f[
  /// {(\rm{result})}_{ij...} = (\rm{this})_{ij...} + (\rm{other})_{ij...}
  /// \f]
  /// \param other The shape to be added to this shape
  /// \return A sum of shapes
  SparseShape_ add(const SparseShape_& other) const {
    TA_ASSERT(!tile_norms_.empty());
    const value_type threshold = threshold_;
    madness::AtomicInt zero_tile_count;
    zero_tile_count = 0;
    auto op = [threshold, &zero_tile_count](value_type left,
                                            const value_type right) {
      left += right;
      if (left < threshold) {
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
  /// {(\rm{result})}_{ji...} = \rm{perm}(i,j) (\rm{this})_{ij...} +
  /// (\rm{other})_{ij...} \f] \param other The shape to be added to this shape
  /// \param perm The permutation that is applied to the result
  /// \return the new shape, equals \c this + \c other
  SparseShape_ add(const SparseShape_& other, const Permutation& perm) const {
    TA_ASSERT(!tile_norms_.empty());
    const value_type threshold = threshold_;
    madness::AtomicInt zero_tile_count;
    zero_tile_count = 0;
    auto op = [threshold, &zero_tile_count](value_type left,
                                            const value_type right) {
      left += right;
      if (left < threshold) {
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
  /// {(\rm{result})}_{ij...} = |(\rm{factor})| ((\rm{this})_{ij...} +
  /// (\rm{other})_{ij...}) \f] \tparam Factor The scaling factor type \note
  /// expression abs(Factor) must be well defined (by default, std::abs will be
  /// used) \param other The shape to be added to this shape \param factor The
  /// scaling factor \return A scaled sum of shapes
  template <typename Factor>
  SparseShape_ add(const SparseShape_& other, const Factor factor) const {
    TA_ASSERT(!tile_norms_.empty());
    const value_type threshold = threshold_;
    const value_type abs_factor = to_abs_factor(factor);
    madness::AtomicInt zero_tile_count;
    zero_tile_count = 0;
    auto op = [threshold, &zero_tile_count, abs_factor](
                  value_type left, const value_type right) {
      left += right;
      left *= abs_factor;
      if (left < threshold) {
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
  /// {(\rm{result})}_{ij...} = |(\rm{factor})| ((\rm{this})_{ij...} +
  /// (\rm{other})_{ij...}) \f] \tparam Factor The scaling factor type \note
  /// expression abs(Factor) must be well defined (by default, std::abs will be
  /// used) \param other The shape to be added to this shape \param factor The
  /// scaling factor \param perm The permutation that is applied to the result
  /// \return A scaled and permuted sum of shapes
  template <typename Factor>
  SparseShape_ add(const SparseShape_& other, const Factor factor,
                   const Permutation& perm) const {
    TA_ASSERT(!tile_norms_.empty());
    const value_type threshold = threshold_;
    const value_type abs_factor = to_abs_factor(factor);
    madness::AtomicInt zero_tile_count;
    zero_tile_count = 0;
    auto op = [threshold, &zero_tile_count, abs_factor](
                  value_type left, const value_type right) {
      left += right;
      left *= abs_factor;
      if (left < threshold) {
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
    TA_ASSERT(!tile_norms_.empty());
    const value_type threshold = threshold_;
    madness::AtomicInt zero_tile_count;
    zero_tile_count = 0;

    Tensor<T> result_tile_norms(tile_norms_.range());

    value = std::abs(value);
    const unsigned int dim = tile_norms_.range().rank();
    const vector_type* MADNESS_RESTRICT const size_vectors =
        size_vectors_.get();

    if (dim == 1u) {
      auto add_const_op = [threshold, &zero_tile_count, value](
                              value_type norm, const value_type size) {
        norm += value / std::sqrt(size);
        if (norm < threshold) {
          norm = 0;
          ++zero_tile_count;
        }
        return norm;
      };

      // This is the easy case where the data is a vector and can be
      // normalized directly.
      math::vector_op(add_const_op, size_vectors[0].size(),
                      result_tile_norms.data(), tile_norms_.data(),
                      size_vectors[0].data());

    } else {
      // Here the normalization constants are computed and multiplied by the
      // norm data using a recursive, outer algorithm. This is done to
      // minimize temporary memory requirements, memory bandwidth, and work.

      auto inv_sqrt_vec_op = [](const vector_type size_vector) {
        return vector_type(size_vector, [](const value_type size) {
          return value_type(1) / std::sqrt(size);
        });
      };

      // Compute the left and right outer products
      const unsigned int middle = (dim >> 1u) + (dim & 1u);
      const vector_type left =
          recursive_outer_product(size_vectors, middle, inv_sqrt_vec_op);
      const vector_type right = recursive_outer_product(
          size_vectors + middle, dim - middle, inv_sqrt_vec_op);

      math::outer_fill(
          left.size(), right.size(), left.data(), right.data(),
          tile_norms_.data(), result_tile_norms.data(),
          [threshold, &zero_tile_count, value](
              value_type& norm, const value_type x, const value_type y) {
            norm += value * x * y;
            if (norm < threshold) {
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

  SparseShape_ subt(const SparseShape_& other) const { return add(other); }

  SparseShape_ subt(const SparseShape_& other, const Permutation& perm) const {
    return add(other, perm);
  }

  template <typename Factor>
  SparseShape_ subt(const SparseShape_& other, const Factor factor) const {
    return add(other, factor);
  }

  template <typename Factor>
  SparseShape_ subt(const SparseShape_& other, const Factor factor,
                    const Permutation& perm) const {
    return add(other, factor, perm);
  }

  SparseShape_ subt(const value_type value) const { return add(value); }

  SparseShape_ subt(const value_type value, const Permutation& perm) const {
    return add(value, perm);
  }

  SparseShape_ mult(const SparseShape_& other) const {
    // TODO: Optimize this function so that the tensor arithmetic and
    // scale_tile_norms operations are performed in one step instead of two.

    TA_ASSERT(!tile_norms_.empty());
    Tensor<T> result_tile_norms = tile_norms_.mult(other.tile_norms_);
    const size_type zero_tile_count = scale_tile_norms<ScaleBy::Volume>(
        result_tile_norms, size_vectors_.get());

    return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count);
  }

  SparseShape_ mult(const SparseShape_& other, const Permutation& perm) const {
    // TODO: Optimize this function so that the tensor arithmetic and
    // scale_tile_norms operations are performed in one step instead of two.

    TA_ASSERT(!tile_norms_.empty());
    Tensor<T> result_tile_norms = tile_norms_.mult(other.tile_norms_, perm);
    std::shared_ptr<vector_type> result_size_vector = perm_size_vectors(perm);
    const size_type zero_tile_count = scale_tile_norms<ScaleBy::Volume>(
        result_tile_norms, result_size_vector.get());

    return SparseShape_(result_tile_norms, result_size_vector, zero_tile_count);
  }

  /// \tparam Factor The scaling factor type
  /// \note expression abs(Factor) must be well defined (by default, std::abs
  /// will be used)
  template <typename Factor>
  SparseShape_ mult(const SparseShape_& other, const Factor factor) const {
    // TODO: Optimize this function so that the tensor arithmetic and
    // scale_tile_norms operations are performed in one step instead of two.

    TA_ASSERT(!tile_norms_.empty());
    const value_type abs_factor = to_abs_factor(factor);
    Tensor<T> result_tile_norms =
        tile_norms_.mult(other.tile_norms_, abs_factor);
    const size_type zero_tile_count = scale_tile_norms<ScaleBy::Volume>(
        result_tile_norms, size_vectors_.get());

    return SparseShape_(result_tile_norms, size_vectors_, zero_tile_count);
  }

  /// \tparam Factor The scaling factor type
  /// \note expression abs(Factor) must be well defined (by default, std::abs
  /// will be used)
  template <typename Factor>
  SparseShape_ mult(const SparseShape_& other, const Factor factor,
                    const Permutation& perm) const {
    // TODO: Optimize this function so that the tensor arithmetic and
    // scale_tile_norms operations are performed in one step instead of two.

    TA_ASSERT(!tile_norms_.empty());
    const value_type abs_factor = to_abs_factor(factor);
    Tensor<T> result_tile_norms =
        tile_norms_.mult(other.tile_norms_, abs_factor, perm);
    std::shared_ptr<vector_type> result_size_vector = perm_size_vectors(perm);
    const size_type zero_tile_count = scale_tile_norms<ScaleBy::Volume>(
        result_tile_norms, result_size_vector.get());

    return SparseShape_(result_tile_norms, result_size_vector, zero_tile_count);
  }

  /// \tparam Factor The scaling factor type
  /// \note expression abs(Factor) must be well defined (by default, std::abs
  /// will be used)
  template <typename Factor>
  SparseShape_ gemm(const SparseShape_& other, const Factor factor,
                    const math::GemmHelper& gemm_helper) const {
    TA_ASSERT(!tile_norms_.empty());

    const value_type abs_factor = to_abs_factor(factor);
    const value_type threshold = threshold_;
    madness::AtomicInt zero_tile_count;
    zero_tile_count = 0;
    using integer = TiledArray::math::blas::integer;
    integer M = 0, N = 0, K = 0;
    gemm_helper.compute_matrix_sizes(M, N, K, tile_norms_.range(),
                                     other.tile_norms_.range());

    // Allocate memory for the contracted size vectors
    std::shared_ptr<vector_type> result_size_vectors(
        new vector_type[gemm_helper.result_rank()],
        std::default_delete<vector_type[]>());

    // Initialize the result size vectors
    unsigned int x = 0ul;
    for (unsigned int i = gemm_helper.left_outer_begin();
         i < gemm_helper.left_outer_end(); ++i, ++x)
      result_size_vectors.get()[x] = size_vectors_.get()[i];
    for (unsigned int i = gemm_helper.right_outer_begin();
         i < gemm_helper.right_outer_end(); ++i, ++x)
      result_size_vectors.get()[x] = other.size_vectors_.get()[i];

    // Compute the number of inner ranks
    const unsigned int k_rank =
        gemm_helper.left_inner_end() - gemm_helper.left_inner_begin();

    // Construct the result norm tensor
    Tensor<value_type> result_norms(
        gemm_helper.make_result_range<typename Tensor<T>::range_type>(
            tile_norms_.range(), other.tile_norms_.range()),
        0);

    if (k_rank > 0u) {
      // Compute size vector
      const vector_type k_sizes = recursive_outer_product(
          size_vectors_.get() + gemm_helper.left_inner_begin(), k_rank,
          [](const vector_type& size_vector) -> const vector_type& {
            return size_vector;
          });

      // TODO: Make this faster. It can be done without using temporaries
      // for the arguments, but requires a custom matrix multiply.

      Tensor<value_type> left(tile_norms_.range());
      const size_type mk = M * K;
      auto left_op = [](const value_type left, const value_type right) {
        return left * right;
      };
      for (size_type i = 0ul; i < mk; i += K)
        math::vector_op(left_op, K, left.data() + i, tile_norms_.data() + i,
                        k_sizes.data());

      Tensor<value_type> right(other.tile_norms_.range());
      for (integer i = 0ul, k = 0; k < K; i += N, ++k) {
        const value_type factor = k_sizes[k];
        auto right_op = [=](const value_type arg) { return arg * factor; };
        math::vector_op(right_op, N, right.data() + i,
                        other.tile_norms_.data() + i);
      }

      result_norms = left.gemm(right, abs_factor, gemm_helper);

      // Hard zero tiles that are below the zero threshold.
      result_norms.inplace_unary(
          [threshold, &zero_tile_count](value_type& value) {
            if (value < threshold) {
              value = value_type(0);
              ++zero_tile_count;
            }
          });

    } else {
      // This is an outer product, so the inputs can be used directly
      math::outer_fill(M, N, tile_norms_.data(), other.tile_norms_.data(),
                       result_norms.data(),
                       [threshold, &zero_tile_count, abs_factor](
                           const value_type left, const value_type right) {
                         value_type norm = left * right * abs_factor;
                         if (norm < threshold) {
                           norm = value_type(0);
                           ++zero_tile_count;
                         }
                         return norm;
                       });
    }

    return SparseShape_(result_norms, result_size_vectors, zero_tile_count);
  }

  /// \tparam Factor The scaling factor type
  /// \note expression abs(Factor) must be well defined (by default, std::abs
  /// will be used)
  template <typename Factor>
  SparseShape_ gemm(const SparseShape_& other, const Factor factor,
                    const math::GemmHelper& gemm_helper,
                    const Permutation& perm) const {
    return gemm(other, factor, gemm_helper).perm(perm);
  }

  template <typename Archive,
            typename std::enable_if<madness::is_input_archive_v<
                std::decay_t<Archive>>>::type* = nullptr>
  void serialize(Archive& ar) {
    ar & tile_norms_;
    const unsigned int dim = tile_norms_.range().rank();
    // allocate size_vectors_
    size_vectors_ = std::move(std::shared_ptr<vector_type>(
        new vector_type[dim], std::default_delete<vector_type[]>()));
    for (unsigned d = 0; d != dim; ++d) ar & size_vectors_.get()[d];
    ar & zero_tile_count_;
  }

  template <typename Archive,
            typename std::enable_if<madness::is_output_archive_v<
                std::decay_t<Archive>>>::type* = nullptr>
  void serialize(Archive& ar) const {
    ar & tile_norms_;
    const unsigned int dim = tile_norms_.range().rank();
    for (unsigned d = 0; d != dim; ++d) ar & size_vectors_.get()[d];
    ar & zero_tile_count_;
  }

 private:
  template <typename Factor>
  static value_type to_abs_factor(const Factor factor) {
    using std::abs;
    const auto cast_abs_factor = static_cast<value_type>(abs(factor));
    TA_ASSERT(std::isfinite(cast_abs_factor));
    return cast_abs_factor;
  }

};  // class SparseShape

// Static member initialization
template <typename T>
typename SparseShape<T>::value_type SparseShape<T>::threshold_ =
    std::numeric_limits<T>::epsilon();

/// Add the shape to an output stream

/// \tparam T the numeric type supporting the type of \c shape
/// \param os The output stream
/// \param shape the SparseShape<T> object
/// \return A reference to the output stream
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const SparseShape<T>& shape) {
  os << "SparseShape<" << typeid(T).name() << ">:" << std::endl
     << shape.data() << std::endl;
  return os;
}

/// collective bitwise-compare-reduce for SparseShape objects

/// @param world the World object
/// @param[in] shape the SparseShape object
/// @return true if \c shape is bitwise identical across \c world
/// @note must be invoked on every rank of World
template <typename T>
bool is_replicated(World& world, const SparseShape<T>& shape) {
  const auto volume = shape.data().size();
  std::vector<T> data(shape.data().data(), shape.data().data() + volume);
  world.gop.max(data.data(), volume);
  bool result = true;
  for (size_t i = 0; i != data.size(); ++i) {
    if (data[i] != shape.data()[i]) {
      result = false;
      break;
    }
  }
  world.gop.logic_and(&result, 1);
  return result;
}

#ifndef TILEDARRAY_HEADER_ONLY

extern template class SparseShape<float>;

#endif  // TILEDARRAY_HEADER_ONLY

}  // namespace TiledArray

#endif  // TILEDARRAY_SPASE_SHAPE_H__INCLUDED
