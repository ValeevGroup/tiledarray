/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  perm_index.h
 *  Oct 10, 2014
 *
 */

#ifndef TILEDARRAY_PERM_INDEX_H__INCLUDED
#define TILEDARRAY_PERM_INDEX_H__INCLUDED

#include <TiledArray/range.h>

namespace TiledArray {
namespace detail {

/// A functor that permutes ordinal indices

/// The purpose of this functor is to allow fast, repeated permutations of
/// ordinal indices.
class PermIndex {
  std::size_t* weights_;  ///< A pointer that stores both the input and
                          ///< output weights (or strides).
  unsigned int
      ndim_;  ///< The number of dimensions in the coordinate index space

 public:
  /// Default constructor
  PermIndex() : weights_(NULL), ndim_(0) {}

  /// Construct permuting functor

  /// \param range The input range of ordinal indices
  PermIndex(const Range& range, const Permutation& perm)
      : weights_(NULL), ndim_(perm.size()) {
    if (ndim_ > 0) {
      // Check the input data
      TA_ASSERT(range.rank() == perm.size());

      // Construct the inverse permutation
      const Permutation inv_perm_ = -perm;

      // Allocate memory for this object
      weights_ = static_cast<std::size_t*>(
          malloc((ndim_ + ndim_) * sizeof(std::size_t)));
      if (!weights_) throw std::bad_alloc();

      // Construct MADNESS_RESTRICTed pointers to the input data
      const auto* MADNESS_RESTRICT const inv_perm = &inv_perm_.data().front();
      const auto* MADNESS_RESTRICT const range_size = range.extent_data();
      const auto* MADNESS_RESTRICT const range_weight = range.stride_data();

      // Construct MADNESS_RESTRICTed pointers to the object data
      std::size_t* MADNESS_RESTRICT const input_weight = weights_;
      std::size_t* MADNESS_RESTRICT const output_weight = weights_ + ndim_;

      // Initialize input and output weights
      std::size_t volume = 1ul;
      for (int i = int(ndim_) - 1; i >= 0; --i) {
        // Load input data for iteration i.
        const auto inv_perm_i = inv_perm[i];
        const auto weight = range_weight[i];
        const auto size = range_size[inv_perm_i];

        // Store the input and output weights
        output_weight[inv_perm_i] = volume;
        volume *= size;
        input_weight[i] = weight;
      }
    }
  }

  PermIndex(const PermIndex& other) : weights_(NULL), ndim_(other.ndim_) {
    if (ndim_) {
      // Allocate memory for this object
      weights_ = static_cast<std::size_t*>(
          malloc((ndim_ + ndim_) * sizeof(std::size_t)));
      if (!weights_) throw std::bad_alloc();

      // Copy data
      memcpy(weights_, other.weights_, (ndim_ + ndim_) * sizeof(std::size_t));
    }
  }

  ~PermIndex() {
    free(weights_);
    weights_ = NULL;
  }

  PermIndex& operator=(const PermIndex& other) {
    // Deallocate memory
    if (ndim_ && (ndim_ != other.ndim_)) {
      free(weights_);
      weights_ = NULL;
    }

    const std::size_t bytes = (other.ndim_ + other.ndim_) * sizeof(std::size_t);

    if (!weights_ && bytes) {
      // Allocate new memory
      weights_ = static_cast<std::size_t*>(malloc(bytes));
      if (!weights_) throw std::bad_alloc();
    }

    // copy the data (safe if ndim_ == 0)
    ndim_ = other.ndim_;
    memcpy(weights_, other.weights_, bytes);

    return *this;
  }

  /// Dimension accessor

  /// \return The dimension of the indices that can be permuted
  int dim() const { return ndim_; }

  /// Data accessor

  /// \return A pointer to the result data
  const std::size_t* data() const { return weights_; }

  /// Compute the permuted index for the current block
  std::size_t operator()(std::size_t index) const {
    TA_ASSERT(ndim_);
    TA_ASSERT(weights_);

    // Construct MADNESS_RESTRICTed pointers to data
    const std::size_t* MADNESS_RESTRICT const input_weight = weights_;
    const std::size_t* MADNESS_RESTRICT const output_weight = weights_ + ndim_;

    // create result index
    std::size_t perm_index = 0ul;

    for (unsigned int i = 0u; i < ndim_; ++i) {
      const std::size_t input_weight_i = input_weight[i];
      const std::size_t output_weight_i = output_weight[i];
      perm_index += index / input_weight_i * output_weight_i;
      index %= input_weight_i;
    }

    return perm_index;
  }

  // Check for valid permutation
  operator bool() const { return ndim_; }
};  // class PermIndex

}  // namespace detail
}  // namespace TiledArray

#endif  // MADNESS_PERM_INDEX_H__INCLUDED
TILEDARRAY_PERM_INDEX_H__INCLUDED
