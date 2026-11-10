/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
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
 */

#ifndef TILEDARRAY_TEST_SPARSE_THRESHOLD_GUARD_H__INCLUDED
#define TILEDARRAY_TEST_SPARSE_THRESHOLD_GUARD_H__INCLUDED

#include "TiledArray/sparse_shape.h"

namespace TiledArray::test {

/// Changes the default threshold of `SparseShape<T>` and, when destroyed,
/// restores the one in effect when it was constructed, however the test exits.

/// A test that only builds shapes can change the threshold on its own rank. A
/// test that builds `DistArray`s must change it on every rank at once: the
/// threshold is read by tasks and by the shape replication in `TensorImpl`, so
/// use the constructor that takes a `World`, which changes it through
/// `World::gop.serial_invoke`.
template <typename T>
class ScopedSparseThreshold {
 public:
  /// sets the threshold to \p t on this rank only
  explicit ScopedSparseThreshold(T t)
      : prior_(SparseShape<T>::threshold()), world_(nullptr) {
    set(t);
  }

  /// sets the threshold to \p t on every rank of \p world
  /// \note collective on \p world, and so is the destructor
  ScopedSparseThreshold(T t, World& world)
      : prior_(SparseShape<T>::threshold()), world_(&world) {
    set(t);
  }

  ScopedSparseThreshold(const ScopedSparseThreshold&) = delete;
  ScopedSparseThreshold& operator=(const ScopedSparseThreshold&) = delete;

  ~ScopedSparseThreshold() { set(prior_); }

  /// sets the threshold to \p t, on the ranks this was constructed for; the
  /// threshold restored on destruction does not change
  void set(T t) const {
    if (world_)
      world_->gop.serial_invoke([t] { SparseShape<T>::threshold(t); });
    else
      SparseShape<T>::threshold(t);
  }

 private:
  const T prior_;       ///< the threshold restored on destruction
  World* const world_;  ///< if not null, the threshold is changed on every
                        ///< rank of this World
};

}  // namespace TiledArray::test

#endif  // TILEDARRAY_TEST_SPARSE_THRESHOLD_GUARD_H__INCLUDED
