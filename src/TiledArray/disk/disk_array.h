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
 *
 *  disk_array.h
 */

#ifndef TILEDARRAY_DISK_DISK_ARRAY_H__INCLUDED
#define TILEDARRAY_DISK_DISK_ARRAY_H__INCLUDED

#include <TiledArray/disk/disk_tile.h>
#include <TiledArray/disk/store.h>
#include <TiledArray/dist_array.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace TiledArray {

/// An array whose tiles are kept in scratch files and loaded when needed.

/// A DiskArray can be used as an argument of an expression like any other
/// array; each of its tiles is read from disk when the expression needs it
/// and released when it is done with it. To bring the data back into
/// memory assign it to an ordinary array, whole
/// \code
/// DistArray<Tile, Policy> a;
/// a("i,j") = d("i,j");
/// \endcode
/// or by tile block, which reads only the tiles of the block:
/// \code
/// a("i,j") = d("i,j").block(lo, hi);
/// \endcode
/// A DiskArray is produced by to_disk() or tile by tile with a
/// DiskArrayWriter. Its tiles are written once and never modified.
///
/// Every process keeps the tiles it owns in a scratch file on its own node
/// (see set_disk_scratch_dir()); the files are private to the process, are
/// gone when the array is, and do not outlive the process.
template <typename Tile, typename Policy>
using DiskArray = DistArray<DiskTile<Tile>, Policy>;

/// Builds a DiskArray tile by tile, so that the data never has to be in
/// memory all at once.

/// Construct one on every process (collectively, like a DistArray), then on
/// each process set() the tiles that process owns. The scratch file of a
/// process is created by its first set(): a process that sets no tile
/// creates no file. Copies of a writer refer to the same array.
template <typename Tile, typename Policy>
class DiskArrayWriter {
 public:
  using array_type = DiskArray<Tile, Policy>;
  using shape_type = typename array_type::shape_type;
  using pmap_interface = typename array_type::pmap_interface;

  /// \param world the world of the array
  /// \param trange the tiled range of the array
  /// \param shape the shape of the array; for a sparse policy it is final:
  ///        exactly its nonzero tiles are to be set
  /// \param pmap the process map of the array (default: the default map)
  DiskArrayWriter(World& world, const TiledRange& trange,
                  const shape_type& shape,
                  const std::shared_ptr<const pmap_interface>& pmap = {})
      : array_(world, trange, shape, pmap), state_(std::make_shared<State>()) {}

  /// Writes a tile to disk and makes it tile \p ord of the array.

  /// May be called from several threads at once (for different tiles).
  /// \param ord the ordinal index of the tile; must be owned by this process
  ///        and nonzero in the shape, and must not have been set before
  /// \param tile the tile; it is not kept
  /// \throw TiledArray::Exception if the tile cannot be written
  void set(std::size_t ord, const Tile& tile) {
    TA_ASSERT(array_.is_local(ord));
    TA_ASSERT(!array_.is_zero(ord));
    array_.set(ord, DiskTile<Tile>(tile, state_->store()));
  }

  /// \return the array; tiles not set (yet) are unset
  const array_type& array() const { return array_; }

 private:
  struct State {
    std::mutex mtx;
    std::shared_ptr<detail::DiskStore> store_;
    /// \return this process's scratch file, created on first use
    std::shared_ptr<detail::DiskStore> store() {
      std::scoped_lock lock(mtx);
      if (!store_) store_ = std::make_shared<detail::DiskStore>();
      return store_;
    }
  };

  array_type array_;
  std::shared_ptr<State> state_;
};

/// Copies an array to disk.

/// The result has the tiled range, shape and process map of \p array ; every
/// process writes the tiles it owns to a scratch file on its own node. This
/// is a collective operation. \p array itself is not changed: to release its
/// memory, drop it (and every other reference to its tiles) afterwards. At no
/// point is a second copy of the data in memory.
/// \param array the array to copy
/// \return the disk-backed copy
/// \throw TiledArray::Exception if a tile cannot be written
template <typename Tile, typename Policy>
DiskArray<Tile, Policy> to_disk(const DistArray<Tile, Policy>& array) {
  DiskArrayWriter<Tile, Policy> writer(array.world(), array.trange(),
                                       array.shape(), array.pmap());
  for (auto it = array.pmap()->begin(); it != array.pmap()->end(); ++it) {
    const auto ord = *it;
    if (array.is_zero(ord)) continue;
    writer.set(ord, array.find_local(ord).get());
  }
  array.world().gop.fence();
  return writer.array();
}

}  // namespace TiledArray

#endif  // TILEDARRAY_DISK_DISK_ARRAY_H__INCLUDED
