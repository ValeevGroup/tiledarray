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
 *  disk_tile.h
 */

#ifndef TILEDARRAY_DISK_DISK_TILE_H__INCLUDED
#define TILEDARRAY_DISK_DISK_TILE_H__INCLUDED

#include <TiledArray/disk/store.h>
#include <TiledArray/error.h>
#include <TiledArray/tensor/type_traits.h>
#include <TiledArray/tile_interface/clone.h>
#include <TiledArray/type_traits.h>

#include <madness/world/buffer_archive.h>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace TiledArray {

/// A tile kept in a scratch file, loaded when it is needed.

/// DiskTile is a lazy tile: its evaluation type is \p Tile , and a
/// DistArray<DiskTile<Tile>, Policy> can appear wherever an expression
/// accepts an array argument. Evaluating a DiskTile (converting it to
/// \p Tile ) reads the tile from disk; the DiskTile itself keeps nothing in
/// memory, so every evaluation is a read and the memory of the evaluated
/// tile is released when the consumer is done with it.
///
/// A DiskTile is in one of three states:
/// - \e empty (default-constructed);
/// - <em>on disk</em>: it refers to a byte range of a detail::DiskStore, a
///   scratch file private to the process that wrote it. Copies share the
///   store, which lives until the last tile referring to it is gone;
/// - <em>in memory</em>: it holds a \p Tile . This is the state of a tile
///   received from another process.
///
/// The scratch file is local to the node of the process that wrote it and
/// cannot be reached from anywhere else. Accordingly, serializing an on-disk
/// tile serializes its \e data , and deserialization always yields an
/// in-memory tile: a tile never refers to a file of another process.
///
/// A DiskTile is immutable once constructed, hence safe to evaluate and to
/// serialize from several threads at once.
///
/// \tparam Tile the tile type this evaluates to; must be default
///         constructible and serializable by a MADNESS buffer archive
template <typename Tile>
class DiskTile {
 public:
  using eval_type = Tile;
  using value_type = typename Tile::value_type;
  using numeric_type = typename Tile::numeric_type;
  using scalar_type = typename Tile::scalar_type;
  using range_type = typename Tile::range_type;

  /// Constructs an empty tile
  DiskTile() = default;

  /// Writes \p tile to \p store ; the result is on disk.
  /// \param tile the tile to store
  /// \param store the scratch file to append the tile to
  /// \throw TiledArray::Exception if the write fails
  DiskTile(const Tile& tile, std::shared_ptr<detail::DiskStore> store) {
    if (!store)
      TA_EXCEPTION("DiskTile: constructed from a tile without a store");
    madness::archive::BufferOutputArchive count;
    count & tile;
    const std::size_t nbytes = count.size();
    const std::unique_ptr<unsigned char[]> buffer(new unsigned char[nbytes]);
    madness::archive::BufferOutputArchive out(buffer.get(), nbytes);
    out & tile;
    const std::size_t offset = store->append(buffer.get(), nbytes);
    extent_ = std::make_shared<const Extent>(std::move(store), offset, nbytes);
    detail::disk_io_count_tiles(1, 0);
  }

  /// Evaluates this tile: reads it from disk, or copies the tile held in
  /// memory. Either way the result shares no data with this tile or with any
  /// other evaluation of it, so the caller is free to modify it.
  /// \throw TiledArray::Exception if this tile is empty
  explicit operator Tile() const {
    // a deep copy: copying a tile may be shallow, and whoever evaluates a
    // lazy tile may work on the result in place
    if (mem_) return TiledArray::clone(*mem_);
    if (!extent_) TA_EXCEPTION("DiskTile: an empty tile cannot be evaluated");
    const auto buffer = read_bytes();
    madness::archive::BufferInputArchive in(buffer.get(), extent_->nbytes);
    Tile tile;
    in & tile;
    return tile;
  }

  /// \return true if this tile holds no data
  bool empty() const { return !extent_ && !mem_; }

  /// \return true if this tile's data is in a scratch file (of this process)
  bool on_disk() const { return static_cast<bool>(extent_); }

  /// \return the scratch file this tile is in; null unless on_disk()
  std::shared_ptr<detail::DiskStore> store() const {
    return extent_ ? extent_->store : nullptr;
  }

  /// \return the number of bytes this tile occupies in a MADNESS buffer
  ///         archive
  std::size_t serialized_size() const {
    std::size_t n = flag_size();
    if (extent_)
      n += extent_->nbytes;
    else if (mem_) {
      madness::archive::BufferOutputArchive count;
      count&* mem_;
      n += count.size();
    }
    return n;
  }

  /// (De)serializes this tile.

  /// Output: writes the tile's data. An on-disk tile is read from disk
  /// exactly once per (non-counting) serialization, and not at all when the
  /// archive only counts bytes, as MADNESS does to size a message before
  /// filling it. Input: the result is in memory (or empty).
  template <typename Archive>
  void serialize(Archive& ar) {
    if constexpr (madness::is_output_archive_v<Archive>) {
      const bool has_data = !empty();
      ar & has_data;
      if (!has_data) return;
      if (mem_) {
        ar&* mem_;
        return;
      }
      // on disk: the stored bytes are exactly what a buffer archive holds
      // for the tile, so a buffer archive takes them as they are
      if constexpr (std::is_same_v<std::remove_cv_t<Archive>,
                                   madness::archive::BufferOutputArchive>) {
        if (ar.count_only()) {
          // only the size is wanted: known since the tile was written
          ar.store(static_cast<const unsigned char*>(nullptr),
                   static_cast<long>(extent_->nbytes));
        } else {
          const auto buffer = read_bytes();
          ar.store(buffer.get(), static_cast<long>(extent_->nbytes));
        }
      } else {
        const Tile tile = static_cast<Tile>(*this);
        ar & tile;
      }
    } else {
      bool has_data = false;
      ar & has_data;
      extent_.reset();
      if (has_data) {
        Tile tile;
        ar & tile;
        mem_ = std::make_shared<const Tile>(std::move(tile));
      } else
        mem_.reset();
    }
  }

 private:
  /// \return the number of bytes of the "has data" flag in a buffer archive
  static std::size_t flag_size() {
    madness::archive::BufferOutputArchive count;
    const bool flag = true;
    count & flag;
    return count.size();
  }

  /// Reads this (on-disk) tile's bytes from its store.
  std::unique_ptr<unsigned char[]> read_bytes() const {
    std::unique_ptr<unsigned char[]> buffer(new unsigned char[extent_->nbytes]);
    extent_->store->read(extent_->offset, buffer.get(), extent_->nbytes);
    detail::disk_io_count_tiles(0, 1);
    return buffer;
  }

  /// The bytes of a tile in a store. Shared by the copies of a tile; gives
  /// the bytes back to the store when the last copy is gone.
  struct Extent {
    Extent(std::shared_ptr<detail::DiskStore> s, std::size_t o, std::size_t n)
        : store(std::move(s)), offset(o), nbytes(n) {}
    ~Extent() { store->release(offset, nbytes); }
    Extent(const Extent&) = delete;
    Extent& operator=(const Extent&) = delete;
    std::shared_ptr<detail::DiskStore> store;
    std::size_t offset;
    std::size_t nbytes;
  };

  std::shared_ptr<const Extent> extent_;  ///< set iff on disk
  std::shared_ptr<const Tile> mem_;       ///< set iff in memory
};

namespace detail {

template <typename Tile>
struct is_contiguous_tensor_helper<DiskTile<Tile>>
    : public is_contiguous_tensor_helper<Tile> {};

}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_DISK_DISK_TILE_H__INCLUDED
