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
 *  store.h
 */

#ifndef TILEDARRAY_DISK_STORE_H__INCLUDED
#define TILEDARRAY_DISK_STORE_H__INCLUDED

#include <cstddef>
#include <filesystem>
#include <mutex>

namespace TiledArray {

/// Sets the directory in which disk-backed arrays keep their scratch files.

/// The scratch directory is, in order of precedence: the last non-empty
/// value set here; else the value of the \c TA_DISK_SCRATCH_DIR environment
/// variable; else \c std::filesystem::temp_directory_path() . The directory
/// must be on a filesystem local to this process's node: a scratch file is
/// private to the process that created it.
/// \param dir the scratch directory; an empty path reverts to the
///        environment variable / system default
void set_disk_scratch_dir(std::filesystem::path dir);

/// \return the directory in which a scratch file created now would be placed
///         (see set_disk_scratch_dir() )
std::filesystem::path disk_scratch_dir();

/// Disk I/O done by disk-backed arrays in this process
struct DiskIOStats {
  std::size_t tiles_written = 0;  ///< # of tiles written to disk
  std::size_t tiles_read = 0;     ///< # of tiles read from disk
  std::size_t bytes_written = 0;  ///< # of bytes written to disk
  std::size_t bytes_read = 0;     ///< # of bytes read from disk
  double seconds_writing = 0;     ///< time spent in writes, summed over threads
  double seconds_reading = 0;     ///< time spent in reads, summed over threads
};

/// \return the disk I/O done by this process since the last
///         reset_disk_io_stats() (or since start)
DiskIOStats disk_io_stats();

/// Zeroes the counters reported by disk_io_stats()
void reset_disk_io_stats();

namespace detail {

/// The largest single read/write request made to the operating system;
/// longer transfers are split. Exposed so that tests can exercise the
/// splitting without multi-GiB data.
std::size_t& disk_io_chunk();

/// Records \p n tiles written (\p n may be 0) and \p m tiles read.
void disk_io_count_tiles(std::size_t n_written, std::size_t n_read);

/// A scratch file private to this process.

/// The file is created in disk_scratch_dir() and unlinked immediately, so it
/// has no name: no other process can open it, and its space is reclaimed by
/// the operating system when this object is destroyed or the process dies,
/// however it dies. The store is append-only. All members are thread-safe.
class DiskStore {
 public:
  /// Creates the scratch file.
  /// \throw TiledArray::Exception naming the scratch directory if the file
  ///        cannot be created
  DiskStore();
  ~DiskStore();
  DiskStore(const DiskStore&) = delete;
  DiskStore& operator=(const DiskStore&) = delete;

  /// Appends \p n bytes starting at \p data .
  /// \return the offset of the appended bytes
  /// \throw TiledArray::Exception if the write fails (e.g. the disk is full)
  std::size_t append(const void* data, std::size_t n);

  /// Reads \p n bytes at \p offset into \p data .
  /// \throw TiledArray::Exception if the range was not appended in full or
  ///        the read fails
  void read(std::size_t offset, void* data, std::size_t n) const;

  /// \return the number of bytes appended so far
  std::size_t size() const;

 private:
  int fd_ = -1;
  mutable std::mutex mtx_;  // guards size_
  std::size_t size_ = 0;
};

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_DISK_STORE_H__INCLUDED
