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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <utility>

namespace TiledArray {

/// Sets the directory in which disk-backed arrays keep their scratch files.

/// The scratch directory is, in order of precedence: the last non-empty
/// value set here; else the value of the \c TA_DISK_SCRATCH_DIR environment
/// variable; else \c std::filesystem::temp_directory_path() . The directory
/// must be on a filesystem local to this process's node: a scratch file is
/// used only by the process that created it.
/// \param dir the scratch directory; an empty path reverts to the
///        environment variable / system default
/// \throw TiledArray::Exception if \p dir is not empty and is not an existing
///        directory this process can create files in; the setting is then
///        unchanged
/// \note \c TA_DISK_SCRATCH_DIR is checked the same way by
///       TiledArray::initialize()
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
/// longer transfers are split (a value of 0 is taken as 1). Exposed so that
/// tests can exercise the splitting without multi-GiB data.
std::atomic<std::size_t>& disk_io_chunk();

/// \return the directory given to set_disk_scratch_dir(), empty if none is
///         in effect. Unlike disk_scratch_dir() this is the setting itself,
///         not what it resolves to: it is what to hand back to
///         set_disk_scratch_dir() to restore an earlier state.
std::filesystem::path disk_scratch_dir_setting();

/// Records \p n_written tiles written and \p n_read tiles read (either may
/// be 0).
void disk_io_count_tiles(std::size_t n_written, std::size_t n_read);

/// Checks that \p dir can hold scratch files.
/// \throw TiledArray::Exception naming \p dir if it is not an existing
///        directory this process can create files in
void check_disk_scratch_dir(const std::filesystem::path& dir);

/// \return a checksum of the \p n bytes at \p data , used to verify that
///         what is read from a scratch file is what was written to it. Not
///         cryptographic: it guards against storage errors, not tampering.
std::uint64_t disk_checksum(const void* data, std::size_t n);

/// A scratch file of this process.

/// The file is created in disk_scratch_dir() without a name (where the
/// operating system cannot do that, it is unlinked immediately): it is not
/// visible in the directory, it is not inherited by child processes, and its
/// space is reclaimed by the operating system when this object is destroyed
/// or the process dies, however it dies.
/// \warning The file is not protected from other processes of the same user
/// (or a privileged one), which can reach an open file of this process
/// through the operating system (e.g. \c /proc/<pid>/fd on Linux). The data
/// is not encrypted, and space that is released is not erased.
///
/// Data is written with write(), which returns where it went, and its
/// space is given back with release(); released space is reused by later
/// writes before the file is extended, so the file does not grow while data
/// of the same sizes is replaced. All members are thread-safe.
class DiskStore {
 public:
  /// Creates the scratch file. Its descriptor is not inherited by child
  /// processes.
  /// \throw TiledArray::Exception naming the scratch directory if the file
  ///        cannot be created, or if there is no scratch directory to use
  DiskStore();
  ~DiskStore();
  DiskStore(const DiskStore&) = delete;
  DiskStore& operator=(const DiskStore&) = delete;

  /// Writes \p n bytes starting at \p data , into released space if some
  /// fits (the smallest range that does), else at the end of the file.
  /// \note the cost of finding released space is logarithmic in the number
  ///       of released ranges. The file itself never shrinks.
  /// \return the offset of the written bytes
  /// \throw TiledArray::Exception if the write fails (e.g. the disk is full)
  /// \warning an error the operating system reports only later, when it
  ///       moves the data to the device, is not seen here; DiskTile detects
  ///       its effect when the data is read (see disk_checksum())
  std::size_t write(const void* data, std::size_t n);

  /// Gives back the \p n bytes at \p offset , which must be a range (or part
  /// of one) returned by write() and not yet released. They must not be
  /// read afterwards.
  void release(std::size_t offset, std::size_t n);

  /// Reads \p n bytes at \p offset into \p data .
  /// \throw TiledArray::Exception if the range is not within space returned
  ///        by write() and not released, or the read fails
  /// \note a range whose write() has not returned yet cannot be told from
  ///       one that has: its offset is not known to anyone until then
  void read(std::size_t offset, void* data, std::size_t n) const;

  /// \return the extent of the file: the end of the last range handed out by
  ///         write() so far
  std::size_t size() const;

  /// \return the number of bytes written and not released
  std::size_t bytes_in_use() const;

 private:
  int fd_ = -1;
  mutable std::mutex mtx_;  // guards the members below
  std::size_t size_ = 0;
  std::size_t in_use_ = 0;
  std::map<std::size_t, std::size_t> free_;  // offset -> length, coalesced
  std::set<std::pair<std::size_t, std::size_t>>
      free_by_size_;  // (length, offset) of the ranges in free_

  // add/remove a released range to/from both indices; mtx_ must be held
  void add_free(std::size_t offset, std::size_t n);
  void remove_free(std::map<std::size_t, std::size_t>::iterator it);
};

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_DISK_STORE_H__INCLUDED
