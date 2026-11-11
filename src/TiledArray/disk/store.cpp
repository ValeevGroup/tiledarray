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
 *  store.cpp
 */

#include <TiledArray/disk/store.h>

#include <TiledArray/error.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace TiledArray {

namespace {

std::mutex& scratch_dir_mutex() {
  static std::mutex mtx;
  return mtx;
}

std::filesystem::path& scratch_dir_setting() {
  static std::filesystem::path dir;
  return dir;
}

struct Counters {
  std::atomic<std::size_t> tiles_written{0}, tiles_read{0}, bytes_written{0},
      bytes_read{0};
  std::atomic<long long> ns_writing{0}, ns_reading{0};
};

Counters& counters() {
  static Counters c;
  return c;
}

[[noreturn]] void io_failure(const std::string& what, int err) {
  throw TiledArray::Exception("TiledArray: disk-backed array I/O: " + what +
                              ": " + std::strerror(err));
}

long long elapsed_ns(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

}  // namespace

void set_disk_scratch_dir(std::filesystem::path dir) {
  // a directory that cannot be used is reported here, to the caller, rather
  // than when the first tile is written, which is typically inside a task
  if (!dir.empty()) detail::check_disk_scratch_dir(dir);
  std::scoped_lock lock(scratch_dir_mutex());
  scratch_dir_setting() = std::move(dir);
}

std::filesystem::path disk_scratch_dir() {
  {
    std::scoped_lock lock(scratch_dir_mutex());
    if (!scratch_dir_setting().empty()) return scratch_dir_setting();
  }
  if (const char* env = std::getenv("TA_DISK_SCRATCH_DIR"); env && *env)
    return env;
  return std::filesystem::temp_directory_path();
}

DiskIOStats disk_io_stats() {
  auto const& c = counters();
  DiskIOStats s;
  s.tiles_written = c.tiles_written;
  s.tiles_read = c.tiles_read;
  s.bytes_written = c.bytes_written;
  s.bytes_read = c.bytes_read;
  s.seconds_writing = 1e-9 * static_cast<double>(c.ns_writing.load());
  s.seconds_reading = 1e-9 * static_cast<double>(c.ns_reading.load());
  return s;
}

void reset_disk_io_stats() {
  auto& c = counters();
  c.tiles_written = c.tiles_read = c.bytes_written = c.bytes_read = 0;
  c.ns_writing = c.ns_reading = 0;
}

namespace detail {

std::atomic<std::size_t>& disk_io_chunk() {
  static std::atomic<std::size_t> chunk{std::size_t{1} << 30};
  return chunk;
}

std::filesystem::path disk_scratch_dir_setting() {
  std::scoped_lock lock(scratch_dir_mutex());
  return scratch_dir_setting();
}

void disk_io_count_tiles(std::size_t n_written, std::size_t n_read) {
  counters().tiles_written += n_written;
  counters().tiles_read += n_read;
}

void check_disk_scratch_dir(const std::filesystem::path& dir) {
  const std::string what =
      "TiledArray: disk-backed array I/O: scratch directory " + dir.string();
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec))
    throw TiledArray::Exception(what + " is not an existing directory");
  if (::access(dir.c_str(), W_OK | X_OK) != 0)
    throw TiledArray::Exception(what +
                                " is not writable: " + std::strerror(errno));
}

std::uint64_t disk_checksum(const void* data, std::size_t n) {
  // every step is a bijection of h for a given word, so data that differs
  // in one word (anywhere) never has the same checksum; a word at a time,
  // to stay well ahead of the disk
  const unsigned char* p = static_cast<const unsigned char*>(data);
  std::uint64_t h = 0x9e3779b97f4a7c15ull ^ static_cast<std::uint64_t>(n);
  auto const mix = [&h](std::uint64_t w) {
    h = (h ^ w) * 0xff51afd7ed558ccdull;
    h ^= h >> 32;
  };
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    std::uint64_t w;
    std::memcpy(&w, p + i, 8);
    mix(w);
  }
  if (i < n) {
    std::uint64_t w = 0;
    std::memcpy(&w, p + i, n - i);
    mix(w);
  }
  return h;
}

DiskStore::DiskStore() {
  std::filesystem::path dir;
  try {
    dir = disk_scratch_dir();
  } catch (const std::filesystem::filesystem_error& e) {
    // no usable system temporary directory
    throw TiledArray::Exception(
        std::string("TiledArray: disk-backed array I/O: cannot determine the "
                    "scratch directory: ") +
        e.what());
  }
  std::string templ = (dir / "ta-disk-XXXXXX").string();
  std::vector<char> name(templ.begin(), templ.end());
  name.push_back('\0');
  // close-on-exec, set as the file is created: a child process must not
  // inherit the descriptor, which would let it read the data and would keep
  // the file's space allocated for as long as the child lives
#ifdef O_TMPFILE
  // where the filesystem can do it, a file that never has a name at all
  fd_ = ::open(dir.c_str(), O_TMPFILE | O_RDWR | O_CLOEXEC, S_IRUSR | S_IWUSR);
  if (fd_ != -1) return;
#endif
  fd_ = ::mkostemp(name.data(), O_CLOEXEC);
  if (fd_ == -1)
    io_failure("cannot create a scratch file in " + dir.string(), errno);
  // no name from here on: gone when this process is
  if (::unlink(name.data()) != 0) {
    const int err = errno;
    ::close(fd_);
    fd_ = -1;
    io_failure("cannot unlink scratch file " + std::string(name.data()), err);
  }
}

DiskStore::~DiskStore() {
  if (fd_ != -1) ::close(fd_);
}

void DiskStore::add_free(std::size_t offset, std::size_t n) {
  free_.emplace(offset, n);
  free_by_size_.emplace(n, offset);
}

void DiskStore::remove_free(std::map<std::size_t, std::size_t>::iterator it) {
  free_by_size_.erase({it->second, it->first});
  free_.erase(it);
}

std::size_t DiskStore::write(const void* data, std::size_t n) {
  // reserve the byte range, then write outside the lock: concurrent writes
  // go to disjoint ranges
  std::size_t offset;
  {
    std::scoped_lock lock(mtx_);
    // best fit: the smallest released range that is large enough (the
    // lowest of those of equal size)
    auto const fit = free_by_size_.lower_bound({n, 0});
    if (n != 0 && fit != free_by_size_.end()) {
      offset = fit->second;
      const std::size_t left = fit->first - n;
      remove_free(free_.find(offset));
      if (left != 0) add_free(offset + n, left);
    } else {
      offset = size_;
      size_ += n;
    }
    in_use_ += n;
  }
  auto const t0 = std::chrono::steady_clock::now();
  const char* p = static_cast<const char*>(data);
  std::size_t done = 0;
  while (done < n) {
    const std::size_t request =
        std::min(n - done, std::max<std::size_t>(disk_io_chunk().load(), 1));
    const ssize_t w =
        ::pwrite(fd_, p + done, request, static_cast<off_t>(offset + done));
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) {
      // give the reserved range back: the store stays usable
      const int err = w < 0 ? errno : ENOSPC;
      release(offset, n);
      io_failure("write of " + std::to_string(n) + " bytes failed", err);
    }
    done += static_cast<std::size_t>(w);
  }
  counters().bytes_written += n;
  counters().ns_writing += elapsed_ns(t0);
  return offset;
}

void DiskStore::read(std::size_t offset, void* data, std::size_t n) const {
  if (n == 0) return;
  {
    // the range must be in space handed out by write() and not released
    std::scoped_lock lock(mtx_);
    bool valid = offset + n <= size_ && offset + n > offset;
    if (valid) {
      auto const next = free_.upper_bound(offset);  // first range past offset
      if (next != free_.end() && next->first < offset + n) valid = false;
      if (valid && next != free_.begin()) {
        auto const prev = std::prev(next);
        if (prev->first + prev->second > offset) valid = false;
      }
    }
    if (!valid)
      throw TiledArray::Exception(
          "TiledArray: disk-backed array I/O: read of a range that is not in "
          "use (" +
          std::to_string(n) + " bytes at offset " + std::to_string(offset) +
          ")");
  }
  auto const t0 = std::chrono::steady_clock::now();
  char* p = static_cast<char*>(data);
  std::size_t done = 0;
  while (done < n) {
    const std::size_t request =
        std::min(n - done, std::max<std::size_t>(disk_io_chunk().load(), 1));
    const ssize_t r =
        ::pread(fd_, p + done, request, static_cast<off_t>(offset + done));
    if (r < 0) {
      if (errno == EINTR) continue;
      io_failure("read of " + std::to_string(n) + " bytes failed", errno);
    }
    if (r == 0)
      throw TiledArray::Exception(
          "TiledArray: disk-backed array I/O: read past the end of a scratch "
          "file (" +
          std::to_string(n) + " bytes at offset " + std::to_string(offset) +
          ")");
    done += static_cast<std::size_t>(r);
  }
  counters().bytes_read += n;
  counters().ns_reading += elapsed_ns(t0);
}

void DiskStore::release(std::size_t offset, std::size_t n) {
  if (n == 0) return;
  std::scoped_lock lock(mtx_);
  TA_ASSERT(offset + n <= size_ && n <= in_use_);
  in_use_ -= n;
  // merge with the free range that follows, then with the one that precedes
  auto next = free_.lower_bound(offset);
  TA_ASSERT(next == free_.end() || offset + n <= next->first);
  if (next != free_.end() && next->first == offset + n) {
    n += next->second;
    auto const merged = next++;
    remove_free(merged);
  }
  if (next != free_.begin()) {
    auto prev = std::prev(next);
    TA_ASSERT(prev->first + prev->second <= offset);
    if (prev->first + prev->second == offset) {
      offset = prev->first;
      n += prev->second;
      remove_free(prev);
    }
  }
  add_free(offset, n);
}

std::size_t DiskStore::size() const {
  std::scoped_lock lock(mtx_);
  return size_;
}

std::size_t DiskStore::bytes_in_use() const {
  std::scoped_lock lock(mtx_);
  return in_use_;
}

}  // namespace detail
}  // namespace TiledArray
