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
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

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

std::size_t& disk_io_chunk() {
  static std::size_t chunk = std::size_t{1} << 30;
  return chunk;
}

void disk_io_count_tiles(std::size_t n_written, std::size_t n_read) {
  counters().tiles_written += n_written;
  counters().tiles_read += n_read;
}

DiskStore::DiskStore() {
  auto const dir = disk_scratch_dir();
  std::string templ = (dir / "ta-disk-XXXXXX").string();
  std::vector<char> name(templ.begin(), templ.end());
  name.push_back('\0');
  fd_ = ::mkstemp(name.data());
  if (fd_ == -1)
    io_failure("cannot create a scratch file in " + dir.string(), errno);
  // no name from here on: private to this process, gone when it is
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

std::size_t DiskStore::append(const void* data, std::size_t n) {
  // reserve the byte range, then write outside the lock: concurrent appends
  // go to disjoint ranges
  std::size_t offset;
  {
    std::scoped_lock lock(mtx_);
    auto it = free_.begin();
    while (it != free_.end() && it->second < n) ++it;  // first fit
    if (it != free_.end()) {
      offset = it->first;
      const std::size_t left = it->second - n;
      free_.erase(it);
      if (left != 0) free_.emplace(offset + n, left);
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
    const std::size_t request = std::min(n - done, disk_io_chunk());
    const ssize_t w =
        ::pwrite(fd_, p + done, request, static_cast<off_t>(offset + done));
    if (w < 0) {
      if (errno == EINTR) continue;
      io_failure("write of " + std::to_string(n) + " bytes failed", errno);
    }
    done += static_cast<std::size_t>(w);
  }
  counters().bytes_written += n;
  counters().ns_writing += elapsed_ns(t0);
  return offset;
}

void DiskStore::read(std::size_t offset, void* data, std::size_t n) const {
  auto const t0 = std::chrono::steady_clock::now();
  char* p = static_cast<char*>(data);
  std::size_t done = 0;
  while (done < n) {
    const std::size_t request = std::min(n - done, disk_io_chunk());
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
    next = free_.erase(next);
  }
  if (next != free_.begin()) {
    auto prev = std::prev(next);
    TA_ASSERT(prev->first + prev->second <= offset);
    if (prev->first + prev->second == offset) {
      prev->second += n;
      return;
    }
  }
  free_.emplace(offset, n);
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
