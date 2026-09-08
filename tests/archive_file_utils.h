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

#ifndef TILEDARRAY_TEST_ARCHIVE_FILE_UTILS_H__INCLUDED
#define TILEDARRAY_TEST_ARCHIVE_FILE_UTILS_H__INCLUDED

#include <madness/world/madness_exception.h>

#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <string>

namespace TiledArray::test {

/// The file a `madness::archive::ParallelOutputArchive` actually writes for
/// \p prefix_name on \p rank -- `<prefix>.<rank>`, rank zero-padded to five
/// digits. A parallel archive never creates \p prefix_name itself, so removing
/// the prefix does *not* clean up after one; remove this instead.
/// \param prefix_name The archive prefix handed to the archive constructor
/// \param rank The writing rank
/// \return The per-rank file name
inline std::string to_parallel_archive_file_name(const char* prefix_name,
                                                 int rank) {
  char buf[256];
  MADNESS_ASSERT(strlen(prefix_name) + 7 <= sizeof(buf));
  snprintf(buf, sizeof(buf), "%s.%5.5d", prefix_name, rank);
  return buf;
}

/// Replace the trailing `XXXXXX` in \p name_template with a unique suffix.

/// Uses mkstemp + close + remove, so the resulting *name* is free for the
/// caller to open itself (single-file archive) or to use as a prefix for
/// per-rank files (parallel archive); no file is left behind by this call.
/// Unlike mktemp(3) -- which clang/macOS flags as deprecated -- this is
/// race-free against other in-process callers.
/// \param[in,out] name_template A writable `...XXXXXX` template
inline void make_unique_filename_template(char* name_template) {
  const int fd = mkstemp(name_template);
  MADNESS_ASSERT(fd != -1);
  ::close(fd);
  std::remove(name_template);
}

}  // namespace TiledArray::test

#endif  // TILEDARRAY_TEST_ARCHIVE_FILE_UTILS_H__INCLUDED
