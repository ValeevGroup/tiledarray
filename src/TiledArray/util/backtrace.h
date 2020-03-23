//
// backtrace.h
//
// Copyright (C) 1996 Limit Point Systems, Inc.
//
// Author: Curtis Janssen <cljanss@limitpt.com>
// Maintainer: LPS
//
// This file is part of the SC Toolkit.
//
// The SC Toolkit is free software; you can redistribute it and/or modify
// it under the terms of the GNU Library General Public License as published by
// the Free Software Foundation; either version 2, or (at your option)
// any later version.
//
// The SC Toolkit is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Library General Public License for more details.
//
// You should have received a copy of the GNU Library General Public License
// along with the SC Toolkit; see the file COPYING.LIB.  If not, write to
// the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
//
// The U.S. Government is granted a limited license as per AL 91-7.
//

#ifndef MPQC4_SRC_MPQC_UTIL_CORE_BACKTRACE_H_
#define MPQC4_SRC_MPQC_UTIL_CORE_BACKTRACE_H_

#include <string>
#include <vector>

namespace TiledArray {
namespace detail {
/**
 * Creates a backtrace of a running program/thread. Example of use:
 * @code
 * void make_omelet(int num_eggs) {
 *   if (num_eggs < 1) {
 *     TiledArray::detail::Backtrace bt("breakfast fail:");
 *     throw std::runtime_error(bt.str());
 *   }
 *   stove.on();
 *   // etc.
 * }
 * @endcode
 *
 */
class Backtrace {
 public:
  /**
   * @param prefix will be prepended to each line
   */
  Backtrace(const std::string& prefix = std::string(""));
  Backtrace(const Backtrace&);

  /**
   * @return true if did not get a backtrace
   */
  bool empty() const { return frames_.empty(); }

  /**
   * converts to a string
   * @param nframes_to_skip how many frames to skip
   * @return string representation of Backtrace, with each frame on a separate
   * line, from bottom to top
   */
  std::string str(const size_t nframes_to_skip = 0) const;

 private:
  /// frames_.begin() is the bottom of the stack
  std::vector<std::string> frames_;
  /// prepended to each line
  std::string prefix_;

  /// demangles a symbol
  static std::string __demangle(const std::string& symbol);
};
}  // namespace detail
}  // namespace TiledArray

#endif  // MPQC4_SRC_MPQC_UTIL_CORE_BACKTRACE_H_
