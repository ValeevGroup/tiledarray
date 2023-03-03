//
// backtrace.cpp
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

#include <TiledArray/util/backtrace.h>

#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>

#include <madness/madness_config.h>
#ifdef MADNESS_HAS_LIBUNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#else
#if __has_include(<execinfo.h>)
#define HAVE_BACKTRACE
#include <execinfo.h>
#endif
#endif

#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
#define HAVE_CXA_DEMANGLE
#endif

namespace TiledArray {
namespace detail {
Backtrace::Backtrace(const std::string &prefix) : prefix_(prefix) {
#ifdef MADNESS_HAS_LIBUNWIND
  {
    unw_cursor_t cursor;
    unw_context_t uc;
    unw_word_t ip, sp, offp;
    int frame = 0;

    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);
    while (unw_step(&cursor) > 0) {
      unw_get_reg(&cursor, UNW_REG_IP, &ip);
      unw_get_reg(&cursor, UNW_REG_SP, &sp);
      char name[32768];
      unw_get_proc_name(&cursor, name, 32768, &offp);
      std::ostringstream oss;
      oss << prefix_ << "frame " << frame << ": "
          << "ip = 0x" << (long)ip << " sp = 0x" << (long)sp
          << " symbol = " << __demangle(name);
      frames_.push_back(oss.str());
      ++frame;
    }
  }
#elif defined(HAVE_BACKTRACE)  // !MADNESS_HAS_LIBUNWIND
  void *stack_addrs[1024];
  const int naddrs = backtrace(stack_addrs, 1024);
  char **frame_symbols = backtrace_symbols(stack_addrs, naddrs);
  // starting @ 1 to skip this function
  for (int i = 1; i < naddrs; ++i) {
    // extract (mangled) function name
    // parsing frame_symbols[i] is OS-specific
    // for unknown OS ... just return the whole string
    std::string mangled_function_name = frame_symbols[i];
#if defined(__APPLE__)
    {
      // "frame_id /path/to/exec address symbol"
      std::istringstream iss(std::string(frame_symbols[i]),
                             std::istringstream::in);
      std::string frame, file, address;
      iss >> frame >> file >> address >> mangled_function_name;
    }
#elif defined(__linux__)
    {
      // "/path/to/exec(symbol+0x...) [address]"
      // parse from the back to avoid dealing with parentheses in the path
      const auto last_right_bracket = mangled_function_name.rfind(']');
      const auto last_left_bracket =
          mangled_function_name.rfind('[', last_right_bracket);
      const auto last_right_parens =
          mangled_function_name.rfind(')', last_left_bracket);
      const auto offset = mangled_function_name.rfind("+0x", last_right_parens);
      const auto last_left_parens =
          mangled_function_name.rfind('(', last_right_parens);
      if (last_left_parens + 1 < mangled_function_name.size()) {
        mangled_function_name = mangled_function_name.substr(
            last_left_parens + 1, offset - last_left_parens - 1);
      }
    }
#endif

    std::ostringstream oss;
    oss << prefix_ << "frame " << i << ": return address = " << stack_addrs[i]
        << std::endl
        << "  symbol = " << __demangle(mangled_function_name);
    frames_.push_back(oss.str());
  }
  free(frame_symbols);
#else  // !MADNESS_HAS_LIBUNWIND && !HAVE_BACKTRACE
#if defined(SIMPLE_STACK)
  int bottom = 0x1234;
  void **topstack = (void **)0xffffffffL;
  void **botstack = (void **)0x70000000L;
  // signal handlers can put weird things in the return address slot,
  // so it is usually best to keep toptext large.
  void **toptext = (void **)0xffffffffL;
  void **bottext = (void **)0x00010000L;
#endif  // SIMPLE_STACK

#if (defined(linux) && defined(i386))
  topstack = (void **)0xc0000000;
  botstack = (void **)0xb0000000;
#endif
#if (defined(__OSF1__) && defined(i860))
  topstack = (void **)0x80000000;
  botstack = (void **)0x70000000;
#endif

#if defined(SIMPLE_STACK)
  // This will go through the stack assuming a simple linked list
  // of pointers to the previous frame followed by the return address.
  // It trys to be careful and avoid creating new exceptions, but there
  // are no guarantees.
  void **stack = (void **)&bottom;

  void **frame_pointer = (void **)stack[3];
  while (frame_pointer >= botstack && frame_pointer < topstack &&
         frame_pointer[1] >= bottext && frame_pointer[1] < toptext) {
    std::ostringstream oss;
    oss << prefix_ << "frame: " << (void *)frame_pointer;
    oss << "  retaddr: " << frame_pointer[1];
    frames_.push_back(oss.str());

    frame_pointer = (void **)*frame_pointer;
  }
#endif  // SIMPLE_STACK
#endif  // HAVE_BACKTRACE
}

Backtrace::Backtrace(const Backtrace &other)
    : frames_(other.frames_), prefix_(other.prefix_) {}

std::string Backtrace::str(size_t nframes_to_skip) const {
  std::ostringstream oss;
  std::copy(frames_.begin() + nframes_to_skip, frames_.end(),
            std::ostream_iterator<std::string>(oss, "\n"));
  return oss.str();
}

std::string Backtrace::__demangle(const std::string &symbol) {
  std::string dsymbol;
#ifdef HAVE_CXA_DEMANGLE
  {
    int status;
    char *dsymbol_char = abi::__cxa_demangle(symbol.c_str(), 0, 0, &status);
    if (status == 0) {  // success
      dsymbol = dsymbol_char;
      free(dsymbol_char);
    } else  // fail
      dsymbol = symbol;
  }
#else
  dsymbol = symbol;
#endif
  return dsymbol;
}

}  // namespace detail
}  // namespace TiledArray

extern "C" void tiledarray_dump_backtrace_to_std_cout() {
  TiledArray::detail::Backtrace bt("tiledarray_dump_backtrace: ");
  std::cout << bt.str() << std::endl;
}
