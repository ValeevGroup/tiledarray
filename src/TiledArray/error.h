/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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

#ifndef TILEDARRAY_ERROR_H__INCLUDED
#define TILEDARRAY_ERROR_H__INCLUDED

#include <TiledArray/config.h>

#ifndef TA_ASSERT_POLICY
#define TA_ASSERT_POLICY TA_ASSERT_THROW
#endif

#define TA_STRINGIZE_IMPL(s) #s
#define TA_STRINGIZE(s) TA_STRINGIZE_IMPL(s)

#define TA_ASSERT_MESSAGE(EXPR, ...)            \
  __FILE__ ":" TA_STRINGIZE(__LINE__) ": "      \
  "TA_ASSERT failed: " TA_STRINGIZE(EXPR)

#if TA_ASSERT_POLICY == TA_ASSERT_IGNORE
#define TA_ASSERT(...) do { } while(0)
#else
#define TA_ASSERT(EXPR, ...)                                            \
  do {                                                                  \
    if (!(EXPR))                                                        \
      TiledArray::assert_failed(TA_ASSERT_MESSAGE(EXPR, __VA_ARGS__));  \
  } while(0)

#endif

#include <stdexcept>
#include <string>

namespace TiledArray {

void ta_abort();

void ta_abort(const std::string &m);

class Exception : public std::runtime_error {
  using std::runtime_error::runtime_error;
};  // class Exception

/// Place a break point on this function to stop before TiledArray exceptions
/// are thrown.
inline void exception_break() {}

inline void assert_failed(const std::string &m) {
#if TA_ASSERT_POLICY == TA_ASSERT_THROW
  TiledArray::exception_break();
  throw TiledArray::Exception(m);
#elif TA_ASSERT_POLICY == TA_ASSERT_ABORT
  TiledArray::ta_abort(m);
#elif TA_ASSERT_POLICY != TA_ASSERT_IGNORE
#error Invalid TA_ASSERT_POLICY parameter
#endif
}

} // namespace TiledArray

#define TA_EXCEPTION_MESSAGE(file, line, mess) \
  "TiledArray: exception at " file "(" TA_STRINGIZE(line) "): " mess

/// throws TiledArray::Exception with message \p m annotated with the file name
/// and line number
/// \param m a C-style string constant
#define TA_EXCEPTION(m)                                                       \
  do {                                                                        \
    TiledArray::exception_break();                                            \
    throw TiledArray::Exception(TA_EXCEPTION_MESSAGE(__FILE__, __LINE__, m)); \
  } while (0)

#ifdef TILEDARRAY_NO_USER_ERROR_MESSAGES
#define TA_USER_ERROR_MESSAGE(m)
#else
#include <iostream>
#define TA_USER_ERROR_MESSAGE(m) \
  std::cerr << "!! ERROR TiledArray: " << m << "\n";
#endif  // TILEDARRAY_NO_USER_ERROR_MESSAGES

#endif  // TILEDARRAY_ERROR_H__INCLUDED
