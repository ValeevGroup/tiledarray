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

// Check for default error checking method, which is determined by
// TA_DEFAULT_ERROR, which is defined in TiledArray/config.h.
#ifdef TA_DEFAULT_ERROR
#if !defined(TA_EXCEPTION_ERROR) && !defined(TA_ASSERT_ERROR) && \
    !defined(TA_NO_ERROR) && !defined(TA_ABORT_ERROR)
#if TA_DEFAULT_ERROR == 0
#define TA_NO_ERROR
#elif TA_DEFAULT_ERROR == 1
#define TA_EXCEPTION_ERROR
#elif TA_DEFAULT_ERROR == 2
#define TA_ASSERT_ERROR
#elif TA_DEFAULT_ERROR == 3
#define TA_ABORT_ERROR
#endif  // TA_DEFAULT_ERROR == ?
#endif  // !defined(TA_EXCEPTION_ERROR) && !defined(TA_ASSERT_ERROR) &&
        // !defined(TA_NO_ERROR) && !defined(TA_ABORT_ERROR)
#endif  // TA_DEFAULT_ERROR

#include <exception>
namespace TiledArray {

class Exception : public std::exception {
 public:
  Exception(const char* m) : message_(m) {}

  virtual const char* what() const noexcept { return message_; }

 private:
  const char* message_;
};  // class Exception

/// Place a break point on this function to stop before TiledArray exceptions
/// are thrown.
inline void exception_break() {}
}  // namespace TiledArray

#define TA_STRINGIZE(s) #s

#define TA_EXCEPTION_MESSAGE(file, line, mess) \
  "TiledArray: exception at " file "(" TA_STRINGIZE(line) "): " mess

#define TA_EXCEPTION(m)                                                       \
  do {                                                                        \
    TiledArray::exception_break();                                            \
    throw TiledArray::Exception(TA_EXCEPTION_MESSAGE(__FILE__, __LINE__, m)); \
  } while (0)

#ifdef TA_EXCEPTION_ERROR
// This section defines the behavior for TiledArray assertion error checking
// which will throw exceptions.
#ifdef TA_ASSERT_ERROR
#undef TA_ASSERT_ERROR
#endif
#ifdef TA_ABORT_ERROR
#undef TA_ABORT_ERROR
#endif

#define TA_ASSERT(a)                             \
  do {                                           \
    if (!(a)) TA_EXCEPTION("assertion failure"); \
  } while (0)
#define TA_TEST(a) TA_ASSERT(a)

#elif defined(TA_ASSERT_ERROR)
// This sections defines behavior for TiledArray assertion error checking which
// uses assertions.
#include <cassert>
#define TA_ASSERT(a) assert(a)
#define TA_TEST(a) TA_ASSERT(a)
#elif defined(TA_ABORT_ERROR)
// This sections defines behavior for TiledArray assertion error checking which
// calls std::abort
#include <cstdlib>
#define TA_ASSERT(a)        \
  do {                      \
    if (!(a)) std::abort(); \
  } while (0)
#define TA_TEST(a) TA_ASSERT(a)
#else
// This section defines behavior for TiledArray assertion error checking which
// does no error checking.
// WARNING: TiledArray will perform no error checking.
// this avoids unused variable warnings, see
// http://cnicholson.net/2009/02/stupid-c-tricks-adventures-in-assert/
#define TA_ASSERT(a) \
  do {               \
    (void)sizeof(a); \
  } while (0)
#define TA_TEST(a) (a)

#endif  // TA_EXCEPTION_ERROR

#define TA_CHECK(a)                          \
  do {                                       \
    if (!(a)) TA_EXCEPTION("check failure"); \
  } while (0)

#ifdef TILEDARRAY_NO_USER_ERROR_MESSAGES
#define TA_USER_ERROR_MESSAGE(m)
#else
#include <iostream>
#define TA_USER_ERROR_MESSAGE(m) \
  std::cerr << "!! ERROR TiledArray: " << m << "\n";
#endif  // TILEDARRAY_NO_USER_ERROR_MESSAGES

#endif  // TILEDARRAY_ERROR_H__INCLUDED
