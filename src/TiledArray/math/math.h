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

#ifndef TILEDARRAY_MATH_MATH_H__INCLUDED
#define TILEDARRAY_MATH_MATH_H__INCLUDED

#include <TiledArray/error.h>
#include <cmath>
#include <stdint.h>

namespace TiledArray {
  namespace math {


    template <typename T>
    inline T abs(const T t) { return std::abs(t); }

    template <typename T>
    inline T max(const T t1, const T t2) { return std::max(t1, t2); }

    template <typename T>
    inline T min(const T t1, const T t2) { return std::min(t1, t2); }

    template <typename T>
    inline T maxabs(const T t1, const T t2) { return std::max(std::abs(t1), std::abs(t2)); }

    template <typename T>
    inline T minabs(const T t1, const T t2) { return std::min(std::abs(t1), std::abs(t2)); }

    /// Compute the integer log_2 of x

    /// \param x The value to be evaluated
    /// \return The floor of log_2(x)
    inline uint32_t log2(uint32_t x) {
      TA_ASSERT(x != 0u);
#if defined(__i386) || defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
      asm ( "\tbsr %0, %0\n" : "=r"(x) : "0" (x) );
      return x;
#elif defined(__powerpc__) || defined(__ppc64__) || defined(__ppc__)
//      uint32_t lz;
//      asm ("cntlzw %0,%1" : "=r" (lz) : "r" (x));
//      return 32u - lz;
      // Not sure this will work. If not, use the code above.
      asm ("cntlzw %0,%1" : "=r" (x) : "r" (x));
      return 32u - x;
#else
      uint_type result = 0ul;
      while (x >>= 1) ++result;
      return result;
#endif
    }

  }  // namespace math
}  // namespace TiledArray

#endif // TILEDARRAY_MATH_MATH_H__INCLUDED
