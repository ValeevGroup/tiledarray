/*
*  This file is a part of TiledArray.
*  Copyright (C) 2025  Virginia Tech
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
*  Eduard Valeyev
*  Department of Chemistry, Virginia Tech
*
*  print.ipp
*  Mar 14, 2025
*
*/

#include <TiledArray/tensor/print.h>

#include <iosfwd>
#include <iomanip>
#include <complex>

namespace TiledArray {

namespace detail {

// Class to print n-dimensional arrays in NumPy style but with curly braces
template <typename T, typename Index, typename Char, typename CharTraits>
void NDArrayPrinter::printArray(const T* data, const std::size_t order,
                                const Index* extents, const Index* strides,
                                std::basic_ostream<Char, CharTraits>& os,
                                size_t level, size_t offset,
                                size_t extra_indentation) {
  if (level >= order) {
    return;
  }

  if (level == 0 && extra_indentation > 0)
    os << std::basic_string<Char, CharTraits>(extra_indentation, ' ');
  os << "{";

  for (size_t i = 0; i < extents[level]; ++i) {
    if (level == order - 1) {
      // At the deepest level, print the actual values
      os << std::fixed << std::setprecision(precision) << std::setw(width) << std::setfill(Char(' '))
         << data[offset + i * strides[level]];
      if (i < extents[level] - 1) {
        os << ", ";
      }
    } else {
      // For higher levels, recurse deeper
      printArray(data, order, extents, strides, os, level + 1, offset + i * strides[level],
                 extra_indentation);
      if (i < extents[level] - 1) {
        os << ",\n" << std::basic_string<Char, CharTraits>(level + 1 + extra_indentation, ' ');
      }
    }
  }
  os << "}";
}

// Print a row-major array to a stream
template <typename T, typename Char, typename Index, typename CharTraits>
void NDArrayPrinter::print(const T* data, const std::size_t order,
                           const Index* extents, const Index* strides,
                           std::basic_ostream<Char, CharTraits>& os,
                           std::size_t extra_indentation) {
  // Note: Can't validate data size with raw pointers, caller must ensure data has sufficient size

  printArray(data, order, extents, strides, os, 0, 0, extra_indentation);
}

// Helper function to create a string representation
template <typename T, typename Char, typename Index, typename CharTraits>
std::basic_string<Char, CharTraits> NDArrayPrinter::toString(
    const T* data, const std::size_t order, const Index* extents,
    const Index* strides) {
  std::basic_stringstream<Char, CharTraits> oss;
  print(data, order, extents, strides, oss, /* extra_indentation = */ 0);
  return oss.str();
}

}  // namespace detail

}  // namespace TiledArray
