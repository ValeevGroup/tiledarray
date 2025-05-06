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
 *  print.h
 *  Mar 14, 2025
 *
 */

#ifndef TILEDARRAY_SRC_TILEDARRAY_TENSOR_PRINT_H__INCLUDED
#define TILEDARRAY_SRC_TILEDARRAY_TENSOR_PRINT_H__INCLUDED

#include <TiledArray/range1.h>

#include <complex>
#include <iomanip>
#include <iosfwd>

namespace TiledArray {

namespace detail {

// Class to print n-dimensional arrays in NumPy style but with curly braces
class NDArrayPrinter {
 public:
  NDArrayPrinter(int width = 10, int precision = 6)
      : width(width),
        precision(precision),
        truncate_(0.5 * std::pow(10., -precision)) {}

 private:
  int width = 10;
  int precision = 10;

  /// truncates (=sets to zero) small floating-point numbers
  class FloatTruncate {
   public:
    /// truncates numbers smaller than @p threshold
    FloatTruncate(double threshold) noexcept : threshold_{threshold} {}

    [[nodiscard]] auto operator()(std::floating_point auto val) const noexcept {
      return std::abs(val) < threshold_ ? decltype(val){0} : val;
    }

    template <typename T>
      requires detail::is_complex_v<T> &&
               std::floating_point<typename T::value_type>
    [[nodiscard]] auto operator()(T const& val) const noexcept {
      using std::imag;
      using std::real;
      return T{(*this)(real(val)), (*this)(imag(val))};
    }

    template <typename T>
      requires(!(std::floating_point<T> || detail::is_complex_v<T>))
    [[nodiscard]] auto operator()(T const& val) const noexcept {
      return val;
    }

   private:
    double threshold_;
  };

  FloatTruncate truncate_;

  // Helper function to recursively print the array
  template <typename T, typename Index = Range1::index1_type,
            typename Char = char, typename CharTraits = std::char_traits<Char>>
  void printArray(const T* data, const std::size_t order, const Index* extents,
                  const Index* strides,
                  std::basic_ostream<Char, CharTraits>& os, size_t level = 0,
                  size_t offset = 0, size_t extra_indentation = 0);

 public:
  // Print a row-major array to a stream
  template <typename T, typename Char = char,
            typename Index = Range1::index1_type,
            typename CharTraits = std::char_traits<Char>>
  void print(const T* data, const std::size_t order, const Index* extents,
             const Index* strides, std::basic_ostream<Char, CharTraits>& os,
             std::size_t extra_indentation = 0);

  // Helper function to create a string representation
  template <typename T, typename Char = char,
            typename Index = Range1::index1_type,
            typename CharTraits = std::char_traits<Char>>
  std::basic_string<Char, CharTraits> toString(const T* data,
                                               const std::size_t order,
                                               const Index* extents,
                                               const Index* strides);
};

// Explicit template instantiations
#define TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(T, C)                   \
  extern template void NDArrayPrinter::print<T, C>(                           \
      const T* data, const std::size_t order,                                 \
      const Range1::index1_type* extents, const Range1::index1_type* strides, \
      std::basic_ostream<C>&, std::size_t);                                   \
  extern template std::basic_string<C> NDArrayPrinter::toString<T, C>(        \
      const T* data, const std::size_t order,                                 \
      const Range1::index1_type* extents, const Range1::index1_type* strides);

TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(double, char);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(double, wchar_t);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(std::complex<double>, char);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(std::complex<double>, wchar_t);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(float, char);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(float, wchar_t);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(std::complex<float>, char);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(std::complex<float>, wchar_t);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(int, char);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(int, wchar_t);

#undef TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION

}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_SRC_TILEDARRAY_TENSOR_PRINT_H__INCLUDED
