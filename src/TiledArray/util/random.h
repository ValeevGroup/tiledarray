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

#ifndef TILEDARRAY_RANDOM_H__INCLUDED
#define TILEDARRAY_RANDOM_H__INCLUDED

#include <complex>      // for std::complex
#include <cstdlib>      // for std::rand
#include <type_traits>  // for true_type, false_type, and enable_if

namespace TiledArray::detail {

//------------------------------------------------------------------------------
//                            CanMakeRandom
//------------------------------------------------------------------------------
/// Determines whether or not we can generate a random value of type ValueType
///
/// The CanMakeRandom<ValueType> class contains a boolean member value `value`,
/// which will be set to `true` if we can generate a random value of type
/// ValueType and `false` otherwise. By default `value` will be `false` for all
/// types. Specific types can be enabled by specializing CanMakeRandom.
///
/// \tparam ValueType The type of random value we are attempting to generate.
template <typename ValueType>
struct CanMakeRandom : std::false_type {};

/// Enables generating random int values
template <>
struct CanMakeRandom<int> : std::true_type {};

/// Enables generating random float values
template <>
struct CanMakeRandom<float> : std::true_type {};

/// Enables generating random double values
template <>
struct CanMakeRandom<double> : std::true_type {};

/// Enables generating random std::complex<float> values
template <>
struct CanMakeRandom<std::complex<float>> : std::true_type {};

/// Enables generating random std::complex<double> values
template <>
struct CanMakeRandom<std::complex<double>> : std::true_type {};

/// Variable for whether or not we can make a random value of type ValueType
///
/// This global variable is a convenience variable for accessing the `value`
/// member value of a particular instantiation of the CanMakeRandom class. For
/// example `can_make_random_v<T>` is shorthand for `CanMakeRandom<T>::value`.
///
/// \tparam ValueType the type of random value we are attempting to make.
template <typename ValueType>
static constexpr auto can_make_random_v = CanMakeRandom<ValueType>::value;

/// Enables a function only when we can generate a random value of type `T`
///
/// \tparam T The type of random value we are attempting to generate.
template <typename T>
using enable_if_can_make_random_t = std::enable_if_t<can_make_random_v<T>>;

//------------------------------------------------------------------------------
//                       MakeRandom
//------------------------------------------------------------------------------

/// Struct wrapping the process of generating a random value of type `ValueType`
///
/// MakeRandom contains a single static member function `generate_value`, which
/// generates a random value using `std::rand()`. The default implementation is
/// only provided for fundamental types:
/// - for a floating-point type this returns a random value in [-1,1].
/// - for a signed integral type this returns a random value in [-4,4].
/// - for an unsigned integral type this returns a random value in [0,8].
/// Users can specialize the MakeRandom class to control how random
/// values of other types are formed.
///
/// \tparam ValueType The type of random value to generate
template <typename ValueType>
struct MakeRandom {
  /// Generates a random value of type ValueType
  static ValueType generate_value() {
    static_assert(std::is_floating_point_v<ValueType> ||
                  std::is_integral_v<ValueType>);
    if constexpr (std::is_floating_point_v<ValueType>)
      return (2 * static_cast<ValueType>(std::rand()) / RAND_MAX) - 1;
    else if constexpr (std::is_integral_v<ValueType>) {
      static_assert(RAND_MAX == 2147483647);
      static_assert(RAND_MAX % 2 == 1);
      constexpr std::int64_t RAND_MAX_DIVBY_9 =
          (static_cast<std::int64_t>(RAND_MAX) + 8) / 9;
      const ValueType v = static_cast<ValueType>(
          static_cast<std::int64_t>(std::rand()) / RAND_MAX_DIVBY_9);
      if constexpr (std::is_signed_v<ValueType>) {
        return v - 4;
      } else {
        return v;
      }
    }
  }
};

/// Specializes MakeRandom to complex types.
///
/// To generate a random complex value we need to generate two random values:
/// one for the real component and one for the imaginary component. This
/// specialization does that by relying MakeRandom specializations for the types
/// of the components.
///
/// \tparam ScalarType The type used to hold the real and imaginary components
///                    of the complex value.
template <typename ScalarType>
struct MakeRandom<std::complex<ScalarType>> {
  /// Generates a random complex number.
  static auto generate_value() {
    static_assert(
        std::is_floating_point_v<ScalarType>);  // std::complex is only defined
                                                // for fundamental
                                                // floating-point types
    const ScalarType real = MakeRandom<ScalarType>::generate_value();
    const ScalarType imag = MakeRandom<ScalarType>::generate_value();
    return std::complex<ScalarType>(real, imag);
  }
};

}  // namespace TiledArray::detail

#endif
