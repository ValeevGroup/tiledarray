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

#ifndef TILEDARRAY_TYPE_TRAITS_H__INCLUDED
#define TILEDARRAY_TYPE_TRAITS_H__INCLUDED

#include <TiledArray/config.h>
#include <iterator>
#include <madness/world/type_traits.h>
#include <complex>

namespace Eigen {

  template <typename, int, int, int, int, int> class Matrix;
  template <typename, int, int, int, int, int> class Array;
  template <typename, int, typename> class Map;

} // namespace Eigen

namespace TiledArray {

  template <bool condition, typename T = void>
  using enable_if_t = typename std::enable_if<condition, T>::type;

  namespace detail {

    template <typename T>
    struct is_numeric : public std::false_type { };

    template <>
    struct is_numeric<short int> : public std::true_type { };

    template <>
    struct is_numeric<int> : public std::true_type { };

    template <>
    struct is_numeric<unsigned int> : public std::true_type { };

    template <>
    struct is_numeric<long int> : public std::true_type { };

    template <>
    struct is_numeric<unsigned long int> : public std::true_type { };

    template <>
    struct is_numeric<float> : public std::true_type { };

    template <>
    struct is_numeric<double> : public std::true_type { };

#ifdef TILEDARRAY_HAS_LONG_DOUBLE

    template <>
    struct is_numeric<long double> : public std::true_type { };

#endif //TILEDARRAY_HAS_LONG_DOUBLE

#ifdef TILEDARRAY_HAS_LONG_LONG

    template <>
    struct is_numeric<long long int> : public std::true_type { };

    template <>
    struct is_numeric<unsigned long long int> : public std::true_type { };

#endif // TILEDARRAY_HAS_LONG_LONG

    template <typename T>
    struct is_numeric<std::complex<T> > : public is_numeric<T> { };

    template <typename T>
    struct is_complex : public std::false_type { };

    template <typename T>
    struct is_complex<std::complex<T> > : public std::true_type { };

    // helps to implement other metafunctions
    template<typename> struct is_type : public std::true_type { };

    /// Type trait for extracting the scalar numeric type of tensors and arrays.

    /// \tparam T The type to extract a numeric type from
    /// \tparam Enabler Type used to selectively implement partial specializations
    /// -# if T is numeric, scalar_type<T>::type evaluates to T
    /// -# if T is not numeric and T::value_type is a valid type, will evaluate to scalar_type<T::value_type>::type,
    ///   and so on recursively
    /// -# otherwise it's undefined
    template <typename T, typename Enabler = void> struct scalar_type;

    template <typename T>
    struct scalar_type<T, typename std::enable_if<is_numeric<T>::value>::type> {
      typedef T type;
    };

    template <typename T>
    struct scalar_type<T, typename std::enable_if<is_type<typename T::value_type>::value>::type> :
        public scalar_type<typename T::value_type>
    { };

    template <typename T, int Rows, int Cols, int Opts, int MaxRows, int MaxCols>
    struct scalar_type<Eigen::Matrix<T, Rows, Cols, Opts, MaxRows, MaxCols>, void> :
        public scalar_type<typename Eigen::Matrix<T, Rows, Cols, Opts, MaxRows, MaxCols>::Scalar>
    { };

    template <typename T, int Rows, int Cols, int Opts, int MaxRows, int MaxCols>
    struct scalar_type<Eigen::Array<T, Rows, Cols, Opts, MaxRows, MaxCols>, void> :
        public scalar_type<typename Eigen::Matrix<T, Rows, Cols, Opts, MaxRows, MaxCols>::Scalar>
    { };

    template <typename PlainObjectType, int MapOptions, typename StrideType>
    struct scalar_type<Eigen::Map<PlainObjectType, MapOptions, StrideType>, void> :
        public scalar_type<PlainObjectType>
    { };

    template <typename...> struct is_integral_list_helper;

    template <typename T, typename...Ts>
    struct is_integral_list_helper<T, Ts...> {
      static constexpr bool value = std::is_integral<T>::value && is_integral_list_helper<Ts...>::value;
    };

    template <> struct is_integral_list_helper<> { static constexpr bool value = true; };

    ///
    template <typename...Ts>
    struct is_integral_list : std::conditional<(sizeof...(Ts) > 0ul),
        is_integral_list_helper<Ts...>,
        std::false_type>::type
    { };


    /// Describes traits of nodes in TiledArray expressions
    template <typename T, typename Enabler = void>
    struct eval_trait {
      typedef T type;
    };

    template <typename T>
    struct eval_trait<T, typename std::enable_if<is_type<typename T::eval_type>::value>::type>  {
        typedef typename T::eval_type type;
    };

    /// Remove const, volatile, and reference qualifiers.
    template <typename T>
    struct remove_cvr {
      typedef typename std::remove_cv<typename std::remove_reference<T>::type>::type type;
    };

    template <typename T, typename Enabler = void>
    struct param {
      typedef typename std::add_lvalue_reference<typename std::add_const<T>::type>::type type;
    };

    template <typename T>
    struct param<T, typename std::enable_if<is_numeric<T>::value>::type> {
      typedef typename std::add_const<T>::type type;
    };

    template <typename T>
    struct param<T, typename std::enable_if<std::is_reference<T>::value>::type> {
      typedef T type;
    };


    template <typename T>
    struct param<T, typename std::enable_if<std::is_pointer<T>::value>::type> {
      typedef typename std::add_const<T>::type type;
    };

    template <typename U>
    using param_type = typename param<U>::type;


    struct non_iterator_tag { };

    template <typename T, typename Enabler = void>
    struct is_iterator : public std::false_type {
      typedef non_iterator_tag iterator_category;
    };

    template <typename T>
    struct is_iterator<T, typename std::enable_if<is_type<typename T::iterator_category>::value>::type > : public std::true_type {
      typedef typename std::iterator_traits<T>::iterator_category iterator_category;
    };

    template <typename T>
    struct is_iterator<T*, void> : std::true_type {
      typedef std::random_access_iterator_tag iterator_category;
    };

    template <typename T>
    struct is_iterator<const T*, void> : std::true_type {
      typedef std::random_access_iterator_tag iterator_category;
    };

    template <typename T>
    struct is_iterator<T* const, void> : std::true_type {
      typedef std::random_access_iterator_tag iterator_category;
    };

    template <typename T>
    struct is_iterator<const T* const, void> : std::true_type {
      typedef std::random_access_iterator_tag iterator_category;
    };

    template <typename T>
    struct is_input_iterator :
        public std::is_base_of<std::input_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_output_iterator :
        public std::is_base_of<std::output_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_forward_iterator :
        public std::is_base_of<std::forward_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_bidirectional_iterator :
        public std::is_base_of<std::bidirectional_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_random_iterator :
        public std::is_base_of<std::random_access_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

  } // namespace detail
} // namespace TiledArray
#endif // TILEDARRAY_TYPE_TRAITS_H__INCLUDED
