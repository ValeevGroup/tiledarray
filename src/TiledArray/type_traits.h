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

  // Forward declaration
  template <typename> class Tile;
  struct ZeroTensor;
  template <typename, typename> class DistArray;
  namespace detail {
    template <typename, typename> class LazyArrayTile;

    // helps to implement other metafunctions
    template<typename> struct is_type : public std::true_type { };

  }  // namespace detail

  /**
   * \addtogroup TileInterface
   * @{
   */

  /// Determine the object type used in the evaluation of tensor expressions

  /// This trait class allows user to specify the object type used in an
  /// expression by providing a (partial) template specialization of this class
  /// for a user defined tile types. This allows users to use lazy tile
  /// construction inside tensor expressions. If no evaluation type is
  /// specified, the lazy tile evaluation is disabled.
  /// \tparam T The lazy tile type
  /// \tparam Enabler Internal use only
  template <typename T, typename Enabler = void>
  struct eval_trait {
    typedef T type;
  }; // struct eval_trait

  /// Determine the object type used in the evaluation of tensor expressions

  /// This trait class allows user to specify the object type used in an
  /// expression by providing a member type <tt>T::eval_type</tt>. This allows
  /// users to use lazy tile  construction inside tensor expressions. If no
  /// evaluation type is specified, the lazy tile evaluation is disabled.
  /// \tparam T The lazy tile type
  template <typename T>
  struct eval_trait<T, typename std::enable_if<detail::is_type<typename T::eval_type>::value>::type>  {
    typedef typename T::eval_type type;
  }; // struct eval_trait


  /// Detect lazy evaluation tiles

  /// \c is_lazy_type evaluates to \c std::true_type when T is a tile that
  /// uses the lazy evaluation mechanism (i.e. when <tt>T != T::eval_type</tt>),
  /// otherwise it evaluates to \c std::false_type .
  /// \tparam T The tile type to test
  template <typename T>
  struct is_lazy_tile :
      public std::integral_constant<bool, ! std::is_same<T, typename eval_trait<T>::type>::value>
  { }; // struct is_lazy_tile

  template <typename Tile, typename Policy>
  struct is_lazy_tile<DistArray<Tile, Policy> > : public std::false_type { };


  /// Consumable tile type trait

  /// This trait is used to determine if a tile type is consumable in tensor
  /// arithmetic operations. That is, temporary tiles may appear on the left-
  /// hand side of add-to (+=), subtract-to (-=), and multiply-to (*=)
  /// operations. By default, all tile types are assumed to be
  /// consumable except lazy tiles. Users should provide a (partial)
  /// specialization of this `struct` to disable consumable tile operations.
  /// \tparam T The tile type
  template <typename T>
  struct is_consumable_tile :
      public std::integral_constant<bool, ! is_lazy_tile<T>::value>
  { };

  template <>
  struct is_consumable_tile<ZeroTensor> : public std::false_type { };



  /** @}*/


  namespace detail {

    template <typename T>
    struct is_complex : public std::false_type { };

    template <typename T>
    struct is_complex<std::complex<T> > : public std::true_type { };

    template <typename T>
    struct is_char : public std::false_type { };

    template <>
    struct is_char<char> : public std::true_type { };

    template <>
    struct is_char<char16_t> : public std::true_type { };

    template <>
    struct is_char<char32_t> : public std::true_type { };

    template <>
    struct is_char<wchar_t> : public std::true_type { };

    template <>
    struct is_char<signed char> : public std::true_type { };

    template <typename T>
    struct is_bool : public std::false_type { };

    template <>
    struct is_bool<bool> : public std::true_type { };

    template <typename T>
    struct is_numeric : public std::is_arithmetic<T> { };

    template <typename T>
    struct is_numeric<std::complex<T> > : public is_numeric<T> { };

    template <>
    struct is_numeric<char> : public std::false_type { };

    template <>
    struct is_numeric<char16_t> : public std::false_type { };

    template <>
    struct is_numeric<char32_t> : public std::false_type { };

    template <>
    struct is_numeric<wchar_t> : public std::false_type { };

    template <>
    struct is_numeric<signed char> : public std::false_type { };

    template <>
    struct is_numeric<bool> : public std::false_type { };

    template <typename T>
    struct is_scalar : public is_numeric<T> { };

    template <typename T>
    struct is_scalar<std::complex<T> > : public std::false_type { };





    /// Detect tiles used by \c ArrayEvalImpl

    /// \c is_lazy_type evaluates to \c std::true_type when T is a tile from
    /// \c ArrayEvalImpl (i.e. when <tt>T != LazyArrayTile</tt>),
    /// otherwise it evaluates to \c std::false_type .
    /// \tparam T The tile type to test
    template <typename T>
    struct is_array_tile : public std::false_type { };

    template <typename T, typename Op>
    struct is_array_tile<TiledArray::detail::LazyArrayTile<T, Op> > :
        public std::true_type
    { }; // struct is_array_tile

    /// Detect a lazy evaluation tile that are not a \c LazyArrayTile

    /// \c is_non_array_lazy_tile evaluates to \c std::true_type when T is a
    /// tile that uses the lazy evaluation mechanism (i.e. when
    /// <tt>T != T::eval_type</tt>), and not a \c LazyArrayTile , otherwise it
    /// evaluates to \c std::false_type .
    /// \tparam T The tile type to test
    template <typename T>
    struct is_non_array_lazy_tile :
        public std::integral_constant<bool, is_lazy_tile<T>::value && (! is_array_tile<T>::value)>
    { }; // struct is_non_array_lazy_tile


    /// Type trait for extracting the numeric type of tensors and arrays.

    /// \tparam T The type to extract a numeric type from
    /// \tparam Enabler Type used to selectively implement partial specializations
    /// -# if T is numeric, scalar_type<T>::type evaluates to T
    /// -# if T is not numeric and T::value_type is a valid type, will evaluate to scalar_type<T::value_type>::type,
    ///   and so on recursively
    /// -# otherwise it's undefined
    template <typename T, typename Enabler = void> struct numeric_type;

    template <typename T>
    struct numeric_type<T,
        typename std::enable_if<is_numeric<T>::value>::type>
    {
      typedef T type;
    };

    template <typename T>
    struct numeric_type<T, typename std::enable_if<is_type<typename T::value_type>::value
          && ! is_numeric<T>::value>::type> :
        public numeric_type<typename T::value_type>
    { };


    template <typename T>
    struct numeric_type<T, typename std::enable_if<is_lazy_tile<T>::value
          && ! is_numeric<T>::value>::type> :
        public numeric_type<typename eval_trait<T>::type>
    { };

    template <typename T, int Rows, int Cols, int Opts, int MaxRows, int MaxCols>
    struct numeric_type<Eigen::Matrix<T, Rows, Cols, Opts, MaxRows, MaxCols>, void> :
        public numeric_type<typename Eigen::Matrix<T, Rows, Cols, Opts, MaxRows, MaxCols>::Scalar>
    { };

    template <typename T, int Rows, int Cols, int Opts, int MaxRows, int MaxCols>
    struct numeric_type<Eigen::Array<T, Rows, Cols, Opts, MaxRows, MaxCols>, void> :
        public numeric_type<typename Eigen::Matrix<T, Rows, Cols, Opts, MaxRows, MaxCols>::Scalar>
    { };

    template <typename T>
    struct numeric_type<Tile<T>, void> :
        public numeric_type<typename Tile<T>::tensor_type>
    { };

    template <typename PlainObjectType, int MapOptions, typename StrideType>
    struct numeric_type<Eigen::Map<PlainObjectType, MapOptions, StrideType>, void> :
        public numeric_type<PlainObjectType>
    { };

    template <typename T>
    struct scalar_type : public numeric_type<T> { };

    template <typename T>
    struct scalar_type<std::complex<T> > : public scalar_type<T> { };


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

    // Type traits used to determine the result of arithmetic operations between
    // two types.

    template <typename Scalar1, typename Scalar2>
    using add_t = decltype(std::declval<Scalar1>() + std::declval<Scalar2>());


    template <typename Scalar1, typename Scalar2>
    using subt_t = decltype(std::declval<Scalar1>() - std::declval<Scalar2>());

    template <typename Scalar1, typename Scalar2>
    using mult_t = decltype(std::declval<Scalar1>() * std::declval<Scalar2>());


  } // namespace detail

} // namespace TiledArray
#endif // TILEDARRAY_TYPE_TRAITS_H__INCLUDED
