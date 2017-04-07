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
#include <utility>

namespace Eigen {

  template <typename, int, int, int, int, int> class Matrix;
  template <typename, int, int, int, int, int> class Array;
  template <typename, int, typename> class Map;

} // namespace Eigen

namespace TiledArray {

  // Forward declaration
  template <typename> class Tile;
  class DensePolicy;
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


  template <typename> struct eval_trait;

  namespace detail {

    template <typename T, typename Enabler = void>
    struct eval_trait_base {
      typedef T type;
      static constexpr bool is_consumable = false;
    }; // struct eval_trait

    template <typename T>
    struct eval_trait_base<T, typename std::enable_if<
        detail::is_type<typename T::eval_type>::value>::type>
    {
      typedef typename T::eval_type type;
      static constexpr bool is_consumable = false;
    }; // struct eval_trait

  } // namespace detail


  /// Determine the object type used in the evaluation of tensor expressions

  /// This trait class allows user to specify the object type used in an
  /// expression by providing a (partial) template specialization of this class
  /// for a user defined tile types. The default implementation uses
  /// `T::eval_type` the evaluation type, if no specialization has been
  /// provided. If no specialization is provided and the tile does not define
  /// an `eval_type`, the tile is not treated as a lazy tile. This class also
  /// provides the `is_consumable` flag that indicates if the evaluated tile
  /// object is consumable.
  /// \tparam T The lazy tile type
  template <typename T>
  struct eval_trait : public TiledArray::detail::eval_trait_base<T> { };

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
    struct is_numeric : public std::is_arithmetic<T> { };

    template <typename T>
    struct is_numeric<std::complex<T> > : public is_numeric<T> { };

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
    /// -# if T is numeric, numeric_type<T>::type evaluates to T
    /// -# if T is not numeric and T::value_type is a valid type, will evaluate to numeric_type<T::value_type>::type,
    ///    and so on recursively
    /// -# otherwise it's undefined
    template <typename T, typename Enabler = void> struct numeric_type;

    template <typename T>
    struct numeric_type<T,
        typename std::enable_if<is_numeric<T>::value>::type>
    {
      typedef T type;
    };

    template <typename T>
    struct numeric_type<T, typename std::enable_if<
          is_type<typename T::value_type>::value &&
          (! is_lazy_tile<T>::value) &&
          (! is_numeric<T>::value)>::type> :
        public numeric_type<typename T::value_type>
    { };


    template <typename T>
    struct numeric_type<T, typename std::enable_if<
          is_lazy_tile<T>::value
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
    using numeric_t = typename TiledArray::detail::numeric_type<T>::type;

    /// Type trait for extracting the scalar type of tensors and arrays.

    /// \tparam T The type to extract a numeric type from
    /// \tparam Enabler Type used to selectively implement partial
    /// specializations
    /// -# if T is a scalar type, i.e. \c is_scalar<T>::value is true (e.g. \c
    ///    int or \c float), \c scalar_type<T>::type evaluates to \c T
    /// -# if T is std::complex<U>, scalar_type<T>::type evaluates to U
    /// -# if T is not a scalar or complex type, will evaluate to \c
    ///    scalar_type<numeric_type<T>::type>::type, and so on recursively
    /// -# otherwise it's undefined
    template <typename T, typename Enabler = void>
    struct scalar_type;

    template <typename T>
    struct scalar_type<
        T, typename std::enable_if<is_scalar<T>::value>::type> {
      typedef T type;
    };

    template <typename T>
    struct scalar_type<std::complex<T>, void > : public scalar_type<T> { };

    template <typename T>
    struct scalar_type<T, typename std::enable_if<!is_numeric<T>::value>::type > :
    public scalar_type<typename numeric_type<T>::type> { };

    template <typename T>
    using scalar_t = typename TiledArray::detail::scalar_type<T>::type;


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

    /// prepends \c const to \c T if \c B is \c true
    template <bool B, typename T>
    using const_if_t = typename std::conditional<B, const T, T>::type;

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


    /// is_dense<T> is a true type if `T` is a dense array
    template <typename T>
    struct is_dense : public std::false_type { };

    template <typename Tile>
    struct is_dense<DistArray<Tile, DensePolicy> > : public std::true_type { };

    template <typename T>
    using trange_t = typename T::trange_type;

    template <typename T>
    using shape_t = typename T::shape_type;

    template <typename T>
    using pmap_t = typename T::pmap_interface;

    template <typename Array>
    struct policy_type;

    template <typename Tile, typename Policy>
    struct policy_type<DistArray<Tile, Policy> > {
      typedef Policy type;
    };

    template <typename T>
    using policy_t = typename policy_type<T>::type;

    /// If \c Base is a base of \c Derived, \c if_same_or_derived::value is \c true.
    /// \tparam Base supposed base class
    /// \tparam Derived supposed derived class
    template<typename Base, typename Derived>
    struct is_same_or_derived : std::conditional<
          std::is_base_of<Base,typename std::decay<Derived>::type>::value,
          std::true_type, std::false_type
        >::type {};

    //////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct is_pair : public std::false_type { };

    template <typename T1, typename T2>
    struct is_pair<std::pair<T1,T2> > : public std::true_type { };

    //////////////////////////////////////////////////////////////////////////////////////////////
    // see https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Member_Detector

#define GENERATE_HAS_MEMBER(member)                                               \
                                                                                  \
template < class T >                                                              \
class __has_member_##member                                                       \
{                                                                                 \
private:                                                                          \
    using Yes = char[2];                                                          \
    using  No = char[1];                                                          \
                                                                                  \
    struct Fallback { int member; };                                              \
    struct Derived : T, Fallback { };                                             \
                                                                                  \
    template < class U >                                                          \
    static No& test ( decltype(U::member)* );                                     \
    template < typename U >                                                       \
    static Yes& test ( U* );                                                      \
                                                                                  \
public:                                                                           \
    static constexpr bool value = sizeof(test<Derived>(nullptr)) == sizeof(Yes);  \
};                                                                                \
                                                                                  \
template < class T >                                                              \
struct has_member_##member                                                        \
: public std::integral_constant<bool, __has_member_##member<T>::value>            \
{                                                                                 \
};

#define GENERATE_HAS_MEMBER_TYPE(Type)                                            \
                                                                                  \
template < class T >                                                              \
class __has_member_type_##Type                                                    \
{                                                                                 \
private:                                                                          \
    using Yes = char[2];                                                          \
    using  No = char[1];                                                          \
                                                                                  \
    struct Fallback { struct Type { }; };                                         \
    struct Derived : T, Fallback { };                                             \
                                                                                  \
    template < class U >                                                          \
    static No& test ( typename U::Type* );                                        \
    template < typename U >                                                       \
    static Yes& test ( U* );                                                      \
                                                                                  \
public:                                                                           \
    static constexpr bool value = sizeof(test<Derived>(nullptr)) == sizeof(Yes);  \
};                                                                                \
                                                                                  \
template < class T >                                                              \
struct has_member_type_##Type                                                     \
: public std::integral_constant<bool, __has_member_type_##Type<T>::value>         \
{ };

#define GENERATE_HAS_MEMBER_FUNCTION(member)                                      \
template<typename T, typename Result, typename ... Args>                          \
class __has_member_function_##member                                              \
{                                                                                 \
  template <typename U, Result (U::*)(Args...)> struct Check;                     \
  template <typename U> static char func(Check<U, &U::member> *);                 \
  template <typename U> static int func(...);                                     \
public:                                                                           \
  static constexpr const bool value = sizeof(func<T>(0)) == sizeof(char);         \
};                                                                                \
template < class T, typename Result, typename ... Args>                           \
struct has_member_function_##member                                               \
: public std::integral_constant<bool, __has_member_function_##member<T, Result, Args...>::value>\
{                                                                                 \
};

#define GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(member)                            \
template<typename T, typename ... Args>                                           \
class __has_member_function_##member##_anyreturn                                  \
{                                                                                 \
  template <typename U, typename ... Args_> static auto func(void*) ->            \
    decltype(std::add_pointer_t<decltype(std::declval<U>().member(std::declval<Args_>()...))>{}, char{});\
  template <typename U, typename ... Args_> static int func(...);                 \
public:                                                                           \
  static constexpr const bool value = sizeof(func<T, Args...>(0)) == sizeof(char);\
};                                                                                \
template < class T, typename ... Args>                                            \
struct has_member_function_##member##_anyreturn                                   \
: public std::integral_constant<bool, __has_member_function_##member##_anyreturn<T, Args...>::value>\
{                                                                                 \
};

#define GENERATE_IS_FREE_FUNCTION_ANYRETURN(fn)                                   \
template<typename ... Args>                                                       \
class __is_free_function_##fn##_anyreturn                                         \
{                                                                                 \
  template <typename ... Args_> static auto func(void*) ->                        \
    decltype(std::add_pointer_t<decltype(fn(std::declval<Args_>()...))>{}, char{}); \
  template <typename ...> static int func(...);                                   \
public:                                                                           \
  static constexpr const bool value = sizeof(func<Args...>(0)) == sizeof(char);   \
};                                                                                \
template <typename ... Args>                                                      \
struct is_free_function_##fn##_anyreturn                                          \
: public std::integral_constant<bool, __is_free_function_##fn##_anyreturn<Args...>::value>\
{                                                                                 \
};

GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(size)
GENERATE_HAS_MEMBER_FUNCTION(size)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(empty)
GENERATE_HAS_MEMBER_FUNCTION(empty)

GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(begin)
GENERATE_HAS_MEMBER_FUNCTION(begin)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(end)
GENERATE_HAS_MEMBER_FUNCTION(end)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(cbegin)
GENERATE_HAS_MEMBER_FUNCTION(cbegin)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(cend)
GENERATE_HAS_MEMBER_FUNCTION(cend)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(rbegin)
GENERATE_HAS_MEMBER_FUNCTION(rbegin)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(rend)
GENERATE_HAS_MEMBER_FUNCTION(rend)

GENERATE_IS_FREE_FUNCTION_ANYRETURN(max)

  } // namespace detail

} // namespace TiledArray
#endif // TILEDARRAY_TYPE_TRAITS_H__INCLUDED
