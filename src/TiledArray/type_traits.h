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

/*
 * N.B. this must be pure c++, usable without any context other than
 *      the current compiler + library + C++ standard.
 *      DO NOT include non-standard headers here!
 */
#include <cassert>
#include <complex>
#include <functional>
#include <iterator>
#include <utility>

#if __cplusplus <= 201703L
#include <boost/tuple/tuple.hpp>
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
// implement C++17's type traits features if using CUDA with older C++ compiler
#if __cplusplus <= 201402L

// GNU stdlibc++ provides void_t if -gnu++11 or -gnu++14 are given
#if __GNUC__ && defined(__GLIBCXX__) && !__STRICT_ANSI__ && \
    __cplusplus >= 201103L
#define HAVE_VOID_T
#endif

#ifndef HAVE_VOID_T  // implement void_t if needed
namespace std {
template <typename... Ts>
struct make_void {
  using type = void;
};
template <typename... Ts>
using void_t = typename make_void<Ts...>::type;
}  // namespace std
#endif

namespace std {
template <class T>
inline constexpr bool is_integral_v = is_integral<T>::value;
}
#endif  // C++14 only

//////////////////////////////////////////////////////////////////////////////////////////////
// forward declarations

namespace Eigen {

template <typename, int, int, int, int, int>
class Matrix;
template <typename, int, int, int, int, int>
class Array;
template <typename, int, typename>
class Map;

}  // namespace Eigen

namespace madness {

template <typename T>
class Future;

}  // namespace madness

namespace TiledArray {
template <typename>
class Tile;
class DensePolicy;
struct ZeroTensor;
template <typename, typename>
class DistArray;
namespace detail {
template <typename, typename>
class LazyArrayTile;
}  // namespace detail
}  // namespace TiledArray

//////////////////////////////////////////////////////////////////////////////////////////////
// see https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Member_Detector

/// this generates struct \c has_member_##Type<T> whose
/// public constexpr member variable \c value is true if \c T::Member is a
/// member variable or function.
#define GENERATE_HAS_MEMBER(Member)                                            \
  template <typename T, typename Enabler = void>                               \
  class __has_member_##Member : public std::false_type {};                     \
  template <typename T>                                                        \
  class __has_member_##Member<                                                 \
      T, typename std::enable_if<std::is_class<T>::value ||                    \
                                 std::is_union<T>::value>::type> {             \
    using Yes = char[2];                                                       \
    using No = char[1];                                                        \
                                                                               \
    struct Fallback {                                                          \
      int Member;                                                              \
    };                                                                         \
    struct Derived : T, Fallback {};                                           \
                                                                               \
    template <class U>                                                         \
    static No& test(decltype(U::Member)*);                                     \
    template <typename U>                                                      \
    static Yes& test(U*);                                                      \
                                                                               \
   public:                                                                     \
    static constexpr bool value =                                              \
        sizeof(test<Derived>(nullptr)) == sizeof(Yes);                         \
  };                                                                           \
                                                                               \
  template <class T>                                                           \
  struct has_member_##Member                                                   \
      : public std::integral_constant<bool, __has_member_##Member<T>::value> { \
  };

/// this generates struct \c has_member_type_##Type<T> whose
/// public constexpr member variable \c value is true if \c T::Member is a
/// valid type.
#define GENERATE_HAS_MEMBER_TYPE(Type)                                        \
  template <typename T, typename Enabler = void>                              \
  class __has_member_type_##Type : public std::false_type {};                 \
  template <typename T>                                                       \
  class __has_member_type_##Type<                                             \
      T, typename std::enable_if<std::is_class<T>::value ||                   \
                                 std::is_union<T>::value>::type> {            \
    using Yes = char[2];                                                      \
    using No = char[1];                                                       \
                                                                              \
    struct Fallback {                                                         \
      struct Type {};                                                         \
    };                                                                        \
    struct Derived : T, Fallback {};                                          \
                                                                              \
    template <class U>                                                        \
    static No& test(typename U::Type*);                                       \
    template <typename U>                                                     \
    static Yes& test(U*);                                                     \
                                                                              \
   public:                                                                    \
    static constexpr bool value =                                             \
        sizeof(test<Derived>(nullptr)) == sizeof(Yes);                        \
  };                                                                          \
                                                                              \
  template <class T>                                                          \
  struct has_member_type_##Type                                               \
      : public std::integral_constant<bool,                                   \
                                      __has_member_type_##Type<T>::value> {}; \
                                                                              \
  template <class T>                                                          \
  constexpr const bool has_member_type_##Type##_v =                           \
      has_member_type_##Type<T>::value;

/// this generates struct \c has_member_function_Member<T,R,Args...> whose
/// public constexpr member variable \c value is true if \c T::Member is a
/// member function that takes \c Args and returns \c R .
/// \note if T is a const type, only const member functions will be
/// detected, hence
/// \c has_member_function_Member_anyreturn<const T,R,Args...>::value &&
/// \c !has_member_function_Member_anyreturn<T,R,Args...>::value
/// evaluates to true if there is only a const T::Member
#define GENERATE_HAS_MEMBER_FUNCTION(Member)                                   \
  template <typename T, typename Result, typename... Args>                     \
  class __has_member_function_##Member {                                       \
    using Yes = char;                                                          \
    using No = int;                                                            \
    template <typename U, Result (U::*)(Args...)>                              \
    struct Check;                                                              \
    template <typename U, Result (U::*)(Args...) const>                        \
    struct CheckConst;                                                         \
    template <typename U>                                                      \
    static Yes test_const(CheckConst<U, &U::Member>*);                         \
    template <typename U>                                                      \
    static No test_const(...);                                                 \
    template <typename U>                                                      \
    static Yes test_nonconst(Check<U, &U::Member>*);                           \
    template <typename U>                                                      \
    static No test_nonconst(...);                                              \
                                                                               \
   public:                                                                     \
    static constexpr const bool value =                                        \
        sizeof(test_const<T>(0)) == sizeof(Yes) ||                             \
        (!std::is_const<T>::value ? sizeof(test_nonconst<T>(0)) == sizeof(Yes) \
                                  : false);                                    \
  };                                                                           \
  template <class T, typename Result, typename... Args>                        \
  struct has_member_function_##Member                                          \
      : public std::integral_constant<                                         \
            bool, __has_member_function_##Member<T, Result, Args...>::value> { \
  };                                                                           \
                                                                               \
  template <class T, typename Result, typename... Args>                        \
  constexpr const bool has_member_function_##Member##_v =                      \
      has_member_function_##Member<T, Result, Args...>::value;

/// this generates struct \c has_member_function_Member_anyreturn<T,Args...>
/// whose public constexpr member variable \c value is true if \c T::Member
/// is a member function that takes \c Args and returns any type.
/// \note if T is a const type, only const member functions will be
/// detected, hence
/// \c has_member_function_Member_anyreturn<const T,Args...>::value &&
/// \c !has_member_function_Member_anyreturn<T,Args...>::value
/// evaluates to true if there is only a const T::Member
#define GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(Member)                         \
  template <typename T, typename... Args>                                      \
  class __has_member_function_##Member##_anyreturn {                           \
    using Yes = char;                                                          \
    using No = int;                                                            \
    template <typename U, typename... Args_>                                   \
    static auto func(void*)                                                    \
        -> decltype(std::add_pointer_t<decltype(std::declval<U>().Member(      \
                        std::declval<Args_>()...))>{},                         \
                    Yes{});                                                    \
    template <typename U, typename... Args_>                                   \
    static No func(...);                                                       \
                                                                               \
   public:                                                                     \
    static constexpr const bool value =                                        \
        sizeof(func<T, Args...>(0)) == sizeof(Yes);                            \
  };                                                                           \
  template <class T, typename... Args>                                         \
  struct has_member_function_##Member##_anyreturn                              \
      : public std::integral_constant<                                         \
            bool,                                                              \
            __has_member_function_##Member##_anyreturn<T, Args...>::value> {}; \
                                                                               \
  template <class T, typename... Args>                                         \
  constexpr const bool has_member_function_##Member##_anyreturn_v =            \
      has_member_function_##Member##_anyreturn<T, Args...>::value;

/// this generates struct \c is_free_function_Function_anyreturn<Args...> whose
/// public constexpr member variable \c value is true if \c Function is a
/// free function that takes \c Args and returns any value.
/// \note to ensure that \c Function can be looked up if
///       it can't be found via ADL, it may be necessary
///       to add \c "using namespace::Function" BEFORE using this macro.
#define GENERATE_IS_FREE_FUNCTION_ANYRETURN(Function)                          \
  template <typename... Args>                                                  \
  class __is_free_function_##Function##_anyreturn {                            \
    using Yes = char;                                                          \
    using No = int;                                                            \
    template <typename... Args_>                                               \
    static auto func(void*) -> decltype(                                       \
        std::add_pointer_t<decltype(Function(std::declval<Args_>()...))>{},    \
        Yes{});                                                                \
    template <typename...>                                                     \
    static No func(...);                                                       \
                                                                               \
   public:                                                                     \
    static constexpr const bool value =                                        \
        sizeof(func<Args...>(0)) == sizeof(Yes);                               \
  };                                                                           \
  template <typename... Args>                                                  \
  struct is_free_function_##Function##_anyreturn                               \
      : public std::integral_constant<                                         \
            bool, __is_free_function_##Function##_anyreturn<Args...>::value> { \
  };                                                                           \
                                                                               \
  template <typename... Args>                                                  \
  constexpr const bool is_free_function_##Function##_anyreturn_v =             \
      is_free_function_##Function##_anyreturn<Args...>::value;

/// this generates struct \c is_free_function_std_Function_anyreturn<Args...>
/// whose public constexpr member variable \c value is true if \c
/// ::std::Function is a free function that takes \c Args and returns any value.
#define GENERATE_IS_FREE_FUNCTION_STD_ANYRETURN(Function)                      \
  template <typename... Args>                                                  \
  class __is_free_function_std_##Function##_anyreturn {                        \
    using Yes = char;                                                          \
    using No = int;                                                            \
    template <typename... Args_>                                               \
    static auto func(void*)                                                    \
        -> decltype(std::add_pointer_t<decltype(                               \
                        ::std::Function(std::declval<Args_>()...))>{},         \
                    Yes{});                                                    \
    template <typename...>                                                     \
    static No func(...);                                                       \
                                                                               \
   public:                                                                     \
    static constexpr const bool value =                                        \
        sizeof(func<Args...>(0)) == sizeof(Yes);                               \
  };                                                                           \
  template <typename... Args>                                                  \
  struct is_free_function_std_##Function##_anyreturn                           \
      : public std::integral_constant<                                         \
            bool,                                                              \
            __is_free_function_std_##Function##_anyreturn<Args...>::value> {}; \
                                                                               \
  template <typename... Args>                                                  \
  constexpr const bool is_free_function_std_##Function##_anyreturn_v =         \
      is_free_function_std_##Function##_anyreturn<Args...>::value;

namespace TiledArray {
namespace detail {

/////////////////////////////
// standard container traits (incomplete)

GENERATE_HAS_MEMBER_TYPE(value_type)
GENERATE_HAS_MEMBER_TYPE(allocator_type)
GENERATE_HAS_MEMBER_TYPE(size_type)
GENERATE_HAS_MEMBER_TYPE(difference_type)
GENERATE_HAS_MEMBER_TYPE(reference)
GENERATE_HAS_MEMBER_TYPE(const_reference)
GENERATE_HAS_MEMBER_TYPE(pointer)
GENERATE_HAS_MEMBER_TYPE(const_pointer)
GENERATE_HAS_MEMBER_TYPE(iterator)
GENERATE_HAS_MEMBER_TYPE(const_iterator)
GENERATE_HAS_MEMBER_TYPE(reverse_iterator)
GENERATE_HAS_MEMBER_TYPE(const_reverse_iterator)
GENERATE_HAS_MEMBER_TYPE(key_type)
GENERATE_HAS_MEMBER_TYPE(mapped_type)

GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(size)
GENERATE_HAS_MEMBER_FUNCTION(size)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(data)
GENERATE_HAS_MEMBER_FUNCTION(data)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(empty)
GENERATE_HAS_MEMBER_FUNCTION(empty)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(clear)
GENERATE_HAS_MEMBER_FUNCTION(clear)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(resize)

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
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(crbegin)
GENERATE_HAS_MEMBER_FUNCTION(crbegin)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(crend)
GENERATE_HAS_MEMBER_FUNCTION(crend)

/////////////////////////////
// standard iterator traits
// GENERATE_HAS_MEMBER_TYPE(value_type)
// GENERATE_HAS_MEMBER_TYPE(difference_type)
// GENERATE_HAS_MEMBER_TYPE(reference)
// GENERATE_HAS_MEMBER_TYPE(pointer)
GENERATE_HAS_MEMBER_TYPE(iterator_category)

// these are useful to detect presence of overloads
GENERATE_IS_FREE_FUNCTION_ANYRETURN(size)
GENERATE_IS_FREE_FUNCTION_ANYRETURN(data)
GENERATE_IS_FREE_FUNCTION_ANYRETURN(empty)

//////// Tile concept checks

GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(permute)
GENERATE_HAS_MEMBER_FUNCTION(permute)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(add)
GENERATE_HAS_MEMBER_FUNCTION(add)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(add_to)
GENERATE_HAS_MEMBER_FUNCTION(add_to)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(subt)
GENERATE_HAS_MEMBER_FUNCTION(subt)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(subt_to)
GENERATE_HAS_MEMBER_FUNCTION(subt_to)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(mult)
GENERATE_HAS_MEMBER_FUNCTION(mult)
GENERATE_HAS_MEMBER_FUNCTION_ANYRETURN(mult_to)
GENERATE_HAS_MEMBER_FUNCTION(mult_to)

GENERATE_IS_FREE_FUNCTION_ANYRETURN(permute)

}  // namespace detail
}  // namespace TiledArray

namespace TiledArray {
namespace detail {
/// @brief helper to implement other metafunctions
/// @c is_type<T>::value is true if @c T is a valid type
/// @tparam T a type
template <typename>
struct is_type : public std::true_type {};
/// @tparam T a type
/// @c is_type_v<T> is an alias for @c is_type<T>::value
template <typename T>
constexpr const bool is_type_v = is_type<T>::value;

/// @brief helper to implement other metafunctions
/// @c is_complete_type<T>::value is true if @c T is a complete type
/// @tparam T a type
/// @note see
/// https://stackoverflow.com/questions/1625105/how-to-write-is-complete-template
template <typename T, typename = void>
struct is_complete_type : std::false_type {};

template <>
struct is_complete_type<void> : std::false_type {};

template <typename T>
struct is_complete_type<T, decltype(void(sizeof(T)))> : std::true_type {};

/// @tparam T a type
/// @c is_complete_type_v<T> is an alias for @c is_complete_type<T>::value
template <typename T>
constexpr const bool is_complete_type_v = is_complete_type<T>::value;

}  // namespace detail
}  // namespace TiledArray

namespace TiledArray {

/**
 * \addtogroup TileInterface
 * @{
 */

template <typename>
struct eval_trait;

namespace detail {

GENERATE_HAS_MEMBER_TYPE(eval_type)

/// evaluates to true if \c From has an (explicit or implicit) conversion
/// function that produces \c To from \c From , i.e. there exists \c
/// From::operator \c To() \note I do not yet know how to distinguish explicit
/// from implicit operators; some observable behavior does depend on this (e.g.
/// given an explicit converting ctor A::A(C) and an B::operator C(), explicit
/// conversion of B into A will be possible if B::operator C is implicit.
#if !defined(__INTEL_COMPILER_BUILD_DATE)
template <typename From, typename To, typename Enabler = void>
struct has_conversion_operator : std::false_type {};

template <typename From, typename To>
struct has_conversion_operator<
    From, To,
    typename std::enable_if<
        is_type<decltype(std::declval<From>().operator To())>::value>::type>
    : std::true_type {};

template <typename From, typename To>
struct has_conversion_operator<
    From, To,
    typename std::enable_if<
        is_type<decltype(std::declval<From>().operator To&())>::value>::type>
    : std::true_type {};
#else
template <typename From, typename To>
struct has_conversion_operator {
  /*
   * see
   * https://stackoverflow.com/questions/87372/check-if-a-class-has-a-member-function-of-a-given-signature#answer-10707822
   * this works for icc and all other compilers tested:
   * <iframe width="800px" height="200px"
   * src="https://godbolt.org/e?readOnly=true&hideEditorToolbars=true#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAM1QDsCBlZAQwBtMQBGAFlICsQAJlKtmtUMgCkggELSZpAM6Z2yAnjqVMtdAGFUrAK4BbWl1Lb0AGTy1MAORMAjTMRDcArKQAOqRYQ1afSNTc19/dTobO0djFzdPJRVMNUCGAmZiAmCTM04k1UjadMyCaIdnV3cvRQysnND82tLy2PjqgEolVENiZA4AcmkAZltkIywAaklh3UN1VkIATxnsSQAGAEENzYJMY29RPenZgiXvbWZjTEmAMWJUY1JJs4vaK5uAFVRVnfHmRSKSYIAEAfTQtAAbq5/HRQagLsRmERiDtJAB2ORbSY4l7nTBYKiTZAg4iTOiYGZYza4ya1YiGNQvADuqGmmOJpMmAA9JB4ZII%2BQARKnskXDak7Wl7A5HG4zXSvS7XO7PJXvFWfNX4jXy4ZCukEdAgECXJzsUF4KigggK2rGkB4RTg0SAhVYcZKiD2k0e1iQtgKiG1O7SABsqwgHQAdAjXMiSJNPlGOqsTQGjJThmts4bkXhkOS7C9MLUIJM/V6fSA/Rmg3QQ7dw5GY3GkSikymAFSTVMSqW4mWHZF6xU6j6qvFvCefVZ59SFgisktl6NrvuSrbeQzmgsgAc47QmdkySYZwx6g3%2BABemFQVAge1qCvuj2e30j6w6HRO4qveFve8IBJTIfwxcVJXRCC0S2EYxgmUclRtJFCEUX5YK2AB6LswDAJMEBuIc5WmMN1hBZ1gwIUFiEwKhXG0fp4W8BUtRzNZSMmTBuV8LIgWYA9JicVADEwMRiQbDJ6BI9Zz0IkECEmJ0XgZG4rSJSRSNnDjvAeSE8CwNDsVxDT1mwJMTTbBNiC7KNxNoZ9SJ2LtMOmLYiJHBUp2VG5XyeLzdSTH5sx2elGQU8jwToaFiFhWgmPjFFBVgzEBOwyYGFuABJexNjMyyUQAWnIgq0GIGi1AK/xgEmEAf2cgT3L2BV1QnbZgqMnFmgXQ0HQIFSbXxFcCAgb5JggTYTS7DpbMosCUo62lJhogheloHqTT6i8BouKMqQE8CYJpXE0oy7LcvJRErIKrinQIIFasmeqFsaxDxxVNqzIErqCwrFJWC9J9hvDCaQHyxNvm/VyjpxQGIErfEIGBizLo7CHSEhVB9MeubqUW46ewAeRRxMbtqRRo0mAB1eTJmYIT5jpPBgBmW5HswgTpXxQlfs9BHYaR0HibJCGf2W1btqzU8OdxMXiDW2W4q9Dd9qgw7aROrKcrM4AJ1QaKavLJ7oZLWUPNON6bg%2Br6Mm66sqDYZQJaGiA12jHHpZxBX1pAe3WEdpXRRVkUMONtLPgQJTmTwVhWEEm5MEIAiyRM6tNswCWTPJZPSLth306VTOjc5i5ufhnbAZfB5jE/Ugvx/Av%2BxD2lvsLSjBOE2PZN/Lz0zYC9RTS6mCwQRSgUIFm2bRVXG82UKmU%2BdYTwOme54Uz5OCX6fqVXu4l6MrjDgLQgLoS8H1hmiSTyWzAVrlzeILFKfoJDuCqG50FQWyz5sCsUFdAJgAsgABUylYbAAAlUEMgACqoChSgiFJsb%2BaJBDDGZJkWgthqrSEEJlXQugKyoFLJMWgqAFKsmIAAa0mFHAgI86E3CoHgGKCloqxXJESBhylmDj0EElbYqCVDKBCjbAsoIATKCyBACKEI2GBHiu2EgL53zrDTCAWSzwcE9BJKovhytYKCJ0FaER%2BZkDiMBK4YauEZFRRhPIsGqJZi3HfJwNRGjph8O0QgVxei9pbBbuYyRw0bFQjsXCBxgonEqLcX3TAmjPGMgQLowQ%2BjZ6iLMRIyxEBrFglkWEuKETlFJlcdmXumZ4mCC8T4lJVIBhdFYCAAYHgBikDMAMdYLTUCNN0PIeQdIeh9HlKgwQLSCCNI6d%2BUglD3DrGjMMYYHhOCCAABxhjDB4bgYYACc6JlkiEadwFpbSOmkC6QMFpigQDrFIGM9pdTSBwFgEgNABxo6uHIJQF53g3luALMgTgWz8hMP%2BjCSgThxktKcLYTISxGmcBaS8649ACa0FYLCu5pAsDGDEMAdgELMXMJSOoaElyMVcRSPMQY8LyD0BUPixYTgkTECWPoLAcLRnEDwMYdlXQaD0CYGwDgPB%2BDmFEOIFAvTZAiDwE4S5kAugIiKKSyqRoZhCikLIeQPALnJFSJoCAlgGh5AsDoVolQ3D5HCAEOgRqwh%2BGtbQM1cQqhNF1UUEo9QDC5HMMoQoaQ6hlFsBUZ1FqlABttU0ANTr2icC6IoAZ/QhD1Mac01p%2BKznclWQVMM3BFLIELAC6MsyN4QFwIQRMIxhCTH0K89gydUGxurZKmQoyIWTIIswLAbgoxTKEIIOZ3BuDDHRJ4dYHhlnrEEFs5ZwgGkDEOWmjFZyLlXJuW2h5MBECg3mNuAgHzgKPG%2BXW8wBJy1uGEHyxgLA8XCuZEibwPL9lNKOemxpIwN60JHpmsM2bc1/MmIW4tra7mTOmcOuZgh1jcC2WGGdyz0SCAQ7wOdC7jmdMaSu65tyJnJoGCMxdJzl1rpA10ORmhuBAA"></iframe>
   */

  /* operator has correct sig */
  template <typename A>
  static std::true_type test(To (A::*)() const) {
    return std::true_type();
  }

  /* operator exists */
  template <typename A>
  static decltype(test(&A::operator To)) test(decltype(&A::operator To),
                                              void*) {
    /* Operator exists. What about sig? */
    typedef decltype(test(&A::operator To)) return_type;
    return return_type();
  }

  /* operator does not exist */
  template <typename A>
  static std::false_type test(...) {
    return std::false_type();
  }

  /* This will be either `std::true_type` or `std::false_type` */
  typedef decltype(test<From>(0, 0)) type;

  static const bool value = type::value; /* Which is it? */
};
#endif

/// \c has_conversion_operator_v<From, To> is an alias for \c
/// has_conversion_operator<From, To>::value
template <class From, class To>
constexpr const bool has_conversion_operator_v =
    has_conversion_operator<From, To>::value;

/// evaluates to true if can construct \c To from \c From , i.e. if there is
/// a converting constructor \c To::To(From) or if \c From has an implicit
/// or explicit conversion function to \c To, i.e. \c operator \c To()
template <class From, class To>
struct is_explicitly_convertible : public std::is_constructible<To, From> {};

/// \c is_explicitly_convertible_v<From, To> is an alias for \c
/// is_explicitly_convertible<From, To>::value
template <class From, class To>
constexpr const bool is_explicitly_convertible_v =
    is_explicitly_convertible<From, To>::value;

/// evaluates to true if can implicitly convert \c To from \c From , i.e.
/// if \c From has an implicit
/// conversion function to \c To, i.e. \c operator \c To()
/// \note this is just an alias to std::is_convertible
template <class From, class To>
struct is_implicitly_convertible : public std::is_convertible<From, To> {};

/// \c is_implicitly_convertible_v<From, To> is an alias for \c
/// is_implicitly_convertible<From, To>::value
template <class From, class To>
constexpr const bool is_implicitly_convertible_v =
    is_implicitly_convertible<From, To>::value;

/// evaluates to true if can convert \c To from \c From , either explicitly
/// or implicitly
/// \note contrast to std::is_convertible which checks for implicit conversion
/// only
template <class From, class To>
struct is_convertible
    : public std::integral_constant<
          bool, is_implicitly_convertible<From, To>::value ||
                    is_explicitly_convertible<From, To>::value> {};

/// \c is_convertible_v<From, To> is an alias for \c is_convertible<From,
/// To>::value
template <class From, class To>
constexpr const bool is_convertible_v = is_convertible<From, To>::value;

template <typename T, typename Enabler = void>
struct eval_trait_base {
  typedef T type;
  static constexpr bool is_consumable = false;
  static constexpr bool nonblocking = false;
};  // struct eval_trait

template <typename T>
struct eval_trait_base<
    T,
    typename std::enable_if<
        has_member_type_eval_type<T>::value &&
        (detail::is_explicitly_convertible<T, typename T::eval_type>::value ||
         detail::is_explicitly_convertible<
             T, madness::Future<typename T::eval_type>>::value ||
         detail::is_implicitly_convertible<T, typename T::eval_type>::value ||
         detail::is_implicitly_convertible<
             T, madness::Future<typename T::eval_type>>::value)>::type> {
  typedef typename T::eval_type type;
  static constexpr bool is_consumable = false;
  static constexpr bool nonblocking =
      detail::is_explicitly_convertible<
          T, madness::Future<typename T::eval_type>>::value ||
      detail::is_implicitly_convertible<
          T, madness::Future<typename T::eval_type>>::value;
};  // struct eval_trait

}  // namespace detail

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
struct eval_trait : public TiledArray::detail::eval_trait_base<T> {};

/// Detect lazy evaluation tiles

/// \c is_lazy_tile evaluates to \c std::true_type when T is a tile that
/// uses the lazy evaluation mechanism (i.e. when <tt>T != T::eval_type</tt>),
/// otherwise it evaluates to \c std::false_type .
/// \tparam T The tile type to test
template <typename T>
struct is_lazy_tile
    : public std::integral_constant<
          bool, !std::is_same<T, typename eval_trait<T>::type>::value> {};

template <typename Tile, typename Policy>
struct is_lazy_tile<DistArray<Tile, Policy>> : public std::false_type {};

/// \c is_lazy_tile_v<T> is an alias for \c is_lazy_tile<T>::value
template <typename T>
constexpr const bool is_lazy_tile_v = is_lazy_tile<T>::value;

/// Consumable tile type trait

/// This trait is used to determine if a tile type is consumable in tensor
/// arithmetic operations. That is, temporary tiles may appear on the left-
/// hand side of add-to (+=), subtract-to (-=), and multiply-to (*=)
/// operations. By default, all tile types are assumed to be
/// consumable except lazy tiles. Users should provide a (partial)
/// specialization of this `struct` to disable consumable tile operations.
/// \tparam T The tile type
template <typename T>
struct is_consumable_tile
    : public std::integral_constant<bool, !is_lazy_tile<T>::value> {};

template <>
struct is_consumable_tile<ZeroTensor> : public std::false_type {};

/// \c is_consumable_tile_v<T> is an alias for \c is_consumable_tile<T>::value
template <typename T>
constexpr const bool is_consumable_tile_v = is_consumable_tile<T>::value;

/** @}*/

namespace detail {

template <typename T>
struct is_complex : public std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : public std::true_type {};

/// \c is_complex_v<T> is an alias for \c is_complex<T>::value
template <typename T>
constexpr const bool is_complex_v = is_complex<T>::value;

template <typename T>
struct is_numeric : public std::is_arithmetic<T> {};

template <typename T>
struct is_numeric<std::complex<T>> : public is_numeric<T> {};

template <>
struct is_numeric<bool> : public std::false_type {};

/// \c is_numeric_v<T> is an alias for \c is_numeric<T>::value
template <typename T>
constexpr const bool is_numeric_v = is_numeric<T>::value;

/// SFINAE type for enabling code when \c T is a numeric type
template <typename T, typename U = void>
using enable_if_numeric_t = std::enable_if_t<is_numeric_v<T>, U>;

template <typename T>
struct is_scalar : public is_numeric<T> {};

template <typename T>
struct is_scalar<std::complex<T>> : public std::false_type {};

/// \c is_scalar_v<T> is an alias for \c is_scalar_v<T>
template <typename T>
constexpr const bool is_scalar_v = is_scalar<T>::value;

/// Detect tiles used by \c ArrayEvalImpl

/// \c is_array_tile evaluates to \c std::true_type when \c T is a \c
/// LazyArrayTile<U> , i.e. when it is a lazy tile wrapper used by e.g. \c
/// ArrayEvalImpl . otherwise it evaluates to \c std::false_type . Note that \c
/// is_array_tile<T> implies \c is_lazy_tile<T> , but \c is_lazy_tile<T> does
/// not imply \c is_array_tile<T> . \tparam T The tile type to test
template <typename T>
struct is_array_tile : public std::false_type {};

template <typename T, typename Op>
struct is_array_tile<TiledArray::detail::LazyArrayTile<T, Op>>
    : public std::true_type {};  // struct is_array_tile

/// \c is_array_tile_v<T> is an alias for \c is_array_tile<T>::value
template <typename T>
constexpr const bool is_array_tile_v = is_array_tile<T>::value;

/// Detect a lazy evaluation tile that are not a \c LazyArrayTile

/// \c is_non_array_lazy_tile evaluates to \c std::true_type when T is a
/// tile that uses the lazy evaluation mechanism (i.e. when
/// <tt>T != T::eval_type</tt>), and not a \c LazyArrayTile , otherwise it
/// evaluates to \c std::false_type .
/// \tparam T The tile type to test
template <typename T>
struct is_non_array_lazy_tile
    : public std::integral_constant<bool, is_lazy_tile<T>::value &&
                                              (!is_array_tile<T>::value)> {
};  // struct is_non_array_lazy_tile

/// \c is_non_array_lazy_tile_v<T> is an alias for \c
/// is_non_array_lazy_tile<T>::value
template <typename T>
constexpr const bool is_non_array_lazy_tile_v =
    is_non_array_lazy_tile<T>::value;

/// Type trait for extracting the numeric type of tensors and arrays.

/// \tparam T The type to extract a numeric type from
/// \tparam Enabler Type used to selectively implement partial specializations
/// -# if T is numeric, numeric_type<T>::type evaluates to T
/// -# if T is not numeric and T::value_type is a valid type, will evaluate to
/// numeric_type<T::value_type>::type,
///    and so on recursively
/// -# otherwise it's undefined
template <typename T, typename Enabler = void>
struct numeric_type;

template <typename T>
struct numeric_type<T, typename std::enable_if<is_numeric_v<T>>::type> {
  typedef T type;
};

template <typename T>
struct numeric_type<
    T, typename std::enable_if<has_member_type_value_type<T>::value &&
                               (!is_lazy_tile<T>::value) &&
                               (!is_numeric_v<T>)>::type>
    : public numeric_type<typename T::value_type> {};

template <typename T>
struct numeric_type<T, typename std::enable_if<is_lazy_tile<T>::value &&
                                               !is_numeric_v<T>>::type>
    : public numeric_type<typename eval_trait<T>::type> {};

template <typename T, int Rows, int Cols, int Opts, int MaxRows, int MaxCols>
struct numeric_type<Eigen::Matrix<T, Rows, Cols, Opts, MaxRows, MaxCols>, void>
    : public numeric_type<typename Eigen::Matrix<T, Rows, Cols, Opts, MaxRows,
                                                 MaxCols>::Scalar> {};

template <typename T, int Rows, int Cols, int Opts, int MaxRows, int MaxCols>
struct numeric_type<Eigen::Array<T, Rows, Cols, Opts, MaxRows, MaxCols>, void>
    : public numeric_type<typename Eigen::Matrix<T, Rows, Cols, Opts, MaxRows,
                                                 MaxCols>::Scalar> {};

template <typename PlainObjectType, int MapOptions, typename StrideType>
struct numeric_type<Eigen::Map<PlainObjectType, MapOptions, StrideType>, void>
    : public numeric_type<PlainObjectType> {};

/// \c numeric_t<T> is an alias for \c numeric_type<T>::type
template <typename T>
using numeric_t = typename TiledArray::detail::numeric_type<T>::type;

/// Type trait for extracting the scalar type of tensors and arrays.

/// \tparam T The type to extract a numeric type from
/// \tparam Enabler Type used to selectively implement partial
/// specializations
/// -# if T is a scalar type, i.e. \c is_scalar_v<T> is true (e.g. \c
///    int or \c float), \c scalar_type<T>::type evaluates to \c T
/// -# if T is std::complex<U>, scalar_type<T>::type evaluates to U
/// -# if T is not a scalar or complex type, will evaluate to \c
///    scalar_type<numeric_type<T>::type>::type, and so on recursively
/// -# otherwise it's undefined
template <typename T, typename Enabler = void>
struct scalar_type;

template <typename T>
struct scalar_type<T, typename std::enable_if<is_scalar_v<T>>::type> {
  typedef T type;
};

template <typename T>
struct scalar_type<std::complex<T>, void> : public scalar_type<T> {};

template <typename T>
struct scalar_type<T, typename std::enable_if<!is_numeric_v<T>>::type>
    : public scalar_type<typename numeric_type<T>::type> {};

/// \c scalar_t<T> is an alias for \c scalar_type<T>::type
template <typename T>
using scalar_t = typename TiledArray::detail::scalar_type<T>::type;

template <typename T>
struct is_strictly_ordered_helper {
  using Yes = char;
  using No = int;
  template <typename U>
  static auto test(void*) -> decltype(
      std::add_pointer_t<decltype(std::declval<U>() < std::declval<U>())>{},
      Yes{});
  template <typename...>
  static No test(...);

 public:
  static constexpr const bool value = sizeof(test<T>(0)) == sizeof(Yes);
};

/// \c is_strictly_ordered<T>::value is true if strict order is defined for T,
/// i.e. "T < T" is defined
template <typename T>
struct is_strictly_ordered
    : public std::integral_constant<bool,
                                    is_strictly_ordered_helper<T>::value> {};

/// \c is_strictly_ordered_v<T> is an alias for \c is_strictly_ordered<T>::value
template <typename T>
constexpr const bool is_strictly_ordered_v = is_strictly_ordered<T>::value;

template <typename...>
struct is_integral_list_helper;

template <typename T, typename... Ts>
struct is_integral_list_helper<T, Ts...> {
  static constexpr bool value =
      std::is_integral<T>::value && is_integral_list_helper<Ts...>::value;
};

template <>
struct is_integral_list_helper<> {
  static constexpr bool value = true;
};

/// @tparam Ts parameter pack
/// @c is_integral_list<Ts...>::value is true if for every type @c T in @c Ts...
/// std::is_integral<T>::value is true
template <typename... Ts>
struct is_integral_list
    : std::conditional<(sizeof...(Ts) > 0ul), is_integral_list_helper<Ts...>,
                       std::false_type>::type {};

/// \c is_integral_list_v<T> is an alias for \c is_integral_list<T>::value
template <typename... Ts>
constexpr const bool is_integral_list_v = is_integral_list<Ts...>::value;

//////////////////////////////////////////////////////////////////////////////////////////////

/// @tparam T a type
/// @c is_tuple<T>::value is true if @c T is an @c std::tuple<...>
template <class T>
struct is_tuple : std::false_type {};

template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

/// \c is_tuple_v<T> is an alias for \c is_tuple<T>::value
template <typename T>
constexpr const bool is_tuple_v = is_tuple<T>::value;

//////////////////////////////////////////////////////////////////////////////////////////////

#if __cplusplus <= 201703L
template <std::size_t I, typename T, typename = void>
struct is_std_gettable : std::false_type {};

template <std::size_t I, typename T>
struct is_std_gettable<
    I, T, std::void_t<decltype(::std::get<I>(std::declval<const T&>()))>>
    : std::true_type {};

template <std::size_t I, typename T>
constexpr const bool is_std_gettable_v = is_std_gettable<I, T>::value;

template <std::size_t I, typename T, typename = void>
struct is_boost_gettable : std::false_type {};

template <std::size_t I, typename T>
struct is_boost_gettable<
    I, T, std::void_t<decltype(::boost::get<I>(std::declval<const T&>()))>>
    : std::true_type {};

template <std::size_t I, typename T>
constexpr const bool is_boost_gettable_v = is_boost_gettable<I, T>::value;

template <std::size_t I, typename T>
constexpr const bool is_gettable_v =
    is_std_gettable_v<I, T> || is_boost_gettable_v<I, T>;

template <std::size_t I, typename T>
auto get(T&& t) {
  using boost::get;
  using std::get;
  return get<I>(std::forward<T>(t));
}
#else  // C++20
template <std::size_t I, typename T, typename = void>
struct is_gettable : std::false_type {};

template <std::size_t I, typename T>
struct is_gettable<I, T, std::void_t<decltype(get<I>(std::declval<T>()))>>
    : std::true_type {};

template <std::size_t I, typename T>
constexpr const bool is_gettable_v = is_gettable<I, T>::value;
#endif

/// @tparam T a type
/// @c is_gettable_pair<T>::value is true if @c TiledArray::detail::get<0>(T)
/// and @c TiledArray::detail::get<1>(T) are valid expressions
template <class T, typename Enabler = void>
struct is_gettable_pair : std::false_type {};

template <typename T>
struct is_gettable_pair<
    T, std::enable_if_t<is_gettable_v<0, T> && is_gettable_v<1, T>>>
    : std::true_type {};

/// \c is_gettable_pair_v<T> is an alias for \c is_gettable_pair<T>::value
template <typename T>
constexpr const bool is_gettable_pair_v = is_gettable_pair<T>::value;

//////////////////////////////////////////////////////////////////////////////////////////////

template <class T, class = void>
struct is_integral_pair_ : std::false_type {};
template <class T1, class T2>
struct is_integral_pair_<
    std::pair<T1, T2>,
    typename std::enable_if<std::is_integral<T1>::value &&
                            std::is_integral<T2>::value>::type>
    : std::true_type {};
/// @tparam T a type
/// @c is_integral_pair<T>::value is true if @c T is an @c std::pair<T1,T2> and
/// both @c std::is_integral<T1>::value and @c std::is_integral<T2>::value are
/// true
template <class T>
struct is_integral_pair : is_integral_pair_<T> {};

/// \c is_integral_pair_v<T> is an alias for \c is_integral_pair<T>::value
template <typename T>
constexpr const bool is_integral_pair_v = is_integral_pair<T>::value;

//////////

template <typename...>
struct is_integral_pair_list_helper;

template <typename T, typename... Ts>
struct is_integral_pair_list_helper<T, Ts...> {
  static constexpr bool value =
      is_integral_pair<T>::value && is_integral_pair_list_helper<Ts...>::value;
};

template <>
struct is_integral_pair_list_helper<> {
  static constexpr bool value = true;
};

/// @tparam Ts a parameter pack
/// @c is_integral_pair_list<Ts...>::value is true if for every @c T in @c Ts...
/// @c is_integral_pair<T>::value is true
template <typename... Ts>
struct is_integral_pair_list
    : std::conditional<(sizeof...(Ts) > 0ul),
                       is_integral_pair_list_helper<Ts...>,
                       std::false_type>::type {};

/// \c is_integral_pair_list_v<T> is an alias for \c
/// is_integral_pair_list<T>::value
template <typename... Ts>
constexpr const bool is_integral_pair_list_v =
    is_integral_pair_list<Ts...>::value;

/// @tparam T a type
/// @c is_integral_tuple<T>::value is true if @c T is @c std::tuple<Ts...> and
/// @c std::is_integral_list<Ts...>::value is true
template <typename T>
struct is_integral_tuple : std::false_type {};
template <typename... Ts>
struct is_integral_tuple<std::tuple<Ts...>> : is_integral_list<Ts...> {};

/// \c is_integral_tuple_v<T> is an alias for \c is_integral_tuple<T>::value
template <typename T>
constexpr const bool is_integral_tuple_v = is_integral_tuple<T>::value;

/// Remove const, volatile, and reference qualifiers.
template <typename T>
struct remove_cvr {
  typedef typename std::remove_cv<typename std::remove_reference<T>::type>::type
      type;
};

template <typename T>
using remove_cvr_t = typename remove_cvr<T>::type;

/// prepends \c const to \c T if \c B is \c true
template <bool B, typename T>
using const_if_t = typename std::conditional<B, const T, T>::type;

template <typename T, typename Enabler = void>
struct param {
  typedef
      typename std::add_lvalue_reference<typename std::add_const<T>::type>::type
          type;
};

template <typename T>
struct param<T, typename std::enable_if<is_numeric_v<T>>::type> {
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

struct non_iterator_tag {};

template <typename T, typename Enabler = void>
struct is_iterator : public std::false_type {
  typedef non_iterator_tag iterator_category;
};

template <typename T>
struct is_iterator<T, typename std::enable_if<
                          has_member_type_iterator_category<T>::value>::type>
    : public std::true_type {
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
struct is_input_iterator
    : public std::is_base_of<std::input_iterator_tag,
                             typename is_iterator<T>::iterator_category> {};

template <typename T>
struct is_output_iterator
    : public std::is_base_of<std::output_iterator_tag,
                             typename is_iterator<T>::iterator_category> {};

template <typename T>
struct is_forward_iterator
    : public std::is_base_of<std::forward_iterator_tag,
                             typename is_iterator<T>::iterator_category> {};

template <typename T>
struct is_bidirectional_iterator
    : public std::is_base_of<std::bidirectional_iterator_tag,
                             typename is_iterator<T>::iterator_category> {};

template <typename T>
struct is_random_iterator
    : public std::is_base_of<std::random_access_iterator_tag,
                             typename is_iterator<T>::iterator_category> {};

//////////////////////////////////////////////////////////////////////////////////////////////

/// @tparam T a type
/// @c is_range<T>::value is true if @c std::begin(T&) and @c std::end(T&) are
/// defined
/// @warning will be replaced by C++20 concepts
template <typename T, typename Enabler = void>
struct is_range : std::false_type {};

template <typename T>
struct is_range<T, std::void_t<decltype(std::begin(std::declval<T&>()),
                                        std::end(std::declval<T&>()))>>
    : std::true_type {};

/// \c is_range_v<T> is an alias for \c is_range<T>::value
template <typename T>
static constexpr bool is_range_v = is_range<T>::value;

/// @tparam T a type
/// @c is_sized_range<T>::value is true if `is_range_v<T>` is true and
/// `std::size(T&)` is valid
/// @warning will be replaced by C++20 concepts
template <typename T, typename Enabler = void>
struct is_sized_range : std::false_type {};

template <typename T>
struct is_sized_range<T, std::void_t<decltype(std::size(std::declval<T&>()))>>
    : is_range<T> {};

/// `is_sized_range_v<T>` is an alias for `is_sized_range<T>::value`
template <typename T>
static constexpr bool is_sized_range_v = is_sized_range<T>::value;

/// @tparam T a type
/// @c is_contiguous_range<T>::value is true if `is_range_v<T>` is true and
/// `std::data(T&)` is valid
/// @warning will be replaced by C++20 concepts
template <typename T, typename Enabler = void>
struct is_contiguous_range : std::false_type {};

template <typename T>
struct is_contiguous_range<T,
                           std::void_t<decltype(std::data(std::declval<T&>()))>>
    : is_range<T> {};

/// `is_contiguous_range_v<T>` is an alias for `is_contiguous_range<T>::value`
template <typename T>
static constexpr bool is_contiguous_range_v = is_contiguous_range<T>::value;

/// @tparam T a range type
/// @c iterator_t<T> is the iterator type, i.e. the type returned by @c
/// std::begin(T&)
/// @warning will be replaced by C++20 ranges::iterator_t
template <class T>
using iterator_t = decltype(std::begin(std::declval<T&>()));

/// @tparam T a range type
/// @c value_t<T> is the value type, i.e. the type to which @c std::begin(T&)
/// dereferences to
/// @warning will be replaced by C++20 ranges::value_t
template <class T>
using value_t = remove_cvr_t<decltype(*std::begin(std::declval<T&>()))>;

/// @tparam T a type
/// `is_integral_range<T>::value` is true if @p T is a range type that
/// dereferences to values for which `std::is_integral_v` is true
template <typename T, typename Enabler = void>
struct is_integral_range : std::false_type {};

template <typename T>
struct is_integral_range<
    T, std::enable_if_t<std::is_integral_v<value_t<T>> && is_range_v<T>>>
    : std::true_type {};

/// `is_integral_range_v<T>` is an alias for `is_integral_range<T>::value`
template <typename T>
static constexpr bool is_integral_range_v = is_integral_range<T>::value;

/// @tparam T a type
/// @c is_integral_sized_range<T>::value is true if @p T is a sized range type
/// that dereferences to values for which std::is_integral is true
template <typename T, typename Enabler = void>
struct is_integral_sized_range : std::false_type {};

template <typename T>
struct is_integral_sized_range<
    T, std::enable_if_t<std::is_integral_v<value_t<T>> && is_sized_range_v<T>>>
    : std::true_type {};

/// \c is_integral_sized_range_v<T> is an alias for \c
/// is_integral_sized_range<T>::value
template <typename T>
static constexpr bool is_integral_sized_range_v =
    is_integral_sized_range<T>::value;

//////////////////////////////////////////////////////////////////////////////////////////////

// Type traits used to determine the result of arithmetic operations between
// two types.

template <typename Scalar1, typename Scalar2>
using add_t = decltype(std::declval<Scalar1>() + std::declval<Scalar2>());

template <typename Scalar1, typename Scalar2>
using subt_t = decltype(std::declval<Scalar1>() - std::declval<Scalar2>());

template <typename Scalar1, typename Scalar2>
using mult_t = decltype(std::declval<Scalar1>() * std::declval<Scalar2>());

template <typename T>
struct is_array : public std::false_type {};

template <typename T, typename P>
struct is_array<DistArray<T, P>> : public std::true_type {};

template <typename T>
static constexpr bool is_array_v = is_array<T>::value;

template <typename T>
using trange_t = typename T::trange_type;

template <typename T>
using shape_t = typename T::shape_type;

template <typename T>
using pmap_t = typename T::pmap_interface;

template <typename T>
using policy_t = typename T::policy_type;

/// If \c Base is a base of \c Derived, \c if_same_or_derived::value is \c true.
/// \tparam Base supposed base class
/// \tparam Derived supposed derived class
template <typename Base, typename Derived>
struct is_same_or_derived
    : std::conditional<
          std::is_base_of<Base, typename std::decay<Derived>::type>::value,
          std::true_type, std::false_type>::type {};

//////////////////////////////////////////////////////////////////////////////////////////////

/// @tparam T a type
/// @c is_pair<T>::value is true if @c T is an @c std::pair<...>
template <typename T>
struct is_pair : public std::false_type {};

template <typename T1, typename T2>
struct is_pair<std::pair<T1, T2>> : public std::true_type {};

/// \c is_pair_v<T> is an alias for \c is_pair<T>::value
template <typename T>
constexpr const bool is_pair_v = is_pair<T>::value;

//////////////////////////////////////////////////////////////////////////////////////////////

/** @brief Primary template for detecting if @p T is an std::initializer_list
 *
 *  This is the primary template for detecting if a type @p T is an
 *  std::initializer_list it will be selected when @p T is **NOT** an
 *  std::initializer_list and will contain a constexpr static member `value`,
 *  which is always false.
 *
 *  @tparam T The type we are testing for its initializer_list-ness
 */
template <typename T>
struct is_initializer_list : std::false_type {};

/** @brief Specialization of is_initializer_list for an std::initializer_list
 *
 *  This specialization is selected if the template type parameter to
 *  is_initializer_list is of the form `std::initializer_list<T>` and will
 * contain a constexpr static member `value` which is always true.
 *
 *  @tparam T The type we are testing for its initializer_list-ness
 */
template <typename T>
struct is_initializer_list<std::initializer_list<T>> : std::true_type {};

/** @brief Helper variable template for the is_initializer_list struct.
 *
 *  This helper variable conforms to the STL's practice of declaring a helper
 *  variable to retrieve the static member `value` of a struct. The value of
 *  `is_initializer_list_v<T>` will be the same as
 * `is_initializer_list<T>::value`
 *
 *  @tparam T The type we want to know the initializer_list-ness of.
 */
template <typename T>
static constexpr bool is_initializer_list_v = is_initializer_list<T>::value;

//////////////////////////////////////////////////////////////////////////////////////////////

/// This evaluates to true if \c T is a generalized pair, namely if expressions
/// `get<0>(T&)` and `get<1>(T&)` are valid, or \c T is a contiguous range
template <typename T>
static constexpr bool is_gpair_v =
    is_contiguous_range_v<T> || is_gettable_pair_v<T>;

/// @tparam T a type
/// `is_gpair_range<T>::value` is true if @p T is a range type that
/// dereferences to values for which is_gpair_v is true
template <typename T, typename Enabler = void>
struct is_gpair_range : std::false_type {};

template <typename T>
struct is_gpair_range<T,
                      std::enable_if_t<is_gpair_v<value_t<T>> && is_range_v<T>>>
    : std::true_type {};

/// `is_gpair_range_v<T>` is an alias for `is_gpair_range<T>::value`
template <typename T>
static constexpr bool is_gpair_range_v = is_gpair_range<T>::value;

/// \c at(pair, i) extracts i-th element from gpair
template <typename GeneralizedPair,
          typename = std::enable_if_t<is_gpair_v<GeneralizedPair>>>
decltype(auto) at(GeneralizedPair&& v, std::size_t idx) {
  assert(idx == 0 || idx == 1);
  if constexpr (is_gettable_pair_v<std::decay_t<decltype(v)>>) {
#if __cplusplus <= 201703L
    return idx == 0 ? detail::get<0>(v) : detail::get<1>(v);
#else
    return idx == 0 ? get<0>(v) : get<1>(v);
#endif
  } else if constexpr (is_contiguous_range_v<std::decay_t<decltype(v)>>) {
    assert(std::size(v) == 2);
    return std::data(v)[idx];
  }
  abort();  // unreachable
};

//////////////////////////////////////////////////////////////////////////////////////////////
// https://stackoverflow.com/questions/51187974/can-stdis-invocable-be-emulated-within-c11

template <typename F, typename... Args>
struct is_invocable
    : std::is_constructible<
          std::function<void(Args...)>,
          std::reference_wrapper<typename std::remove_reference<F>::type>> {};

template <typename R, typename F, typename... Args>
struct is_invocable_r
    : std::is_constructible<
          std::function<R(Args...)>,
          std::reference_wrapper<typename std::remove_reference<F>::type>> {};

template <typename Enabler, typename F, typename... Args>
struct is_invocable_void_helper : std::false_type {};
template <typename F, typename... Args>
struct is_invocable_void_helper<
    std::enable_if_t<std::is_void<std::result_of_t<F(Args...)>>::value, void>,
    F, Args...> : std::true_type {};
template <typename F, typename... Args>
struct is_invocable_void : is_invocable_void_helper<void, F, Args...> {};

template <typename T>
struct type_printer;
}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_TYPE_TRAITS_H__INCLUDED
