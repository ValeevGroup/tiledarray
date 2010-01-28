#ifndef TILEDARRAY_TYPE_TRAITS_H__INCLUDED
#define TILEDARRAY_TYPE_TRAITS_H__INCLUDED

#include <iterator>
#include <boost/type_traits.hpp>

namespace TiledArray {
  namespace detail {

    /// adds const to the type if the boolean parameter is true
    template<bool B, typename T>
    struct add_const {
      typedef const typename boost::remove_const<T>::type type;
    }; // struct add_const

    template<typename T>
    struct add_const<false, T> {
      typedef typename boost::remove_const<T>::type type;
    }; // struct add_const<false, T>

    template<class UnaryFunc>
    struct unary_functor_types {
      typedef typename UnaryFunc::result_type result_type;
      typedef typename UnaryFunc::argument_type argument_type;
    }; // struct unary_functor_types

    template<class R, class A>
    struct unary_functor_types<R(*)(A)> {
      typedef R result_type;
      typedef A argument_type;
    }; // struct unary_functor_types<Return(*)(Argument)>

    template<class BinaryFunc>
    struct binary_functor_types {
      typedef typename BinaryFunc::result_type result_type;
      typedef typename BinaryFunc::first_argument_type first_argument_type;
      typedef typename BinaryFunc::second_argument_type second_argument_type;
    }; // struct binary_functor_types

    template<class R, class A1, class A2>
    struct binary_functor_types<R(*)(A1, A2)> {
      typedef R result_type;
      typedef A1 first_argument_type;
      typedef A2 second_argument_type;
    }; // struct binary_functor_types<Return(*)(Argument)>

    /// The static member value is true if the type T is a random access iterator
    template<typename T>
    struct is_random_iterator {
      static const bool value = boost::is_same<std::random_access_iterator_tag,
          typename std::iterator_traits<T>::iterator_category>::value;
    }; // struct is_random_iterator

    /// The static member value is true if the type T is a bidirectional iterator
    template<typename T>
    struct is_bidirectional_iterator {
      static const bool value = is_random_iterator<T>::value ||
          boost::is_same<std::bidirectional_iterator_tag,
          typename std::iterator_traits<T>::iterator_category>::value;
    }; // struct is_bidirectional_iterator

    /// The static member value is true if the type T is a forward iterator
    template<typename T>
    struct is_forward_iterator {
      static const bool value = is_bidirectional_iterator<T>::value ||
          boost::is_same<std::forward_iterator_tag,
          typename std::iterator_traits<T>::iterator_category>::value;
    }; // struct is_forward_iterator

    /// The static member value is true if the type T is an output iterator
    template<typename T>
    struct is_output_iterator {
      static const bool value = is_forward_iterator<T>::value ||
          boost::is_same<std::output_iterator_tag,
          typename std::iterator_traits<T>::iterator_category>::value;
    }; // struct is_output_iterator

    /// The static member value is true if the type T is an input iterator
    template<typename T>
    struct is_input_iterator {
      static const bool value = is_forward_iterator<T>::value ||
          boost::is_same<std::input_iterator_tag,
          typename std::iterator_traits<T>::iterator_category>::value;
    }; // struct is_input_iterator

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// value is equal to the number of template parameters.
    template<typename... Args> struct Count;

    template<typename T, typename... Args>
    struct Count<T, Args...> {
      static const unsigned int value = Count<Args...>::value + 1u;
    }; // struct Count<T, Args...>

    template<>
    struct Count<> {
      static const unsigned int value = 0u;
    }; // struct Count<>

    /// value is true if all template parameter types are integral types.
    template<typename... Args> struct is_integral_list;

    template<typename T, typename... Args>
    struct is_integral_list<T, Args...> {
      static const bool value = is_integral_list<Args...>::value && boost::is_integral<T>::value;
    }; // struct is_integral_list<T, Args...>

    template<>
    struct is_integral_list<> {
      static const bool value = true;
    }; // is_integral_list<>
#endif // __GXX_EXPERIMENTAL_CXX0X__

  } // namespace detail
} // namespace TiledArray
#endif // TILEDARRAY_TYPE_TRAITS_H__INCLUDED
