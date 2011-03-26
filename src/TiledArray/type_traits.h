#ifndef TILEDARRAY_TYPE_TRAITS_H__INCLUDED
#define TILEDARRAY_TYPE_TRAITS_H__INCLUDED

#include <TiledArray/config.h>
#include <iterator>
#include <boost/iterator/iterator_traits.hpp>
#include <world/enable_if.h>
#include <world/typestuff.h>

namespace TiledArray {
  namespace detail {

    template <class T, class U>
    class is_convertible
    {
       typedef char true_t;
       class false_t { char dummy[2]; };
       static true_t dispatch(U);
       static false_t dispatch(...);
       static T trigger();
       public:
       enum { value = sizeof(dispatch(trigger())) == sizeof(true_t) };
    };

    template <typename T>
    struct has_iterator_catagory
    {
        // yes and no are guaranteed to have different sizes,
        // specifically sizeof(yes) == 1 and sizeof(no) == 2
        typedef char yes[1];
        typedef char no[2];

        template <typename C>
        static yes& test(typename C::iterator_category*);

        template <typename>
        static no& test(...);

        // if the sizeof the result of calling test<T>(0) is equal to the sizeof(yes),
        // the first overload worked and T has a nested type named type.
        static const bool value = sizeof(test<T>(0)) == sizeof(yes);
    };

    struct none_iterator_tag { };

    template <typename T, typename Enabler = void>
    struct is_iterator : public std::false_type {
      typedef none_iterator_tag iterator_category;
    };

    template <typename T>
    struct is_iterator<T, typename madness::enable_if_c<has_iterator_catagory<T>::value >::type > : public std::true_type {
      typedef typename std::iterator_traits<T>::iterator_category iterator_category;
    };

    template <typename T>
    struct is_iterator<T*, void> : std::true_type {
      typedef typename std::iterator_traits<T*>::iterator_category iterator_category;
    };

    template <typename T>
    struct is_iterator<const T*, void> : std::true_type {
      typedef typename std::iterator_traits<const T*>::iterator_category iterator_category;
    };

    template <typename T>
    struct is_input_iterator :
        public boost::is_base_of<std::input_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_output_iterator :
        public boost::is_base_of<std::output_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_forward_iterator :
        public boost::is_base_of<std::forward_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_bidirectional_iterator :
        public boost::is_base_of<std::bidirectional_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template <typename T>
    struct is_random_iterator :
        public boost::is_base_of<std::random_access_iterator_tag, typename is_iterator<T>::iterator_category>
    { };

    template<typename T>
    struct remove_cr {
      typedef typename std::remove_const<typename std::remove_reference<T>::type>::type type;
    };

    template<typename Func>
    struct unary_functor_types {
      typedef typename Func::argument_type argument_type;
      typedef typename Func::result_type result_type;
      typedef result_type(*func_ptr_type)(argument_type);
    }; // struct unary_functor_types

    template<typename Arg, typename Res>
    struct unary_functor_types<Res(*)(Arg)> {
      typedef Arg argument_type;
      typedef Res result_type;
      typedef result_type(*func_ptr_type)(argument_type);
    }; // struct unary_functor_types<Res(*)(Arg)>

    template<typename Func>
    struct binary_functor_types {
      typedef typename Func::first_argument_type first_argument_type;
      typedef typename Func::second_argument_type second_argument_type;
      typedef typename Func::result_type result_type;
      typedef result_type(*func_ptr_type)(first_argument_type, second_argument_type);
    }; // struct binary_functor_types

  } // namespace detail
} // namespace TiledArray
#endif // TILEDARRAY_TYPE_TRAITS_H__INCLUDED
