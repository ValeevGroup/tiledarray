#ifndef TILEDARRAY_UTILITY_H__INCLUDED
#define TILEDARRAY_UTILITY_H__INCLUDED

#include <TiledArray/type_traits.h>
#include <utility>
#include <functional>
#include <boost/type_traits/integral_constant.hpp>

namespace TiledArray {

  template <typename I, unsigned int DIM, typename Tag>
  class ArrayCoordinate;

  namespace detail {

    // Forward declarations
    template<typename Key1, typename Key2>
    class Key;

    // Local forward declarations
    template<typename Op, typename F0, typename F1>
    struct binary_transform;
    template<typename Op, typename F>
    struct unary_transform;
    template<typename Op, typename F0, typename F1>
    binary_transform<Op, F0, F1> make_binary_transform(Op op, F0 f0, F1 f1);
    template<typename Op, typename F>
    unary_transform<Op, F> make_unary_transform(Op op, F f);
    template <typename Value, typename OutIter>
    void initialize_from_values(Value val, OutIter result, unsigned int, boost::true_type);
    template <typename InIter, typename OutIter>
    void initialize_from_values(InIter, OutIter, unsigned int, boost::false_type);
    template<typename I, unsigned int DIM, typename Tag, typename OutIter>
    void initialize_from_values(Key<I, ArrayCoordinate<I, DIM, Tag> > k, OutIter result, unsigned int, boost::false_type);
    template<typename I, unsigned int DIM, typename Tag, typename OutIter>
    void initialize_from_values(Key<ArrayCoordinate<I, DIM, Tag>, I > k, OutIter result, unsigned int, boost::false_type);
    template<typename I, unsigned int DIM, typename Tag, typename OutIter>
    void initialize_from_values(Key<ArrayCoordinate<I, DIM, Tag>, I >, OutIter, unsigned int, boost::false_type);

    template<typename P>
    struct pair_first : public std::unary_function<P, typename P::first_type> {
      typename P::first_type& operator ()(P& p) const { return p.first; }
      typename P::first_type& operator ()(const P& p) const { return p.first; }

    };

    template<typename P>
    struct pair_second : public std::unary_function<P, typename P::second_type> {
      typename P::second_type& operator ()(P& p) const { return p.second; }
      typename P::second_type& operator ()(const P& p) const { return p.second; }
    };

    template<typename T>
    struct null_func : public std::unary_function<T,T> {
      const T& operator()(const T& t) const { return t; }
    };

    template<typename Op, typename F = null_func<typename Op::argument_type> >
    struct unary_transform : public std::unary_function<typename F::argument_type,
        typename Op::result_type> {
      unary_transform() : f_(F()), op_(Op()) { }
      unary_transform(Op op = Op(), F f = F()) : f_(f), op_(op) { }

      typename Op::result_type operator()(const typename F::argument_type& a) {
        return op_(f_(a));
      }

    private:
      F f_;
      Op op_;
    };

    template<typename Op, typename F0 = null_func<typename Op::first_argument_type>,
        typename F1 = null_func<typename Op::second_argument_type> >
    struct binary_transform : public std::binary_function<typename F0::argument_type,
        typename F1::argument_type, typename Op::result_type> {
      binary_transform() : f0_(F0()), f1_(F1()), op_(Op()) { }
      binary_transform(F0 f0 = F0(), F1 f1 = F1(), Op op = Op()) : f0_(f0), f1_(f1), op_(op) { }

      typename Op::result_type operator()(const typename F0::argument_type& a0,
          const typename F1::argument_type a1) {
        return op_(f0_(a0), f1_(a1));
      }

    private:
      F0 f0_;
      F1 f1_;
      Op op_;
    };

    template<typename Op, typename F0, typename F1>
    binary_transform<Op, F0, F1> make_binary_transform(Op op, F0 f0, F1 f1) {
      return binary_transform<Op, F0, F1>(op, f0, f1);
    }

    template<typename Op, typename F>
    unary_transform<Op, F> make_unary_transform(Op op, F f) {
      return unary_transform<Op, F>(op, f);
    }

    // help with initialization of classes with constructors that need disambiguation
    template <typename Value, typename OutIter>
    void initialize_from_values(Value val, OutIter result, unsigned int, boost::true_type) {
      BOOST_STATIC_ASSERT(detail::is_output_iterator<OutIter>::value);
      *result = val;
    }
    template <typename InIter, typename OutIter>
    void initialize_from_values(InIter first, OutIter result, unsigned int size, boost::false_type) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      BOOST_STATIC_ASSERT(detail::is_output_iterator<OutIter>::value);
      for(unsigned int i = 0; i < size; ++i, ++result, ++first)
        *result = *first;
    }

    template<typename I, unsigned int DIM, typename Tag, typename OutIter>
    void initialize_from_values(Key<I, ArrayCoordinate<I, DIM, Tag> > k,
        OutIter result, unsigned int, boost::false_type)
    {
      std::copy(k.key2().begin(), k.key2().end(), result);
    }

    template<typename I, unsigned int DIM, typename Tag, typename OutIter>
    void initialize_from_values(Key<ArrayCoordinate<I, DIM, Tag>, I > k,
        OutIter result, unsigned int, boost::false_type)
    {
      std::copy(k.key1().begin(), k.key1().end(), result);
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    template<typename OutIter, typename V>
    void fill(OutIter it, V v) {
      BOOST_STATIC_ASSERT(detail::is_output_iterator<OutIter>::value);
      *it = v;
    }

    template <typename OutIter, typename V, typename... Params>
    void fill(OutIter it, V v, Params... params) {
      BOOST_STATIC_ASSERT(detail::is_output_iterator<OutIter>::value);
      *it = v;
      fill(++it, params...);
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

#if 0
    template<unsigned int DIM, typename V, typename... Params>
    struct FillCoord {
    private:
      BOOST_STATIC_ASSERT((Count<V, Params...>::value) == DIM);
    public:
      template<typename OutIter>
      void operator()(OutIter it, V v, Params... p) {
        BOOST_STATIC_ASSERT(detail::is_output_iterator<OutIter>::value);
        *it = v;
        FillCoord<DIM - 1, Params...> f; // this is not implemented yet
        f(++it, p...);
      }
    };

    template<typename V>
    struct FillCoord<1u, V> {
      template<typename OutIter>
      void operator()(OutIter it, V v) {
        BOOST_STATIC_ASSERT(detail::is_output_iterator<OutIter>::value);
        *it = v;
      }
    };
#endif

    template<typename T>
    struct bit_and : public std::binary_function<T, T, T> {
      T operator()(const T& t1, const T& t2) const {
        return t1 & t2;
      }
    }; // struct bit_and

    template<typename T>
    struct bit_or : public std::binary_function<T, T, T> {
      T operator()(const T& t1, const T& t2) const {
        return t1 | t2;
      }
    }; // struct bit_or

    template<typename T>
    struct bit_xor : public std::binary_function<T, T, T> {
      T operator()(const T& t1, const T& t2) const {
        return t1 ^ t2;
      }
    }; // struct bit_xor

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_UTILITY_H__INCLUDED
