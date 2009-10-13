#ifndef TILEDARRAY_UTILITY_H__INCLUDED
#define TILEDARRAY_UTILITY_H__INCLUDED

#include <utility>
#include <functional>

namespace TiledArray {
  namespace detail {

    template<typename P>
    struct pair_first : public std::unary_function<P, typename P::first_type> {
      const typename P::first_type& operator ()(const P& p) const {
        return p.first;
      }
    };

    template<typename P>
    struct pair_second : public std::unary_function<P, typename P::second_type> {
      const typename P::second_type& operator ()(const P& p) const {
        return p.second;
      }
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

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_UTILITY_H__INCLUDED
