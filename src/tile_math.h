#ifndef TILEDARRAY_TILE_MATH_H__INCLUDED
#define TILEDARRAY_TILE_MATH_H__INCLUDED

#include <variable_list.h>
#include <coordinate_system.h>
#include <Eigen/core>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
//#include <boost/tuple/tuple.hpp>
//#include <functional>
#include <numeric>
/*
extern "C" {
#include <cblas.h>
};
*/

namespace TiledArray {

  namespace detail {

    /// Contract a and b, and place the results into c.
    /// c[m,o,n,p] = a[m,i,n] * b[o,i,p]
    template<typename T, DimensionOrderType D>
    void contract(const std::size_t m, const std::size_t n, const std::size_t o,
        const std::size_t p, const std::size_t i, const T* a, const T* b, T* c)
    {
      typedef Eigen::Matrix< T , Eigen::Dynamic , Eigen::Dynamic,
          (D == decreasing_dimension_order ? Eigen::RowMajor : Eigen::ColMajor) | Eigen::AutoAlign > matrix_type;

      const std::size_t ma1 = ( D == increasing_dimension_order ? m : n );
      const std::size_t mb1 = ( D == increasing_dimension_order ? o : p );
      const std::size_t a_step = i * ma1;
      const std::size_t b_step = i * mb1;
      const std::size_t c_step = ma1 * mb1;


      const T* a_begin = NULL;
      const T* b_begin = NULL;
      T* c_begin = c;
      const T* const a_end = a + (m * i * n);
      const T* const b_end = b + (o * i * p);
//      const T* const c_end = c + (m * n * o * p);

      for(a_begin = a; a_begin != a_end; a_begin += a_step) {
        Eigen::Map<matrix_type> ma(a_begin, i, ma1);
        for(b_begin = b; b_begin != b_end; b_begin += b_step, c_begin += c_step) {
          Eigen::Map<matrix_type> mb(b_begin, i, mb1);
          Eigen::Map<matrix_type> mc(c_begin, ma1, mb1);

          mc = ma.transpose() * mb;
        }
      }
    }

  } // namespace detail

  namespace math {

    template<typename T, TiledArray::detail::DimensionOrderType O>
    class AnnotatedTile;
    template<typename Exp0, typename Exp1, template<typename> class Op>
    struct BinaryTileExp;

    /// ValueExp holds a constant value expression.
    template<typename T>
    struct ValueExp {
      typedef typename boost::remove_const<T>::type type;

      ValueExp(const type& v) : v_(v) { }

      const type eval() const { return v_; }
    private:
      ValueExp();
      const type v_;
    }; // class ValueExp

    template<typename T>
    struct ExpType {
      typedef ValueExp<T> type;
      typedef T value_type;
    };

    template<typename T, TiledArray::detail::DimensionOrderType O>
    struct ExpType<AnnotatedTile<T,O> > {
      typedef AnnotatedTile<T,O> type;
      typedef typename AnnotatedTile<T,O>::value_type value_type;
    };

    template<typename Exp0, typename Exp1, template<typename> class Op>
    struct ExpType<BinaryTileExp<Exp0, Exp1, Op> > {
      typedef typename BinaryTileExp<Exp0, Exp1, Op>::result_type type;
      typedef typename BinaryTileExp<Exp0, Exp1, Op>::value_type value_type;
    };

    template<typename T0, typename T1, typename R, template<typename> class Op >
    struct ZipOp : public std::unary_function<const boost::tuple<const T0&, const T1&>&, R>
    {
      typedef Op<R> op_type;

      ZipOp() : op_(op_type()) { }
      ZipOp(op_type op) : op_(op) { }

      R operator()(const boost::tuple<const T0&, const T0&>& t) const
      {
        return op_(boost::get<0>(t), boost::get<1>(t));
      }

    private:
      op_type op_;
    };


    template<typename Exp0, typename Exp1, typename Res, template<typename> class Op>
    struct TileOp {
      typedef typename ExpType<Exp0>::type exp0_type;
      typedef typename ExpType<Exp1>::type exp1_type;
      typedef typename ExpType<Res>::type result_type;
      typedef typename ExpType<Res>::value_type value_type;
      typedef ZipOp<typename exp0_type::value_type,
          typename exp1_type::value_type, value_type, Op> op_type;
      typedef boost::transform_iterator<op_type,
          boost::zip_iterator<boost::tuple<typename exp0_type::const_iterator,
          typename exp1_type::const_iterator> > > const_iterator;

      result_type operator ()(const exp0_type& e0, const exp0_type& e1) {
        result_type result(e0.size(), e0.vars(), begin(e0, e1), end(e0, e1));
        return result;
      }

      static const_iterator begin(const exp0_type& e0, const exp0_type& e1) {
        return boost::make_transform_iterator(boost::make_zip_iterator(
            boost::make_tuple(e0.begin(), e1.begin())), op_type());
      }

      static const_iterator end(const exp0_type& e0, const exp0_type& e1) {
        return boost::make_transform_iterator(boost::make_zip_iterator(
            boost::make_tuple(e0.end(), e1.end())), op_type());
      }

    private:


    };

    template<typename T, typename U, TiledArray::detail::DimensionOrderType O, typename Res>
    struct TileOp<AnnotatedTile<T,O>, AnnotatedTile<U,O>, Res, std::multiplies> {
      typedef AnnotatedTile<T,O> exp0_type;
      typedef AnnotatedTile<U,O> exp1_type;
      typedef typename ExpType<Res>::type result_type;
      typedef typename ExpType<Res>::value_type value_type;
      typedef ZipOp< typename exp0_type::const_iterator,
          typename exp1_type::const_iterator, value_type, std::multiplies> op_type;
      typedef boost::transform_iterator<op_type,
          boost::zip_iterator<boost::tuple<typename exp0_type::const_iterator,
          typename exp1_type::const_iterator> > > const_iterator;

      result_type operator ()(const exp0_type& e0, const exp0_type& e1) {
        typedef std::pair<VariableList::const_iterator, VariableList::const_iterator> it_pair;

        // find common variable lists
        std::multiplies<VariableList> v_op;
        it_pair e0_common;
        it_pair e1_common;
        VariableList vars = v_op(e0.vars(), e1.vars());
        find_common(e0.vars().begin(), e0.vars().end(), e1.vars().begin(),
            e1.vars().end(), e0_common, e1_common);

        // find dimensions of the result tile
        typename result_type::size_array size(vars.dim(), 1);
        typename result_type::size_array::iterator it = size.begin();
        VariableList::const_iterator v_it = vars.begin();
        VariableList::const_iterator e_it;
        for(; it != size.end(); ++it, ++v_it) {
          if((e_it = std::find(e0.vars().begin(), e0.vars().end(), *v_it)) != e0.vars().end()) {
            *it = e0.size()[std::distance(e0.vars().begin(), e_it)];
          } else {
            e_it = std::find(e1.vars().begin(), e1.vars().end(), *v_it);
            *it = e1.size()[std::distance(e1.vars().begin(), e_it)];
          }
        }

        // calculate packed tile dimensions
        const std::size_t init = 1;
        const std::size_t m = std::accumulate(e0.size().begin(), e0.size().begin() +
            std::distance(e0.vars().begin(), e0_common.first), init,
            std::multiplies<std::size_t>());
        const std::size_t n = std::accumulate(e0.size().begin() +
            std::distance(e0.vars().begin(), e0_common.second), e0.size().end(),
            init, std::multiplies<std::size_t>());
        const std::size_t o = std::accumulate(e1.size().begin(), e1.size().begin() +
            std::distance(e1.vars().begin(), e1_common.first), init,
            std::multiplies<std::size_t>());
        const std::size_t p = std::accumulate(e1.size().begin() +
            std::distance(e1.vars().begin(), e1_common.second), e1.size().end(),
            init, std::multiplies<std::size_t>());
        const std::size_t i = std::accumulate(e0.size().begin() +
            std::distance(e0.vars().begin(), e0_common.first), e0.size().begin()
            + std::distance(e0.vars().begin(), e0_common.second), init,
            std::multiplies<std::size_t>());

        // construct result tile
        result_type result(size, vars);
        TiledArray::detail::contract<value_type, O>(m, n, o, p, i, e0.data(),
            e1.data(), result.data());

        return result;
      }

    private:

      static const_iterator begin(const exp0_type& e0, const exp0_type& e1) {
        return boost::make_transform_iterator(boost::make_zip_iterator(
            boost::make_tuple(e0.begin(), e1.begin())), op_type());
      }

      static const_iterator end(const exp0_type& e0, const exp0_type& e1) {
        return boost::make_transform_iterator(boost::make_zip_iterator(
            boost::make_tuple(e0.end(), e1.end())), op_type());
      }
    };

    template<typename T, typename Exp1, typename Res, template<typename> class Op>
    struct TileOp<ValueExp<T>, Exp1, Res, Op > {
      typedef ValueExp<T> exp0_type;
      typedef typename ExpType<Exp1>::type exp1_type;
      typedef typename ExpType<Res>::type result_type;
      typedef typename ExpType<Res>::value_type value_type;
      typedef std::binder1st< Op<value_type> > op_type;
      typedef boost::transform_iterator<op_type, typename Exp1::const_iterator > const_iterator;

      result_type operator ()(const exp0_type& e0, const exp1_type& e1) {
        return result_type(e1.size(), e1.vars(), begin(e0, e1), end(e0, e1));
      }

    private:

      static const_iterator begin(const exp0_type& e0, const exp1_type& e1) {
        return boost::make_transform_iterator(e1.begin(), op_type(Op<value_type>(), e0.eval()));
      }

      static const_iterator end(const exp0_type& e0, const exp1_type& e1) {
        return boost::make_transform_iterator(e1.end(), op_type(Op<value_type>(), e0.eval()));
      }
    };

    template<typename Exp0, typename T, typename Res, template<typename> class Op>
    struct TileOp<Exp0, ValueExp<T>, Res, Op > {
      typedef typename ExpType<Exp0>::type exp0_type;
      typedef ValueExp<T> exp1_type;
      typedef typename ExpType<Res>::type result_type;
      typedef typename ExpType<Res>::value_type value_type;
      typedef std::binder2nd< Op<value_type> > op_type;
      typedef boost::transform_iterator<op_type, typename Exp0::const_iterator > const_iterator;

      result_type operator ()(const exp0_type& e0, const exp1_type& e1) {
        return result_type(e0.size(), e0.vars(), begin(e0, e1), end(e0, e1));
      }

    private:

      static const_iterator begin(const exp0_type& e0, const exp1_type& e1) {
        return boost::make_transform_iterator(e0.begin(), op_type(Op<value_type>(), e1.eval()));
      }

      static const_iterator end(const exp0_type& e0, const exp1_type& e1) {
        return boost::make_transform_iterator(e0.end(), op_type(Op<value_type>(), e1.eval()));
      }
    };

    template<typename Exp0, typename Exp1>
    struct ExpPair {
      typedef typename ExpType<Exp0>::value_type value_type;
      typedef AnnotatedTile<value_type, ExpType<Exp0>::type::order > result_type;
    };

    template<typename T, typename Exp1>
    struct ExpPair<ValueExp<T>, Exp1> {
      typedef typename ExpType<Exp1>::value_type value_type;
      typedef AnnotatedTile<value_type, ExpType<Exp1>::type::order > result_type;
    };

    template<typename Exp0, typename T>
    struct ExpPair<Exp0, ValueExp<T> > {
      typedef typename ExpType<Exp0>::value_type value_type;
      typedef AnnotatedTile<value_type, ExpType<Exp0>::type::order > result_type;
    };

    template<typename Exp0, typename Exp1, template<typename> class Op>
    struct BinaryTileExp {
      typedef typename ExpType<Exp0>::type exp0_type;
      typedef typename ExpType<Exp1>::type exp1_type;
      typedef typename ExpPair<exp0_type, exp1_type>::result_type result_type;
      typedef typename ExpPair<exp0_type, exp1_type>::value_type value_type;
      typedef TileOp<exp0_type, exp1_type, result_type, Op> op_type;
      typedef typename result_type::const_iterator const_iterator;

      static const TiledArray::detail::DimensionOrderType order;

      BinaryTileExp(const Exp0& e0, const Exp1& e1) : e0_(e0), e1_(e1) { }

      result_type eval() const {
        op_type op;
        return op(eval_(e0_), eval_(e1_));
      }

    private:

      BinaryTileExp();

      template<typename E0, typename E1, template<typename> class EOp >
      static typename BinaryTileExp<E0, E1, EOp>::result_type
      eval_(const BinaryTileExp<E0, E1, EOp>& e) { return e.eval(); }

      template<typename T>
      static ValueExp<T> eval_(const ValueExp<T>& e) { return e.eval(); }

      template<typename T>
      static ValueExp<T> eval_(const T& e) { return ValueExp<T>(e); }

      template<typename T, TiledArray::detail::DimensionOrderType O>
      static AnnotatedTile<T, O> eval_(const AnnotatedTile<T, O>& e) { return e; }

      const Exp0& e0_;
      const Exp1& e1_;
    };

    template<typename Exp0, typename Exp1, template<typename> class Op >
    const TiledArray::detail::DimensionOrderType BinaryTileExp<Exp0, Exp1, Op>::order =
        BinaryTileExp<Exp0, Exp1, Op>::result_type::order;


    template<typename Exp0, typename Exp1>
    BinaryTileExp<Exp0, Exp1, std::plus> operator +(const Exp0& e0, const Exp1& e1) {
      return BinaryTileExp<Exp0, Exp1, std::plus>(e0, e1);
    }

    template<typename Exp0, typename Exp1>
    BinaryTileExp<Exp0, Exp1, std::minus> operator -(const Exp0& e0, const Exp1& e1) {
      return BinaryTileExp<Exp0, Exp1, std::minus>(e0, e1);
    }

    template<typename Exp0, typename Exp1>
    BinaryTileExp<Exp0, Exp1, std::multiplies> operator *(const Exp0& e0, const Exp1& e1) {
      return BinaryTileExp<Exp0, Exp1, std::multiplies>(e0, e1);
    }

    template<typename Exp0>
    BinaryTileExp<Exp0, typename Exp0::value_type, std::multiplies> operator -(const Exp0& e0) {
      const typename Exp0::value_type neg = -1;
      return BinaryTileExp<Exp0, typename Exp0::value_type, std::multiplies>(e0, neg);
    }

  } // namespace math

} // namespace TiledArray

#endif // TILEDARRAY_TILE_MATH_H__INCLUDED
