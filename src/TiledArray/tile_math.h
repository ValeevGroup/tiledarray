#ifndef TILEDARRAY_TILE_MATH_H__INCLUDED
#define TILEDARRAY_TILE_MATH_H__INCLUDED

#include <TiledArray/variable_list.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/array_ref.h>
#include <TiledArray/config.h>
#include <Eigen/Core>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/functional.hpp>
//#include <boost/tuple/tuple.hpp>
//#include <functional>
#include <numeric>
#ifdef TA_USE_CBLAS
#include <cblas.h>
#endif // TA_USE_CBLAS

namespace TiledArray {

  namespace expressions {
    namespace tile {
      template<typename T>
      class AnnotatedTile;
      template<typename T>
      class ArrayRef;
      template<typename Exp0, typename Exp1, template<typename> class Op >
      struct Expression;
      template<typename Exp, typename Op>
      struct UnaryTileExp;
      template<typename Exp0, typename Exp1>
      typename Expression<Exp0, Exp1, std::plus>::exp_type
      operator +(const Exp0& e0, const Exp1& e1);
      template<typename Exp0, typename Exp1>
      typename Expression<Exp0, Exp1, std::minus>::exp_type
      operator -(const Exp0& e0, const Exp1& e1);
      template<typename Exp0, typename Exp1>
      typename Expression<Exp0, Exp1, std::multiplies>::exp_type
      operator *(const Exp0& e0, const Exp1& e1);
      template<typename Exp>
      UnaryTileExp<Exp, std::negate<typename Exp::value_type> >
      operator -(const Exp& e);
    } // namesapce tile
  } // namespace expressions

  namespace math {

    template<typename T, detail::DimensionOrderType D>
    void contract(const std::size_t, const std::size_t, const std::size_t,
        const std::size_t, const std::size_t, const T*, const T*, T*);

    /// Contract a and b, and place the results into c.
    /// c[m,o,n,p] = a[m,i,n] * b[o,i,p]
    template<typename T, detail::DimensionOrderType D>
    void contract(const std::size_t m, const std::size_t n, const std::size_t o,
        const std::size_t p, const std::size_t i, const T* a, const T* b, T* c)
    {
      typedef Eigen::Matrix< T , Eigen::Dynamic , Eigen::Dynamic,
          (D == detail::decreasing_dimension_order ? Eigen::RowMajor : Eigen::ColMajor) | Eigen::AutoAlign > matrix_type;

      // determine the lower order dimension size
      const std::size_t ma1 = ( D == detail::increasing_dimension_order ? m : n );
      const std::size_t mb1 = ( D == detail::increasing_dimension_order ? o : p );

      // calculate iterator step sizes.
      const std::size_t a_step = i * ma1;
      const std::size_t b_step = i * mb1;
      const std::size_t c_step = ma1 * mb1;

      // calculate iterator boundaries
      const T* a_begin = NULL;
      const T* b_begin = NULL;
      T* c_begin = c;
      const T* const a_end = a + (m * i * n);
      const T* const b_end = b + (o * i * p);
//      const T* const c_end = c + (m * n * o * p);

      // iterate over the highest order dimensions of a and b, and store the
      // results of the matrix-matrix multiplication.
      for(a_begin = a; a_begin != a_end; a_begin += a_step) {
        Eigen::Map<matrix_type> ma(a_begin, i, ma1);
        for(b_begin = b; b_begin != b_end; b_begin += b_step, c_begin += c_step) {
          Eigen::Map<matrix_type> mb(b_begin, i, mb1);
          Eigen::Map<matrix_type> mc(c_begin, ma1, mb1);

          mc = ma.transpose() * mb;
        }
      }
    }

    /// Zip operator adapter.

    /// This adapter is used convert a binary operation to a unary operation that
    /// operates on a two element tuple.
    template<typename T0, typename T1, typename R, typename Op >
    struct ZipOp : public std::unary_function<const boost::tuple<const T0&, const T1&>&, R>
    {
      typedef Op op_type;

      ZipOp() : op_(op_type()) { }
      ZipOp(op_type op) : op_(op) { }

      R operator()(const boost::tuple<const T0&, const T0&>& t) const
      {
        return op_(boost::get<0>(t), boost::get<1>(t));
      }

    private:
      op_type op_;
    }; // struct ZipOp

    /// Tile operation

    /// Performs an element wise binary operation (e.g. std::plus<T>,
    /// std::minus<T>) on two annotated tiles. The value type of the different
    /// tiles may be different, but the value types of expression one and two
    /// must be implicitly convertible to the result value type.
    template<typename Arg1, typename Arg2, typename Res, typename Op>
    struct BinaryTileOp {
      typedef const Arg1& first_argument_type;
      typedef const Arg2& second_argument_type;
      typedef Res result_type;
      typedef typename Res::value_type value_type;
      typedef ZipOp<typename Arg1::value_type,
          typename Arg2::value_type, value_type, Op> op_type;
      typedef boost::transform_iterator<op_type,
          boost::zip_iterator<boost::tuple<typename Arg1::const_iterator,
          typename Arg2::const_iterator> > > const_iterator;

      BinaryTileOp() : op_(Op()) { }
      BinaryTileOp(Op op) : op_(op) { }

      result_type operator ()(first_argument_type e0, second_argument_type e1) {
        result_type result(e0.size(), e0.vars(), begin(e0, e1), end(e0, e1));
        return result;
      }

    private:
      const_iterator begin(first_argument_type e0, second_argument_type e1) {
        return boost::make_transform_iterator(boost::make_zip_iterator(
            boost::make_tuple(e0.begin(), e1.begin())), op_);
      }

      const_iterator end(first_argument_type e0, second_argument_type e1) {
        return boost::make_transform_iterator(boost::make_zip_iterator(
            boost::make_tuple(e0.end(), e1.end())), op_);
      }

      op_type op_;
    }; // struct BinaryTileOp

    /// Tile operation, contraction specialization

    /// This specialization of the tile operation performs a contraction between
    /// two tiles. If more than one index will be contracted, all contracted
    /// indexes must be adjacent.
    template<typename T, typename U, typename Res>
    struct BinaryTileOp<expressions::tile::AnnotatedTile<T>, expressions::tile::AnnotatedTile<U>,
        Res, std::multiplies<typename Res::value_type> >
    {
      typedef expressions::tile::AnnotatedTile<T> first_argument_type;
      typedef expressions::tile::AnnotatedTile<U> second_argument_type;
      typedef Res result_type;
      typedef typename Res::value_type value_type;
      typedef ZipOp< typename first_argument_type::const_iterator,
          typename second_argument_type::const_iterator, value_type, std::multiplies<typename Res::value_type> > op_type;
      typedef boost::transform_iterator<op_type,
          boost::zip_iterator<boost::tuple<typename first_argument_type::const_iterator,
          typename second_argument_type::const_iterator> > > const_iterator;

      BinaryTileOp() { }
      BinaryTileOp(std::multiplies<typename Res::value_type>) { }

      result_type operator ()(const first_argument_type& e0, const second_argument_type& e1) {
        typedef std::pair<expressions::VariableList::const_iterator,
            expressions::VariableList::const_iterator> it_pair;

        // find common variable lists
        std::multiplies<expressions::VariableList> v_op;
        it_pair e0_common;
        it_pair e1_common;
        expressions::VariableList vars = v_op(e0.vars(), e1.vars());
        expressions::find_common(e0.vars().begin(), e0.vars().end(), e1.vars().begin(),
            e1.vars().end(), e0_common, e1_common);

        // find dimensions of the result tile
        std::vector<typename result_type::size_array::value_type> size(vars.dim(), 1);
        typename std::vector<typename result_type::size_array::value_type>::iterator it = size.begin();
        typename std::vector<typename result_type::size_array::value_type>::iterator size_end = size.end();
        expressions::VariableList::const_iterator v_it = vars.begin();
        expressions::VariableList::const_iterator evar0_begin = e0.vars().begin();
        expressions::VariableList::const_iterator evar0_end = e0.vars().end();
        expressions::VariableList::const_iterator evar1_begin = e1.vars().begin();
        expressions::VariableList::const_iterator evar1_end = e1.vars().end();
        expressions::VariableList::const_iterator e_it;
        for(; it != size_end; ++it, ++v_it) {
          if((e_it = std::find(evar0_begin, evar0_end, *v_it)) != e0.vars().end()) {
            *it = e0.size()[std::distance(evar0_begin, e_it)];
          } else {
            e_it = std::find(evar1_begin, evar1_end, *v_it);
            *it = e1.size()[std::distance(evar1_begin, e_it)];
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
        result_type result(size, vars, value_type());
        if(e0.order() == TiledArray::detail::decreasing_dimension_order)
          contract<value_type, TiledArray::detail::decreasing_dimension_order>(m, n, o, p, i, e0.data(), e1.data(), result.data());
        else
          contract<value_type, TiledArray::detail::increasing_dimension_order>(m, n, o, p, i, e0.data(), e1.data(), result.data());

        return result;
      }
    }; // struct BinaryTileOp<AnnotatedTile<T,O>, AnnotatedTile<U,O>, Res, std::multiplies>

    /// Unary tile operation

    /// Performs an element wise unary operation on a tile.
    template<typename Arg, typename Res, typename Op>
    struct UnaryTileOp {
      typedef Arg argument_type;
      typedef Res result_type;
      typedef typename Res::value_type value_type;
      typedef Op op_type;
      typedef boost::transform_iterator<op_type, typename argument_type::const_iterator > const_iterator;

      UnaryTileOp() : op_(op_type()) { }
      UnaryTileOp(op_type op) : op_(op) { }

      result_type operator ()(const argument_type& e) {
        return result_type(e.size(), e.vars(), begin(e), end(e));
      }

    private:
      const_iterator begin(const argument_type& e) {
        return boost::make_transform_iterator(e.begin(), op_);
      }

      const_iterator end(const argument_type& e) {
        return boost::make_transform_iterator(e.end(), op_);
      }

      op_type op_;
    }; // struct UnaryTileOp

  } // namespace math
/*
  namespace expressions {

    namespace tile {

      template<typename Exp0, typename Exp1, typename Op>
      struct BinaryTileExp;
      template<typename Exp, typename Op>
      struct UnaryTileExp;

      /// ValueExp holds a constant value expression.
      template<typename T>
      struct ValueExp {
        typedef typename boost::remove_const<T>::type value_type;

        ValueExp(const value_type& v) : v_(v) { }

        const value_type eval() const { return v_; }
      private:
        ValueExp();
        const value_type v_;
      }; // class ValueExp

      /// Expression Type for constant values

      /// This class is used to ensure that constant values are correctly converted
      /// to ValueExp<T> type, and to make sure the underlying type is maintained.
      template<typename T>
      struct ExpType {
        typedef ValueExp<T> type;
        typedef ValueExp<T> result_type;
        typedef T value_type;
      }; // struct ExpType

      /// Expression Type for constant values

      /// This class is used to ensure that constant values are correctly converted
      /// to ValueExp<T> type, and to make sure the underlying type is maintained.
      template<typename T>
      struct ExpType<ValueExp<T> > {
        typedef ValueExp<T> type;
        typedef ValueExp<T> result_type;
        typedef T value_type;
      }; // struct ExpType<ValueExp<T> >

      /// Expression Type for annotated tiles.

      /// This class is used to determine the type of the tile and the element type.
      template<typename T>
      struct ExpType<AnnotatedTile<T> > {
        typedef AnnotatedTile<T> type;
        typedef AnnotatedTile<T> result_type;
        typedef typename AnnotatedTile<T>::value_type value_type;
      }; // struct ExpType<AnnotatedTile<T,O> >

      /// Expression Type for Binary Expressions.

      /// This class is used to determine the return type for the expression and
      /// the element type.
      template<typename Exp0, typename Exp1, typename Op>
      struct ExpType<BinaryTileExp<Exp0, Exp1, Op> > {
        typedef BinaryTileExp<Exp0, Exp1, Op> type;
        typedef typename BinaryTileExp<Exp0, Exp1, Op>::result_type result_type;
        typedef typename BinaryTileExp<Exp0, Exp1, Op>::value_type value_type;
      }; // struct ExpType<BinaryTileExp<Exp0, Exp1, Op> >

      /// Expression Type for Binary Expressions.

      /// This class is used to determine the return type for the expression and
      /// the element type.
      template<typename Exp, typename Op>
      struct ExpType<UnaryTileExp<Exp, Op> > {
        typedef UnaryTileExp<Exp, Op> type;
        typedef typename UnaryTileExp<Exp, Op>::result_type result_type;
        typedef typename UnaryTileExp<Exp, Op>::value_type value_type;
      }; // struct ExpType<UnaryTileExp<Exp, Op> >

      /// Expression pair

      /// Determines the value and result type of an expression given two
      /// Expressions. The first expression value type is favored.
      template<typename Exp0, typename Exp1>
      struct ExpPair {
        typedef typename ExpType<Exp0>::value_type value_type;
        typedef AnnotatedTile<value_type> result_type;
      }; // ExpPair

      /// Expression pair, constant value first argument specialization

      /// Determines the value and result type of an expression given two
      /// Expressions. The second expression value type is favored.
      template<typename T, typename Exp1>
      struct ExpPair<ValueExp<T>, Exp1> {
        typedef typename ExpType<Exp1>::value_type value_type;
        typedef AnnotatedTile<value_type> result_type;
      }; // struct ExpPair<ValueExp<T>, Exp1>

      /// Expression pair, constant value second argument specialization

      /// Determines the value and result type of an expression given two
      /// Expressions. The first expression value type is favored.
      template<typename Exp0, typename T>
      struct ExpPair<Exp0, ValueExp<T> > {
        typedef typename ExpType<Exp0>::value_type value_type;
        typedef AnnotatedTile<value_type> result_type;
      }; // struct ExpPair<Exp0, ValueExp<T> >

      /// Expression evaluation

      /// This structure contains the methods for evaluating various expression
      /// types.
      struct ExpEval {
        template<typename E0, typename E1, typename EOp >
        static typename BinaryTileExp<E0, E1, EOp>::result_type
        eval(const BinaryTileExp<E0, E1, EOp>& e) { return e.eval(); }

        template<typename E, typename EOp >
        static typename UnaryTileExp<E, EOp>::result_type
        eval(const UnaryTileExp<E, EOp>& e) { return e.eval(); }

        template<typename T>
        static ValueExp<T> eval(const ValueExp<T>& e) { return e.eval(); }

        template<typename T>
        static ValueExp<T> eval(const T& e) { return ValueExp<T>(e); }

        template<typename T>
        static AnnotatedTile<T> eval(const AnnotatedTile<T>& e) { return e; }
      }; // struct ExpEval

      /// Binary Tile Expression

      /// This structure represents a binary tile math expression. The Op type
      /// represents the basic math operation that will be performed on a pair of
      /// elements (one from each tile). The expression types may be annotated
      /// tiles, fundamental types, or other tile expressions. They may be combined
      /// in any combination (except two fundamental types). Both expression types
      /// must have the same storage order. When constructed, the expression stores
      /// a reference to the two expressions it will operate on. The expression
      /// does a lazy evaluation, i.e. it is only evaluated when the eval()
      /// function is explicitly called. If one of the arguments is another
      /// expression, it will be evaluated before this expression.
      template<typename Exp0, typename Exp1, typename Op>
      struct BinaryTileExp {
        typedef typename ExpType<Exp0>::result_type exp0_type;
        typedef typename ExpType<Exp1>::result_type exp1_type;
        typedef typename ExpPair<exp0_type, exp1_type>::result_type result_type;
        typedef typename ExpPair<exp0_type, exp1_type>::value_type value_type;
        typedef math::BinaryTileOp<exp0_type, exp1_type, result_type, Op> op_type;
  //      typedef typename result_type::const_iterator const_iterator;

        BinaryTileExp(const Exp0& e0, const Exp1& e1, Op op = Op()) :
            e0_(e0), e1_(e1), op_(op) { }

        result_type eval() const {
          op_type tile_op(op_);
          return tile_op(ExpEval::eval(e0_), ExpEval::eval(e1_));
        }

      private:

        BinaryTileExp();

        const Exp0& e0_;
        const Exp1& e1_;
        Op op_;
      }; // struct BinaryTileExp

      /// Unary Tile Expression

      /// This structure represents a unary tile math expression. The Op type
      /// represents the basic math operation that will be performed on each
      /// element. The expression types may be annotated tiles, fundamental types,
      /// or other tile expressions. The expression does a lazy evaluation, i.e.
      /// it is only evaluated when the eval() function is explicitly called. If
      /// the argument is another expression, it will be evaluated before this
      /// expression.
      template<typename Exp, typename Op>
      struct UnaryTileExp {
        typedef typename ExpType<Exp>::result_type exp_type;
        typedef exp_type result_type;
        typedef typename exp_type::value_type value_type;
        typedef math::UnaryTileOp<exp_type, result_type, Op> op_type;
  //      typedef typename result_type::const_iterator const_iterator;

        UnaryTileExp(const Exp& e, Op op = Op()) : e_(e), op_(op) { }

        result_type eval() const {
          op_type tile_op(op_);
          return tile_op(ExpEval::eval(e_));
        }

      private:

        UnaryTileExp();

        const Exp& e_;
        Op op_;
      }; // struct UnaryTileExp

      template<typename Exp0, typename Exp1, template<typename> class Op>
      struct ExpConstruct {
        typedef Op<typename ExpPair<Exp0, Exp1>::value_type> op_type;
        typedef BinaryTileExp<Exp0, Exp1, op_type> exp_type;

        static exp_type make_exp(const Exp0& e0, const Exp1& e1) {
          return exp_type(e0, e1, op_type());
        }
      };

      template<typename Exp0, typename T, template<typename> class Op>
      struct ExpConstruct<Exp0, ValueExp<T>, Op> {
        typedef boost::binder2nd< Op<typename Exp0::value_type> > op_type;
        typedef UnaryTileExp<Exp0, op_type> exp_type;

        static exp_type make_exp(const Exp0& e0, const ValueExp<T> e1) {
          return exp_type(e0, op_type(Op<typename Exp0::value_type>(), e1.eval()));
        }
      };

      template<typename T, typename Exp1, template<typename> class Op>
      struct ExpConstruct<ValueExp<T>, Exp1, Op> {
        typedef boost::binder1st< Op<typename Exp1::value_type> > op_type;
        typedef UnaryTileExp<Exp1, op_type> exp_type;

        static exp_type make_exp(const ValueExp<T> e0, const Exp1& e1) {
          return exp_type(e1, op_type(Op<typename Exp1::value_type>(), e0.eval()));
        }
      };

      template<typename Exp0, typename Exp1, template<typename> class Op >
      struct Expression :
          public tile::ExpConstruct<typename ExpType<Exp0>::type, typename ExpType<Exp1>::type, Op> {

      }; // struct ExpPairOp

      /// Tile expression addition operation

      /// This operator constructs a tile binary, addition expression object. The
      /// expression is not immediately evaluated.
      template<typename Exp0, typename Exp1>
      typename tile::Expression<Exp0, Exp1, std::plus>::exp_type
      operator +(const Exp0& e0, const Exp1& e1) {
        return tile::Expression<Exp0, Exp1, std::plus>::make_exp(e0, e1);
      }

      /// Tile expression subtraction operation

      /// This operator constructs a tile binary, subtraction expression object.
      /// The expression is not immediately evaluated.
      template<typename Exp0, typename Exp1>
      typename tile::Expression<Exp0, Exp1, std::minus>::exp_type
      operator -(const Exp0& e0, const Exp1& e1) {
        return tile::Expression<Exp0, Exp1, std::minus>::make_exp(e0, e1);
      }

      /// Tile expression multiplication or contraction operation

      /// This operator constructs a tile binary, multiplication or contraction
      /// expression object. A multiplication expression is constructed when one
      /// of the expressions is a constant value. Otherwise, a contraction
      /// expression is constructed if both expressions will evaluate to annotated
      /// tiles. The expression is not immediately evaluated.
      template<typename Exp0, typename Exp1>
      typename tile::Expression<Exp0, Exp1, std::multiplies>::exp_type
      operator *(const Exp0& e0, const Exp1& e1) {
        return tile::Expression<Exp0, Exp1, std::multiplies>::make_exp(e0, e1);
      }

      /// Tile expression negate operation

      /// This operator constructs a negation expression object. The expression is
      /// not immediately evaluated.
      template<typename Exp>
      UnaryTileExp<Exp, std::negate<typename Exp::value_type> >
      operator -(const Exp& e) {
        return UnaryTileExp<Exp, std::negate<typename Exp::value_type> >(e);
      }

    } // namespace tile

  } // namespace expressions
*/
} // namespace TiledArray

#endif // TILEDARRAY_TILE_MATH_H__INCLUDED
