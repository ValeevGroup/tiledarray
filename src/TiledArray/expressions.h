#ifndef TILEDARRAY_EXPRESSIONS_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_H__INCLUDED

#include <TiledArray/annotated_tile.h>
#include <TiledArray/annotated_array.h>
#include <TiledArray/tile_math.h>
#include <TiledArray/array_math.h>
#include <boost/mpl/if.hpp>
#include <boost/type_traits.hpp>

// The expression templates in this file are taken from Boost uBLAS.

namespace TiledArray {
  namespace expressions {

    template<typename T>
    struct ValueExp;
    template<typename Exp>
    struct ExpType;
    template<typename Exp, template<typename> class Op>
    struct UnaryExp;
    template<typename Exp1, typename Exp2, template<typename> class Op>
    struct BinaryExp;
    template<typename Exp, template<typename> class Op>
    struct UnaryTileExp;
    template<typename Exp1, typename Exp2, template<typename> class Op>
    struct BinaryTileExp;
    template<typename Exp, template<typename> class Op>
    struct UnaryArrayExp;
    template<typename Exp1, typename Exp2, template<typename> class Op>
    struct BinaryArrayExp;


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
    struct ExpType<tile::AnnotatedTile<T> > {
      typedef tile::AnnotatedTile<T> type;
      typedef tile::AnnotatedTile<T> result_type;
      typedef typename tile::AnnotatedTile<T>::value_type value_type;
    }; // struct ExpType<AnnotatedTile<T,O> >

    /// Expression Type for Binary Expressions.

    /// This class is used to determine the return type for the expression and
    /// the element type.
    template<typename Exp0, typename Exp1, template<typename> class Op>
    struct ExpType<BinaryTileExp<Exp0, Exp1, Op> > {
      typedef BinaryTileExp<Exp0, Exp1, Op> type;
      typedef typename BinaryTileExp<Exp0, Exp1, Op>::result_type result_type;
      typedef typename BinaryTileExp<Exp0, Exp1, Op>::value_type value_type;
    }; // struct ExpType<BinaryTileExp<Exp0, Exp1, Op> >

    /// Expression Type for Binary Expressions.

    /// This class is used to determine the return type for the expression and
    /// the element type.
    template<typename Exp, template<typename> class Op>
    struct ExpType<UnaryTileExp<Exp, Op> > {
      typedef UnaryTileExp<Exp, Op> type;
      typedef typename UnaryTileExp<Exp, Op>::result_type result_type;
      typedef typename UnaryTileExp<Exp, Op>::value_type value_type;
    }; // struct ExpType<UnaryTileExp<Exp, Op> >

    /// Expression Type for annotated tiles.

    /// This class is used to determine the type of the tile and the element type.
    template<typename T>
    struct ExpType<array::AnnotatedArray<T> > {
      typedef array::AnnotatedArray<T> type;
      typedef array::AnnotatedArray<T> result_type;
      typedef typename array::AnnotatedArray<T>::value_type value_type;

    }; // struct ExpType<AnnotatedArray<T,O> >

    /// Expression Type for Binary Expressions.

    /// This class is used to determine the return type for the expression and
    /// the element type.
    template<typename Exp0, typename Exp1, template<typename> class Op>
    struct ExpType<BinaryArrayExp<Exp0, Exp1, Op> > {
      typedef BinaryArrayExp<Exp0, Exp1, Op> type;
      typedef typename BinaryArrayExp<Exp0, Exp1, Op>::result_type result_type;
      typedef typename BinaryArrayExp<Exp0, Exp1, Op>::value_type value_type;
    }; // struct ExpType<BinaryArrayExp<Exp0, Exp1, Op> >

    /// Expression Type for Binary Expressions.

    /// This class is used to determine the return type for the expression and
    /// the element type.
    template<typename Exp, template<typename> class Op>
    struct ExpType<UnaryArrayExp<Exp, Op> > {
      typedef UnaryArrayExp<Exp, Op> type;
      typedef typename UnaryArrayExp<Exp, Op>::result_type result_type;
      typedef typename UnaryArrayExp<Exp, Op>::value_type value_type;
    }; // struct ExpType<UnaryArrayExp<Exp, Op> >

    /// Expression pair

    /// Determines the value and result type of an expression given two
    /// Expressions. The first expression value type is favored.
    template<typename Exp0, typename Exp1>
    struct ExpPair {
      typedef typename ExpType<Exp0>::value_type value_type;
      typedef tile::AnnotatedTile<value_type> result_type;
    }; // ExpPair

    /// Expression pair, constant value first argument specialization

    /// Determines the value and result type of an expression given two
    /// Expressions. The second expression value type is favored.
    template<typename T, typename Exp1>
    struct ExpPair<ValueExp<T>, Exp1> {
      typedef typename ExpType<Exp1>::value_type value_type;
      typedef tile::AnnotatedTile<value_type> result_type;
    }; // struct ExpPair<ValueExp<T>, Exp1>

    /// Expression pair, constant value second argument specialization

    /// Determines the value and result type of an expression given two
    /// Expressions. The first expression value type is favored.
    template<typename Exp0, typename T>
    struct ExpPair<Exp0, ValueExp<T> > {
      typedef typename ExpType<Exp0>::value_type value_type;
      typedef tile::AnnotatedTile<value_type> result_type;
    }; // struct ExpPair<Exp0, ValueExp<T> >

    /// Expression evaluation

    /// This structure contains the methods for evaluating various expression
    /// types.
    struct ExpEval {
      template<typename E0, typename E1, template<typename> class EOp >
      static typename BinaryTileExp<E0, E1, EOp>::result_type
      eval(const BinaryTileExp<E0, E1, EOp>& e) { return e.eval(); }

      template<typename E, template<typename> class EOp >
      static typename UnaryTileExp<E, EOp>::result_type
      eval(const UnaryTileExp<E, EOp>& e) { return e.eval(); }

      template<typename T>
      static ValueExp<T> eval(const ValueExp<T>& e) { return e.eval(); }

      template<typename T>
      static ValueExp<T> eval(const T& e) { return ValueExp<T>(e); }

      template<typename T>
      static tile::AnnotatedTile<T> eval(const tile::AnnotatedTile<T>& e) { return e; }
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
    template<typename Exp0, typename Exp1, template<typename> class Op>
    struct BinaryTileExp {
      typedef typename ExpType<Exp0>::result_type exp0_type;
      typedef typename ExpType<Exp1>::result_type exp1_type;
      typedef typename ExpPair<exp0_type, exp1_type>::result_type result_type;
      typedef typename ExpPair<exp0_type, exp1_type>::value_type value_type;
      typedef math::BinaryTileOp<exp0_type, exp1_type, result_type, Op > op_type;
//      typedef typename result_type::const_iterator const_iterator;

      BinaryTileExp(const Exp0& e0, const Exp1& e1, Op<value_type> op = Op<value_type>()) :
          e0_(e0), e1_(e1), op_(op) { }

      result_type eval() const {
        op_type tile_op(op_);
        return tile_op(ExpEval::eval(e0_), ExpEval::eval(e1_));
      }

    private:

      BinaryTileExp();

      const Exp0& e0_;
      const Exp1& e1_;
      Op<value_type> op_;
    }; // struct BinaryTileExp

    /// Unary Tile Expression

    /// This structure represents a unary tile math expression. The Op type
    /// represents the basic math operation that will be performed on each
    /// element. The expression types may be annotated tiles, fundamental types,
    /// or other tile expressions. The expression does a lazy evaluation, i.e.
    /// it is only evaluated when the eval() function is explicitly called. If
    /// the argument is another expression, it will be evaluated before this
    /// expression.
    template<typename Exp, template<typename> class Op>
    struct UnaryTileExp {
      typedef typename ExpType<Exp>::result_type exp_type;
      typedef exp_type result_type;
      typedef typename exp_type::value_type value_type;
      typedef math::UnaryTileOp<exp_type, result_type, Op> op_type;
//      typedef typename result_type::const_iterator const_iterator;

      UnaryTileExp(const Exp& e, Op<value_type> op = Op<value_type>()) : e_(e), op_(op) { }

      result_type eval() const {
        op_type tile_op(op_);
        return tile_op(ExpEval::eval(e_));
      }

    private:

      UnaryTileExp();

      const Exp& e_;
      Op<value_type> op_;
    }; // struct UnaryTileExp

    template<typename Exp0, typename Exp1, template<typename> class Op>
    struct ExpConstruct {
      typedef Op<typename ExpPair<Exp0, Exp1>::value_type> op_type;
      typedef BinaryTileExp<Exp0, Exp1, Op> exp_type;

      static exp_type make_exp(const Exp0& e0, const Exp1& e1) {
        return exp_type(e0, e1, op_type());
      }
    };

    template<typename Exp0, typename T, template<typename> class Op>
    struct ExpConstruct<Exp0, ValueExp<T>, Op> {
      typedef boost::binder2nd< Op<typename Exp0::value_type> > op_type;
      typedef UnaryTileExp<Exp0, Op> exp_type;

      static exp_type make_exp(const Exp0& e0, const ValueExp<T> e1) {
        return exp_type(e0, op_type(Op<typename Exp0::value_type>(), e1.eval()));
      }
    };

    template<typename T, typename Exp1, template<typename> class Op>
    struct ExpConstruct<ValueExp<T>, Exp1, Op> {
      typedef boost::binder1st< Op<typename Exp1::value_type> > op_type;
      typedef UnaryTileExp<Exp1, Op> exp_type;

      static exp_type make_exp(const ValueExp<T> e0, const Exp1& e1) {
        return exp_type(e1, op_type(Op<typename Exp1::value_type>(), e0.eval()));
      }
    };



    template<typename Exp>
    struct Arguments {
      typedef typename Exp::result_type argument_type;
      typedef typename Exp::result_type result_type;
    };

    template<typename Exp1, typename Exp2>
    struct Arguments<ExpPair<Exp1, Exp2> > {
      typedef typename ExpPair<Exp1, Exp2>::first_argument_type first_argument_type;
      typedef typename ExpPair<Exp1, Exp2>::second_argument_type second_argument_type;
      typedef typename ExpPair<Exp1, Exp2>::result_type result_type;
    };

    template<typename Exp, template<typename> class Op >
    struct Construct : public Arguments<Exp> {
      void make(const Exp&) {

      }
    };

    template<typename Exp1, typename Exp2, template<typename> class Op >
    struct Construct<ExpPair<Exp1, Exp2>, Op> {

    };

    struct Function {

    };

    template<typename Exp, template<typename> class Op >
    struct Expression : public Construct<Exp, Op>

    {

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
    UnaryTileExp<Exp, std::negate >
    operator -(const Exp& e) {
      return UnaryTileExp<Exp, std::negate>(e);
    }


  }  // namespace expressions

}  // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_H__INCLUDED
