#ifndef TILEDARRAY_ARRAY_MATH_H__INCLUDED
#define TILEDARRAY_ARRAY_MATH_H__INCLUDED

#include <TiledArray/variable_list.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/madness_runtime.h>
#include <Eigen/Core>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/functional.hpp>
#include <boost/mpl/if.hpp>
#include <numeric>


namespace TiledArray {

  namespace math {

    template<typename Arg1, typename Arg2, typename Res, typename Op>
    struct BinaryTileOp;
    template<typename Arg, typename Res, typename Op>
    struct UnaryTileOp;

    /// Binary task operation

    /// This functor is used to convert a binary functor or function into a task
    /// based function (i.e. the functor will spawn a task to do the work
    /// instead of executing the functor or function directly). The new functor
    /// will accept madness futures as its arguments and return a future to the
    /// result. The future template parameters for each argument and the result
    /// correspond to the argument and result type of the original functor or
    /// function. Example:
    /// \code
    /// BinaryTaskOp<std::plus<int> > op;
    /// madness::Future<int> a;
    /// madness::Future<int> b;
    /// madness::Future<int> c;
    /// c = op(a, b);
    /// \endcode
    /// This will spawn a task that sums the values of the futures \c a and \c b
    /// and places the result in the future c. The Op template argument may be
    /// ether a function object or a function pointer.
    /// \var \c Op is the functor or function pointer type, and the operation that will be performed by the task.
    template<typename Op>
    struct BinaryTaskOp {
    private:
      BinaryTaskOp();
      typedef typename detail::remove_cr<typename detail::binary_functor_types<Op>::first_argument_type>::type first_value_type;
      typedef typename detail::remove_cr<typename detail::binary_functor_types<Op>::second_argument_type>::type second_value_type;
      typedef typename detail::remove_cr<typename detail::binary_functor_types<Op>::result_type>::type result_value_type;
    public:
      typedef const madness::Future<first_value_type>& first_argument_type;
      typedef const madness::Future<second_value_type>& second_argument_type;
      typedef madness::Future<result_value_type> result_type;

      /// Primary constructor

      /// Constructs a binary task object. If no functor is provided, the default
      /// constructor will be used to create it.
      /// \var \c w is the world object used to spawn tasks.
      /// \var \c o is the functor object which performs the task work (optional).
      BinaryTaskOp(madness::World& w, Op o = Op(), madness::TaskAttributes a = madness::TaskAttributes()) :
          world_(w), attr_(a), op_(o)
      { }

      /// Set the task attributes to the the given attributes.

      /// Change the task attributes to a new value. If no attribute flag is
      /// Provided, the attributes will be set to the default value.
      /// \var \c a is the new attribute to be used when generating tasks (optional).
      void reset(madness::TaskAttributes a = madness::TaskAttributes()) { attr_ = a; }

      /// Returns a reference to the world object where tasks will be spawned.
      madness::World& get_world() const { return world_; }

      /// Creates a task.

      /// This will generate a task on the local task queue
      result_type operator() (first_argument_type fut1, second_argument_type fut2) {
        return world_.taskq.add(op_, & Op::operator(), fut1, fut2, attr_);
      }

    private:
      madness::World& world_;         ///< Reference to the world object used to
                                      ///< create tasks.
      madness::TaskAttributes attr_;  ///< Task attribute object.
      Op op_;                         ///< Functor which does the task work.
    }; // struct BinaryTaskOp


    /// Unary task operation

    /// This functor is used to convert a unary functor or function into a task
    /// based function (i.e. the functor will spawn a task to do the work
    /// instead of executing the functor or function directly). The new functor
    /// will accept a madness futures as its argument and return a future to the
    /// result. The future template parameters for the argument and the result
    /// correspond to the argument and result type of the original functor or
    /// function. Example:
    /// \code
    /// UnaryTaskOp<int, int, std::negate<int> > op;
    /// madness::Future<int> a;
    /// madness::Future<int> b;
    /// b = op(a);
    /// \endcode
    /// This will spawn a task that sums the values of the futures \c a and \c b
    /// and places the result in the future c.
    /// \var \c Op is the functor or function pointer type, and the operation that will be performed by the task.
    template<typename Op>
    struct UnaryTaskOp {
    private:
      UnaryTaskOp();
    public:
      typedef madness::Future<typename detail::unary_functor_types<Op>::argument_type>
          argument_type;
      typedef madness::Future<typename detail::unary_functor_types<Op>::result_type>
          result_type;

      /// Primary constructor

      /// Constructs a binary task object. If no functor is provided, the default
      /// constructor will be used to create it.
      /// \var \c w is the world object used to spawn tasks.
      /// \var \c o is the functor object which performs the task work (optional).
      UnaryTaskOp(madness::World& w, Op o = Op(),
          madness::TaskAttributes a = madness::TaskAttributes()) :
          world_(w), attr_(a), op_(o)
      { }

      /// Set the task attributes to the the given attributes.

      /// Change the task attributes to a new value. If no attribute flag is
      /// Provided, the attributes will be set to the default value.
      /// \var \c a is the new attribute to be used when generating tasks (optional).
      void reset(madness::TaskAttributes a = madness::TaskAttributes()) { attr_ = a; }

      /// Returns a reference to the world object where tasks will be spawned.
      madness::World& get_world() const { return world_; }

      /// Creates a task.

      /// This will generate a task on the local task queue
      result_type operator() (const argument_type& fut) const {
        return world_.taskq.add(op_, fut, attr_);
      }

    private:
      madness::World& world_;         ///< Reference to the world object used to
                                      ///< create tasks.
      madness::TaskAttributes attr_;  ///< Task attribute object.
      Op op_;                         ///< Functor which does the task work.
    }; // struct UnaryTaskOp

    /// Array operation

    /// Performs an element wise binary operation (e.g. std::plus<T>,
    /// std::minus<T>) on two annotated tiles. The value type of the different
    /// tiles may be different, but the value types of expression one and two
    /// must be implicitly convertible to the result value type.
    template<typename Arg1, typename Arg2, typename Res, typename Op>
    struct BinaryArrayOp {
    private:
      BinaryArrayOp();

    public:
      typedef const Arg1& first_argument_type;  ///< first array argument type.
      typedef const Arg2& second_argument_type; ///< second array argument type.
      typedef Res result_type;            ///< result array type.

    private:
      /// Binary tile-task operation type.
      typedef BinaryTaskOp<BinaryTileOp<typename Arg1::tile_type,
          typename Arg2::tile_type, typename result_type::tile_type, Op> > op_type;

    public:
      /// operation constructor
      /// \arg \c w is a reference to the world object where tasks will be spawned.
      /// \arg \c o is the functor or function that will be used in tile operations.
      /// \arg \c a is the task attributes that will be used when spawning tile, task operations.
      BinaryArrayOp(madness::World& w, Op o = Op(), madness::TaskAttributes a = madness::TaskAttributes()) :
          op_(w, o, a) { }

      /// Constructs a series of tasks for the given arrays.
      result_type operator ()(first_argument_type a1, second_argument_type a2) {
        // Here we assume that the array tiles have compatible sizes because it
        // is checked in the expression generation functions (if error checking
        // is enabled.
        result_type result(a1.get_world(), a1.range(), a1.vars(), a1.order());
        for(typename Arg1::const_iterator it = a1.begin(); it != a1.end(); ++it) {
          const typename Arg1::index_type i = it->first;
          madness::Future<typename Arg1::tile_type> t1 = it->second;
          madness::Future<typename Arg2::tile_type> t2 = a2.find(i)->second;
          madness::Future<typename Res::tile_type> tr = op_(t1, t2);

          result.insert(i, tr);
        }
        return result;
      }

    private:
      op_type op_; ///< Binary task operation object.
    }; // struct BinaryArrayOp

    // Todo: Add BinaryArrayOp specialization for contractions.

    /// Unary tile operation

    /// Performs an element wise unary operation on a tile.
    template<typename Arg, typename Res, typename Op>
    struct UnaryArrayOp {
      typedef Arg& argument_type;
      typedef Res result_type;
      typedef UnaryTaskOp<UnaryTileOp<typename Arg::tile_type,
          typename result_type::tile_type, Op> > op_type;

    private:
      UnaryArrayOp();

    public:
      /// operation constructor
      /// \arg \c w is a reference to the world object where tasks will be spawned.
      /// \arg \c o is the functor or function that will be used in tile operations.
      /// \arg \c a is the task attributes that will be used when spawning tile, task operations.
      UnaryArrayOp(madness::World& w, Op o = Op(), madness::TaskAttributes a = madness::TaskAttributes()) :
          op_(w, o, a) { }

      /// Constructs a series of tasks for the given arrays.
      result_type operator ()(argument_type a) const {
        typedef typename boost::mpl::if_<boost::is_const<Arg>,
            typename Arg::const_iterator, typename Arg::iterator>::type iterator_type;

        result_type result = a.clone(op_.get_world(), false);
        for(iterator_type it = a.begin(); it == a.end(); ++it)
          result.insert(it->first, op_(it->second));
        return result;
      }

    private:
      op_type op_; ///< Binary task operation object.
    }; // struct UnaryArrayOp

  } // namespace math
/*
  namespace expressions {

    namespace array {

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
      struct ExpType<AnnotatedArray<T> > {
        typedef AnnotatedArray<T> type;
        typedef AnnotatedArray<T> result_type;
        typedef typename AnnotatedArray<T>::value_type value_type;

      }; // struct ExpType<AnnotatedArray<T,O> >

      /// Expression Type for Binary Expressions.

      /// This class is used to determine the return type for the expression and
      /// the element type.
      template<typename Exp0, typename Exp1, typename Op>
      struct ExpType<BinaryArrayExp<Exp0, Exp1, Op> > {
        typedef BinaryArrayExp<Exp0, Exp1, Op> type;
        typedef typename BinaryArrayExp<Exp0, Exp1, Op>::result_type result_type;
        typedef typename BinaryArrayExp<Exp0, Exp1, Op>::value_type value_type;
      }; // struct ExpType<BinaryArrayExp<Exp0, Exp1, Op> >

      /// Expression Type for Binary Expressions.

      /// This class is used to determine the return type for the expression and
      /// the element type.
      template<typename Exp, typename Op>
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
        typedef AnnotatedArray<value_type> result_type;
        typedef typename result_type::tile_type tile_type;
      }; // ExpPair

      /// Expression pair, constant value first argument specialization

      /// Determines the value and result type of an expression given two
      /// Expressions. The second expression value type is favored.
      template<typename T, typename Exp1>
      struct ExpPair<ValueExp<T>, Exp1> {
        typedef typename ExpType<Exp1>::value_type value_type;
        typedef AnnotatedArray<value_type> result_type;
        typedef typename result_type::tile_type tile_type;
      }; // struct ExpPair<ValueExp<T>, Exp1>

      /// Expression pair, constant value second argument specialization

      /// Determines the value and result type of an expression given two
      /// Expressions. The first expression value type is favored.
      template<typename Exp0, typename T>
      struct ExpPair<Exp0, ValueExp<T> > {
        typedef typename ExpType<Exp0>::value_type value_type;
        typedef AnnotatedArray<value_type> result_type;
        typedef typename result_type::tile_type tile_type;
      }; // struct ExpPair<Exp0, ValueExp<T> >

      /// Expression evaluation

      /// This structure contains the methods for evaluating various expression
      /// types.
      struct ExpEval {
        template<typename E0, typename E1, typename EOp >
        static typename BinaryArrayExp<E0, E1, EOp>::result_type
        eval(const BinaryArrayExp<E0, E1, EOp>& e) { return e.eval(); }

        template<typename E, typename EOp >
        static typename UnaryArrayExp<E, EOp>::result_type
        eval(const UnaryArrayExp<E, EOp>& e) { return e.eval(); }

        template<typename T>
        static ValueExp<T> eval(const ValueExp<T>& e) { return e.eval(); }

        template<typename T>
        static ValueExp<T> eval(const T& e) { return ValueExp<T>(e); }

        template<typename T>
        static AnnotatedArray<T> eval(const AnnotatedArray<T>& e) { return e; }
      }; // struct ExpEval

      /// Binary Array Expression

      /// This structure represents a binary array math expression. The Op type
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
      struct BinaryArrayExp {
        typedef typename ExpType<Exp0>::result_type exp0_type;
        typedef typename ExpType<Exp1>::result_type exp1_type;
        typedef typename ExpPair<exp0_type, exp1_type>::result_type result_type;
        typedef typename ExpPair<exp0_type, exp1_type>::value_type value_type;
        typedef typename ExpPair<exp0_type, exp1_type>::tile_type tile_type;
        typedef Op op_type;
  //      typedef typename result_type::const_iterator const_iterator;
      private:
        BinaryArrayExp();

      public:
        BinaryArrayExp(const Exp0& e0, const Exp1& e1, op_type op) :
            e0_(e0), e1_(e1), op_(op) { }

        /// Evaluate this expression
        result_type eval() const {
          return op_(ExpEval::eval(e0_), ExpEval::eval(e1_));
        }

        /// Return a reference to the world object associated with this expression.
        madness::World& get_world() const { return e0_.get_world(); }

      private:
        const Exp0& e0_;
        const Exp1& e1_;
        op_type op_;
      }; // struct BinaryArrayExp

      /// Unary Array Expression

      /// This structure represents a unary tile math expression. The Op type
      /// represents the basic math operation that will be performed on each
      /// element. The expression types may be annotated tiles, fundamental types,
      /// or other tile expressions. The expression does a lazy evaluation, i.e.
      /// it is only evaluated when the eval() function is explicitly called. If
      /// the argument is another expression, it will be evaluated before this
      /// expression.
      template<typename Exp, typename Op>
      struct UnaryArrayExp {
        typedef typename ExpType<Exp>::result_type exp_type;
        typedef exp_type result_type;
        typedef typename exp_type::value_type value_type;
        typedef typename exp_type::tile_type tile_type;
        typedef Op op_type;
  //      typedef typename result_type::const_iterator const_iterator;

        UnaryArrayExp(const Exp& e, Op op) : e_(e), op_(op) { }

        /// Evaluate this expression.
        result_type eval() const {
          return op_(ExpEval::eval(e_));
        }

        /// Return a reference to the world object associated with this expression.
        madness::World& get_world() const { return e_.get_world(); }

      private:

        UnaryArrayExp();

        const Exp& e_;
        op_type op_;
      }; // struct UnaryArrayExp

      template<typename Exp0, typename Exp1, template<typename> class Op>
      struct ExpConstruct {
        typedef typename ExpPair<Exp0, Exp1>::result_type result_type;
        typedef math::BinaryArrayOp<typename ExpType<Exp0>::result_type,
            typename ExpType<Exp1>::result_type, result_type,
            Op<typename result_type::value_type> > op_type;
        typedef BinaryArrayExp<Exp0, Exp1, op_type> exp_type;

        static exp_type make_exp(const Exp0& e0, const Exp1& e1) {
          return exp_type(e0, e1, op_type(e0.get_world()));
        }
      };

      template<typename Exp0, typename T, template<typename> class Op>
      struct ExpConstruct<Exp0, ValueExp<T>, Op> {
        typedef math::UnaryArrayOp<
            typename ExpType<Exp0>::result_type,
            typename ExpType<Exp0>::result_type,
            boost::binder2nd< Op<typename Exp0::value_type> > > op_type;
        typedef UnaryArrayExp<Exp0, op_type> exp_type;

        static exp_type make_exp(const Exp0& e0, const ValueExp<T> e1) {
          return exp_type(e0, op_type(e0.get_world(), boost::bind2nd(Op<typename Exp0::value_type>(), e1.eval())));
        }
      };

      template<typename T, typename Exp1, template<typename> class Op>
      struct ExpConstruct<ValueExp<T>, Exp1, Op> {
        typedef math::UnaryArrayOp<
            typename ExpType<Exp1>::result_type,
            typename ExpType<Exp1>::result_type,
            boost::binder1st< Op<typename Exp1::value_type> > > op_type;
        typedef UnaryArrayExp<Exp1, op_type> exp_type;

        static exp_type make_exp(const ValueExp<T> e0, const Exp1& e1) {
          return exp_type(e1, op_type(e1.get_world(), boost::bind1st(Op<typename Exp1::value_type>(), e0.eval())));
        }
      };

      template<typename Exp0, typename Exp1, template<typename> class Op >
      struct Expression :
          public array::ExpConstruct<typename ExpType<Exp0>::type, typename ExpType<Exp1>::type, Op> {

      }; // struct Expression

      /// Array expression addition operation

      /// This operator constructs a tile binary, addition expression object. The
      /// expression is not immediately evaluated.
      template<typename Exp0, typename Exp1>
      typename array::Expression<Exp0, Exp1, std::plus>::exp_type
      operator +(const Exp0& e0, const Exp1& e1) {
        return array::Expression<Exp0, Exp1, std::plus>::make_exp(e0, e1);
      }

      /// Array expression subtraction operation

      /// This operator constructs a tile binary, subtraction expression object.
      /// The expression is not immediately evaluated.
      template<typename Exp0, typename Exp1>
      typename array::Expression<Exp0, Exp1, std::minus>::exp_type
      operator -(const Exp0& e0, const Exp1& e1) {
        return array::Expression<Exp0, Exp1, std::minus>::make_exp(e0, e1);
      }

      /// Array expression multiplication or contraction operation

      /// This operator constructs a tile binary, multiplication or contraction
      /// expression object. A multiplication expression is constructed when one
      /// of the expressions is a constant value. Otherwise, a contraction
      /// expression is constructed if both expressions will evaluate to annotated
      /// tiles. The expression is not immediately evaluated.
      template<typename Exp0, typename Exp1>
      typename array::Expression<Exp0, Exp1, std::multiplies>::exp_type
      operator *(const Exp0& e0, const Exp1& e1) {
        return array::Expression<Exp0, Exp1, std::multiplies>::make_exp(e0, e1);
      }

      /// Array expression negate operation

      /// This operator constructs a negation expression object. The expression is
      /// not immediately evaluated.
      template<typename Exp>
      UnaryArrayExp<Exp, math::UnaryArrayOp<
          typename ExpType<Exp>::result_type,
          typename ExpType<Exp>::result_type,
          std::negate<typename Exp::value_type> > >
      operator -(const Exp& e) {
        typedef math::UnaryArrayOp<
            typename ExpType<Exp>::result_type,
            typename ExpType<Exp>::result_type,
            std::negate<typename Exp::value_type> > op_type;
        return UnaryArrayExp<Exp, op_type>(e, op_type(e.get_world()));
      }

    } // namespace array

  } // namespace expressions
*/
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_MATH_H__INCLUDED
