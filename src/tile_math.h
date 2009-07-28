#ifndef TILEDARRAY_TILE_MATH_H__INCLUDED
#define TILEDARRAY_TILE_MATH_H__INCLUDED

#include <tile.h>
#include <boost/iterator/iterator_facade.hpp>

namespace TiledArray {

  namespace detail {

    template<typename T>
    struct diff_type {
      typedef typename T::difference_type difference_type;
    };

    template<typename T>
    struct diff_type<T*> {
      typedef std::size_t difference_type;
    };

    /// Step iterator

    /// This iterator will iterate in steps through a container. The base
    /// iterator must be a random access iterator or pointer.
    template<typename RandIter>
    class StepIterator : public boost::iterator_facade< StepIterator<RandIter>,
        typename RandIter::value_type, std::random_access_iterator_tag,
        typename RandIter::reference, typename diff_type<RandIter>::difference_type >
    {
    private:
      typedef boost::iterator_facade< StepIterator<RandIter>,
      typename RandIter::value_type, std::random_access_iterator_tag,
      typename RandIter::reference, typename diff_type<RandIter>::difference_type >
      base_type;
    public:
      typedef StepIterator<RandIter> StepIterator_;
      typedef RandIter base_iterator;
      typedef typename base_type::difference_type difference_type;

      /// Primary constructor

      /// \arg \c it is a \c base_iterator that identifies the current position
      /// of the iterator.
      /// \arg \c step is the step size for iterator increment.
      StepIterator(base_iterator it, const difference_type step) :
        i_(it), s_(step) { }

      StepIterator(const StepIterator_& other) : i_(other.i_), s_(other.s_) { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move constructor
      StepIterator(StepIterator_&& other) : i_(std::move(other.i_)), s_(other.s_) { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      StepIterator_& operator =(const StepIterator_& other) {
        i_ = other.i_;
        s_ = other.s_;

        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      StepIterator_& operator =(StepIterator_&& other) {
        i_ = std::move(other.i_);
        s_ = std::move(other.s_);

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Returns a reference to the base iterator.
      base_iterator& base() { return i_; }
      /// Returns a constant reference to the base iterator.
      const base_iterator& base() const { return i_; }

    private:
      friend class boost::iterator_core_access;

      /// No default construction allowed.
      StepIterator();

      /// Returns a reference to the the element pointed to by
      typename base_type::reference dereference() const { return *i_; }

      /// Returns true if the base iterators are equal. Throws std::runtime_error
      /// if the step sizes are not equal.
      bool equal(const StepIterator_& other) const {
        TA_ASSERT(s_ == other.s_,
            std::runtime_error("StepIterator<...>::equal(...): The step sizes are not equal."));
        return i_ == other.i_;
      }

      /// advances the base iterator by step.
      void increment() { i_ += s_; }

      /// reverse advance the base iterator by step.
      void decrement() { i_ -= s_; }

      /// advances the base iterator by step * n.
      void advance(const difference_type n) { i_ += n * s_; }

      /// Returns the difference between this iterator and the other iterator
      /// divided by step. Throws if the step sizes of this and the other
      /// iterator are different or if the other base iterator does not fall
      /// on a step boundary of the this base iterator.
      difference_type distance_to(const StepIterator_& other) const {
        TA_ASSERT(s_ == other.s_,
            std::runtime_error("StepIterator<...>::distance_to(...): The step sizes are not equal."));
        TA_ASSERT((((other.i_ -  i_) % s_) == 0),
            std::runtime_error("StepIterator<...>::distance_to(...): The other iterator is not on a valid "));

        return (other.i_ -  i_) / s_;
      }

      base_iterator i_;
      difference_type s_;
    };

    /// Tile math expression
    template <typename T0, typename T1, typename TR, typename Op>
    struct TileExp : public std::binary_function<detail::AnnotatedTile<T0>,
        detail::AnnotatedTile<T1>, detail::AnnotatedTile<TR> >
    {
      typedef std::binary_function<detail::AnnotatedTile<T0>,
          detail::AnnotatedTile<T1>, detail::AnnotatedTile<TR> > func_type;
      typedef typename func_type::first_argument_type first_argument_type;
      typedef typename func_type::second_argument_type second_argument_type;
      typedef typename func_type::result_type result_type;

    };


































    /// Applies a unary operation to the base iterator when it is dereferenced.

    /// When this iterator is dereferenced, the base iterator is dereferenced
    /// and the unary operation is applied its value. The dereference operation
    /// returns the transformed result by value (not reference). This iterator
    /// category is an input iterator.
    template<typename Iter, typename Op>
    class UnaryTransformIterator : public boost::iterator_facade<
        UnaryTransformIterator<Iter, Op>, typename Op::result_type,
        std::input_iterator_tag, typename Op::result_type >
    {
    private:
      typedef typename boost::iterator_facade< UnaryTransformIterator<Iter, Op>,
          typename Op::result_type, std::input_iterator_tag,
          typename Op::result_type > base_type;

    public:
      typedef UnaryTransformIterator<Iter, Op> UnaryTransformIterator_;
      typedef Iter base_iterator;
      typedef Op operator_type;

      /// Primary constructor

      /// \arg \c i is the base iterator used by the transform iterator.
      /// \arg \c op is the unary operator applied to the dereferenced base iterator (optional).
      UnaryTransformIterator(base_iterator i, Op op = Op()) : i_(i), op_(op) { }

      /// Copy constructor

      /// \arg \c other is the \c UnaryTransformIterator to be copied.
      UnaryTransformIterator(const UnaryTransformIterator_& other) :
          i_(other.i_), op_(other.op_)
      { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move constructor

      /// \arg \c other is the \c UnaryTransformIterator to be copied.
      UnaryTransformIterator(UnaryTransformIterator_&& other) :
          i_(std::move(other.i_)), op_(std::move(other.op_))
      { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      ~UnaryTransformIterator() { }

      /// Assignement operator
      UnaryTransformIterator_& operator =(const UnaryTransformIterator_& other) {
        i_ = other.i_;
        op_ = other.op_;

        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move assignment operator
      UnaryTransformIterator_& operator =(UnaryTransformIterator_&& other) {
        i_ = std::move(other.i_);
        op_ = std::move(other.op_);

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// returns a constant reference to the operator.
      const operator_type& functor() const { return op_; }

      /// returns a constant reference to the base iterator.
      const base_iterator& base() const { return i_; }

    private:
      friend class boost::iterator_core_access;

      /// Default construction not allowed.
      UnaryTransformIterator();

      /// Returns the transformed value of the base iterator. The result is
      /// returned by value (not reference).
      typename base_type::value_type dereference() const { return op_(*i_); }

      /// Returns true when the base iterator of this and the other iterators
      /// are equal.
      bool equal(const UnaryTransformIterator_& other) const { return i_ == other.i_; }

      /// Increments the base iterator.
      void increment() { ++i_; }

      base_iterator i_; ///< base iterator
      Op op_;           ///< unary operator, which transforms the base iterator
    }; // class UnaryTransformIterator


    /// Applies a binary operation to a pair of base iterators when it is dereferenced.

    /// This iterator applies a binary operation to a pair of iterators (i.e.
    /// op(*iterator0, *iterator1); ). When dereferenced the result of the
    /// transformation is returned by value (not by reference). When this
    /// iterator is incremented, both iterator0 and iterator1 are incremented.
    /// This iterator category is an input iterator. This iterator combines the
    /// Boost transform and zip iterator concepts with a binary transformation.
    template<typename Iter0, typename Iter1, typename Op>
    class BinaryTransformIterator : public boost::iterator_facade<
        BinaryTransformIterator<Iter0, Iter1, Op>, typename Op::result_type,
        std::input_iterator_tag, typename Op::result_type >
    {
    private:
      typedef typename boost::iterator_facade<
          BinaryTransformIterator<Iter0, Iter1, Op>, typename Op::result_type,
          std::input_iterator_tag > base_type;

    public:
      typedef BinaryTransformIterator<Iter0, Iter1, Op> BinaryTransformIterator_;
      typedef Iter0 base_iterator0;
      typedef Iter1 base_iterator1;
      typedef Op operator_type;

      /// Primary constructor.

      /// \arg \c i0 is the base iterator0.
      /// \arg \c i1 is the base iterator1.
      /// \arg \c op is the binary operator applied to the dereferenced base iterators (optional).
      BinaryTransformIterator(base_iterator0 i0, base_iterator1 i1, Op op = Op()) : i0_(i0), i1_(i1), op_(op) { }

      /// Copy constructor
      BinaryTransformIterator(const BinaryTransformIterator_& other) :
          i0_(other.i0_), i1_(other.i1_), op_(other.op_)
      { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move constructor
      BinaryTransformIterator(BinaryTransformIterator_&& other) :
          i0_(std::move(other.i0_)), i1_(std::move(other.i1_)), op_(std::move(other.op_))
      { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      ~BinaryTransformIterator() { }

      /// Assignment operator
      BinaryTransformIterator_& operator =(const BinaryTransformIterator_& other) {
        i0_ = other.i0_;
        i1_ = other.i1_;
        op_ = other.op_;

        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move assignment operator.
      BinaryTransformIterator_& operator =(BinaryTransformIterator_&& other) {
        i0_ = std::move(other.i0_);
        i1_ = std::move(other.i1_);
        op_ = std::move(other.op_);

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Returns a constant reference to the operator.
      const operator_type& functor() const { return op_; }

      /// Returns a constant reference to base iterator0.
      const base_iterator0& base0() const { return i0_; }

      /// Return a constant reference to base iterator1.
      const base_iterator1& base1() const { return i1_; }

    private:
      friend class boost::iterator_core_access;

      /// Default construction not allowed.
      BinaryTransformIterator();

      /// Returns the transformed iterator pair.

      /// The result of the transformation is returned by value (not by
      /// reference). The operation preformed is op(*iterator0, *iterator1);.
      typename base_type::value_type dereference() const { return op_(*i0_, *i1_); }

      /// Returns true if one of the two iterators are equal to its conterpart
      /// in other.

      /// Returns true if iterator0 or iterator1 are equal to iterator0 or
      /// iterator1 of \c other transform iterator respectively. This is done so
      /// that in the case where there are different number of elements in the
      /// range of iterators, an end transform iterator will stop the loop at
      /// the first end encountered. Note: if you another type of comparison
      /// scheme, you can compare the base iterators directly with base0() and
      /// base1() functions.
      /// \arg \c other is the transform iterator to be compared with this one.
      bool equal(const BinaryTransformIterator_& other) const { return i0_ == other.i0_ || i1_ == other.i1_; }

      /// increments both iterator0 and iterator1
      void increment() { ++i0_; ++i1_; }

      base_iterator0 i0_; ///< base iterator0
      base_iterator1 i1_; ///< base iterator1
      Op op_;             ///< binary transform operation
    }; // class BinaryTransformIterator

    template<typename Op>
    class SudoTransformIterator : public boost::iterator_facade<
        SudoTransformIterator<Op>, typename Op::result_type,
        std::input_iterator_tag, typename Op::result_type >
    {
    private:
      typedef boost::iterator_facade<SudoTransformIterator<Op>,
          typename Op::result_type, std::input_iterator_tag,
          typename Op::result_type > base_type;

    public:
      typedef SudoTransformIterator<Op> SudoTransformIterator_;
      typedef Op operator_type;
      typedef typename Op::first_argument_type arg0;
      typedef typename Op::second_argument_type arg1;

      /// Primary constructor, which evaluates and stores the expression.
      SudoTransformIterator(const arg0 a0, const arg1 a1, operator_type op = operator_type()) :
          v_(op(a0,a1))
      { }

      /// Copy constructor
      SudoTransformIterator(const SudoTransformIterator& other) : v_(other.v_) { }

      ~SudoTransformIterator() { }

    private:

      /// Default construction not allowed.
      SudoTransformIterator();

      typename base_type::value_type dereference() { return v_; }
      bool equal() const { return false; }
      void increment() const { }

      const typename base_type::value_type v_; ///< value of evaluated expression.
    }; // class SudoTransformIterator

    /// Value holds a constant value expression.
    template<typename T>
    class Value {
    public:
      typedef boost::remove_const<T> type;
      Value(const type v) : v_(v) { }
      Value(const Value<type>& other) : v_(other.v_) { }

      static const VariableList& vars() { return var_; }
      const type eval() const { return v_; }
    private:
      const type v_;
      static const VariableList var_;
    }; // class value

    template<typename T>
    const VariableList Value<T>::var_ = VariableList();

    // Expression types are used to so constant values are handled correctly.

    /// Base case for expression type
    template<typename Exp>
    struct ExpType {
      typedef typename boost::remove_const<typename boost::remove_reference<Exp>::type>::type type;
      typedef typename type::value_type value_type;
    };

    /// int expression type specialization
    template<>
    struct ExpType<int> {
      typedef Value<int> type;
      typedef int value_type;
    };

    /// long expression type specialization
    template<>
    struct ExpType<long> {
      typedef Value<long> type;
      typedef long value_type;
    };

    /// long long expression type specialization
    template<>
    struct ExpType<long long> {
      typedef Value<long long> type;
      typedef long long value_type;
    };

    /// unsigned int expression type specialization
    template<>
    struct ExpType<unsigned int> {
      typedef Value<unsigned int> type;
      typedef unsigned int value_type;
    };

    /// unsigned long expression type specialization
    template<>
    struct ExpType<unsigned long> {
      typedef Value<unsigned long> type;
      typedef unsigned long value_type;
    };

    /// unsigned long long expression type specialization
    template<>
    struct ExpType<unsigned long long> {
      typedef Value<unsigned long long> type;
      typedef unsigned long long value_type;
    };

    /// double expression type specialization
    template<>
    struct ExpType<double> {
      typedef Value<double> type;
      typedef double value_type;
    };

    /// float expression type specialization
    template<>
    struct ExpType<float> {
      typedef Value<float> type;
      typedef float value_type;
    };

    template<typename T, typename U>
    struct TypeSelector {
      typedef T type;
    };

    template<typename T>
    struct TypeSelector<T, T> {
      typedef T type;
    };

    //TODO: add specializations for TypeSelector to make sure the correct type
    // is selected.

    /// Expression pair class provides appropriate iterator types and
    /// factory functions.
    template<typename Exp0, typename Exp1, typename Op>
    struct ExpPair {
      typedef BinaryTransformIterator<typename Exp0::iterator, typename Exp1::iterator, Op> const_iterator;

      static const_iterator begin(const Exp0& e0, const Exp1& e1, const Op& op) {
        return const_iterator(e0.begin(), e1.begin(), op);
      }

      static const_iterator end(const Exp0& e0, const Exp1& e1, const Op& op) {
        return const_iterator(e0.end(), e1.end(), op);
      }

    };

    /// Expression pair one value specialization.
    template<typename Exp0, typename Op>
    struct ExpPair<Exp0, Value<typename Op::second_argument_type>, Op > {
      typedef UnaryTransformIterator<typename Exp0::iterator, std::binder2nd<Op> > const_iterator;

      static const_iterator begin(const Exp0& e0, const Value<typename Op::second_argument_type>& v1, const Op& op) {
        return const_iterator(e0.begin(), std::bind2nd(op, v1.eval()));
      }

      static const_iterator end(const Exp0& e0, const Value<typename Op::second_argument_type>& v1, const Op& op) {
        return const_iterator(e0.end(), std::bind2nd(op, v1.eval()));
      }
    };

    /// Expression pair one value specialization.
    template<typename Exp1, typename Op>
    struct ExpPair<Value<typename Op::first_argument_type>, Exp1, Op > {
      typedef UnaryTransformIterator<typename Exp1::iterator, std::binder1st<Op> > const_iterator;

      static const_iterator begin(const Value<typename Op::first_argument_type>& v0, const Exp1& e1, const Op& op) {
        return const_iterator(e1.begin(), std::bind1st(op, v0.eval()));
      }

      static const_iterator end(const Value<typename Op::first_argument_type>& v0, const Exp1& e1, const Op& op) {
        return const_iterator(e1.end(), std::bind1st(op, v0.eval()));
      }
    };

/*
    /// Expression pair two value specialization.
    template<typename Op>
    struct ExpPair<Value<typename Op::first_argument_type>, Value<typename Op::second_argument_type>, Op> {
      typedef SudoTransformIterator<Op> const_iterator;

      static const_iterator begin(const Value<typename Op::first_argument_type>& v0, const Value<typename Op::second_argument_type>& v1, const Op& op) {
        return const_iterator(v0.eval(), v1.eval(), op);
      }

      static const_iterator end(const Value<typename Op::first_argument_type>& v0, const Value<typename Op::second_argument_type>& v1, const Op& op) {
        return const_iterator(v0.eval(), v1.eval(), op);
      }
    };
*/

    /// Expression pair value-tile specialization.
    template<typename T, typename Op>
    struct ExpPair<Value<typename Op::first_argument_type>, detail::AnnotatedTile<T>, Op> {
      typedef UnaryTransformIterator<typename detail::AnnotatedTile<T>::const_iterator, std::binder1st<Op> > const_iterator;

      static const_iterator begin(const Value<typename Op::first_argument_type>& v0, const detail::AnnotatedTile<T>& t1, const Op& op) {
        return const_iterator(t1.begin(), std::bind1st(op, v0));
      }

      static const_iterator end(const Value<typename Op::first_argument_type>& v0, const detail::AnnotatedTile<T>& t1, const Op& op) {
        return const_iterator(t1.end(), std::bind1st(op, v0));
      }
    };

    /// Expression pair tile-value specialization.
    template<typename T, typename Op>
    struct ExpPair<detail::AnnotatedTile<T>, Value<typename Op::second_argument_type>, Op> {
      typedef UnaryTransformIterator<typename detail::AnnotatedTile<T>::const_iterator, std::binder2nd<Op> > const_iterator;

      static const_iterator begin(const detail::AnnotatedTile<T>& t0, const Value<typename Op::second_argument_type>& v1, const Op& op) {
        return const_iterator(t0.begin(), std::bind2nd(op, v1));
      }

      static const_iterator end(const detail::AnnotatedTile<T>& t0, const Value<typename Op::second_argument_type>& v1, const Op& op) {
        return const_iterator(t0.end(), std::bind2nd(op, v1));
      }
    };

    /// Expression pair expression-tile specialization.
    template<typename Exp0, typename T, typename Op>
    struct ExpPair<Exp0, detail::AnnotatedTile<T>, Op> {
      typedef BinaryTransformIterator<typename Exp0::iterator, typename detail::AnnotatedTile<T>::const_iterator, Op> const_iterator;

      static const_iterator begin(const Exp0& e0, const detail::AnnotatedTile<T>& t1, const Op& op) {
        return const_iterator(e0.begin(), t1.begin(), op);
      }

      static const_iterator end(const Exp0& e0, const detail::AnnotatedTile<T>& t1, const Op& op) {
        return const_iterator(e0.end(), t1.end(), op);
      }
    };

    /// Expression pair tile-expression specialization.
    template<typename T, typename Exp1, typename Op>
    struct ExpPair<detail::AnnotatedTile<T>, Exp1, Op> {
      typedef BinaryTransformIterator<typename detail::AnnotatedTile<T>::const_iterator, typename Exp1::iterator, Op> const_iterator;

      static const_iterator begin(const detail::AnnotatedTile<T>& t0, const Exp1& e1, const Op& op) {
        return const_iterator(t0.begin(), e1.begin(), op);
      }

      static const_iterator end(const detail::AnnotatedTile<T>& t0, const Exp1& e1, const Op& op) {
        return const_iterator(t0.end(), e1.end(), op);
      }
    };

    /// Expression pair tile-tile specialization.
    template<typename T, typename U, typename Op>
    struct ExpPair<detail::AnnotatedTile<T>, detail::AnnotatedTile<U>, Op> {
      typedef BinaryTransformIterator<typename detail::AnnotatedTile<T>::const_iterator,
          typename detail::AnnotatedTile<U>::const_iterator, Op> const_iterator;

      static const_iterator begin(const detail::AnnotatedTile<T>& t0, const detail::AnnotatedTile<T>& t1, const Op& op) {
        return const_iterator(t0.begin(), t1.begin(), op);
      }

      static const_iterator end(const detail::AnnotatedTile<T>& t0, const detail::AnnotatedTile<T>& t1, const Op& op) {
        return const_iterator(t0.end(), t1.end(), op);
      }
    };

    template<typename Op>
    struct VarListOp {
      const VariableList& operator()(const VariableList& v0, const VariableList& v1) {
        if((v0 == v1) && (v0.count() != 0) && (v1.count() != 0))
          return v0;
        else if(v0.count() == 0 && v1.count() != 0)
          return v1;
        else if(v1.count() == 0 && v0.count() != 0)
          return v0;
        else
          throw std::runtime_error("VarListOp<Op>::operator(): Variable lists do not match or are both zero length.");
      }
    };

    template<>
    struct VarListOp<std::multiplies<VariableList> > {
      const VariableList operator()(const VariableList& v0, const VariableList& v1) {
        return m_(v0, v1);
      }

    private:
      std::multiplies<VariableList> m_;
    };

    /// Binary array expression.

    /// The binary expression class is used to evaluate arrays
    template<typename Exp0, typename Exp1, template<typename T> class Op>
    class BinaryArrayExp {
    public:
      typedef BinaryArrayExp<Exp0,Exp1,Op> BinaryArrayExp_;
      typedef typename ExpType<Exp0>::type exp0_type;
      typedef typename ExpType<Exp1>::type exp1_type;
      typedef typename TypeSelector<typename exp0_type::value_type, typename exp1_type::value_type>::type value_type;
      typedef Op<value_type> operator_type;
      typedef typename ExpPair<exp0_type, exp1_type, operator_type>::const_iterator const_iterator;

      BinaryArrayExp(const exp0_type& e0, const exp1_type& e1, const operator_type& op = operator_type()) :
          e0_(e0), e1_(e1), op_(op), v_()
      {
        VarListOp< Op<VariableList> > vop;
        v_ = vop(e0.vars(), e1.vars());
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      BinaryArrayExp(exp0_type&& e0, exp1_type&& e1, operator_type&& op = operator_type()) :
          e0_(std::move(e0)), e1_(std::move(e1)), op_(op), v_()
      {
        VarListOp< Op<VariableList> > vop;
        v_ = vop(e0_.vars(), e1_.vars());
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      BinaryArrayExp(const BinaryArrayExp_& other) :
          e0_(other.e0_), e1_(other.e1_), op_(op_), v_(other.v_)
      { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      BinaryArrayExp(BinaryArrayExp_&& other) :
          e0_(std::move(other.e0_)), e1_(std::move(other.e1_)), op_(std::move(other.op_)), v_(std::move(other.v_))
      { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      const_iterator begin() const { return ExpPair<exp0_type, exp1_type, operator_type>::begin(e0_, e1_, op_); }
      const_iterator end() const { return ExpPair<exp0_type, exp1_type, operator_type>::end(e0_, e1_, op_); }

      const VariableList& vars() const { return v_; }

    private:
      exp0_type e0_;
      exp1_type e1_;
      operator_type op_;
      VariableList v_;

    };

    template< template<typename T> class Op, typename Exp0, typename Exp1>
    BinaryArrayExp<Exp0, Exp1, Op>
    make_array_exp(Exp0&& e0, Exp1&& e1) {
      return BinaryArrayExp<Exp0, Exp1, Op>(std::forward<Exp0>(e0), std::forward<Exp1>(e1));
    }

    template<typename Exp0, typename Exp1>
    BinaryArrayExp<Exp0, Exp1, std::plus > operator +(Exp0&& e0, Exp1&& e1) {
      return make_array_exp<std::plus>(std::forward<Exp0>(e0), std::forward<Exp1>(e1));
    }

    template<typename Exp0, typename Exp1>
    BinaryArrayExp<Exp0, Exp1, std::minus > operator -(Exp0&& e0, Exp1&& e1) {
      return make_array_exp<std::minus>(std::forward<Exp0>(e0), std::forward<Exp1>(e1));
    }


/*

    /// Contraction of two rank 3 tensors.
    /// r[a,b,c,d] = t0[a,i,c] * t1[b,i,d]
    template<typename T, detail::DimensionOrderType Order>
    void contract_aic_x_bid(const Tile<T,3,CoordinateSystem<3,Order> >& t0, const Tile<T,3,CoordinateSystem<3,Order> >& t1, Tile<T,4,CoordinateSystem<4,Order> >& tr) {
      typedef Tile<T,3,CoordinateSystem<3,Order> > Tile3;
      typedef Tile<T,4,CoordinateSystem<4,Order> > Tile4;
      typedef Eigen::Matrix< T , Eigen::Dynamic , Eigen::Dynamic, (Order == decreasing_dimension_order ? Eigen::RowMajor : Eigen::ColMajor) | Eigen::AutoAlign > matrix_type;
      TA_ASSERT(t0.size()[1] == t1.size()[1],
          std::runtime_error("void contract(const contraction_pair<T,3>& t0, const contraction_pair<T,3>& t1, contraction_pair<T,4>& tr): t0[1] != t1[1]."));

      const unsigned int i0 = Tile3::coordinate_system::ordering().order2dim(0);
      const unsigned int i1 = Tile3::coordinate_system::ordering().order2dim(1);
      const unsigned int i2 = Tile3::coordinate_system::ordering().order2dim(2);

      typename Tile4::size_array s;
      typename Tile4::coordinate_system::const_reverse_iterator it = Tile4::coordinate_system::rbegin();
      s[*it++] = t0.size()[i2];
      s[*it++] = t1.size()[i2];
      s[*it++] = t0.size()[i0];
      s[*it] = t1.size()[i0];

      if(tr.size() != s)
        tr.resize(s);

      const typename Tile3::ordinal_type step0 = t0.weight()[i2];
      const typename Tile3::ordinal_type step1 = t1.weight()[i2];
      const typename Tile4::ordinal_type stepr = t0.size()[i0] * t1.size()[i0];
      const typename Tile3::value_type* p0_begin = NULL;
      const typename Tile3::value_type* p0_end = t0.data() + step0 * t0.size()[i2];
      const typename Tile3::value_type* p1_begin = NULL;
      const typename Tile3::value_type* p1_end = t1.data() + step1 * t1.size()[i2];
      typename Tile4::value_type* pr = tr.data();

      for(p0_begin = t0.data(); p0_begin != p0_end; p0_begin += step0) {
        Eigen::Map<matrix_type> m0(p0_begin, t0.size()[i1], t0.size()[i0]);
        for(p1_begin = t1.begin(); p1_begin != p1_end; p1_begin += step1, pr += stepr) {
          Eigen::Map<matrix_type> mr(pr, t0.size()[i0], t1.size()[i0]);
          Eigen::Map<matrix_type> m1(p1_begin, t0.size()[i1], t1.size()[i0]);

          mr = m0.transpose() * m1;
        }
      }
    }

    /// Contraction of a rank 3 and rank 2 tensor.
    /// r[a,b,c] = t0[a,i,b] * t1[i,c]
    template<typename T, detail::DimensionOrderType Order>
    void contract_aib_x_ic(const Tile<T,3,CoordinateSystem<3,Order> >& t0, const Tile<T,2,CoordinateSystem<2,Order> >& t1, Tile<T,3,CoordinateSystem<3,Order> >& tr) {
      typedef Tile<T,2,CoordinateSystem<2,Order> > Tile2;
      typedef Tile<T,3,CoordinateSystem<3,Order> > Tile3;
      typedef Eigen::Matrix< T , Eigen::Dynamic , Eigen::Dynamic, (Order == decreasing_dimension_order ? Eigen::RowMajor : Eigen::ColMajor) | Eigen::AutoAlign > matrix_type;

      const unsigned int i2_0 = Tile2::coordinate_system::ordering().order2dim(0);
      const unsigned int i2_1 = Tile2::coordinate_system::ordering().order2dim(1);

      const unsigned int i3_0 = Tile3::coordinate_system::ordering().order2dim(0);
      const unsigned int i3_1 = Tile3::coordinate_system::ordering().order2dim(1);
      const unsigned int i3_2 = Tile3::coordinate_system::ordering().order2dim(2);

      TA_ASSERT(t0.size()[i3_1] == t1.size()[i2_1],
          std::runtime_error("void contract(const contraction_pair<T,3>& t0, const contraction_pair<T,3>& t1, contraction_pair<T,4>& tr): t0[1] != t1[1]."));

      typename Tile3::size_array s;
      typename Tile3::coordinate_system::const_reverse_iterator it = Tile3::coordinate_system::rbegin();
      s[*it++] = t0.size()[i3_2];
      s[*it++] = t0.size()[i3_0];
      s[*it] = t1.size()[i2_0];

      if(tr.size() != s)
        tr.resize(s);

      const typename Tile3::ordinal_type step0 = t0.weight()[i3_2];
      const typename Tile3::ordinal_type stepr = t0.size()[i3_0] * t1.size()[i2_0];
      const typename Tile3::value_type* p0_begin = NULL;
      const typename Tile3::value_type* p0_end = t0.data() + step0 * t0.size()[i3_2];
      typename Tile3::value_type* pr = tr.data();

      Eigen::Map<matrix_type> m1(t1.begin(), t0.size()[i2_1], t1.size()[i2_0]);

      for(p0_begin = t0.data(); p0_begin != p0_end; p0_begin += step0, pr += stepr) {
        Eigen::Map<matrix_type> m0(p0_begin, t0.size()[i3_1], t0.size()[i3_0]);
        Eigen::Map<matrix_type> mr(pr, t0.size()[i3_0], t1.size()[i2_0]);
        mr = m0.transpose() * m1;
      }
    }

    /// Contraction of two 3D arrays.
    /// r = t0[i] * t1[i]
    template<typename T, typename CS>
    void contract(const Tile<T,1,CS>& t0, const Tile<T,1,CS>& t1, T& tr) {
      TA_ASSERT(t0.volume() == t1.volume(),
          std::runtime_error("void contract(const contraction_pair<T,1>& t0, const contraction_pair<T,1>& t1, T& tr): t0[0] != t1[0]."));

      tr = 0;
      for(std::size_t i = 0; i < t0.volume(); ++i)
        tr += t0[i] * t1[i];
    }

*/
  } // namespace detail

} // namespace TiledArray

#endif // TILEDARRAY_TILE_MATH_H__INCLUDED
