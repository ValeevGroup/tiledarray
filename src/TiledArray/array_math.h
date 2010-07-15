#ifndef TILEDARRAY_ARRAY_MATH_H__INCLUDED
#define TILEDARRAY_ARRAY_MATH_H__INCLUDED

namespace TiledArray {

  namespace math {







/*
    /// Array operation

    /// Performs an element wise binary operation (e.g. std::plus<T>,
    /// std::minus<T>) on two annotated tiles. The value type of the different
    /// tiles may be different, but the value types of expression one and two
    /// must be implicitly convertible to the result value type.
    template<typename Arg1, typename Arg2, typename Res, template <typename> class Op>
    struct BinaryArrayOp {
    private:
      BinaryArrayOp();
      typedef BinaryArrayOp<Arg1, Arg2, Res, Op> BinaryArrayOp_;

    public:
      typedef const Arg1& first_argument_type;  ///< first array argument type.
      typedef const Arg2& second_argument_type; ///< second array argument type.
      typedef Res result_type;                  ///< result array type.
      typedef BinaryTileOp<typename Arg1::tile_type, typename Arg2::tile_type,
          typename result_type::tile_type, Op> tile_op; ///< Binary tile operation

    private:

      /// Copies of this object are passed to the madness task queue.

      /// This object handles decision making which operations should be run
      /// based on the existence or non-existence of tiles.
      class ProbeOp {
        ProbeOp();
      public:
        ProbeOp(tile_op o) : op_(o) { }

        /// Executes operations for local left hand tile operands.

        /// This function checks for the existence of the right hand operand.
        /// If it exists, the binary operation is performed and the result is
        ///
        typename result_type::tile_type left_op(const typename Arg1::tile_type& t1, const typename Arg2::tile_type& t2) const {
          if(t2.initialized())
            return op_(t1, t2);

          return op_(t1, 0);
        }

        typename result_type::tile_type right_op(bool p1, typename Arg2::tile_type t2) const {
          if(!p1)
            return op_(0, t2);

          typename result_type::tile_type result;
          return result;
        }

      private:
        tile_op op_; ///< Tile operation to be executed.
      }; // class ProbeOp

    public:
      /// operation constructor
      /// \arg \c w is a reference to the world object where tasks will be spawned.
      /// \arg \c o is the functor or function that will be used in tile operations.
      /// \arg \c a is the task attributes that will be used when spawning tile, task operations.
      BinaryArrayOp(madness::World& w, tile_op o = tile_op(), madness::TaskAttributes a = madness::TaskAttributes()) :
          world_(w), op_(o), attr_(a)
      { }

      /// Constructs a series of tasks for the given arrays.
      result_type operator ()(first_argument_type a1, second_argument_type a2) {
        // Here we assume that the array tiles have compatible sizes because it
        // is checked in the expression generation functions (if error checking
        // is enabled.

        // This loop will generate the appropriate tasks for the cases where the
        // left and right tile exist and only the left tile exists.
        result_type result(world_, a1.range(), a1.vars(), a1.order());
        ProbeOp probe(op_);
        for(typename Arg1::const_iterator it = a1.begin(); it != a1.end(); ++it) {
          const typename Arg1::index_type i = it->first;
          madness::Future<typename Arg1::tile_type> f1 = it->second;
          madness::Future<typename Arg2::tile_type> f2 = a2.find(i)->second;

          madness::Future<typename result_type::tile_type> fr =
              world_.taskq.add(probe, &ProbeOp::left_op, f1, f2, attr_);

          result.insert(i, fr);
        }

        // This loop will generate the appropriate tasks for the cases where
        // only the right tile exists.
        for(typename Arg2::const_iterator it = a2.begin(); it != a2.end(); ++it) {
          const typename Arg2::index_type i = it->first;
          madness::Future<bool> f1 = a1.probe(i);
          madness::Future<typename Arg2::tile_type> f2 = it->second;

          madness::Future<typename result_type::tile_type> fr =
              world_.taskq.add(probe, &ProbeOp::right_op, f1, f2, attr_);

          result.insert(i, fr);
        }

        // No tasks are generated where neither the left nor the right tiles exits.

        return result;
      }

    private:
      madness::World& world_;
      tile_op op_;
      madness::TaskAttributes attr_;
    }; // struct BinaryArrayOp


    /// Array operation

    /// Performs an element wise binary operation (e.g. std::plus<T>,
    /// std::minus<T>) on two annotated tiles. The value type of the different
    /// tiles may be different, but the value types of expression one and two
    /// must be implicitly convertible to the result value type.
    template<typename Arg1, typename Arg2, typename Res>
    struct BinaryArrayOp<Arg1, Arg2, Res, std::multiplies> {
    private:
      BinaryArrayOp();
      typedef BinaryArrayOp<Arg1, Arg2, Res, std::multiplies> BinaryArrayOp_;

    public:
      typedef const Arg1& first_argument_type;  ///< first array argument type.
      typedef const Arg2& second_argument_type; ///< second array argument type.
      typedef Res result_type;                  ///< result array type.
      typedef BinaryTileOp<typename Arg1::tile_type, typename Arg2::tile_type,
          typename result_type::tile_type, std::multiplies> tile_multiplies_op; ///< Tile contraction operation
      typedef BinaryTileOp<typename Arg1::tile_type, typename Arg2::tile_type,
          typename result_type::tile_type, std::plus> tile_plus_op; ///< Tile contraction operation

    private:

      /// Copies of this object are passed to the madness task queue.

      /// This object handles decision making which operations should be run
      /// based on the existence or non-existence of tiles.
      class ProbeOp {
        ProbeOp();
      public:
        ProbeOp(tile_op o) : op_(o) { }

        /// Executes operations for local left hand tile operands.

        /// This function checks for the existence of the right hand operand.
        /// If it exists, the binary operation is performed and the result is
        ///
        typename result_type::tile_type left_op(const typename Arg1::tile_type& t1, const typename Arg2::tile_type& t2) const {
          if(t2.initialized())
            return op_(t1, t2);

          return op_(t1, 0);
        }

        typename result_type::tile_type right_op(bool p1, typename Arg2::tile_type t2) const {
          if(!p1)
            return op_(0, t2);

          return result_type::tile_type();
        }

      private:
        tile_op op_; ///< Tile operation to be executed.
      }; // class ProbeOp

    public:
      /// operation constructor
      /// \arg \c w is a reference to the world object where tasks will be spawned.
      /// \arg \c o is the functor or function that will be used in tile operations.
      /// \arg \c a is the task attributes that will be used when spawning tile, task operations.
      BinaryArrayOp(madness::World& w, tile_op o = tile_op(), madness::TaskAttributes a = madness::TaskAttributes()) :
          world_(w), op_(o), attr_(a)
      { }

      /// Constructs a series of tasks for the given arrays.
      result_type operator ()(first_argument_type a1, second_argument_type a2) {
        // Here we assume that the array tiles have compatible sizes because it
        // is checked in the expression generation functions (if error checking
        // is enabled.
        result_type result(world_, a1.range(), a1.vars(), a1.order());
        ProbeOp probe(result, op_);
        for(typename Arg1::const_iterator it = a1.begin(); it != a1.end(); ++it) {
          const typename Arg1::index_type i = it->first;
          madness::Future<typename Arg1::tile_type> f1 = it->second;
          madness::Future<typename Arg2::tile_type> f2 = a2.find(i)->second;

          madness::Future<typename result_type::tile_type> fr =
              world_.taskq.add(probe, &ProbeOp::left_op, i, f1, f2, attr_);

          result.insert(i, fr);
        }

        for(typename Arg2::const_iterator it = a2.begin(); it != a2.end(); ++it) {
          const typename Arg2::index_type i = it->first;
          madness::Future<bool> f1 = a1.probe(i);
          madness::Future<typename Arg2::tile_type> f2 = it->second;

          madness::Future<typename result_type::tile_type> fr =
              world_.taskq.add(probe, &ProbeOp::right_op, i, f1, f2, attr_);

          result.insert(i, fr);
        }
        return result;
      }

    private:
      madness::World& world_;
      tile_op op_;
      madness::TaskAttributes attr_;
    }; // struct BinaryArrayOp

    /// Unary tile operation

    /// Performs an element wise unary operation on a tile.
    template<typename Arg, typename Res, template <typename> class Op>
    struct UnaryArrayOp {
      typedef Arg& argument_type;
      typedef Res result_type;
      typedef UnaryTileOp<typename Arg::tile_type, typename result_type::tile_type, Op> tile_op;

    private:
      UnaryArrayOp();

    public:
      /// operation constructor
      /// \arg \c w is a reference to the world object where tasks will be spawned.
      /// \arg \c o is the functor or function that will be used in tile operations.
      /// \arg \c a is the task attributes that will be used when spawning tile, task operations.
      UnaryArrayOp(madness::World& w, tile_op o = tile_op(), madness::TaskAttributes a = madness::TaskAttributes()) :
          world_(w), op_(o), attr_(a)
      { }

      /// Constructs a series of tasks for the given arrays.
      result_type operator ()(argument_type a) const {
        typedef typename boost::mpl::if_<boost::is_const<Arg>,
            typename Arg::const_iterator, typename Arg::iterator>::type iterator_type;

        result_type result(world_, a.range(), a.vars(), a.order());
        for(iterator_type it = a.begin(); it == a.end(); ++it)
          result.insert(it->first, world_.taskq.add(op_, it->second));
        return result;
      }

    private:
      madness::World& world_;
      tile_op op_;
      madness::TaskAttributes attr_;
    }; // struct UnaryArrayOp
*/
  } // namespace math

} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_MATH_H__INCLUDED
