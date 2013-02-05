#ifndef TILEDARRAY_EXPRESSIONS_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_H__INCLUDED

#include <TiledArray/unary_tensor.h>
#include <TiledArray/binary_tensor.h>
#include <TiledArray/contraction_tensor.h>
#include <TiledArray/functional.h>
#include <TiledArray/tensor_reduce.h>
#include <world/typestuff.h>

namespace TiledArray {
  namespace expressions {

    // Tensor expression factory functions

    /// Tensor expression addition operator

    /// Add \c left and \c right tensor expression to give a new tensor
    /// expression. The variable lists must be the same for the left- and
    /// right-hand arguments, though the order may differ. The tiled range
    /// dimensions must match where the variable list for the two expressions
    /// match.
    /// \f[
    /// result_{i_1, i_2, \dots} =  left_{i_1, i_2, \dots} - right_{i_1, i_2, \dots}
    /// \f]
    /// \tparam LTile The left-hand tensor expression tile type
    /// \tparam RTile The right-hand tensor expression tile type
    /// \param left The left-hand tensor expression
    /// \param right The right-hand tensor expression
    /// \return A tensor expression that is the difference of \c left and \c right
    template <typename LTile, typename RTile>
    inline TensorExpression<typename BinaryOpSelect<std::plus<typename LTile::value_type> >::type::result_type>
    operator+(const TensorExpression<LTile>& left, const TensorExpression<RTile>& right) {
      return make_binary_tensor(left, right, make_binary_tile_op(
          std::plus<typename LTile::value_type>()));
    }

    /// Tensor expression add-scalar operator

    /// \f[
    /// result_{i_1, i_2, \dots} =  left + right_{i_1, i_2, \dots}
    /// \f]
    /// \tparam LValue A numerical type
    /// \tparam RTile The tensor expression tile type
    /// \param left A numerical constant
    /// \param right The value to be subracted from the tensor elements
    /// \return The tensor expression that is the sum of left and right tensor
    /// elements.
    template <typename LValue, typename RTile>
    inline TensorExpression<RTile>
    operator+(const LValue& left, const TensorExpression<RTile>& right) {
      return make_unary_tensor(right, make_unary_tile_op(
          std::bind1st(std::plus<typename TensorExpression<RTile>::value_type::value_type>(), left)));
    }

    /// Tensor expression add-scalar operator

    /// \f[
    /// result_{i_1, i_2, \dots} =  left_{i_1, i_2, \dots} + right
    /// \f]
    /// \tparam LTile The tensor expression tile type
    /// \tparam RValue A numerical type
    /// \param left The tensor expression to be scaled
    /// \param right A numerical constant
    /// \return The tensor expression that is the sum of left tensor elements
    /// and right.
    template <typename LTile, typename RValue>
    inline TensorExpression<LTile>
    operator+(const TensorExpression<LTile>& left, const RValue& right) {
      return make_unary_tensor(left, make_unary_tile_op(
          std::bind2nd(std::plus<typename LTile::value_type>(), right)));
    }

    /// Tensor expression subraction operator

    /// Subtract \c left and \c right tensor expression to give a new tensor
    /// expression. The variable lists must be the same for the left- and
    /// right-hand arguments, though the order may differ. The tiled range
    /// dimensions must match where the variable list for the two expressions
    /// match.
    /// \f[
    /// result_{i_1, i_2, \dots} =  left_{i_1, i_2, \dots} - right_{i_1, i_2, \dots}
    /// \f]
    /// \tparam LTile The tensor expression tile type
    /// \tparam RValue A numerical type
    /// \param left The value to be subracted from the tensor elements
    /// \param right The tensor expression to
    /// \return A tensor expression that is the difference of \c left and
    /// \c right .
    template <typename LTile, typename RTile>
    inline TensorExpression<typename BinaryOpSelect<std::minus<typename LTile::value_type> >::type::result_type>
    operator-(const TensorExpression<LTile>& left, const TensorExpression<RTile>& right) {
      return make_binary_tensor(left, right, make_binary_tile_op(
          std::minus<typename LTile::value_type>()));
    }

    /// Tensor expression subtract-scalar operator

    /// \f[
    /// result_{i_1, i_2, \dots} =  left - right_{i_1, i_2, \dots}
    /// \f]
    /// \tparam LValue A numerical type
    /// \tparam RTile The tensor expression tile type
    /// \param left A numeric constant
    /// \param right The tensor expression
    /// \return The tensor expression that is the difference between the left
    /// and right tensor elements.
    template <typename LValue, typename RTile>
    inline TensorExpression<RTile>
    operator-(const LValue& left, const TensorExpression<RTile>& right) {
      return make_unary_tensor(right, make_unary_tile_op(
          std::bind1st(std::minus<typename RTile::value_type>(), left)));
    }

    /// Tensor expression subtract-scalar operator

    /// \f[
    /// result_{i_1, i_2, \dots} =  left_{i_1, i_2, \dots} - right
    /// \f]
    /// \tparam LTile The tensor expression tile type
    /// \tparam RValue A numerical type
    /// \param left The tensor expression
    /// \param right A numerical constant
    /// \return The tensor expression that is the difference between the left
    /// tensor elements and right.
    template <typename LTile, typename RValue>
    inline TensorExpression<LTile>
    operator-(const TensorExpression<LTile>& left, const RValue& right) {
      return make_unary_tensor(left, make_unary_tile_op(
          std::bind2nd(std::minus<typename LTile::value_type>(), right)));
    }

    /// Tensor expression contraction operator

    /// Contract \c left and \c right tensor expression to give a new tensor
    /// expression. The contracted indices will be the indices that are the
    /// same variables in the left- and right-hand arguments.
    /// \f[
    /// result_{l_1, l_2, \dots, r_1, r_2, \dots} = \sum_{i_1, i_2, \dots}
    ///   left_{l_1, l_2, \dots, i_1, i_2, \dots} \times right_{i_1, i_2, \dots,
    ///   r_1, r_2, \dots}
    /// \f]
    /// \tparam LTile The left-hand tensor expression tile type
    /// \tparam RTile The right-hand tensor expression tile type
    /// \param left The left-hand tensor expression
    /// \param right The right-hand tensor expression
    /// \return A tensor contraction expression
    template <typename LTile, typename RTile>
    inline typename detail::ContractionExp<TensorExpression<LTile>, TensorExpression<RTile> >::type
    operator*(const TensorExpression<LTile>& left, const TensorExpression<RTile>& right) {
      return make_contraction_tensor(left, right);
    }

    /// Tensor expression scaling operator

    /// \f[
    /// result_{i_1, i_2, \dots} =  left \times right_{i_1, i_2, \dots}
    /// \f]
    /// \tparam LValue A numerical type
    /// \tparam RTile The tensor expression tile type
    /// \param left a numerical constant
    /// \param right The tensor expression
    /// \return The right tensor expression that has been scaled by left.
    template <typename LValue, typename RTile>
    inline const TensorExpression<RTile>&
    operator*(const LValue& left, const TensorExpression<RTile>& right) {
      const_cast<TensorExpression<RTile>&>(right).scale(left);
      return right;
    }

    /// Tensor expression scaling operator

    /// \f[
    /// result_{i_1, i_2, \dots} = left_{i_1, i_2, \dots} \times right
    /// \f]
    /// \tparam LTile The tensor expression tile type
    /// \tparam RValue A numerical type
    /// \param left The tensor expression to be scaled
    /// \param right The scaling factor
    /// \return The left tensor expression that has been scaled by right.
    template <typename LTile, typename RValue>
    inline const TensorExpression<LTile>&
    operator*(const TensorExpression<LTile>& left, const RValue& right) {
      const_cast<TensorExpression<LTile>&>(left).scale(right);
      return left;
    }

    /// Tensor expression negate operator

    /// \f[
    /// result_{i_1, i_2, \dots} =  -arg_{i_1, i_2, \dots}
    /// \f]
    /// \tparam Tile The tensor expression tile type
    /// \param arg The tensor expression to be negated
    /// \return A tensor expression that has the negative value of arg.
    template <typename Tile>
    inline const TensorExpression<Tile>&
    operator-(const TensorExpression<Tile>& arg) {
      const_cast<TensorExpression<Tile>&>(arg).scale(-1);
      return arg;
    }

    namespace detail {

      /// Square norm2 reduction operation

      /// Reduction operation taht computes the square of the norm2 of a tensor
      /// expression. This is equal to the dot product of the expression with
      /// itself.
      /// \tparam Tile The tensor expression tile type
      template <typename Exp>
      class square_norm2_op {
      public:
        typedef typename Exp::value_type argument_type; ///< The tile type
        typedef typename argument_type::value_type result_type; ///< The result type
        typedef std::plus<result_type> remote_op_type; ///< Remote reduction operation type

        /// Create a result type object

        /// Initialize a result object for subsequent reductions
        result_type operator()() const {
          return result_type(0);
        }

        /// Reduce two result objects

        /// Add \c arg to \c result .
        /// \param[in,out] result The result object that will be the reduction target
        /// \param[in] arg The argument that will be added to \c result
        void operator()(result_type& result, const result_type& arg) const {
          result += arg;
        }

        /// Square norm2 of a pair of tiles

        /// Contracte \c left and \c right and add the result to \c result.
        /// \param[in,out] result The result object that will be the reduction target
        /// \param[in] left The left-hand tile to be contracted
        /// \param[in] right The right-hand tile to be contracted
        void operator()(result_type& result, const argument_type& first) const {
          result += math::square_norm(first.size(), first.begin());
        }

        /// Square norm2 of two pairs of tiles

        /// Contracte \c left1 with \c right1 and \c left2 with \c right2 ,
        /// and add the two results.
        /// \param[in] left The first left-hand tile to be contracted
        /// \param[in] right The first right-hand tile to be contracted
        /// \param[in] left The second left-hand tile to be contracted
        /// \param[in] right The second right-hand tile to be contracted
        /// \return A tile that contains the sum of the two contractions.
        void operator()(result_type& result, const argument_type& first, const argument_type& second) const {
          result += math::square_norm(first.size(), first.begin())
              + math::square_norm(second.size(), second.begin());
        }

      }; // class square_norm2_op


      /// Dot product reduction operation

      /// Reduction operation that computes the dot product of two tensor
      /// expressions.
      /// \tparam LExp Left tensor expression type
      /// \tparam RExp Right tensor expression type
      template <typename LExp, typename RExp>
      class dot_op {
      public:
        typedef typename LExp::value_type first_argument_type; ///< The left tile type
        typedef typename RExp::value_type second_argument_type; ///< The right tile type
        typedef typename ContractionResult<LExp, RExp>::type::value_type result_type; ///< The result type

        /// Create a result type object

        /// Initialize a result object for subsequent reductions
        result_type operator()() const {
          return result_type(0);
        }

        /// Reduce two result objects

        /// Add \c arg to \c result .
        /// \param[in,out] result The result object that will be the reduction target
        /// \param[in] arg The argument that will be added to \c result
        void operator()(result_type& result, const result_type& arg) const {
          result += arg;
        }

        /// Dot product of a pair of tiles

        /// Contracte \c left and \c right and add the result to \c result.
        /// \param[in,out] result The result object that will be the reduction target
        /// \param[in] left The left-hand tile to be contracted
        /// \param[in] right The right-hand tile to be contracted
        void operator()(result_type& result, const first_argument_type& first, const second_argument_type& second) const {
          TA_ASSERT(first.range() == second.range());
          result += math::dot(first.size(), first.begin(), second.begin());
        }

        /// Dot product of two pairs of tiles

        /// Contracte \c left1 with \c right1 and \c left2 with \c right2 ,
        /// and add the two results.
        /// \param[in] left The first left-hand tile to be contracted
        /// \param[in] right The first right-hand tile to be contracted
        /// \param[in] left The second left-hand tile to be contracted
        /// \param[in] right The second right-hand tile to be contracted
        /// \return A tile that contains the sum of the two contractions.
        result_type operator()(const first_argument_type& first1, const second_argument_type& second1,
            const first_argument_type& first2, const second_argument_type& second2) const {
          TA_ASSERT(first1.range() == second1.range());
          TA_ASSERT(first2.range() == second2.range());

          result_type result = math::dot(first1.size(), first1.begin(), second1.begin())
              + math::dot(first2.size(), first2.begin(), second2.begin());

          return result;
        }

      }; // class dot_reduce_op

    } // namespace detail


    /// Reduce a \c TensorExpression

    /// This function will reduce all elements of \c arg . The result of the
    /// reduction is returned on all nodes. The function will block, until the
    /// reduction is complete, but it will continue to process tasks while
    /// waiting. Reduction operation objects must have the following definition:
    /// \code
    /// class reduction_op {
    /// public:
    ///   typedef ... argument_type; // tile type
    ///   typedef ... result_type; // result type of the reduction operation
    ///   typedef ... remote_op_type; // remote reduction operation type (e.g. std::plus<result_type>)
    ///
    ///   // Default construction of a result object
    ///   result_type operator()() const { ... }
    ///
    ///   // Reduce two result objects
    ///   void operator()(result_type& result, const result_type& arg) const { ... }
    ///
    ///   // Reduce an argument object to a result object
    ///   void operator()(result_type& result, const argument_type& first) const { ... }
    ///
    ///   // Reduce a two arguments to a single result object
    ///   void operator()(result_type& result, const argument_type& first, const argument_type& second) const { ... }
    /// }; // class reduction_op
    /// \endcode
    /// \tparam Tile The tensor expression tile type
    /// \tparam Op The reduction operation type
    /// \param arg The tensor expression object to be reduced
    /// \param op The reduction operation
    /// \return The reduced value of the tensor.
    template <typename Tile, typename Op>
    inline typename madness::detail::result_of<Op>::type
    reduce(const TensorExpression<Tile>& arg, const Op& op) {

      // Spawn a task that will generate reduction tasks for each local tile
      ReduceTensorExpression<TensorExpression<Tile>, Op>* reduce_task =
          new ReduceTensorExpression<TensorExpression<Tile>, Op>(arg, op);

      // Spawn the task
      madness::Future<typename madness::detail::result_of<Op>::type> local_result =
          reduce_task->result();
      arg.get_world().taskq.add(reduce_task);

      // Wait for the local reduction result
      typename madness::detail::result_of<Op>::type result = local_result.get();

      // All to all global reduction
      arg.get_world().gop.reduce(& result, 1, typename Op::remote_op_type());

      return result;
    }

    /// Element-wise tensor multiplication

    /// \f[
    /// C_{i_1, i_2, \dots} = left_{i_1, i_2, \dots} right_{i_1, i_2, \dots}
    /// \f]
    /// The tiled ranges of the tensor expressions, \c left and \c right , must
    /// be identical, and the variable lists for the expressions must contain
    /// the same set of variables, though the order of the variables may differ.
    /// \tparam LExp The left-hand tensor expression type
    /// \tparam RExp The right-hand tensor expression type
    /// \param left The left-hand tensor expression
    /// \param right The right-hand tensor expression
    /// \return A result tensor expression
    template <typename LTile, typename RTile>
    inline TensorExpression<typename BinaryOpSelect<std::multiplies<typename LTile::value_type> >::type::result_type>
    multiply(const TensorExpression<LTile>& left, const TensorExpression<RTile>& right) {
      return make_binary_tensor(left, right, make_binary_tile_op(
          std::multiplies<typename LTile::value_type>()));
    }

    /// Calculate the dot product of two tensor expressions

    /// \f[
    /// C = \sum_{i_1, i_2, \dots}  A_{i_1, i_2, \dots} B_{i_1, i_2, \dots}
    /// \f]
    /// \tparam LExp Left-hand tensor expression type
    /// \tparam RExp Right-hand tensor expression type
    /// \param left The left tensor argument ( \c A )
    /// \param right The right tensor argument ( \c B )
    /// \return The sum of the products of each element in \c left and \c right ( \c C )
    template <typename LTile, typename RTile>
    inline typename ReduceTensorExpressionPair<TensorExpression<LTile>,
        TensorExpression<RTile>, detail::dot_op<TensorExpression<LTile>, TensorExpression<RTile> > >::result_type
    dot(const TensorExpression<LTile>& left, const TensorExpression<RTile>& right) {
      // Typedefs
      typedef detail::dot_op<TensorExpression<LTile>,TensorExpression<RTile> > dot_op_type;
      typedef ReduceTensorExpressionPair<TensorExpression<LTile>, TensorExpression<RTile>, dot_op_type> dot_task_type;

      // Construct the dot task
      dot_task_type* dot_task = new dot_task_type(left, right, dot_op_type());
      madness::Future<typename dot_task_type::result_type>
          local_result = dot_task->result();

      // Submit the dot task
      left.get_world().taskq.add(dot_task);

      // Get the result
      typename dot_task_type::result_type result = local_result.get();

      // All reduce the result
      left.get_world().gop.sum(&result, 1);
      return result;
    }

    /// Compute the norm2 of \c arg

    /// 2-norm:
    /// \f[
    /// ||arg||_2 = \sqrt{\sum_{i_1, i_2, \dots} (arg_{i_1, i_2, \dots})^2 }
    /// \f]
    /// This function will compute the norm2 of the tensor expression, \c arg ,
    /// across all nodes. The function will block, until the computation is
    /// complete, but it will continue to process tasks while waiting. The same
    /// result is returned on all nodes.
    /// \tparam Exp Tensor expression type
    /// \param arg The tensor expression
    /// \return The norm2 of the tensor
    template <typename Exp>
    inline typename madness::enable_if<is_tensor_expression<Exp>,
        typename Exp::value_type::value_type>::type
    norm2(const Exp& arg) {
      return std::sqrt(reduce(arg, detail::square_norm2_op<Exp>()));
    }



  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_H__INCLUDED
