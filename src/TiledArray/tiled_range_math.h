#ifndef TILEDARRAY_TILED_RANGE_MATH_H__INCLUDED
#define TILEDARRAY_TILED_RANGE_MATH_H__INCLUDED

#include <TiledArray/error.h>

namespace TiledArray {

  template <typename>
  class TiledRange;
  template <typename, typename, typename>
  class Array;

  namespace expressions {

    template <typename>
    class AnnotatedArray;

  } // namespace expressions

  namespace math {


    /// Default binary operation for \c TiledRange objects

    /// \tparam ArrayType The array type of the annotated array objects
    template <typename T, typename CS, typename C, template <typename> class Op>
    class BinaryOp<TiledRange<CS>, expressions::AnnotatedArray<Array<T, CS, C> >,
    expressions::AnnotatedArray<Array<T, CS, C> >, Op>
    {
    public:
      typedef const expressions::AnnotatedArray<Array<T, CS, C> >& first_argument_type;
      typedef const expressions::AnnotatedArray<Array<T, CS, C> >& second_argument_type;
      typedef TiledRange<CS>& result_type;
      typedef expressions::VariableList VarList;

      /// Set \c result to hold the resulting variable list for this operation

      /// The default behavior for this operation is to set copy left into
      /// result.
      /// \param result The result range
      /// \param result_vars The result variable list
      /// \param left The range for the left-hand argument
      /// \param right The range for the right-hand argument
      /// \throw std::runtime_error When \c left range is not equal to \c right
      /// range.
      /// \throw std::runtime_error When \c left variable list is not equal to
      /// the \c right variable list.
      /// \throw std::runtime_error When \c result_vars is not equal to the
      /// argument variable lists.
      result_type operator ()(result_type result, const VarList& result_vars,
          first_argument_type left, second_argument_type right) const
      {
        TA_ASSERT(left.range() == right.range(), std::runtime_error,
            "Left and right ranges must match.");
        TA_ASSERT(left.vars() == right.vars(), std::runtime_error,
            "Left and right variables must match.");
        TA_ASSERT(result_vars == left.vars(), std::runtime_error,
            "Result variables must match the argument variables.");
        result = left.range();
        return result;
      }
    }; // BinaryOp<Range<typename ArrayType::coordinate_system>, expressions::AnnotatedArray<ArrayType>, expressions::AnnotatedArray<ArrayType>, Op>


    /// Contraction operation for \c Range objects

    /// \tparam ArrayType The array type of the annotated array objects
    template <typename ArrayType>
    class BinaryOp<Range<typename ArrayType::coordinate_system>,
    expressions::AnnotatedArray<ArrayType>, expressions::AnnotatedArray<ArrayType>, std::multiplies>
    {
    public:
      typedef const expressions::AnnotatedArray<ArrayType>& first_argument_type;
      typedef const expressions::AnnotatedArray<ArrayType>& second_argument_type;
      typedef Range<typename ArrayType::coordinate_system>& result_type;
      typedef expressions::VariableList VarList;

      /// Set \c result to hold the resulting variable list for this operation

      /// The default behavior for this operation is to set copy left into
      /// result.
      /// \param result The result range
      /// \param result_vars The result variable list
      /// \param left The annotated array for the left-hand argument
      /// \param right The annotated array for the right-hand argument
      result_type operator ()(result_type result, const VarList& result_vars,
          first_argument_type left, second_argument_type right) const
      {
        typename Range<typename ArrayType::coordinate_system>::index start;
        typename Range<typename ArrayType::coordinate_system>::index finish;

        std::size_t left_it = 0;
        std::size_t right_it = 0;
        for(std::size_t it = 0; it != result_vars.dim(); ++it) {
          if(left.vars()[left_it] == result_vars[it]) {
            start[it] = left.range().start()[left_it];
            finish[it] = left.range().finish()[left_it];
            ++left_it;
          } else { // *right_it == *it
            start[it] = right.range().start()[right_it];
            finish[it] = right.range().finish()[right_it];
            ++right_it;
          }
        }

        result.resize(start, finish);
        return result;
      }
    }; // BinaryOp<Range<typename ArrayType::coordinate_system>, expressions::AnnotatedArray<ArrayType>, expressions::AnnotatedArray<ArrayType>, std::multiplies>

    template <typename T, typename CS, typename C, template <typename> class Op>
    class UnaryOp<TiledRange<CS>, expressions::AnnotatedArray<Array<T, CS, C> >, Op> {
    public:
      typedef const expressions::AnnotatedArray<Array<T, CS, C> >& argument_type;
      typedef TiledRange<CS>& result_type;
      typedef expressions::VariableList VarList;


      /// Set \c result to hold the resulting variable list for this operation

      /// The default behavior for this operation is to set copy left into
      /// result.
      /// \param result The result range
      /// \param result_vars The result variable list
      /// \param arg The annotated array argument
      /// \throw std::runtime_error When \c result_vars is not equal to the
      /// \c arg variables.
      expressions::VariableList& operator ()(result_type result,
          const VarList& result_vars, argument_type arg) const
      {
        TA_ASSERT(result_vars == arg.vars(), std::runtime_error,
            "Result variables must match the argument variables.");
        result = arg;
        return result;
      }
    }; // class UnaryOp<expressions::VariableList, Op>

  }  // namespace math
}  // namespace TiledArray
#endif // TILEDARRAY_TILED_RANGE_MATH_H__INCLUDED
