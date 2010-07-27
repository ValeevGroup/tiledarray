#ifndef TILEDARRAY_RANGE_MATH_H__INCLUDED
#define TILEDARRAY_RANGE_MATH_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/variable_list.h>
#include <boost/static_assert.hpp>

namespace TiledArray {

  template <typename>
  class Range;

  namespace expressions {

    template <typename>
    class AnnotatedArray;

  } // namespace expressions

  namespace math {

    // Forward declarations
    template <typename, typename, typename, template <typename> class>
    class BinaryOp;

    template <typename, typename, template <typename> class>
    class UnaryOp;

    /// Default binary operation for \c Range objects

    /// \tparam ArrayType The array type of the annotated array objects
    template <typename ArrayType, template <typename> class Op>
    class BinaryOp<Range<typename ArrayType::coordinate_system>,
        expressions::AnnotatedArray<ArrayType>, expressions::AnnotatedArray<ArrayType>, Op>
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
      /// \param left The range for the left-hand argument
      /// \param right The range for the right-hand argument
      /// \throw std::runtime_error When \c left range is not equal to \c right
      /// range.
      /// \throw std::runtime_error When \c left variable list is not equal to
      /// the \c right variable list.
      /// \throw std::runtime_error When \c result_vars is not equal to the
      /// argument variable lists.
      result_type operator ()(result_type result, const VarList& result_vars,
          first_argument_type left, second_argument_type& right) const
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
    template <typename CS, typename LeftArrayType, typename RightArrayType>
    class BinaryOp<Range<CS>,
        expressions::AnnotatedArray<LeftArrayType>, expressions::AnnotatedArray<RightArrayType>, std::multiplies>
    {
      // Check that the coordinate systems are compatible.
      BOOST_STATIC_ASSERT( (detail::compatible_coordinate_system<LeftArrayType, RightArrayType>::value) );

    public:
      typedef const expressions::AnnotatedArray<LeftArrayType>& first_argument_type;
      typedef const expressions::AnnotatedArray<RightArrayType>& second_argument_type;
      typedef Range<CS>& result_type;
      typedef expressions::VariableList VarList;

      /// Set \c result to hold the resulting variable list for this operation

      /// The default behavior for this operation is to set copy left into
      /// result.
      /// \param result The result range
      /// \param result_vars The result variable list
      /// \param left The annotated array for the left-hand argument
      /// \param right The annotated array for the right-hand argument
      result_type operator ()(result_type result, const VarList& result_vars,
          first_argument_type left, second_argument_type& right) const
      {
        typename Range<CS>::index start;
        typename Range<CS>::index finish;

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

    template <typename ArrayType, template <typename> class Op>
    class UnaryOp<Range<typename ArrayType::coordinate_system>, expressions::AnnotatedArray<ArrayType>, Op>
    {
    public:
      typedef const expressions::AnnotatedArray<ArrayType>& argument_type;
      typedef Range<typename ArrayType::coordinate_system>& result_type;
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

  } // namespace math
} // namespace TiledArray


#endif // TILEDARRAY_RANGE_MATH_H__INCLUDED
