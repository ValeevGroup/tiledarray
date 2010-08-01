#ifndef TILEDARRAY_ARRAY_MATH_H__INCLUDED
#define TILEDARRAY_ARRAY_MATH_H__INCLUDED

#include <TiledArray/coordinate_system.h>
#include <boost/static_assert.hpp>
#include <functional>

namespace TiledArray {

  template <typename, typename, typename>
  class Array;

  namespace expressions {

    template <typename>
    class AnnotatedArray;
    class VariableList;

  }  // namespace expressions

  namespace math {

    // Forward declarations
    template <typename, typename, typename, template <typename> class>
    class BinaryOp;

    template <typename, typename, template <typename> class>
    class UnaryOp;


    /// Default binary operation for \c Array objects

    /// \tparam TO Output array element type
    /// \tparam CSO Output array coordinate system type
    /// \tparam PO Output array policy type
    /// \tparam TL Left-hand argument array element type
    /// \tparam CSL Left-hand argument array coordinate system type
    /// \tparam PL Left-hand argument array policy type
    /// \tparam TR Right-hand argument array element type
    /// \tparam CSR Right-hand argument array coordinate system type
    /// \tparam PR Right-hand argument array policy type
    /// \tparam Op The binary operation to be performed
    template <typename T, typename CSO, typename PO,
                          typename CSL, typename PL,
                          typename CSR, typename PR,
              template <typename> class Op>
    class BinaryOp<
        expressions::AnnotatedArray<Array<T, CSO, PO> >,
        expressions::AnnotatedArray<Array<T, CSL, PL> >,
        expressions::AnnotatedArray<Array<T, CSR, PR> >,
        Op>
    {
    private:
      BOOST_STATIC_ASSERT( (detail::compatible_coordinate_system<CSO, CSL>::value) );
      BOOST_STATIC_ASSERT( (detail::compatible_coordinate_system<CSO, CSR>::value) );
      BOOST_STATIC_ASSERT( (detail::same_cs_dim<CSO, CSL>::value) );
      BOOST_STATIC_ASSERT( (detail::same_cs_dim<CSO, CSR>::value) );

    public:
      typedef const expressions::AnnotatedArray<Array<T, CSO, PL> >& first_argument_type;
      typedef const expressions::AnnotatedArray<Array<T, CSL, PR> >& second_argument_type;
      typedef expressions::AnnotatedArray<Array<T, CSR, PL> >& result_type;
      typedef expressions::VariableList VarList;

      /// Set \c result to hold the resulting variable list for this operation

      /// The default behavior for this operation is to set copy left into
      /// result.
      /// \param result The result range
      /// \param left The array for the left-hand argument
      /// \param right The array for the right-hand argument
      result_type operator ()(result_type result, first_argument_type left,
          second_argument_type& right) const
      {
        // Todo: Create result variable list and compare it with result variable list.
        // Todo: Spawn task to construct a new tiled range
        // Todo: Spawn task to construct a new pmap
        // Todo: Spawn task to construct a new shape
        // Todo: Spawn task to construct a new tile container
        return result;
      }
    }; // BinaryOp for AnnotatedArray<Array<> >


    /// Default binary operation for \c Array objects

    /// \tparam TO Output array element type
    /// \tparam CSO Output array coordinate system type
    /// \tparam PO Output array policy type
    /// \tparam TL Left-hand argument array element type
    /// \tparam CSL Left-hand argument array coordinate system type
    /// \tparam PL Left-hand argument array policy type
    /// \tparam TR Right-hand argument array element type
    /// \tparam CSR Right-hand argument array coordinate system type
    /// \tparam PR Right-hand argument array policy type
    /// \tparam Op The binary operation to be performed
    template <typename T, typename CSO, typename PO,
                          typename CSL, typename PL,
                          typename CSR, typename PR>
    class BinaryOp<
        expressions::AnnotatedArray<Array<T, CSO, PO> >,
        expressions::AnnotatedArray<Array<T, CSL, PL> >,
        expressions::AnnotatedArray<Array<T, CSR, PR> >,
        std::multiplies>
    {
      // Check that the coordinate systems are compatible.
      BOOST_STATIC_ASSERT( (detail::compatible_coordinate_system<CSO, CSL>::value) );
      BOOST_STATIC_ASSERT( (detail::compatible_coordinate_system<CSO, CSR>::value) );

    public:
      typedef const expressions::AnnotatedArray<Array<T, CSO, PL> >& first_argument_type;
      typedef const expressions::AnnotatedArray<Array<T, CSL, PR> >& second_argument_type;
      typedef expressions::AnnotatedArray<Array<T, CSR, PL> >& result_type;
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

        return result;
      }
    }; // BinaryOp for AnnotatedArray<Array<> > with Op = std::multiplies

    /// Default binary operation for \c Array objects

    /// \tparam TO Output array element type
    /// \tparam CSO Output array coordinate system type
    /// \tparam PO Output array policy type
    /// \tparam TL Left-hand argument array element type
    /// \tparam CSL Left-hand argument array coordinate system type
    /// \tparam PL Left-hand argument array policy type
    /// \tparam TR Right-hand argument array element type
    /// \tparam CSR Right-hand argument array coordinate system type
    /// \tparam PR Right-hand argument array policy type
    /// \tparam Op The binary operation to be performed
    template <typename T, typename CS, typename PR, typename PA, template <typename> class Op>
    class UnaryOp<
        expressions::AnnotatedArray<Array<T, CS, PR> >,
        expressions::AnnotatedArray<Array<T, CS, PA> >,
        Op>
    {
    public:
      typedef const expressions::AnnotatedArray<Array<T, CS, PA> >& argument_type;
      typedef expressions::AnnotatedArray<Array<T, CS, PR> >& result_type;
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
        return result;
      }
    }; // class UnaryOp for AnnotatedArray<Array<> >



  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_MATH_H__INCLUDED
