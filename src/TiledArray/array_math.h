#ifndef TILEDARRAY_ARRAY_MATH_H__INCLUDED
#define TILEDARRAY_ARRAY_MATH_H__INCLUDED

#include <TiledArray/coordinate_system.h>
#include <TiledArray/permutation.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/madness_runtime.h>
#include <functional>

namespace TiledArray {

  template <typename, typename, typename>
  class Array;

  template <typename>
  class DenseShape;
  template <typename>
  class SparseShape;

  namespace expressions {

    template <typename>
    class AnnotatedArray;
    class VariableList;

  }  // namespace expressions

  namespace math {

    // Forward declarations
    template <typename, typename, typename, template <typename> class>
    struct BinaryOp;

    template <typename, typename, template <typename> class>
    struct UnaryOp;


    /// Default binary operation for \c Array objects

    /// \tparam T Element type of the arrays
    /// \tparam CSO Output array coordinate system type
    /// \tparam PO Output array policy type
    /// \tparam CSL Left-hand argument array coordinate system type
    /// \tparam PL Left-hand argument array policy types
    /// \tparam CSR Right-hand argument array coordinate system type
    /// \tparam PR Right-hand argument array policy type
    /// \tparam Op The binary operation to be performed
    template <typename T, typename CSO, typename PO,
                          typename CSL, typename PL,
                          typename CSR, typename PR,
              template <typename> class Op>
    struct BinaryOp<
        expressions::AnnotatedArray<Array<T, CSO, PO> >,
        expressions::AnnotatedArray<Array<T, CSL, PL> >,
        expressions::AnnotatedArray<Array<T, CSR, PR> >,
        Op>
    {
    private:
      TA_STATIC_ASSERT( (detail::compatible_coordinate_system<CSO, CSL>::value) );
      TA_STATIC_ASSERT( (detail::compatible_coordinate_system<CSO, CSR>::value) );
      TA_STATIC_ASSERT( (detail::same_cs_dim<CSO, CSL>::value) );
      TA_STATIC_ASSERT( (detail::same_cs_dim<CSO, CSR>::value) );

      typedef Array<T, CSO, PO> oarray_type;
      typedef Array<T, CSL, PL> larray_type;
      typedef Array<T, CSR, PR> rarray_type;


      typedef BinaryOp<expressions::AnnotatedArray<typename oarray_type::value_type>,
          expressions::AnnotatedArray<typename larray_type::value_type>,
          expressions::AnnotatedArray<typename rarray_type::value_type>, Op> BinaryTileOp;

      typedef expressions::VariableList var_type;

      typedef Permutation<oarray_type::coordinate_system::dim> permD;
      typedef typename oarray_type::tiled_range_type otrange_type;
      typedef typename larray_type::tiled_range_type ltrange_type;
      typedef typename rarray_type::tiled_range_type rtrange_type;
      typedef madness::Future<otrange_type> fut_trange;

      typedef typename oarray_type::pmap_interface_type pmap_type;
      typedef std::shared_ptr<pmap_type> pmap_ptr;
      typedef madness::Future<pmap_ptr> fut_pmap;

      typedef typename oarray_type::shape_type shape_type;
      typedef std::shared_ptr<shape_type> shape_ptr;
      typedef madness::Future<shape_ptr> fut_shape;

      typedef typename oarray_type::container_type container_type;
      typedef madness::Future<container_type> fut_container;


    public:
      typedef const expressions::AnnotatedArray<Array<T, CSO, PO> >& first_argument_type;
      typedef const expressions::AnnotatedArray<Array<T, CSL, PR> >& second_argument_type;
      typedef expressions::AnnotatedArray<Array<T, CSR, PL> >& result_type;
      typedef expressions::VariableList VarList;

    public:
      /// Set \c result to hold the resulting variable list for this operation

      /// This operation performs does the following:
      /// result = Op(left, right);
      /// \param result The result range
      /// \param left The array for the left-hand argument
      /// \param right The array for the right-hand argument
      result_type operator ()(result_type result, first_argument_type left,
          second_argument_type right) const
      {
        TA_ASSERT(left.array().tiling() == right.array().tiling(), std::runtime_error,
            "Left and right ranges must match.");
        TA_ASSERT(left.vars() == right.vars(), std::runtime_error,
            "Left and right variables must match.");

        // Task attributes for all tasks spawned here.
        madness::TaskAttributes attr();
        madness::World& w = result.array().get_world();
        result.array().preassign(world_, left.array().tiling());

        fut_shape shape = make_shape(result.array(), left.array(), right.array());
        madness::Future<madness::Void> tiles_added =
            w.taskq.add(result.array().pimpl_.get(),
                & oarray_type::impl_type::set_shape, shape, attr);

        w.taskq.add(make_tiles, result.array(), left.array(), right.array(), tiles_added, attr);
        return result;
      }

    private:

      shape_ptr make_shape(const oarray_type& result, const larray_type& left, rarray_type& right) const {
        typedef DenseShape<typename larray_type::coordinate_system> ldense_array;
        typedef DenseShape<typename rarray_type::coordinate_system> rdense_array;
        typedef SparseShape<typename larray_type::coordinate_system> lsparse_array;
        typedef SparseShape<typename rarray_type::coordinate_system> rsparse_array;

        // Output shape is determined as follows:
        //
        // D = dense
        // S = sparse
        // P = predicate
        // P* = predicate
        //
        // Note: P and P* are used to indicate predicated shapes with different
        // predicates.
        //
        // +/-| D  S  P  P*
        // ---+--------------
        //  D | D  D  D  D
        //  S | D  S  S  S
        //  P | D  S  P  S
        //  P*| D  S  S  P*
        //
        // There are 4 combinations here that need to be considered: Dense | Dense,
        // Sparse | Sparse, Predicate & Predicate (where the predicates are the
        // same), and Predicate & Predicates (where the predicates are different).
        //

        // Todo: The sparse shape algorithms are very bad and will not scale.
        // At some point this needs to be fixed.
        if(is_dense(left.pimpl_->get_shape()) || is_dense(right.pimpl_->get_shape())) {
          // output shape is dense
          return fut_shape(oarray_type::impl_type::make_shape(result.array().tiles()));

        } else if(is_sparse_shape(left.pimpl_->get_shape()) || is_sparse_shape(right.pimpl_->get_shape())) {
          // output shape is sparse
          std::vector<typename larray_type::ordinal_index> tiles;
          for(typename larray_type::ordinal_index i = 0; i < left.tiles().volume(); ++i)
            if(left.pimpl_->get_shape()->includes(i).get() && right.pimpl_->get_shape()->includes(i).get())
              tiles.push_back(i);
          return fut_shape(oarray_type::impl_type::make_shape(result.array().get_world(),
              result.array().pimpl_->get_pmap(), tiles.begin(), tiles.end()));

        } else if(left.pimpl_->get_shape().type() == right.pimpl_->get_shape()) {
          // Output shape is a predicate shape
          return fut_shape(left.pimpl_->clone());

        } else {
          // Output shape is sparse
          std::vector<typename larray_type::ordinal_index> tiles;
          for(typename larray_type::ordinal_index i = 0; i < left.tiles().volume(); ++i)
            if(left.pimpl_->get_shape()->local_and_includes(i) && right.pimpl_->get_shape()->local_and_includes(i))
              tiles.push_back(i);
          return oarray_type::impl_type::make_shape(result.array().get_world(),
              result.array().pimpl_->get_pmap(), tiles.begin(), tiles.end());
        }
      }

      static void make_tiles(const VarList& vars, const oarray_type& result, const larray_type& left, rarray_type& right, madness::Void) {
        for(typename oarray_type::ordinal_type i = 0; i < result.tiles().volume(); ++i)
          if(result->pimpl_->get_shape()->local_and_includes(i))
            result.get_world().taskq.add(& do_tile_op, vars, i,
                left.remote_find(i), right.remote_find(i));
      }

      static void do_tile_op(const oarray_type& result, const typename oarray_type::ordinal_type& i,
          const typename larray_type::value_type& left, const typename rarray_type::value_type& right, const VarList& vars) {
        BinaryTileOp tile_op;
        typename oarray_type::data_type fut_tile = result.find(i);

        if(left.range().volume() == 0ul) {
          if(right.range().volume() == 0ul) {
            result.pimpl_.set(i);
          } else {
            result.pimpl_.set(i, right);
          }
        } else {
          if(right.range().volume() == 0ul) {
            result.pimpl_.set(i, left);
          } else {
            typename oarray_type::value_type temp;
            tile_op(temp(vars), left(vars), right(vars));
            result.pimpl_.set(i, temp);
          }
        }
      }

    private:
      madness::World* world_;

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
    struct BinaryOp<
        expressions::AnnotatedArray<Array<T, CSO, PO> >,
        expressions::AnnotatedArray<Array<T, CSL, PL> >,
        expressions::AnnotatedArray<Array<T, CSR, PR> >,
        std::multiplies>
    {
      // Check that the coordinate systems are compatible.
      TA_STATIC_ASSERT( (detail::compatible_coordinate_system<CSO, CSL>::value) );
      TA_STATIC_ASSERT( (detail::compatible_coordinate_system<CSO, CSR>::value) );

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
          first_argument_type left, second_argument_type right) const
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
    struct UnaryOp<
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
