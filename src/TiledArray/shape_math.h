#ifndef TILEDARRAY_SHAPE_MATH_H__INCLUDED
#define TILEDARRAY_SHAPE_MATH_H__INCLUDED

#include <TiledArray/dense_shape.h>
#include <TiledArray/pred_shape.h>
#include <TiledArray/sparse_shape.h>
#include <TiledArray/tile_math.h>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/make_shared.hpp>
#include <algorithm>

namespace TiledArray {
  namespace expressions {
    class VariableList;
  } // namespace expressions
  namespace math {
/*

    template<typename I, template <typename> class Op >
    struct MakeShapePolicy {
      typedef madness::WorldDCPmapInterface<I> pmap_type;

      detail::ShapeType select_shape_type(const boost::shared_ptr<Shape<I> >& lshape,
          const boost::shared_ptr<Shape<I> >& rshape)
      {
        if(lshape->type() == detail::dense_shape)
          return rshape->type();
        else if(rshape->type() == detail::dense_shape)
          return lshape->type();
        else
          return detail::sparse_shape;
      }

      static boost::shared_ptr<SparseShape<I> >
      add_tiles(madness::World& world, const boost::shared_ptr<pmap_type> pmap,
          const boost::shared_ptr<Shape<I> >& lshape, const expressions::VariableList& lvars,
          const boost::shared_ptr<Shape<I> >& rshape, const expressions::VariableList& rvars)
      {
        TA_ASSERT(std::lexicographical_compare(lshape.start().begin(), lshape.start().end(),
            rshape.start().begin(), rshape.start().end()), std::runtime_error,
            "Shape start indices do not match.");
        TA_ASSERT(std::lexicographical_compare(lshape.finish().begin(), lshape.finish().end(),
            rshape.finish().begin(), rshape.finish().end()), std::runtime_error,
            "Shape finish indices do not match.");

        SparseShape<I>* result =
            new SparseShape<I>(world, lshape->start(), lshape->finish(), pmap);
        for(I i = 0; i < result->volume(); ++i) {
          if(result->is_local(i))
              result->add(i, lshape->includes(i), rshape->includes(i), std::logical_or<I>());
        }
      }
    }; // struct MakeShapePolicy

    template<typename I>
    struct MakeShapePolicy<I, std::multiplies> {
      typedef madness::WorldDCPmapInterface<I> pmap_type;

      detail::ShapeType select_shape_type(const boost::shared_ptr<Shape<I> >& lshape,
          const boost::shared_ptr<Shape<I> >& rshape)
      {
        if(lshape->type() == detail::dense_shape && rshape->type() == detail::dense_shape)
          return detail::dense_shape;
        else
          return detail::sparse_shape;
      }

      static boost::shared_ptr<SparseShape<I> >
      add_tiles(madness::World& world, const boost::shared_ptr<pmap_type> pmap,
          const boost::shared_ptr<Shape<I> >& lshape, const expressions::VariableList& lvars,
          const boost::shared_ptr<Shape<I> >& rshape, const expressions::VariableList& rvars)
      {
        boost::shared_array<unsigned int> a = create_int_array(lshape);
        boost::shared_array<unsigned int> b = create_int_array(rshape);

        ContractedData<std::size_t> result_data(lshape.start(), lshape.finish(),
            lvars, rshape.start(), lshape.finish(), rvars, lshape.order());

        SparseShape<I>* result =
            new SparseShape<I>(world, result_data.start(), result_data.finish(), pmap);

        boost::shared_array<unsigned int> c(new unsigned int[result->volume()]);
        std::fill(c.get(), c.get() + result->volume(), 0u);

        contract(result_data, a.get(), b.get(), c.get());

        for(I i = 0; i < result->volume(); ++i) {
          if(result->is_local(i) && c[i] != 0)
              result->add(i);
        }
      }

    private:
      static boost::shared_array<unsigned int> create_int_array(const boost::shared_ptr<Shape<I> >& shape) {
        const typename Shape<I>::volume_type vol = shape->volume;
        boost::shared_array<unsigned int> result(new int[vol]);
        std::fill(result.get(), result.get() + vol, 0u);
        for(I i = 0; i < vol; ++i)
          if(shape->includes(i).get())
            result[i] = 1;

        return result;
      }
    }; // struct MakeShapePolicy<I, std::multiplies>


    template<typename I, template <typename> class Op >
    class BinaryShapeOp {
      BinaryShapeOp();
    public:
      typedef madness::WorldDCPmapInterface<I> pmap_type;

      BinaryShapeOp(madness::World& w, const boost::shared_ptr<pmap_type>& pm) :
          world_(w), pmap_(pm)
      {  }

      boost::shared_ptr<Shape<I> > operator()(const boost::shared_ptr<Shape<I> >& lshape,
          const expressions::VariableList& lvars, const boost::shared_ptr<Shape<I> >& rshape,
          const expressions::VariableList& rvars)
      {
        TA_ASSERT(lshape.order() == rshape.order(), std::runtime_error,
            "Shape dimension orders do not match.");


      }

    private:



      template<typename Index>
      static boost::shared_ptr<DenseShape<I> > make_dense(const Index& start, const Index& finish, detail::DimensionOrderType o) {
        return boost::make_shared<DenseShape<I> >(start, finish, o);
      }

      template<typename Index>
      static boost::shared_ptr<PredShape<I> > make_predicate(const Index& start,
          const Index& finish, detail::DimensionOrderType o,
          boost::shared_ptr<typename PredShape<I>::PredInterface> pred)
      {
        boost::make_shared<PredShape<I> >(start, finish, o, pred);
      }


      template<typename Index>
      boost::shared_ptr<DenseShape<I> > make_sparse(const Index& start,
          const Index& finish, detail::DimensionOrderType o,
          const boost::shared_ptr<Shape<I> >& s1, const boost::shared_ptr<Shape<I> >& s2)
      {
        boost::shared_ptr<SparseShape<I> > sparse_shape =
            boost::make_shared<SparseShape<I> >(world_, start, finish, o, pmap_);
        Op<I> logic();
        for(I i = 0; i < sparse_shape->volume(); ++i) {
          if(sparse_shape->is_local(i))
              sparse_shape->add(i, s1->includes(i), s2->includes(i), logic);
        }
        return boost::dynamic_pointer_cast<Shape<I> >(sparse_shape);
      }

      /// Returns a copy of the shapes predicate.
      static boost::shared_ptr<typename PredShape<I>::PredInterface> get_pred(const boost::shared_ptr<Shape<I> >& s) {
        TA_ASSERT(s.type() == detail::predicated_shape, std::runtime_error,
            "Shape is not a predicate shape.");
        boost::shared_ptr<PredShape<I> > pred_shape =
            boost::dynamic_pointer_cast<PredShape<I> >(s);
        TA_ASSERT(pred_shape.get() != NULL, std::runtime_error,
            "Dynamic pointer cast failed.");
        return pred_shape->clone_predicate();
      }


      template<typename OtherShape>
      static boost::shared_ptr<Shape<I> > cast_shape(const boost::shared_ptr<OtherShape>& other) {
        return boost::dynamic_pointer_cast<Shape<I> >(other);
      }

      madness::World& world_;
      const boost::shared_ptr<pmap_type> pmap_;
    }; // struct ShapeOp
*/
  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_SHAPE_MATH_H__INCLUDED
