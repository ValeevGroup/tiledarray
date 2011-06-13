#ifndef TILEDARRAY_ARRAY_MATH_H__INCLUDED
#define TILEDARRAY_ARRAY_MATH_H__INCLUDED

#include <TiledArray/coordinate_system.h>
#include <TiledArray/permutation.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/madness_runtime.h>
#include <world/worldrange.h>
#include <world/make_task.h>
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

    /// Default binary operation for \c Array objects
    template <typename ResArray, typename LeftArray, typename RightArray, typename Op>
    struct BinaryOp {
      typedef BinaryOp<ResArray, LeftArray, RightArray, Op> BinaryOp_;
      typedef expressions::AnnotatedArray<ResArray> result_array_type;
      typedef expressions::AnnotatedArray<LeftArray> left_array_type;
      typedef expressions::AnnotatedArray<RightArray> right_array_type;
      typedef Op op_type;

      BinaryOp(result_array_type& result, const Op& op = Op()) :
          result_(result), op_(op)
      {}

      BinaryOp(const BinaryOp_& other) :
          result_(other.result_), op_(other.op_)
      {}

      BinaryOp_& operator=(const BinaryOp_& other) {
        result_ = other.result_;
        op_ = other.op_;
        return *this;
      }

      madness::Void operator()(const left_array_type& left, const right_array_type& right) {
        TA_ASSERT(left.array().tiling() == right.array().tiling(), std::runtime_error,
            "The tiling of left and right arrays in binary operations must be identical.");

        madness::World& world = result_->array().get_world();
        const unsigned int version = result_->array().pimpl_->version() + 1u;

        if(is_dense_shape(left.array().pimpl_->shape()) || is_dense_shape(right.array().pimpl_->shape())) {
          ResArray(world, left.array().tiling(), version).swap(result_->array());
        } else {

          ResArray(world, left.array().tiling(), left.array().pimpl_->shape().make_shape_map() +
              right.array().pimpl_->shape().make_shape_map(), version).swap(result_->array());
        }

        LeftOp left_op(world, result_, right, op_);
        world.taskq.for_each(madness::Range<typename left_array_type::const_iterator>(left.begin(), left.end()), left_op);

        RightOp right_op(world, result_, left, op_);
        world.taskq.for_each(madness::Range<typename right_array_type::const_iterator>(right.begin(), right.end()), right_op);

        return madness::None;
      }

      template <typename Archive>
      void serialize(const Archive& ar) {
        TA_ASSERT(false, std::runtime_error, "Serialization not allowed.");
      }

    private:

      template <typename lshape, typename rshape>
      static std::vector<typename ResArray::shape_type::array_type::ordinal_index>
      make_sparse_list(const lshape& left_shape, const rshape& right_shape) {
        // Generate the new shape map
        typename ResArray::shape_type::array_type result_shape_map =
            left_shape->make_shape_map() + right_shape->make_shape_map();

        std::vector<typename ResArray::shape_type::array_type::ordinal_index> tiles;
        tiles.reserve(result_shape_map.range().volume());
        for(typename ResArray::shape_type::array_type::ordinal_index i = 0; i != result_shape_map.range().volume(); ++i)
          if(result_shape_map[i] != 0)
            tiles.push_back(i);
      }

      struct LeftOp {
        LeftOp(madness::World& world, result_array_type& result, const right_array_type& right, op_type op) :
            world_(world), result_(&result), right_(&right), op_(op)
        { }

        LeftOp(const LeftOp& other) :
            world_(other.world_), result_(other.result_), right_(other.right_), op_(other.op_)
        { }

        madness::Void operator()(typename left_array_type::const_iterator it) {
          if(! right_->array().zero(it.index()))
            result_->array().set(it.index(), world_.taskq.add(madness::make_task(op_,
                *it, right_->find(it.index()))));
          else
            result_->array().set(it.index(), world_.taskq.add(madness::make_task(op_,
                *it, right_array_type::array_type::value_type())));

          return madness::None;
        }

      private:
        madness::World& world_;
        result_array_type result_;
        const right_array_type* right_;
        op_type op_;
      }; // struct LeftOp

      struct RightOp {
        RightOp(madness::World& world, result_array_type& result, const left_array_type& left, op_type op) :
            world_(world), result_(&result), left_(&left), op_(op)
        { }

        RightOp(const RightOp& other) :
            world_(other.world_), result_(other.result_), left_(other.left_), op_(other.op_)
        { }

        madness::Void operator()(typename right_array_type::const_iterator it) {
          if(left_->array().zero(it.index()))
            result_->array().set(it.index(), world_.taskq.add(madness::make_task(op_,
                left_array_type::array_type::value_type(), *it)));

          return madness::None;
        }

      private:
        madness::World& world_;
        result_array_type* result_;
        const left_array_type* left_;
        op_type op_;
      }; // struct LeftOp

      result_array_type* result_;
      Op op_;
    }; // class BinaryOp

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_MATH_H__INCLUDED
