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

      BinaryOp(madness::World& world, unsigned int version, const Op& op = Op()) :
          world_(&world), version_(version + 1), op_(op)
      {}

      BinaryOp(const BinaryOp_& other) :
          world_(other.world_), op_(other.op_)
      {}

      BinaryOp_& operator=(const BinaryOp_& other) {
        world_ = other.world_;
        op_ = other.op_;
        return *this;
      }

      ResArray operator()(const left_array_type& left, const right_array_type& right) {
        TA_ASSERT(left.array().tiling() == right.array().tiling(), std::runtime_error,
            "The tiling of left and right arrays in binary operations must be identical.");

        // Construct the new array
        ResArray result(*world_, left.array().tiling(), left.array(), right.array(),
            version_);

        LeftOp left_op(world_, result, right, op_);
        world_->taskq.for_each(madness::Range<typename left_array_type::const_iterator>(left.begin(), left.end()), left_op);

        RightOp right_op(world_, result, left, op_);
        world_->taskq.for_each(madness::Range<typename right_array_type::const_iterator>(right.begin(), right.end()), right_op);

        return result;
      }

      template <typename Archive>
      void serialize(const Archive& ar) {
        TA_ASSERT(false, std::runtime_error, "Serialization not allowed.");
      }

    private:

      struct LeftOp {
        LeftOp(madness::World* world, const ResArray& result, const right_array_type& right, op_type op) :
            world_(world), result_(result), right_(right), op_(op)
        { }

        LeftOp(const LeftOp& other) :
            world_(other.world_), result_(other.result_), right_(other.right_), op_(other.op_)
        { }

        LeftOp& operator=(const LeftOp& other) {
          world_ = other.world_;
          result_ = other.result_;
          right_ = other.right_;

          return *this;
        }

        bool operator()(typename left_array_type::const_iterator it) const {
          if(! right_.array().zero(it.index()))
            result_.set(it.index(), world_->taskq.add(madness::make_task(op_,
                *it, right_.array().find(it.index()))));
          else
            result_.set(it.index(), world_->taskq.add(madness::make_task(op_,
                *it, typename right_array_type::array_type::value_type())));

          return true;
        }

      private:
        madness::World* world_;
        mutable ResArray result_;
        right_array_type right_;
        op_type op_;
      }; // struct LeftOp

      struct RightOp {
        RightOp(madness::World* world, const ResArray& result, const left_array_type& left, op_type op) :
            world_(world), result_(result), left_(left), op_(op)
        { }

        RightOp(const RightOp& other) :
            world_(other.world_), result_(other.result_), left_(other.left_), op_(other.op_)
        { }

        RightOp& operator=(const RightOp& other) {
          world_ = other.world_;
          result_ = other.result_;
          left_ = other.left_;

          return *this;
        }

        bool operator()(typename right_array_type::const_iterator it) const {
          if(left_.array().zero(it.index()))
            result_.set(it.index(), world_->taskq.add(madness::make_task(op_,
                typename left_array_type::array_type::value_type(), *it)));

          return true;
        }

      private:
        madness::World* world_;
        mutable ResArray result_;
        left_array_type left_;
        op_type op_;
      }; // struct LeftOp

      madness::World* world_;
      unsigned int version_;
      Op op_;
    }; // class BinaryOp

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_MATH_H__INCLUDED
