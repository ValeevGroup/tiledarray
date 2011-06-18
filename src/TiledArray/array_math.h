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

        ArrayOp array_op(result, left, right, op_);
        world_->taskq.for_each(madness::Range<typename left_array_type::range_type::volume_type>(0,
            result.tiles().volume()), array_op);

        return result;
      }

      template <typename Archive>
      void serialize(const Archive& ar) {
        TA_ASSERT(false, std::runtime_error, "Serialization not allowed.");
      }

    private:

      struct ArrayOp {
        ArrayOp(const ResArray& result, const left_array_type& left, const right_array_type& right, op_type op) :
            world_(& result.get_world()), result_(result), left_(left), right_(right), op_(op)
        { }

        ArrayOp(const ArrayOp& other) :
            world_(other.world_), result_(other.result_), left_(other.left_), right_(other.right_), op_(other.op_)
        { }

        ArrayOp& operator=(const ArrayOp& other) {
          world_ = other.world_;
          result_ = other.result_;
          left_ = other.left_;
          right_ = other.right_;

          return *this;
        }

        bool operator()(typename ResArray::range_type::volume_type i) const {
          if(result_.is_local(i))
            if(! result_.is_zero(i))
              result_.set(i, world_->taskq.add(madness::make_task(op_,
                  left_.array().find(i), right_.array().find(i))));

          return true;
        }

        template <typename Archive>
        void serialize(const Archive& ar) {
          TA_ASSERT(false, std::runtime_error, "Serialization not allowed.");
        }

      private:
        madness::World* world_;
        mutable ResArray result_;
        left_array_type left_;
        right_array_type right_;
        op_type op_;
      }; // struct ArrayOp

      madness::World* world_;
      unsigned int version_;
      Op op_;
    }; // class BinaryOp

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_MATH_H__INCLUDED
