#ifndef TILEDARRAY_ARRAY_MATH_H__INCLUDED
#define TILEDARRAY_ARRAY_MATH_H__INCLUDED

#include <TiledArray/coordinate_system.h>
#include <TiledArray/permutation.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/math.h>
#include <TiledArray/reduce_task.h>
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
      typedef Op op_type;
      typedef BinaryOp<ResArray, LeftArray, RightArray, Op> BinaryOp_;
      typedef expressions::AnnotatedArray<ResArray> result_array_type;
      typedef expressions::AnnotatedArray<LeftArray> left_array_type;
      typedef expressions::AnnotatedArray<RightArray> right_array_type;

      BinaryOp(madness::World& world, unsigned int version, const Op& op) :
          world_(&world), version_(version + 1), op_(op)
      {}

      BinaryOp(const BinaryOp_& other) :
          world_(other.world_), version_(other.version_), op_(other.op_)
      {}

      BinaryOp_& operator=(const BinaryOp_& other) {
        world_ = other.world_;
        version_ = other.version_;
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

        bool operator()(typename ResArray::ordinal_index i) const {
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

    /// Default binary operation for \c Array objects
    template <typename ResArray, typename LeftArray, typename RightArray>
    struct BinaryOp<ResArray, LeftArray, RightArray, TileContract<typename ResArray::value_type,
        typename LeftArray::value_type, typename RightArray::value_type> >
    {
      typedef TileContract<typename ResArray::value_type, typename LeftArray::value_type,
          typename RightArray::value_type> contraction_op_type;
      typedef std::plus<typename ResArray::value_type> addtion_op_type;
      typedef BinaryOp<ResArray, LeftArray, RightArray, contraction_op_type> BinaryOp_;
      typedef expressions::AnnotatedArray<ResArray> result_array_type;
      typedef expressions::AnnotatedArray<LeftArray> left_array_type;
      typedef expressions::AnnotatedArray<RightArray> right_array_type;
      typedef Contraction<typename ResArray::ordinal_index> cont_type;
      typedef typename ResArray::ordinal_index ordinal_index;

    private:
      template <unsigned int DIM, typename CS>
      struct packed_coordinate_system {
        typedef CoordinateSystem<DIM, CS::level, CS::order,
            typename CS::ordinal_index> coordinate_system;
      };

      typedef typename packed_coordinate_system<4,
          typename ResArray::coordinate_system>::coordinate_system res_packed_cs;
      typedef typename packed_coordinate_system<3,
          typename ResArray::coordinate_system>::coordinate_system left_packed_cs;
      typedef typename packed_coordinate_system<3,
          typename ResArray::coordinate_system>::coordinate_system right_packed_cs;

    public:

      BinaryOp(madness::World& world, unsigned int version) :
          world_(&world), version_(version + 1)
      {}

      BinaryOp(const BinaryOp_& other) :
          world_(other.world_), version_(other.version_)
      {}

      BinaryOp_& operator=(const BinaryOp_& other) {
        world_ = other.world_;
        version_ = other.version_;
        return *this;
      }

      ResArray operator()(const left_array_type& left, const right_array_type& right) {
        TA_ASSERT(left.array().tiling() == right.array().tiling(), std::runtime_error,
            "The tiling of left and right arrays in binary operations must be identical.");

        std::shared_ptr<cont_type> cont(new cont_type(left.vars(), right.vars()));

        typename ResArray::tiled_range_type tiling;
        cont->contract_trange(tiling, left.array().tiling(), right.array().tiling());

        ResArray result(*world_, tiling, cont, left.array(), right.array(), version_);

        ArrayOp array_op(cont, result, left, right);

        world_->taskq.for_each(array_op.range(), array_op);

        return result;
      }

      template <typename Archive>
      void serialize(const Archive& ar) {
        TA_ASSERT(false, std::runtime_error, "Serialization not allowed.");
      }

    private:

      struct ArrayOpImpl {
        typedef typename Range<res_packed_cs>::index res_packed_index;
        typedef typename Range<left_packed_cs>::index left_packed_index;
        typedef typename Range<right_packed_cs>::index right_packed_index;
        typedef madness::Range<typename Range<res_packed_cs>::const_iterator> range_type;

        ArrayOpImpl(const std::shared_ptr<cont_type>& cont, const ResArray& result,
          const left_array_type& left, const right_array_type& right) :
            world_(& result.get_world()),
            contraction_(cont),
            result_(result),
            left_(left.array()),
            right_(right.array()),
            res_range_(),
            left_range_(),
            right_range_()
        {
          // Get packed sizes
          const typename cont_type::packed_size_array size =
              cont->pack_arrays(left_.range().size(), right_.range().size());

          // Set packed range dimensions
          res_range_.resize(res_packed_index(0),
              res_packed_index(size[0], size[2], size[1], size[3]));
          left_range_.resize(left_packed_index(0),
              left_packed_index(size[0], size[4], size[1]));
          right_range_.resize(right_packed_index(0),
              right_packed_index(size[2], size[4], size[3]));
        }

        range_type range() const {
          return range_type(res_range_.begin(), res_range_.end());
        }

        void generate_tasks(const typename Range<res_packed_cs>::const_iterator& it) const {
          const ordinal_index res_index = res_range_.ord(*it);

          // Check that the result tile has a value
          if(result_.is_zero(res_index))
            return;

          ordinal_index I = left_range_.size()[1];
          // Reduction objects
          std::vector<ProcessID> reduce_grp;
          reduce_grp.reserve(I);
          detail::ReduceTask<typename ResArray::value_type, addtion_op_type >
              local_reduce_op(*world_, addtion_op_type());

          for(ordinal_index i = 0; i < I; ++i) {
            // Get the a and b index
            const ordinal_index left_index = left_ord(*it, i);
            const ordinal_index right_index = right_ord(*it, i);

            // Check for non-zero contraction.
            if((! left_.is_zero(left_index)) && (! right_.is_zero(right_index))) {

              // Add to the list nodes involved in the reduction reduction group
              reduce_grp.push_back(left_.owner(left_index));

              if(left_.is_local(left_index)) {
                // Do the tile-tile contraction and add to local reduction list
                local_reduce_op.add(world_->taskq.add(madness::make_task(make_cont_op(res_index),
                    left_.find(left_index), right_.find(right_index))));
              }
            }
          }

          // Do tile-contraction reduction
          std::stringstream ss;

          if(local_reduce_op.size() != 0) {
            TA_ASSERT(std::find_if(reduce_grp.begin(), reduce_grp.end(),
                std::bind2nd(std::equal_to<ProcessID>(), world_->rank())) != reduce_grp.end(),
                std::runtime_error, "This node is not in the group to be reduced.");

            madness::Future<typename ResArray::value_type> local_red = local_reduce_op();

            result_.reduce(res_index, local_red, reduce_grp.begin(),
                reduce_grp.end(), addtion_op_type());
          } else {
            TA_ASSERT(std::find_if(reduce_grp.begin(), reduce_grp.end(), std::bind2nd(std::equal_to<ProcessID>(), world_->rank())) == reduce_grp.end(),
                std::runtime_error, "This node is not in the group to be reduced.");
          }
        }

      private:

        /// Contraction operation factory

        /// \param index The ordinal index of the result tile
        /// \retur The contraction operation for \c index
        contraction_op_type make_cont_op(const ordinal_index& index) const {
          return contraction_op_type(contraction_,
              result_.tiling().make_tile_range(index));
        }

        ordinal_index left_ord(const res_packed_index& res_index, ordinal_index i) const {
          const typename Range<left_packed_cs>::size_array& weight = left_range_.weight();
          return res_index[0] * weight[0] + i * weight[1] + res_index[2] * weight[2];
        }

        ordinal_index right_ord(const res_packed_index& res_index, ordinal_index i) const {
          const typename Range<right_packed_cs>::size_array& weight = right_range_.weight();
          return res_index[1] * weight[0] + i * weight[1] + res_index[3] * weight[2];
        }

        madness::World* world_;
        std::shared_ptr<cont_type> contraction_;
        mutable ResArray result_;
        LeftArray left_;
        RightArray right_;
        Range<res_packed_cs> res_range_;
        Range<left_packed_cs> left_range_;
        Range<right_packed_cs> right_range_;
      }; // struct ArrayOpImpl

      struct ArrayOp {

        ArrayOp(const std::shared_ptr<cont_type>& cont, const ResArray& result,
          const left_array_type& left, const right_array_type& right) :
            pimpl_(new ArrayOpImpl(cont, result, left, right))
        { }

        ArrayOp(const ArrayOp& other) :
            pimpl_(other.pimpl_)
        { }

        ArrayOp& operator=(const ArrayOp& other) {
          pimpl_ = other.pimpl_;
          return *this;
        }

        typename ArrayOpImpl::range_type range() const { return pimpl_->range(); }

        bool operator()(const typename Range<res_packed_cs>::const_iterator& it) const {
          pimpl_->generate_tasks(it);
          return true;
        }

        template <typename Archive>
        void serialize(const Archive& ar) {
          TA_ASSERT(false, std::runtime_error, "Serialization not allowed.");
        }

      private:

        std::shared_ptr<ArrayOpImpl> pimpl_;
      };


      madness::World* world_;
      unsigned int version_;
    }; // class BinaryOp

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_MATH_H__INCLUDED
