#ifndef TILEDARRAY_CONTRACTION_H__INCLUDED
#define TILEDARRAY_CONTRACTION_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/range.h>
#include <TiledArray/bitset.h>
#include <world/array.h>
#include <iterator>
#include <numeric>
#include <functional>

namespace TiledArray {
  namespace math {

    /// A utility class for contraction operations.

    /// \param I The index type
    class Contraction {
    public:
      typedef std::array<std::size_t, 5> packed_size_array;

      Contraction(const expressions::VariableList& left,
          const expressions::VariableList& right)
      {
        typedef std::pair<expressions::VariableList::const_iterator,
            expressions::VariableList::const_iterator> vars_iter_pair;

        vars_iter_pair lvars_common;
        vars_iter_pair rvars_common;
        expressions::detail::find_common(left.begin(), left.end(), right.begin(),
            right.end(), lvars_common, rvars_common);

        left_[0] = std::distance(left.begin(), lvars_common.first);
        left_[1] = std::distance(left.begin(), lvars_common.second);
        left_[2] = left.dim();

        right_[0] = std::distance(right.begin(), rvars_common.first);
        right_[1] = std::distance(right.begin(), rvars_common.second);
        right_[2] = right.dim();

        dim_ = left_[2] - left_[1] + left_[0] + right_[2] - right_[1] + right_[0];
      }

      unsigned int dim() const { return dim_; }

      template <typename LeftArray, typename RightArray>
      packed_size_array pack_arrays(const LeftArray& left, const RightArray& right) {
        TA_ASSERT(left.size() == left_[2]);
        TA_ASSERT(right.size() == right_[2]);

        // Get the left iterator
        typename LeftArray::const_iterator l0 = left.begin();
        typename LeftArray::const_iterator l1 = l0;
        std::advance(l1, left_[0]);
        typename LeftArray::const_iterator l2 = l0;
        std::advance(l2, left_[1]);
        typename LeftArray::const_iterator l3 = left.end();

        // Get the right array iterator boundaries.
        typename RightArray::const_iterator r0 = right.begin();
        typename RightArray::const_iterator r1 = r0;
        std::advance(r1, right_[0]);
        typename RightArray::const_iterator r2 = r0;
        std::advance(r2, right_[1]);
        typename RightArray::const_iterator r3 = right.end();

        // Make accumulator operation object
        std::multiplies<std::size_t> acc_op;

        // Calculate packed dimensions.
        packed_size_array result = {{
            std::accumulate(l0, l1, 1ul, acc_op),
            std::accumulate(l2, l3, 1ul, acc_op),
            std::accumulate(r0, r1, 1ul, acc_op),
            std::accumulate(r2, r3, 1ul, acc_op),
            std::accumulate(l1, l2, 1ul, acc_op)
          }};

        return result;
      }

      /// Contract array

      /// a{{a1...ai},{ai+1...aj-1},{aj...ak}} + b{{b1...bi},{bi+1...bj-1},{bj...bk}}
      /// ==> c{{a1...ai},{b1...bi},{aj...ak},{bj...bk}}
      template <typename ResArray, typename LeftArray, typename RightArray>
      void contract_array(ResArray& res, const LeftArray& left, const RightArray& right) {
        TA_ASSERT(left.size() == left_[2]);
        TA_ASSERT(right.size() == right_[2]);
        TA_ASSERT(res.size() == dim_);

        std::size_t l, r, i = 0;

        for(l = 0; l < left_[0]; ++l) res[i++] = left[l];
        for(r = 0; r < right_[0]; ++r) res[i++] = right[r];
        for(l = left_[1]; l < left_[2]; ++l) res[i++] = left[l];
        for(r = right_[1]; r < right_[2]; ++r) res[i++] = right[r];
      }

      template <typename LeftRange, typename RightRange>
      DynamicRange contract_range(const LeftRange& left, const RightRange& right) {
        TA_ASSERT(left.order() == right.order());
        typename DynamicRange::index start(dim_), finish(dim_);
        contract_array(start, left.start(), right.start());
        contract_array(finish, left.finish(), right.finish());
        return DynamicRange(start, finish, left.order());
      }

      template <typename ResTRange, typename LeftTRange, typename RightTRange>
      void contract_trange(ResTRange& res, const LeftTRange& left, const RightTRange& right) {
        typename ResTRange::Ranges ranges;
        contract_array(ranges, left.data(), right.data());
        res.resize(ranges.begin(), ranges.end());
      }

      // Definition is in contraction_tensor.h
      template <typename LeftTensor, typename RightTensor>
      inline detail::Bitset<> contract_shape(const LeftTensor& left, const RightTensor& right);

    private:


      typedef std::array<std::size_t, 3> pack_boundary_array;

      pack_boundary_array left_;
      pack_boundary_array right_;
      unsigned int dim_;
    }; // class Contraction

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_CONTRACTION_H__INCLUDED
