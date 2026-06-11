/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  tile_op/batched_contract_reduce.h
 *  Jun 11, 2026
 *
 */

#ifndef TILEDARRAY_TILE_OP_BATCHED_CONTRACT_REDUCE_H__INCLUDED
#define TILEDARRAY_TILE_OP_BATCHED_CONTRACT_REDUCE_H__INCLUDED

#include <TiledArray/tile_op/contract_reduce.h>

namespace TiledArray {
namespace detail {

/// Batched (fused-index) tile contract/reduce operation

/// Adapts a ContractReduce op built for the *folded* (fused-mode-free) tile
/// shapes to tiles that carry \c nfused leading fused (Hadamard/batch) modes
/// on both arguments and the result, i.e. evaluates
/// \f$ C_{h,e_A,e_B} \mathrel{+}= \sum_c A_{h,e_A,c} B_{h,c,e_B} \f$
/// per tile. The fused modes are folded into the tile batch dimension by a
/// zero-copy reshape (they are leading, so the fold preserves layout) and the
/// per-batch GEMM loop of \c Tensor::gemm does the rest; the accumulated
/// result tile is allocated with its full (h, e_A, e_B) range up front, so no
/// unfold is ever needed.
///
/// Models the ReducePairTask op concept (seed / combine / pair-accumulate /
/// finalize). The result permutation must be handled outside this op (the
/// wrapped op must be perm-free).
///
/// \tparam Op the folded-shape contract/reduce op (ContractReduce
///         instantiated with the fused-mode-free ranks)
template <typename Op>
class BatchedContractReduce {
 public:
  typedef BatchedContractReduce<Op> BatchedContractReduce_;
  typedef Op op_type;
  typedef typename Op::result_type result_type;
  typedef typename Op::first_argument_type first_argument_type;
  typedef typename Op::second_argument_type second_argument_type;

 private:
  op_type op_;               ///< The folded-shape contract/reduce op
  unsigned int nfused_ = 0;  ///< The number of leading fused modes

  /// \return the range spanned by modes [nfused_, rank) of \p r, rebased to
  /// zero lobounds (the folded view is a GEMM scratch view; only extents
  /// matter)
  template <typename Range_>
  Range_ fold_range(const Range_& r) const {
    const auto* extent = r.extent_data();
    container::svector<typename Range_::index1_type> extents(extent + nfused_,
                                                             extent + r.rank());
    return Range_(extents);
  }

  /// \return the number of fused elements (product of the extents of the
  /// leading \c nfused_ modes) of \p r
  template <typename Range_>
  std::size_t fused_volume(const Range_& r) const {
    const auto* extent = r.extent_data();
    std::size_t n = 1ul;
    for (unsigned int d = 0u; d < nfused_; ++d) n *= extent[d];
    return n;
  }

 public:
  BatchedContractReduce() = default;
  BatchedContractReduce(const BatchedContractReduce_&) = default;
  BatchedContractReduce(BatchedContractReduce_&&) = default;
  ~BatchedContractReduce() = default;
  BatchedContractReduce_& operator=(const BatchedContractReduce_&) = default;
  BatchedContractReduce_& operator=(BatchedContractReduce_&&) = default;

  /// \param op the folded-shape contract/reduce op; must be perm-free (the
  ///        full-range result of this op cannot host the folded-rank result
  ///        permutation)
  /// \param nfused the number of leading fused modes carried by both
  ///        arguments and the result
  BatchedContractReduce(const op_type& op, const unsigned int nfused)
      : op_(op), nfused_(nfused) {
    TA_ASSERT(!op_.perm());
  }

  /// \return the wrapped folded-shape op
  const op_type& op() const { return op_; }
  /// \return the number of leading fused modes
  unsigned int nfused() const { return nfused_; }
  /// \return the GEMM helper of the wrapped (folded-shape) op
  const TiledArray::math::GemmHelper& gemm_helper() const {
    return op_.gemm_helper();
  }

  /// Create a new, empty result object
  result_type operator()() const { return result_type(); }

  /// Post processing step (no result permutation supported)
  result_type operator()(const result_type& temp) const {
    using TiledArray::empty;
    TA_ASSERT(!empty(temp));
    return temp;
  }

  /// Reduce two result objects (both carry the full fused range)
  void operator()(result_type& result, const result_type& arg) const {
    op_(result, arg);
  }

  /// Contract a pair of fused-mode-carrying tiles and add to the target tile
  void operator()(result_type& result, const first_argument_type& left,
                  const second_argument_type& right) const {
    // the fold relies on TA::Tensor's zero-copy reshape + batched GEMM
    constexpr bool supported_tiles =
        TiledArray::detail::is_ta_tensor_v<result_type> &&
        TiledArray::detail::is_ta_tensor_v<
            std::remove_cv_t<std::remove_reference_t<first_argument_type>>> &&
        TiledArray::detail::is_ta_tensor_v<
            std::remove_cv_t<std::remove_reference_t<second_argument_type>>>;
    if constexpr (!supported_tiles) {
      TA_EXCEPTION(
          "BatchedContractReduce supports only TiledArray::Tensor tiles");
    } else {
      contract_pair(result, left, right);
    }
  }

 private:
  /// The TA::Tensor implementation of the pair contraction
  void contract_pair(result_type& result, const first_argument_type& left,
                     const second_argument_type& right) const {
    using TiledArray::empty;
    if (empty(left) || empty(right)) return;

    const auto& gh = op_.gemm_helper();
    const unsigned int nc = gh.num_contract_ranks();
    const unsigned int neA = gh.left_rank() - nc;
    const unsigned int neB = gh.right_rank() - nc;

    // both args must carry the fused modes as their leading modes, with
    // equal extents
    TA_ASSERT(left.range().rank() == nfused_ + neA + nc);
    TA_ASSERT(right.range().rank() == nfused_ + nc + neB);
    TA_ASSERT(left.nbatch() == 1ul);
    TA_ASSERT(right.nbatch() == 1ul);
    const std::size_t batch = fused_volume(left.range());
    TA_ASSERT(batch == fused_volume(right.range()));

    // folded, zero-copy argument views
    auto left_folded = left.reshape(fold_range(left.range()), batch);
    auto right_folded = right.reshape(fold_range(right.range()), batch);

    if (empty(result)) {
      // let the wrapped op allocate (and zero- or beta-0-initialize) the
      // result in *folded* form -- this also engages its tile-type-specific
      // result construction (e.g. the arena reserve for tensor-of-tensor
      // tiles) -- then unfold by a zero-copy reshape: the data layout of the
      // folded (range = (e_A, e_B), nbatch = batch) result coincides with
      // the full (h, e_A, e_B) row-major layout because the fused modes
      // lead. The full bounds: fused + left-external from the left tile,
      // right-external from the right tile.
      result_type result_folded;
      op_(result_folded, left_folded, right_folded);

      using index1_type = typename result_type::range_type::index1_type;
      container::svector<index1_type> lobounds, upbounds;
      lobounds.reserve(nfused_ + neA + neB);
      upbounds.reserve(nfused_ + neA + neB);
      for (unsigned int d = 0u; d < nfused_ + neA; ++d) {
        lobounds.push_back(left.range().lobound_data()[d]);
        upbounds.push_back(left.range().upbound_data()[d]);
      }
      for (unsigned int d = nfused_ + nc; d < nfused_ + nc + neB; ++d) {
        lobounds.push_back(right.range().lobound_data()[d]);
        upbounds.push_back(right.range().upbound_data()[d]);
      }
      result = result_folded.reshape(
          typename result_type::range_type(lobounds, upbounds));
    } else {
      // accumulate through a folded, zero-copy view of the result
      auto result_folded = result.reshape(fold_range(result.range()), batch);
      op_(result_folded, left_folded, right_folded);
    }
  }

};  // class BatchedContractReduce

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_OP_BATCHED_CONTRACT_REDUCE_H__INCLUDED
