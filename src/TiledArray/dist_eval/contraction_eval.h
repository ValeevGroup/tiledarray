/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 */

#ifndef TILEDARRAY_DIST_EVAL_CONTRACTION_EVAL_H__INCLUDED
#define TILEDARRAY_DIST_EVAL_CONTRACTION_EVAL_H__INCLUDED

#include <TiledArray/dist_eval/dist_eval.h>
#include <TiledArray/pmap/cyclic_pmap.h>

namespace TiledArray {
  namespace detail {

    template <typename, typename, typename>
    class ContractionEvalImpl;

    /// Contract and reduce operation

    /// This object handles contraction and reduction of tensor tiles.
    template <typename Left, typename Right, typename Op>
    class ContractReduceOp {
    public:
      typedef typename ContractionEvalImpl<Left, Right, Op>::left_value_type first_argument_type; ///< The left tile type
      typedef typename ContractionEvalImpl<Left, Right, Op>::right_value_type second_argument_type; ///< The right tile type
      typedef typename ContractionEvalImpl<Left, Right, Op>::value_type result_type; ///< The result tile type.

      /// Construct contract/reduce functor

      /// \param cont Shared pointer to contraction definition object
      explicit ContractReduceOp(const ContractionEvalImpl<Left, Right, Op>& owner) :
          owner_(& owner)
      { TA_ASSERT(owner_); }

      /// Functor copy constructor

      /// Shallow copy of this functor
      /// \param other The functor to be copied
      ContractReduceOp(const ContractionEvalImpl<Left, Right, Op>& other) :
        owner_(other.owner_)
      { }

      /// Functor assignment operator

      /// Shallow copy of this functor
      /// \param other The functor to be copied
      ContractReduceOp& operator=(const ContractionEvalImpl<Left, Right, Op>& other) {
        owner_ = other.owner_;
        return *this;
      }


      /// Create a result type object

      /// Initialize a result object for subsequent reductions
      result_type operator()() const {
        return result_type();
      }

      /// Reduce two result objects

      /// Add \c arg to \c result .
      /// \param[in,out] result The result object that will be the reduction target
      /// \param[in] arg The argument that will be added to \c result
      void operator()(result_type& result, const result_type& arg) const {
        result += arg;
      }


      /// Contract a pair of tiles and add to a target tile

      /// Contract \c left and \c right and add the result to \c result.
      /// \param[in,out] result The result object that will be the reduction target
      /// \param[in] left The left-hand tile to be contracted
      /// \param[in] right The right-hand tile to be contracted
      void operator()(result_type& result, const first_argument_type& first, const second_argument_type& second) const {
        owner_->contract(result, first, second);
      }

      /// Contract a pair of tiles and add to a target tile

      /// Contract \c left1 with \c right1 and \c left2 with \c right2 ,
      /// and add the two results.
      /// \param[in] left The first left-hand tile to be contracted
      /// \param[in] right The first right-hand tile to be contracted
      /// \param[in] left The second left-hand tile to be contracted
      /// \param[in] right The second right-hand tile to be contracted
      /// \return A tile that contains the sum of the two contractions.
      result_type operator()(const first_argument_type& first1, const second_argument_type& second1,
          const first_argument_type& first2, const second_argument_type& second2) const {
        result_type result = operator()();

        owner_->contract(result, first1, second1);
        owner_->contract(result, first2, second2);

        return result;
      }

    private:
      const ContractionEvalImpl<Left, Right, Op>* owner_; ///< The contraction definition object pointer
    }; // class contract_reduce_op



    template <typename Left, typename Right, typename Op>
    class ContractionEvalImpl : public DistEvalImpl<typename Op::result_type> {
    public:
      // Base class typedefs
      typedef Op op_type;
      typedef DistEvalImpl<typename Op::result_type> DistEvalImpl_; ///< Base class type
      typedef typename DistEvalImpl_::TensorImpl_ TensorImpl_;
      typedef ContractionEvalImpl<Left, Right, Op> ContractionTensorImpl_;

      typedef Left left_type; ///< The left tensor type
      typedef typename left_type::value_type left_value_type; /// The left tensor value type
      typedef Right right_type; ///< The right tensor type
      typedef typename right_type::value_type right_value_type; ///< The right tensor value type

      typedef typename TensorImpl_::size_type size_type; ///< size type
      typedef typename TensorImpl_::pmap_interface pmap_interface; ///< The process map interface type
      typedef typename TensorImpl_::trange_type trange_type;
      typedef typename TensorImpl_::range_type range_type;
      typedef typename TensorImpl_::shape_type shape_type;
      typedef typename TensorImpl_::value_type value_type; ///< The result value type
      typedef typename TensorImpl_::storage_type::const_iterator const_iterator; ///< Tensor const iterator
      typedef typename TensorImpl_::storage_type::future const_reference;

    private:
      op_type op_; ///< The contraction operation
      left_type left_; ///< The left argument tensor
      right_type right_; /// < The right argument tensor

      size_type left_inner_; ///< The number of inner indices in the left tensor argument
      size_type left_outer_; ///< The number of outer indices in the left tensor argument
      size_type right_inner_; ///< The number of inner indices in the right tensor argument
      size_type right_outer_; ///< The number of outer indices in the right tensor argument

    protected:
      const ProcessID rank_; ///< This process's rank
      const ProcessID size_; ///< Then number of processes
      size_type m_; ///< Number of element rows in the result and left matrix
      size_type n_; ///< Number of element columns in the result matrix and rows in the right argument matrix
      size_type k_; ///< Number of element columns in the left and right argument matrices
      size_type mk_; ///< Number of elements in left matrix
      size_type kn_; ///< Number of elements in right matrix
      size_type proc_cols_; ///< Number of columns in the result process map
      size_type proc_rows_; ///< Number of rows in the result process map
      size_type proc_size_; ///< Number of process in the process map. This may be
                         ///< less than the number of processes in world.
      ProcessID rank_row_; ///< This node's row in the process map
      ProcessID rank_col_; ///< This node's column in the process map
      size_type local_rows_; ///< The number of local element rows
      size_type local_cols_; ///< The number of local element columns
      size_type local_size_; ///< Number of local elements

    public:

      ContractionEvalImpl(const left_type& left, const right_type& right,
          const op_type& op, const Permutation& perm, madness::World& world,
          const trange_type& trange, const shape_type& shape,
          const std::shared_ptr<pmap_interface>& pmap) :
        DistEvalImpl_(world, perm, trange, shape, pmap),
        op_(op), left_(left), right_(right),
        left_inner_(0ul), left_outer_(0ul), right_inner_(0ul), right_outer_(0ul),
        rank_(TensorImpl_::get_world().rank()), size_(TensorImpl_::get_world().size()),
        m_(1ul), n_(1ul), k_(1ul), mk_(1ul), kn_(1ul),
        proc_cols_(0ul), proc_rows_(0ul), proc_size_(0ul),
        rank_row_(-1), rank_col_(-1),
        local_rows_(0ul), local_cols_(0ul), local_size_(0ul)
      {
        // Calculate the size of the inner dimension, k.
        k_ = op_(left_.range().size(), right_.range().size());

        // Calculate the dimensions of the inner and outer fused tensors
        mk_ = left_.range().volume();
        kn_ = right_.range().volume();
        m_ = mk_ / k_;
        n_ = kn_ / k_;
        left_outer_ = left_.range().dim() - left_inner_;
        right_inner_ = left_inner_;
        right_outer_ = right_.range().dim() - right_inner_;

        // Calculate the process map dimensions and size
        proc_cols_ = std::min(size_ / std::max(std::min<std::size_t>(std::sqrt(size_ * m_ / n_), size_), 1ul), n_);
        proc_rows_ = std::min(size_ / proc_cols_, m_);
        proc_size_ = proc_cols_ * proc_rows_;

        if(rank_ < proc_size_) {
          // Calculate this rank's row and column
          rank_row_ = rank_ / proc_cols_;
          rank_col_ = rank_ % proc_cols_;

          // Calculate the local tile dimensions and size
          local_rows_ = (m_ / proc_rows_) + (rank_row_ < (m_ % proc_rows_) ? 1 : 0);
          local_cols_ = (n_ / proc_cols_) + (rank_col_ < (n_ % proc_cols_) ? 1 : 0);
          local_size_ = local_rows_ * local_cols_;
        }
      }

      virtual ~ContractionEvalImpl() { }

      const left_type& left() const { return left_; }

      left_type& left() { return left_; }

      const right_type& right() const { return right_; }

      right_type& right() { return right_; }

      const op_type& op() const { return op_; }

      /// Construct the left argument process map
      std::shared_ptr<pmap_interface> make_left_pmap() const {
        return std::shared_ptr<pmap_interface>(new TiledArray::detail::CyclicPmap(
            TensorImpl_::get_world(), m_, k_, proc_rows_, proc_cols_));
      }

      /// Construct the right argument process map
      std::shared_ptr<pmap_interface> make_right_pmap() const {
        return std::shared_ptr<pmap_interface>(new TiledArray::detail::CyclicPmap(
            TensorImpl_::get_world(), k_, n_, proc_rows_, proc_cols_));
      }

    protected:
      /// Contraction operation

      /// Contract \c left and \c right to \c result .
      /// \param[out] result The tensor that will store the result
      /// \param[in] left The left hand tensor argument
      /// \param[in] right The right hand tensor argument
      void contract(value_type& result, const left_value_type& left, const right_value_type& right) const {
        op_(result, left, right);
      }

    private:

      virtual void eval_children(madness::AtomicInt& counter, int& task_count) {
        left_.eval(counter, task_count);
        right_.eval(counter, task_count);
      }
    }; // class ContractionTensorImpl

  } // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_DIST_EVAL_CONTRACTION_EVAL_H__INCLUDED
