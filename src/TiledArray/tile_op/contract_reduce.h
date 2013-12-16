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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  contract_reduce.h
 *  Oct 9, 2013
 *
 */

#ifndef TILEDARRAY_CONTRACT_REDUCE_H__INCLUDED
#define TILEDARRAY_CONTRACT_REDUCE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/math/blas.h>
#include <TiledArray/tile_op/permute.h>

namespace TiledArray {
  namespace math {

    /// Contract and reduce operation

    /// This object uses a tile contraction operation to form a pair reduction
    /// operation.
    template <typename Result, typename Left, typename Right>
    class ContractReduce {
    public:
      typedef ContractReduce ContractReduce_; ///< This class type
      typedef const Left& first_argument_type; ///< The left tile type
      typedef const Right& second_argument_type; ///< The right tile type
      typedef Result result_type; ///< The result tile type.
      typedef typename TiledArray::detail::scalar_type<result_type>::type scalar_type;

    private:

      madness::cblas::CBLAS_TRANSPOSE left_op_;
      madness::cblas::CBLAS_TRANSPOSE right_op_;
      scalar_type alpha_;
      Permutation perm_;
      unsigned int left_dim_;
      unsigned int right_dim_;
      unsigned int result_dim_;
      unsigned int left_inner_begin_;
      unsigned int left_inner_end_;
      unsigned int left_outer_begin_;
      unsigned int left_outer_end_;
      unsigned int right_inner_begin_;
      unsigned int right_inner_end_;
      unsigned int right_outer_begin_;
      unsigned int right_outer_end_;

      static const scalar_type zero_; ///< Constant equal to 0
      static const scalar_type one_; ///< Constant equal to 1

      /// Contraction operation

      /// Contract \c left and \c right to \c result .
      /// \param[out] result The tensor that will store the result
      /// \param[in] left The left hand tensor argument
      /// \param[in] right The right hand tensor argument
      void contract(result_type& result, first_argument_type left, second_argument_type right) const {
        // Check that the arguments are not empty and have the correct dimension sizes
        TA_ASSERT(!left.empty());
        TA_ASSERT(!right.empty());
        TA_ASSERT(left.range().dim() == left_dim_);
        TA_ASSERT(right.range().dim() == right_dim_);

        // Allocate the result tile if it is uninitialized
        scalar_type beta = one_;
        if(result.empty()) {
          // Create the start and finish indices
          std::vector<std::size_t> start, finish;
          start.reserve(result_dim_);
          finish.reserve(result_dim_);
          for(std::size_t i = left_outer_begin_; i < left_outer_end_; ++i) {
            start.push_back(left.range().start()[i]);
            finish.push_back(left.range().finish()[i]);
          }
          for(std::size_t i = right_outer_begin_; i < right_outer_end_; ++i) {
            start.push_back(right.range().start()[i]);
            finish.push_back(right.range().finish()[i]);
          }

          // Construct the result tile
          result_type(typename result_type::range_type(start, finish)).swap(result);

          beta = zero_;
        }

        // Check that the result is not empty and has the correct dimension size
        TA_ASSERT(!result.empty());
        TA_ASSERT(result.range().dim() == result_dim_);

        // Check that the outer dimensions of left match the the corresponding dimensions in result
        TA_ASSERT(std::equal(left.range().start().begin() + left_outer_begin_,
            left.range().start().begin() + left_outer_end_, result.range().start().begin()));
        TA_ASSERT(std::equal(left.range().finish().begin() + left_outer_begin_,
            left.range().finish().begin() + left_outer_end_, result.range().finish().begin()));
        TA_ASSERT(std::equal(left.range().size().begin() + left_outer_begin_,
            left.range().size().begin() + left_outer_end_, result.range().size().begin()));

        // Check that the outer dimensions of right match the the corresponding dimensions in result
        TA_ASSERT(std::equal(right.range().start().begin() + right_outer_begin_,
            right.range().start().begin() + right_outer_end_,
            result.range().start().begin() + (left_outer_end_ - left_outer_begin_)));
        TA_ASSERT(std::equal(right.range().finish().begin() + right_outer_begin_,
            right.range().finish().begin() + right_outer_end_,
            result.range().finish().begin() + (left_outer_end_ - left_outer_begin_)));
        TA_ASSERT(std::equal(right.range().size().begin() + right_outer_begin_,
            right.range().size().begin() + right_outer_end_,
            result.range().size().begin() + (left_outer_end_ - left_outer_begin_)));

        // Check that the inner dimensions of left and right match
        TA_ASSERT(std::equal(left.range().start().begin() + left_inner_begin_,
            left.range().start().begin() + left_inner_end_,
            right.range().start().begin() + right_inner_begin_));
        TA_ASSERT(std::equal(left.range().finish().begin() + left_inner_begin_,
            left.range().finish().begin() + left_inner_end_,
            right.range().finish().begin() + right_inner_begin_));
        TA_ASSERT(std::equal(left.range().size().begin() + left_inner_begin_,
            left.range().size().begin() + left_inner_end_,
            right.range().size().begin() + right_inner_begin_));

        // Calculate the fused tile dimension
        integer m = 1, n = 1, k = 1;
        for(std::size_t i = left_outer_begin_; i < left_outer_end_; ++i)
          m *= left.range().size()[i];
        for(std::size_t i = left_inner_begin_; i < left_inner_end_; ++i)
          k *= left.range().size()[i];
        for(std::size_t i = right_outer_begin_; i < right_outer_end_; ++i)
          n *= right.range().size()[i];


        // Do the contraction
        gemm(left_op_, right_op_, m, n, k, alpha_, left.data(), right.data(),
            beta, result.data());
      }

    public:

      /// Construct contract/reduce functor

      /// \param num_cont_ranks The number of contracted ranks
      ContractReduce(const madness::cblas::CBLAS_TRANSPOSE left_op,
          madness::cblas::CBLAS_TRANSPOSE right_op, scalar_type alpha,
          const unsigned int result_dim, const unsigned int left_dim,
          const unsigned int right_dim, const Permutation& perm = Permutation()) :
        left_op_(left_op), right_op_(right_op), alpha_(alpha),
        perm_(perm),
        left_dim_(left_dim), right_dim_(right_dim), result_dim_(result_dim),
        left_inner_begin_(0u), left_inner_end_(0u),
        left_outer_begin_(0u), left_outer_end_(0u),
        right_inner_begin_(0u), right_inner_end_(0u),
        right_outer_begin_(0u), right_outer_end_(0u)
      {
        // Compute the number of contracted dimensions in left and right.
        const unsigned int contract_size = (left_dim_ + right_dim_ - result_dim_) >> 1;

        // Store the inner and outer dimension ranges for the left-hand argument.
        if(left_op_ == madness::cblas::NoTrans) {
          left_outer_end_ = left_inner_begin_ = contract_size;
          left_inner_end_ = left_dim_;
        } else {
          left_inner_end_ = left_outer_begin_ = contract_size;
          left_outer_end_ = left_dim_;
        }

        // Store the inner and outer dimension ranges for the right-hand argument.
        if(right_op_ == madness::cblas::NoTrans) {
          right_inner_end_ = right_outer_begin_ = contract_size;
          right_outer_end_ = right_dim_;
        } else {
          right_outer_end_ = right_inner_begin_ = contract_size;
          right_inner_end_ = right_dim_;
        }
      }

      /// Functor copy constructor

      /// Shallow copy of this functor
      /// \param other The functor to be copied
      ContractReduce(const ContractReduce_& other) :
        left_op_(other.left_op_), right_op_(other.right_op_),
        alpha_(other.alpha_), perm_(other.perm_), left_dim_(other.left_dim_),
        right_dim_(other.right_dim_), result_dim_(other.result_dim_),
        left_inner_begin_(other.left_inner_begin_), left_inner_end_(other.left_inner_end_),
        left_outer_begin_(other.left_outer_begin_), left_outer_end_(other.left_outer_end_),
        right_inner_begin_(other.right_inner_begin_), right_inner_end_(other.right_inner_end_),
        right_outer_begin_(other.right_outer_begin_), right_outer_end_(other.right_outer_end_)
      { }

      /// Functor assignment operator

      /// \param other The functor to be copied
      ContractReduce_& operator=(const ContractReduce_& other) {
        left_op_ = other.left_op_;
        right_op_ = other.right_op_;
        alpha_ = other.alpha_;
        perm_ = other.perm_;
        left_dim_ = other.left_dim_;
        right_dim_ = other.right_dim_;
        result_dim_ = other.result_dim_;
        left_inner_begin_ = other.left_inner_begin_;
        left_inner_end_ = other.left_inner_end_;
        left_outer_begin_ = other.left_outer_begin_;
        left_outer_end_ = other.left_outer_end_;
        right_inner_begin_ = other.right_inner_begin_;
        right_inner_end_ = other.right_inner_end_;
        right_outer_begin_ = other.right_outer_begin_;
        right_outer_end_ = other.right_outer_end_;

        return *this;
      }

      /// Create a result type object

      /// Initialize a result object for subsequent reductions
      result_type operator()() const {
        return result_type();
      }

      /// Post processing step
      result_type operator()(const result_type& temp) const {
        result_type result;

        if(! temp.empty()) {
          if(perm_.dim() < 1u)
            result = temp;
          else
            TiledArray::math::permute(result, perm_, temp);
        }

        return result;
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
      void operator()(result_type& result, first_argument_type first,
          second_argument_type second) const
      {
        contract(result, first, second);
      }

      /// Contract a pair of tiles and add to a target tile

      /// Contract \c left1 with \c right1 and \c left2 with \c right2 ,
      /// and add the two results.
      /// \param[in] left The first left-hand tile to be contracted
      /// \param[in] right The first right-hand tile to be contracted
      /// \param[in] left The second left-hand tile to be contracted
      /// \param[in] right The second right-hand tile to be contracted
      /// \return A tile that contains the sum of the two contractions.
      result_type operator()(first_argument_type first1, second_argument_type second1,
          first_argument_type first2, second_argument_type second2) const
      {
        result_type result = operator()();

        contract(result, first1, second1);
        contract(result, first2, second2);

        return result;
      }

    }; // class ContractReduce

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_CONTRACT_REDUCE_H__INCLUDED
