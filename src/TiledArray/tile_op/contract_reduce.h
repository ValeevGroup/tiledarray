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
              ///< Transpose operation that is applied to the left-hand argument
      madness::cblas::CBLAS_TRANSPOSE right_op_;
              ///< Transpose operation that is applied to the right-hand argument
      scalar_type alpha_; ///< Scaling factor applied to the contraction of the left- and right-hand arguments
      Permutation perm_; ///< Permutation that is applied to the final result tensor
      unsigned int result_rank_; ///< The rank of the result tensor

      /// Contraction argument range data

      /// The range data held by this object is the range of the inner and outer
      /// dimensions of the argument tensor. It is assumed that the inner and
      /// outer dimensions are contiguous.
      struct ContractArg {
        unsigned int inner[2]; ///< The inner dimension range
        unsigned int outer[2]; ///< The outer dimension range
        unsigned int rank; ///< Rank of the argument tensor
      }
        left_, ///< Left-hand argument range data
        right_; ///< Right-hand argument range data

      static const scalar_type zero_; ///< Constant equal to 0
      static const scalar_type one_; ///< Constant equal to 1

      void check_dims(first_argument_type left, second_argument_type right, const result_type& result) const {

        // Check that the outer dimensions of left match the the corresponding dimensions in result
        TA_ASSERT(std::equal(left.range().start().begin() + left_.outer[0],
            left.range().start().begin() + left_.outer[1], result.range().start().begin()));
        TA_ASSERT(std::equal(left.range().finish().begin() + left_.outer[0],
            left.range().finish().begin() + left_.outer[1], result.range().finish().begin()));
        TA_ASSERT(std::equal(left.range().size().begin() + left_.outer[0],
            left.range().size().begin() + left_.outer[1], result.range().size().begin()));

        // Check that the outer dimensions of right match the the corresponding dimensions in result
        TA_ASSERT(std::equal(right.range().start().begin() + right_.outer[0],
            right.range().start().begin() + right_.outer[1],
            result.range().start().begin() + (left_.outer[1] - left_.outer[0])));
        TA_ASSERT(std::equal(right.range().finish().begin() + right_.outer[0],
            right.range().finish().begin() + right_.outer[1],
            result.range().finish().begin() + (left_.outer[1] - left_.outer[0])));
        TA_ASSERT(std::equal(right.range().size().begin() + right_.outer[0],
            right.range().size().begin() + right_.outer[1],
            result.range().size().begin() + (left_.outer[1] - left_.outer[0])));

        // Check that the inner dimensions of left and right match
        TA_ASSERT(std::equal(left.range().start().begin() + left_.inner[0],
            left.range().start().begin() + left_.inner[1],
            right.range().start().begin() + right_.inner[0]));
        TA_ASSERT(std::equal(left.range().finish().begin() + left_.inner[0],
            left.range().finish().begin() + left_.inner[1],
            right.range().finish().begin() + right_.inner[0]));
        TA_ASSERT(std::equal(left.range().size().begin() + left_.inner[0],
            left.range().size().begin() + left_.inner[1],
            right.range().size().begin() + right_.inner[0]));

      }

      result_type make_result(first_argument_type left, second_argument_type right) const {
        // Create the start and finish indices
        std::vector<std::size_t> start, finish;
        start.reserve(result_rank_);
        finish.reserve(result_rank_);

        // Copy left-hand argument outer dimensions to start and finish
        for(unsigned int i = left_.outer[0]; i < left_.outer[1]; ++i) {
          start.push_back(left.range().start()[i]);
          finish.push_back(left.range().finish()[i]);
        }

        // Copy right-hand argument outer dimensions to start and finish
        for(unsigned int i = right_.outer[0]; i < right_.outer[1]; ++i) {
          start.push_back(right.range().start()[i]);
          finish.push_back(right.range().finish()[i]);
        }

        // Construct the result tile
        return result_type(typename result_type::range_type(start, finish));
      }

    public:

      /// Construct contract/reduce functor

      /// \param num_cont_ranks The number of contracted ranks
      ContractReduce(const madness::cblas::CBLAS_TRANSPOSE left_op,
          madness::cblas::CBLAS_TRANSPOSE right_op, scalar_type alpha,
          const unsigned int result_dim, const unsigned int left_dim,
          const unsigned int right_dim, const Permutation& perm = Permutation()) :
        left_op_(left_op), right_op_(right_op), alpha_(alpha),
        perm_(perm), result_rank_(result_dim), left_(), right_()
      {
        // Compute the number of contracted dimensions in left and right.
        TA_ASSERT(((left_dim + right_dim - result_dim) % 2) == 0u);
        const unsigned int contract_size = (left_dim + right_dim - result_dim) >> 1;

        left_.rank = left_dim;
        right_.rank = right_dim;

        // Store the inner and outer dimension ranges for the left-hand argument.
        if(left_op_ == madness::cblas::NoTrans) {
          left_.outer[0] = 0u;
          left_.outer[1] = left_.inner[0] = left_dim - contract_size;
          left_.inner[1] = left_dim;
        } else {
          left_.inner[0] = 0ul;
          left_.inner[1] = left_.outer[0] = contract_size;
          left_.outer[1] = left_dim;
        }

        // Store the inner and outer dimension ranges for the right-hand argument.
        if(right_op_ == madness::cblas::NoTrans) {
          right_.inner[0] = 0u;
          right_.inner[1] = right_.outer[0] = contract_size;
          right_.outer[1] = right_dim;
        } else {
          right_.outer[0] = 0u;
          right_.outer[1] = right_.inner[0] = right_dim - contract_size;
          right_.inner[1] = right_dim;
        }
      }

      /// Functor copy constructor

      /// Shallow copy of this functor
      /// \param other The functor to be copied
      ContractReduce(const ContractReduce_& other) :
        left_op_(other.left_op_), right_op_(other.right_op_),
        alpha_(other.alpha_), perm_(other.perm_), result_rank_(other.result_rank_),
        left_(other.left_), right_(other.right_)
      { }

      /// Functor assignment operator

      /// \param other The functor to be copied
      ContractReduce_& operator=(const ContractReduce_& other) {
        left_op_ = other.left_op_;
        right_op_ = other.right_op_;
        alpha_ = other.alpha_;
        perm_ = other.perm_;
        result_rank_ = other.result_rank_;
        left_ = other.left_;
        right_ = other.right_;

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
      void operator()(result_type& result, first_argument_type left,
          second_argument_type right) const
      {
        // Check that the arguments are not empty and have the correct ranks
        TA_ASSERT(!left.empty());
        TA_ASSERT(!right.empty());
        TA_ASSERT(left.range().dim() == left_.rank);
        TA_ASSERT(right.range().dim() == right_.rank);

        scalar_type beta = one_;
        if(result.empty()) {
          result = make_result(left, right);
          beta = zero_;
        }

        // Check that the result is not empty and has the correct rank
        TA_ASSERT(!result.empty());
        TA_ASSERT(result.range().dim() == result_rank_);

        check_dims(left, right, result);

        // Compute fused dimension sizes
        integer m = 1, n = 1, k = 1;
        for(unsigned int i = left_.outer[0]; i < left_.outer[1]; ++i)
          m *= left.range().size()[i];
        for(unsigned int i = left_.inner[0]; i < left_.inner[1]; ++i)
          k *= left.range().size()[i];
        for(unsigned int i = right_.outer[0]; i < right_.outer[1]; ++i)
          n *= right.range().size()[i];

        // Do the contraction
        gemm(left_op_, right_op_, m, n, k, alpha_, left.data(), right.data(),
            beta, result.data());
      }

      /// Contract a pair of tiles and add to a target tile

      /// Contract \c left1 with \c right1 and \c left2 with \c right2 ,
      /// and add the two results.
      /// \param[in,out] result The object that will hold the result of this
      /// reduction operation.
      /// \param[in] left1 The first left-hand tile to be contracted
      /// \param[in] right1 The first right-hand tile to be contracted
      /// \param[in] left2 The second left-hand tile to be contracted
      /// \param[in] right2 The second right-hand tile to be contracted
      void operator()(result_type& result,
          first_argument_type left1, second_argument_type right1,
          first_argument_type left2, second_argument_type right2) const
      {
        // Check that the arguments are not empty and have the correct ranks
        TA_ASSERT(!left1.empty());
        TA_ASSERT(!right1.empty());
        TA_ASSERT(!left2.empty());
        TA_ASSERT(!right2.empty());
        TA_ASSERT(left1.range().dim() == left_.rank);
        TA_ASSERT(right1.range().dim() == right_.rank);
        TA_ASSERT(left2.range().dim() == left_.rank);
        TA_ASSERT(right2.range().dim() == right_.rank);

        scalar_type beta = one_;
        if(result.empty()) {
          result = make_result(left1, right1);
          beta = zero_;
        }

        // Check that the result is not empty and has the correct rank
        TA_ASSERT(!result.empty());
        TA_ASSERT(result.range().dim() == result_rank_);

        check_dims(left1, right1, result);

        // Compute fused dimension sizes for the first contraction
        integer m = 1, n = 1, k = 1;
        for(unsigned int i = left_.outer[0]; i < left_.outer[1]; ++i)
          m *= left1.range().size()[i];
        for(unsigned int i = right_.outer[0]; i < right_.outer[1]; ++i)
          n *= right1.range().size()[i];
        for(unsigned int i = left_.inner[0]; i < left_.inner[1]; ++i)
          k *= left1.range().size()[i];

        TA_ASSERT(left1.size() == (m * k));
        TA_ASSERT(right1.size() == (k * n));
        TA_ASSERT(result.size() == (m * n));

        // Do the contraction with first pair
        gemm(left_op_, right_op_, m, n, k, alpha_, left1.data(), right1.data(),
            beta, result.data());

        check_dims(left2, right2, result);

        // Compute fused inner dimension size for the second contraction
        k = 1;
        for(unsigned int i = left_.inner[0]; i < left_.inner[1]; ++i)
          k *= left2.range().size()[i];

        TA_ASSERT(left2.size() == (m * k));
        TA_ASSERT(right2.size() == (k * n));
        TA_ASSERT(result.size() == (m * n));

        // Do contraction with second pair
        gemm(left_op_, right_op_, m, n, k, alpha_, left2.data(), right2.data(),
            one_, result.data());
      }

    }; // class ContractReduce


    // Initialize static constants
    template <typename Result, typename Left, typename Right>
    const typename ContractReduce<Result, Left, Right>::scalar_type
    ContractReduce<Result, Left, Right>::zero_(0);

    template <typename Result, typename Left, typename Right>
    const typename ContractReduce<Result, Left, Right>::scalar_type
    ContractReduce<Result, Left, Right>::one_(1);

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_CONTRACT_REDUCE_H__INCLUDED
