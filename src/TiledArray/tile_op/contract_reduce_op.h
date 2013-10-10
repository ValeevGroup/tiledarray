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
 *  contract_reduce_op.h
 *  Oct 9, 2013
 *
 */

#ifndef TILEDARRAY_CONTRACT_REDUCE_OP_H__INCLUDED
#define TILEDARRAY_CONTRACT_REDUCE_OP_H__INCLUDED

namespace TiledArray {
  namespace math {

    /// Contract and reduce operation

    /// This object uses a tile contraction operation to form a pair reduction
    /// operation.
    template <typename Op>
    class ContractReduceOp {
    public:
      typedef ContractReduceOp<Op> ContractReduceOp_; ///< This class type
      typedef typename Op::first_argument_type first_argument_type; ///< The left tile type
      typedef typename Op::second_argument_type second_argument_type; ///< The right tile type
      typedef typename Op::result_type result_type; ///< The result tile type.

    private:
      Op op_; ///< The tile contraction operation

    public:

      /// Construct contract/reduce functor

      /// \param op The contraction operation
      explicit ContractReduceOp(const Op& op) : op_(op) { }

      /// Functor copy constructor

      /// Shallow copy of this functor
      /// \param other The functor to be copied
      ContractReduceOp(const ContractReduceOp_& other) : op_(other.op_) { }

      /// Functor assignment operator

      /// \param other The functor to be copied
      ContractReduceOp_& operator=(const ContractReduceOp_& other) {
        op_ = other.op_;
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
      void operator()(result_type& result, const first_argument_type& first,
          const second_argument_type& second) const
      {
        op_(result, first, second);
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
          const first_argument_type& first2, const second_argument_type& second2) const
      {
        result_type result = operator()();

        op_(result, first1, second1);
        op_(result, first2, second2);

        return result;
      }

    }; // class ContractReduceOp

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_CONTRACT_REDUCE_OP_H__INCLUDED
