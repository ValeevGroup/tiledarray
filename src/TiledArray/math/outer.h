/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  outer.h
 *  Feb 16, 2014
 *
 */

#ifndef TILEDARRAY_MATH_OUTER_H__INCLUDED
#define TILEDARRAY_MATH_OUTER_H__INCLUDED

#include <TiledArray/math/vector_op.h>
#include <TiledArray/utility.h>

namespace TiledArray {
  namespace math {

    /// Outer algorithm automatic loop unwinding

    /// \tparam N The number of steps to unwind
    template <std::size_t N>
    class OuterVectorOpUnwind;

    template <>
    class OuterVectorOpUnwind<0> {
    public:

      static const std::size_t offset = TILEDARRAY_LOOP_UNWIND - 1;

      template <typename X, typename Y, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      outer(const X* const x_block, const Y* const y_block,
          Result* const result, const std::size_t /*stride*/, const Op& op)
      {
        TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];
        copy_block(result_block, result);

        const X x = x_block[offset];
        for_each_block([x,&op] (Result& r, const Y y) { op(r, x, y); },
            result_block, y_block);

        copy_block(result, result_block);
      }

      template <typename X, typename Y, typename Init, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      fill(const X* const x_block, const Y* const y_block,
          const Init* const init, Result* const result,
          const std::size_t /*stride*/, const Op& op)
      {
        TILEDARRAY_ALIGNED_STORAGE Init init_block[TILEDARRAY_LOOP_UNWIND];
        copy_block(init_block, init);

        const X x = x_block[offset];
        for_each_block([x,&op] (Init& i, const Y y) { op(i, x, y); },
            init_block, y_block);

        copy_block(result, init_block);
      }

      template <typename X, typename Y, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      fill(const X* const x_block, const Y* const y_block,
          Result* const result, const std::size_t /*stride*/, const Op& op)
      {
        TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];

        const X x = x_block[offset];

        for_each_block([x,&op] (Result& res, const Y y) { res = op(x, y); },
            result_block, y_block);

        copy_block(result, result_block);
      }
    }; // class OuterVectorOpUnwind<0>


    template <std::size_t N>
    class OuterVectorOpUnwind : public OuterVectorOpUnwind<N - 1> {
    public:

      typedef OuterVectorOpUnwind<N - 1> OuterVectorOpUnwindN1;

      static const std::size_t offset = TILEDARRAY_LOOP_UNWIND - N - 1;

      template <typename X, typename Y, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      outer(const X* const x_block, const Y* const y_block,
          Result* const result, const std::size_t stride, const Op& op)
      {
        {
          TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];
          copy_block(result_block, result);

          const X x = x_block[offset];
          for_each_block([x,&op] (Result& r, const Y y) { op(r, x, y); },
              result_block, y_block);

          copy_block(result, result_block);
        }

        OuterVectorOpUnwindN1::outer(x_block, y_block, result + stride, stride, op);
      }


      template <typename X, typename Y, typename Init, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      fill(const X* const x_block, const Y* const y_block,
          const Init* const init, Result* const result,
          const std::size_t stride, const Op& op)
      {
        {
          TILEDARRAY_ALIGNED_STORAGE Init init_block[TILEDARRAY_LOOP_UNWIND];
          copy_block(init_block, init);

          const X x = x_block[offset];
          for_each_block([x,&op] (Init& i, const Y y) { op(i, x, y); },
              init_block, y_block);

          copy_block(result, init_block);
        }

        OuterVectorOpUnwindN1::fill(x_block, y_block, init + stride, result + stride, stride, op);
      }

      template <typename X, typename Y, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      fill(const X* const x_block, const Y* const y_block,
          Result* MADNESS_RESTRICT const result, const std::size_t stride, const Op& op)
      {
        {
          TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];

          const X x = x_block[offset];

          for_each_block([x,&op] (Result& res, const Y y) { res = op(x, y); },
              result_block, y_block);

          copy_block(result, result_block);
        }

        OuterVectorOpUnwindN1::fill(x_block, y_block, result + stride, stride, op);
      }
    }; // class OuterVectorOpUnwind

    // Convenience typedef
    typedef OuterVectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1> OuterVectorOpUnwindN;


    /// Compute and store outer of \c x and \c y in \c a

    /// <tt>a[i][j] = op(x[i], y[j])</tt>.
    /// \tparam X The left-hand vector element type
    /// \tparam Y The right-hand vector element type
    /// \tparam A The a matrix element type
    /// \param[in] m The size of the left-hand vector
    /// \param[in] n The size of the right-hand vector
    /// \param[in] x The left-hand vector
    /// \param[in] y The right-hand vector
    /// \param[out] a The result matrix of size \c m*n
    /// \param[in] op The operation that will compute the outer product elements
    template <typename X, typename Y, typename A, typename Op>
    void outer_fill(const std::size_t m, const std::size_t n,
        const X* const x, const Y* const y, A* a, const Op& op)
    {
      std::size_t i = 0ul;

      // Compute block iteration limit
      const std::size_t mx = m & index_mask::value; // = m - m % TILEDARRAY_LOOP_UNWIND
      const std::size_t nx = n & index_mask::value; // = n - n % TILEDARRAY_LOOP_UNWIND
      const std::size_t a_block_stride = n * TILEDARRAY_LOOP_UNWIND;

      for(; i < mx; i += TILEDARRAY_LOOP_UNWIND, a += a_block_stride) {

        // Load x block
        TILEDARRAY_ALIGNED_STORAGE X x_block[TILEDARRAY_LOOP_UNWIND];
        copy_block(x_block, x + i);

        std::size_t j = 0ul;
        for(; j < nx; j += TILEDARRAY_LOOP_UNWIND) {

          // Load y block
          TILEDARRAY_ALIGNED_STORAGE Y y_block[TILEDARRAY_LOOP_UNWIND];
          copy_block(y_block, y + j);

          // Compute and store a block
          OuterVectorOpUnwindN::fill(x_block, y_block, a + j, n, op);

        }

        for(; j < n; ++j) {

          // Load y block
          const Y y_j = y[j];

          // Compute a block
          TILEDARRAY_ALIGNED_STORAGE A a_block[TILEDARRAY_LOOP_UNWIND];

          const auto bind_first_op = [y_j,&op] (A& a_ij, const X x_i) { a_ij = op(x_i, y_j); };
          for_each_block(bind_first_op, a_block, x_block);

          // Store a block
          scatter_block(a + j, n, a_block);
        }
      }

      for(; i < m; ++i, a += n) {

        const X x_i = x[i];
        vector_op([x_i,&op] (const Y y_j)
            { return op(x_i, y_j); }, n, a, y);

      }
    }

    /// Compute the outer of \c x and \c y to modify \c a

    /// Compute <tt>op(a[i][j], x[i], y[j])</tt> for each \c i and \c j pair,
    /// where \c a[i][j] is modified by \c op.
    /// \tparam X The left hand vector element type
    /// \tparam Y The right-hand vector element type
    /// \tparam A The a matrix element type
    /// \tparam Op The operation that will compute outer product elements
    /// \param[in] m The size of the left-hand vector
    /// \param[in] n The size of the right-hand vector
    /// \param[in] x The left-hand vector
    /// \param[in] y The right-hand vector
    /// \param[in,out] a The result matrix of size \c m*n
    /// \param[in] op The operation used to generate the result
    template <typename X, typename Y, typename A, typename Op>
    void outer(const std::size_t m, const std::size_t n,
        const X* const x, const Y* const y, A* a, const Op& op)
    {
      std::size_t i = 0ul;

      // Compute block iteration limit
      const std::size_t mx = m & index_mask::value; // = m - m % TILEDARRAY_LOOP_UNWIND
      const std::size_t nx = n & index_mask::value; // = n - n % TILEDARRAY_LOOP_UNWIND
      const std::size_t a_block_stride = n * TILEDARRAY_LOOP_UNWIND;

      for(; i < mx; i += TILEDARRAY_LOOP_UNWIND, a += a_block_stride) {

        // Load x block
        TILEDARRAY_ALIGNED_STORAGE X x_block[TILEDARRAY_LOOP_UNWIND];
        copy_block(x_block, x + i);

        std::size_t j = 0ul;
        for(; j < nx; j += TILEDARRAY_LOOP_UNWIND) {

          // Load y block
          TILEDARRAY_ALIGNED_STORAGE Y y_block[TILEDARRAY_LOOP_UNWIND];
          copy_block(y_block, y + j);

          // Load, compute, and store a block
          OuterVectorOpUnwindN::outer(x_block, y_block, a + j, n, op);

        }

        for(; j < n; ++j) {

          // Load a block
          A* const a_ij = a + j;
          TILEDARRAY_ALIGNED_STORAGE A a_block[TILEDARRAY_LOOP_UNWIND];
          gather_block(a_block, a_ij, n);

          // Load y block
          const Y y_j = y[j];

          // Compute a block
          for_each_block([y_j,&op] (A& a_ij, const X x_i)
              { return op(a_ij, x_i, y_j); },
              a_block, x_block);

          // Store a block
          scatter_block(a_ij, n, a_block);

        }
      }

      for(; i < m; ++i, a += n) {
        const X x_i = x[i];
        inplace_vector_op([x_i,&op] (A& a_ij, const Y y_j)
            { return op(a_ij, x_i, y_j); },
            n, a, y);
      }
    }


    /// Compute the outer of \c x, \c y, and \c a, and store the result in \c b

    /// Store a modified copy of \c a in \c b, where modified elements are
    /// generated using the following algorithm:
    /// \code
    /// A temp = a[i][j];
    /// op(temp, x[i], y[j]);
    /// b[i][j] = temp;
    /// \endcode
    /// for each unique pair of \c i and \c j.
    /// \tparam X The left hand vector element type
    /// \tparam Y The right-hand vector element type
    /// \tparam A The a matrix element type
    /// \tparam B The b matrix element type
    /// \tparam Op The operation that will compute outer product elements
    /// \param[in] m The size of the left-hand vector
    /// \param[in] n The size of the right-hand vector
    /// \param[in] x The left-hand vector
    /// \param[in] y The right-hand vector
    /// \param[in] a The input matrix of size \c m*n
    /// \param[out] b The output matrix of size \c m*n
    /// \param[in] op The operation that will compute the outer product elements
    template <typename X, typename Y, typename A, typename B, typename Op>
    void outer_fill(const std::size_t m, const std::size_t n,
        const X* MADNESS_RESTRICT const x, const Y* MADNESS_RESTRICT const y,
        const A* MADNESS_RESTRICT a, B* MADNESS_RESTRICT b, const Op& op)
    {
      std::size_t i = 0ul;

      // Compute block iteration limit
      const std::size_t mx = m & index_mask::value; // = m - m % TILEDARRAY_LOOP_UNWIND
      const std::size_t nx = n & index_mask::value; // = n - n % TILEDARRAY_LOOP_UNWIND
      const std::size_t a_block_stride = n * TILEDARRAY_LOOP_UNWIND;

      for(; i < mx; i += TILEDARRAY_LOOP_UNWIND, a += a_block_stride, b += a_block_stride) {

        // Load x block
        TILEDARRAY_ALIGNED_STORAGE X x_block[TILEDARRAY_LOOP_UNWIND];
        copy_block(x_block, x + i);

        std::size_t j = 0ul;
        for(; j < nx; j += TILEDARRAY_LOOP_UNWIND) {

          // Load y block
          TILEDARRAY_ALIGNED_STORAGE Y y_block[TILEDARRAY_LOOP_UNWIND];
          copy_block(y_block, y + j);

          // Load, compute, and store a block
          OuterVectorOpUnwindN::fill(x_block, y_block, a + j, b + j, n, op);

        }

        for(; j < n; ++j) {

          // Load a block
          TILEDARRAY_ALIGNED_STORAGE A a_block[TILEDARRAY_LOOP_UNWIND];
          gather_block(a_block, a + j, n);

          // Load y block
          const Y y_j = y[j];

          // Compute a block
          for_each_block([y_j,&op] (A& a_ij, const X x_i)
              { return op(a_ij, x_i, y_j); },
              a_block, x_block);

          // Store a block
          scatter_block(b + j, n, a_block);
        }
      }

      for(; i < m; ++i, a += n, b += n) {

        // Load x block
        const X x_i = x[i];

        std::size_t j = 0ul;

        for(; j < nx; j += TILEDARRAY_LOOP_UNWIND) {

          // Load a block
          TILEDARRAY_ALIGNED_STORAGE A a_block[TILEDARRAY_LOOP_UNWIND];
          copy_block(a_block, a + j);

          // Load y block
          TILEDARRAY_ALIGNED_STORAGE Y y_block[TILEDARRAY_LOOP_UNWIND];
          copy_block(y_block, y + j);

          // Compute outer block
          for_each_block([x_i,&op] (A& a_ij, const Y y_j)
              { return op(a_ij, x_i, y_j); },
              a_block, y_block);

          // Store a block
          copy_block(b + j, a_block);

        }

        for(; j < n; ++j) {

          A a_ij = a[j];
          const Y y_j = y[j];
          op(a_ij, x_i, y_j);
          b[j] = a_ij;
        }
      }
    }

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_MATH_OUTER_H__INCLUDED
