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

#include <TiledArray/madness.h>

namespace TiledArray {
  namespace math {

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
    void outer(const std::size_t m, const std::size_t n,
        const X* restrict const x, const Y* restrict const y,
        A* restrict const a, const Op& op)
    {
      // Compute limits of block iteration

      const std::size_t mask = ~(std::size_t(3));
      const std::size_t m4 = m & mask; // = m - m % 4
      const std::size_t n4 = n & mask; // = n - n % 4

      std::size_t i = 0ul;
      for(; i < m4; i += 4ul) {

        // Load x block

        const X x_0 = x[i];
        const X x_1 = x[i + 1];
        const X x_2 = x[i + 2];
        const X x_3 = x[i + 3];

        // Compute a block pointers

        A* restrict const a_0 = a + (i * n);
        A* restrict const a_1 = a_0 + n;
        A* restrict const a_2 = a_1 + n;
        A* restrict const a_3 = a_2 + n;

        std::size_t j = 0ul;
        for(; j < n4; j += 4ul) {

          // Compute j offsets

          const std::size_t j1 = j + 1;
          const std::size_t j2 = j + 2;
          const std::size_t j3 = j + 3;

          // Load y block

          const Y y_0 = y[j ];
          const Y y_1 = y[j1];
          const Y y_2 = y[j2];
          const Y y_3 = y[j3];

          // Compute the outer product

          const A temp_00 = op(x_0, y_0);
          const A temp_01 = op(x_0, y_1);
          const A temp_02 = op(x_0, y_2);
          const A temp_03 = op(x_0, y_3);

          const A temp_10 = op(x_1, y_0);
          const A temp_11 = op(x_1, y_1);
          const A temp_12 = op(x_1, y_2);
          const A temp_13 = op(x_1, y_3);

          const A temp_20 = op(x_2, y_0);
          const A temp_21 = op(x_2, y_1);
          const A temp_22 = op(x_2, y_2);
          const A temp_23 = op(x_2, y_3);

          const A temp_30 = op(x_3, y_0);
          const A temp_31 = op(x_3, y_1);
          const A temp_32 = op(x_3, y_2);
          const A temp_33 = op(x_3, y_3);

          // Store a block

          a_0[j ] = temp_00;
          a_0[j1] = temp_01;
          a_0[j2] = temp_02;
          a_0[j3] = temp_03;

          a_1[j ] = temp_10;
          a_1[j1] = temp_11;
          a_1[j2] = temp_12;
          a_1[j3] = temp_13;

          a_2[j ] = temp_20;
          a_2[j1] = temp_21;
          a_2[j2] = temp_22;
          a_2[j3] = temp_23;

          a_3[j ] = temp_30;
          a_3[j1] = temp_31;
          a_3[j2] = temp_32;
          a_3[j3] = temp_33;

        }

        for(; j < n; ++j) {


          // Load y block

          const Y y_j = y[j];

          // Compute the outer product

          const A temp_0 = op(x_0, y_j);
          const A temp_1 = op(x_1, y_j);
          const A temp_2 = op(x_2, y_j);
          const A temp_3 = op(x_3, y_j);

          // Store a block

          a_0[j] = temp_0;
          a_1[j] = temp_1;
          a_2[j] = temp_2;
          a_3[j] = temp_3;
        }
      }

      for(; i < m; ++i) {

        // Load x

        const X x_i = x[i];

        // Compute a block pointer

        A* restrict const a_i = a + (i * n);

        std::size_t j = 0ul;
        for(; j < n4; j += 4) {

          // Compute j offsets

          const std::size_t j1 = j + 1;
          const std::size_t j2 = j + 2;
          const std::size_t j3 = j + 3;

          // Load y block

          const Y y_0 = y[j ];
          const Y y_1 = y[j1];
          const Y y_2 = y[j2];
          const Y y_3 = y[j3];

          // Compute the outer product

          const A temp_i0 = op(x_i, y_0);
          const A temp_i1 = op(x_i, y_1);
          const A temp_i2 = op(x_i, y_2);
          const A temp_i3 = op(x_i, y_3);

          // Store a block

          a_i[j ] = temp_i0;
          a_i[j1] = temp_i1;
          a_i[j2] = temp_i2;
          a_i[j3] = temp_i3;

        }

        for(; j < n; ++j) {

          // Load y
          const Y y_j = y[j];

          // Compute the product
          const A temp_ij = op(x_i, y_j);

          // Store a
          a_i[j] = temp_ij;
        }
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
    /// \param[in] alpha The scaling factor
    /// \param[in] x The left-hand vector
    /// \param[in] y The right-hand vector
    /// \param[in,out] a The result matrix of size \c m*n
    /// \param[in]
    template <typename X, typename Y, typename A, typename Op>
    void outer_product_to(const std::size_t m, const std::size_t n,
        const X* restrict const x, const Y* restrict const y,
        A* restrict const a, const Op& op)
    {
      // Compute limits of block iteration

      const std::size_t mask = ~(std::size_t(3));
      const std::size_t m4 = m & mask; // = m - m % 4
      const std::size_t n4 = n & mask; // = n - n % 4

      std::size_t i = 0ul;
      for(; i < m4; i += 4ul) {

        // Load x block

        const X x_0 = x[i];
        const X x_1 = x[i + 1];
        const X x_2 = x[i + 2];
        const X x_3 = x[i + 3];

        // Compute a block pointers

        A* restrict const a_0 = a + (i * n);
        A* restrict const a_1 = a_0 + n;
        A* restrict const a_2 = a_1 + n;
        A* restrict const a_3 = a_2 + n;

        std::size_t j = 0ul;
        for(; j < n4; j += 4ul) {

          // Compute j offsets

          const std::size_t j1 = j + 1;
          const std::size_t j2 = j + 2;
          const std::size_t j3 = j + 3;

          // Load the a data

          A temp_00 = a_0[j ];
          A temp_01 = a_0[j1];
          A temp_02 = a_0[j2];
          A temp_03 = a_0[j3];

          A temp_10 = a_1[j ];
          A temp_11 = a_1[j1];
          A temp_12 = a_1[j2];
          A temp_13 = a_1[j3];

          A temp_20 = a_2[j ];
          A temp_21 = a_2[j1];
          A temp_22 = a_2[j2];
          A temp_23 = a_2[j3];

          A temp_30 = a_3[j ];
          A temp_31 = a_3[j1];
          A temp_32 = a_3[j2];
          A temp_33 = a_3[j3];

          // Load y block

          const Y y_0 = y[j ];
          const Y y_1 = y[j1];
          const Y y_2 = y[j2];
          const Y y_3 = y[j3];

          // Compute the outer product

          op(temp_00, x_0, y_0);
          op(temp_01, x_0, y_1);
          op(temp_02, x_0, y_2);
          op(temp_03, x_0, y_3);

          op(temp_10, x_1, y_0);
          op(temp_11, x_1, y_1);
          op(temp_12, x_1, y_2);
          op(temp_13, x_1, y_3);

          op(temp_20, x_2, y_0);
          op(temp_21, x_2, y_1);
          op(temp_22, x_2, y_2);
          op(temp_23, x_2, y_3);

          op(temp_30, x_3, y_0);
          op(temp_31, x_3, y_1);
          op(temp_32, x_3, y_2);
          op(temp_33, x_3, y_3);

          // Store a block

          a_0[j ] = temp_00;
          a_0[j1] = temp_01;
          a_0[j2] = temp_02;
          a_0[j3] = temp_03;

          a_1[j ] = temp_10;
          a_1[j1] = temp_11;
          a_1[j2] = temp_12;
          a_1[j3] = temp_13;

          a_2[j ] = temp_20;
          a_2[j1] = temp_21;
          a_2[j2] = temp_22;
          a_2[j3] = temp_23;

          a_3[j ] = temp_30;
          a_3[j1] = temp_31;
          a_3[j2] = temp_32;
          a_3[j3] = temp_33;

        }

        for(; j < n; ++j) {


          // Load a block

          A temp_0 = a_0[j];
          A temp_1 = a_1[j];
          A temp_2 = a_2[j];
          A temp_3 = a_3[j];

          // Load the y

          const Y y_j = y[j];

          // Compute the outer product

          op(temp_0, x_0, y_j);
          op(temp_1, x_1, y_j);
          op(temp_2, x_2, y_j);
          op(temp_3, x_3, y_j);

          // Store a block

          a_0[j] = temp_0;
          a_1[j] = temp_1;
          a_2[j] = temp_2;
          a_3[j] = temp_3;
        }
      }

      for(; i < m; ++i) {

        // Load x

        const X x_i = x[i];

        // Compute a block pointer

        A* restrict const a_i = a + (i * n);

        std::size_t j = 0ul;
        for(; j < n4; j += 4) {

          // Compute j offsets

          const std::size_t j1 = j + 1;
          const std::size_t j2 = j + 2;
          const std::size_t j3 = j + 3;

          // Load a block

          A temp_i0 = a_i[j ];
          A temp_i1 = a_i[j1];
          A temp_i2 = a_i[j2];
          A temp_i3 = a_i[j3];

          // Load y block

          const Y y_0 = y[j ];
          const Y y_1 = y[j1];
          const Y y_2 = y[j2];
          const Y y_3 = y[j3];

          // Compute outer product

          op(temp_i0, x_i, y_0);
          op(temp_i1, x_i, y_1);
          op(temp_i2, x_i, y_2);
          op(temp_i3, x_i, y_3);

          // Store a block

          a_i[j ] = temp_i0;
          a_i[j1] = temp_i1;
          a_i[j2] = temp_i2;
          a_i[j3] = temp_i3;

        }

        for(; j < n; ++j) {

          // Load a

          const A temp_ij = a_i[j];

          // Load y

          const Y y_j = y[j];

          // Compute outer product

          op(temp_ij, x_i, y_j);

          // Store a

          a_i[j] = temp_ij;
        }
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
    /// \param[in] alpha The scaling factor
    /// \param[in] x The left-hand vector
    /// \param[in] y The right-hand vector
    /// \param[in] a The input matrix of size \c m*n
    /// \param[out] b The output matrix of size \c m*n
    template <typename X, typename Y, typename A, typename B, typename Op>
    void outer_to(const std::size_t m, const std::size_t n,
        const X* restrict const x, const Y* restrict const y,
        const A* restrict const a, B* restrict const b, const Op& op)
    {
      // Compute limits of block iteration

      const std::size_t mask = ~(std::size_t(3));
      const std::size_t m4 = m & mask; // = m - m % 4
      const std::size_t n4 = n & mask; // = n - n % 4

      std::size_t i = 0ul;
      for(; i < m4; i += 4ul) {

        // Load x block

        const X x_0 = x[i];
        const X x_1 = x[i + 1];
        const X x_2 = x[i + 2];
        const X x_3 = x[i + 3];

        // Compute a & b block pointers

        const A* restrict const a_0 = a + (i * n);
        const A* restrict const a_1 = a_0 + n;
        const A* restrict const a_2 = a_1 + n;
        const A* restrict const a_3 = a_2 + n;

        B* restrict const b_0 = b + (i * n);
        B* restrict const b_1 = b_0 + n;
        B* restrict const b_2 = b_1 + n;
        B* restrict const b_3 = b_2 + n;

        std::size_t j = 0ul;
        for(; j < n4; j += 4ul) {

          // Compute j offsets

          const std::size_t j1 = j + 1;
          const std::size_t j2 = j + 2;
          const std::size_t j3 = j + 3;

          // Load a block

          A temp_00 = a_0[j ];
          A temp_01 = a_0[j1];
          A temp_02 = a_0[j2];
          A temp_03 = a_0[j3];

          A temp_10 = a_1[j ];
          A temp_11 = a_1[j1];
          A temp_12 = a_1[j2];
          A temp_13 = a_1[j3];

          A temp_20 = a_2[j ];
          A temp_21 = a_2[j1];
          A temp_22 = a_2[j2];
          A temp_23 = a_2[j3];

          A temp_30 = a_3[j ];
          A temp_31 = a_3[j1];
          A temp_32 = a_3[j2];
          A temp_33 = a_3[j3];

          // Load y block

          const Y y_0 = y[j ];
          const Y y_1 = y[j1];
          const Y y_2 = y[j2];
          const Y y_3 = y[j3];

          // Compute outer

          op(temp_00, x_0, y_0);
          op(temp_01, x_0, y_1);
          op(temp_02, x_0, y_2);
          op(temp_03, x_0, y_3);

          op(temp_10, x_1, y_0);
          op(temp_11, x_1, y_1);
          op(temp_12, x_1, y_2);
          op(temp_13, x_1, y_3);

          op(temp_20, x_2, y_0);
          op(temp_21, x_2, y_1);
          op(temp_22, x_2, y_2);
          op(temp_23, x_2, y_3);

          op(temp_30, x_3, y_0);
          op(temp_31, x_3, y_1);
          op(temp_32, x_3, y_2);
          op(temp_33, x_3, y_3);

          // Store b block

          b_0[j ] = temp_00;
          b_0[j1] = temp_01;
          b_0[j2] = temp_02;
          b_0[j3] = temp_03;

          b_1[j ] = temp_10;
          b_1[j1] = temp_11;
          b_1[j2] = temp_12;
          b_1[j3] = temp_13;

          b_2[j ] = temp_20;
          b_2[j1] = temp_21;
          b_2[j2] = temp_22;
          b_2[j3] = temp_23;

          b_3[j ] = temp_30;
          b_3[j1] = temp_31;
          b_3[j2] = temp_32;
          b_3[j3] = temp_33;

        }

        for(; j < n; ++j) {

          // Load a block

          A temp_0 = a_0[j];
          A temp_1 = a_1[j];
          A temp_2 = a_2[j];
          A temp_3 = a_3[j];

          // Load y

          const Y y_j = y[j];

          // Compute the outer product

          op(temp_0, x_0, y_j);
          op(temp_1, x_1, y_j);
          op(temp_2, x_2, y_j);
          op(temp_3, x_3, y_j);

          // Store b block

          b_0[j] = temp_0;
          b_1[j] = temp_1;
          b_2[j] = temp_2;
          b_3[j] = temp_3;
        }
      }

      for(; i < m; ++i) {

        // Load x

        const X x_i = x[i];

        // Compute a & b block pointers

        const A* restrict const a_i = a + (i * n);

        B* restrict const b_i = b + (i * n);

        std::size_t j = 0ul;
        for(; j < n4; j += 4) {

          // Compute j offsets

          const std::size_t j1 = j + 1;
          const std::size_t j2 = j + 2;
          const std::size_t j3 = j + 3;

          // Load a block

          A temp_i0 = a_i[j ];
          A temp_i1 = a_i[j1];
          A temp_i2 = a_i[j2];
          A temp_i3 = a_i[j3];

          // Load y block

          const Y y_0 = y[j ];
          const Y y_1 = y[j1];
          const Y y_2 = y[j2];
          const Y y_3 = y[j3];

          // Compute outer product

          op(temp_i0, x_i, y_0);
          op(temp_i1, x_i, y_1);
          op(temp_i2, x_i, y_2);
          op(temp_i3, x_i, y_3);

          // Store a block

          b_i[j ] = temp_i0;
          b_i[j1] = temp_i1;
          b_i[j2] = temp_i2;
          b_i[j3] = temp_i3;

        }

        for(std::size_t j = n4; j < n; ++j) {

          // Load a

          A temp_ij = a_i[j];

          // Load y

          const Y y_j = y[j];

          // Compute outer product

          op(temp_ij, x_i, y_j);

          // Store b

          b_i[j] = temp_ij;
        }
      }
    }

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_MATH_OUTER_H__INCLUDED
