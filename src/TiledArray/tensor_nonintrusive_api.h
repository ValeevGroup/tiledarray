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

#ifndef TILEDARRAY_NONINTRUSIVE_API_TENSOR_H__INCLUDED
#define TILEDARRAY_NONINTRUSIVE_API_TENSOR_H__INCLUDED

namespace TiledArray {
  namespace math {

    /// Computes the result of applying permutation \c perm to \c arg
    template <typename Tensor>
    Tensor permute(const Tensor& arg,
                   const TiledArray::Permutation& perm) {
      return arg.permute(perm);
    }

    /// Addition operations

    /// result[i] = arg1[i] + arg2[i]
    template <typename Tensor>
    Tensor add(const Tensor& arg1,
               const Tensor& arg2) {
      return arg1.add(arg2);
    }

    /// result[i] = (arg1[i] + arg2[i]) * factor
    template <typename Tensor>
    Tensor add(const Tensor& arg1,
               const Tensor& arg2,
               const typename Tensor::numeric_type factor) {
      return arg1.add(arg2,factor);
    }

    /// result[perm ^ i] = arg1[i] + arg2[i]
    template <typename Tensor>
    Tensor add(const Tensor& arg1,
               const Tensor& arg2,
               const TiledArray::Permutation& perm) {
      return arg1.add(arg2,perm);
    }

    /// result[perm ^ i] = (arg1[i] + arg2[i]) * factor
    template <typename Tensor>
    Tensor add(const Tensor& arg1,
               const Tensor& arg2,
               const typename Tensor::numeric_type factor,
               const TiledArray::Permutation& perm) {
      return arg1.add(arg2,factor,perm);
    }

    /// result[perm ^ i] = arg[i] + value
    template <typename Tensor>
    Tensor add(const Tensor& arg,
               const typename Tensor::value_type& value,
               const TiledArray::Permutation& perm) {
      return arg.add(value,perm);
    }

    /// result[i] = arg[i] + value
    template <typename Tensor>
    Tensor add(const Tensor& arg,
               const typename Tensor::value_type value) {
      return arg.add(value);
    }

    /// result[i] += arg[i]
    template <typename Tensor>
    void add_to(Tensor& result,
                const Tensor& arg) {
      result.add_to(arg);
    }

    /// (result[i] += arg[i]) *= factor
    template <typename Tensor>
    void add_to(Tensor& result,
                const Tensor& arg,
                const typename Tensor::numeric_type factor) {
      result.add_to(arg,factor);
    }

    /// result[i] += value
    template <typename Tensor>
    void add_to(Tensor& result,
                const typename Tensor::value_type& value) {
      result.add_to(value);
    }

    /// Subtraction operations

    /// result[i] = arg1[i] - arg2[i]
    template <typename Tensor>
    Tensor subt(const Tensor& arg1,
                  const Tensor& arg2) {
      return arg1.subt(arg2);
    }

    /// result[i] = (arg1[i] - arg2[i]) * factor
    template <typename Tensor>
    Tensor subt(const Tensor& arg1,
                  const Tensor& arg2,
                  const typename Tensor::numeric_type factor) {
      return arg1.subt(arg2, factor);
    }

    // result[i] = arg[i] - value
    template <typename Tensor>
    Tensor subt(const Tensor& arg,
                  const typename Tensor::value_type& value) {
      return arg.subt(value);
    }

    // result[perm ^ i] = arg1[i] - arg2[i]
    template <typename Tensor>
    Tensor subt(const Tensor& arg1,
                  const Tensor& arg2,
                  const TiledArray::Permutation& perm) {
      return arg1.subt(arg2, perm);
    }

    // result[perm ^ i] = (arg1[i] - arg2[i]) * factor
    template <typename Tensor>
    Tensor subt(const Tensor& arg1,
                  const Tensor& arg2,
                  const typename Tensor::numeric_type factor,
                  const TiledArray::Permutation& perm) {
      return arg1.subt(arg2, factor, perm);
    }

    // result[perm ^ i] = arg[i] - value
    template <typename Tensor>
    Tensor subt(const Tensor& arg,
                  const typename Tensor::value_type value,
                  const TiledArray::Permutation& perm) {
      return arg.subt(value, perm);
    }

    // result[i] -= arg[i]
    template <typename Tensor>
    void subt_to(Tensor& result,
                 const Tensor& arg) {
      result.subt_to(arg);
    }

    // (result[i] -= arg[i]) *= factor
    template <typename Tensor>
    void subt_to(Tensor& result,
                 const Tensor& arg,
                 const typename Tensor::numeric_type factor) {
      result.subt_to(arg, factor);
    }

    // result[i] -= value
    template <typename Tensor>
    void subt_to(Tensor& result,
                 const typename Tensor::value_type& value) {
      result.subt_to(value);
    }

    /// Multiplication operations

    /// result[i] = arg1[i] * arg2[i]
    template <typename Tensor>
    Tensor mult(const Tensor& arg1,
                const Tensor& arg2) {
      return arg1.mult(arg2);
    }

    /// result[i] = (arg1[i] * arg2[i]) * factor
    template <typename Tensor>
    Tensor mult(const Tensor& arg1,
                const Tensor& arg2,
                const typename Tensor::numeric_type factor) {
      return arg1.mult(arg2,factor);
    }

    /// result[perm ^ i] = arg1[i] * arg2[i]
    template <typename Tensor>
    Tensor mult(const Tensor& arg1,
                const Tensor& arg2,
                const TiledArray::Permutation& perm) {
      return arg1.mult(arg2,perm);
    }

    /// result[perm^ i] = (arg1[i] * arg2[i]) * factor
    template <typename Tensor>
    Tensor mult(const Tensor& arg1,
                const Tensor& arg2,
                const typename Tensor::numeric_type factor,
                const TiledArray::Permutation& perm) {
      return arg1.mult(arg2,factor,perm);
    }

    /// result[i] *= arg[i]
    template <typename Tensor>
    void mult_to(Tensor& result,
                 const Tensor& arg) {
      result.mult_to(arg);
    }

    /// (result[i] *= arg[i]) *= factor
    template <typename Tensor>
    void mult_to(Tensor& result,
                 const Tensor& arg,
                 const typename Tensor::numeric_type factor) {
      result.mult_to(arg,factor);
    }

    /// Negation operations

    /// result[i] = -(arg[i])
    template <typename Tensor>
    Tensor neg(const Tensor& arg) {
      return arg.neg();
    }

    /// result[perm ^ i] = -(arg[i])
    template <typename Tensor>
    Tensor neg(const Tensor& arg,
                 const TiledArray::Permutation& perm) {
      return arg.neg(perm);
    }

    /// result[i] = -(result[i])
    template <typename Tensor>
    void neg_to(Tensor& result) {
      result.neg();
    }

    /// Contraction operations

    /// GEMM operation with fused indices as defined by gemm_config; multiply arg1 by arg2, return the result
    template <typename Tensor>
    Tensor gemm(const Tensor& arg1,
                const Tensor& arg2,
                const typename Tensor::numeric_type factor,
                const TiledArray::math::GemmHelper& gemm_config) {
      return arg1.gemm(arg2,factor,gemm_config);
    }

    /// GEMM operation with fused indices as defined by gemm_config; multiply left by right, store to result
    template <typename Tensor>
    void gemm(Tensor& result,
              const Tensor& arg1,
              const Tensor& arg2,
              const typename Tensor::numeric_type factor,
              const TiledArray::math::GemmHelper& gemm_config) {
      result.gemm(arg1,arg2,factor,gemm_config);
    }

    /// Reduction operations

    /// Sum of hyper diagonal elements
    template <typename Tensor>
    typename Tensor::numeric_type trace(const Tensor& arg) {
      return arg.trace();
    }

    /// foreach(i) result += arg[i]
    template <typename Tensor>
    typename Tensor::numeric_type sum(const Tensor& arg) {
      return arg.sum();
    }

    /// foreach(i) result *= arg[i]
    template <typename Tensor>
    typename Tensor::numeric_type prod(const Tensor& arg) {
      return arg.prod();
    }

    /// foreach(i) result += arg[i] * arg[i]
    template <typename Tensor>
    typename Tensor::numeric_type norm_sqared(const Tensor& arg) {
      return arg.norm_sqared();
    }

    /// sqrt(norm_squared(arg)), i.e 2-norm
    template <typename Tensor>
    typename Tensor::numeric_type norm(const Tensor& arg) {
      return arg.norm();
    }

    /// foreach(i) result = max(result, arg[i])
    template <typename Tensor>
    typename Tensor::numeric_type max(const Tensor& arg) {
      return arg.max();
    }

    /// foreach(i) result = min(result, arg[i])
    template <typename Tensor>
    typename Tensor::numeric_type min(const Tensor& arg) {
      return arg.min();
    }

    /// foreach(i) result = max(result, abs(arg[i]))
    template <typename Tensor>
    typename Tensor::numeric_type abs_max(const Tensor& arg) {
      return arg.abs_max();
    }

    /// foreach(i) result = max(result, abs(arg[i]))
    template <typename Tensor>
    typename Tensor::numeric_type abs_min(const Tensor& arg) {
      return arg.abs_min();
    }

    /// foreach(i) result += left[i] * right[i]
    template <typename Tensor>
    typename Tensor::numeric_type norm_sqared(const Tensor& arg1, const Tensor& arg2) {
      return arg1.norm_sqared(arg2);
    }

  } // namespace TiledArray::math
} // namespace TiledArray

#endif /* TILEDARRAY_NONINTRUSIVE_API_TENSOR_H__INCLUDED */
