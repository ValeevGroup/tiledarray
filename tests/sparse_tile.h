/*
 * This file is a part of TiledArray.
 * Copyright (C) 2015  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#ifndef TILEDARRAY_TEST_SPARSE_TILE_H__INCLUDED
#define TILEDARRAY_TEST_SPARSE_TILE_H__INCLUDED

#include <Eigen/SparseCore>
#include <memory>
#include <tuple>

#include <TiledArray/external/madness.h>

// Array class
#include <TiledArray/tensor.h>
#include <TiledArray/tile.h>
#include <TiledArray/tile_op/tile_interface.h>

// Array policy classes
#include <TiledArray/policies/dense_policy.h>
#include <TiledArray/policies/sparse_policy.h>

#include <TiledArray/tile_interface/add.h>

// sparse 2-dimensional matrix type, with tag type thrown in to make expression
// engine work harder
template <typename T, typename TagType = std::tuple<>>
class EigenSparseTile {
 public:
  // Concept typedefs
  typedef TiledArray::Range range_type;  // Tensor range type
  typedef T value_type;                  // Element type
  typedef T numeric_type;  // The scalar type that is compatible with value_type
  typedef size_t size_type;  // Size type
  typedef const T& const_reference;
  typedef size_type ordinal_type;
  // other typedefs
  typedef Eigen::SparseMatrix<T, Eigen::RowMajor> matrix_type;

  typedef std::tuple<matrix_type, range_type> impl_type;

 public:
  /// makes an uninitialized matrix
  EigenSparseTile() = default;

  /// Shallow copy constructor; see EigenSparseTile::clone() for deep copy
  EigenSparseTile(const EigenSparseTile&) = default;

  /// Shallow assignment operator; see EigenSparseTile::clone() for deep copy
  EigenSparseTile& operator=(const EigenSparseTile& other) = default;

  /// makes an uninitialized matrix
  explicit EigenSparseTile(const range_type& r)
      : impl_(std::make_shared<impl_type>(
            std::make_tuple(matrix_type(r.extent()[0], r.extent()[1]), r))) {
    TA_ASSERT(r.extent()[0] > 0);
    TA_ASSERT(r.extent()[1] > 0);
  }

  /// ctor using sparse matrix
  EigenSparseTile(matrix_type&& mat, const range_type& range)
      : impl_(std::make_shared<impl_type>(
            std::make_tuple(std::move(mat), range))) {
    using extent_type = typename range_type::extent_type::value_type;
    TA_ASSERT(static_cast<extent_type>(mat.rows()) == range.extent()[0]);
    TA_ASSERT(static_cast<extent_type>(mat.cols()) == range.extent()[1]);
  }

  /// ctor using sparse matrix
  EigenSparseTile(const matrix_type& mat, const range_type& range)
      : impl_(std::make_shared<impl_type>(std::make_tuple(mat, range))) {
    using extent_type = typename range_type::extent_type::value_type;
    TA_ASSERT(static_cast<extent_type>(mat.rows()) == range.extent()[0]);
    TA_ASSERT(static_cast<extent_type>(mat.cols()) == range.extent()[1]);
  }

  // Deep copy
  EigenSparseTile clone() const {
    EigenSparseTile result;
    result.impl_ = std::make_shared<impl_type>(*(this->impl_));
    return result;
  }

  // copies
  template <typename AnotherTagType>
  explicit operator EigenSparseTile<T, AnotherTagType>() const {
    return EigenSparseTile<T, AnotherTagType>{this->matrix(), this->range()};
  }

  explicit operator TiledArray::Tensor<T>() const {
    TiledArray::Tensor<T> result(this->range(), T(0));
    auto nrows = range().extent()[0];
    auto ncols = range().extent()[1];
    Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
        Eigen::AutoAlign>
        result_view(result.data(), nrows, ncols);
    result_view = matrix();
    return result;
  }

  // Tile range accessor
  const range_type& range() const { return std::get<1>(*impl_); }

  // const data matrix access
  const matrix_type& matrix() const { return std::get<0>(*impl_); }

  // const data matrix access
  matrix_type& matrix() { return std::get<0>(*impl_); }

  /// data read-write accessor
  template <typename Index, typename = std::enable_if_t<
                                detail::is_integral_sized_range_v<Index>>>
  value_type& operator[](const Index& idx) {
    auto start = range().lobound_data();
    return matrix().coeffRef(idx[0] - start[0], idx[1] - start[1]);
  }

  /// data read-write accessor
  template <typename Ordinal,
            std::enable_if_t<std::is_integral_v<Ordinal>>* = nullptr>
  value_type& operator[](const Ordinal& ord) {
    auto idx = range().idx(ord);
    auto start = range().lobound_data();
    return matrix().coeffRef(idx[0] - start[0], idx[1] - start[1]);
  }

  /// data read-only accessor
  template <typename Index>
  std::enable_if_t<detail::is_integral_sized_range_v<Index>, const value_type&>
  operator[](const Index& idx) const {
    static const value_type zero = 0;
    auto start = range().lobound_data();
    auto* ptr = coeffPtr(idx[0] - start[0], idx[1] - start[1]);
    return ptr == nullptr ? zero : *ptr;
  }

  /// data read-only accessor
  template <typename Ordinal,
            typename = std::enable_if_t<std::is_integral_v<Ordinal>>>
  const value_type& operator[](const Ordinal& ord) const {
    static const value_type zero = 0;
    auto idx = range().idx(ord);
    auto start = range().lobound_data();
    auto* ptr = coeffPtr(idx[0] - start[0], idx[1] - start[1]);
    return ptr == nullptr ? zero : *ptr;
  }

  const value_type& at_ordinal(const ordinal_type index_ordinal) const {
    return this->operator[](index_ordinal);
  }

  value_type& at_ordinal(const ordinal_type index_ordinal) {
    return this->operator[](index_ordinal);
  }

  /// Maximum # of elements in the tile
  size_type size() const { return std::get<0>(*impl_).volume(); }

  // Initialization check. False if the tile is fully initialized.
  bool empty() const { return impl_.get() == nullptr; }

  // MADNESS compliant serialization

  // output
  template <typename Archive,
            typename std::enable_if<
                madness::is_output_archive_v<Archive>>::type* = nullptr>
  void serialize(Archive& ar) {
    if (impl_) {
      ar & true;
      auto mat = this->matrix();
      std::vector<Eigen::Triplet<T>> datavec;
      datavec.reserve(mat.size());
      typedef typename matrix_type::Index idx_t;
      for (idx_t k = 0; k < mat.outerSize(); ++k)
        for (typename matrix_type::InnerIterator it(mat, k); it; ++it) {
          datavec.push_back(Eigen::Triplet<T>(it.row(), it.col(), it.value()));
        }
      ar & datavec& this->range();
    } else {
      ar & false;
    }
  }

  // output
  template <typename Archive,
            typename std::enable_if<
                madness::is_input_archive_v<Archive>>::type* = nullptr>
  void serialize(Archive& ar) {
    bool have_impl = false;
    ar & have_impl;
    if (have_impl) {
      std::vector<Eigen::Triplet<T>> datavec;
      range_type range;
      ar & datavec & range;
      auto extents = range.extent();
      matrix_type mat(extents[0], extents[1]);
      mat.setFromTriplets(datavec.begin(), datavec.end());
      impl_ = std::make_shared<impl_type>(
          std::make_pair(std::move(mat), std::move(range)));
    } else {
      impl_ = 0;
    }
  }

  // Scaling operations

  // result[i] = (*this)[i] * factor
  EigenSparseTile scale(const numeric_type factor) const;
  // result[perm ^ i] = (*this)[i] * factor
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  EigenSparseTile scale(const numeric_type factor, const Perm& perm) const;
  // (*this)[i] *= factor
  EigenSparseTile& scale_to(const numeric_type factor) const;

 private:
  std::shared_ptr<impl_type> impl_;

  // pointer-based coeffRef
  const value_type* coeffPtr(Eigen::Index row, Eigen::Index col) const {
    auto& mat = matrix();
    constexpr bool IsRowMajor =
        std::decay_t<decltype(mat)>::Flags & Eigen::RowMajorBit ? 1 : 0;
    using Eigen::Index;
    const Index outer = IsRowMajor ? row : col;
    const Index inner = IsRowMajor ? col : row;

    auto* outerIndexPtr = mat.outerIndexPtr();
    auto* innerNonZeros = mat.innerNonZeroPtr();
    const auto start = outerIndexPtr[outer];
    const auto end = innerNonZeros ? outerIndexPtr[outer] + innerNonZeros[outer]
                                   : outerIndexPtr[outer + 1];
    TA_ASSERT(end >= start &&
              "you probably called coeffRef on a non finalized matrix");
    if (end <= start) return nullptr;
    const Index p = mat.data().searchLowerIndex(
        start, end - 1,
        (typename std::decay_t<decltype(mat)>::StorageIndex)inner);
    if ((p < end) && (mat.data().index(p) == inner))
      return &(mat.data().value(p));
    else
      return nullptr;
  }

};  // class EigenSparseTile

// configure TA traits to be usable as tile
namespace TiledArray {
namespace detail {
template <typename T, typename TagType>
struct is_tensor_helper<EigenSparseTile<T, TagType>> : public std::true_type {};
template <typename T, typename TagType>
struct is_contiguous_tensor_helper<EigenSparseTile<T, TagType>>
    : public std::false_type {};
}  // namespace detail
}  // namespace TiledArray

// Permutation operation

// returns a tile for which result[perm ^ i] = tile[i]
template <
    typename T, typename TagType, typename Perm,
    typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
EigenSparseTile<T, TagType> permute(const EigenSparseTile<T, TagType>& tile,
                                    const Perm& perm) {
  return EigenSparseTile<T, TagType>(tile.matrix().transpose(),
                                     perm * tile.range());
}

// Addition operations

// sparse_result[i] = sparse_arg1[i] + sparse_arg2[i]
template <typename T, typename TagType>
EigenSparseTile<T, TagType> add(const EigenSparseTile<T, TagType>& arg1,
                                const EigenSparseTile<T, TagType>& arg2) {
  TA_ASSERT(arg1.range() == arg2.range());

  return EigenSparseTile<T, TagType>(arg1.matrix() + arg2.matrix(),
                                     arg1.range());
}

//// dense_result[i] = dense_arg1[i] + sparse_arg2[i]
// template <typename T, typename TagType>
// TiledArray::Tensor<T> add(const TiledArray::Tensor<T>& arg1,
//                           const EigenSparseTile<T, TagType>& arg2) {
//   TA_ASSERT(arg1.range() == arg2.range());
//
//   // this could be done better ...
//   return TiledArray::add(arg1, static_cast<TiledArray::Tensor<T>>(arg2));
// }
//
//// dense_result[i] = sparse_arg1[i] + dense_arg2[i]
// template <typename T, typename TagType>
// TiledArray::Tensor<T> add(const EigenSparseTile<T, TagType>& arg1,
//                           const TiledArray::Tensor<T>& arg2) {
//   return TiledArray::add(arg2, static_cast<TiledArray::Tensor<T>>(arg1));
// }

// dense_result[perm ^ i] = dense_arg1[i] + sparse_arg2[i]
template <
    typename T, typename TagType, typename Perm,
    typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
TiledArray::Tensor<T> add(const TiledArray::Tensor<T>& arg1,
                          const EigenSparseTile<T, TagType>& arg2,
                          const Perm& perm) {
  TA_ASSERT(arg1.range() == arg2.range());

  // this could be done better ...
  return TiledArray::permute(
      TiledArray::add(arg1, static_cast<TiledArray::Tensor<T>>(arg2)), perm);
}

// sparse_result[i] += sparse_arg[i]
template <typename T, typename TagType>
EigenSparseTile<T, TagType>& add_to(EigenSparseTile<T, TagType>& result,
                                    const EigenSparseTile<T, TagType>& arg) {
  TA_ASSERT(result.range() == arg.range());

  result.matrix() += arg.matrix();
  return result;
}

// dense_result[i] += sparse_arg[i]
template <typename T, typename TagType>
TiledArray::Tensor<T>& add_to(TiledArray::Tensor<T>& result,
                              const EigenSparseTile<T, TagType>& arg) {
  TA_ASSERT(result.range() == arg.range());

  auto nrows = result.range().extent()[0];
  auto ncols = result.range().extent()[1];
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
             Eigen::AutoAlign>
      result_view(result.data(), nrows, ncols);
  result_view += arg.matrix();
  return result;
}

#define MULT_DENSE_SPARSE_TO_SPARSE 0

#if MULT_DENSE_SPARSE_TO_SPARSE
// Multiplication operations (Hadamard product)

// sparse_result[perm ^ i] = dense_arg1[i] * sparse_arg2[i]
template <
    typename T, typename TagType, typename Perm,
    typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
EigenSparseTile<T, TagType> mult(const TiledArray::Tensor<T>& arg1,
                                 const EigenSparseTile<T, TagType>& arg2,
                                 const Perm& perm) {
  TA_ASSERT(arg1.range() == arg2.range());
  TA_ASSERT(perm.size() == 2);
  const auto identity_perm = (perm[0] == 0);

  typedef typename EigenSparseTile<T, TagType>::matrix_type matrix_type;
  typedef typename matrix_type::Index idx_t;
  auto arg2_mat = arg2.matrix();
  auto lobound = arg2.range().lobound_data();
  std::vector<Eigen::Triplet<T>> datavec;
  // drive Hadamard by the sparse matrix
  for (idx_t k = 0; k < arg2_mat.outerSize(); ++k)
    for (typename matrix_type::InnerIterator it(arg2_mat, k); it; ++it) {
      auto row = it.row();
      auto col = it.col();
      datavec.push_back(Eigen::Triplet<T>(
          row, col, it.value() * arg1(row + lobound[0], col + lobound[1])));
    }
  matrix_type result(arg2_mat.rows(), arg2_mat.cols());
  result.setFromTriplets(datavec.begin(), datavec.end());
  if (not identity_perm) result = result.transpose();
  return EigenSparseTile<T, TagType>(result, arg2.range());
}

// sparse_result[i] = dense_arg1[i] * sparse_arg2[i]
template <typename T, typename TagType>
EigenSparseTile<T, TagType> mult(const TiledArray::Tensor<T>& arg1,
                                 const EigenSparseTile<T, TagType>& arg2) {
  static_assert(
      !TiledArray::detail::is_tensor_of_tensor_v<TiledArray::Tensor<T>>);
  auto iperm = Permutation::identity(2);
  return mult(arg1, arg2, iperm);
}

// sparse_result[i] *= dense_arg1[i]
template <typename T, typename TagType>
EigenSparseTile<T, TagType>& mult_to(EigenSparseTile<T, TagType>& result,
                                     const TiledArray::Tensor<T>& arg1) {
  TA_ASSERT(result.range() == arg1.range());

  typedef typename EigenSparseTile<T, TagType>::matrix_type matrix_type;
  auto mat = result.matrix();
  auto lobound = result.range().lobound_data();
  typedef typename matrix_type::Index idx_t;
  // drive Hadamard by the sparse matrix
  for (idx_t k = 0; k < mat.outerSize(); ++k)
    for (typename matrix_type::InnerIterator it(mat, k); it; ++it) {
      auto row = it.row();
      auto col = it.col();
      it.valueRef() *= arg1(row + lobound[0], col + lobound[1]);
    }
  return result;
}

// sparse_result[perm ^ i] = binary(dense_arg1[i], sparse_arg2[i], op)
template <
    typename T, typename TagType, typename Op, typename Perm,
    typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
EigenSparseTile<T, TagType> binary(const TiledArray::Tensor<T>& arg1,
                                   const EigenSparseTile<T, TagType>& arg2,
                                   Op&& op, const Perm& perm) {
  abort();
  return {};
}

// sparse_result[i] = binary(dense_arg1[i], sparse_arg2[i], op)
template <typename T, typename TagType, typename Op>
EigenSparseTile<T, TagType> binary(const TiledArray::Tensor<T>& arg1,
                                   const EigenSparseTile<T, TagType>& arg2,
                                   Op&& op) {
  static_assert(
      !TiledArray::detail::is_tensor_of_tensor_v<TiledArray::Tensor<T>>);
  auto iperm = Permutation::identity(2);
  return binary(arg1, arg2, std::forward<Op>(op), iperm);
}

// Contraction operation

// GEMM operation with fused indices as defined by gemm_config:
// sparse_result[i,j] = dense_arg1[i,k] * sparse_arg2[k,j]
template <typename T, typename TagType>
EigenSparseTile<T, TagType> gemm(
    const TiledArray::Tensor<T>& arg1, const EigenSparseTile<T, TagType>& arg2,
    const typename std::common_type<
        typename TiledArray::Tensor<T>::numeric_type,
        typename EigenSparseTile<T, TagType>::numeric_type>::type factor,
    const TiledArray::math::GemmHelper& gemm_config) {
  abort();
  return {};
}

// GEMM operation with fused indices as defined by gemm_config:
// sparse_result[i,j] = dense_arg1[i,k] * sparse_arg2[k,j]
template <typename T, typename TagType>
void gemm(EigenSparseTile<T, TagType>& result,
          const TiledArray::Tensor<T>& arg1,
          const EigenSparseTile<T, TagType>& arg2,
          const typename std::common_type<
              typename TiledArray::Tensor<T>::numeric_type,
              typename EigenSparseTile<T, TagType>::numeric_type>::type factor,
          const TiledArray::math::GemmHelper& gemm_config) {
  abort();
}

#else  // not MULT_DENSE_SPARSE_TO_SPARSE

// Multiplication operations (Hadamard product)

// dense_result[perm ^ i] = dense_arg1[i] * sparse_arg2[i]
template <
    typename T, typename TagType, typename Perm,
    typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
TiledArray::Tensor<T> mult(const TiledArray::Tensor<T>& arg1,
                           const EigenSparseTile<T, TagType>& arg2,
                           const Perm& perm) {
  TA_ASSERT(arg1.range() == arg2.range());
  TA_ASSERT(perm.size() == 2);
  const auto identity_perm = (*perm.begin() == 0);

  typedef typename EigenSparseTile<T, TagType>::matrix_type matrix_type;
  typedef typename matrix_type::Index idx_t;
  auto arg2_mat = arg2.matrix();
  auto lobound = arg2.range().lobound_data();
  TiledArray::Tensor<T> result(perm * arg1.range(), 0);

  // drive Hadamard by the sparse matrix
  for (idx_t k = 0; k < arg2_mat.outerSize(); ++k)
    for (typename matrix_type::InnerIterator it(arg2_mat, k); it; ++it) {
      auto row = it.row();
      auto col = it.col();
      auto drow = row + lobound[0];
      auto dcol = col + lobound[1];
      if (identity_perm)
        result(drow, dcol) = arg1(drow, dcol) * it.value();
      else
        result(dcol, drow) = arg1(drow, dcol) * it.value();
    }

  return result;
}

// dense_result[i] = dense_arg1[i] * sparse_arg2[i]
template <typename T, typename TagType>
TiledArray::Tensor<T> mult(const TiledArray::Tensor<T>& arg1,
                           const EigenSparseTile<T, TagType>& arg2) {
  auto iperm = Permutation::identity(2);
  return mult(arg1, arg2, iperm);
}

// dense_result[i] *= sparse_arg1[i]
template <typename T, typename TagType>
TiledArray::Tensor<T>& mult_to(TiledArray::Tensor<T>& result,
                               const EigenSparseTile<T, TagType>& arg1) {
  TA_ASSERT(result.range() == arg1.range());

  typedef typename EigenSparseTile<T, TagType>::matrix_type matrix_type;
  auto mat = arg1.matrix();
  auto lobound = arg1.range().lobound_data();
  typedef typename matrix_type::Index idx_t;
  // drive Hadamard by the sparse matrix
  for (idx_t k = 0; k < mat.outerSize(); ++k)
    for (typename matrix_type::InnerIterator it(mat, k); it; ++it) {
      auto row = it.row();
      auto col = it.col();
      result(row + lobound[0], col + lobound[1]) *= it.value();
    }
  return result;
}

// dense_result[perm ^ i] = binary(dense_arg1[i], sparse_arg2[i], op)
template <
    typename T, typename TagType, typename Op, typename Perm,
    typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
TiledArray::Tensor<T> binary(const TiledArray::Tensor<T>& arg1,
                             const EigenSparseTile<T, TagType>& arg2, Op&& op,
                             const Perm& perm) {
  abort();
  return {};
}

// dense_result[i] = binary(dense_arg1[i], sparse_arg2[i], op)
template <typename T, typename TagType, typename Op>
TiledArray::Tensor<T> binary(const TiledArray::Tensor<T>& arg1,
                             const EigenSparseTile<T, TagType>& arg2, Op&& op) {
  static_assert(
      !TiledArray::detail::is_tensor_of_tensor_v<TiledArray::Tensor<T>>);
  auto iperm = Permutation::identity(2);
  return binary(arg1, arg2, std::forward<Op>(op), iperm);
}

// Contraction operation

// GEMM operation with fused indices as defined by gemm_config:
// dense_result[i,j] = dense_arg1[i,k] * sparse_arg2[k,j]
template <typename T, typename TagType>
TiledArray::Tensor<T> gemm(
    const TiledArray::Tensor<T>& arg1, const EigenSparseTile<T, TagType>& arg2,
    const typename std::common_type<
        typename TiledArray::Tensor<T>::numeric_type,
        typename EigenSparseTile<T, TagType>::numeric_type>::type factor,
    const TiledArray::math::GemmHelper& gemm_config) {
  // only simple outer product implemented at the moment
  TA_ASSERT(gemm_config.result_rank() ==
            gemm_config.left_rank() + gemm_config.right_rank());
  TA_ASSERT(gemm_config.left_rank() == arg1.range().rank());
  TA_ASSERT(gemm_config.right_rank() == arg2.range().rank());

  auto result_range = gemm_config.make_result_range<TiledArray::Range>(
      arg1.range(), arg2.range());
  TiledArray::Tensor<T> result(result_range, 0);

  auto arg1_lobound = arg1.range().lobound_data();
  auto arg1_upbound = arg1.range().upbound_data();
  typedef typename EigenSparseTile<T, TagType>::matrix_type matrix_type;
  typedef typename matrix_type::Index idx_t;
  auto arg2_mat = arg2.matrix();
  auto arg2_lobound = arg2.range().lobound_data();

  // drive outer product by the sparse matrix
  for (idx_t k = 0; k < arg2_mat.outerSize(); ++k)
    for (typename matrix_type::InnerIterator it(arg2_mat, k); it; ++it) {
      auto row = it.row();
      auto col = it.col();
      auto value = it.value();
      auto drow = row + arg2_lobound[0];
      auto dcol = col + arg2_lobound[1];

      // make a slice of the result ...
      // TODO can this be done via outer product of TensorInterfaces?
      //        auto result_slice = result.block(
      //          {arg1_lobound[0],arg1_lobound[1],drow,dcol},
      //          {arg1_upbound[0],arg1_upbound[1],drow+1,dcol+1}
      //        );

      // and evaluate the result slice
      for (auto i0 = arg1_lobound[0]; i0 != arg1_upbound[0]; ++i0) {
        for (auto i1 = arg1_lobound[1]; i1 != arg1_upbound[1]; ++i1) {
          result(i0, i1, drow, dcol) = arg1(i0, i1) * value;
        }
      }
    }

  return result;
}

// GEMM operation with fused indices as defined by gemm_config:
// dense_result[i,j] = dense_arg1[i,k] * sparse_arg2[k,j]
template <typename T, typename TagType>
void gemm(TiledArray::Tensor<T>& result, const TiledArray::Tensor<T>& arg1,
          const EigenSparseTile<T, TagType>& arg2,
          const typename std::common_type<
              typename TiledArray::Tensor<T>::numeric_type,
              typename EigenSparseTile<T, TagType>::numeric_type>::type factor,
          const TiledArray::math::GemmHelper& gemm_config) {
  abort();
}
#endif

// Reduction operations

// Sum of hyper diagonal elements
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type trace(
    const EigenSparseTile<T, TagType>& arg);
// foreach(i) result += arg[i]
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type sum(
    const EigenSparseTile<T, TagType>& arg);
// foreach(i) result *= arg[i]
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type product(
    const EigenSparseTile<T, TagType>& arg);
// foreach(i) result += arg[i] * arg[i]
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type squared_norm(
    const EigenSparseTile<T, TagType>& arg);
// sqrt(squared_norm(arg))
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type norm(
    const EigenSparseTile<T, TagType>& arg);
// foreach(i) result = max(result, arg[i])
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type max(
    const EigenSparseTile<T, TagType>& arg);
// foreach(i) result = min(result, arg[i])
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type min(
    const EigenSparseTile<T, TagType>& arg);
// foreach(i) result = max(result, abs(arg[i]))
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type abs_max(
    const EigenSparseTile<T, TagType>& arg);
// foreach(i) result = max(result, abs(arg[i]))
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type abs_min(
    const EigenSparseTile<T, TagType>& arg);

namespace TiledArray {

// convert TiledArray::Tensor<T> to EigenSparseTile<T>
template <typename T, typename TagType>
class Cast<EigenSparseTile<T, TagType>, TiledArray::Tensor<T>> {
 public:
  typedef EigenSparseTile<T, TagType> result_type;
  typedef TiledArray::Tensor<T> tile_type;

  result_type operator()(const tile_type& arg) const {
    typedef Eigen::Triplet<T> Triplet;
    std::vector<Triplet> tripletList;
    tripletList.reserve(arg.size());
    auto extent = arg.range().extent_data();
    auto lobound = arg.range().lobound_data();
    auto nrows = extent[0];
    auto ncols = extent[1];
    for (decltype(nrows) r = 0; r != nrows; ++r) {
      for (decltype(ncols) c = 0; c != ncols; ++c) {
        auto v_rc = arg(r + lobound[0], c + lobound[1]);
        if (v_rc != 0) tripletList.push_back(Triplet(r, c, v_rc));
      }
    }

    typename EigenSparseTile<T, TagType>::matrix_type result(nrows, ncols);
    result.setFromTriplets(tripletList.begin(), tripletList.end());

    return EigenSparseTile<T, TagType>(std::move(result), arg.range());
  }
};

}  // namespace TiledArray

namespace madness {
namespace archive {
template <class Archive, typename T>
struct ArchiveLoadImpl<Archive, Eigen::Triplet<T>> {
  static inline void load(const Archive& ar, Eigen::Triplet<T>& obj) {
    int row, col;
    T value;
    ar & row & col & value;
    obj = Eigen::Triplet<T>(row, col, value);
  }
};

template <class Archive, typename T>
struct ArchiveStoreImpl<Archive, Eigen::Triplet<T>> {
  static inline void store(const Archive& ar, const Eigen::Triplet<T>& obj) {
    ar & obj.row() & obj.col() & obj.value();
  }
};
}  // namespace archive
}  // namespace madness

#endif  // TILEDARRAY_TEST_SPARSE_TILE_H__INCLUDED
