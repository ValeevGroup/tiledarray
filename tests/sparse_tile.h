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

#include <tuple>
#include <memory>
#include <Eigen/SparseCore>

#include <tiledarray_fwd.h>

#include <TiledArray/madness.h>

// Array class
#include <TiledArray/tensor.h>
#include <TiledArray/tile.h>
#include <TiledArray/tile_op/tile_interface.h>

// Array policy classes
#include <TiledArray/policies/dense_policy.h>
#include <TiledArray/policies/sparse_policy.h>

// sparse 2-dimensional matrix type
template <typename T, typename TagType = std::tuple<> >
class EigenSparseTile {
public:
  // Concept typedefs
  typedef TiledArray::Range range_type; // Tensor range type
  typedef T value_type; // Element type
  typedef T numeric_type; // The scalar type that is compatible with value_type
  typedef size_t size_type; // Size type
  // other typedefs
  typedef Eigen::SparseMatrix<T> matrix_type;

  typedef std::tuple<matrix_type, range_type> impl_type;
public:

  /// makes an uninitialized matrix
  EigenSparseTile() = default;

  /// Shallow copy constructor; see MyTensor::clone() for deep copy
  EigenSparseTile(const EigenSparseTile&) = default;

  /// Shallow assignment operator; see MyTensor::clone() for deep copy
  EigenSparseTile&
  operator=(const EigenSparseTile& other) = default;

  /// makes an uninitialized matrix
  explicit EigenSparseTile(const range_type& r) :
          impl_(
              std::make_shared < impl_type
                  > (std::make_tuple(matrix_type(r.extent()[0], r.extent()[1]), r)))
  {
    TA_ASSERT(r.extent()[0] > 0);
    TA_ASSERT(r.extent()[1] > 0);
  }

  /// ctor using sparse matrix
  EigenSparseTile(matrix_type&& mat, const range_type& range) :
      impl_(std::make_shared < impl_type > (std::make_tuple(std::move(mat), range)))
  {
    TA_ASSERT(mat.rows() == range.extent()[0]);
    TA_ASSERT(mat.cols() == range.extent()[1]);
  }

  /// ctor using sparse matrix
  EigenSparseTile(const matrix_type& mat, const range_type& range) :
      impl_(std::make_shared < impl_type > (std::make_tuple(mat, range)))
  {
    TA_ASSERT(mat.rows() == range.extent()[0]);
    TA_ASSERT(mat.cols() == range.extent()[1]);
  }

  // Deep copy
  EigenSparseTile clone() const {
    EigenSparseTile result;
    result.impl_ = std::make_shared < impl_type > (this->impl_);
    return result;
  }

  // copies
  template <typename AnotherTagType>
  explicit operator EigenSparseTile<T,AnotherTagType>() const {
    return EigenSparseTile<T, AnotherTagType> {this->data(), this->range()};
  }

  explicit operator TiledArray::Tensor<T>() const {
    TiledArray::Tensor<T> result(this->range(), T(0));
    auto nrows = range().extent()[0];
    auto ncols = range().extent()[1];
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
      result_view(result.data(), nrows, ncols);
    result_view = data();
    return result;
  }

  // Tile range accessor
  const range_type& range() const {
    return std::get < 1 > (*impl_);
  }

  // data matrix access
  const matrix_type& data() const {
    return std::get < 0 > (*impl_);
  }

  /// data read-write accessor
  template <typename Index>
  value_type& operator[](const Index& idx) {
    auto start = range().lobound_data();
    return std::get < 0 > (*impl_).coeffRef(idx[0] - start[0], idx[1] - start[1]);
  }

  /// Maximum # of elements in the tile
  size_type size() const {
    std::get < 0 > (*impl_).volume();
  }

  // Initialization check. False if the tile is fully initialized.
  bool empty() const {
    return impl_;
  }

  // MADNESS compliant serialization

  // output
  template <typename Archive, typename std::enable_if<
      madness::archive::is_output_archive<Archive>::value>::type* = nullptr>
  void serialize(Archive& ar) {
    if(impl_) {
      ar & true;
      auto mat = this->data();
      std::vector < Eigen::Triplet < T >> datavec;
      datavec.reserve(mat.size());
      for(size_t k = 0; k < mat.outerSize(); ++k)
        for(typename matrix_type::InnerIterator it(mat, k); it; ++it) {
          datavec.push_back(Eigen::Triplet < T > (it.row(), it.col(), it.value()));
        }
      ar & datavec & this->range();
    } else {
      ar & false;
    }
  }

  // output
  template <typename Archive, typename std::enable_if<
      madness::archive::is_input_archive<Archive>::value>::type* = nullptr>
  void serialize(Archive& ar) {
    bool have_impl;
    ar & have_impl;
    if(have_impl) {
      std::vector < Eigen::Triplet < T >> datavec;
      range_type range;
      ar & datavec & range;
      auto extents = range.extent();
      matrix_type mat(extents[0], extents[1]);
      mat.setFromTriplets(datavec.begin(), datavec.end());
      impl_ = std::make_shared < impl_type > (std::make_pair(std::move(mat), std::move(range)));
    } else {
      impl_ = 0;
    }
  }

  // Scaling operations

  // result[i] = (*this)[i] * factor
  EigenSparseTile
  scale(const numeric_type factor) const;
  // result[perm ^ i] = (*this)[i] * factor
  EigenSparseTile
  scale(const numeric_type factor, const TiledArray::Permutation& perm) const;
  // (*this)[i] *= factor
  EigenSparseTile&
  scale_to(const numeric_type factor) const;

private:
  std::shared_ptr<impl_type> impl_;

}; // class EigenSparseTile


  // Permutation operation

  // returns a tile for which result[perm ^ i] = tile[i]
  template <typename T, typename TagType>
  EigenSparseTile<T, TagType> permute(const EigenSparseTile<T, TagType>& tile,
				      const TiledArray::Permutation& perm) {
    TA_ASSERT(perm[0] != 0);
    return EigenSparseTile<T, TagType>(tile.data().transpose(),
				       perm * tile.range());
  }

#if 0
  // Addition operations

  // result[i] = arg1[i] + arg2[i]
  MyTensor add(const MyTensor& arg1,
      const MyTensor& arg2);
  // result[i] = (arg1[i] + arg2[i]) * factor
  MyTensor add(const MyTensor& arg1,
      const MyTensor& arg2,
      const MyTensor::value_type factor);
  // result[i] = arg[i] + value
  MyTensor add(const MyTensor& arg,
      const MyTensor::value_type& value);

  // result[perm ^ i] = arg1[i] + arg2[i]
  MyTensor add(const MyTensor& arg1,
      const MyTensor& arg2,
      const TiledArray::Permutation& perm);
  // result[perm ^ i] = (arg1[i] + arg2[i]) * factor
  MyTensor add(const MyTensor& arg1,
      const MyTensor& arg2,
      const MyTensor::numeric_type factor,
      const TiledArray::Permutation& perm);
  // result[perm ^ i] = arg[i] + value
  MyTensor add(const MyTensor& arg,
      const MyTensor::value_type& value,
      const TiledArray::Permutation& perm);

  // result[i] += arg[i]
  void add_to(MyTensor& result,
      const MyTensor& arg);
  // (result[i] += arg[i]) *= factor
  void add_to(MyTensor& result,
      const MyTensor& arg,
      const MyTensor::numeric_type factor);
  // result[i] += value
  void add_to(MyTensor& result,
      const MyTensor::value_type& value);

  // Subtraction operations

  // result[i] = arg1[i] - arg2[i]
  MyTensor subt(const MyTensor& arg1,
      const MyTensor& arg2);
  // result[i] = (arg1[i] - arg2[i]) * factor
  MyTensor subt(const MyTensor& arg1,
      const MyTensor& arg2,
      const MyTensor::numeric_type factor);
  // result[i] = arg[i] - value
  MyTensor subt(const MyTensor& arg,
      const MyTensor::value_type& value);

  // result[perm ^ i] = arg1[i] - arg2[i]
  MyTensor subt(const MyTensor& arg1,
      const MyTensor& arg2,
      const TiledArray::Permutation& perm);
  // result[perm ^ i] = (arg1[i] - arg2[i]) * factor
  MyTensor subt(const MyTensor& arg1,
      const MyTensor& arg2,
      const MyTensor::numeric_type factor,
      const TiledArray::Permutation& perm);
  // result[perm ^ i] = arg[i] - value
  MyTensor subt(const MyTensor& arg,
      const MyTensor::value_type value,
      const TiledArray::Permutation& perm);

  // result[i] -= arg[i]
  void subt_to(MyTensor& result,
      const MyTensor& arg);
  // (result[i] -= arg[i]) *= factor
  void subt_to(MyTensor& result,
      const MyTensor& arg,
      const MyTensor::numeric_type factor);
  // result[i] -= value
  void subt_to(MyTensor& result,
      const MyTensor::value_type& value);

  // Multiplication operations

  // result[i] = arg1[i] * arg2[i]
  MyTensor mult(const MyTensor& arg1,
      const MyTensor& arg2);
  // result[i] = (arg1[i] * arg2[i]) * factor
  MyTensor mult(const MyTensor& arg1,
      const MyTensor& arg2,
      const MyTensor::numeric_type factor);

  // result[perm ^ i] = arg1[i] * arg2[i]
  MyTensor mult(const MyTensor& arg1,
      const MyTensor& arg2,
      const TiledArray::Permutation& perm);
  // result[perm^ i] = (arg1[i] * arg2[i]) * factor
  MyTensor mult(const MyTensor& arg1,
      const MyTensor& arg2,
      const MyTensor::numeric_type factor,
      const TiledArray::Permutation& perm);

  // result[i] *= arg[i]
  void mult_to(MyTensor& result,
      const MyTensor& arg);
  // (result[i] *= arg[i]) *= factor
  void mult_to(MyTensor& result,
      const MyTensor& arg,
      const MyTensor::numeric_type factor);

  // Negation operations

  // result[i] = -(arg[i])
  MyTensor neg(const MyTensor& arg);
  // result[perm ^ i] = -(arg[i])
  MyTensor neg(const MyTensor& arg,
      const TiledArray::Permutation& perm);
  // result[i] = -(result[i])
  void neg_to(MyTensor& result);

  // Contraction operations

  // GEMM operation with fused indices as defined by gemm_config; multiply arg1 by arg2, return the result
  MyTensor gemm(const MyTensor& arg1,
      const MyTensor& arg2,
      const MyTensor::numeric_type factor,
      const TiledArray::math::GemmHelper& gemm_config);

  // GEMM operation with fused indices as defined by gemm_config; multiply left by right, store to result
  void gemm(MyTensor& result,
      const MyTensor& arg1,
      const MyTensor& arg2,
      const MyTensor::numeric_type factor,
      const TiledArray::math::GemmHelper& gemm_config);
#endif

// Reduction operations

// Sum of hyper diagonal elements
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type
trace(const EigenSparseTile<T, TagType>& arg);
// foreach(i) result += arg[i]
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type
sum(const EigenSparseTile<T, TagType>& arg);
// foreach(i) result *= arg[i]
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type
product(const EigenSparseTile<T, TagType>& arg);
// foreach(i) result += arg[i] * arg[i]
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type
squared_norm(const EigenSparseTile<T, TagType>& arg);
// sqrt(squared_norm(arg))
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type
norm(const EigenSparseTile<T, TagType>& arg);
// foreach(i) result = max(result, arg[i])
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type
max(const EigenSparseTile<T, TagType>& arg);
// foreach(i) result = min(result, arg[i])
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type
min(const EigenSparseTile<T, TagType>& arg);
// foreach(i) result = max(result, abs(arg[i]))
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type
abs_max(const EigenSparseTile<T, TagType>& arg);
// foreach(i) result = max(result, abs(arg[i]))
template <typename T, typename TagType>
typename EigenSparseTile<T, TagType>::numeric_type
abs_min(const EigenSparseTile<T, TagType>& arg);

namespace TiledArray {

  // convert TiledArray::Tensor<T> to EigenSparseTile<T>
  template <typename T, typename TagType>
  class Cast<EigenSparseTile<T, TagType>, TiledArray::Tensor<T> > {
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
      for(auto r = 0; r != nrows; ++r) {
        for(auto c = 0; c != ncols; ++c) {
          auto v_rc = arg(r + lobound[0], c + lobound[1]);
          if(v_rc != 0)
            tripletList.push_back(Triplet(r, c, v_rc));
        }
      }

      typename EigenSparseTile<T, TagType>::matrix_type result(nrows, ncols);
      result.setFromTriplets(tripletList.begin(), tripletList.end());

      return EigenSparseTile<T, TagType>(std::move(result), arg.range());
    }
  };

} // namespace TiledArray

namespace madness {
  namespace archive {
    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, Eigen::Triplet<T>> {
      static inline void load(const Archive& ar, Eigen::Triplet<T>& obj) {
        int row, col;
        T value;
        ar & row & col & value;
        obj = Eigen::Triplet < T > (row, col, value);
      }
    };

    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, Eigen::Triplet<T>> {
      static inline void store(const Archive& ar, const Eigen::Triplet<T>& obj) {
        ar & obj.row() & obj.col() & obj.value();
      }
    };
  }
}

#endif // TILEDARRAY_TEST_SPARSE_TILE_H__INCLUDED
