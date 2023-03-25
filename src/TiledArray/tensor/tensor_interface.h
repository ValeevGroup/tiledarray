/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  tensor_interface.h
 *  May 29, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_TENSOR_VIEW_H__INCLUDED
#define TILEDARRAY_TENSOR_TENSOR_VIEW_H__INCLUDED

#include <TiledArray/tensor/complex.h>
#include <TiledArray/tensor/kernels.h>
#include <TiledArray/tile_interface/permute.h>
#include <TiledArray/type_traits.h>

namespace TiledArray {

// Forward declarations
template <typename T, typename A>
class Tensor;
class Range;
namespace detail {
template <typename T, typename Range, typename OpResult>
class TensorInterface;
}

template <typename T, typename Range_, typename OpResult, typename Index1,
          typename Index2>
void remap(detail::TensorInterface<T, Range_, OpResult>&, T* const,
           const Index1&, const Index2&);
template <typename T, typename Range_, typename OpResult, typename Index1,
          typename Index2>
void remap(detail::TensorInterface<const T, Range_, OpResult>&, T* const,
           const Index1&, const Index2&);
template <typename T, typename Range_, typename OpResult, typename Index1,
          typename Index2>
void remap(detail::TensorInterface<T, Range_, OpResult>&, T* const,
           const std::initializer_list<Index1>&,
           const std::initializer_list<Index2>&);
template <typename T, typename Range_, typename OpResult, typename Index1,
          typename Index2>
void remap(detail::TensorInterface<const T, Range_, OpResult>&, T* const,
           const std::initializer_list<Index1>&,
           const std::initializer_list<Index2>&);

namespace detail {

/// Tensor interface for external data

/// This class allows users to construct a tensor object using data from an
/// external source. \c TensorInterface objects can be used
/// with each other and \c Tensor objects in any of the arithmetic
/// operations.
/// \warning No ownership of the data pointer used to construct
/// \c TensorInterface objects. Therefore, it is the user's responsibility
/// to manage the lifetime of the data.
/// \warning This is not appropriate for use as a tile object as it does not
/// own the data it references. Use \c Tensor for that purpose.
/// \tparam T The tensor value type
/// \tparam R The range type
/// \tparam OpResult tensor type used as the return type from operations that
/// produce a tensor
template <typename T, typename R, typename OpResult>
class TensorInterface {
 public:
  typedef TensorInterface<T, R, OpResult>
      TensorInterface_;                                  ///< This class type
  typedef R range_type;                                  ///< Tensor range type
  typedef typename range_type::index1_type index1_type;  ///< 1-index type
  typedef typename range_type::ordinal_type ordinal_type;  ///< Ordinal type
  using size_type = ordinal_type;
  typedef
      typename std::remove_const<T>::type value_type;  ///< Array element type
  typedef typename std::add_lvalue_reference<T>::type
      reference;  ///< Element reference type
  typedef
      typename std::add_lvalue_reference<typename std::add_const<T>::type>::type
          const_reference;  ///< Element reference type
  typedef typename std::add_pointer<T>::type pointer;  ///< Element pointer type
  typedef typename std::add_pointer<typename std::add_const<T>::type>::type
      const_pointer;                                ///< Element pointer type
  typedef typename std::ptrdiff_t difference_type;  ///< Difference type
  typedef typename detail::numeric_type<value_type>::type
      numeric_type;  ///< the numeric type that supports T
  typedef typename detail::scalar_type<value_type>::type
      scalar_type;  ///< the scalar type that supports T

  typedef OpResult result_tensor;
  ///< Tensor type used as the return type from operations that produce a tensor

 private:
  template <typename X>
  using numeric_t = typename TiledArray::detail::numeric_type<X>::type;

  template <typename, typename, typename>
  friend class TensorInterface;

  template <typename U, typename Range_, typename OpResult_, typename Index1,
            typename Index2>
  friend void TiledArray::remap(detail::TensorInterface<U, Range_, OpResult_>&,
                                U* const, const Index1&, const Index2&);
  template <typename U, typename Range_, typename OpResult_, typename Index1,
            typename Index2>
  friend void TiledArray::remap(
      detail::TensorInterface<const U, Range_, OpResult_>&, U* const,
      const Index1&, const Index2&);
  template <typename U, typename Range_, typename OpResult_, typename Index1,
            typename Index2>
  friend void TiledArray::remap(detail::TensorInterface<U, Range_, OpResult_>&,
                                U* const, const std::initializer_list<Index1>&,
                                const std::initializer_list<Index2>&);
  template <typename U, typename Range_, typename OpResult_, typename Index1,
            typename Index2>
  friend void TiledArray::remap(
      detail::TensorInterface<const U, Range_, OpResult_>&, U* const,
      const std::initializer_list<Index1>&,
      const std::initializer_list<Index2>&);

  range_type range_;  ///< View sub-block range
  pointer data_;      ///< Pointer to the original tensor data

 public:
  /// Compiler generated functions
  TensorInterface() = delete;
  ~TensorInterface() = default;
  TensorInterface(const TensorInterface_&) = default;
  TensorInterface(TensorInterface_&&) = default;
  TensorInterface_& operator=(const TensorInterface_&) = delete;
  TensorInterface_& operator=(TensorInterface_&&) = delete;

  /// Type conversion copy constructor

  /// \tparam U The value type of the other view
  /// \param other The other tensor view to be copied
  template <typename U, typename UOpResult,
            typename std::enable_if<std::is_convertible<
                typename TensorInterface<U, R, UOpResult>::pointer,
                pointer>::value>::type* = nullptr>
  TensorInterface(const TensorInterface<U, R, UOpResult>& other)
      : range_(other.range_), data_(other.data_) {}

  /// Type conversion move constructor

  /// \tparam U The value type of the other tensor view
  /// \param other The other tensor view to be moved
  template <typename U, typename UOpResult,
            typename std::enable_if<std::is_convertible<
                typename TensorInterface<U, R, UOpResult>::pointer,
                pointer>::value>::type* = nullptr>
  TensorInterface(TensorInterface<U, R, UOpResult>&& other)
      : range_(std::move(other.range_)), data_(other.data_) {
    other.data_ = nullptr;
  }

  /// Construct a new view of \c tensor

  /// \param range The range of this tensor
  /// \param data The data pointer for this tensor
  TensorInterface(const range_type& range, pointer data)
      : range_(range), data_(data) {
    TA_ASSERT(data);
  }

  /// Construct a new view of \c tensor

  /// \param range The range of this tensor
  /// \param data The data pointer for this tensor
  TensorInterface(range_type&& range, pointer data)
      : range_(std::move(range)), data_(data) {
    TA_ASSERT(data);
  }

  template <typename T1, typename std::enable_if<
                             detail::is_tensor<T1>::value>::type* = nullptr>
  TensorInterface_& operator=(const T1& other) {
    if constexpr (std::is_same_v<numeric_type, numeric_t<T1>>) {
      TA_ASSERT(data_ != other.data());
    }

    detail::inplace_tensor_op([](numeric_type& MADNESS_RESTRICT result,
                                 const numeric_t<T1> arg) { result = arg; },
                              *this, other);

    return *this;
  }

  /// Tensor range object accessor

  /// \return The tensor range object
  const range_type& range() const { return range_; }

  /// Tensor dimension size accessor

  /// \return The number of elements in the tensor
  ordinal_type size() const { return range_.volume(); }

  /// Data pointer accessor

  /// \return The data pointer of the parent tensor
  pointer data() const { return data_; }

  /// Element subscript accessor

  /// \param index The ordinal element index
  /// \return A const reference to the element at \c index.
  const_reference operator[](const ordinal_type index) const {
    TA_ASSERT(range_.includes(index));
    return data_[range_.ordinal(index)];
  }

  /// Element subscript accessor

  /// \param index The ordinal element index
  /// \return A const reference to the element at \c index.
  reference operator[](const ordinal_type index) {
    TA_ASSERT(range_.includes(index));
    return data_[range_.ordinal(index)];
  }

  /// Element accessor

  /// \tparam Index An integral type pack or a single coodinate index type
  /// \param idx The index pack
  template <typename... Index>
  reference operator()(const Index&... idx) {
    TA_ASSERT(range_.includes(idx...));
    return data_[range_.ordinal(idx...)];
  }

  /// Element accessor

  /// \tparam Index An integral type pack or a single coodinate index type
  /// \param idx The index pack
  template <typename... Index>
  const_reference operator()(const Index&... idx) const {
    TA_ASSERT(range_.includes(idx...));
    return data_[range_.ordinal(idx...)];
  }

  /// Check for empty view

  /// \return \c false
  constexpr bool empty() const { return false; }

  /// Swap tensor views

  /// \param other The view to be swapped
  void swap(TensorInterface_& other) {
    range_.swap(other.range_);
    std::swap(data_, other.data_);
  }

  /// Shift the lower and upper bound of this tensor view

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return Reference to \c this
  template <typename Index,
            typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
  TensorInterface_& shift_to(const Index& bound_shift) {
    range_.inplace_shift(bound_shift);
    return *this;
  }

  /// Shift the lower and upper bound of this tensor view

  /// \tparam Index An integral type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return Reference to \c this
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  TensorInterface_& shift_to(const std::initializer_list<Index>& bound_shift) {
    range_.inplace_shift(bound_shift);
    return *this;
  }

  /// Make a copy of this view shited by \p bound_shift

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return A copy of the shifted view
  template <typename Index,
            typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
  result_tensor shift(const Index& bound_shift) const {
    return result_tensor(range_.shift(bound_shift), data_);
  }

  /// Make a copy of this view shited by \p bound_shift

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return A copy of the shifted view
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  result_tensor shift(const std::initializer_list<Index>& bound_shift) const {
    return result_tensor(range_.shift(bound_shift), data_);
  }

  // Generic vector operations

  /// Use a binary, element wise operation to construct a new tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Op The binary operation type
  /// \param right The right-hand argument in the binary operation
  /// \param op The binary, element-wise operation
  /// \return A tensor where element \c i of the new tensor is equal to
  /// \c op(*this[i],other[i])
  template <typename Right, typename Op,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  result_tensor binary(const Right& right, Op&& op) const {
    //        return result_tensor(*this, right, op);
    result_tensor new_tensor(detail::clone_range(*this));
    detail::tensor_init(std::forward<Op>(op), new_tensor, *this, right);
    return new_tensor;
  }

  /// Use a binary, element wise operation to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Op The binary operation type
  /// \param right The right-hand argument in the binary operation
  /// \param op The binary, element-wise operation
  /// \param perm The permutation to be applied to this tensor
  /// \return A tensor where element \c i of the new tensor is equal to
  /// \c op(*this[i],other[i])
  template <
      typename Right, typename Op, typename Perm,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  result_tensor binary(const Right& right, Op&& op, const Perm& perm) const {
    //        return result_tensor(*this, right, op, perm);
    result_tensor new_tensor(outer(perm) * this->range());
    detail::tensor_init(std::forward<Op>(op), outer(perm), new_tensor, *this,
                        right);

    // If we actually have a ToT the inner permutation was not applied above so
    // we do that now
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<TensorInterface>;
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    if constexpr (is_tot && is_bperm) {
      if (inner_size(perm) != 0) {
        auto inner_perm = inner(perm);
        Permute<T, T> p;
        for (auto& x : new_tensor) x = p(x, inner_perm);
      }
    }

    return new_tensor;
  }

  /// Use a binary, element wise operation to modify this tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Op The binary operation type
  /// \param right The right-hand argument in the binary operation
  /// \param op The binary, element-wise operation
  /// \return A reference to this object
  /// \throw TiledArray::Exception When this tensor is empty.
  /// \throw TiledArray::Exception When \c other is empty.
  /// \throw TiledArray::Exception When the range of this tensor is not equal
  /// to the range of \c other.
  /// \throw TiledArray::Exception When this and \c other are the same.
  template <typename Right, typename Op,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  TensorInterface_& inplace_binary(const Right& right, Op&& op) {
    detail::inplace_tensor_op(std::forward<Op>(op), *this, right);
    return *this;
  }

  /// Use a unary, element wise operation to construct a new tensor

  /// \tparam Op The unary operation type
  /// \param op The unary, element-wise operation
  /// \return A tensor where element \c i of the new tensor is equal to
  /// \c op(*this[i])
  /// \throw TiledArray::Exception When this tensor is empty.
  template <typename Op>
  result_tensor unary(Op&& op) const {
    //        return result_tensor(*this, op);
    result_tensor new_tensor(detail::clone_range(*this));
    detail::tensor_init(std::forward<Op>(op), new_tensor, *this);
    return new_tensor;
  }

  /// Use a unary, element wise operation to construct a new, permuted tensor

  /// \tparam Op The unary operation type
  /// \param op The unary operation
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted tensor with elements that have been modified by \c op
  /// \throw TiledArray::Exception When this tensor is empty.
  /// \throw TiledArray::Exception The dimension of \c perm does not match
  /// that of this tensor.
  template <typename Op, typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_tensor unary(Op&& op, const Perm& perm) const {
    result_tensor new_tensor(outer(perm) * this->range());
    detail::tensor_init(std::forward<Op>(op), outer(perm), new_tensor, *this);

    // If we actually have a ToT the inner permutation was not applied above so
    // we do that now
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<TensorInterface>;
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    if constexpr (is_tot && is_bperm) {
      if (inner_size(perm) != 0) {
        auto inner_perm = inner(perm);
        Permute<T, T> p;
        for (auto& x : new_tensor) x = p(x, inner_perm);
      }
    }

    return new_tensor;
  }

  /// Use a unary, element wise operation to modify this tensor

  /// \tparam Op The unary operation type
  /// \param op The unary, element-wise operation
  /// \return A reference to this object
  /// \throw TiledArray::Exception When this tensor is empty.
  template <typename Op>
  TensorInterface_& inplace_unary(Op&& op) {
    detail::inplace_tensor_op(std::forward<Op>(op), *this);
    return *this;
  }

  // permute operation
  // construct a permuted copy of this tensor
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_tensor permute(const Perm& perm) const {
    auto op = [](const T& arg) { return arg; };
    result_tensor new_tensor(outer(perm) * this->range());
    TiledArray::detail::tensor_init(op, outer(perm), new_tensor, *this);

    // If we actually have a ToT the inner permutation was not applied above so
    // we do that now
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<TensorInterface>;
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    if constexpr (is_tot && is_bperm) {
      if (inner_size(perm) != 0) {
        auto inner_perm = inner(perm);
        Permute<T, T> p;
        for (auto& x : new_tensor) x = p(x, inner_perm);
      }
    }

    return new_tensor;
  }

  // Scale operation

  /// Construct a scaled copy of this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A new tensor where the elements of this tensor are scaled by
  /// \c factor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  result_tensor scale(const Scalar factor) const {
    return unary(
        [factor](const numeric_type a) -> numeric_type { return a * factor; });
  }

  /// Construct a scaled and permuted copy of this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements of this tensor are scaled by
  /// \c factor and permuted
  template <
      typename Scalar, typename Perm,
      typename std::enable_if<detail::is_numeric_v<Scalar> &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  result_tensor scale(const Scalar factor, const Perm& perm) const {
    return unary(
        [factor](const numeric_type a) -> numeric_type { return a * factor; },
        perm);
  }

  /// Scale this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  TensorInterface_& scale_to(const Scalar factor) {
    return inplace_unary(
        [factor](numeric_type& MADNESS_RESTRICT res) { res *= factor; });
  }

  // Addition operations

  /// Add this and \c other to construct a new tensors

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be added to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  result_tensor add(const Right& right) const {
    return binary(
        right,
        [](const numeric_type l, const numeric_t<Right> r) -> numeric_type {
          return l + r;
        });
  }

  /// Add this and \c other to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be added to this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other
  template <
      typename Right, typename Perm,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  result_tensor add(const Right& right, const Perm& perm) const {
    return binary(
        right,
        [](const numeric_type l, const numeric_t<Right> r) -> numeric_type {
          return l + r;
        },
        perm);
  }

  /// Scale and add this and \c other to construct a new tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be added to this tensor
  /// \param factor The scaling factor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other, scaled by \c factor
  template <
      typename Right, typename Scalar,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  result_tensor add(const Right& right, const Scalar factor) const {
    return binary(right,
                  [factor](const numeric_type l, const numeric_t<Right> r)
                      -> numeric_type { return (l + r) * factor; });
  }

  /// Scale and add this and \c other to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be added to this tensor
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other, scaled by \c factor
  template <typename Right, typename Scalar, typename Perm,
            typename std::enable_if<
                is_tensor<Right>::value && detail::is_numeric_v<Scalar> &&
                detail::is_permutation_v<Perm>>::type* = nullptr>
  result_tensor add(const Right& right, const Scalar factor,
                    const Perm& perm) const {
    return binary(
        right,
        [factor](const numeric_type l, const numeric_t<Right> r)
            -> numeric_type { return (l + r) * factor; },
        perm);
  }

  /// Add a constant to a copy of this tensor

  /// \param value The constant to be added to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c value
  result_tensor add(const numeric_type value) const {
    return unary(
        [value](const numeric_type a) -> numeric_type { return a + value; });
  }

  /// Add a constant to a permuted copy of this tensor

  /// \param value The constant to be added to this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c value
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_tensor add(const numeric_type value, const Perm& perm) const {
    return unary(
        [value](const numeric_type a) -> numeric_type { return a + value; },
        perm);
  }

  /// Add \c other to this tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be added to this tensor
  /// \return A reference to this tensor
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  TensorInterface_& add_to(const Right& right) {
    return inplace_binary(right, [](numeric_type& MADNESS_RESTRICT l,
                                    const numeric_t<Right> r) { l += r; });
  }

  /// Add \c other to this tensor, and scale the result

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be added to this tensor
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <
      typename Right, typename Scalar,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  TensorInterface_& add_to(const Right& right, const Scalar factor) {
    return inplace_binary(
        right, [factor](numeric_type& MADNESS_RESTRICT l,
                        const numeric_t<Right> r) { (l += r) *= factor; });
  }

  /// Add a constant to this tensor

  /// \param value The constant to be added
  /// \return A reference to this tensor
  TensorInterface_& add_to(const numeric_type value) {
    return inplace_unary(
        [value](numeric_type& MADNESS_RESTRICT res) { res += value; });
  }

  // Subtraction operations

  /// Subtract this and \c right to construct a new tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be subtracted from this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c right
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  result_tensor subt(const Right& right) const {
    return binary(
        right,
        [](const numeric_type l, const numeric_t<Right> r) -> numeric_type {
          return l - r;
        });
  }

  /// Subtract this and \c right to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be subtracted from this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c right
  template <
      typename Right, typename Perm,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  result_tensor subt(const Right& right, const Perm& perm) const {
    return binary(
        right,
        [](const numeric_type l, const numeric_t<Right> r) -> numeric_type {
          return l - r;
        },
        perm);
  }

  /// Scale and subtract this and \c right to construct a new tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be subtracted from this tensor
  /// \param factor The scaling factor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c right, scaled by \c factor
  template <
      typename Right, typename Scalar,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  result_tensor subt(const Right& right, const Scalar factor) const {
    return binary(right,
                  [factor](const numeric_type l, const numeric_t<Right> r)
                      -> numeric_type { return (l - r) * factor; });
  }

  /// Scale and subtract this and \c right to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be subtracted from this tensor
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c right, scaled by \c factor
  template <typename Right, typename Scalar, typename Perm,
            typename std::enable_if<
                is_tensor<Right>::value && detail::is_numeric_v<Scalar> &&
                detail::is_permutation_v<Perm>>::type* = nullptr>
  result_tensor subt(const Right& right, const Scalar factor,
                     const Perm& perm) const {
    return binary(
        right,
        [factor](const numeric_type l, const numeric_t<Right> r)
            -> numeric_type { return (l - r) * factor; },
        perm);
  }

  /// Subtract a constant from a copy of this tensor

  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c value
  result_tensor subt(const numeric_type value) const { return add(-value); }

  /// Subtract a constant from a permuted copy of this tensor

  /// \param value The constant to be subtracted
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c value
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_tensor subt(const numeric_type value, const Perm& perm) const {
    return add(-value, perm);
  }

  /// Subtract \c right from this tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be subtracted from this tensor
  /// \return A reference to this tensor
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  TensorInterface_& subt_to(const Right& right) {
    return inplace_binary(right, [](numeric_type& MADNESS_RESTRICT l,
                                    const numeric_t<Right> r) { l -= r; });
  }

  /// Subtract \c right from and scale this tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be subtracted from this tensor
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <
      typename Right, typename Scalar,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  TensorInterface_& subt_to(const Right& right, const Scalar factor) {
    return inplace_binary(
        right, [factor](numeric_type& MADNESS_RESTRICT l,
                        const numeric_t<Right> r) { (l -= r) *= factor; });
  }

  /// Subtract a constant from this tensor

  /// \return A reference to this tensor
  TensorInterface_& subt_to(const numeric_type value) { return add_to(-value); }

  // Multiplication operations

  /// Multiply this by \c right to create a new tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be multiplied by this tensor
  /// \return A new tensor where the elements are the product of the elements
  /// of \c this and \c right
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  result_tensor mult(const Right& right) const {
    return binary(
        right,
        [](const numeric_type l, const numeric_t<Right> r) -> numeric_type {
          return l * r;
        });
  }

  /// Multiply this by \c right to create a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be multiplied by this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the product of the elements
  /// of \c this and \c right
  template <
      typename Right, typename Perm,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  result_tensor mult(const Right& right, const Perm& perm) const {
    return binary(
        right,
        [](const numeric_type l, const numeric_t<Right> r) -> numeric_type {
          return l * r;
        },
        perm);
  }

  /// Scale and multiply this by \c right to create a new tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be multiplied by this tensor
  /// \param factor The scaling factor
  /// \return A new tensor where the elements are the product of the elements
  /// of \c this and \c right, scaled by \c factor
  template <
      typename Right, typename Scalar,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  result_tensor mult(const Right& right, const Scalar factor) const {
    return binary(right,
                  [factor](const numeric_type l, const numeric_t<Right> r)
                      -> numeric_type { return (l * r) * factor; });
  }

  /// Scale and multiply this by \c right to create a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be multiplied by this tensor
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the product of the elements
  /// of \c this and \c right, scaled by \c factor
  template <typename Right, typename Scalar, typename Perm,
            typename std::enable_if<
                is_tensor<Right>::value && detail::is_numeric_v<Scalar> &&
                detail::is_permutation_v<Perm>>::type* = nullptr>
  result_tensor mult(const Right& right, const Scalar factor,
                     const Perm& perm) const {
    return binary(
        right,
        [factor](const numeric_type l, const numeric_t<Right> r)
            -> numeric_type { return (l * r) * factor; },
        perm);
  }

  /// Multiply this tensor by \c right

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be multiplied by this tensor
  /// \return A reference to this tensor
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  TensorInterface_& mult_to(const Right& right) {
    return inplace_binary(right, [](numeric_type& MADNESS_RESTRICT l,
                                    const numeric_t<Right> r) { l *= r; });
  }

  /// Scale and multiply this tensor by \c right

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be multiplied by this tensor
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <
      typename Right, typename Scalar,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  TensorInterface_& mult_to(const Right& right, const Scalar factor) {
    return inplace_binary(
        right, [factor](numeric_type& MADNESS_RESTRICT l,
                        const numeric_t<Right> r) { (l *= r) *= factor; });
  }

  // Negation operations

  /// Create a negated copy of this tensor

  /// \return A new tensor that contains the negative values of this tensor
  result_tensor neg() const {
    return unary([](const numeric_type r) -> numeric_type { return -r; });
  }

  /// Create a negated and permuted copy of this tensor

  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor that contains the negative values of this tensor
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_tensor neg(const Perm& perm) const {
    return unary([](const numeric_type l) -> numeric_type { return -l; }, perm);
  }

  /// Negate elements of this tensor

  /// \return A reference to this tensor
  TensorInterface_& neg_to() {
    return inplace_unary([](numeric_type& MADNESS_RESTRICT l) { l = -l; });
  }

  /// Create a complex conjugated copy of this tensor

  /// \return A copy of this tensor that contains the complex conjugate the
  /// values
  result_tensor conj() const { return scale(conj_op()); }

  /// Create a complex conjugated and scaled copy of this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A copy of this tensor that contains the scaled complex
  /// conjugate the values
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  result_tensor conj(const Scalar factor) const {
    return scale(conj_op(factor));
  }

  /// Create a complex conjugated and permuted copy of this tensor

  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted copy of this tensor that contains the complex
  /// conjugate values
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_tensor conj(const Perm& perm) const {
    return scale(conj_op(), perm);
  }

  /// Create a complex conjugated, scaled, and permuted copy of this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted copy of this tensor that contains the complex
  /// conjugate values
  template <
      typename Scalar, typename Perm,
      typename std::enable_if<detail::is_numeric_v<Scalar> &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  result_tensor conj(const Scalar factor, const Perm& perm) const {
    return scale(conj_op(factor), perm);
  }

  /// Complex conjugate this tensor

  /// \return A reference to this tensor
  TensorInterface_& conj_to() { return scale_to(conj_op()); }

  /// Complex conjugate and scale this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  TensorInterface_& conj_to(const Scalar factor) {
    return scale_to(conj_op(factor));
  }

  /// Unary reduction operation

  /// Perform an element-wise reduction of the data by
  /// executing <tt>join_op(result, reduce_op(*this[i]))</tt> for each
  /// \c i in the index range of \c this . \c result is initialized to \c
  /// identity . If HAVE_INTEL_TBB is defined, and this is a contiguous tensor,
  /// the reduction will be executed in an undefined order, otherwise will
  /// execute in the order of increasing \c i .
  /// \tparam ReduceOp The reduction operation type
  /// \tparam JoinOp The join operation type
  /// \tparam Identity a type that can be used as argument to ReduceOp
  /// \param reduce_op The element-wise reduction operation
  /// \param join_op The join result operation
  /// \param identity The identity value of the reduction
  /// \return The reduced value
  template <typename ReduceOp, typename JoinOp, typename Identity>
  decltype(auto) reduce(ReduceOp&& reduce_op, JoinOp&& join_op,
                        Identity&& identity) const {
    return detail::tensor_reduce(std::forward<ReduceOp>(reduce_op),
                                 std::forward<JoinOp>(join_op),
                                 std::forward<Identity>(identity), *this);
  }

  /// Binary reduction operation

  /// Perform an element-wise binary reduction of the data of \c this and \c
  /// other by executing <tt>join_op(result, reduce_op(*this[i], other[i]))</tt>
  /// for each \c i in the index range of \c this . \c result is initialized to
  /// \c identity . If HAVE_INTEL_TBB is defined, and this is a contiguous
  /// tensor, the reduction will be executed in an undefined order, otherwise
  /// will execute in the order of increasing \c i .
  /// \tparam Right The right-hand argument tensor type
  /// \tparam ReduceOp The reduction operation type
  /// \tparam JoinOp The join operation type
  /// \tparam Identity a type that can be used as argument to ReduceOp
  /// \param other The right-hand argument of the binary reduction
  /// \param reduce_op The element-wise reduction operation
  /// \param join_op The join result operation
  /// \param identity The identity value of the reduction
  /// \return The reduced value
  template <typename Right, typename ReduceOp, typename JoinOp,
            typename Identity,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  decltype(auto) reduce(const Right& other, ReduceOp&& reduce_op,
                        JoinOp&& join_op, Identity&& identity) const {
    return detail::tensor_reduce(
        std::forward<ReduceOp>(reduce_op), std::forward<JoinOp>(join_op),
        std::forward<Identity>(identity), *this, other);
  }

  /// Sum of elements

  /// \return The sum of all elements of this tensor
  numeric_type sum() const {
    auto sum_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type arg) { res += arg; };
    return reduce(sum_op, sum_op, numeric_type(0));
  }

  /// Product of elements

  /// \return The product of all elements of this tensor
  numeric_type product() const {
    auto mult_op = [](numeric_type& MADNESS_RESTRICT res,
                      const numeric_type arg) { res *= arg; };
    return reduce(mult_op, mult_op, numeric_type(1));
  }

  /// Square of vector 2-norm

  /// \return The vector norm of this tensor
  scalar_type squared_norm() const {
    auto square_op = [](scalar_type& MADNESS_RESTRICT res,
                        const numeric_type arg) {
      res += TiledArray::detail::norm(arg);
    };
    auto sum_op = [](scalar_type& MADNESS_RESTRICT res, const scalar_type arg) {
      res += arg;
    };
    return reduce(square_op, sum_op, scalar_type(0));
  }

  /// Vector 2-norm

  /// \tparam ResultType the return type
  /// \note This evaluates \c std::sqrt(ResultType(this->squared_norm()))
  /// \return The vector norm of this tensor
  template <typename ResultType = numeric_type>
  ResultType norm() const {
    return std::sqrt(static_cast<ResultType>(squared_norm()));
  }

  /// Minimum element

  /// \return The minimum elements of this tensor
  numeric_type min() const {
    auto min_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type arg) { res = std::min(res, arg); };
    return reduce(min_op, min_op, std::numeric_limits<numeric_type>::max());
  }

  /// Maximum element

  /// \return The maximum elements of this tensor
  numeric_type max() const {
    auto max_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type arg) { res = std::max(res, arg); };
    return reduce(max_op, max_op, std::numeric_limits<numeric_type>::min());
  }

  /// Absolute minimum element

  /// \return The minimum elements of this tensor
  scalar_type abs_min() const {
    auto abs_min_op = [](scalar_type& MADNESS_RESTRICT res,
                         const numeric_type arg) {
      res = std::min(res, std::abs(arg));
    };
    auto min_op = [](scalar_type& MADNESS_RESTRICT res, const scalar_type arg) {
      res = std::min(res, arg);
    };
    return reduce(abs_min_op, min_op, std::numeric_limits<scalar_type>::max());
  }

  /// Absolute maximum element

  /// \return The maximum elements of this tensor
  scalar_type abs_max() const {
    auto abs_max_op = [](scalar_type& MADNESS_RESTRICT res,
                         const numeric_type arg) {
      res = std::max(res, std::abs(arg));
    };
    auto max_op = [](scalar_type& MADNESS_RESTRICT res, const scalar_type arg) {
      res = std::max(res, arg);
    };
    return reduce(abs_max_op, max_op, scalar_type(0));
  }

  /// Vector dot product

  /// \tparam Right The right-hand tensor type
  /// \param other The right-hand tensor to be reduced
  /// \return The inner product of the this and \c other
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  numeric_type dot(const Right& other) const {
    auto mult_add_op = [](numeric_type& res, const numeric_type l,
                          const numeric_t<Right> r) { res += l * r; };
    auto add_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type value) { res += value; };
    return reduce(other, mult_add_op, add_op, numeric_type(0));
  }

  /// Vector inner product
  /// \tparam Right The right-hand tensor type
  /// \param other The right-hand tensor to be reduced
  /// \return The dot product of the this and \c other
  /// If numeric_type is real, this is equivalent to dot product
  /// \sa Tensor::dot
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  numeric_type inner_product(const Right& other) const {
    auto mult_add_op = [](numeric_type& res, const numeric_type l,
                          const numeric_t<Right> r) {
      res += TiledArray::detail::inner_product(l, r);
    };
    auto add_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type value) { res += value; };
    return reduce(other, mult_add_op, add_op, numeric_type(0));
  }

};  // class TensorInterface

/// Shallow comparison operator

/// \return true if \p first and \p second view the same data block through
/// equivalent ranges
template <typename T, typename Range, typename OpResult>
bool operator==(const TensorInterface<T, Range, OpResult>& first,
                const TensorInterface<T, Range, OpResult>& second) {
  return first.data() == second.data() && first.range() == second.range();
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_TENSOR_VIEW_H__INCLUDED
