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
 *  tensor_view.h
 *  May 29, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_TENSOR_VIEW_H__INCLUDED
#define TILEDARRAY_TENSOR_TENSOR_VIEW_H__INCLUDED

#include <TiledArray/block_range.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/size_array.h>
#include <TiledArray/tensor.h>
#include <TiledArray/utility.h>

namespace TiledArray {
  namespace detail {

    /// Sub-block view of an N-dimensional Tensor

    /// TensorView is a shallow reference to sub-block of a Tensor object.
    /// \note It is the user's responsibility to ensure the original tensor does
    template <typename T, typename RangeType>
    class TensorInterface {
    public:
      typedef TensorInterface<T, RangeType> TensorInterface_; ///< This class type
      typedef RangeType range_type; ///< Tensor range type
      typedef typename range_type::size_type size_type; ///< size type
      typedef typename std::remove_const<T>::type value_type; ///< Array element type
      typedef typename std::add_lvalue_reference<T>::type reference; ///< Element reference type
      typedef typename std::add_lvalue_reference<typename std::add_const<T>::type>::type const_reference; ///< Element reference type
      typedef typename std::add_pointer<T>::type pointer; ///< Element pointer type
      typedef typename std::add_pointer<typename std::add_const<T>::type>::type const_pointer; ///< Element pointer type
      typedef typename std::ptrdiff_t difference_type; ///< Difference type
      typedef typename detail::scalar_type<value_type>::type
          numeric_type; ///< the numeric type that supports T

    private:
      template <typename, typename>
      friend class TensorInterface;

      range_type range_; ///< View sub-block range
      pointer data_; ///< Pointer to the original tensor data

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
      template <typename U,
          typename std::enable_if<std::is_convertible<
              typename TensorInterface<U, RangeType>::pointer, pointer>::value
          >::type* = nullptr>
      TensorInterface(const TensorInterface<U, RangeType>& other) :
        range_(other.range_), data_(other.data_)
      { }

      /// Type conversion move constructor

      /// \tparam U The value type of the other tensor view
      /// \param other The other tensor view to be moved
      template <typename U,
          typename std::enable_if<std::is_convertible<
              typename TensorInterface<U, RangeType>::pointer, pointer>::value
          >::type* = nullptr>
      TensorInterface(TensorInterface<U, RangeType>&& other) :
        range_(std::move(other.range_)), data_(other.data_)
      {
        other.data_ = nullptr;
      }

      /// Construct a new view of \c tensor

      /// \param range The range of this tensor
      /// \param data The data pointer for this tensor
      TensorInterface(const range_type& range, pointer data) :
        range_(range), data_(data)
      {
        TA_ASSERT(data);
      }

      /// Construct a new view of \c tensor

      /// \param range The range of this tensor
      /// \param data The data pointer for this tensor
      TensorInterface(range_type&& range, pointer data) :
        range_(std::move(range)), data_(data)
      {
        TA_ASSERT(data);
      }

      template <typename T1,
          typename std::enable_if<detail::is_tensor<T1>::value>::type* = nullptr>
      TensorInterface_& operator=(const T1& other) {
        TA_ASSERT(data_ != other.data());

        detail::inplace_tensor_op([] (numeric_type& restrict result,
            const typename T1::numeric_type arg)
            { result = arg; }, *this, other);

        return *this;
      }

      /// Tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const { return range_; }

      /// Tensor dimension size accessor

      /// \return The number of elements in the tensor
      size_type size() const { return range_.volume(); }


      /// Data pointer accessor

      /// \return The data pointer of the parent tensor
      pointer data() const { return data_; }

      /// Element subscript accessor

      /// \param index The ordinal element index
      /// \return A const reference to the element at \c index.
      const_reference operator[](const size_type index) const {
        TA_ASSERT(range_.includes(index));
        return data_[range_.ord(index)];
      }

      /// Element subscript accessor

      /// \param index The ordinal element index
      /// \return A const reference to the element at \c index.
      reference operator[](const size_type index) {
        TA_ASSERT(range_.includes(index));
        return data_[range_.ord(index)];
      }


      /// Element accessor

      /// \tparam Index An integral type pack or a single coodinate index type
      /// \param idx The index pack
      template<typename... Index>
      reference operator()(const Index&... idx) {
        TA_ASSERT(range_.includes(idx...));
        return data_[range_.ord(idx...)];
      }

      /// Element accessor

      /// \tparam Index An integral type pack or a single coodinate index type
      /// \param idx The index pack
      template<typename... Index>
      const_reference operator()(const Index&... idx) const {
        TA_ASSERT(range_.includes(idx...));
        return data_[range_.ord(idx...)];
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


      /// Shift the lower and upper bound of this tensor

      /// \tparam Index The shift array type
      /// \param bound_shift The shift to be applied to the tensor range
      /// \return A reference to this tensor
      template <typename Index>
      TensorInterface_& shift_to(const Index& bound_shift) {
        range_.inplace_shift(bound_shift);
        return *this;
      }

      /// Shift the lower and upper bound of this range

      /// \tparam Index The shift array type
      /// \param bound_shift The shift to be applied to the tensor range
      /// \return A shifted copy of this tensor
      template <typename Index>
      Tensor<T> shift(const Index& bound_shift) const {
        return Tensor<T>(range_.shift(bound_shift), data_);
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
      Tensor<T> binary(const Right& right, Op&& op) const {
        return Tensor<T>(*this, right, std::forward<Op>(op));
      }

      /// Use a binary, element wise operation to construct a new, permuted tensor

      /// \tparam Right The right-hand tensor type
      /// \tparam Op The binary operation type
      /// \param right The right-hand argument in the binary operation
      /// \param op The binary, element-wise operation
      /// \param perm The permutation to be applied to this tensor
      /// \return A tensor where element \c i of the new tensor is equal to
      /// \c op(*this[i],other[i])
      template <typename Right, typename Op,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> binary(const Right& right, Op&& op, const Permutation& perm) const {
        return Tensor<T>(*this, right, std::forward<Op>(op), perm);
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
      Tensor<T> unary(Op&& op) const {
        return Tensor<T>(*this, std::forward<Op>(op));
      }

      /// Use a unary, element wise operation to construct a new, permuted tensor

      /// \tparam Op The unary operation type
      /// \param op The unary operation
      /// \param perm The permutation to be applied to this tensor
      /// \return A permuted tensor with elements that have been modified by \c op
      /// \throw TiledArray::Exception When this tensor is empty.
      /// \throw TiledArray::Exception The dimension of \c perm does not match
      /// that of this tensor.
      template <typename Op>
      Tensor<T> unary(Op&& op, const Permutation& perm) const {
        return Tensor<T>(*this, std::forward<Op>(op), perm);
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

      // Scale operation

      /// Construct a scaled copy of this tensor

      /// \param factor The scaling factor
      /// \return A new tensor where the elements are the sum of the elements of
      /// \c this are scaled by \c factor
      Tensor<T> scale(numeric_type factor) const {
        return unary([=] (const numeric_type a) { return a * factor; });
      }

      /// Construct a scaled and permuted copy of this tensor

      /// \param factor The scaling factor
      /// \param perm The permutation to be applied to this tensor
      /// \return A new tensor where the elements are the sum of the elements of
      /// \c this are scaled by \c factor
      Tensor<T> scale(numeric_type factor, const Permutation& perm) const {
        return unary([=] (const numeric_type a) { return a * factor; }, perm);
      }

      /// Scale this tensor

      /// \param factor The scaling factor
      /// \return A reference to this tensor
      TensorInterface_& scale_to(numeric_type factor) {
        return inplace_unary([=] (numeric_type& restrict res) { res *= factor; });
      }

      // Addition operations

      /// Add this and \c other to construct a new tensors

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be added to this tensor
      /// \return A new tensor where the elements are the sum of the elements of
      /// \c this and \c other
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> add(const Right& right) const {
        return binary(right, [] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return l + r; });
      }

      /// Add this and \c other to construct a new, permuted tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be added to this tensor
      /// \param perm The permutation to be applied to this tensor
      /// \return A new tensor where the elements are the sum of the elements of
      /// \c this and \c other
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> add(const Right& right, const Permutation& perm) const {
        return binary(right, [] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return l + r; }, perm);
      }

      /// Scale and add this and \c other to construct a new tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be added to this tensor
      /// \param factor The scaling factor
      /// \return A new tensor where the elements are the sum of the elements of
      /// \c this and \c other, scaled by \c factor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> add(const Right& right, const numeric_type factor) const {
        return binary(right, [=] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return (l + r) * factor; });
      }

      /// Scale and add this and \c other to construct a new, permuted tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be added to this tensor
      /// \param factor The scaling factor
      /// \param perm The permutation to be applied to this tensor
      /// \return A new tensor where the elements are the sum of the elements of
      /// \c this and \c other, scaled by \c factor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> add(const Right& right, const numeric_type factor,
          const Permutation& perm) const
      {
        return binary(right,  [=] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return (l + r) * factor; }, perm);
      }

      /// Add a constant to a copy of this tensor

      /// \param value The constant to be added to this tensor
      /// \return A new tensor where the elements are the sum of the elements of
      /// \c this and \c value
      Tensor<T> add(const numeric_type value) const {
        return unary([=] (const numeric_type a) { return a + value; });
      }

      /// Add a constant to a permuted copy of this tensor

      /// \param value The constant to be added to this tensor
      /// \param perm The permutation to be applied to this tensor
      /// \return A new tensor where the elements are the sum of the elements of
      /// \c this and \c value
      Tensor<T> add(const numeric_type value, const Permutation& perm) const {
        return unary([=] (const numeric_type a) { return a + value; }, perm);
      }

      /// Add \c other to this tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be added to this tensor
      /// \return A reference to this tensor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      TensorInterface_& add_to(const Right& right) {
        return inplace_binary(right, [] (numeric_type& restrict l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { l += r; });
      }

      /// Add \c other to this tensor, and scale the result

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be added to this tensor
      /// \param factor The scaling factor
      /// \return A reference to this tensor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      TensorInterface_& add_to(const Right& right, const numeric_type factor) {
        return inplace_binary(right, [=] (numeric_type& restrict l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { (l += r) *= factor; });
      }

      /// Add a constant to this tensor

      /// \param value The constant to be added
      /// \return A reference to this tensor
      TensorInterface_& add_to(const numeric_type value) {
        return inplace_unary([=] (numeric_type& restrict res) { res += value; });
      }

      // Subtraction operations

      /// Subtract this and \c right to construct a new tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be subtracted from this tensor
      /// \return A new tensor where the elements are the different between the
      /// elements of \c this and \c right
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> subt(const Right& right) const {
        return binary(right, [] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return l - r; });
      }

      /// Subtract this and \c right to construct a new, permuted tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be subtracted from this tensor
      /// \param perm The permutation to be applied to this tensor
      /// \return A new tensor where the elements are the different between the
      /// elements of \c this and \c right
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> subt(const Right& right, const Permutation& perm) const {
        return binary(right, [] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return l - r; }, perm);
      }

      /// Scale and subtract this and \c right to construct a new tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be subtracted from this tensor
      /// \param factor The scaling factor
      /// \return A new tensor where the elements are the different between the
      /// elements of \c this and \c right, scaled by \c factor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> subt(const Right& right, const numeric_type factor) const {
        return binary(right, [=] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return (l - r) * factor; });
      }

      /// Scale and subtract this and \c right to construct a new, permuted tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be subtracted from this tensor
      /// \param factor The scaling factor
      /// \param perm The permutation to be applied to this tensor
      /// \return A new tensor where the elements are the different between the
      /// elements of \c this and \c right, scaled by \c factor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> subt(const Right& right, const numeric_type factor,
          const Permutation& perm) const
      {
        return binary(right, [=] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return (l - r) * factor; }, perm);
      }

      /// Subtract a constant from a copy of this tensor

      /// \return A new tensor where the elements are the different between the
      /// elements of \c this and \c value
      Tensor<T> subt(const numeric_type value) const {
        return add(-value);
      }

      /// Subtract a constant from a permuted copy of this tensor

      /// \param value The constant to be subtracted
      /// \param perm The permutation to be applied to this tensor
      /// \return A new tensor where the elements are the different between the
      /// elements of \c this and \c value
      Tensor<T> subt(const numeric_type value, const Permutation& perm) const {
        return add(-value, perm);
      }

      /// Subtract \c right from this tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be subtracted from this tensor
      /// \return A reference to this tensor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      TensorInterface_& subt_to(const Right& right) {
        return inplace_binary(right, [] (numeric_type& restrict l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { l -= r; });
      }

      /// Subtract \c right from and scale this tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be subtracted from this tensor
      /// \param factor The scaling factor
      /// \return A reference to this tensor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      TensorInterface_& subt_to(const Right& right, const numeric_type factor) {
        return inplace_binary(right, [=] (numeric_type& restrict l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { (l -= r) *= factor; });
      }

      /// Subtract a constant from this tensor

      /// \return A reference to this tensor
      TensorInterface_& subt_to(const numeric_type value) {
        return add_to(-value);
      }

      // Multiplication operations

      /// Multiply this by \c right to create a new tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be multiplied by this tensor
      /// \return A new tensor where the elements are the product of the elements
      /// of \c this and \c right
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> mult(const Right& right) const {
        return binary(right, [] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return l * r; });
      }

      /// Multiply this by \c right to create a new, permuted tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be multiplied by this tensor
      /// \param perm The permutation to be applied to this tensor
      /// \return A new tensor where the elements are the product of the elements
      /// of \c this and \c right
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> mult(const Right& right, const Permutation& perm) const {
        return binary(right, [] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return l * r; }, perm);
      }

      /// Scale and multiply this by \c right to create a new tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be multiplied by this tensor
      /// \param factor The scaling factor
      /// \return A new tensor where the elements are the product of the elements
      /// of \c this and \c right, scaled by \c factor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> mult(const Right& right, const numeric_type factor) const {
        return binary(right, [=] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return (l * r) * factor; });
      }

      /// Scale and multiply this by \c right to create a new, permuted tensor

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be multiplied by this tensor
      /// \param factor The scaling factor
      /// \param perm The permutation to be applied to this tensor
      /// \return A new tensor where the elements are the product of the elements
      /// of \c this and \c right, scaled by \c factor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      Tensor<T> mult(const Right& right, const numeric_type factor,
          const Permutation& perm) const
      {
        return binary(right,  [=] (const numeric_type l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { return (l * r) * factor; }, perm);
      }

      /// Multiply this tensor by \c right

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be multiplied by this tensor
      /// \return A reference to this tensor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      TensorInterface_& mult_to(const Right& right) {
        return inplace_binary(right, [] (numeric_type& restrict l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { l *= r; });
      }

      /// Scale and multiply this tensor by \c right

      /// \tparam Right The right-hand tensor type
      /// \param right The tensor that will be multiplied by this tensor
      /// \param factor The scaling factor
      /// \return A reference to this tensor
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      TensorInterface_& mult_to(const Right& right, const numeric_type factor) {
        return inplace_binary(right, [=] (numeric_type& restrict l,
            const typename TiledArray::detail::scalar_type<Right>::type r)
            { (l *= r) *= factor; });
      }

      // Negation operations

      /// Create a negated copy of this tensor

      /// \return A new tensor that contains the negative values of this tensor
      Tensor<T> neg() const {
        return unary([] (const numeric_type r) { return -r; });
      }

      /// Create a negated and permuted copy of this tensor

      /// \param perm The permutation to be applied to this tensor
      /// \return A new tensor that contains the negative values of this tensor
      Tensor<T> neg(const Permutation& perm) const {
        return unary([] (const numeric_type l) { return -l; }, perm);
      }

      /// Negate elements of this tensor

      /// \return A reference to this tensor
      TensorInterface_& neg_to() {
        return inplace_unary([] (numeric_type& restrict l) { l = -l; });
      }


      /// Unary reduction operation

      /// Perform an element-wise reduction of the tile data.
      /// \tparam ReduceOp The reduction operation type
      /// \tparam JoinOp The join operation type
      /// \param reduce_op The element-wise reduction operation
      /// \param join_op The join result operation
      /// \param identity The identity value of the reduction
      /// \return The reduced value
      template <typename ReduceOp, typename JoinOp>
      numeric_type reduce(ReduceOp&& reduce_op, JoinOp&& join_op,
          const numeric_type identity) const
      {
        return detail::tensor_reduce(std::forward<ReduceOp>(reduce_op),
            std::forward<JoinOp>(join_op), identity, *this);
      }

      /// Binary reduction operation

      /// Perform an element-wise reduction of the tile data.
      /// \tparam Right The right-hand argument tensor type
      /// \tparam ReduceOp The reduction operation type
      /// \tparam JoinOp The join operation type
      /// \param other The right-hand argument of the binary reduction
      /// \param reduce_op The element-wise reduction operation
      /// \param join_op The join result operation
      /// \param identity The identity value of the reduction
      /// \return The reduced value
      template <typename Right, typename ReduceOp, typename JoinOp,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      numeric_type reduce(const Right& other, ReduceOp&& reduce_op, JoinOp&& join_op,
          const numeric_type identity) const
      {
        return detail::tensor_reduce(std::forward<ReduceOp>(reduce_op),
            std::forward<JoinOp>(join_op), identity, *this, other);
      }

      /// Sum of elements

      /// \return The sum of all elements of this tensor
      numeric_type sum() const {
        auto sum_op = [] (numeric_type& restrict res, const numeric_type arg)
                { res += arg; };
        return reduce(sum_op, sum_op, numeric_type(0));
      }

      /// Product of elements

      /// \return The product of all elements of this tensor
      numeric_type product() const {
        auto mult_op = [] (numeric_type& restrict res, const numeric_type arg)
                { res *= arg; };
        return reduce(mult_op, mult_op, numeric_type(1));
      }

      /// Square of vector 2-norm

      /// \return The vector norm of this tensor
      numeric_type squared_norm() const {
        auto square_op = [] (numeric_type& restrict res, const numeric_type arg)
                { res += arg * arg; };
        auto sum_op = [] (numeric_type& restrict res, const numeric_type arg)
                { res += arg; };
        return reduce(square_op, sum_op, numeric_type(0));
      }

      /// Vector 2-norm

      /// \return The vector norm of this tensor
      numeric_type norm() const {
        return std::sqrt(squared_norm());
      }

      /// Minimum element

      /// \return The minimum elements of this tensor
      numeric_type min() const {
        auto min_op = [] (numeric_type& restrict res, const numeric_type arg)
                { res = std::min(res, arg); };
        return reduce(min_op, min_op, std::numeric_limits<numeric_type>::max());
      }

      /// Maximum element

      /// \return The maximum elements of this tensor
      numeric_type max() const {
        auto max_op = [] (numeric_type& restrict res, const numeric_type arg)
                { res = std::max(res, arg); };
        return reduce(max_op, max_op, std::numeric_limits<numeric_type>::min());
      }

      /// Absolute minimum element

      /// \return The minimum elements of this tensor
      numeric_type abs_min() const {
        auto abs_min_op = [] (numeric_type& restrict res, const numeric_type arg)
                { res = std::min(res, std::abs(arg)); };
        auto min_op = [] (numeric_type& restrict res, const numeric_type arg)
                { res = std::min(res, arg); };
        return reduce(abs_min_op, min_op, std::numeric_limits<numeric_type>::max());
      }

      /// Absolute maximum element

      /// \return The maximum elements of this tensor
      numeric_type abs_max() const {
        auto abs_max_op = [] (numeric_type& restrict res, const numeric_type arg)
                { res = std::max(res, std::abs(arg)); };
        auto max_op = [] (numeric_type& restrict res, const numeric_type arg)
                { res = std::max(res, arg); };
        return reduce(abs_max_op, max_op, numeric_type(0));
      }

      /// Vector dot product

      /// \tparam Right The right-hand tensor type
      /// \param other The right-hand tensor to be reduced
      /// \return The inner product of the this and \c other
      template <typename Right,
          typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
      numeric_type dot(const Right& other) const {
        auto mult_add_op = [] (numeric_type& res, const numeric_type l,
                  const typename TiledArray::detail::scalar_type<Right>::type r)
                  { res += l * r; };
        auto add_op = [] (numeric_type& restrict res, const numeric_type value)
              { res += value; };
        return reduce(other, mult_add_op, add_op, numeric_type(0));
      }

    }; // class TensorInterface

  } // namespace detail

  // Template aliases for TensorInterface objects

  template <typename T>
  using TensorView =
      detail::TensorInterface<T, BlockRange>;

  template <typename T>
  using TensorConstView =
      detail::TensorInterface<typename std::add_const<T>::type, BlockRange>;


  template <typename T>
  using TensorMap =
      detail::TensorInterface<T, Range>;

  template <typename T>
  using TensorConstMap =
      detail::TensorInterface<typename std::add_const<T>::type, Range>;

  template <typename T, typename Index>
  TensorMap<T> make_map(T* const data, const Index& lower_bound,
      const Index& upper_bound)
  { return TensorMap<T>(Range(lower_bound, upper_bound), data); }

  template <typename T, typename Index>
  TensorMap<T> make_map(T* const data,
      const std::initializer_list<Index>& lower_bound,
      const std::initializer_list<Index>& upper_bound)
  { return TensorMap<T>(Range(lower_bound, upper_bound), data); }

  template <typename T>
  TensorMap<T> make_map(T* const data, const Range& range)
  { return TensorMap<T>(range, data); }

  template <typename T, typename Index>
  TensorConstMap<T> make_map(const T* const data, const Index& lower_bound,
      const Index& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }


  template <typename T, typename Index>
  TensorConstMap<T> make_map(const T* const data,
      const std::initializer_list<Index>& lower_bound,
      const std::initializer_list<Index>& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }

  template <typename T>
  TensorConstMap<T> make_map(const T* const data, const Range& range)
  { return TensorConstMap<T>(range, data); }


  template <typename T, typename Index>
  TensorConstMap<T> make_const_map(const T* const data, const Index& lower_bound,
      const Index& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }


  template <typename T, typename Index>
  TensorConstMap<T> make_const_map(const T* const data,
      const std::initializer_list<Index>& lower_bound,
      const std::initializer_list<Index>& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }

  template <typename T>
  TensorConstMap<T> make_const_map(const T* const data, const Range& range)
  { return TensorConstMap<T>(range, data); }

} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_TENSOR_VIEW_H__INCLUDED
