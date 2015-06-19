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

#ifndef TILEDARRAY_TENSOR_TENSOR_H__INCLUDED
#define TILEDARRAY_TENSOR_TENSOR_H__INCLUDED

#include <TiledArray/math/gemm_helper.h>
#include <TiledArray/math/blas.h>
#include <TiledArray/tensor/kernels.h>

namespace TiledArray {

  /// An N-dimensional tensor object

  /// \tparam T the value type of this tensor
  /// \tparam A The allocator type for the data
  template <typename T, typename A = Eigen::aligned_allocator<T> >
  class Tensor {
  public:
    typedef Tensor<T, A> Tensor_; ///< This class type
    typedef Range range_type; ///< Tensor range type
    typedef typename range_type::size_type size_type; ///< size type
    typedef A allocator_type; ///< Allocator type
    typedef typename allocator_type::value_type value_type; ///< Array element type
    typedef typename allocator_type::reference reference; ///< Element reference type
    typedef typename allocator_type::const_reference const_reference; ///< Element reference type
    typedef typename allocator_type::pointer pointer; ///< Element pointer type
    typedef typename allocator_type::const_pointer const_pointer; ///< Element const pointer type
    typedef typename allocator_type::difference_type difference_type; ///< Difference type
    typedef pointer iterator; ///< Element iterator type
    typedef const_pointer const_iterator; ///< Element const iterator type
    typedef typename TiledArray::detail::scalar_type<T>::type
        numeric_type; ///< the numeric type that supports T

  private:

    /// Evaluation tensor

    /// This tensor is used as an evaluated intermediate for other tensors.
    class Impl : public allocator_type {
    public:

      /// Default constructor

      /// Construct an empty tensor that has no data or dimensions
      Impl() : allocator_type(), range_(), data_(NULL) { }

      /// Construct with range

      /// \param range The N-dimensional range for this tensor
      explicit Impl(const range_type& range) :
        allocator_type(), range_(range), data_(NULL)
      {
        data_ = allocator_type::allocate(range.volume());
      }

      ~Impl() {
        math::destroy_vector(range_.volume(), data_);
        allocator_type::deallocate(data_, range_.volume());
        data_ = NULL;
      }

      range_type range_; ///< Tensor size info
      pointer data_; ///< Tensor data
    }; // class Impl

    template <typename... Ts>
    struct is_tensor {
      static constexpr bool value =
          detail::is_tensor<Ts...>::value || detail::is_tensor_of_tensor<Ts...>::value;
    };

    template <typename U, typename std::enable_if<std::is_scalar<U>::value>::type* = nullptr>
    static void default_init(size_type, U*) { }

    template <typename U, typename std::enable_if<! std::is_scalar<U>::value>::type* = nullptr>
    static void default_init(size_type n, U* u) {
      math::uninitialized_fill_vector(n, U(), u);
    }

    std::shared_ptr<Impl> pimpl_; ///< Shared pointer to implementation object
    static const range_type empty_range_; ///< Empty range

  public:

    // Compiler generated functions
    Tensor() = default;
    Tensor(const Tensor_&) = default;
    Tensor(Tensor_&&) = default;
    ~Tensor() = default;
    Tensor_& operator=(const Tensor_& other) = default;
    Tensor_& operator=(Tensor_&&) = default;

    /// Construct tensor

    /// Construct a tensor with a range equal to \c range. The data is
    /// uninitialized.
    /// \param range The range of the tensor
    Tensor(const range_type& range) :
      pimpl_(new Impl(range))
    {
      default_init(range.volume(), pimpl_->data_);
    }

    /// Construct a tensor with a fill value

    /// \param range An array with the size of of each dimension
    /// \param value The value of the tensor elements
    Tensor(const range_type& range, const numeric_type value) :
      pimpl_(new Impl(range))
    {
      detail::tensor_init([=] () { return value; }, *this);
    }

    /// Construct an evaluated tensor
    template <typename InIter,
        typename std::enable_if<TiledArray::detail::is_input_iterator<InIter>::value &&
            ! std::is_pointer<InIter>::value>::type* = nullptr>
    Tensor(const range_type& range, InIter it) :
      pimpl_(new Impl(range))
    {
      size_type n = range.volume();
      pointer restrict const data = pimpl_->data_;
      for(size_type i = 0ul; i < n; ++i)
        data[i] = *it++;
    }

    template <typename U>
    Tensor(const Range& range, const U* u) :
      pimpl_(new Impl(range))
    {
      math::uninitialized_copy_vector(range.volume(), u, pimpl_->data_);
    }

    /// Construct a copy of a tensor interface object

    /// \tparam T1 A tensor type
    /// \param other The tensor to be copied
    template <typename T1,
        typename std::enable_if<is_tensor<T1>::value &&
            ! std::is_same<T1, Tensor_>::value>::type* = nullptr>
    Tensor(const T1& other) :
      pimpl_(new Impl(detail::clone_range(other)))
    {
      auto op =
          [] (const typename T1::numeric_type arg) -> typename T1::numeric_type
      { return arg; };

      detail::tensor_init(op, *this, other);
    }

    /// Construct a permuted tensor copy

    /// \tparam T1 A tensor type
    /// \param other The tensor to be copied
    /// \param perm The permutation that will be applied to the copy
    template <typename T1,
        typename std::enable_if<is_tensor<T1>::value>::type* = nullptr>
    Tensor(const T1& other, const Permutation& perm) :
      pimpl_(new Impl(perm * other.range()))
    {
      auto op =
          [] (const typename T1::numeric_type arg) -> typename T1::numeric_type
      { return arg; };

      detail::tensor_init(op, perm, *this, other);
    }

    /// Copy and modify the data from \c other

    /// \tparam T1 A tensor type
    /// \tparam Op An element-wise operation type
    /// \param other The tensor argument
    /// \param op The element-wise operation
    template <typename T1, typename Op,
        typename std::enable_if<is_tensor<T1>::value
                 && ! std::is_same<typename std::decay<Op>::type,
                 Permutation>::value>::type* = nullptr>
    Tensor(const T1& other, Op&& op) :
      pimpl_(new Impl(detail::clone_range(other)))
    {
      detail::tensor_init(op, *this, other);
    }

    /// Copy, modify, and permute the data from \c other

    /// \tparam T1 A tensor type
    /// \tparam Op An element-wise operation type
    /// \param other The tensor argument
    /// \param op The element-wise operation
    template <typename T1, typename Op,
        typename std::enable_if<is_tensor<T1>::value>::type* = nullptr>
    Tensor(const T1& other, Op&& op, const Permutation& perm) :
      pimpl_(new Impl(perm * other.range()))
    {
      detail::tensor_init(std::forward<Op>(op), perm, *this, other);
    }

    /// Copy and modify the data from \c left, and \c right

    /// \tparam T1 A tensor type
    /// \tparam T2 A tensor type
    /// \tparam Op An element-wise operation type
    /// \param left The left-hand tensor argument
    /// \param right The right-hand tensor argument
    /// \param op The element-wise operation
    template <typename T1, typename T2, typename Op,
        typename std::enable_if<is_tensor<T1, T2>::value>::type* = nullptr>
    Tensor(const T1& left, const T2& right, Op&& op) :
      pimpl_(new Impl(detail::clone_range(left)))
    {
      detail::tensor_init(std::forward<Op>(op), *this, left, right);
    }

    /// Copy, modify, and permute the data from \c left, and \c right

    /// \tparam T1 A tensor type
    /// \tparam T2 A tensor type
    /// \tparam Op An element-wise operation type
    /// \param left The left-hand tensor argument
    /// \param right The right-hand tensor argument
    /// \param op The element-wise operation
    /// \param perm The permutation that will be applied to the arguments
    template <typename T1, typename T2, typename Op,
        typename std::enable_if<is_tensor<T1, T2>::value>::type* = nullptr>
    Tensor(const T1& left, const T2& right, Op&& op, const Permutation& perm) :
      pimpl_(new Impl(perm * left.range()))
    {
      detail::tensor_init(std::forward<Op>(op), perm, *this, left, right);
    }

    Tensor_ clone() const {
      Tensor_ result;
      if(pimpl_) {
        result = detail::tensor_op<Tensor_>(
            [] (const numeric_type value) -> numeric_type { return value; },
            *this);
      }
      return result;
    }

    template <typename T1,
        typename std::enable_if<is_tensor<T1>::value>::type* = nullptr>
    Tensor_& operator=(const T1& other) {
      detail::inplace_tensor_op([] (reference restrict tr,
          typename T1::const_reference restrict t1) { tr = t1; }, *this, other);

      return *this;
    }

    /// Tensor range object accessor

    /// \return The tensor range object
    const range_type& range() const {
      return (pimpl_ ? pimpl_->range_ : empty_range_);
    }

    /// Tensor dimension size accessor

    /// \return The number of elements in the tensor
    size_type size() const {
      return (pimpl_ ? pimpl_->range_.volume() : 0ul);
    }

    /// Element accessor

    /// \return The element at the \c i position.
    const_reference operator[](const size_type i) const {
      TA_ASSERT(pimpl_);
      TA_ASSERT(pimpl_->range_.includes(i));
      return pimpl_->data_[i];
    }

    /// Element accessor

    /// \return The element at the \c i position.
    /// \throw TiledArray::Exception When this tensor is empty.
    reference operator[](const size_type i) {
      TA_ASSERT(pimpl_);
      TA_ASSERT(pimpl_->range_.includes(i));
      return pimpl_->data_[i];
    }


    /// Element accessor

    /// \return The element at the \c i position.
    /// \throw TiledArray::Exception When this tensor is empty.
    template <typename Index,
        typename std::enable_if<
            ! std::is_integral<Index>::value>::type* = nullptr>
    const_reference operator[](const Index& i) const {
      TA_ASSERT(pimpl_);
      TA_ASSERT(pimpl_->range_.includes(i));
      return pimpl_->data_[pimpl_->range_.ordinal(i)];
    }

    /// Element accessor

    /// \return The element at the \c i position.
    /// \throw TiledArray::Exception When this tensor is empty.
    template <typename Index,
      typename std::enable_if<
          ! std::is_integral<Index>::value>::type* = nullptr>
    reference operator[](const Index& i) {
      TA_ASSERT(pimpl_);
      TA_ASSERT(pimpl_->range_.includes(i));
      return pimpl_->data_[pimpl_->range_.ordinal(i)];
    }

    /// Element accessor

    /// \tparam Index index type pack
    /// \param idx The index pack
    template<typename... Index>
    reference operator()(const Index&... idx) {
      TA_ASSERT(pimpl_);
      TA_ASSERT(pimpl_->range_.includes(idx...));
      return pimpl_->data_[pimpl_->range_.ordinal(idx...)];
    }

    /// Element accessor

    /// \tparam Index index type pack
    /// \param idx The index pack
    template<typename... Index>
    const_reference operator()(const Index&... idx) const {
      TA_ASSERT(pimpl_);
      TA_ASSERT(pimpl_->range_.includes(idx...));
      return pimpl_->data_[pimpl_->range_.ordinal(idx...)];
    }

    /// Iterator factory

    /// \return An iterator to the first data element
    const_iterator begin() const { return (pimpl_ ? pimpl_->data_ : NULL); }

    /// Iterator factory

    /// \return An iterator to the first data element
    iterator begin() { return (pimpl_ ? pimpl_->data_ : NULL); }

    /// Iterator factory

    /// \return An iterator to the last data element
    const_iterator end() const {
      return (pimpl_ ? pimpl_->data_ + pimpl_->range_.volume() : NULL);
    }

    /// Iterator factory

    /// \return An iterator to the last data element
    iterator end() {
      return (pimpl_ ? pimpl_->data_ + pimpl_->range_.volume() : NULL);
    }

    /// Data direct access

    /// \return A const pointer to the tensor data
    const_pointer data() const { return (pimpl_ ? pimpl_->data_ : NULL); }

    /// Data direct access

    /// \return A const pointer to the tensor data
    pointer data() { return (pimpl_ ? pimpl_->data_ : NULL); }

    /// Test if the tensor is empty

    /// \return \c true if this tensor was default constructed (contains no
    /// data), otherwise \c false.
    bool empty() const { return !pimpl_; }

    /// Output serialization function

    /// This function enables serialization within MADNESS
    /// \tparam Archive The output archive type
    /// \param[out] ar The output archive
    template <typename Archive,
        typename std::enable_if<
          madness::archive::is_output_archive<Archive>::value>::type* = nullptr>
    void serialize(Archive& ar) {
      if(pimpl_) {
        ar & pimpl_->range_.volume();
        ar & madness::archive::wrap(pimpl_->data_, pimpl_->range_.volume());
        ar & pimpl_->range_;
      } else {
        ar & size_type(0ul);
      }
    }

    /// Input serialization function

    /// This function enables serialization within MADNESS
    /// \tparam Archive The input archive type
    /// \param[out] ar The input archive
    template <typename Archive,
        typename std::enable_if<
          madness::archive::is_input_archive<Archive>::value>::type* = nullptr>
    void serialize(Archive& ar) {
      size_type n = 0ul;
      ar & n;
      if(n) {
        std::shared_ptr<Impl> temp(new Impl());
        temp->data_ = temp->allocate(n);
        try {
          ar & madness::archive::wrap(temp->data_, n);
          ar & temp->range_;
        } catch(...) {
          temp->deallocate(temp->data_, n);
          throw;
        }

        pimpl_ = temp;
      } else {
        pimpl_.reset();
      }
    }

    /// Swap tensor data

    /// \param other The tensor to swap with this
    void swap(Tensor_& other) {
      std::swap(pimpl_, other.pimpl_);
    }

    template <typename Index>
    detail::TensorInterface<T, BlockRange>
    block(const Index& lower_bound, const Index& upper_bound) {
      TA_ASSERT(pimpl_);
      return detail::TensorInterface<T, BlockRange>(BlockRange(pimpl_->range_,
          lower_bound, upper_bound), pimpl_->data_);
    }

    detail::TensorInterface<T, BlockRange>
    block(const std::initializer_list<size_type>& lower_bound,
        const std::initializer_list<size_type>& upper_bound)
    {
      TA_ASSERT(pimpl_);
      return detail::TensorInterface<T, BlockRange>(BlockRange(pimpl_->range_,
          lower_bound, upper_bound), pimpl_->data_);
    }

    template <typename Index>
    detail::TensorInterface<const T, BlockRange>
    block(const Index& lower_bound, const Index& upper_bound) const {
      TA_ASSERT(pimpl_);
      return detail::TensorInterface<const T, BlockRange>(BlockRange(pimpl_->range_,
          lower_bound, upper_bound), pimpl_->data_);
    }

    detail::TensorInterface<const T, BlockRange>
    block(const std::initializer_list<size_type>& lower_bound,
        const std::initializer_list<size_type>& upper_bound) const
    {
      TA_ASSERT(pimpl_);
      return detail::TensorInterface<const T, BlockRange>(BlockRange(pimpl_->range_,
          lower_bound, upper_bound), pimpl_->data_);
    }

    /// Create a permuted copy of this tensor

    /// \param perm The permutation to be applied to this tensor
    /// \return A permuted copy of this tensor
    Tensor_ permute(const Permutation& perm) const {
      return Tensor_(*this, perm);
    }


    /// Shift the lower and upper bound of this tensor

    /// \tparam Index The shift array type
    /// \param bound_shift The shift to be applied to the tensor range
    /// \return A reference to this tensor
    template <typename Index>
    Tensor_& shift_to(const Index& bound_shift) {
      TA_ASSERT(pimpl_);
      pimpl_->range_.inplace_shift(bound_shift);
      return *this;
    }

    /// Shift the lower and upper bound of this range

    /// \tparam Index The shift array type
    /// \param bound_shift The shift to be applied to the tensor range
    /// \return A shifted copy of this tensor
    template <typename Index>
    Tensor_ shift(const Index& bound_shift) const {
      TA_ASSERT(pimpl_);
      Tensor_ result = clone();
      result.shift_to(bound_shift);
      return result;
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
    Tensor_ binary(const Right& right, Op&& op) const {
      return Tensor_(*this, right, std::forward<Op>(op));
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
    Tensor_ binary(const Right& right, Op&& op, const Permutation& perm) const {
      return Tensor_(*this, right, std::forward<Op>(op), perm);
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
    Tensor_& inplace_binary(const Right& right, Op&& op) {
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
    Tensor_ unary(Op&& op) const {
      return Tensor_(*this, std::forward<Op>(op));
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
    Tensor_ unary(Op&& op, const Permutation& perm) const {
      return Tensor_(*this, std::forward<Op>(op), perm);
    }

    /// Use a unary, element wise operation to modify this tensor

    /// \tparam Op The unary operation type
    /// \param op The unary, element-wise operation
    /// \return A reference to this object
    /// \throw TiledArray::Exception When this tensor is empty.
    template <typename Op>
    Tensor_& inplace_unary(Op&& op) {
      detail::inplace_tensor_op(std::forward<Op>(op), *this);
      return *this;
    }

    // Scale operation

    /// Construct a scaled copy of this tensor

    /// \param factor The scaling factor
    /// \return A new tensor where the elements are the sum of the elements of
    /// \c this are scaled by \c factor
    Tensor_ scale(numeric_type factor) const {
      return unary([=] (const numeric_type a) { return a * factor; });
    }

    /// Construct a scaled and permuted copy of this tensor

    /// \param factor The scaling factor
    /// \param perm The permutation to be applied to this tensor
    /// \return A new tensor where the elements are the sum of the elements of
    /// \c this are scaled by \c factor
    Tensor_ scale(numeric_type factor, const Permutation& perm) const {
      return unary([=] (const numeric_type a) { return a * factor; }, perm);
    }

    /// Scale this tensor

    /// \param factor The scaling factor
    /// \return A reference to this tensor
    Tensor_& scale_to(numeric_type factor) {
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
    Tensor_ add(const Right& right) const {
      return binary(right, [] (const numeric_type l,
          const typename TiledArray::detail::scalar_type<Right>::type r) { return l + r; });
    }

    /// Add this and \c other to construct a new, permuted tensor

    /// \tparam Right The right-hand tensor type
    /// \param right The tensor that will be added to this tensor
    /// \param perm The permutation to be applied to this tensor
    /// \return A new tensor where the elements are the sum of the elements of
    /// \c this and \c other
    template <typename Right,
        typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
    Tensor_ add(const Right& right, const Permutation& perm) const {
      return binary(right, [] (const numeric_type l,
          const typename TiledArray::detail::scalar_type<Right>::type r)
          { return l + r; },
          perm);
    }

    /// Scale and add this and \c other to construct a new tensor

    /// \tparam Right The right-hand tensor type
    /// \param right The tensor that will be added to this tensor
    /// \param factor The scaling factor
    /// \return A new tensor where the elements are the sum of the elements of
    /// \c this and \c other, scaled by \c factor
    template <typename Right,
        typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
    Tensor_ add(const Right& right, const numeric_type factor) const {
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
    Tensor_ add(const Right& right, const numeric_type factor,
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
    Tensor_ add(const numeric_type value) const {
      return unary([=] (const numeric_type a) { return a + value; });
    }

    /// Add a constant to a permuted copy of this tensor

    /// \param value The constant to be added to this tensor
    /// \param perm The permutation to be applied to this tensor
    /// \return A new tensor where the elements are the sum of the elements of
    /// \c this and \c value
    Tensor_ add(const numeric_type value, const Permutation& perm) const {
      return unary([=] (const numeric_type a) { return a + value; }, perm);
    }

    /// Add \c other to this tensor

    /// \tparam Right The right-hand tensor type
    /// \param right The tensor that will be added to this tensor
    /// \return A reference to this tensor
    template <typename Right,
        typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
    Tensor_& add_to(const Right& right) {
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
    Tensor_& add_to(const Right& right, const numeric_type factor) {
      return inplace_binary(right, [=] (numeric_type& restrict l,
          const typename TiledArray::detail::scalar_type<Right>::type r)
          { (l += r) *= factor; });
    }

    /// Add a constant to this tensor

    /// \param value The constant to be added
    /// \return A reference to this tensor
    Tensor_& add_to(const numeric_type value) {
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
    Tensor_ subt(const Right& right) const {
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
    Tensor_ subt(const Right& right, const Permutation& perm) const {
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
    Tensor_ subt(const Right& right, const numeric_type factor) const {
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
    Tensor_ subt(const Right& right, const numeric_type factor,
        const Permutation& perm) const
    {
      return binary(right, [=] (const numeric_type l,
          const typename TiledArray::detail::scalar_type<Right>::type r)
          { return (l - r) * factor; }, perm);
    }

    /// Subtract a constant from a copy of this tensor

    /// \return A new tensor where the elements are the different between the
    /// elements of \c this and \c value
    Tensor_ subt(const numeric_type value) const {
      return add(-value);
    }

    /// Subtract a constant from a permuted copy of this tensor

    /// \param value The constant to be subtracted
    /// \param perm The permutation to be applied to this tensor
    /// \return A new tensor where the elements are the different between the
    /// elements of \c this and \c value
    Tensor_ subt(const numeric_type value, const Permutation& perm) const {
      return add(-value, perm);
    }

    /// Subtract \c right from this tensor

    /// \tparam Right The right-hand tensor type
    /// \param right The tensor that will be subtracted from this tensor
    /// \return A reference to this tensor
    template <typename Right,
        typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
    Tensor_& subt_to(const Right& right) {
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
    Tensor_& subt_to(const Right& right, const numeric_type factor) {
      return inplace_binary(right, [=] (numeric_type& restrict l,
          const typename TiledArray::detail::scalar_type<Right>::type r)
          { (l -= r) *= factor; });
    }

    /// Subtract a constant from this tensor

    /// \return A reference to this tensor
    Tensor_& subt_to(const numeric_type value) {
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
    Tensor_ mult(const Right& right) const {
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
    Tensor_ mult(const Right& right, const Permutation& perm) const {
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
    Tensor_ mult(const Right& right, const numeric_type factor) const {
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
    Tensor_ mult(const Right& right, const numeric_type factor,
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
    Tensor_& mult_to(const Right& right) {
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
    Tensor_& mult_to(const Right& right, const numeric_type factor) {
      return inplace_binary(right, [=] (numeric_type& restrict l,
          const typename TiledArray::detail::scalar_type<Right>::type r)
          { (l *= r) *= factor; });
    }

    // Negation operations

    /// Create a negated copy of this tensor

    /// \return A new tensor that contains the negative values of this tensor
    Tensor_ neg() const {
      return unary([] (const numeric_type r) { return -r; });
    }

    /// Create a negated and permuted copy of this tensor

    /// \param perm The permutation to be applied to this tensor
    /// \return A new tensor that contains the negative values of this tensor
    Tensor_ neg(const Permutation& perm) const {
      return unary([] (const numeric_type l) { return -l; }, perm);
    }

    /// Negate elements of this tensor

    /// \return A reference to this tensor
    Tensor_& neg_to() {
      return inplace_unary([] (numeric_type& restrict l) { l = -l; });
    }

    // GEMM operations

    /// Contract this tensor with \c other

    /// \tparam U The other tensor element type
    /// \tparam AU The other tensor allocator type
    /// \param other The tensor that will be contracted with this tensor
    /// \param factor The scaling factor
    /// \param gemm_helper The *GEMM operation meta data
    /// \return A new tensor which is the result of contracting this tensor with
    /// \c other
    /// \throw TiledArray::Exception When this tensor is empty.
    /// \throw TiledArray::Exception When \c other is empty.
    template <typename U, typename AU>
    Tensor_ gemm(const Tensor<U, AU>& other, const numeric_type factor,
        const math::GemmHelper& gemm_helper) const
    {
      // Check that this tensor is not empty and has the correct rank
      TA_ASSERT(pimpl_);
      TA_ASSERT(pimpl_->range_.rank() == gemm_helper.left_rank());

      // Check that the arguments are not empty and have the correct ranks
      TA_ASSERT(!other.empty());
      TA_ASSERT(other.range().rank() == gemm_helper.right_rank());

      // Construct the result Tensor
      Tensor_ result(gemm_helper.make_result_range<range_type>(pimpl_->range_, other.range()));

      // Check that the inner dimensions of left and right match
      TA_ASSERT(gemm_helper.left_right_coformal(pimpl_->range_.lobound_data(), other.range().lobound_data()));
      TA_ASSERT(gemm_helper.left_right_coformal(pimpl_->range_.upbound_data(), other.range().upbound_data()));
      TA_ASSERT(gemm_helper.left_right_coformal(pimpl_->range_.extent_data(), other.range().extent_data()));


      // Compute gemm dimensions
      integer m = 1, n = 1, k = 1;
      gemm_helper.compute_matrix_sizes(m, n, k, pimpl_->range_, other.range());

      // Get the leading dimension for left and right matrices.
      const integer lda = (gemm_helper.left_op() == madness::cblas::NoTrans ? k : m);
      const integer ldb = (gemm_helper.right_op() == madness::cblas::NoTrans ? n : k);

      math::gemm(gemm_helper.left_op(), gemm_helper.right_op(), m, n, k, factor,
          pimpl_->data_, lda, other.data(), ldb, numeric_type(0), result.data(), n);

      return result;
    }

    /// Contract two tensors and store the result in this tensor

    /// \tparam U The left-hand tensor element type
    /// \tparam AU The left-hand tensor allocator type
    /// \tparam V The right-hand tensor element type
    /// \tparam AV The right-hand tensor allocator type
    /// \param left The left-hand tensor that will be contracted
    /// \param right The right-hand tensor that will be contracted
    /// \param factor The scaling factor
    /// \param gemm_helper The *GEMM operation meta data
    /// \return A new tensor which is the result of contracting this tensor with
    /// other
    /// \throw TiledArray::Exception When this tensor is empty.
    template <typename U, typename AU, typename V, typename AV>
    Tensor_& gemm(const Tensor<U, AU>& left, const Tensor<V, AV>& right,
        const numeric_type factor, const math::GemmHelper& gemm_helper)
    {
      // Check that this tensor is not empty and has the correct rank
      TA_ASSERT(pimpl_);
      TA_ASSERT(pimpl_->range_.rank() == gemm_helper.result_rank());

      // Check that the arguments are not empty and have the correct ranks
      TA_ASSERT(!left.empty());
      TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
      TA_ASSERT(!right.empty());
      TA_ASSERT(right.range().rank() == gemm_helper.right_rank());

      // Check that the outer dimensions of left match the the corresponding
      // dimensions in result
      TA_ASSERT(gemm_helper.left_result_coformal(left.range().lobound_data(),
          pimpl_->range_.lobound_data()));
      TA_ASSERT(gemm_helper.left_result_coformal(left.range().upbound_data(),
          pimpl_->range_.upbound_data()));
      TA_ASSERT(gemm_helper.left_result_coformal(left.range().extent_data(),
          pimpl_->range_.extent_data()));

      // Check that the outer dimensions of right match the the corresponding
      // dimensions in result
      TA_ASSERT(gemm_helper.right_result_coformal(right.range().lobound_data(),
          pimpl_->range_.lobound_data()));
      TA_ASSERT(gemm_helper.right_result_coformal(right.range().upbound_data(),
          pimpl_->range_.upbound_data()));
      TA_ASSERT(gemm_helper.right_result_coformal(right.range().extent_data(),
          pimpl_->range_.extent_data()));

      // Check that the inner dimensions of left and right match
      TA_ASSERT(gemm_helper.left_right_coformal(left.range().lobound_data(),
          right.range().lobound_data()));
      TA_ASSERT(gemm_helper.left_right_coformal(left.range().upbound_data(),
          right.range().upbound_data()));
      TA_ASSERT(gemm_helper.left_right_coformal(left.range().extent_data(),
          right.range().extent_data()));

      // Compute gemm dimensions
      integer m, n, k;
      gemm_helper.compute_matrix_sizes(m, n, k, left.range(), right.range());

      // Get the leading dimension for left and right matrices.
      const integer lda =
          (gemm_helper.left_op() == madness::cblas::NoTrans ? k : m);
      const integer ldb =
          (gemm_helper.right_op() == madness::cblas::NoTrans ? n : k);

      math::gemm(gemm_helper.left_op(), gemm_helper.right_op(), m, n, k, factor,
          left.data(), lda, right.data(), ldb, numeric_type(1), pimpl_->data_, n);

      return *this;
    }

    // Reduction operations

    /// Generalized tensor trace

    /// This function will compute the sum of the hyper diagonal elements of
    /// tensor.
    /// \return The trace of this tensor
    /// \throw TiledArray::Exception When this tensor is empty.
    value_type trace() const {
      TA_ASSERT(pimpl_);

      // Get pointers to the range data
      const size_type n = pimpl_->range_.rank();
      const size_type* restrict const lower = pimpl_->range_.lobound_data();
      const size_type* restrict const upper = pimpl_->range_.upbound_data();
      const size_type* restrict const stride = pimpl_->range_.stride_data();

      // Search for the largest lower bound and the smallest upper bound
      size_type lower_max = 0ul, upper_min =
          std::numeric_limits<size_type>::max();
      for(size_type i = 0ul; i < n; ++i) {
        const size_type lower_i = lower[i];
        const size_type upper_i = upper[i];

        lower_max = std::max(lower_max, lower_i);
        upper_min = std::min(upper_min, upper_i);
      }

      value_type result = 0;

      if(lower_max < upper_min) {
        // Compute the first and last ordinal index
        size_type first = 0ul, last = 0ul, trace_stride = 0ul;
        for(size_type i = 0ul; i < n; ++i) {
          const size_type lower_i = lower[i];
          const size_type stride_i = stride[i];

          first += (lower_max - lower_i) * stride_i;
          last += (upper_min - lower_i) * stride_i;
          trace_stride += stride_i;
        }

        // Compute the trace
        const value_type* restrict const data = pimpl_->data_;
        for(; first < last; first += trace_stride)
          result += data[first];
      }

      return result;
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

  }; // class Tensor

  template <typename T, typename A>
  const typename Tensor<T, A>::range_type Tensor<T, A>::empty_range_;

} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_TENSOR_H__INCLUDED
