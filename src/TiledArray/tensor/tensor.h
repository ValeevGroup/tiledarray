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
#include <TiledArray/tensor/complex.h>

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
    typedef typename TiledArray::detail::numeric_type<T>::type
        numeric_type; ///< the numeric type that supports T
    typedef typename TiledArray::detail::scalar_type<T>::type
        scalar_type; ///< the scalar type that supports T

  private:

    template <typename X>
    using numeric_t = typename TiledArray::detail::numeric_type<X>::type;

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
    Tensor() : pimpl_() { }
    Tensor(const Tensor_& other) : pimpl_(other.pimpl_) { }
    Tensor(Tensor_&& other) : pimpl_(std::move(other.pimpl_)) { }
    ~Tensor() { }
    Tensor_& operator=(const Tensor_& other) {
      pimpl_ = other.pimpl_;
      return *this;
    }
    Tensor_& operator=(Tensor_&& other)  {
      pimpl_ = std::move(other.pimpl_);
      return *this;
    }

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
    template <typename Value,
        typename std::enable_if<std::is_same<Value, value_type>::value &&
        detail::is_tensor<Value>::value>::type* = nullptr>
    Tensor(const range_type& range, const Value& value) :
      pimpl_(new Impl(range))
    {
      const size_type n = pimpl_->range_.volume();
      pointer MADNESS_RESTRICT const data = pimpl_->data_;
      for(size_type i = 0ul; i < n; ++i)
        new(data + i) value_type(value.clone());
    }

    /// Construct a tensor with a fill value

    /// \param range An array with the size of of each dimension
    /// \param value The value of the tensor elements
    template <typename Value,
        typename std::enable_if<detail::is_numeric<Value>::value>::type* = nullptr>
    Tensor(const range_type& range, const Value& value) :
      pimpl_(new Impl(range))
    {
      detail::tensor_init([=] () -> Value { return value; }, *this);
    }

    /// Construct an evaluated tensor
    template <typename InIter,
        typename std::enable_if<TiledArray::detail::is_input_iterator<InIter>::value &&
            ! std::is_pointer<InIter>::value>::type* = nullptr>
    Tensor(const range_type& range, InIter it) :
      pimpl_(new Impl(range))
    {
      size_type n = range.volume();
      pointer MADNESS_RESTRICT const data = pimpl_->data_;
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
          [] (const numeric_t<T1> arg) -> numeric_t<T1>
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
          [] (const numeric_t<T1> arg) -> numeric_t<T1>
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
      detail::tensor_init(op, perm, *this, other);
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
      detail::tensor_init(op, *this, left, right);
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
      detail::tensor_init(op, perm, *this, left, right);
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
      detail::inplace_tensor_op([] (reference MADNESS_RESTRICT tr,
          typename T1::const_reference MADNESS_RESTRICT t1) { tr = t1; }, *this, other);

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
      return Tensor_(*this, right, op);
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
      return Tensor_(*this, right, op, perm);
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
      detail::inplace_tensor_op(op, *this, right);
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
      return Tensor_(*this, op);
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
      return Tensor_(*this, op, perm);
    }

    /// Use a unary, element wise operation to modify this tensor

    /// \tparam Op The unary operation type
    /// \param op The unary, element-wise operation
    /// \return A reference to this object
    /// \throw TiledArray::Exception When this tensor is empty.
    template <typename Op>
    Tensor_& inplace_unary(Op&& op) {
      detail::inplace_tensor_op(op, *this);
      return *this;
    }

    // Scale operation

    /// Construct a scaled copy of this tensor

    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \return A new tensor where the elements of this tensor are scaled by
    /// \c factor
    template <typename Scalar,
        typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_ scale(const Scalar factor) const {
      return unary([=] (const numeric_type a) -> numeric_type
          { return a * factor; });
    }

    /// Construct a scaled and permuted copy of this tensor

    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param perm The permutation to be applied to this tensor
    /// \return A new tensor where the elements of this tensor are scaled by
    /// \c factor and permuted
    template <typename Scalar,
        typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_ scale(const Scalar factor, const Permutation& perm) const {
      return unary([=] (const numeric_type a) -> numeric_type
          { return a * factor; }, perm);
    }

    /// Scale this tensor

    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \return A reference to this tensor
    template <typename Scalar,
        typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_& scale_to(const Scalar factor) {
      return inplace_unary([=] (numeric_type& MADNESS_RESTRICT res) { res *= factor; });
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
          const numeric_t<Right> r)
          -> numeric_type { return l + r; });
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
          const numeric_t<Right> r)
          -> numeric_type { return l + r; }, perm);
    }

    /// Scale and add this and \c other to construct a new tensor

    /// \tparam Right The right-hand tensor type
    /// \tparam Scalar A scalar type
    /// \param right The tensor that will be added to this tensor
    /// \param factor The scaling factor
    /// \return A new tensor where the elements are the sum of the elements of
    /// \c this and \c other, scaled by \c factor
    template <typename Right, typename Scalar,
        typename std::enable_if<is_tensor<Right>::value &&
        detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_ add(const Right& right, const Scalar factor) const {
      return binary(right, [=] (const numeric_type l,
          const numeric_t<Right> r)
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
    template <typename Right, typename Scalar,
        typename std::enable_if<is_tensor<Right>::value &&
        detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_ add(const Right& right, const Scalar factor,
        const Permutation& perm) const
    {
      return binary(right,  [=] (const numeric_type l,
          const numeric_t<Right> r) -> numeric_type
          { return (l + r) * factor; }, perm);
    }

    /// Add a constant to a copy of this tensor

    /// \param value The constant to be added to this tensor
    /// \return A new tensor where the elements are the sum of the elements of
    /// \c this and \c value
    Tensor_ add(const numeric_type value) const {
      return unary([=] (const numeric_type a) -> numeric_type
          { return a + value; });
    }

    /// Add a constant to a permuted copy of this tensor

    /// \param value The constant to be added to this tensor
    /// \param perm The permutation to be applied to this tensor
    /// \return A new tensor where the elements are the sum of the elements of
    /// \c this and \c value
    Tensor_ add(const numeric_type value, const Permutation& perm) const {
      return unary([=] (const numeric_type a) -> numeric_type
          { return a + value; }, perm);
    }

    /// Add \c other to this tensor

    /// \tparam Right The right-hand tensor type
    /// \param right The tensor that will be added to this tensor
    /// \return A reference to this tensor
    template <typename Right,
        typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
    Tensor_& add_to(const Right& right) {
      return inplace_binary(right, [] (numeric_type& MADNESS_RESTRICT l,
          const numeric_t<Right> r) { l += r; });
    }

    /// Add \c other to this tensor, and scale the result

    /// \tparam Right The right-hand tensor type
    /// \tparam Scalar A scalar type
    /// \param right The tensor that will be added to this tensor
    /// \param factor The scaling factor
    /// \return A reference to this tensor
    template <typename Right, typename Scalar,
        typename std::enable_if<is_tensor<Right>::value &&
        detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_& add_to(const Right& right, const Scalar factor) {
      return inplace_binary(right, [=] (numeric_type& MADNESS_RESTRICT l,
          const numeric_t<Right> r)
          { (l += r) *= factor; });
    }

    /// Add a constant to this tensor

    /// \param value The constant to be added
    /// \return A reference to this tensor
    Tensor_& add_to(const numeric_type value) {
      return inplace_unary([=] (numeric_type& MADNESS_RESTRICT res) { res += value; });
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
          const numeric_t<Right> r)
          -> numeric_type { return l - r; });
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
          const numeric_t<Right> r)
          -> numeric_type { return l - r; }, perm);
    }

    /// Scale and subtract this and \c right to construct a new tensor

    /// \tparam Right The right-hand tensor type
    /// \tparam Scalar A scalar type
    /// \param right The tensor that will be subtracted from this tensor
    /// \param factor The scaling factor
    /// \return A new tensor where the elements are the different between the
    /// elements of \c this and \c right, scaled by \c factor
    template <typename Right, typename Scalar,
        typename std::enable_if<is_tensor<Right>::value &&
        detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_ subt(const Right& right, const Scalar factor) const {
      return binary(right, [=] (const numeric_type l,
          const numeric_t<Right> r)
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
    template <typename Right, typename Scalar,
        typename std::enable_if<is_tensor<Right>::value &&
        detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_ subt(const Right& right, const Scalar factor,
        const Permutation& perm) const
    {
      return binary(right, [=] (const numeric_type l,
          const numeric_t<Right> r)
          -> numeric_type { return (l - r) * factor; }, perm);
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
      return inplace_binary(right, [] (numeric_type& MADNESS_RESTRICT l,
          const numeric_t<Right> r)
          { l -= r; });
    }

    /// Subtract \c right from and scale this tensor

    /// \tparam Right The right-hand tensor type
    /// \tparam Scalar A scalar type
    /// \param right The tensor that will be subtracted from this tensor
    /// \param factor The scaling factor
    /// \return A reference to this tensor
    template <typename Right, typename Scalar,
        typename std::enable_if<is_tensor<Right>::value &&
        detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_& subt_to(const Right& right, const Scalar factor) {
      return inplace_binary(right, [=] (numeric_type& MADNESS_RESTRICT l,
          const numeric_t<Right> r)
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
          const numeric_t<Right> r)
          -> numeric_type { return l * r; });
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
          const numeric_t<Right> r)
          -> numeric_type { return l * r; }, perm);
    }

    /// Scale and multiply this by \c right to create a new tensor

    /// \tparam Right The right-hand tensor type
    /// \tparam Scalar A scalar type
    /// \param right The tensor that will be multiplied by this tensor
    /// \param factor The scaling factor
    /// \return A new tensor where the elements are the product of the elements
    /// of \c this and \c right, scaled by \c factor
    template <typename Right, typename Scalar,
        typename std::enable_if<is_tensor<Right>::value &&
        detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_ mult(const Right& right, const Scalar factor) const {
      return binary(right, [=] (const numeric_type l,
          const numeric_t<Right> r)
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
    template <typename Right, typename Scalar,
        typename std::enable_if<is_tensor<Right>::value &&
        detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_ mult(const Right& right, const Scalar factor,
        const Permutation& perm) const
    {
      return binary(right,  [=] (const numeric_type l,
          const numeric_t<Right> r)
          -> numeric_type { return (l * r) * factor; }, perm);
    }

    /// Multiply this tensor by \c right

    /// \tparam Right The right-hand tensor type
    /// \param right The tensor that will be multiplied by this tensor
    /// \return A reference to this tensor
    template <typename Right,
        typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
    Tensor_& mult_to(const Right& right) {
      return inplace_binary(right, [] (numeric_type& MADNESS_RESTRICT l,
          const numeric_t<Right> r)
          { l *= r; });
    }

    /// Scale and multiply this tensor by \c right

    /// \tparam Right The right-hand tensor type
    /// \tparam Scalar A scalar type
    /// \param right The tensor that will be multiplied by this tensor
    /// \param factor The scaling factor
    /// \return A reference to this tensor
    template <typename Right, typename Scalar,
        typename std::enable_if<is_tensor<Right>::value &&
        detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_& mult_to(const Right& right, const Scalar factor) {
      return inplace_binary(right, [=] (numeric_type& MADNESS_RESTRICT l,
          const numeric_t<Right> r)
          { (l *= r) *= factor; });
    }

    // Negation operations

    /// Create a negated copy of this tensor

    /// \return A new tensor that contains the negative values of this tensor
    Tensor_ neg() const {
      return unary([] (const numeric_type r) -> numeric_type { return -r; });
    }

    /// Create a negated and permuted copy of this tensor

    /// \param perm The permutation to be applied to this tensor
    /// \return A new tensor that contains the negative values of this tensor
    Tensor_ neg(const Permutation& perm) const {
      return unary([] (const numeric_type l) -> numeric_type { return -l; },
          perm);
    }

    /// Negate elements of this tensor

    /// \return A reference to this tensor
    Tensor_& neg_to() {
      return inplace_unary([] (numeric_type& MADNESS_RESTRICT l) { l = -l; });
    }


    /// Create a complex conjugated copy of this tensor

    /// \return A copy of this tensor that contains the complex conjugate the
    /// values
    Tensor_ conj() const {
      TA_ASSERT(pimpl_);
      return scale(detail::conj_op());
    }

    /// Create a complex conjugated and scaled copy of this tensor

    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \return A copy of this tensor that contains the scaled complex
    /// conjugate the values
    template <typename Scalar,
        typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_ conj(const Scalar factor) const {
      TA_ASSERT(pimpl_);
      return scale(detail::conj_op(factor));
    }

    /// Create a complex conjugated and permuted copy of this tensor

    /// \param perm The permutation to be applied to this tensor
    /// \return A permuted copy of this tensor that contains the complex
    /// conjugate values
    Tensor_ conj(const Permutation& perm) const {
      TA_ASSERT(pimpl_);
      return scale(detail::conj_op(), perm);
    }

    /// Create a complex conjugated, scaled, and permuted copy of this tensor

    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param perm The permutation to be applied to this tensor
    /// \return A permuted copy of this tensor that contains the complex
    /// conjugate values
    template <typename Scalar,
        typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_ conj(const Scalar factor, const Permutation& perm) const {
      TA_ASSERT(pimpl_);
      return scale(detail::conj_op(factor), perm);
    }

    /// Complex conjugate this tensor

    /// \return A reference to this tensor
    Tensor_& conj_to() {
      TA_ASSERT(pimpl_);
      return scale_to(detail::conj_op());
    }

    /// Complex conjugate and scale this tensor

    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \return A reference to this tensor
    template <typename Scalar,
        typename std::enable_if<detail::is_numeric<Scalar>::value>::type* = nullptr>
    Tensor_& conj_to(const Scalar factor) {
      TA_ASSERT(pimpl_);
      return scale_to(detail::conj_op(factor));
    }

    // GEMM operations

    /// Contract this tensor with \c other

    /// \tparam U The other tensor element type
    /// \tparam AU The other tensor allocator type
    /// \tparam V The type of \c factor scalar
    /// \param other The tensor that will be contracted with this tensor
    /// \param factor Multiply the result by this constant
    /// \param gemm_helper The *GEMM operation meta data
    /// \return A new tensor which is the result of contracting this tensor with
    /// \c other and scaled by \c factor
    /// \throw TiledArray::Exception When this tensor is empty.
    /// \throw TiledArray::Exception When \c other is empty.
    template <typename U, typename AU, typename V>
    Tensor_ gemm(const Tensor<U, AU>& other, const V factor,
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

    /// Gemm is limited to matrix like contractions. For example, the following
    /// contractions are supported:
    /// \code
    /// C[a,b] = A[a,i,j] * B[i,j,b]
    /// C[a,b] = A[a,i,j] * B[b,i,j]
    /// C[a,b] = A[i,j,a] * B[i,j,b]
    /// C[a,b] = A[i,j,a] * B[b,i,j]
    ///
    /// C[a,b,c,d] = A[a,b,i,j] * B[i,j,c,d]
    /// C[a,b,c,d] = A[a,b,i,j] * B[c,d,i,j]
    /// C[a,b,c,d] = A[i,j,a,b] * B[i,j,c,d]
    /// C[a,b,c,d] = A[i,j,a,b] * B[c,d,i,j]
    /// \endcode
    /// Notice that in the above contractions, the inner and outer indices of
    /// the arguments for exactly two contiguous groups in each tensor and that
    /// each group is in the same order in all tensors. That is, the indices of
    /// the tensors must fit the one of the following patterns:
    /// \code
    /// C[M...,N...] = A[M...,K...] * B[K...,N...]
    /// C[M...,N...] = A[M...,K...] * B[N...,K...]
    /// C[M...,N...] = A[K...,M...] * B[K...,N...]
    /// C[M...,N...] = A[K...,M...] * B[N...,K...]
    /// \endcode
    /// This allows use of optimized BLAS functions to evaluate tensor
    /// contractions. Tensor contractions that do not fit this pattern require
    /// one or more tensor permutation so that the tensors fit the required
    /// pattern.
    /// \tparam U The left-hand tensor element type
    /// \tparam AU The left-hand tensor allocator type
    /// \tparam V The right-hand tensor element type
    /// \tparam AV The right-hand tensor allocator type
    /// \tparam W The type of the scaling factor
    /// \param left The left-hand tensor that will be contracted
    /// \param right The right-hand tensor that will be contracted
    /// \param factor The scaling factor
    /// \param gemm_helper The *GEMM operation meta data
    /// \return A new tensor which is the result of contracting this tensor with
    /// other
    /// \throw TiledArray::Exception When this tensor is empty.
    template <typename U, typename AU, typename V, typename AV, typename W>
    Tensor_& gemm(const Tensor<U, AU>& left, const Tensor<V, AV>& right,
        const W factor, const math::GemmHelper& gemm_helper)
    {
      // Check that this tensor is not empty and has the correct rank
      TA_ASSERT(pimpl_);
      TA_ASSERT(pimpl_->range_.rank() == gemm_helper.result_rank());

      // Check that the arguments are not empty and have the correct ranks
      TA_ASSERT(!left.empty());
      TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
      TA_ASSERT(!right.empty());
      TA_ASSERT(right.range().rank() == gemm_helper.right_rank());

      // Check that the outer dimensions of left match the corresponding
      // dimensions in result
      TA_ASSERT(gemm_helper.left_result_coformal(left.range().lobound_data(),
          pimpl_->range_.lobound_data()));
      TA_ASSERT(gemm_helper.left_result_coformal(left.range().upbound_data(),
          pimpl_->range_.upbound_data()));
      TA_ASSERT(gemm_helper.left_result_coformal(left.range().extent_data(),
          pimpl_->range_.extent_data()));

      // Check that the outer dimensions of right match the corresponding
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
      const size_type* MADNESS_RESTRICT const lower = pimpl_->range_.lobound_data();
      const size_type* MADNESS_RESTRICT const upper = pimpl_->range_.upbound_data();
      const size_type* MADNESS_RESTRICT const stride = pimpl_->range_.stride_data();

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
        const value_type* MADNESS_RESTRICT const data = pimpl_->data_;
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
    template <typename ReduceOp, typename JoinOp, typename Scalar>
    decltype(auto) reduce(ReduceOp&& reduce_op, JoinOp&& join_op,
                Scalar identity) const
    {
      return detail::tensor_reduce(reduce_op, join_op, identity, *this);
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
    template <typename Right, typename ReduceOp, typename JoinOp, typename Scalar,
        typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
    decltype(auto) reduce(const Right& other, ReduceOp&& reduce_op, JoinOp&& join_op,
                Scalar identity) const
    {
      return detail::tensor_reduce(reduce_op, join_op, identity, *this, other);
    }

    /// Sum of elements

    /// \return The sum of all elements of this tensor
    numeric_type sum() const {
      auto sum_op = [] (numeric_type& MADNESS_RESTRICT res, const numeric_type arg)
              { res += arg; };
      return reduce(sum_op, sum_op, numeric_type(0));
    }

    /// Product of elements

    /// \return The product of all elements of this tensor
    numeric_type product() const {
      auto mult_op = [] (numeric_type& MADNESS_RESTRICT res, const numeric_type arg)
              { res *= arg; };
      return reduce(mult_op, mult_op, numeric_type(1));
    }

    /// Square of vector 2-norm

    /// \return The vector norm of this tensor
    scalar_type squared_norm() const {
      auto square_op = [] (scalar_type& MADNESS_RESTRICT res, const numeric_type arg)
              { res += TiledArray::detail::norm(arg); };
      auto sum_op = [] (scalar_type& MADNESS_RESTRICT res, const scalar_type arg)
              { res += arg; };
      return detail::tensor_reduce(square_op, sum_op, scalar_type(0), *this);
    }

    /// Vector 2-norm

    /// \return The vector norm of this tensor
    scalar_type norm() const {
      return std::sqrt(squared_norm());
    }

    /// Minimum element

    /// \return The minimum elements of this tensor
    template <typename Numeric = numeric_type>
    numeric_type min(typename std::enable_if<
                         detail::is_strictly_ordered<Numeric>::value>::type* =
                         nullptr) const {
      auto min_op = [](numeric_type& MADNESS_RESTRICT res, const numeric_type arg) {
        res = std::min(res, arg);
      };
      return reduce(min_op, min_op, std::numeric_limits<numeric_type>::max());
    }

    /// Maximum element

    /// \return The maximum elements of this tensor
    template <typename Numeric = numeric_type>
    numeric_type max(typename std::enable_if<
                         detail::is_strictly_ordered<Numeric>::value>::type* =
                         nullptr) const {
      auto max_op = [](numeric_type& MADNESS_RESTRICT res, const numeric_type arg) {
        res = std::max(res, arg);
      };
      return reduce(max_op, max_op, std::numeric_limits<scalar_type>::min());
    }

    /// Absolute minimum element

    /// \return The minimum elements of this tensor
    scalar_type abs_min() const {
      auto abs_min_op = [] (scalar_type& MADNESS_RESTRICT res, const numeric_type arg)
              { res = std::min(res, std::abs(arg)); };
      auto min_op = [] (scalar_type& MADNESS_RESTRICT res, const scalar_type arg)
              { res = std::min(res, arg); };
      return reduce(abs_min_op, min_op, std::numeric_limits<scalar_type>::max());
    }

    /// Absolute maximum element

    /// \return The maximum elements of this tensor
    scalar_type abs_max() const {
      auto abs_max_op = [] (scalar_type& MADNESS_RESTRICT res, const numeric_type arg)
              { res = std::max(res, std::abs(arg)); };
      auto max_op = [] (scalar_type& MADNESS_RESTRICT res, const scalar_type arg)
              { res = std::max(res, arg); };
      return reduce(abs_max_op, max_op, scalar_type(0));
    }

    /// Vector dot product

    /// \tparam Right The right-hand tensor type
    /// \param other The right-hand tensor to be reduced
    /// \return The inner product of the this and \c other
    template <typename Right,
        typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
    numeric_type dot(const Right& other) const {
      auto mult_add_op = [] (numeric_type& res, const numeric_type l,
                const numeric_t<Right> r)
                { res += l * r; };
      auto add_op = [] (numeric_type& MADNESS_RESTRICT res, const numeric_type value)
            { res += value; };
      return reduce(other, mult_add_op, add_op, numeric_type(0));
    }

  }; // class Tensor

  template <typename T, typename A>
  const typename Tensor<T, A>::range_type Tensor<T, A>::empty_range_;

#ifndef TILEDARRAY_HEADER_ONLY

  extern template
  class Tensor<double, Eigen::aligned_allocator<double> >;
  extern template
  class Tensor<float, Eigen::aligned_allocator<float> >;
  extern template
  class Tensor<int, Eigen::aligned_allocator<int> >;
  extern template
  class Tensor<long, Eigen::aligned_allocator<long> >;
//  extern template
//  class Tensor<std::complex<double>, Eigen::aligned_allocator<std::complex<double> > >;
//  extern template
//  class Tensor<std::complex<float>, Eigen::aligned_allocator<std::complex<float> > >;

#endif // TILEDARRAY_HEADER_ONLY

} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_TENSOR_H__INCLUDED
