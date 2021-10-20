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

#include "TiledArray/host/allocator.h"

#include "TiledArray/math/blas.h"
#include "TiledArray/math/gemm_helper.h"
#include "TiledArray/tensor/complex.h"
#include "TiledArray/tensor/kernels.h"
#include "TiledArray/tile_interface/clone.h"
#include "TiledArray/tile_interface/permute.h"
#include "TiledArray/tile_interface/trace.h"
#include "TiledArray/util/logger.h"

namespace TiledArray {

template <typename Alpha, typename... As, typename... Bs, typename Beta,
          typename... Cs>
void gemm(Alpha alpha, const Tensor<As...>& A, const Tensor<Bs...>& B,
          Beta beta, Tensor<Cs...>& C, const math::GemmHelper& gemm_helper);

namespace detail {

#ifdef TA_TENSOR_MEM_PROFILE
inline static std::mutex
    ta_tensor_mem_profile_mtx;  // protects the following statics
inline static std::uint64_t nbytes_allocated = 0;
inline static std::uint64_t max_nbytes_allocated = 0;
#endif  // TA_TENSOR_MEM_PROFILE

/// Signals that we can take the trace of a Tensor<T, A> (for numeric \c T)
template <typename T, typename A>
struct TraceIsDefined<Tensor<T, A>, enable_if_numeric_t<T>> : std::true_type {};

}  // namespace detail

/// An N-dimensional tensor object

/// A contiguous row-major tensor with shallow-copy semantics.
/// As of TiledArray 1.1 Tensor represents a batch of tensors with same Range
/// (the default batch size = 1).
/// \tparam T the value type of this tensor
/// \tparam A The allocator type for the data
template <typename T, typename Allocator>
class Tensor {
  // meaningful error if T& is not assignable, see
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=48101
  static_assert(
      std::is_assignable<std::add_lvalue_reference_t<T>, T>::value,
      "Tensor<T>: T must be an assignable type (e.g. cannot be const)");

 public:
  typedef Range range_type;                              ///< Tensor range type
  typedef typename range_type::index1_type index1_type;  ///< 1-index type
  typedef typename range_type::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename range_type::ordinal_type
      size_type;  ///< Size type (to meet the container concept)
  typedef Allocator allocator_type;  ///< Allocator type
  typedef
      typename allocator_type::value_type value_type;  ///< Array element type
  typedef
      typename allocator_type::reference reference;  ///< Element reference type
  typedef typename allocator_type::const_reference
      const_reference;                               ///< Element reference type
  typedef typename allocator_type::pointer pointer;  ///< Element pointer type
  typedef typename allocator_type::const_pointer
      const_pointer;  ///< Element const pointer type
  typedef typename allocator_type::difference_type
      difference_type;                   ///< Difference type
  typedef pointer iterator;              ///< Element iterator type
  typedef const_pointer const_iterator;  ///< Element const iterator type
  typedef typename TiledArray::detail::numeric_type<T>::type
      numeric_type;  ///< the numeric type that supports T
  typedef typename TiledArray::detail::scalar_type<T>::type
      scalar_type;  ///< the scalar type that supports T

 private:
  template <typename X>
  using numeric_t = typename TiledArray::detail::numeric_type<X>::type;

#ifdef TA_TENSOR_MEM_PROFILE
  enum class MemOp { Alloc, Dealloc };
  void alloc_record(std::uint64_t n, MemOp action) {
    const double to_MiB = 1 / (1024.0 * 1024.0); /* Convert from bytes to MiB */
    const auto nbytes = n * sizeof(value_type);
    {
      std::scoped_lock lock(detail::ta_tensor_mem_profile_mtx);
      if (action == MemOp::Alloc) {
        detail::nbytes_allocated += nbytes;
        detail::max_nbytes_allocated =
            std::max(detail::nbytes_allocated, detail::max_nbytes_allocated);
      } else
        detail::nbytes_allocated -= nbytes;
    }
    char buf[1024];
    auto value_type_str = []() {
      if constexpr (std::is_same_v<value_type, double>)
        return "double";
      else if constexpr (std::is_same_v<value_type, float>)
        return "float";
      else if constexpr (std::is_same_v<value_type, std::complex<double>>)
        return "zdouble";
      else if constexpr (std::is_same_v<value_type, std::complex<float>>)
        return "zfloat";
      else
        return "";
    };
    std::snprintf(
        buf, 1023,
        "TA::Tensor<%s>: %sallocated %lf MiB [wm = %lf MiB hwm = %lf MiB]\n",
        value_type_str(), (action == MemOp::Dealloc ? "de" : "  "),
        nbytes * to_MiB, detail::nbytes_allocated * to_MiB,
        detail::max_nbytes_allocated * to_MiB);
    auto& os = madness::print_meminfo_ostream();
    os << buf;
    os.flush();
  }
#endif

  template <typename... Ts>
  struct is_tensor {
    static constexpr bool value = detail::is_tensor<Ts...>::value ||
                                  detail::is_tensor_of_tensor<Ts...>::value;
  };

  using default_construct = bool;

  Tensor(const range_type& range, size_t batch_size, bool default_construct)
      : range_(range), batch_size_(batch_size) {
    size_t size = range_.volume() * batch_size;
    allocator_type allocator;
    auto* ptr = allocator.allocate(size);
#ifdef TA_TENSOR_MEM_PROFILE
    alloc_record(size, MemOp::Alloc);
#endif
    if (default_construct) {
      std::uninitialized_default_construct_n(ptr, size);
      // std::uninitialized_value_construct_n(ptr, size);
    }
    auto deleter = [this, allocator = std::move(allocator),
                    size](auto&& ptr) mutable {
      std::destroy_n(ptr, size);
      allocator.deallocate(ptr, size);
#ifdef TA_TENSOR_MEM_PROFILE
      alloc_record(size, MemOp::Dealloc);
#endif
    };
    this->data_ = std::shared_ptr<value_type>(ptr, std::move(deleter));
  }

  Tensor(range_type&& range, size_t batch_size, bool default_construct)
      : range_(std::move(range)), batch_size_(batch_size) {
    size_t size = range_.volume() * batch_size;
    allocator_type allocator;
    auto* ptr = allocator.allocate(size);
#ifdef TA_TENSOR_MEM_PROFILE
    alloc_record(size, MemOp::Alloc);
#endif
    if (default_construct) {
      std::uninitialized_default_construct_n(ptr, size);
      // std::uninitialized_value_construct_n(ptr, size);
    }
    auto deleter = [this, allocator = std::move(allocator),
                    size](auto&& ptr) mutable {
      std::destroy_n(ptr, size);
      allocator.deallocate(ptr, size);
#ifdef TA_TENSOR_MEM_PROFILE
      alloc_record(size, MemOp::Dealloc);
#endif
    };
    this->data_ = std::shared_ptr<value_type>(ptr, std::move(deleter));
  }

  /// Construct a tensor with a range equal to \c range. The data is
  /// uninitialized.
  /// \param range The range of the tensor
  /// \param batch_size The batch size
  /// \param data shared pointer to the data
  Tensor(const range_type& range, size_t batch_size,
         std::shared_ptr<value_type> data)
      : range_(range), batch_size_(batch_size), data_(data) {}

  range_type range_;  ///< range
  size_t batch_size_ = 1;
  std::shared_ptr<value_type> data_;  ///< Shared pointer to the data

 public:
  // Compiler generated functions
  Tensor() = default;

  /// Construct a tensor with a range equal to \c range. The data is
  /// uninitialized.
  /// \param range The range of the tensor
  explicit Tensor(const range_type& range)
      : Tensor(range, 1, default_construct{true}) {}

  /// Construct a tensor with a fill value

  /// \param range An array with the size of of each dimension
  /// \param value The value of the tensor elements
  template <
      typename Value,
      typename std::enable_if<std::is_same<Value, value_type>::value &&
                              detail::is_tensor<Value>::value>::type* = nullptr>
  Tensor(const range_type& range, const Value& value)
      : Tensor(range, 1, default_construct{false}) {
    const auto n = this->size();
    pointer MADNESS_RESTRICT const data = this->data();
    Clone<Value, Value> cloner;
    for (size_type i = 0ul; i < n; ++i)
      new (data + i) value_type(cloner(value));
  }

  /// Construct a tensor with a fill value

  /// \param range An array with the size of of each dimension
  /// \param value The value of the tensor elements
  template <typename Value, typename std::enable_if<
                                detail::is_numeric_v<Value>>::type* = nullptr>
  Tensor(const range_type& range, const Value& value)
      : Tensor(range, 1, default_construct{false}) {
    detail::tensor_init([value]() -> Value { return value; }, *this);
  }

  /// Construct an evaluated tensor
  template <typename InIter,
            typename std::enable_if<
                TiledArray::detail::is_input_iterator<InIter>::value &&
                !std::is_pointer<InIter>::value>::type* = nullptr>
  Tensor(const range_type& range, InIter it)
      : Tensor(range, 1, default_construct{false}) {
    auto n = range.volume();
    pointer MADNESS_RESTRICT const data = this->data();
    for (size_type i = 0ul; i < n; ++i, ++it) data[i] = *it;
  }

  template <typename U>
  Tensor(const Range& range, const U* u)
      : Tensor(range, 1, default_construct{false}) {
    math::uninitialized_copy_vector(range.volume(), u, this->data());
  }

  Tensor(const Range& range, std::initializer_list<T> il)
      : Tensor(range, il.begin()) {}

  /// Construct a copy of a tensor interface object

  /// \tparam T1 A tensor type
  /// \param other The tensor to be copied
  /// \note this constructor is disabled if \p T1 already has a conversion
  ///       operator to this type
  template <
      typename T1,
      typename std::enable_if<
          is_tensor<T1>::value && !std::is_same<T1, Tensor>::value &&
          !detail::has_conversion_operator_v<T1, Tensor>>::type* = nullptr>
  explicit Tensor(const T1& other)
      : Tensor(detail::clone_range(other), 1, default_construct{false}) {
    auto op = [](const numeric_t<T1> arg) -> numeric_t<T1> { return arg; };

    detail::tensor_init(op, *this, other);
  }

  /// Construct a permuted tensor copy

  /// \tparam T1 A tensor type
  /// \tparam Perm A permutation type
  /// \param other The tensor to be copied
  /// \param perm The permutation that will be applied to the copy
  template <
      typename T1, typename Perm,
      typename std::enable_if<is_tensor<T1>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor(const T1& other, const Perm& perm)
      : Tensor(outer(perm) * other.range(), 1, default_construct{false}) {
    auto op = [](const numeric_t<T1> arg) -> numeric_t<T1> { return arg; };

    detail::tensor_init(op, outer(perm), *this, other);

    // If we actually have a ToT the inner permutation was not applied above so
    // we do that now
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    // static_assert(is_tot || (!is_tot && !is_bperm), "Permutation type does
    // not match Tensor");
    if constexpr (is_tot && is_bperm) {
      if (inner_size(perm) != 0) {
        auto inner_perm = inner(perm);
        Permute<value_type, value_type> p;
        for (auto& x : *this) x = p(x, inner_perm);
      }
    }
  }

  /// Copy and modify the data from \c other

  /// \tparam T1 A tensor type
  /// \tparam Op An element-wise operation type
  /// \param other The tensor argument
  /// \param op The element-wise operation
  template <typename T1, typename Op,
            typename std::enable_if_t<
                is_tensor<T1>::value &&
                !detail::is_permutation_v<std::decay_t<Op>>>* = nullptr>
  Tensor(const T1& other, Op&& op)
      : Tensor(detail::clone_range(other), 1, default_construct{false}) {
    detail::tensor_init(op, *this, other);
  }

  /// Copy, modify, and permute the data from \c other

  /// \tparam T1 A tensor type
  /// \tparam Op An element-wise operation type
  /// \tparam Perm A permutation type
  /// \param other The tensor argument
  /// \param op The element-wise operation
  template <
      typename T1, typename Op, typename Perm,
      typename std::enable_if_t<is_tensor<T1>::value &&
                                detail::is_permutation_v<Perm>>* = nullptr>
  Tensor(const T1& other, Op&& op, const Perm& perm)
      : Tensor(outer(perm) * other.range(), 1, default_construct{false}) {
    detail::tensor_init(op, outer(perm), *this, other);
    // If we actually have a ToT the inner permutation was not applied above so
    // we do that now
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    // static_assert(is_tot || (!is_tot && !is_bperm), "Permutation type does
    // not match Tensor");
    if constexpr (is_tot && is_bperm) {
      if (inner_size(perm) != 0) {
        auto inner_perm = inner(perm);
        Permute<value_type, value_type> p;
        for (auto& x : *this) x = p(x, inner_perm);
      }
    }
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
  Tensor(const T1& left, const T2& right, Op&& op)
      : Tensor(detail::clone_range(left), 1, default_construct{false}) {
    detail::tensor_init(op, *this, left, right);
  }

  /// Copy, modify, and permute the data from \c left, and \c right

  /// \tparam T1 A tensor type
  /// \tparam T2 A tensor type
  /// \tparam Op An element-wise operation type
  /// \tparam Perm A permutation tile
  /// \param left The left-hand tensor argument
  /// \param right The right-hand tensor argument
  /// \param op The element-wise operation
  /// \param perm The permutation that will be applied to the arguments
  template <
      typename T1, typename T2, typename Op, typename Perm,
      typename std::enable_if<is_tensor<T1, T2>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor(const T1& left, const T2& right, Op&& op, const Perm& perm)
      : Tensor(outer(perm) * left.range(), 1, default_construct{false}) {
    detail::tensor_init(op, outer(perm), *this, left, right);
    // If we actually have a ToT the inner permutation was not applied above so
    // we do that now
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    // static_assert(is_tot || (!is_tot && !is_bperm), "Permutation type does
    // not match Tensor");
    if constexpr (is_tot && is_bperm) {
      if (inner_size(perm) != 0) {
        auto inner_perm = inner(perm);
        Permute<value_type, value_type> p;
        for (auto& x : *this) x = p(x, inner_perm);
      }
    }
  }

  /// The batch size accessor

  /// @return the size of tensor batch represented by `*this`
  size_t batch_size() const { return this->batch_size_; }

  /// @param[in] idx the batch index
  /// @pre `idx < this->batch_size()`
  /// @return (plain, i.e. batch_size=1) Tensor representing element \p idx of
  /// the batch
  Tensor batch(size_t idx) const {
    TA_ASSERT(idx < this->batch_size());
    std::shared_ptr<value_type> data(this->data_,
                                     this->data_.get() + idx * this->size());
    return Tensor(this->range(), 1, data);
  }

  /// Returns Tensor representing the data using another range and batch size

  /// @param[in] range the Range of the result
  /// @param[in] batch_size the batch size of the result
  /// @return Tensor object representing `this->data()` using @p range and @p
  /// batch_size
  auto reshape(const range_type& range, size_t batch_size = 1) const {
    TA_ASSERT(this->range().volume() * this->batch_size() ==
              range.volume() * batch_size);
    return Tensor(range, batch_size, this->data_);
  }

  /// @return a deep copy of `*this`
  Tensor clone() const {
    Tensor result;
    if (data_) {
      result = detail::tensor_op<Tensor>(
          [](const numeric_type value) -> numeric_type { return value; },
          *this);
    }
    return result;
  }

  template <typename T1,
            typename std::enable_if<is_tensor<T1>::value>::type* = nullptr>
  Tensor& operator=(const T1& other) {
    *this = Tensor(detail::clone_range(other), 1, default_construct{false});
    detail::inplace_tensor_op(
        [](reference MADNESS_RESTRICT tr,
           typename T1::const_reference MADNESS_RESTRICT t1) { tr = t1; },
        *this, other);

    return *this;
  }

  /// Tensor range object accessor

  /// \return The tensor range object
  const range_type& range() const { return range_; }

  /// Tensor dimension size accessor

  /// \return The number of elements in the tensor
  ordinal_type size() const { return (this->range().volume()); }

  /// Const element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Const reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  const_reference operator[](const Ordinal ord) const {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(ord));
    return this->data()[ord];
  }

  /// Element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  reference operator[](const Ordinal ord) {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(ord));
    return this->data()[ord];
  }

  /// Const element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  const_reference operator[](const Index& i) const {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(i));
    return this->data()[this->range_.ordinal(i)];
  }

  /// Element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  reference operator[](const Index& i) {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(i));
    return this->data()[this->range_.ordinal(i)];
  }

  /// Const element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  const_reference operator[](const std::initializer_list<Integer>& i) const {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(i));
    return this->data()[this->range_.ordinal(i)];
  }

  /// Element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  reference operator[](const std::initializer_list<Integer>& i) {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(i));
    return this->data()[this->range_.ordinal(i)];
  }

  /// Const element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  const_reference operator()(const Index& i) const {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(i));
    return this->data()[this->range_.ordinal(i)];
  }

  /// Element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  reference operator()(const Index& i) {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(i));
    return this->data()[this->range_.ordinal(i)];
  }

  /// Const element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  const_reference operator()(const std::initializer_list<Integer>& i) const {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(i));
    return this->data()[this->range_.ordinal(i)];
  }

  /// Element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  reference operator()(const std::initializer_list<Integer>& i) {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(i));
    return this->data()[this->range_.ordinal(i)];
  }

  /// Const element accessor

  /// \tparam Index an integral list ( see TiledArray::detail::is_integral_list
  /// ) \param[in] i an index \return Const reference to the element at position
  /// \c i . \note This asserts (using TA_ASSERT) that this is not empty and ord
  /// is included in the range
  template <
      typename... Index,
      std::enable_if_t<detail::is_integral_list<Index...>::value>* = nullptr>
  const_reference operator()(const Index&... i) const {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(i...));
    return this->data()[this->range_.ordinal(i...)];
  }

  /// Element accessor

  /// \tparam Index an integral list ( see TiledArray::detail::is_integral_list
  /// ) \param[in] i an index \return Reference to the element at position \c i
  /// . \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <
      typename... Index,
      std::enable_if_t<detail::is_integral_list<Index...>::value>* = nullptr>
  reference operator()(const Index&... i) {
    // TA_ASSERT(pimpl_);
    TA_ASSERT(this->range_.includes(i...));
    return this->data()[this->range_.ordinal(i...)];
  }

  /// Iterator factory

  /// \return An iterator to the first data element
  const_iterator begin() const { return (this->data() ? this->data() : NULL); }

  /// Iterator factory

  /// \return An iterator to the first data element
  iterator begin() { return (this->data() ? this->data() : NULL); }

  /// Iterator factory

  /// \return An iterator to the last data element
  const_iterator end() const {
    return (this->data() ? this->data() + this->size() : NULL);
  }

  /// Iterator factory

  /// \return An iterator to the last data element
  iterator end() { return (this->data() ? this->data() + this->size() : NULL); }

  /// Read-only access to the data

  /// \return A const pointer to the tensor data
  const_pointer data() const { return this->data_.get(); }

  /// Mutable access to the data

  /// \return A const pointer to the tensor data
  pointer data() { return this->data_.get(); }

  /// Test if the tensor is empty

  /// \return \c true if this tensor was default constructed (contains no
  /// data), otherwise \c false.
  bool empty() const { return !this->data_; }

  /// MADNESS serialization function

  /// This function enables serialization within MADNESS
  /// \tparam Archive A MADNESS archive type
  /// \param[out] ar An input/output archive
  template <typename Archive>
  void serialize(Archive& ar) {
    bool empty = this->empty();
    auto range = this->range_;
    auto batch_size = this->batch_size_;
    ar& empty;
    if (!empty) {
      ar& range;
      ar& batch_size;
      if constexpr (madness::is_input_archive_v<Archive>) {
        *this = Tensor(std::move(range), batch_size, default_construct{true});
      }
      ar& madness::archive::wrap(this->data_.get(),
                                 this->range_.volume() * batch_size);
    } else {
      if constexpr (madness::is_input_archive_v<Archive>) {
        *this = Tensor{};
      }
    }
  }

  /// Swap tensor data

  /// \param other The tensor to swap with this
  void swap(Tensor& other) {
    std::swap(data_, other.data_);
    std::swap(range_, other.range_);
    std::swap(batch_size_, other.batch_size_);
  }

  // clang-format off
  /// Constructs a view of the block defined by \p lower_bound and \p upper_bound.

  /// Examples of using this:
  /// \code
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///   auto tview = t.block(lobounds, upbounds);
  ///   assert(tview.range().includes(lobounds));
  ///   assert(tview(lobounds) == t(lobounds));
  /// \endcode
  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param lower_bound The lower bound
  /// \param upper_bound The upper bound
  /// \return a {const,mutable} view of the block defined by \p lower_bound and \p upper_bound
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `lower_bound[i] >= upper_bound[i]`
  // clang-format on
  /// @{
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  detail::TensorInterface<T, BlockRange> block(const Index1& lower_bound,
                                               const Index2& upper_bound) {
    // TA_ASSERT(pimpl_);
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, lower_bound, upper_bound), this->data());
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  detail::TensorInterface<const T, BlockRange> block(
      const Index1& lower_bound, const Index2& upper_bound) const {
    // TA_ASSERT(pimpl_);
    return detail::TensorInterface<const T, BlockRange>(
        BlockRange(this->range_, lower_bound, upper_bound), this->data());
  }
  /// @}

  // clang-format off
  /// Constructs a view of the block defined by \p lower_bound and \p upper_bound.

  /// Examples of using this:
  /// \code
  ///   auto tview = t.block({0, 1, 2}, {4, 6, 8});
  ///   assert(tview.range().includes(lobounds));
  ///   assert(tview(lobounds) == t(lobounds));
  /// \endcode
  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \param lower_bound The lower bound
  /// \param upper_bound The upper bound
  /// \return a {const,mutable} view of the block defined by \p lower_bound and \p upper_bound
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `lower_bound[i] >= upper_bound[i]`
  // clang-format on
  /// @{
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  detail::TensorInterface<T, BlockRange> block(
      const std::initializer_list<Index1>& lower_bound,
      const std::initializer_list<Index2>& upper_bound) {
    // TA_ASSERT(pimpl_);
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, lower_bound, upper_bound), this->data());
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  detail::TensorInterface<const T, BlockRange> block(
      const std::initializer_list<Index1>& lower_bound,
      const std::initializer_list<Index2>& upper_bound) const {
    // TA_ASSERT(pimpl_);
    return detail::TensorInterface<const T, BlockRange>(
        BlockRange(this->range_, lower_bound, upper_bound), this->data());
  }
  /// @}

  // clang-format off
  /// Constructs a view of the block defined by its \p bounds .

  /// Examples of using this:
  /// \code
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///
  ///   // using vector of pairs
  ///   std::vector<std::pair<size_t,size_t>> vpbounds{{0,4}, {1,6}, {2,8}};
  ///   auto tview0 = t.block(vpbounds);
  ///   // using vector of tuples
  ///   std::vector<std::tuple<size_t,size_t>> vtbounds{{0,4}, {1,6}, {2,8}};
  ///   auto tview1 = t.block(vtbounds);
  ///   assert(tview0 == tview1);
  ///
  ///   // using zipped ranges of bounds (using Boost.Range)
  ///   // need to #include <boost/range/combine.hpp>
  ///   auto tview2 = t.block(boost::combine(lobounds, upbounds));
  ///   assert(tview0 == tview2);
  ///
  ///   // using zipped ranges of bounds (using Ranges-V3)
  ///   // need to #include <range/v3/view/zip.hpp>
  ///   auto tview3 = t.block(ranges::views::zip(lobounds, upbounds));
  ///   assert(tview0 == tview3);
  /// \endcode
  /// \tparam PairRange Type representing a range of generalized pairs (see TiledArray::detail::is_gpair_v )
  /// \param bounds The block bounds
  /// \return a {const,mutable} view of the block defined by its \p bounds
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `get<0>(bounds[i]) >= get<1>(bounds[i])`
  // clang-format on
  /// @{
  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  detail::TensorInterface<const T, BlockRange> block(
      const PairRange& bounds) const {
    return detail::TensorInterface<const T, BlockRange>(
        BlockRange(this->range_, bounds), this->data());
  }

  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  detail::TensorInterface<T, BlockRange> block(const PairRange& bounds) {
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, bounds), this->data());
  }
  /// @}

  // clang-format off
  /// Constructs a view of the block defined by its \p bounds .

  /// Examples of using this:
  /// \code
  ///   auto tview0 = t.block({{0,4}, {1,6}, {2,8}});
  /// \endcode
  /// \tparam Index An integral type
  /// \param bounds The block bounds
  /// \return a {const,mutable} view of the block defined by its \p bounds
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `get<0>(bounds[i]) >= get<1>(bounds[i])`
  // clang-format on
  /// @{
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  detail::TensorInterface<const T, BlockRange> block(
      const std::initializer_list<std::initializer_list<Index>>& bounds) const {
    return detail::TensorInterface<const T, BlockRange>(
        BlockRange(this->range_, bounds), this->data());
  }

  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  detail::TensorInterface<T, BlockRange> block(
      const std::initializer_list<std::initializer_list<Index>>& bounds) {
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, bounds), this->data());
  }
  /// @}

  /// Create a permuted copy of this tensor

  /// \tparam Perm A permutation tile
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted copy of this tensor
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor permute(const Perm& perm) const {
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    [[maybe_unused]] constexpr bool is_bperm =
        detail::is_bipartite_permutation_v<Perm>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    // static_assert(is_tot || (!is_tot && !is_bperm), "Permutation type does
    // not match Tensor");
    if constexpr (!is_tot) {
      if constexpr (is_bperm) {
        TA_ASSERT(inner_size(perm) == 0);  // ensure this is a plain permutation
        return Tensor(*this, outer(perm));
      } else
        return Tensor(*this, perm);
    } else {
      // If we have a ToT we need to apply the permutation in two steps. The
      // first step is identical to the non-ToT case (permute the outer modes)
      // the second step does the inner modes
      Tensor rv(*this, outer(perm));
      if constexpr (is_bperm) {
        if (inner_size(perm) != 0) {
          auto inner_perm = inner(perm);
          Permute<value_type, value_type> p;
          for (auto& inner_t : rv) inner_t = p(inner_t, inner_perm);
        }
      }
      return rv;
    }
    abort();  // unreachable
  }

  /// Shift the lower and upper bound of this tensor

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return A reference to this tensor
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  Tensor& shift_to(const Index& bound_shift) {
    // TA_ASSERT(pimpl_);
    this->range_.inplace_shift(bound_shift);
    return *this;
  }

  /// Shift the lower and upper bound of this tensor

  /// \tparam Integer An integral type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return A reference to this tensor
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  Tensor& shift_to(const std::initializer_list<Integer>& bound_shift) {
    // TA_ASSERT(pimpl_);
    this->range_.template inplace_shift<std::initializer_list<Integer>>(
        bound_shift);
    return *this;
  }

  /// Shift the lower and upper bound of this range

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return A shifted copy of this tensor
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  Tensor shift(const Index& bound_shift) const {
    // TA_ASSERT(pimpl_);
    Tensor result = clone();
    result.shift_to(bound_shift);
    return result;
  }

  /// Shift the lower and upper bound of this range

  /// \tparam Integer An integral type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return A shifted copy of this tensor
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  Tensor shift(const std::initializer_list<Integer>& bound_shift) const {
    // TA_ASSERT(pimpl_);
    Tensor result = clone();
    result.template shift_to<std::initializer_list<Integer>>(bound_shift);
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
  Tensor binary(const Right& right, Op&& op) const {
    return Tensor(*this, right, op);
  }

  /// Use a binary, element wise operation to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Op The binary operation type
  /// \tparam Perm A permutation tile
  /// \param right The right-hand argument in the binary operation
  /// \param op The binary, element-wise operation
  /// \param perm The permutation to be applied to this tensor
  /// \return A tensor where element \c i of the new tensor is equal to
  /// \c op(*this[i],other[i])
  template <
      typename Right, typename Op, typename Perm,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor binary(const Right& right, Op&& op, const Perm& perm) const {
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    [[maybe_unused]] constexpr bool is_bperm =
        detail::is_bipartite_permutation_v<Perm>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    // static_assert(is_tot || (!is_tot && !is_bperm), "Permutation type does
    // not match Tensor");
    if constexpr (!is_tot) {
      if constexpr (is_bperm) {
        TA_ASSERT(inner_size(perm) == 0);  // ensure this is a plain permutation
        return Tensor(*this, right, op, outer(perm));
      } else
        return Tensor(*this, right, op, perm);
    } else {
      // AFAIK the other branch fundamentally relies on raw pointer arithmetic,
      // which won't work for ToTs.
      auto temp = binary(right, std::forward<Op>(op));
      Permute<Tensor, Tensor> p;
      return p(temp, perm);
    }
    abort();  // unreachable
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
  Tensor& inplace_binary(const Right& right, Op&& op) {
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
  Tensor unary(Op&& op) const {
    return Tensor(*this, op);
  }

  /// Use a unary, element wise operation to construct a new, permuted tensor

  /// \tparam Op The unary operation type
  /// \tparam Perm A permutation tile
  /// \param op The unary operation
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted tensor with elements that have been modified by \c op
  /// \throw TiledArray::Exception When this tensor is empty.
  /// \throw TiledArray::Exception The dimension of \c perm does not match
  /// that of this tensor.
  template <typename Op, typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor unary(Op&& op, const Perm& perm) const {
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    [[maybe_unused]] constexpr bool is_bperm =
        detail::is_bipartite_permutation_v<Perm>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    // static_assert(is_tot || (!is_tot && !is_bperm), "Permutation type does
    // not match Tensor");
    if constexpr (!is_tot) {
      if constexpr (is_bperm) {
        TA_ASSERT(inner_size(perm) == 0);  // ensure this is a plain permutation
        return Tensor(*this, op, outer(perm));
      } else
        return Tensor(*this, op, perm);
    } else {
      auto temp = unary(std::forward<Op>(op));
      Permute<Tensor, Tensor> p;
      return p(temp, perm);
    }
    abort();  // unreachable
  }

  /// Use a unary, element wise operation to modify this tensor

  /// \tparam Op The unary operation type
  /// \param op The unary, element-wise operation
  /// \return A reference to this object
  /// \throw TiledArray::Exception When this tensor is empty.
  template <typename Op>
  Tensor& inplace_unary(Op&& op) {
    detail::inplace_tensor_op(op, *this);
    return *this;
  }

  // Scale operation

  /// Construct a scaled copy of this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A new tensor where the elements of this tensor are scaled by
  /// \c factor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor scale(const Scalar factor) const {
    return unary(
        [factor](const numeric_type a) -> numeric_type { return a * factor; });
  }

  /// Construct a scaled and permuted copy of this tensor

  /// \tparam Scalar A scalar type
  /// \tparam Perm A permutation tile
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements of this tensor are scaled by
  /// \c factor and permuted
  template <typename Scalar, typename Perm,
            typename = std::enable_if_t<detail::is_numeric_v<Scalar> &&
                                        detail::is_permutation_v<Perm>>>
  Tensor scale(const Scalar factor, const Perm& perm) const {
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
  Tensor& scale_to(const Scalar factor) {
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
  Tensor add(const Right& right) const {
    return binary(
        right,
        [](const numeric_type l, const numeric_t<Right> r) -> numeric_type {
          return l + r;
        });
  }

  /// Add this and \c other to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Perm A permutation tile
  /// \param right The tensor that will be added to this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other
  template <
      typename Right, typename Perm,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor add(const Right& right, const Perm& perm) const {
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
  Tensor add(const Right& right, const Scalar factor) const {
    return binary(right,
                  [factor](const numeric_type l, const numeric_t<Right> r)
                      -> numeric_type { return (l + r) * factor; });
  }

  /// Scale and add this and \c other to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \tparam Perm A permutation tile
  /// \param right The tensor that will be added to this tensor
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other, scaled by \c factor
  template <typename Right, typename Scalar, typename Perm,
            typename std::enable_if<
                is_tensor<Right>::value && detail::is_numeric_v<Scalar> &&
                detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor add(const Right& right, const Scalar factor, const Perm& perm) const {
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
  Tensor add(const numeric_type value) const {
    return unary(
        [value](const numeric_type a) -> numeric_type { return a + value; });
  }

  /// Add a constant to a permuted copy of this tensor

  /// \tparam Perm A permutation tile
  /// \param value The constant to be added to this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c value
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor add(const numeric_type value, const Perm& perm) const {
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
  Tensor& add_to(const Right& right) {
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
  Tensor& add_to(const Right& right, const Scalar factor) {
    return inplace_binary(
        right, [factor](numeric_type& MADNESS_RESTRICT l,
                        const numeric_t<Right> r) { (l += r) *= factor; });
  }

  /// Add a constant to this tensor

  /// \param value The constant to be added
  /// \return A reference to this tensor
  Tensor& add_to(const numeric_type value) {
    return inplace_unary(
        [value](numeric_type& MADNESS_RESTRICT res) { res += value; });
  }

  // Subtraction operations

  /// Subtract \c right from this and return the result

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be subtracted from this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c right
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  Tensor subt(const Right& right) const {
    return binary(
        right,
        [](const numeric_type l, const numeric_t<Right> r) -> numeric_type {
          return l - r;
        });
  }

  /// Subtract \c right from this and return the result permuted by \c perm

  /// \tparam Right The right-hand tensor type
  /// \tparam Perm A permutation type
  /// \param right The tensor that will be subtracted from this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c right
  template <
      typename Right, typename Perm,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor subt(const Right& right, const Perm& perm) const {
    return binary(
        right,
        [](const numeric_type l, const numeric_t<Right> r) -> numeric_type {
          return l - r;
        },
        perm);
  }

  /// Subtract \c right from this and return the result scaled by a scaling \c
  /// factor

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
  Tensor subt(const Right& right, const Scalar factor) const {
    return binary(right,
                  [factor](const numeric_type l, const numeric_t<Right> r)
                      -> numeric_type { return (l - r) * factor; });
  }

  /// Subtract \c right from this and return the result scaled by a scaling \c
  /// factor and permuted by \c perm

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \tparam Perm A permutation type
  /// \param right The tensor that will be subtracted from this tensor
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c right, scaled by \c factor
  template <typename Right, typename Scalar, typename Perm,
            typename std::enable_if<
                is_tensor<Right>::value && detail::is_numeric_v<Scalar> &&
                detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor subt(const Right& right, const Scalar factor, const Perm& perm) const {
    return binary(
        right,
        [factor](const numeric_type l, const numeric_t<Right> r)
            -> numeric_type { return (l - r) * factor; },
        perm);
  }

  /// Subtract a constant from a copy of this tensor

  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c value
  Tensor subt(const numeric_type value) const { return add(-value); }

  /// Subtract a constant from a permuted copy of this tensor

  /// \tparam Perm A permutation tile
  /// \param value The constant to be subtracted
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the different between the
  /// elements of \c this and \c value
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor subt(const numeric_type value, const Perm& perm) const {
    return add(-value, perm);
  }

  /// Subtract \c right from this tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be subtracted from this tensor
  /// \return A reference to this tensor
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  Tensor& subt_to(const Right& right) {
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
  Tensor& subt_to(const Right& right, const Scalar factor) {
    return inplace_binary(
        right, [factor](numeric_type& MADNESS_RESTRICT l,
                        const numeric_t<Right> r) { (l -= r) *= factor; });
  }

  /// Subtract a constant from this tensor

  /// \return A reference to this tensor
  Tensor& subt_to(const numeric_type value) { return add_to(-value); }

  // Multiplication operations

  /// Multiply this by \c right to create a new tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be multiplied by this tensor
  /// \return A new tensor where the elements are the product of the elements
  /// of \c this and \c right
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  Tensor mult(const Right& right) const {
    return binary(
        right,
        [](const numeric_type l, const numeric_t<Right> r) -> numeric_type {
          return l * r;
        });
  }

  /// Multiply this by \c right to create a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Perm a permutation type
  /// \param right The tensor that will be multiplied by this tensor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the product of the elements
  /// of \c this and \c right
  template <
      typename Right, typename Perm,
      typename std::enable_if<is_tensor<Right>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor mult(const Right& right, const Perm& perm) const {
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
  Tensor mult(const Right& right, const Scalar factor) const {
    return binary(right,
                  [factor](const numeric_type l, const numeric_t<Right> r)
                      -> numeric_type { return (l * r) * factor; });
  }

  /// Scale and multiply this by \c right to create a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \tparam Perm A permutation type
  /// \param right The tensor that will be multiplied by this tensor
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor where the elements are the product of the elements
  /// of \c this and \c right, scaled by \c factor
  template <typename Right, typename Scalar, typename Perm,
            typename std::enable_if<
                is_tensor<Right>::value && detail::is_numeric_v<Scalar> &&
                detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor mult(const Right& right, const Scalar factor, const Perm& perm) const {
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
  Tensor& mult_to(const Right& right) {
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
  Tensor& mult_to(const Right& right, const Scalar factor) {
    return inplace_binary(
        right, [factor](numeric_type& MADNESS_RESTRICT l,
                        const numeric_t<Right> r) { (l *= r) *= factor; });
  }

  // Negation operations

  /// Create a negated copy of this tensor

  /// \return A new tensor that contains the negative values of this tensor
  Tensor neg() const {
    return unary([](const numeric_type r) -> numeric_type { return -r; });
  }

  /// Create a negated and permuted copy of this tensor

  /// \tparam Perm A permutation type
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor that contains the negative values of this tensor
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor neg(const Perm& perm) const {
    return unary([](const numeric_type l) -> numeric_type { return -l; }, perm);
  }

  /// Negate elements of this tensor

  /// \return A reference to this tensor
  Tensor& neg_to() {
    return inplace_unary([](numeric_type& MADNESS_RESTRICT l) { l = -l; });
  }

  /// Create a complex conjugated copy of this tensor

  /// \return A copy of this tensor that contains the complex conjugate the
  /// values
  Tensor conj() const {
    // TA_ASSERT(pimpl_);
    return scale(detail::conj_op());
  }

  /// Create a complex conjugated and scaled copy of this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A copy of this tensor that contains the scaled complex
  /// conjugate the values
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor conj(const Scalar factor) const {
    // TA_ASSERT(pimpl_);
    return scale(detail::conj_op(factor));
  }

  /// Create a complex conjugated and permuted copy of this tensor

  /// \tparam Perm A permutation type
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted copy of this tensor that contains the complex
  /// conjugate values
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor conj(const Perm& perm) const {
    // TA_ASSERT(pimpl_);
    return scale(detail::conj_op(), perm);
  }

  /// Create a complex conjugated, scaled, and permuted copy of this tensor

  /// \tparam Scalar A scalar type
  /// \tparam Perm A permutation type
  /// \param factor The scaling factor
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted copy of this tensor that contains the complex
  /// conjugate values
  template <
      typename Scalar, typename Perm,
      typename std::enable_if<detail::is_numeric_v<Scalar> &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor conj(const Scalar factor, const Perm& perm) const {
    // TA_ASSERT(pimpl_);
    return scale(detail::conj_op(factor), perm);
  }

  /// Complex conjugate this tensor

  /// \return A reference to this tensor
  Tensor& conj_to() {
    // TA_ASSERT(pimpl_);
    return scale_to(detail::conj_op());
  }

  /// Complex conjugate and scale this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor& conj_to(const Scalar factor) {
    // TA_ASSERT(pimpl_);
    return scale_to(detail::conj_op(factor));
  }

  // GEMM operations

  /// Contract this tensor with another tensor

  /// \tparam As... Template parameter pack of a tensor type
  /// \tparam V The type of \c alpha scalar
  /// \param A The tensor that will be contracted with this tensor
  /// \param alpha Multiply the result by this constant
  /// \param gemm_helper The *GEMM operation meta data
  /// \return A new tensor which is the result of contracting this tensor with
  /// \c A and scaled by \c alpha
  template <typename... As, typename V>
  Tensor gemm(const Tensor<As...>& A, const V alpha,
              const math::GemmHelper& gemm_helper) const {
    Tensor result;
    result.gemm(*this, A, alpha, gemm_helper);
    return result;
  }

  /// Contract two tensors and accumulate the scaled result to this tensor

  /// GEMM is limited to matrix like contractions. For example, the following
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
  /// \param factor The contraction result will be scaling by this value, then
  /// accumulated into \c this \param gemm_helper The *GEMM operation meta data
  /// \return A reference to \c this
  /// \note if this is uninitialized, i.e., if \c this->empty()==true will
  /// this is equivalent to
  /// \code
  ///   return (*this = left.gemm(right, factor, gemm_helper));
  /// \endcode
  template <typename... As, typename... Bs, typename W>
  Tensor& gemm(const Tensor<As...>& A, const Tensor<Bs...>& B, const W alpha,
               const math::GemmHelper& gemm_helper) {
    numeric_type beta = 1;
    if (this->empty()) {
      *this =
          Tensor(gemm_helper.make_result_range<range_type>(A.range_, B.range()),
                 A.batch_size(), default_construct{true});
      beta = 0;
    }
    TA_ASSERT(this->batch_size() == A.batch_size());
    TA_ASSERT(this->batch_size() == B.batch_size());

    // may need to split gemm into multiply + accumulate for tracing purposes
#ifdef TA_ENABLE_TILE_OPS_LOGGING
    {
      const bool twostep =
          TiledArray::TileOpsLogger<T>::get_instance().gemm &&
          TiledArray::TileOpsLogger<T>::get_instance().gemm_print_contributions;
      std::unique_ptr<T[]> data_copy;
      size_t tile_volume;
      if (twostep) {
        tile_volume = range().volume() * batch_size();
        data_copy = std::make_unique<T[]>(tile_volume);
        std::copy(data_.get(), data_.get() + tile_volume, data_copy.get());
      }
      for (size_t i = 0; i < this->batch_size(); ++i) {
        auto Ci = this->batch(i);
        TiledArray::gemm(alpha, A.batch(i), B.batch(i),
                         twostep ? numeric_type(0) : numeric_type(1), Ci,
                         gemm_helper);
      }
      if (TiledArray::TileOpsLogger<T>::get_instance_ptr() != nullptr &&
          TiledArray::TileOpsLogger<T>::get_instance().gemm) {
        auto& logger = TiledArray::TileOpsLogger<T>::get_instance();
        auto apply = [](auto& fnptr, const Range& arg) {
          return fnptr ? fnptr(arg) : arg;
        };
        auto tformed_left_range =
            apply(logger.gemm_left_range_transform, A.range());
        auto tformed_right_range =
            apply(logger.gemm_right_range_transform, B.range());
        auto tformed_result_range =
            apply(logger.gemm_result_range_transform, this->range_);
        if ((!logger.gemm_result_range_filter ||
             logger.gemm_result_range_filter(tformed_result_range)) &&
            (!logger.gemm_left_range_filter ||
             logger.gemm_left_range_filter(tformed_left_range)) &&
            (!logger.gemm_right_range_filter ||
             logger.gemm_right_range_filter(tformed_right_range))) {
          logger << "TA::Tensor::gemm+: left=" << tformed_left_range
                 << " right=" << tformed_right_range
                 << " result=" << tformed_result_range << std::endl;
          if (TiledArray::TileOpsLogger<T>::get_instance()
                  .gemm_print_contributions) {
            if (!TiledArray::TileOpsLogger<T>::get_instance()
                     .gemm_printer) {  // default printer
              // must use custom printer if result's range transformed
              if (!logger.gemm_result_range_transform)
                logger << *this << std::endl;
              else
                logger << make_map(this->data_.get(), tformed_result_range)
                       << std::endl;
            } else {
              TiledArray::TileOpsLogger<T>::get_instance().gemm_printer(
                  *logger.log, tformed_left_range, A.data(),
                  tformed_right_range, B.data(), tformed_right_range,
                  this->data(), this->batch_size());
            }
          }
        }
      }

      if (twostep) {
        for (size_t v = 0; v != tile_volume; ++v) {
          this->data_.get()[v] += data_copy[v];
        }
      }
    }
#else   // TA_ENABLE_TILE_OPS_LOGGING
    for (size_t i = 0; i < this->batch_size(); ++i) {
      auto Ci = this->batch(i);
      TiledArray::gemm(alpha, A.batch(i), B.batch(i), beta, Ci, gemm_helper);
    }
#endif  // TA_ENABLE_TILE_OPS_LOGGING

    return *this;
  }

  template <typename U, typename AU, typename V, typename AV,
            typename ElementMultiplyAddOp,
            typename = std::enable_if_t<std::is_invocable_r_v<
                void, std::remove_reference_t<ElementMultiplyAddOp>,
                value_type&, const U&, const V&>>>
  Tensor& gemm(const Tensor<U, AU>& left, const Tensor<V, AV>& right,
               const math::GemmHelper& gemm_helper,
               ElementMultiplyAddOp&& elem_muladd_op) {
    // Check that the arguments are not empty and have the correct ranks
    TA_ASSERT(!left.empty());
    TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
    TA_ASSERT(!right.empty());
    TA_ASSERT(right.range().rank() == gemm_helper.right_rank());

    // Check that the inner dimensions of left and right match
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_right_congruent(left.range().lobound_data(),
                                               right.range().lobound_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_right_congruent(left.range().upbound_data(),
                                               right.range().upbound_data()));
    TA_ASSERT(gemm_helper.left_right_congruent(left.range().extent_data(),
                                               right.range().extent_data()));

    if (this->empty()) {  // initialize, if empty
      *this = Tensor(gemm_helper.make_result_range<range_type>(left.range(),
                                                               right.range()));
    } else {
      // Check that the outer dimensions of left match the corresponding
      // dimensions in result
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.left_result_congruent(left.range().lobound_data(),
                                                  this->range_.lobound_data()));
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.left_result_congruent(left.range().upbound_data(),
                                                  this->range_.upbound_data()));
      TA_ASSERT(gemm_helper.left_result_congruent(left.range().extent_data(),
                                                  this->range_.extent_data()));

      // Check that the outer dimensions of right match the corresponding
      // dimensions in result
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.right_result_congruent(
                    right.range().lobound_data(), this->range_.lobound_data()));
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.right_result_congruent(
                    right.range().upbound_data(), this->range_.upbound_data()));
      TA_ASSERT(gemm_helper.right_result_congruent(right.range().extent_data(),
                                                   this->range_.extent_data()));
    }

    // Compute gemm dimensions
    using integer = TiledArray::math::blas::integer;
    integer M, N, K;
    gemm_helper.compute_matrix_sizes(M, N, K, left.range(), right.range());

    // Get the leading dimension for left and right matrices.
    const integer lda =
        (gemm_helper.left_op() == TiledArray::math::blas::NoTranspose ? K : M);
    const integer ldb =
        (gemm_helper.right_op() == TiledArray::math::blas::NoTranspose ? N : K);

    for (integer m = 0; m != M; ++m) {
      for (integer n = 0; n != N; ++n) {
        auto c_offset = m * N + n;
        for (integer k = 0; k != K; ++k) {
          auto a_offset =
              gemm_helper.left_op() == TiledArray::math::blas::NoTranspose
                  ? m * lda + k
                  : k * lda + m;
          auto b_offset =
              gemm_helper.right_op() == TiledArray::math::blas::NoTranspose
                  ? k * ldb + n
                  : n * ldb + k;
          elem_muladd_op(*(this->data() + c_offset), *(left.data() + a_offset),
                         *(right.data() + b_offset));
        }
      }
    }

    return *this;
  }

  // Reduction operations

  /// Generalized tensor trace

  /// This function will compute the sum of the hyper diagonal elements of
  /// tensor.
  /// \return The trace of this tensor
  /// \throw TiledArray::Exception When this tensor is empty.
  template <typename TileType = Tensor,
            typename = detail::enable_if_trace_is_defined_t<TileType>>
  decltype(auto) trace() const {
    return TiledArray::trace(*this);
  }

  /// Unary reduction operation

  /// Perform an element-wise reduction of the data by
  /// executing <tt>join_op(result, reduce_op(*this[i]))</tt> for each
  /// \c i in the index range of \c this . \c result is initialized to \c
  /// identity . If HAVE_INTEL_TBB is defined, and this is a contiguous tensor,
  /// the reduction will be executed in an undefined order, otherwise will
  /// execute in the order of increasing \c i .
  /// \tparam ReduceOp The reduction
  /// operation type
  /// \tparam JoinOp The join operation type
  /// \param reduce_op The
  /// element-wise reduction operation
  /// \param join_op The join result operation
  /// \param identity The identity value of the reduction
  /// \return The reduced value
  template <typename ReduceOp, typename JoinOp, typename Scalar>
  decltype(auto) reduce(ReduceOp&& reduce_op, JoinOp&& join_op,
                        Scalar identity) const {
    return detail::tensor_reduce(reduce_op, join_op, identity, *this);
  }

  /// Binary reduction operation

  /// Perform an element-wise binary reduction of the data of \c this and \c
  /// other by executing <tt>join_op(result, reduce_op(*this[i], other[i]))</tt>
  /// for each \c i in the index range of \c this . \c result is initialized to
  /// \c identity . If HAVE_INTEL_TBB is defined, and this is a contiguous
  /// tensor, the reduction will be executed in an undefined order, otherwise
  /// will execute in the order of increasing \c i .
  /// \tparam Right The
  /// right-hand argument tensor type
  /// \tparam ReduceOp The reduction operation
  /// type
  /// \tparam JoinOp The join operation type
  /// \param other The right-hand
  /// argument of the binary reduction
  /// \param reduce_op The element-wise
  /// reduction operation \param join_op The join result operation
  /// \param identity The identity value of the reduction
  /// \return The reduced value
  template <typename Right, typename ReduceOp, typename JoinOp, typename Scalar,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  decltype(auto) reduce(const Right& other, ReduceOp&& reduce_op,
                        JoinOp&& join_op, Scalar identity) const {
    return detail::tensor_reduce(reduce_op, join_op, identity, *this, other);
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

  /// \tparam ResultType return type
  /// \note This evaluates \c std::sqrt(ResultType(this->squared_norm()))
  /// \return The vector norm of this tensor
  template <typename ResultType = scalar_type>
  ResultType norm() const {
    return std::sqrt(static_cast<ResultType>(squared_norm()));
  }

  /// Minimum element

  /// \return The minimum elements of this tensor
  template <typename Numeric = numeric_type>
  numeric_type min(
      typename std::enable_if<
          detail::is_strictly_ordered<Numeric>::value>::type* = nullptr) const {
    auto min_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type arg) { res = std::min(res, arg); };
    return reduce(min_op, min_op, std::numeric_limits<numeric_type>::max());
  }

  /// Maximum element

  /// \return The maximum elements of this tensor
  template <typename Numeric = numeric_type>
  numeric_type max(
      typename std::enable_if<
          detail::is_strictly_ordered<Numeric>::value>::type* = nullptr) const {
    auto max_op = [](numeric_type& MADNESS_RESTRICT res,
                     const numeric_type arg) { res = std::max(res, arg); };
    return reduce(max_op, max_op, std::numeric_limits<scalar_type>::min());
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

  /// Vector dot (not inner!) product

  /// \tparam Right The right-hand tensor type
  /// \param other The right-hand tensor to be reduced
  /// \return The dot product of the this and \c other
  /// If numeric_type is real, this is equivalent to inner product
  /// \sa Tensor::inner_product
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

};  // class Tensor

/// Contract two tensors and accumulate the scaled result to this tensor

/// GEMM is limited to matrix like contractions. For example, the following
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
/// \param factor The contraction result will be scaling by this value, then
/// accumulated into \c this \param gemm_helper The *GEMM operation meta data
/// \return A reference to \c this
/// \note if this is uninitialized, i.e., if \c this->empty()==true will
/// this is equivalent to
/// \code
///   return (*this = left.gemm(right, factor, gemm_helper));
/// \endcode
template <typename Alpha, typename... As, typename... Bs, typename Beta,
          typename... Cs>
void gemm(Alpha alpha, const Tensor<As...>& A, const Tensor<Bs...>& B,
          Beta beta, Tensor<Cs...>& C, const math::GemmHelper& gemm_helper) {
  static_assert(
      !detail::is_tensor_of_tensor_v<Tensor<As...>, Tensor<Bs...>,
                                     Tensor<Cs...>>,
      "TA::Tensor<T>::gemm without custom element op is only applicable to "
      "plain tensors");
  {
    // Check that tensor C is not empty and has the correct rank
    TA_ASSERT(!C.empty());
    TA_ASSERT(C.range().rank() == gemm_helper.result_rank());

    // Check that the arguments are not empty and have the correct ranks
    TA_ASSERT(!A.empty());
    TA_ASSERT(A.range().rank() == gemm_helper.left_rank());
    TA_ASSERT(!B.empty());
    TA_ASSERT(B.range().rank() == gemm_helper.right_rank());

    TA_ASSERT(A.batch_size() == 1);
    TA_ASSERT(B.batch_size() == 1);
    TA_ASSERT(C.batch_size() == 1);

    // Check that the outer dimensions of left match the corresponding
    // dimensions in result
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_result_congruent(A.range().lobound_data(),
                                                C.range().lobound_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_result_congruent(A.range().upbound_data(),
                                                C.range().upbound_data()));
    TA_ASSERT(gemm_helper.left_result_congruent(A.range().extent_data(),
                                                C.range().extent_data()));

    // Check that the outer dimensions of right match the corresponding
    // dimensions in result
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.right_result_congruent(B.range().lobound_data(),
                                                 C.range().lobound_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.right_result_congruent(B.range().upbound_data(),
                                                 C.range().upbound_data()));
    TA_ASSERT(gemm_helper.right_result_congruent(B.range().extent_data(),
                                                 C.range().extent_data()));

    // Check that the inner dimensions of left and right match
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_right_congruent(A.range().lobound_data(),
                                               B.range().lobound_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_right_congruent(A.range().upbound_data(),
                                               B.range().upbound_data()));
    TA_ASSERT(gemm_helper.left_right_congruent(A.range().extent_data(),
                                               B.range().extent_data()));

    // Compute gemm dimensions
    using integer = TiledArray::math::blas::integer;
    integer m, n, k;
    gemm_helper.compute_matrix_sizes(m, n, k, A.range(), B.range());

    // Get the leading dimension for left and right matrices.
    const integer lda =
        (gemm_helper.left_op() == TiledArray::math::blas::NoTranspose ? k : m);
    const integer ldb =
        (gemm_helper.right_op() == TiledArray::math::blas::NoTranspose ? n : k);

    // may need to split gemm into multiply + accumulate for tracing purposes
#ifdef TA_ENABLE_TILE_OPS_LOGGING
    {
      using numeric_type = typename Tensor<Cs...>::numeric_type;
      using T = numeric_type;
      const bool twostep =
          TiledArray::TileOpsLogger<T>::get_instance().gemm &&
          TiledArray::TileOpsLogger<T>::get_instance().gemm_print_contributions;
      std::unique_ptr<T[]> data_copy;
      size_t tile_volume;
      if (twostep) {
        tile_volume = C.range().volume();
        data_copy = std::make_unique<T[]>(tile_volume);
        std::copy(C.data(), C.data() + tile_volume, data_copy.get());
      }
      non_distributed::gemm(gemm_helper.left_op(), gemm_helper.right_op(), m, n,
                            k, alpha, A.data(), lda, B.data(), ldb,
                            twostep ? numeric_type(0) : beta, C.data(), n);

      if (TiledArray::TileOpsLogger<T>::get_instance_ptr() != nullptr &&
          TiledArray::TileOpsLogger<T>::get_instance().gemm) {
        auto& logger = TiledArray::TileOpsLogger<T>::get_instance();
        auto apply = [](auto& fnptr, const Range& arg) {
          return fnptr ? fnptr(arg) : arg;
        };
        auto tformed_left_range =
            apply(logger.gemm_left_range_transform, A.range());
        auto tformed_right_range =
            apply(logger.gemm_right_range_transform, B.range());
        auto tformed_result_range =
            apply(logger.gemm_result_range_transform, C.range());
        if ((!logger.gemm_result_range_filter ||
             logger.gemm_result_range_filter(tformed_result_range)) &&
            (!logger.gemm_left_range_filter ||
             logger.gemm_left_range_filter(tformed_left_range)) &&
            (!logger.gemm_right_range_filter ||
             logger.gemm_right_range_filter(tformed_right_range))) {
          logger << "TA::Tensor::gemm+: left=" << tformed_left_range
                 << " right=" << tformed_right_range
                 << " result=" << tformed_result_range << std::endl;
          if (TiledArray::TileOpsLogger<T>::get_instance()
                  .gemm_print_contributions) {
            if (!TiledArray::TileOpsLogger<T>::get_instance()
                     .gemm_printer) {  // default printer
              // must use custom printer if result's range transformed
              if (!logger.gemm_result_range_transform)
                logger << C << std::endl;
              else
                logger << make_map(C.data(), tformed_result_range) << std::endl;
            } else {
              TiledArray::TileOpsLogger<T>::get_instance().gemm_printer(
                  *logger.log, tformed_left_range, A.data(),
                  tformed_right_range, B.data(), tformed_right_range, C.data(),
                  C.batch_size());
            }
          }
        }
      }

      if (twostep) {
        for (size_t v = 0; v != tile_volume; ++v) {
          C.data()[v] += data_copy[v];
        }
      }
    }
#else   // TA_ENABLE_TILE_OPS_LOGGING
    math::blas::gemm(gemm_helper.left_op(), gemm_helper.right_op(), m, n, k,
                     alpha, A.data(), lda, B.data(), ldb, beta, C.data(), n);
#endif  // TA_ENABLE_TILE_OPS_LOGGING
  }
}

// template <typename T, typename A>
// const typename Tensor<T, A>::range_type Tensor<T, A>::empty_range_;

template <typename T, typename A>
bool operator==(const Tensor<T, A>& a, const Tensor<T, A>& b) {
  return a.range() == b.range() &&
         std::equal(a.data(), a.data() + a.size(), b.data());
}
template <typename T, typename A>
bool operator!=(const Tensor<T, A>& a, const Tensor<T, A>& b) {
  return !(a == b);
}

namespace detail {

/// Implements taking the trace of a Tensor<T> (\c T is a numeric type)
///
/// \tparam T The type of the elements in the tensor. For this specialization
///           to be considered must satisfy the concept of numeric type.
/// \tparam A The type of the allocator for the tensor
template <typename T, typename A>
struct Trace<Tensor<T, A>, detail::enable_if_numeric_t<T>> {
  decltype(auto) operator()(const Tensor<T>& t) const {
    using size_type = typename Tensor<T>::size_type;
    using value_type = typename Tensor<T>::value_type;
    const auto range = t.range();

    // Get pointers to the range data
    const size_type n = range.rank();
    const auto* MADNESS_RESTRICT const lower = range.lobound_data();
    const auto* MADNESS_RESTRICT const upper = range.upbound_data();
    const auto* MADNESS_RESTRICT const stride = range.stride_data();

    // Search for the largest lower bound and the smallest upper bound
    const size_type lower_max = *std::max_element(lower, lower + n);
    const size_type upper_min = *std::min_element(upper, upper + n);

    value_type result = 0;

    if (lower_max >= upper_min) return result;  // No diagonal element in tile

    // Compute the first and last ordinal index
    size_type first = 0ul, last = 0ul, trace_stride = 0ul;
    for (size_type i = 0ul; i < n; ++i) {
      const size_type lower_i = lower[i];
      const size_type stride_i = stride[i];

      first += (lower_max - lower_i) * stride_i;
      last += (upper_min - lower_i) * stride_i;
      trace_stride += stride_i;
    }

    // Compute the trace
    const value_type* MADNESS_RESTRICT const data = &t[first];
    for (; first < last; first += trace_stride) result += data[first];

    return result;
  }
};

/// specialization of TiledArray::detail::transform for Tensor
template <typename T, typename A>
struct transform<Tensor<T, A>> {
  template <typename Op, typename T1>
  Tensor<T, A> operator()(Op&& op, T1&& t1) const {
    return Tensor<T, A>(std::forward<T1>(t1), std::forward<Op>(op));
  }
  template <typename Op, typename Perm, typename T1,
            typename = std::enable_if_t<
                detail::is_permutation_v<std::remove_reference_t<Perm>>>>
  Tensor<T, A> operator()(Op&& op, Perm&& perm, T1&& t1) const {
    return Tensor<T, A>(std::forward<T1>(t1), std::forward<Op>(op),
                        std::forward<Perm>(perm));
  }
  template <typename Op, typename T1, typename T2>
  Tensor<T, A> operator()(Op&& op, T1&& t1, T2&& t2) const {
    return Tensor<T, A>(std::forward<T1>(t1), std::forward<T2>(t2),
                        std::forward<Op>(op));
  }
  template <typename Op, typename Perm, typename T1, typename T2,
            typename = std::enable_if_t<
                detail::is_permutation_v<std::remove_reference_t<Perm>>>>
  Tensor<T, A> operator()(Op&& op, Perm&& perm, T1&& t1, T2&& t2) const {
    return Tensor<T, A>(std::forward<T1>(t1), std::forward<T2>(t2),
                        std::forward<Op>(op), std::forward<Perm>(perm));
  }
};
}  // namespace detail

#ifndef TILEDARRAY_HEADER_ONLY

extern template class Tensor<double>;
extern template class Tensor<float>;
// extern template class Tensor<int>;
// extern template class Tensor<long>;
extern template class Tensor<std::complex<double>>;
extern template class Tensor<std::complex<float>>;

#endif  // TILEDARRAY_HEADER_ONLY

}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_TENSOR_H__INCLUDED
