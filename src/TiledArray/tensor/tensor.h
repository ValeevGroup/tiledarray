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

#include "TiledArray/config.h"

#include "TiledArray/external/umpire.h"
#include "TiledArray/host/env.h"

#include "TiledArray/math/blas.h"
#include "TiledArray/math/gemm_helper.h"
#include "TiledArray/tensor/complex.h"
#include "TiledArray/tensor/kernels.h"
#include "TiledArray/tile_interface/clone.h"
#include "TiledArray/tile_interface/permute.h"
#include "TiledArray/tile_interface/trace.h"
#include "TiledArray/util/logger.h"
#include "TiledArray/util/ptr_registry.h"

namespace TiledArray {

namespace detail {

/// Signals that we can take the trace of a Tensor<T, A> (for numeric \c T)
template <typename T, typename A>
struct TraceIsDefined<Tensor<T, A>, enable_if_numeric_t<T>> : std::true_type {};

template <typename To, typename From,
          typename = std::enable_if_t<
              detail::is_nested_tensor_v<To, detail::remove_cvr_t<From>>>>
To clone_or_cast(From&& f) {
  if constexpr (std::is_same_v<To, detail::remove_cvr_t<From>>)
    return std::forward<From>(f).clone();
  else if constexpr (detail::is_convertible_v<From, To>) {
    return static_cast<To>(std::forward<From>(f));
  } else if constexpr (detail::is_range_v<To> &&
                       detail::is_range_v<detail::remove_cvr_t<From>>) {
    using std::begin;
    using std::data;
    using std::end;

    To t(f.range());
    if constexpr (detail::is_contiguous_tensor_v<detail::remove_cvr_t<From>>) {
      const auto n = f.range().volume();
      if constexpr (detail::is_contiguous_tensor_v<To>) {
        std::copy(data(f), data(f) + n, data(t));
      } else {
        std::copy(data(f), data(f) + n, begin(t));
      }
    } else {
      if constexpr (detail::is_contiguous_tensor_v<To>) {
        std::copy(begin(f), end(f), data(t));
      } else
        std::copy(begin(f), end(f), begin(t));
    }
    return t;
  } else {
    static_assert(
        !std::is_void_v<To>,
        "clone_or_cast<To,From>: could not figure out how to convert From to "
        "To, either overload of a member function of Tensor is missing or From "
        "need to provide a conversion operator to To");
  }
}

}  // namespace detail

/// An N-dimensional tensor object

/// A contiguous row-major tensor with __shallow-copy__ semantics.
/// As of TiledArray 1.1 Tensor represents a batch of tensors with same Range
/// (the default batch size = 1).
/// \tparam T The value type of this tensor
/// \tparam A The allocator type for the data; only default-constructible
/// allocators are supported to save space
template <typename T, typename Allocator>
class Tensor {
  // meaningful error if T& is not assignable, see
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=48101
  static_assert(std::is_assignable<std::add_lvalue_reference_t<T>, T>::value,
                "Tensor<T,Allocator>: T must be an assignable type (e.g. "
                "cannot be const)");
  // default-constructible Allocator allows to reduce the size of default Tensor
  // and minimize the overhead of null elements in Tensor<Tensor<T>>
  static_assert(
      std::is_default_constructible_v<Allocator>,
      "Tensor<T,Allocator>: only default-constructible Allocator is supported");

#ifdef TA_TENSOR_MEM_TRACE
  template <typename... Ts>
  std::string make_string(Ts&&... ts) {
    std::ostringstream oss;
    (oss << ... << ts);
    return oss.str();
  }
#endif

 public:
  typedef Range range_type;                              ///< Tensor range type
  typedef typename range_type::index1_type index1_type;  ///< 1-index type
  typedef typename range_type::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename range_type::ordinal_type
      size_type;  ///< Size type (to meet the container concept)
  typedef Allocator allocator_type;  ///< Allocator type
  typedef typename std::allocator_traits<allocator_type>::value_type
      value_type;  ///< Array element type
  typedef std::add_lvalue_reference_t<value_type>
      reference;  ///< Element (lvalue) reference type
  typedef std::add_lvalue_reference_t<std::add_const_t<value_type>>
      const_reference;  ///< Element (const lvalue) reference type
  typedef typename std::allocator_traits<allocator_type>::pointer
      pointer;  ///< Element pointer type
  typedef typename std::allocator_traits<allocator_type>::const_pointer
      const_pointer;  ///< Element const pointer type
  typedef typename std::allocator_traits<allocator_type>::difference_type
      difference_type;                   ///< Difference type
  typedef pointer iterator;              ///< Element iterator type
  typedef const_pointer const_iterator;  ///< Element const iterator type
  typedef typename TiledArray::detail::numeric_type<T>::type
      numeric_type;  ///< the numeric type that supports T
  typedef typename TiledArray::detail::scalar_type<T>::type
      scalar_type;  ///< the scalar type that supports T

 private:
  template <typename X>
  using value_t = typename X::value_type;
  template <typename X>
  using numeric_t = typename TiledArray::detail::numeric_type<X>::type;

  template <typename... Ts>
  struct is_tensor {
    static constexpr bool value = detail::is_tensor<Ts...>::value ||
                                  detail::is_tensor_of_tensor<Ts...>::value;
  };

 public:
  /// compute type of Tensor with different element type
  template <typename U,
            typename OtherAllocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<U>>
  using rebind_t = Tensor<U, OtherAllocator>;

  template <typename U, typename V = value_type, typename = void>
  struct rebind_numeric;
  template <typename U, typename V>
  struct rebind_numeric<U, V, std::enable_if_t<is_tensor<V>::value>> {
    using VU = typename V::template rebind_numeric<U>::type;
    using type = Tensor<VU, typename std::allocator_traits<
                                Allocator>::template rebind_alloc<VU>>;
  };
  template <typename U, typename V>
  struct rebind_numeric<U, V, std::enable_if_t<!is_tensor<V>::value>> {
    using type = Tensor<
        U, typename std::allocator_traits<Allocator>::template rebind_alloc<U>>;
  };

  /// compute type of Tensor with different numeric type
  template <typename U>
  using rebind_numeric_t = typename rebind_numeric<U, value_type>::type;

 private:
  using default_construct = bool;

  Tensor(const range_type& range, size_t nbatch, bool default_construct)
      : range_(range), nbatch_(nbatch) {
    size_t size = range_.volume() * nbatch;
    allocator_type allocator;
    auto* ptr = allocator.allocate(size);
    // default construct elements of data only if can have any effect ...
    if constexpr (!std::is_trivially_default_constructible_v<T>) {
      // .. and requested
      if (default_construct) {
        std::uninitialized_default_construct_n(ptr, size);
      }
    }
    auto deleter = [
#ifdef TA_TENSOR_MEM_TRACE
                       this,
#endif
                       allocator = std::move(allocator),
                       size](auto&& ptr) mutable {
      std::destroy_n(ptr, size);
      // N.B. deregister ptr *before* deallocating to avoid possible race
      // between reallocation and deregistering
#ifdef TA_TENSOR_MEM_TRACE
      const auto nbytes = size * sizeof(T);
      if (nbytes >= trace_if_larger_than_) {
        ptr_registry()->erase(ptr, nbytes,
                              make_string("created by TA::Tensor*=", this));
      }
#endif
      allocator.deallocate(ptr, size);
    };
    this->data_ = std::shared_ptr<value_type[]>(ptr, std::move(deleter));
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor::data_.get()=", data_.get()));
      ptr_registry()->insert(data_.get(), nbytes(),
                             make_string("created by TA::Tensor*=", this));
    }
#endif
  }

  Tensor(range_type&& range, size_t nbatch, bool default_construct)
      : range_(std::move(range)), nbatch_(nbatch) {
    size_t size = range_.volume() * nbatch;
    allocator_type allocator;
    auto* ptr = allocator.allocate(size);
    // default construct elements of data only if can have any effect ...
    if constexpr (!std::is_trivially_default_constructible_v<T>) {
      // .. and requested
      if (default_construct) {
        std::uninitialized_default_construct_n(ptr, size);
      }
    }
    auto deleter = [
#ifdef TA_TENSOR_MEM_TRACE
                       this,
#endif
                       allocator = std::move(allocator),
                       size](auto&& ptr) mutable {
      std::destroy_n(ptr, size);
      // N.B. deregister ptr *before* deallocating to avoid possible race
      // between reallocation and deregistering
#ifdef TA_TENSOR_MEM_TRACE
      const auto nbytes = size * sizeof(T);
      if (nbytes >= trace_if_larger_than_) {
        ptr_registry()->erase(ptr, nbytes,
                              make_string("created by TA::Tensor*=", this));
      }
#endif
      allocator.deallocate(ptr, size);
    };
    this->data_ = std::shared_ptr<value_type[]>(ptr, std::move(deleter));
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor::data_.get()=", data_.get()));
      ptr_registry()->insert(data_.get(), nbytes(),
                             make_string("created by TA::Tensor*=", this));
    }
#endif
  }

  template <typename T_>
  static decltype(auto) value_converter(const T_& arg) {
    using arg_type = detail::remove_cvr_t<decltype(arg)>;
    if constexpr (detail::is_tensor_v<arg_type>)  // clone nested tensors
      return arg.clone();
    else if constexpr (!std::is_same_v<arg_type, value_type>) {  // convert
      if constexpr (std::is_convertible_v<arg_type, value_type>)
        return static_cast<value_type>(arg);
      else
        return conversions::to<value_type, arg_type>()(arg);
    } else
      return arg;
  };

  range_type range_;  ///< Range
  /// Number of `range_`-sized blocks in `data_`
  /// \note this is not used for (in)equality comparison
  size_t nbatch_ = 1;
  std::shared_ptr<value_type[]> data_;  ///< Shared pointer to the data

 public:
  /// constructs an empty (null) Tensor
  /// \post `this->empty()`
  Tensor() = default;

  /// copy constructor

  /// \param[in] other an object to copy data from
  /// \post `*this` is a shallow copy of \p other ,
  /// i.e. `*this == other && this->data()==other.data()`
  Tensor(const Tensor& other)
      : range_(other.range_), nbatch_(other.nbatch_), data_(other.data_) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor(const Tensor& other)::data_.get()=",
                            data_.get()));
    }
#endif
  }

  /// move constructor

  /// \param[in,out] other an object to move data from;
  ///                      on return \p other is in empty (null) but not
  ///                      necessarily default state
  /// \post `other.empty()`
  Tensor(Tensor&& other)
      : range_(std::move(other.range_)),
        nbatch_(std::move(other.nbatch_)),
        data_(std::move(other.data_)) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->erase(
          &other,
          make_string("TA::Tensor(Tensor&& other)::data_.get()=", data_.get()));
      ptr_registry()->insert(
          this,
          make_string("TA::Tensor(Tensor&& other)::data_.get()=", data_.get()));
    }
#endif
  }

  ~Tensor() {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->erase(
          this, make_string("TA::~Tensor()::data_.get()=", data_.get()));
    }
#endif
  }

  struct nbatches {
    template <typename Int,
              typename = std::enable_if_t<std::is_integral_v<Int>>>
    nbatches(Int n) : n(n) {}
    template <typename Int,
              typename = std::enable_if_t<std::is_integral_v<Int>>>
    nbatches& operator=(Int n) {
      this->n = n;
    }

    size_type n = 1;
  };

  /// Construct a tensor with a range equal to \c range. The data is
  /// default-initialized (which, for `T` with trivial default constructor,
  /// means data is uninitialized).
  /// \param range The range of the tensor
  /// \param nbatch The number of batches (default is 1)
  explicit Tensor(const range_type& range, nbatches nb = 1)
      : Tensor(range, nb.n, default_construct{true}) {}

  /// Construct a tensor of tensor values, setting all elements to the same
  /// value

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

  /// Construct a tensor of scalars, setting all elements to the same value

  /// \param range An array with the size of of each dimension
  /// \param value The value of the tensor elements
  template <typename Value,
            typename std::enable_if<std::is_convertible_v<Value, value_type> &&
                                    !detail::is_tensor<Value>::value>::type* =
                nullptr>
  Tensor(const range_type& range, const Value& value)
      : Tensor(range, 1, default_construct{false}) {
    detail::tensor_init([value]() -> Value { return value; }, *this);
  }

  /// Construct a tensor with a fill op that takes an element index

  /// \tparam ElementIndexOp callable of signature
  /// `value_type(const Range::index_type&)`
  /// \param range An array with the size of of each dimension
  /// \param element_idx_op a callable of type ElementIndexOp
  template <typename ElementIndexOp,
            typename = std::enable_if_t<std::is_invocable_r_v<
                value_type, ElementIndexOp, const Range::index_type&>>>
  Tensor(const range_type& range, const ElementIndexOp& element_idx_op)
      : Tensor(range, 1, default_construct{false}) {
    pointer MADNESS_RESTRICT const data = this->data();
    for (auto&& element_idx : range) {
      const auto ord = range.ordinal(element_idx);
      new (data + ord) value_type(element_idx_op(element_idx));
    }
  }

  /// Construct an evaluated tensor
  template <typename InIter,
            typename std::enable_if<
                TiledArray::detail::is_input_iterator<InIter>::value &&
                !std::is_pointer<InIter>::value>::type* = nullptr>
  Tensor(const range_type& range, InIter it)
      : Tensor(range, 1, default_construct{false}) {
    auto n = range.volume();
    pointer MADNESS_RESTRICT data = this->data();
    for (size_type i = 0ul; i < n; ++i, ++it, ++data)
      new (data) value_type(*it);
  }

  template <typename U>
  Tensor(const Range& range, const U* u)
      : Tensor(range, 1, default_construct{false}) {
    math::uninitialized_copy_vector(range.volume(), u, this->data());
  }

  explicit Tensor(const Range& range, std::initializer_list<T> il)
      : Tensor(range, il.begin()) {}

  /// Construct a copy of a tensor interface object

  /// \tparam T1 A tensor type
  /// \param other The tensor to be copied
  /// \note this constructor is disabled if \p T1 already has a conversion
  ///       operator to this type
  /// \warning if `T1` is a tensor of tensors its elements are _cloned_ rather
  ///          than copied to make the semantics of  this to be consistent
  ///          between tensors of scalars and tensors of scalars; specifically,
  ///          if `T1` is a tensor of scalars the constructed tensor is
  ///          is independent of \p other, thus should apply clone to inner
  ///          tensor nests to behave similarly for nested tensors
  template <
      typename T1,
      typename std::enable_if<
          is_tensor<T1>::value && !std::is_same<T1, Tensor>::value &&
          !detail::has_conversion_operator_v<T1, Tensor>>::type* = nullptr>
  explicit Tensor(const T1& other)
      : Tensor(detail::clone_range(other), 1, default_construct{false}) {
    detail::tensor_init(value_converter<typename T1::value_type>, *this, other);
  }

  /// Construct a permuted tensor copy

  /// \tparam T1 A tensor type
  /// \tparam Perm A permutation type
  /// \param other The tensor to be copied
  /// \param perm The permutation that will be applied to the copy
  /// \warning if `T1` is a tensor of tensors its elements are _cloned_ rather
  ///          than copied to make the semantics of  this to be consistent
  ///          between tensors of scalars and tensors of tensors; specifically,
  ///          if `T1` is a tensor of scalars the constructed tensor is
  ///          is independent of \p other, thus should apply clone to inner
  ///          tensor nests to behave similarly for nested tensors
  template <
      typename T1, typename Perm,
      typename std::enable_if<detail::is_nested_tensor_v<T1> &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor(const T1& other, const Perm& perm)
      : Tensor(outer(perm) * other.range(), other.nbatch(),
               default_construct{false}) {
    const auto outer_perm = outer(perm);
    if (outer_perm) {
      detail::tensor_init(value_converter<typename T1::value_type>, outer_perm,
                          *this, other);
    } else {
      detail::tensor_init(value_converter<typename T1::value_type>, *this,
                          other);
    }

    // If we actually have a ToT the inner permutation was not applied above so
    // we do that now
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    if constexpr (is_tot && is_bperm) {
      if (inner_size(perm) != 0) {
        const auto inner_perm = inner(perm);
        Permute<value_type, value_type> p;

        auto volume = total_size();
        for (decltype(volume) i = 0; i < volume; ++i) {
          auto& el = *(data() + i);
          if (!el.empty()) el = p(el, inner_perm);
        }
      }
    }
  }

  /// "Element-wise" unary transform of \c other

  /// \tparam T1 A tensor type
  /// \tparam Op A unary callable
  /// \param other The tensor argument
  /// \param op Unary operation that can be invoked on elements of \p other ;
  ///        if it is not, it will be "threaded" over \p other via `tensor_op`
  template <typename T1, typename Op,
            typename std::enable_if_t<
                is_tensor<T1>::value &&
                !detail::is_permutation_v<std::decay_t<Op>>>* = nullptr>
  Tensor(const T1& other, Op&& op)
      : Tensor(detail::clone_range(other), 1, default_construct{false}) {
    detail::tensor_init(op, *this, other);
  }

  /// "Element-wise" unary transform of \c other fused with permutation

  /// equivalent, but more efficient, than `Tensor(other, op).permute(perm)`
  /// \tparam T1 A tensor type
  /// \tparam Op A unary callable
  /// \tparam Perm A permutation type
  /// \param other The tensor argument
  /// \param op Unary operation that can be invoked as` op(other[i]))`;
  ///        if it is not, it will be "threaded" over \p other via `tensor_op`
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
    // tile ops pass bipartite permutations here even if this is a plain tensor
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    if constexpr (is_tot && is_bperm) {
      if (inner_size(perm) != 0) {
        auto inner_perm = inner(perm);
        Permute<value_type, value_type> p;
        for (auto& x : *this) x = p(x, inner_perm);
      }
    }
  }

  /// "Element-wise" binary transform of \c {left,right}

  /// \tparam T1 A tensor type
  /// \tparam T2 A tensor type
  /// \tparam Op A binary callable
  /// \param left The left-hand tensor argument
  /// \param right The right-hand tensor argument
  /// \param op Binary operation that can be invoked as `op(left[i],right[i]))`;
  ///        if it is not, it will be "threaded" over \p other via `tensor_op`
  template <typename T1, typename T2, typename Op,
            typename = std::enable_if_t<detail::is_nested_tensor_v<T1, T2>>>
  Tensor(const T1& left, const T2& right, Op&& op)
      : Tensor(detail::clone_range(left), 1, default_construct{false}) {
    detail::tensor_init(op, *this, left, right);
  }

  /// "Element-wise" binary transform of \c {left,right} fused with permutation

  /// \tparam T1 A tensor type
  /// \tparam T2 A tensor type
  /// \tparam Op A binary callable
  /// \tparam Perm A permutation tile
  /// \param left The left-hand tensor argument
  /// \param right The right-hand tensor argument
  /// \param op Binary operation that can be invoked as `op(left[i],right[i]))`;
  ///        if it is not, it will be "threaded" over \p other via `tensor_op`
  /// \param perm The permutation that will be applied to the arguments
  template <
      typename T1, typename T2, typename Op, typename Perm,
      typename std::enable_if<detail::is_nested_tensor<T1, T2>::value &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  Tensor(const T1& left, const T2& right, Op&& op, const Perm& perm)
      : Tensor(outer(perm) * left.range(), 1, default_construct{false}) {
    detail::tensor_init(op, outer(perm), *this, left, right);
    // If we actually have a ToT the inner permutation was not applied above so
    // we do that now
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    constexpr bool is_bperm = detail::is_bipartite_permutation_v<Perm>;
    if constexpr (is_tot && is_bperm) {
      if (inner_size(perm) != 0) {
        auto inner_perm = inner(perm);
        Permute<value_type, value_type> p;
        for (auto& x : *this) x = p(x, inner_perm);
      }
    }
  }

  /// Construct a tensor with a range equal to \c range using existing data
  /// \param range The range of the tensor
  /// \param nbatch The number of batches
  /// \param data shared pointer to the data
  Tensor(const range_type& range, size_t nbatch,
         std::shared_ptr<value_type[]> data)
      : range_(range), nbatch_(nbatch), data_(std::move(data)) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor(range, nbatch, data)::data_.get()=",
                            data_.get()));
    }
#endif
  }

  /// Construct a tensor with a range equal to \c range using existing data
  /// assuming unit batch size \param range The range of the tensor \param data
  /// shared pointer to the data
  Tensor(const range_type& range, std::shared_ptr<value_type[]> data)
      : range_(range), nbatch_(1), data_(std::move(data)) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this,
          make_string("TA::Tensor(range, data)::data_.get()=", data_.get()));
    }
#endif
  }

  /// The batch size accessor

  /// @return the size of tensor batch represented by `*this`
  size_t nbatch() const { return this->nbatch_; }

  /// @param[in] idx the batch index
  /// @pre `idx < this->nbatch()`
  /// @return (plain, i.e. nbatch=1) Tensor representing element \p idx of
  /// the batch
  Tensor batch(size_t idx) const {
    TA_ASSERT(idx < this->nbatch());
    std::shared_ptr<value_type[]> data(this->data_,
                                       this->data_.get() + idx * this->size());
    return Tensor(this->range(), 1, data);
  }

  /// Returns Tensor representing the data using another range and batch size

  /// @param[in] range the Range of the result
  /// @param[in] nbatch the number of batches of the result
  /// @return Tensor object representing `this->data()` using @p range and @p
  /// nbatch
  auto reshape(const range_type& range, size_t nbatch = 1) const {
    TA_ASSERT(this->range().volume() * this->nbatch() ==
              range.volume() * nbatch);
    return Tensor(range, nbatch, this->data_);
  }

  /// @return a deep copy of `*this`
  Tensor clone() const& {
    Tensor result;
    if (data_) {
      if constexpr (detail::is_tensor_of_tensor_v<Tensor>) {
        result = Tensor(*this, [](value_type const& el) { return el.clone(); });
      } else {
        result = detail::tensor_op<Tensor>(
            [](const numeric_type value) -> numeric_type { return value; },
            *this);
      }
    } else if (range_) {  // corner case: data_ = null implies range_.volume()
                          // == 0;
      TA_ASSERT(range_.volume() == 0);
      result = Tensor(range_);
    }
    return result;
  }

  /// cloning an rvalue ref forwards the contents of this
  /// @return a deep copy of `*this`
  /// @post this is in a moved-from state
  Tensor clone() && { return std::move(*this); }

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

  /// copy assignment operator

  /// \param[in] other an object to copy data from
  /// \post `*this` is a shallow copy of \p other ,
  /// i.e. `*this == other && this->data()==other.data()`
  Tensor& operator=(const Tensor& other) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->erase(
          this,
          make_string("TA::Tensor::operator=(const Tensor&)::data_.get()=",
                      data_.get()));
    }
#endif
    range_ = other.range_;
    nbatch_ = other.nbatch_;
    data_ = other.data_;
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this,
          make_string("TA::Tensor::operator=(const Tensor&)::data_.get()=",
                      data_.get()));
    }
#endif
    return *this;
  }

  /// move assignment operator

  /// \param[in,out] other an object to move data from;
  ///                      on return \p other is in empty (null) but not
  ///                      necessarily default state
  /// \post `other.empty()`
  Tensor& operator=(Tensor&& other) {
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->erase(
          this, make_string("TA::Tensor::operator=(Tensor&&)::data_.get()=",
                            data_.get()));
    }
    if (other.nbytes() >= trace_if_larger_than_) {
      ptr_registry()->erase(
          &other, make_string("TA::Tensor::operator=(Tensor&&)::data_.get()=",
                              data_.get()));
    }
#endif
    range_ = std::move(other.range_);
    nbatch_ = std::move(other.nbatch_);
    data_ = std::move(other.data_);
#ifdef TA_TENSOR_MEM_TRACE
    if (nbytes() >= trace_if_larger_than_) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor::operator=(Tensor&&)::data_.get()=",
                            data_.get()));
    }
#endif
    return *this;
  }

  /// Tensor range object accessor

  /// \return The tensor range object
  const range_type& range() const { return range_; }

  /// Tensor dimension size accessor

  /// \return The number of elements in the tensor
  ordinal_type size() const { return (this->range().volume()); }

  /// \return The number of elements in the tensor by summing up the sizes of
  /// the batches.
  ordinal_type total_size() const { return size() * nbatch(); }

  /// Tensor data size (in bytes) accessor

  /// \return The number of bytes occupied by this tensor's data
  /// \warning this only returns valid value if this is a tensor of scalars
  std::size_t nbytes() const {
    return this->range().volume() * this->nbatch_ * sizeof(T);
  }

  /// Const element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Const reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  const_reference operator[](const Ordinal ord) const {
    TA_ASSERT(!this->empty());
    // can't distinguish between operator[](Index...) and operator[](ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator[](index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  reference operator[](const Ordinal ord) {
    TA_ASSERT(!this->empty());
    // can't distinguish between operator[](Index...) and operator[](ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator[](index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Const element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Const reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  const_reference at_ordinal(const Ordinal ord) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  reference at_ordinal(const Ordinal ord) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Const element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  const_reference operator[](const Index& i) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  reference operator[](const Index& i) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Const element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  const_reference operator[](const std::initializer_list<Integer>& i) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  reference operator[](const std::initializer_list<Integer>& i) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Const element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Const reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral_v<Ordinal>>* = nullptr>
  const_reference operator()(const Ordinal& ord) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    // can't distinguish between operator[](Index...) and operator[](ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator()(index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p ord is
  /// included in the range, and `nbatch()==1`
  template <typename Ordinal,
            std::enable_if_t<std::is_integral_v<Ordinal>>* = nullptr>
  reference operator()(const Ordinal& ord) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    // can't distinguish between operator[](Index...) and operator[](ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator()(index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(this->range_.includes_ordinal(ord));
    return this->data()[ord];
  }

  /// Const element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  const_reference operator()(const Index& i) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  reference operator()(const Index& i) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Const element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  const_reference operator()(const std::initializer_list<Integer>& i) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Element accessor

  /// \tparam Integer An integral type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  reference operator()(const std::initializer_list<Integer>& i) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    const auto iord = this->range_.ordinal(i);
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Const element accessor

  /// \tparam Index an integral list ( see TiledArray::detail::is_integral_list
  /// )
  /// \param[in] i an index \return Const reference to the element at position
  /// \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <
      typename... Index,
      std::enable_if_t<(sizeof...(Index) > 1ul) &&
                       detail::is_integral_list<Index...>::value>* = nullptr>
  const_reference operator()(const Index&... i) const {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range().rank() == sizeof...(Index));
    // can't distinguish between operator()(Index...) and operator()(ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator()(index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    using Int = std::common_type_t<Index...>;
    const auto iord = this->range_.ordinal(
        std::array<Int, sizeof...(Index)>{{static_cast<Int>(i)...}});
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Element accessor

  /// \tparam Index an integral list ( see TiledArray::detail::is_integral_list
  /// )
  /// \param[in] i an index \return Reference to the element at position \c i
  /// .
  /// \note This asserts (using TA_ASSERT) that this is not empty, \p i is
  /// included in the range, and `nbatch()==1`
  template <
      typename... Index,
      std::enable_if_t<(sizeof...(Index) > 1ul) &&
                       detail::is_integral_list<Index...>::value>* = nullptr>
  reference operator()(const Index&... i) {
    TA_ASSERT(!this->empty());
    TA_ASSERT(this->nbatch() == 1);
    TA_ASSERT(this->range().rank() == sizeof...(Index));
    // can't distinguish between operator()(Index...) and operator()(ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range_.rank() != 1 &&
              "use Tensor::operator()(index) or "
              "Tensor::at_ordinal(index_ordinal) if this->range().rank()==1");
    using Int = std::common_type_t<Index...>;
    const auto iord = this->range_.ordinal(
        std::array<Int, sizeof...(Index)>{{static_cast<Int>(i)...}});
    TA_ASSERT(this->range_.includes_ordinal(iord));
    return this->data()[iord];
  }

  /// Iterator factory

  /// \return A const iterator to the first data element
  const_iterator begin() const { return (this->data() ? this->data() : NULL); }

  /// Iterator factory

  /// \return An iterator to the first data element
  iterator begin() { return (this->data() ? this->data() : NULL); }

  /// Iterator factory

  /// \return A const iterator to the last data element
  const_iterator end() const {
    return (this->data() ? this->data() + this->size() : NULL);
  }

  /// Iterator factory

  /// \return An iterator to the last data element
  iterator end() { return (this->data() ? this->data() + this->size() : NULL); }

  /// Iterator factory

  /// \return A const iterator to the first data element
  const_iterator cbegin() const { return (this->data() ? this->data() : NULL); }

  /// Iterator factory

  /// \return A const iterator to the first data element
  const_iterator cbegin() { return (this->data() ? this->data() : NULL); }

  /// Iterator factory

  /// \return A const iterator to the last data element
  const_iterator cend() const {
    return (this->data() ? this->data() + this->size() : NULL);
  }

  /// Iterator factory

  /// \return A const iterator to the last data element
  const_iterator cend() {
    return (this->data() ? this->data() + this->size() : NULL);
  }

  /// Read-only access to the data

  /// \return A const pointer to the tensor data
  const_pointer data() const { return this->data_.get(); }

  /// Mutable access to the data

  /// \return A mutable pointer to the tensor data
  pointer data() { return this->data_.get(); }

  /// @param[in] batch_idx the batch index
  /// @pre `batch_idx < this->nbatch()`
  /// @return A const pointer to the tensor data of the batch \p batch_idx
  const_pointer batch_data(size_t batch_idx) const {
    TA_ASSERT(batch_idx < this->nbatch());
    return data() + batch_idx * size();
  }

  /// @param[in] batch_idx the batch index
  /// @pre `batch_idx < this->nbatch()`
  /// @return A const pointer to the tensor data of the batch \p batch_idx
  pointer batch_data(size_t batch_idx) {
    TA_ASSERT(batch_idx < this->nbatch());
    return data() + batch_idx * size();
  }

  /// Read-only shared_ptr to the data

  /// \return A const shared_ptr to the tensor data
  std::shared_ptr<const value_type[]> data_shared() const {
    return this->data_;
  }

  /// Mutable shared_ptr to the data

  /// \return A mutable shared_ptr to the tensor data
  std::shared_ptr<value_type[]> data_shared() { return this->data_; }

  /// Test if the tensor is empty

  /// \return \c true if this tensor contains no
  ///         data, otherwise \c false.
  /// \note Empty Tensor is defaul_ish_ , i.e. it is *equal* to
  ///       a default-constructed Tensor
  ///       (`this->empty()` is equivalent to `*this == Tensor{}`),
  ///       but is not identical
  ///       to a default-constructed Tensor (e.g., `this->empty()` does not
  ///       imply `this->nbatch() == Tensor{}.nbatch()`)
  bool empty() const {
    // empty data_ implies default values for range_ (but NOT nbatch_)
    TA_ASSERT(
        (this->data_.use_count() == 0 && !this->range_) ||
        (this->data_.use_count() != 0 && this->range_));  // range is empty
    return this->data_.use_count() == 0;
  }

  /// MADNESS serialization function

  /// This function enables serialization within MADNESS
  /// \tparam Archive A MADNESS archive type
  /// \param[out] ar An input/output archive
  template <typename Archive>
  void serialize(Archive& ar) {
    bool empty = this->empty();
    auto range = this->range_;
    auto nbatch = this->nbatch_;
    ar & empty;
    if (!empty) {
      ar & range;
      ar & nbatch;
      if constexpr (madness::is_input_archive_v<Archive>) {
        *this = Tensor(std::move(range), nbatch, default_construct{true});
      }
      ar& madness::archive::wrap(this->data_.get(),
                                 this->range_.volume() * nbatch);
    } else {
      if constexpr (madness::is_input_archive_v<Archive>) {
        *this = Tensor{};
      }
    }
  }

  /// Swap tensor data

  /// \param other The tensor to swap with this
  void swap(Tensor& other) {
#ifdef TA_TENSOR_MEM_TRACE
    bool this_to_be_traced = false;
    bool other_to_be_traced = false;
    if (nbytes() >= trace_if_larger_than_) {
      this_to_be_traced = true;
      ptr_registry()->erase(
          this, make_string("TA::Tensor::swap()::data_.get()=", data_.get()));
    }
    if (other.nbytes() >= trace_if_larger_than_) {
      other_to_be_traced = true;
      ptr_registry()->erase(
          &other,
          make_string("TA::Tensor::swap()::data_.get()=", other.data_.get()));
    }
#endif
    std::swap(data_, other.data_);
    std::swap(range_, other.range_);
    std::swap(nbatch_, other.nbatch_);
#ifdef TA_TENSOR_MEM_TRACE
    if (other_to_be_traced) {
      ptr_registry()->insert(
          this, make_string("TA::Tensor::swap()::data_.get()=", data_.get()));
    }
    if (this_to_be_traced) {
      ptr_registry()->insert(
          &other,
          make_string("TA::Tensor::swap()::data_.get()=", other.data_.get()));
    }
#endif
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
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, lower_bound, upper_bound), this->data());
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  detail::TensorInterface<const T, BlockRange> block(
      const Index1& lower_bound, const Index2& upper_bound) const {
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
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, lower_bound, upper_bound), this->data());
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  detail::TensorInterface<const T, BlockRange> block(
      const std::initializer_list<Index1>& lower_bound,
      const std::initializer_list<Index2>& upper_bound) const {
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
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange> &&
                                        !std::is_same_v<PairRange, Range>>>
  detail::TensorInterface<const T, BlockRange> block(
      const PairRange& bounds) const {
    return detail::TensorInterface<const T, BlockRange>(
        BlockRange(this->range_, bounds), this->data());
  }

  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange> &&
                                        !std::is_same_v<PairRange, Range>>>
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

  // clang-format off
  /// Constructs a view of the block defined by a TiledArray::Range .

  /// Examples of using this:
  /// \code
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///
  ///   auto tview = t.block(TiledArray::Range(lobounds, upbounds));
  /// \endcode
  /// \tparam PairRange Type representing a range of generalized pairs (see TiledArray::detail::is_gpair_v )
  /// \param bounds The block bounds
  /// \return a {const,mutable} view of the block defined by its \p bounds
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `get<0>(bounds[i]) >= get<1>(bounds[i])`
  // clang-format on
  /// @{
  detail::TensorInterface<const T, BlockRange> block(
      const Range& bounds) const {
    return detail::TensorInterface<const T, BlockRange>(
        BlockRange(this->range_, bounds.lobound(), bounds.upbound()),
        this->data());
  }

  detail::TensorInterface<T, BlockRange> block(const Range& bounds) {
    return detail::TensorInterface<T, BlockRange>(
        BlockRange(this->range_, bounds.lobound(), bounds.upbound()),
        this->data());
  }
  /// @}

  /// Create a permuted copy of this tensor

  /// \tparam Perm A permutation tile
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted copy of this tensor
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor permute(const Perm& perm) const {
    return Tensor(*this, perm);
  }

  /// Shift the lower and upper bound of this tensor

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the tensor range
  /// \return A reference to this tensor
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  Tensor& shift_to(const Index& bound_shift) {
// although shift_to is currently fine on shared objects since ranges are
// not shared, this will change in the future
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
    TA_ASSERT(data_.use_count() <= 1);
#endif
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
    // although shift_to is currently fine on shared objects since ranges are
    // not shared, this will change in the future
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
    TA_ASSERT(data_.use_count() <= 1);
#endif
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
  auto binary(const Right& right, Op&& op) const {
    using result_value_type = decltype(op(
        std::declval<const T&>(), std::declval<const value_t<Right>&>()));
    using result_allocator_type = typename std::allocator_traits<
        Allocator>::template rebind_alloc<result_value_type>;
    using ResultTensor = Tensor<result_value_type, result_allocator_type>;
    return ResultTensor(*this, right, op);
  }

  /// Use a binary, element wise operation to construct a new, permuted tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Op The binary operation type
  /// \tparam Perm A permutation tile
  /// \param right The right-hand argument in the binary operation
  /// \param op The binary element-wise operation
  /// \param perm The permutation to be applied to this tensor
  /// \return A tensor where element \c i of the new tensor is equal to
  /// \c op(*this[i],other[i])
  template <typename Right, typename Op, typename Perm,
            typename std::enable_if<is_tensor<Right>::value &&
                                    detail::is_permutation_v<
                                        std::remove_reference_t<Perm>>>::type* =
                nullptr>
  auto binary(const Right& right, Op&& op, Perm&& perm) const {
    using result_value_type = decltype(op(
        std::declval<const T&>(), std::declval<const value_t<Right>&>()));
    using result_allocator_type = typename std::allocator_traits<
        Allocator>::template rebind_alloc<result_value_type>;
    using ResultTensor = Tensor<result_value_type, result_allocator_type>;
    // tile ops pass bipartite permutations here even if the result is a plain
    // tensor
    [[maybe_unused]] constexpr bool is_bperm =
        detail::is_bipartite_permutation_v<Perm>;
    constexpr bool result_is_tot = detail::is_tensor_of_tensor_v<ResultTensor>;

    if constexpr (!result_is_tot) {
      if constexpr (is_bperm) {
        TA_ASSERT(!inner(perm));  // ensure this is a plain permutation since
                                  // ResultTensor is plain
        return ResultTensor(*this, right, op, outer(std::forward<Perm>(perm)));
      } else
        return ResultTensor(*this, right, op, std::forward<Perm>(perm));
    } else {
      // AFAIK the other branch fundamentally relies on raw pointer arithmetic,
      // which won't work for ToTs.
      auto temp = binary(right, std::forward<Op>(op));
      Permute<decltype(temp), decltype(temp)> p;
      return p(temp, std::forward<Perm>(perm));
    }
    abort();  // unreachable
  }

  /// Use a binary, element wise operation to modify this tensor

  /// \tparam Right The right-hand tensor type
  /// \tparam Op The binary operation type
  /// \param right The right-hand argument in the binary operation
  /// \param op The binary element-wise operation
  /// \return A reference to this object
  /// \throw TiledArray::Exception When this tensor is empty.
  /// \throw TiledArray::Exception When \c other is empty.
  /// \throw TiledArray::Exception When the range of this tensor is not equal
  /// to the range of \c other.
  /// \throw TiledArray::Exception When this and \c other are the same.
  template <typename Right, typename Op,
            typename std::enable_if<detail::is_nested_tensor_v<Right>>::type* =
                nullptr>
  Tensor& inplace_binary(const Right& right, Op&& op) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
    TA_ASSERT(data_.use_count() <= 1);
#endif
    detail::inplace_tensor_op(op, *this, right);
    return *this;
  }

  /// Use a unary, element wise operation to construct a new tensor

  /// \tparam Op The unary operation type
  /// \param op The unary element-wise operation
  /// \return A tensor where element \c i of the new tensor is equal to
  /// \c op(*this[i])
  /// \throw TiledArray::Exception When this tensor is empty.
  template <typename Op>
  Tensor unary(Op&& op) const& {
    return Tensor(*this, op);
  }

  /// Use a unary, element wise operation to construct a new tensor

  /// \tparam Op The unary operation type
  /// \param op The unary element-wise operation
  /// \return A tensor where element \c i of the new tensor is equal to
  /// \c op(*this[i])
  /// \throw TiledArray::Exception When this tensor is empty.
  template <typename Op>
  Tensor unary(Op&& op) && {
    inplace_unary(std::forward<Op>(op));
    return std::move(*this);
  }

  /// Use a unary, element wise operation to construct a new, permuted tensor

  /// \tparam Op The unary operation type
  /// \tparam Perm A permutation tile
  /// \param op The unary element-wise operation
  /// \param perm The permutation to be applied to this tensor
  /// \return A permuted tensor with elements that have been modified by \c op
  /// \throw TiledArray::Exception When this tensor is empty.
  /// \throw TiledArray::Exception The dimension of \c perm does not match
  /// that of this tensor.
  template <typename Op, typename Perm,
            typename = std::enable_if_t<
                detail::is_permutation_v<std::remove_reference_t<Perm>>>>
  Tensor unary(Op&& op, Perm&& perm) const {
    constexpr bool is_tot = detail::is_tensor_of_tensor_v<Tensor>;
    [[maybe_unused]] constexpr bool is_bperm =
        detail::is_bipartite_permutation_v<Perm>;
    // tile ops pass bipartite permutations here even if this is a plain tensor
    if constexpr (!is_tot) {
      if (empty()) return {};
      if constexpr (is_bperm) {
        TA_ASSERT(inner_size(perm) == 0);  // ensure this is a plain permutation
        return Tensor(*this, op, outer(std::forward<Perm>(perm)));
      } else
        return Tensor(*this, op, std::forward<Perm>(perm));
    } else {
      auto temp = unary(std::forward<Op>(op));
      Permute<Tensor, Tensor> p;
      return p(temp, std::forward<Perm>(perm));
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
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
    TA_ASSERT(data_.use_count() <= 1);
#endif
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
  Tensor scale(const Scalar factor) const& {
    // early exit for empty this
    if (empty()) return {};

    return unary([factor](const value_type& a) {
      using namespace TiledArray::detail;
      return a * factor;
    });
  }

  /// Construct a scaled copy of this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A new tensor where the elements of this tensor are scaled by
  /// \c factor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor scale(const Scalar factor) && {
    scale_to(factor);
    return std::move(*this);
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
    // early exit for empty this
    if (empty()) return {};

    return unary(
        [factor](const value_type& a) {
          using namespace TiledArray::detail;
          return a * factor;
        },
        perm);
  }

  /// Scale this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor& scale_to(const Scalar factor) {
    // early exit for empty this
    if (empty()) return *this;

    return inplace_unary(
        [factor](value_type& MADNESS_RESTRICT res) { res *= factor; });
  }

  // Addition operations

  /// Add this and \c other to construct a new tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be added to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  Tensor add(const Right& right) const& {
    // early exit for empty right
    if (right.empty()) return this->clone();

    // early exit for empty this
    if (empty()) detail::clone_or_cast<Tensor>(right);

    return binary(
        right,
        [](const value_type& l, const value_t<Right>& r) -> decltype(l + r) {
          if constexpr (detail::is_tensor_v<value_type>) {
            if (l.empty()) {
              if (r.empty())
                return {};
              else
                return r.clone();
            } else {
              if (r.empty())
                return l.clone();
              else
                return l + r;
            }
          }
          return l + r;
        });
  }

  /// Add this and \c other to construct a new tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be added to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c other
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  Tensor add(const Right& right) && {
    add_to(right);
    return std::move(*this);
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
        right, [](const value_type& l, const value_type& r) { return l + r; },
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
    return binary(right, [factor](const value_type& l, const value_type& r) {
      return (l + r) * factor;
    });
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
        [factor](const value_type& l, const value_type& r) {
          return (l + r) * factor;
        },
        perm);
  }

  /// Add a constant to a copy of this tensor

  /// \param value The constant to be added to this tensor
  /// \return A new tensor where the elements are the sum of the elements of
  /// \c this and \c value
  Tensor add(const numeric_type value) const {
    // early exit for empty this
    if (empty()) return {};

    return unary([value](const value_type& a) { return a + value; });
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
    // early exit for empty this
    if (empty()) return {};

    return unary([value](const value_type& a) { return a + value; }, perm);
  }

  /// Add \c other to this tensor

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be added to this tensor
  /// \return A reference to this tensor
  template <typename Right,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  Tensor& add_to(const Right& right) {
    // early exit for empty right
    if (right.empty()) return *this;

    // early exit for empty this
    if (empty()) {
      *this = detail::clone_or_cast<Tensor>(right);
      return *this;
    }

    return inplace_binary(right, [](value_type& MADNESS_RESTRICT l,
                                    const value_t<Right> r) { l += r; });
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
        right, [factor](value_type& MADNESS_RESTRICT l,
                        const value_t<Right> r) { (l += r) *= factor; });
  }

  /// Add a constant to this tensor

  /// \param value The constant to be added
  /// \return A reference to this tensor
  template <typename Scalar,
            typename = std::enable_if_t<detail::is_numeric_v<Scalar>>>
  Tensor& add_to(const Scalar value) {
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
            typename = std::enable_if<
                detail::tensors_have_equal_nested_rank_v<Tensor, Right>>>
  Tensor subt(const Right& right) const {
    return binary(
        right,
        [](const value_type& l, const value_t<Right>& r) -> decltype(l - r) {
          if constexpr (detail::is_tensor_v<value_type>) {
            if (l.empty()) {
              if (r.empty())
                return {};
              else
                return -r;
            } else {
              if (r.empty())
                return l.clone();
              else
                return l - r;
            }
          } else {
            return l - r;
          }
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
        right, [](const value_type& l, const value_type& r) { return l - r; },
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
    return binary(right, [factor](const value_type& l, const value_type& r) {
      return (l - r) * factor;
    });
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
        [factor](const value_type& l, const value_type& r) {
          return (l - r) * factor;
        },
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
    // early exit for empty right
    if (right.empty()) return *this;

    return inplace_binary(
        right, [](auto& MADNESS_RESTRICT l, const auto& r) { l -= r; });
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
    // early exit for empty right
    if (right.empty()) {
      return this->scale_to(factor);
    }

    return inplace_binary(right,
                          [factor](auto& MADNESS_RESTRICT l, const auto& r) {
                            (l -= r) *= factor;
                          });
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
            typename std::enable_if<detail::is_nested_tensor_v<Right>>::type* =
                nullptr>
  decltype(auto) mult(const Right& right) const {
    auto mult_op = [](const value_type& l, const value_t<Right>& r) {
      return l * r;
    };

    if (empty() || right.empty()) {
      using res_t = decltype(std::declval<Tensor>().binary(
          std::declval<Right>(), mult_op));
      return res_t{};
    }

    return binary(right, mult_op);
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
      typename std::enable_if<detail::is_nested_tensor_v<Right> &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  decltype(auto) mult(const Right& right, const Perm& perm) const {
    return binary(
        right,
        [](const value_type& l, const value_t<Right>& r) { return l * r; },
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
      typename std::enable_if<detail::is_nested_tensor_v<Right> &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  decltype(auto) mult(const Right& right, const Scalar factor) const {
    return binary(right,
                  [factor](const value_type& l, const value_t<Right>& r) {
                    return (l * r) * factor;
                  });
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
  template <
      typename Right, typename Scalar, typename Perm,
      typename std::enable_if<detail::is_nested_tensor_v<Right> &&
                              detail::is_numeric_v<Scalar> &&
                              detail::is_permutation_v<Perm>>::type* = nullptr>
  decltype(auto) mult(const Right& right, const Scalar factor,
                      const Perm& perm) const {
    return binary(
        right,
        [factor](const value_type& l, const value_t<Right>& r) {
          return (l * r) * factor;
        },
        perm);
  }

  /// Multiply this tensor by \c right

  /// \tparam Right The right-hand tensor type
  /// \param right The tensor that will be multiplied by this tensor
  /// \return A reference to this tensor
  template <typename Right,
            typename std::enable_if<detail::is_nested_tensor_v<Right>>::type* =
                nullptr>
  Tensor& mult_to(const Right& right) {
    // early exit for empty right
    if (right.empty()) {
      *this = Tensor{};
      return *this;
    }

    // early exit for empty this
    if (empty()) return *this;

    return inplace_binary(right, [](value_type& MADNESS_RESTRICT l,
                                    const value_t<Right>& r) { l *= r; });
  }

  /// Scale and multiply this tensor by \c right

  /// \tparam Right The right-hand tensor type
  /// \tparam Scalar A scalar type
  /// \param right The tensor that will be multiplied by this tensor
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <
      typename Right, typename Scalar,
      typename std::enable_if<detail::is_nested_tensor_v<Right> &&
                              detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor& mult_to(const Right& right, const Scalar factor) {
    // early exit for empty this
    if (empty()) return *this;

    return inplace_binary(
        right, [factor](value_type& MADNESS_RESTRICT l,
                        const value_t<Right>& r) { (l *= r) *= factor; });
  }

  // Negation operations

  /// Create a negated copy of this tensor

  /// \return A new tensor that contains the negative values of this tensor
  Tensor neg() const {
    // early exit for empty this
    if (empty()) return this->clone();

    return unary([](const value_type r) { return -r; });
  }

  /// Create a negated and permuted copy of this tensor

  /// \tparam Perm A permutation type
  /// \param perm The permutation to be applied to this tensor
  /// \return A new tensor that contains the negative values of this tensor
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  Tensor neg(const Perm& perm) const {
    // early exit for empty this
    if (empty()) return this->clone();

    return unary([](const value_type l) { return -l; }, perm);
  }

  /// Negate elements of this tensor

  /// \return A reference to this tensor
  Tensor& neg_to() {
    // early exit for empty this
    if (empty()) return *this;

    return inplace_unary([](numeric_type& MADNESS_RESTRICT l) { l = -l; });
  }

  /// Create a complex conjugated copy of this tensor

  /// \return A copy of this tensor that contains the complex conjugate the
  /// values
  Tensor conj() const { return scale(detail::conj_op()); }

  /// Create a complex conjugated and scaled copy of this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A copy of this tensor that contains the scaled complex
  /// conjugate the values
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor conj(const Scalar factor) const {
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
    return scale(detail::conj_op(factor), perm);
  }

  /// Complex conjugate this tensor

  /// \return A reference to this tensor
  Tensor& conj_to() { return scale_to(detail::conj_op()); }

  /// Complex conjugate and scale this tensor

  /// \tparam Scalar A scalar type
  /// \param factor The scaling factor
  /// \return A reference to this tensor
  template <typename Scalar, typename std::enable_if<
                                 detail::is_numeric_v<Scalar>>::type* = nullptr>
  Tensor& conj_to(const Scalar factor) {
    return scale_to(detail::conj_op(factor));
  }

  // GEMM operations

  /// Contract this tensor with another tensor

  /// \tparam As Template parameter pack of a tensor type
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
  /// \tparam As Template parameter pack of the left-hand tensor type
  /// \tparam Bs Template parameter pack of the right-hand tensor type
  /// \tparam W The type of the scaling factor
  /// \param A The left-hand tensor that will be contracted
  /// \param B The right-hand tensor that will be contracted
  /// \param alpha The contraction result will be scaled by this value, then
  /// accumulated into \c this
  /// \param gemm_helper The *GEMM operation meta data
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
                 A.nbatch(), default_construct{true});
      beta = 0;
    }
    TA_ASSERT(this->nbatch() == A.nbatch());
    TA_ASSERT(this->nbatch() == B.nbatch());

    // may need to split gemm into multiply + accumulate for tracing purposes
#ifdef TA_ENABLE_TILE_OPS_LOGGING
    {
      const bool twostep =
          TiledArray::TileOpsLogger<T>::get_instance().gemm &&
          TiledArray::TileOpsLogger<T>::get_instance().gemm_print_contributions;
      std::unique_ptr<T[]> data_copy;
      size_t tile_volume;
      if (twostep) {
        tile_volume = range().volume() * nbatch();
        data_copy = std::make_unique<T[]>(tile_volume);
        std::copy(data_.get(), data_.get() + tile_volume, data_copy.get());
      }
      for (size_t i = 0; i < this->nbatch(); ++i) {
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
                  this->data(), this->nbatch());
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
    for (size_t i = 0; i < this->nbatch(); ++i) {
      auto Ci = this->batch(i);
      TiledArray::detail::gemm(alpha, A.batch(i), B.batch(i), beta, Ci,
                               gemm_helper);
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
    TA_ASSERT(left.nbatch() == right.nbatch());
    const auto batch_sz = left.nbatch();

    // Check that the inner dimensions of left and right match
    TA_ASSERT(gemm_helper.left_right_congruent(left.range().extent_data(),
                                               right.range().extent_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_right_congruent(left.range().lobound_data(),
                                               right.range().lobound_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_right_congruent(left.range().upbound_data(),
                                               right.range().upbound_data()));

    if (this->empty()) {  // initialize, if empty
      *this = Tensor(gemm_helper.make_result_range<range_type>(left.range(),
                                                               right.range()),
                     batch_sz);
    } else {
      // Check that the outer dimensions of left match the corresponding
      // dimensions in result
      TA_ASSERT(gemm_helper.left_result_congruent(left.range().extent_data(),
                                                  this->range_.extent_data()));
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.left_result_congruent(left.range().lobound_data(),
                                                  this->range_.lobound_data()));
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.left_result_congruent(left.range().upbound_data(),
                                                  this->range_.upbound_data()));

      // Check that the outer dimensions of right match the corresponding
      // dimensions in result
      TA_ASSERT(gemm_helper.right_result_congruent(right.range().extent_data(),
                                                   this->range_.extent_data()));
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.right_result_congruent(
                    right.range().lobound_data(), this->range_.lobound_data()));
      TA_ASSERT(ignore_tile_position() ||
                gemm_helper.right_result_congruent(
                    right.range().upbound_data(), this->range_.upbound_data()));

      // check that batch size of this matches that of left and right
      TA_ASSERT(this->nbatch() == batch_sz);
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

    for (integer b = 0; b != nbatch(); ++b) {
      auto this_data = this->batch_data(b);
      auto left_data = left.batch_data(b);
      auto right_data = right.batch_data(b);
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
            elem_muladd_op(*(this_data + c_offset), *(left_data + a_offset),
                           *(right_data + b_offset));
          }
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
  /// \tparam ReduceOp The reduction operation type
  /// \tparam JoinOp The join operation type
  /// \tparam T a type that can be used as argument to ReduceOp
  /// \param reduce_op The element-wise reduction operation
  /// \param join_op The join result operation
  /// \param identity The identity value of the reduction
  /// \return The reduced value
  template <typename ReduceOp, typename JoinOp, typename Identity>
  auto reduce(ReduceOp&& reduce_op, JoinOp&& join_op,
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
  /// \tparam Identity A type that can be used as argument to ReduceOp
  /// \param other The right-hand argument of the binary reduction
  /// \param reduce_op The element-wise reduction operation
  /// \param join_op The join result operation
  /// \param identity The identity value of the reduction
  /// \return The reduced value
  template <typename Right, typename ReduceOp, typename JoinOp,
            typename Identity,
            typename std::enable_if<is_tensor<Right>::value>::type* = nullptr>
  auto reduce(const Right& other, ReduceOp&& reduce_op, JoinOp&& join_op,
              Identity&& identity) const {
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
      res += TiledArray::detail::squared_norm(arg);
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

  /// @return pointer to the PtrRegistry object used for tracing TA::Tensor
  /// lifetime
  /// @warning only nonnull if configured with `TA_TENSOR_MEM_TRACE=ON`
  static PtrRegistry* ptr_registry() {
#ifdef TA_TENSOR_MEM_TRACE
    static PtrRegistry registry;
    return &registry;
#else
    return nullptr;
#endif
  }

#ifdef TA_TENSOR_MEM_TRACE
  /// @param nbytes sets the minimum size of TA::Tensor objects whose lifetime
  /// will be tracked; must be greater or equal to 1
  static void trace_if_larger_than(std::size_t nbytes) {
    TA_ASSERT(nbytes >= 1);
    trace_if_larger_than_ = nbytes;
  }
  /// @return the minimum size of TA::Tensor objects whose lifetime
  /// will be tracked
  static std::size_t trace_if_larger_than() { return trace_if_larger_than_; }
#endif

 private:
#ifdef TA_TENSOR_MEM_TRACE
  static std::size_t trace_if_larger_than_;
#endif

};  // class Tensor

#ifdef TA_TENSOR_MEM_TRACE
template <typename T, typename A>
std::size_t Tensor<T, A>::trace_if_larger_than_ =
    std::numeric_limits<std::size_t>::max();
#endif

template <typename T, typename A>
Tensor<T, A> operator*(const Permutation& p, const Tensor<T, A>& t) {
  return t.permute(p);
}

// template <typename T, typename A>
// const typename Tensor<T, A>::range_type Tensor<T, A>::empty_range_;

/// equality comparison
/// \param[in] a a Tensor object
/// \param[in] b another Tensor object
/// \return true if ranges and data of \p a and \p b are equal
/// \internal this does not compare nbatch  so any
///           2 empty tensors are equal even if their nbatch
///           differ
template <typename T, typename A>
bool operator==(const Tensor<T, A>& a, const Tensor<T, A>& b) {
  return a.range() == b.range() &&
         std::equal(a.data(), a.data() + a.size(), b.data());
}

/// inequality comparison
/// \param[in] a a Tensor object
/// \param[in] b another Tensor object
/// \return true if ranges and data of \p a and \p b are not equal
template <typename T, typename A>
bool operator!=(const Tensor<T, A>& a, const Tensor<T, A>& b) {
  return !(a == b);
}

namespace detail {

/// Implements taking the trace of a Tensor<T,A>
///
/// \tparam T The type of the elements in the tensor. For this specialization
///           to be considered must satisfy the concept of numeric type.
/// \tparam A The type of the allocator for the tensor
template <typename T, typename A>
struct Trace<Tensor<T, A>, detail::enable_if_numeric_t<T>> {
  decltype(auto) operator()(const Tensor<T, A>& t) const {
    using size_type = typename Tensor<T, A>::size_type;
    using value_type = typename Tensor<T, A>::value_type;
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

namespace detail {

template <typename T, typename A>
struct real_t_impl<Tensor<T, A>> {
  using type = typename Tensor<T, A>::template rebind_numeric_t<
      typename Tensor<T, A>::scalar_type>;
};

template <typename T, typename A>
struct complex_t_impl<Tensor<T, A>> {
  using type = typename Tensor<T, A>::template rebind_numeric_t<
      std::complex<typename Tensor<T, A>::scalar_type>>;
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
