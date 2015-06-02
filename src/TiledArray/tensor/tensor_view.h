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


  /// Sub-block view of an N-dimensional Tensor

  /// TensorView is a shallow reference to sub-block of a Tensor object.
  /// \note It is the user's responsibility to ensure the original tensor does
  template <typename T>
  class TensorView {
  public:
    typedef TensorView<T> TensorView_; ///< This class type
    typedef BlockRange range_type; ///< Tensor range type
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
    template <typename>
    friend class TensorView;


    /// Construct a tensor with the same lower and upper bound as this view

    /// \return A new tensor object
    Tensor<value_type> make_result_tensor() const {
      detail::SizeArray<const size_type>
          lower_bound(range_.start(), range_.start() + range_.rank());
      detail::SizeArray<const size_type>
          upper_bound(range_.finish(), range_.finish() + range_.rank());
      return Tensor<T>(Range(lower_bound, upper_bound));
    }

    /// Get the inner size

    /// \return The largest contiguous, inner-dimension size.
    size_type inner_size() const {
      const size_type* restrict const stride = range_.weight();
      const size_type* restrict const size = range_.size();

      int i = int(range_.rank()) - 1;
      size_type result = size[i];

      for(--i; i >= 0; --i) {
        const size_type stride_i = stride[i];
        const size_type size_i = size[i];
        if(result != stride_i)
          break;
        result *= size_i;
      }

      return result;
    }

    /// Get the inner size

    /// Here we assume the size of the other range is equal to this view.
    /// \return The largest contiguous, inner-dimension size.
    size_type inner_size(const size_type* restrict const other_stride) const {
      const size_type* restrict const stride = range_.weight();
      const size_type* restrict const size = range_.size();

      int i = int(range_.rank()) - 1;
      size_type result = size[i];

      for(--i; i >= 0; --i) {
        const size_type stride_i = stride[i];
        const size_type other_stride_i = other_stride[i];
        const size_type size_i = size[i];
        if((result != stride_i) || (result != other_stride_i))
          break;
        result *= size_i;
      }

      return result;
    }


    range_type range_; ///< View sub-block range
    pointer data_; ///< Pointer to the original tensor data

  public:

    /// Compiler generated functions
    TensorView() = delete;
    ~TensorView() = default;
    TensorView(const TensorView_&) = default;
    TensorView(TensorView_&&) = default;

    /// Type conversion copy constructor

    /// \tparam U The value type of the other view
    /// \param other The other tensor view to be copied
    template <typename U,
        enable_if_t<std::is_convertible<typename TensorView<U>::pointer, pointer>::value>* = nullptr>
    TensorView(const TensorView<U>& other) :
      range_(other.range_), data_(other.data_)
    { }

    /// Type conversion move constructor

    /// \tparam U The value type of the other tensor view
    /// \param other The other tensor view to be moved
    template <typename U,
        enable_if_t<std::is_convertible<typename TensorView<U>::pointer, pointer>::value>* = nullptr>
    TensorView(TensorView<U>&& other) :
      range_(std::move(other.range_)), data_(other.data_)
    {
      other.data_ = nullptr;
    }

    /// Construct a new view of \c tensor

    /// \tparam U The element type of the tensor
    /// \tparam A the allocator type of the tensor
    /// \param tensor The parent tensor object
    /// \param lower_bound The lower bound of the tensor view
    /// \param upper_bound The upper bound of the tensor view
    template <typename U, typename A, typename Index,
        enable_if_t<std::is_convertible<typename Tensor<U, A>::pointer, pointer>::value>* = nullptr>
    TensorView(Tensor<U, A>& tensor, const Index& lower_bound, const Index& upper_bound) :
      range_(tensor.range(), lower_bound, upper_bound), data_(tensor.data())
    { }

    template <typename U, typename A, typename Index,
        enable_if_t<std::is_convertible<typename Tensor<U, A>::const_pointer, pointer>::value>* = nullptr>
    TensorView(const Tensor<U, A>& tensor, const Index& lower_bound, const Index& upper_bound) :
      range_(tensor.range(), lower_bound, upper_bound), data_(tensor.data())
    { }

    TensorView_& operator=(const TensorView_& other) const {
      if(data_ != other.data_) { // Check for self assignment
        inplace_binary(other,
            [] (value_type& restrict result, const value_type& arg)
            { result = arg; });
      }

      return *this;
    }

    template <typename U>
    TensorView_& operator=(const U& other) {
      if(data_ != other.data()) { // Check for self assignment
        detail::inplace_binary(*this, other,
            [] (value_type& restrict result, const value_type& arg)
            { result = arg; });
      }

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
    void swap(TensorView_& other) {
      range_.swap(other.range_);
      std::swap(data_, other.data_);
    }


    // Generic vector operations -----------------------------------------------


    /// Use a unary, element wise operation to construct a new tensor

    /// \tparam Op The unary operation type
    /// \param op The unary, element-wise operation
    /// \return A tensor where element \c i of the new tensor is equal to
    /// \c op(arg[i])
    template <typename Op>
    inline Tensor<value_type> unary(Op&& op) const {
      Tensor<value_type> result = make_result_tensor();
      const size_type stride = range_.size()[range_.rank() - 1];
      const size_type volume = range_.volume();

      for(size_type i = 0ul; i < volume; i += stride)
        math::vector_op(std::forward<Op>(op), stride, result.data() + i,
            data_ + range_.ord(i));

      return result;
    }


    /// Use a unary, element wise operation to modify this tensor

    /// \tparam Op The unary operation type
    /// \param op The unary, element-wise operation
    /// \return A reference to this object
    template <typename Result, typename Op>
    TensorView_& inplace_unary_impl(const Op& op) {
      const size_type stride = range_.size()[range_.rank() - 1];
      const size_type volume = range_.volume();

      for(size_type i = 0ul; i < volume; i += stride)
        math::vector_op(std::forward<Op>(op), stride, data_ + range_.ord(i));

      return *this;
    }

  }; // class TensorView

  /// TensorConstView is a read-only variant of TensorView
  template <typename T>
  using TensorConstView = TensorView<typename std::add_const<T>::type>;

} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_TENSOR_VIEW_H__INCLUDED
