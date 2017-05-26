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
 *  shift_wrapper.h
 *  May 31, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_SHIFT_WRAPPER_H__INCLUDED
#define TILEDARRAY_TENSOR_SHIFT_WRAPPER_H__INCLUDED

#include <TiledArray/type_traits.h>
#include <TiledArray/tensor/kernels.h>

namespace TiledArray {
  namespace detail {

    /// Shift wrapper class

    /// This object is used to disable the global range checks for tensor objects
    /// in various arithmetic operations. The interface of this object is the
    /// minimum required
    template <typename T>
    class ShiftWrapper {
    public:
      typedef ShiftWrapper<T> ShiftWrapper_;
      typedef typename std::remove_const<T>::type tensor_type;
      typedef typename tensor_type::value_type value_type;
      typedef typename tensor_type::size_type size_type;
      typedef typename tensor_type::range_type range_type;
      typedef typename tensor_type::reference reference;
      typedef typename tensor_type::const_reference const_reference;
      typedef typename tensor_type::pointer pointer;
      typedef typename tensor_type::const_pointer const_pointer;


    private:

      T* const tensor_; ///< Tensor object

    public:
      // Compiler generated functions
      ShiftWrapper() = delete;
      ShiftWrapper(const ShiftWrapper&) = default;
      ShiftWrapper(ShiftWrapper&&) = default;
      ~ShiftWrapper() = default;
      ShiftWrapper& operator=(const ShiftWrapper&) = delete;
      ShiftWrapper& operator=(ShiftWrapper&&) = delete;

      ShiftWrapper(T& tensor) : tensor_(&tensor) { }

      /// Assignment operator

      /// This assignment operator is a simple pass through assignment to the
      /// tensor object. It handles both copy and move assignments
      /// \tparam U The right-hand argument type
      /// \param other The right-hand argument
      /// \return A reference to this object
      template <typename U>
      ShiftWrapper<T>& operator=(U&& other) {
        typedef typename std::decay<U>::type arg_type;
        detail::inplace_tensor_op([] (reference MADNESS_RESTRICT l,
            typename arg_type::const_reference MADNESS_RESTRICT r) { l = r; },
            *this, other);
        return *this;
      }

      /// Tensor accessor

      /// \return A reference to the wrapped tensor
      T& get() const { return *tensor_; }

      /// Tensor type conversion operator

      /// \return A reference to the wrapped tensor
//      operator T&() const { return get(); }

      /// Tensor range accessor

      /// \return a const reference to the tensor range object
      decltype(auto) range() const { return tensor_->range(); }

      /// Tensor data pointer accessor

      /// \return A pointer to the tensor data
      decltype(auto) data() const { return tensor_->data(); }

      /// Check for an empty tensor

      /// \return \c true if the tensor is empty, otherwise \c false.
      decltype(auto) empty() const { return tensor_->empty(); }

    }; // class ShiftWrapper


    /// Check for congruent range objects with a shifted tensor

    /// \tparam Left The left-hand tensor type
    /// \tparam Right The right-hand tensor type
    /// \param left The left-hand tensor
    /// \param right The right-hand tensor
    /// \return \c true if the lower and upper bounds of the left- and
    /// right-hand tensor ranges are equal, otherwise \c false
    template <typename Left, typename Right>
    inline bool is_range_congruent(const Left& left, const ShiftWrapper<Right>& right) {
      return (left.range().rank() == right.range().rank()) &&
          std::equal(left.range().extent_data(),
              left.range().extent_data() + left.range().rank(), right.range().extent_data());
    }

  }  // namespace detail

  /// Shift a tensor from one range to another

  /// \tparam T A tensor type
  /// \param tensor The tensor object to shift
  /// \return A shifted tensor object
  template <typename T>
  detail::ShiftWrapper<T> shift(T& tensor) {
    return detail::ShiftWrapper<T>(tensor);
  }


  /// Shift a tensor from one range to another

  /// \tparam T A tensor type
  /// \param tensor The tensor object to shift
  /// \return A shifted tensor object
  template <typename T>
  detail::ShiftWrapper<const T> shift(const T& tensor) {
    return detail::ShiftWrapper<const T>(tensor);
  }

} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_SHIFT_WRAPPER_H__INCLUDED
