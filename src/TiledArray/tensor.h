/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
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

#ifndef TILEDARRAY_TENSOR_H__INCLUDED
#define TILEDARRAY_TENSOR_H__INCLUDED

#include <TiledArray/dense_storage.h>
#include <TiledArray/range.h>
#include <world/move.h>

namespace TiledArray {
  namespace expressions {

    /// Evaluation tensor

    /// This tensor is used as an evaluated intermediate for other tensors.
    /// \tparma T the value type of this tensor
    /// \tparam A The allocator type for the data
    template <typename T, typename A = Eigen::aligned_allocator<T> >
    class Tensor {
    private:
      struct Enabler { };
    public:
      typedef Tensor<T, A> Tensor_;
      typedef DenseStorage<T,A> storage_type;
      typedef Range range_type;
      typedef typename storage_type::value_type value_type;
      typedef typename storage_type::const_reference const_reference;
      typedef typename storage_type::reference reference;
      typedef typename storage_type::const_iterator const_iterator;
      typedef typename storage_type::iterator iterator;
      typedef typename storage_type::difference_type difference_type;
      typedef typename storage_type::const_pointer const_pointer;
      typedef typename storage_type::pointer pointer;
      typedef typename storage_type::size_type size_type;

      /// Default constructor

      /// Construct an empty tensor that has no data or dimensions
      Tensor() : range_(), data_() { }

      /// Construct an evaluated tensor

      /// \param r An array with the size of of each dimension
      /// \param v The value of the tensor elements
      explicit Tensor(const Range& r, const value_type& v = value_type()) :
        range_(r), data_(r.volume(), v)
      { }

      /// Construct an evaluated tensor
      template <typename InIter>
      Tensor(const Range& r, InIter it,
          typename madness::enable_if<TiledArray::detail::is_input_iterator<InIter>, Enabler>::type = Enabler()) :
        range_(r), data_(r.volume(), it)
      { }

      /// Construct an evaluated tensor
      template <typename InIter, typename Op>
      Tensor(const Range& r, InIter it, const Op& op,
          typename madness::enable_if< TiledArray::detail::is_input_iterator<InIter>,
          Enabler>::type = Enabler()) :
        range_(r), data_(r.volume(), it, op)
      { }

      /// Construct an evaluated tensor
      template <typename InIter1, typename InIter2, typename Op>
      Tensor(const Range& r, InIter1 it1, InIter2 it2, const Op& op,
          typename madness::enable_if_c<
            TiledArray::detail::is_input_iterator<InIter1>::value &&
            TiledArray::detail::is_input_iterator<InIter2>::value, Enabler>::type = Enabler()) :
        range_(r), data_(r.volume(), it1, it2, op)
      { }

      /// Copy constructor

      /// Do a deep copy of \c other
      /// \param other The tile to be copied.
      Tensor(const Tensor_& other) :
        range_(other.range_), data_(other.data_)
      { }

      /// Move constructor

      /// Do a shallow copy of \c other
      /// \param other The tile to be Moved
      Tensor(const madness::detail::MoveWrapper<Tensor<T, A> >& wrapper) :
          range_(wrapper.get().range()), data_()
      {
        data_.swap(wrapper.get().data_);
      }

      /// Copy assignment

      /// Evaluate \c other to this tensor
      /// \param other The tensor to be copied
      /// \return this tensor
      Tensor_& operator=(const Tensor_& other) {
        if(range_ == other.range())
          std::copy(other.begin(), other.end(), data_.begin());
        else {
          range_ = other.range();
          data_ = other.data_;
        }

        return *this;
      }

      /// Copy assignment

      /// Evaluate \c other to this tensor
      /// \param other The tensor to be copied
      /// \return this tensor
      template <typename U, typename AU>
      Tensor_& operator=(const Tensor<U, AU>& other) {
        if(range_ == other.range())
          std::copy(other.begin(), other.end(), data_.begin());
        else {
          range_ = other.range();
          storage_type(other.size(), other.begin()).swap(data_);
        }

        return *this;
      }

      /// Move assignment

      /// Evaluate \c other to this tensor
      /// \param other The tensor to be moved
      /// \return this tensor
      Tensor_& operator=(const madness::detail::MoveWrapper<Tensor<T, A> >& other) {
        range_ = other.get().range();
        data_.swap(other.get().data_);
        return *this;
      }

      /// Plus assignment

      /// Evaluate \c other to this tensor
      /// \param other The tensor to be copied
      /// \return this tensor
      template <typename U, typename AU>
      Tensor_& operator+=(const Tensor<U, AU>& other) {
        if(data_.empty()) {
          range_ = other.range();
          storage_type temp(other.range().volume());
          temp.swap(data_);
        }

        TA_ASSERT(range_ == other.range());
        data_ += other;
        return *this;
      }

      /// Minus assignment

      /// Evaluate \c other to this tensor
      /// \param other The tensor to be copied
      /// \return this tensor
      template <typename U, typename AU>
      Tensor_& operator-=(const Tensor<U, AU>& other) {
        if(data_.empty()) {
          range_ = other.range();
          storage_type temp(other.range().volume());
          temp.swap(data_);
        }

        TA_ASSERT(range_ == other.range());
        data_ -= other;

        return *this;
      }

      /// Multiply assignment

      /// Evaluate \c other to this tensor
      /// \param other The tensor to be copied
      /// \return this tensor
      template <typename U, typename RU, typename AU>
      Tensor_& operator*=(const Tensor<U, AU>& other) {
        if(data_.empty()) {
          range_ = other.range();
          storage_type temp(other.range().volume());
          temp.swap(data_);
        }

        TA_ASSERT(range_ == other.range());
        data_ *= other;

        return *this;
      }

      /// Tensor range object accessor

      /// \return The tensor range object
      const range_type& range() const { return range_; }

      /// Tensor dimension size accessor

      /// \return The number of elements in the tensor
      size_type size() const { return range_.volume(); }

      /// Element accessor

      /// \return The element at the \c i position.
      const_reference operator[](size_type i) const { return data_[i]; }

      /// Element accessor

      /// \return The element at the \c i position.
      reference operator[](size_type i) { return data_[i]; }


      /// Element accessor

      /// \return The element at the \c i position.
      template <typename Index>
      typename madness::disable_if<std::is_integral<Index>, const_reference>::type
      operator[](const Index& i) const { return data_[range_.ord(i)]; }

      /// Element accessor

      /// \return The element at the \c i position.
      template <typename Index>
      typename madness::disable_if<std::is_integral<Index>, reference>::type
      operator[](const Index& i) { return data_[range_.ord(i)]; }

      /// Iterator factory

      /// \return An iterator to the first data element
      const_iterator begin() const { return data_.begin(); }

      /// Iterator factory

      /// \return An iterator to the first data element
      iterator begin() { return data_.begin(); }

      /// Iterator factory

      /// \return An iterator to the last data element
      const_iterator end() const { return data_.end(); }

      /// Iterator factory

      /// \return An iterator to the last data element
      iterator end() { return data_.end(); }

      /// Data direct access

      /// \return A const pointer to the tensor data
      const_pointer data() const { return data_.data(); }

      /// Data direct access

      /// \return A const pointer to the tensor data
      pointer data() { return data_.data(); }

      bool empty() const { return data_.empty(); }

      /// Serialize tensor data

      /// \tparam Archive The serialization archive type
      /// \param ar The serialization archive
      template <typename Archive>
      void serialize(Archive& ar) {
        ar & range_ & data_;
      }

      /// Swap tensor data

      /// \param other The tensor to swap with this
      void swap(Tensor_& other) {
        range_.swap(other.range_);
        data_.swap(other.data_);
      }

    private:

      range_type range_; ///< Tensor size info
      storage_type data_; ///< Tensor data
    }; // class Tensor

    template <typename T, typename A>
    std::ostream& operator<<(std::ostream& os, const Tensor<T, A>& t) {
      os << t.range() << " { ";
      for(typename Tensor<T, A>::const_iterator it = t.begin(); it != t.end(); ++it) {
        os << *it << " ";
      }

      os << "}";

      return os;
    }

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_H__INCLUDED
