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

#ifndef TILEDARRAY_TENSOR_H__INCLUDED
#define TILEDARRAY_TENSOR_H__INCLUDED

#include <TiledArray/dense_storage.h>
#include <TiledArray/range.h>
#include <TiledArray/madness.h>
#include <TiledArray/math/functional.h>

namespace TiledArray {

  namespace detail {

    /// Place-holder object for a zero tensor.
    template <typename T>
    struct ZeroTensor {
      typedef T value_type;
    }; // struct ZeroTensor

  }  // namespace detail

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
    typedef Tensor_ eval_type; ///< The type used when evaluating expressions
    typedef DenseStorage<T,A> storage_type;
    typedef Range range_type;
    typedef typename storage_type::value_type value_type;
    typedef typename TiledArray::detail::scalar_type<T>::type
        numeric_type; ///< the numeric type that supports T
    typedef typename storage_type::const_reference const_reference;
    typedef typename storage_type::reference reference;
    typedef typename storage_type::const_iterator const_iterator;
    typedef typename storage_type::iterator iterator;
    typedef typename storage_type::difference_type difference_type;
    typedef typename storage_type::const_pointer const_pointer;
    typedef typename storage_type::pointer pointer;
    typedef typename storage_type::size_type size_type;

  private:

    /// Evaluation tensor

    /// This tensor is used as an evaluated intermediate for other tensors.
    /// \tparma T the value type of this tensor
    /// \tparam A The allocator type for the data
    class Impl {
    public:

      /// Default constructor

      /// Construct an empty tensor that has no data or dimensions
      Impl() : range_(), data_() { }

      /// Construct an evaluated tensor

      /// \param r An array with the size of of each dimension
      /// \param v The value of the tensor elements
      explicit Impl(const Range& r, const value_type& v) :
        range_(r), data_(r.volume(), v)
      { }

      /// Construct an evaluated tensor
      template <typename InIter>
      Impl(const Range& r, InIter it,
          typename madness::enable_if<TiledArray::detail::is_input_iterator<InIter>, Enabler>::type = Enabler()) :
        range_(r), data_(r.volume(), it)
      { }

      /// Construct an evaluated tensor
      template <typename InIter, typename Op>
      Impl(const Range& r, InIter it, const Op& op,
          typename madness::enable_if< TiledArray::detail::is_input_iterator<InIter>,
          Enabler>::type = Enabler()) :
        range_(r), data_(r.volume(), it, op)
      { }

      /// Construct an evaluated tensor
      template <typename InIter1, typename InIter2, typename Op>
      Impl(const Range& r, InIter1 it1, InIter2 it2, const Op& op,
          typename madness::enable_if_c<
            TiledArray::detail::is_input_iterator<InIter1>::value &&
            TiledArray::detail::is_input_iterator<InIter2>::value, Enabler>::type = Enabler()) :
        range_(r), data_(r.volume(), it1, it2, op)
      { }

      /// Copy constructor

      /// Do a deep copy of \c other
      /// \param other The tile to be copied.
      Impl(const Impl& other) :
        range_(other.range_), data_(other.data_)
      { }

      /// Element accessor

      /// Serialize tensor data

      /// \tparam Archive The serialization archive type
      /// \param ar The serialization archive
      template <typename Archive>
      void serialize(Archive& ar) {
        ar & range_ & data_;
      }

      range_type range_; ///< Tensor size info
      storage_type data_; ///< Tensor data
    }; // class Impl

    std::shared_ptr<Impl> pimpl_; ///< Shared pointer to implementation object
    static const range_type empty_range_; ///< Empty range

  public:

    /// Default constructor

    /// Construct an empty tensor that has no data or dimensions
    Tensor() : pimpl_() { }

    /// Construct an evaluated tensor

    /// \param r An array with the size of of each dimension
    /// \param v The value of the tensor elements
    explicit Tensor(const Range& r, const value_type& v = value_type()) :
      pimpl_(new Impl(r, v))
    { }

    /// Construct an evaluated tensor
    template <typename InIter>
    Tensor(const Range& r, InIter it,
        typename madness::enable_if<TiledArray::detail::is_input_iterator<InIter>, Enabler>::type = Enabler()) :
      pimpl_(new Impl(r, it))
    { }

    /// Construct an evaluated tensor
    template <typename InIter, typename Op>
    Tensor(const Range& r, InIter it, const Op& op,
        typename madness::enable_if< TiledArray::detail::is_input_iterator<InIter>,
        Enabler>::type = Enabler()) :
      pimpl_(new Impl(r, it, op))
    { }

    /// Construct an evaluated tensor
    template <typename InIter1, typename InIter2, typename Op>
    Tensor(const Range& r, InIter1 it1, InIter2 it2, const Op& op,
        typename madness::enable_if_c<
          TiledArray::detail::is_input_iterator<InIter1>::value &&
          TiledArray::detail::is_input_iterator<InIter2>::value, Enabler>::type = Enabler()) :
      pimpl_(new Impl(r, it1, it2, op))
    { }

    /// Copy constructor

    /// Do a deep copy of \c other
    /// \param other The tile to be copied.
    Tensor(const Tensor_& other) :
      pimpl_(other.pimpl_)
    { }

    /// Copy assignment

    /// Evaluate \c other to this tensor
    /// \param other The tensor to be copied
    /// \return this tensor
    Tensor_& operator=(const Tensor_& other) {
      pimpl_ = other.pimpl_;

      return *this;
    }

    /// Plus assignment

    /// Evaluate \c other to this tensor
    /// \param other The tensor to be copied
    /// \return this tensor
    template <typename U, typename AU>
    Tensor_& operator+=(const Tensor<U, AU>& other) {
      if(!pimpl_) {
        if(! other.empty())
          pimpl_.reset(new Impl(other.range(), other.begin()));
      } else {
        TA_ASSERT(pimpl_->range_ == other.range());
        pimpl_->data_ += other;
      }

      return *this;
    }

    /// Minus assignment

    /// Evaluate \c other to this tensor
    /// \param other The tensor to be copied
    /// \return this tensor
    template <typename U, typename AU>
    Tensor_& operator-=(const Tensor<U, AU>& other) {
      if(!pimpl_) {
        if(! other.empty())
          pimpl_.reset(new Impl(other.range(), other.begin(), std::bind1st(std::minus<value_type>(), 0)));
      } else {
        TA_ASSERT(pimpl_->range_ == other.range());
        pimpl_->data_ -= other;
      }

      return *this;
    }

    /// Multiply assignment

    /// Evaluate \c other to this tensor
    /// \param other The tensor to be copied
    /// \return this tensor
    template <typename U, typename RU, typename AU>
    Tensor_& operator*=(const Tensor<U, AU>& other) {
      if(!pimpl_) {
        if(! other.empty())
          pimpl_.reset(new Impl(other.range(), 0));
      } else {
        TA_ASSERT(pimpl_->range_ == other.range());
        pimpl_->data_ *= other;
      }

      return *this;
    }

    /// Multiply assignment

    /// Evaluate \c other to this tensor
    /// \param other The tensor to be copied
    /// \return this tensor
    template <typename U>
    typename madness::enable_if<detail::is_numeric<U>, Tensor_&>::type
    operator*=(const U u) {
      if(pimpl_) {
        const iterator end =  pimpl_->data_.end();
        for(iterator it = pimpl_->data_.begin(); it != end; ++it)
          *it *= u;
      }

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
      return pimpl_->data_[i];
    }

    /// Element accessor

    /// \return The element at the \c i position.
    reference operator[](const size_type i) {
      TA_ASSERT(pimpl_);
      return pimpl_->data_[i];
    }


    /// Element accessor

    /// \return The element at the \c i position.
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, const_reference>::type
    operator[](const Index& i) const {
      TA_ASSERT(pimpl_);
      return pimpl_->data_[pimpl_->range_.ord(i)];
    }

    /// Element accessor

    /// \return The element at the \c i position.
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, reference>::type
    operator[](const Index& i) {
      TA_ASSERT(pimpl_);
      return pimpl_->data_[pimpl_->range_.ord(i)];
    }

    /// Iterator factory

    /// \return An iterator to the first data element
    const_iterator begin() const { return (pimpl_ ? pimpl_->data_.begin() : NULL); }

    /// Iterator factory

    /// \return An iterator to the first data element
    iterator begin() { return (pimpl_ ? pimpl_->data_.begin() : NULL); }

    /// Iterator factory

    /// \return An iterator to the last data element
    const_iterator end() const { return (pimpl_ ? pimpl_->data_.end() : NULL); }

    /// Iterator factory

    /// \return An iterator to the last data element
    iterator end() { return (pimpl_ ? pimpl_->data_.end() : NULL); }

    /// Data direct access

    /// \return A const pointer to the tensor data
    const_pointer data() const { return (pimpl_ ? pimpl_->data_.data() : NULL); }

    /// Data direct access

    /// \return A const pointer to the tensor data
    pointer data() { return (pimpl_ ? pimpl_->data_.data() : NULL); }

    bool empty() const { return (pimpl_ ? pimpl_->data_.empty() : true); }

    /// Serialize tensor data

    /// \tparam Archive The serialization archive type
    /// \param ar The serialization archive
    template <typename Archive>
    void serialize(Archive& ar) {
      if(!pimpl_)
        pimpl_.reset(new Impl());
      pimpl_->serialize(ar);
    }

    /// Swap tensor data

    /// \param other The tensor to swap with this
    void swap(Tensor_& other) {
      std::swap(pimpl_, other.pimpl_);
    }

  }; // class Tensor

  template <typename T, typename A>
  const typename Tensor<T, A>::range_type Tensor<T, A>::empty_range_;

  template <typename T, typename AT, typename U, typename AU>
  Tensor<T, AT> operator+(const Tensor<T, AT>& left, const Tensor<U, AU>& right) {
    TA_ASSERT(left.range() == right.range());
    return Tensor<T,AT>(left.range(), left.begin(), right.begin(), std::plus<T>());
  }

  template <typename T, typename AT, typename U, typename AU>
  Tensor<T, AT> operator-(const Tensor<T, AT>& left, const Tensor<U, AU>& right) {
    TA_ASSERT(left.range() == right.range());
    return Tensor<T,AT>(left.range(), left.begin(), right.begin(), std::minus<T>());
  }

  template <typename T, typename AT, typename U, typename AU>
  Tensor<T, AT> operator*(const Tensor<T, AT>& left, const Tensor<U, AU>& right) {
    TA_ASSERT(left.range() == right.range());
    return Tensor<T,AT>(left.range(), left.begin(), right.begin(), std::multiplies<T>());
  }

  template <typename T, typename AT, typename N>
  typename madness::enable_if<TiledArray::detail::is_numeric<N>, Tensor<T, AT> >::type
  operator*(const Tensor<T, AT>& left, N right) {
    return Tensor<T,AT>(left.range(), left.begin(),
        std::bind2nd(TiledArray::detail::multiplies<T, N, T>(),right));
  }

  template <typename T, typename AT, typename N>
  typename madness::enable_if<TiledArray::detail::is_numeric<N>, Tensor<T, AT> >::type
  operator*(N left, const Tensor<T, AT>& right) {
    return Tensor<T,AT>(right.range(), right.begin(),
        std::bind2nd(TiledArray::detail::multiplies<T, N, T>(),left));
  }

  template <typename T, typename A>
  std::ostream& operator<<(std::ostream& os, const Tensor<T, A>& t) {
    os << t.range() << " { ";
    for(typename Tensor<T, A>::const_iterator it = t.begin(); it != t.end(); ++it) {
      os << *it << " ";
    }

    os << "}";

    return os;
  }

} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_H__INCLUDED
