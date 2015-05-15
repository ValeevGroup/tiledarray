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

#ifndef TILEDARRAY_PERMUTATION_H__INCLUED
#define TILEDARRAY_PERMUTATION_H__INCLUED

#include <TiledArray/error.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/utility.h>
#include <array>

namespace TiledArray {

  // Forward declarations
  class Permutation;
  bool operator==(const Permutation&, const Permutation&);
  std::ostream& operator<<(std::ostream&, const Permutation&);
  template <typename T, std::size_t N>
  inline std::array<T,N> operator*(const Permutation&, const std::array<T, N>&);
  template <typename T, std::size_t N>
  inline std::array<T,N>& operator*=(std::array<T,N>&, const Permutation&);
  template <typename T, typename A>
  inline std::vector<T> operator*(const Permutation&, const std::vector<T, A>&);
  template <typename T, typename A>
  inline std::vector<T, A>& operator*=(std::vector<T, A>&, const Permutation&);
  template <typename T>
  inline std::vector<T> operator*(const Permutation&, const T* restrict const);

  namespace detail {

    /// Create a permuted copy of an array

    /// \tparam Perm The permutation type
    /// \tparam Arg The input array type
    /// \tparam Result The output array type
    /// \param[in] perm The permutation
    /// \param[in] arg The input array to be permuted
    /// \param[out] result The output array that will hold the permuted array
    template <typename Perm, typename Arg, typename Result>
    inline void permute_array(const Perm& perm, const Arg& arg, Result& result) {
      TA_ASSERT(perm.dim() == size(arg));
      TA_ASSERT(perm.dim() == size(result));

      const unsigned int n = perm.dim();
      for(unsigned int i = 0u; i < n; ++i) {
        const typename Perm::index_type pi = perm[i];
        result[pi] = arg[i];
      }
    }
  } // namespace detail


  /**
   * \defgroup symmetry Permutation and Permutation Group Symmetry
   * @{
   */

  /// Permutation

  /// Permutation class is used as an argument in all permutation operations on
  /// other objects. Permutations are performed with the following syntax:
  /// \code
  ///   b = p * a; // assign permeation of a into b given the permutation p.
  ///   a *= p;    // permute a given the permutation p.
  /// \endcode
  class Permutation {
  public:
    typedef Permutation Permutation_;
    typedef unsigned int index_type;
    typedef std::vector<index_type> Array;
    typedef std::vector<index_type>::const_iterator const_iterator;

  private:

    /// The permutation stored in an "image form," where the permutation
    ///   ( 0 1 2 3 )
    ///   ( 3 2 1 0 )
    /// is stored as { 3, 2, 1, 0 }
    std::vector<index_type> p_;

    /// Validate input permutation data

    /// \return \c false if each element of [first, last) is unique and less
    /// than the size of the list.
    template <typename InIter>
    bool valid_(InIter first, InIter last) {
      bool result = true;
      const unsigned int n = std::distance(first, last);
      for(; first != last; ++first) {
        const unsigned int value = *first;
        result = result && (value < n) && (std::count(first, last, *first) == 1ul);
      }
      return result;
    }

    // Used to select the correct constructor based on input template types
    struct Enabler { };

  public:

    Permutation() = default;
    Permutation(const Permutation&) = default;
    Permutation(Permutation&&) = default;
    ~Permutation() = default;
    Permutation& operator=(const Permutation&) = default;
    Permutation& operator=(Permutation&& other) = default;

    /// Construct permutation from an iterator

    /// \tparam InIter An input iterator type
    /// \param first The beginning of the iterator range
    /// \param last The end of the iterator range
    /// \throw TiledArray::Exception If the permutation contains any element
    /// that is greater than the size of the permutation or if there are any
    /// duplicate elements.
    template <typename InIter,
        typename std::enable_if<detail::is_input_iterator<InIter>::value>::type* = nullptr>
    Permutation(InIter first, InIter last) :
        p_(first, last)
    {
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    /// Array constructor

    /// Construct permutation from an Array
    /// \param a The permutation array to be moved
    explicit Permutation(const std::vector<index_type>& a) :
        p_(a)
    {
      TA_ASSERT( valid_(a.begin(), a.end()) );
    }

    /// std::vector move constructor

    /// Move the content of the std::vector into this permutation
    /// \param a The permutation array to be moved
    explicit Permutation(std::vector<index_type>&& a) :
        p_(std::move(a))
    {
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    /// Construct permutation with an initializer list

    /// \param list An initializer list of integers
    explicit Permutation(std::initializer_list<index_type> list) :
        Permutation(list.begin(), list.end())
    { }

    /// Permutation dimension accessor

    /// \return The number of elements in the permutation
    unsigned int dim() const { return p_.size(); }

    /// Begin element iterator factory function

    /// \return An iterator that points to the beginning of the element range
    const_iterator begin() const { return p_.begin(); }


    /// End element iterator factory function

    /// \return An iterator that points to the end of the element range
    const_iterator end() const { return p_.end(); }

    /// Element accessor

    /// \param i The element index
    /// \return The i-th element
    index_type operator[](unsigned int i) const { return p_[i]; }


    /// Identity permutation factory function

    /// \param dim The number of dimensions in the
    /// \return An identity permutation for \c dim elements
    static Permutation identity(const unsigned int dim) {
      std::vector<index_type> result;
      result.reserve(dim);
      for(unsigned int i = 0u; i < dim; ++i)
        result.emplace_back(i);
      return Permutation(std::move(result));
    }

    /// Identity permutation factory function

    /// \return An identity permutation
    Permutation identity() const { return identity(p_.size()); }

    /// Multiplication of this permutation with \c other

    /// \param other The right-hand argument
    /// \return The product of this permutation multiplied by \c other
    Permutation mult(const Permutation& other) const {
      TA_ASSERT(p_.size() == other.p_.size());
      std::vector<index_type> temp(p_.size());
      detail::permute_array(*this, other.p_, temp);
      return Permutation(std::move(temp));
    }

    /// Construct the inverse of this permutation

    /// The inverse of the permutation is defined as \f$ P \times P^{-1} = I \f$,
    /// where \f$ I \f$ is the identity permutation.
    /// \return The inverse of this permutation
    Permutation inv() const {
      const std::size_t n = p_.size();
      std::vector<index_type> result;
      result.resize(n, 0ul);
      for(std::size_t i = 0ul; i < n; ++i) {
        const std::size_t pi = p_[i];
        result[pi] = i;
      }
      return Permutation(std::move(result));
    }

  private:

    static Permutation pow(Permutation perm, const int n) {
      if(n > 1) {
        if(n & int(1)) {// if n is odd then
          perm = pow(perm.mult(perm), (n - 1) >> 1).mult(perm);
        } else {
          perm = pow(perm.mult(perm), n >> 1);
        }
      }

      return perm;
    }

  public:

    /// Raise this permutation to the n-th power

    /// Constructs the permutation \f$ P^n \f$, where \f$ P \f$ is this
    /// permutation.
    /// \param n Exponent value
    /// \return This permutation raised to the n-th power
    Permutation pow(int n) const {
      Permutation result;

      if(n < 0)
        result = pow(inv(), -n);
      else if(n == 0)
        result = identity();
      else
        result = pow(*this, n);

      return result;
    }

    /// Bool conversion

    /// \return \c true if the permutation is not empty, otherwise \c false.
    operator bool() const { return ! p_.empty(); }

    /// Not operator

    /// \return \c true if the permutation is empty, otherwise \c false.
    bool operator!() const { return p_.empty(); }

    /// Permutation data accessor

    /// \return A reference to the array of permutation elements
    const std::vector<index_type>& data() const { return p_; }

    /// Serialize permutation

    /// MADNESS compatible serialization function
    /// \tparam Archive The serialization archive type
    /// \param[in,out] ar The serialization archive
    template <typename Archive>
    void serialize(Archive& ar) {
      ar & p_;
    }

  }; // class Permutation

  /// Permutation equality operator

  /// \param p1 The left-hand permutation to be compared
  /// \param p2 The right-hand permutation to be compared
  /// \return \c true if all elements of \c p1 and \c p2 are equal and in the
  /// same order, otherwise \c false.
  inline bool operator==(const Permutation& p1, const Permutation& p2) {
    return (p1.dim() == p2.dim())
        && std::equal(p1.data().begin(), p1.data().end(), p2.data().begin());
  }

  /// Permutation inequality operator

  /// \param p1 The left-hand permutation to be compared
  /// \param p2 The right-hand permutation to be compared
  /// \return \c true if any element of \c p1 is not equal to that of \c p2,
  /// otherwise \c false.
  inline bool operator!=(const Permutation& p1, const Permutation& p2) {
    return ! operator==(p1, p2);
  }

  /// Permutation less-than operator

  /// \param p1 The left-hand permutation to be compared
  /// \param p2 The right-hand permutation to be compared
  /// \return \c true if the elements of \c p1 are lexicographically less than
  /// that of \c p2, otherwise \c false.
  inline bool operator<(const Permutation& p1, const Permutation& p2) {
    return std::lexicographical_compare(p1.data().begin(), p1.data().end(),
        p2.data().begin(), p2.data().end());
  }

  /// Add permutation to an output stream

  /// \param[out] output The output stream
  /// \param[in] p The permutation to be added to the output stream
  /// \return The output stream
  inline std::ostream& operator<<(std::ostream& output, const Permutation& p) {
    std::size_t n = p.dim();
    output << "{";
    for (unsigned int dim = 0; dim < n - 1; ++dim)
      output << dim << "->" << p.data()[dim] << ", ";
    output << n - 1 << "->" << p.data()[n - 1] << "}";
    return output;
  }


  /// Inverse permutation operator

  /// \param perm The permutation to be inverted
  /// \return \c perm.inverse()
  inline Permutation operator -(const Permutation& perm) {
    return perm.inv();
  }

  /// Permutation multiplication operator

  /// \param p1 The left-hand permutation
  /// \param p2 The right-hand permutation
  /// \return The product of p1 and p2 (which is the permutation of \c p2
  /// by \c p1).
  inline Permutation operator*(const Permutation& p1, const Permutation& p2) {
    return p1.mult(p2);
  }


  /// return *this ^ other
  inline Permutation& operator*=(Permutation& p1, const Permutation& p2) {
    return (p1 = p1 * p2);
  }

  /// Raise \c perm to the n-th power

  /// Constructs the permutation \f$ P^n \f$, where \f$ P \f$ is the
  /// permutation \c perm.
  /// \param perm The base permutation
  /// \param n Exponent value
  /// \return This permutation raised to the n-th power
  inline Permutation operator^(const Permutation perm, int n) {
    return perm.pow(n);
  }

  /** @}*/


  /// Permute a \c std::array

  /// \tparam T The element type of the array
  /// \tparam N The size of the array
  /// \param perm The permutation
  /// \param a The array to be permuted
  /// \return A permuted copy of \c a
  /// \throw TiledArray::Exception When the dimension of the permutation is not
  /// equal to the size of \c a.
  template <typename T, std::size_t N>
  inline std::array<T,N> operator*(const Permutation& perm, const std::array<T, N>& a) {
    TA_ASSERT(perm.dim() == a.size());
    std::array<T,N> result;
    detail::permute_array(perm, a, result);
    return result;
  }

  /// In-place permute a \c std::array

  /// \tparam T The element type of the array
  /// \tparam N The size of the array
  /// \param[out] a The array to be permuted
  /// \param[in] perm The permutation
  /// \return A reference to \c a
  /// \throw TiledArray::Exception When the dimension of the permutation is not
  /// equal to the size of \c a.
  template <typename T, std::size_t N>
  inline std::array<T,N>& operator*=(std::array<T,N>& a, const Permutation& perm) {
    TA_ASSERT(perm.dim() == a.size());
    const std::array<T,N> temp = a;
    detail::permute_array(perm, temp, a);
    return a;
  }

  /// permute a \c std::vector<T>

  /// \tparam T The element type of the vector
  /// \tparam A The allocator type of the vector
  /// \param perm The permutation
  /// \param v The vector to be permuted
  /// \return A permuted copy of \c v
  /// \throw TiledArray::Exception When the dimension of the permutation is not
  /// equal to the size of \c v.
  template <typename T, typename A>
  inline std::vector<T> operator*(const Permutation& perm, const std::vector<T, A>& v) {
    TA_ASSERT(perm.dim() == v.size());
    std::vector<T> result(perm.dim());
    detail::permute_array(perm, v, result);
    return result;
  }

  /// In-place permute a \c std::array

  /// \tparam T The element type of the vector
  /// \tparam A The allocator type of the vector
  /// \param[out] v The vector to be permuted
  /// \param[in] perm The permutation
  /// \return A reference to \c v
  /// \throw TiledArray::Exception When the dimension of the permutation is not
  /// equal to the size of \c v.
  template <typename T, typename A>
  inline std::vector<T, A>& operator*=(std::vector<T, A>& v, const Permutation& perm) {
    const std::vector<T, A> temp = v;
    detail::permute_array(perm, temp, v);
    return v;
  }

  /// Permute a memory buffer

  /// \tparam T The element type of the memory buffer
  /// \param perm The permutation
  /// \param ptr A pointer to the memory buffer to be permuted
  /// \return A permuted copy of the memory buffer as a \c std::vector
  template <typename T>
  inline std::vector<T> operator*(const Permutation& perm, const T* restrict const ptr) {
    const unsigned int n = perm.dim();
    std::vector<T> result(n);
    for(unsigned int i = 0u; i < n; ++i) {
      const typename Permutation::index_type perm_i = perm[i];
      const T ptr_i = ptr[i];
      result[perm_i] = ptr_i;
    }
    return result;
  }

} // namespace TiledArray

#endif // TILEDARRAY_PERMUTATION_H__INCLUED
