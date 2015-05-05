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

  // weirdly necessary forward declarations
  class Permutation;
  bool operator==(const Permutation&, const Permutation&);
  std::ostream& operator<<(std::ostream&, const Permutation&);

  namespace detail {

    /// Place holder object to represent a no permutation operation.
    struct NoPermutation {
      const NoPermutation& operator-() const { return *this; }
      template <typename Archive>
      void serialize(Archive&) { }
    };

    /// Copies an iterator range into an array type container.

    /// Permutes iterator range  \c [first_o, \c last_o) base on the permutation
    /// \c [first_p, \c last_p) and places it in \c result. The result object
    /// must define operator[](std::size_t).
    /// \arg \c [first_p, \c last_p) is an iterator range to the permutation
    /// \arg \c [irst_o is an input iterator to the beginning of the original array
    /// \arg \c result is a random access iterator that will contain the resulting permuted array
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

  /// Permutation

  /// Permutation class is used as an argument in all permutation operations on
  /// other objects. Permutations are performed with the following syntax:
  ///
  ///   b = p ^ a; // assign permeation of a into b given the permutation p.
  ///   a ^= p;    // permute a given the permutation p.
  class Permutation {
  private:

    // Used to select the correct constructor based on input template types
    struct Enabler { };

  public:
    typedef Permutation Permutation_;
    typedef unsigned int index_type;
    typedef std::vector<index_type> Array;
    typedef Array::const_iterator const_iterator;

    unsigned int dim() const { return p_.size(); }

    /// Default constructor (defines a null permutation)
    Permutation() : p_() { }

    /// Construct permutation from an iterator

    /// \tparam InIter An input iterator type
    /// \param first The beginning of the iterator range
    /// \param last The end of the iterator range
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
    explicit Permutation(const std::vector<index_type>& a) : p_(a) {
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    /// Array move constructor

    /// Move the content of the array into this permutation
    /// \param a The permutation array to be moved
    explicit Permutation(Array&& a) :
        p_(std::forward<Array>(a))
    {
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    /// Copy constructor

    /// \param other The permutation to be copied
    Permutation(const Permutation& other) : p_(other.p_) { }

    /// Move constructor

    /// \param other The permuatation to be moved
    Permutation(Permutation&& other) : p_(std::move(other.p_)) { }

    /// Construct permuation with an initializer list

    /// \tparam I An integral type
    /// \param list An initializer list of integers
    template <typename I,
        typename std::enable_if<std::is_integral<I>::value>::type* = nullptr>
    explicit Permutation(std::initializer_list<I> list) :
        p_(list.begin(), list.end())
    {
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    ~Permutation() {}

    const_iterator begin() const { return p_.begin(); }
    const_iterator end() const { return p_.end(); }

    const index_type& operator[](unsigned int i) const {
      return p_[i];
    }

    Permutation& operator=(const Permutation& other) { p_ = other.p_; return *this; }

    /// return *this ^ other
    Permutation& operator^=(const Permutation& other) {
      TA_ASSERT(other.p_.size() == p_.size());

      Array result(p_.size());
      detail::permute_array(other, p_, result);
      std::swap(result, p_);
      return *this;
    }

    /// Returns the reverse permutation and will satisfy the following conditions.
    /// given c2 = p ^ c1
    /// c1 == ((-p) ^ c2);
    Permutation operator -() const {
      const std::size_t n = p_.size();
      Permutation result;
      result.p_.resize(n, 0ul);
      for(std::size_t i = 0ul; i < n; ++i) {
        const std::size_t pi = p_[i];
        result.p_[pi] = i;
      }
      return result;
    }

    /// Bool conversion

    /// \return \c true if the permutation is not empty, otherwise \c false.
    operator bool() const { return ! p_.empty(); }

    /// Not operator

    /// \return \c true if the permutation is empty, otherwise \c false.
    bool operator!() const { return p_.empty(); }

    /// Return a reference to the array that represents the permutation.
    const Array& data() const { return p_; }

    template <typename Archive>
    void serialize(Archive& ar) {
      ar & p_;
    }

  private:

    static Permutation make_unit_permutation(std::size_t n) {
      Permutation::Array temp(n);
      for(unsigned int d = 0; d < n; ++d) temp[d] = d;
      return Permutation(std::move(temp));
    }

    // return false if this is not a valid permutation
    template <typename InIter>
    bool valid_(InIter first, InIter last) {
      Array count(std::distance(first, last));
      std::fill(count.begin(), count.end(), 0);
      for(; first != last; ++first) {
        const index_type& i = *first;
        if(count[i] > 0) return false;
        ++count[i];
      }
      return true;
    }

    Array p_;
  };

  inline bool operator==(const Permutation& p1, const Permutation& p2) {
    return (p1.dim() == p2.dim()) && std::equal(p1.data().begin(), p1.data().end(), p2.data().begin());
  }

  inline bool operator!=(const Permutation& p1, const Permutation& p2) {
    return ! operator==(p1, p2);
  }

  inline std::ostream& operator<<(std::ostream& output, const Permutation& p) {
    std::size_t n = p.dim();
    output << "{";
    for (unsigned int dim = 0; dim < n - 1; ++dim)
      output << dim << "->" << p.data()[dim] << ", ";
    output << n - 1 << "->" << p.data()[n - 1] << "}";
    return output;
  }

  /// permute a std::array
  template <typename T, std::size_t N>
  inline std::array<T,N> operator^(const Permutation& perm, const std::array<T, N>& orig) {
    TA_ASSERT(perm.dim() == orig.size());
    std::array<T,N> result;
    detail::permute_array(perm, orig, result);
    return result;
  }

  /// permute a std::vector<T>
  template <typename T, typename A>
  inline std::vector<T> operator^(const Permutation& perm, const std::vector<T, A>& orig) {
    std::vector<T> result(perm.dim());
    detail::permute_array(perm, orig, result);
    return result;
  }

  inline Permutation operator ^(const Permutation& perm, const Permutation& p) {
    TA_ASSERT(perm.dim() == p.dim());
    return Permutation(perm ^ p.data());
  }

  namespace detail {
    /// permute a std::array
    template <typename T, std::size_t N>
    inline const std::array<T,N>& operator^(const NoPermutation&, const std::array<T, N>& orig) {
      return orig;
    }

    /// permute a std::vector<T>
    template <typename T, typename A>
    inline const std::vector<T, A>& operator^(const NoPermutation&, const std::vector<T, A>& orig) {
      return orig;
    }

    inline const Permutation& operator ^(const NoPermutation&, const Permutation& p) {
      return p;
    }

  } // namespace detail

  template <typename T, std::size_t N>
  inline std::array<T,N>& operator ^=(std::array<T,N>& a, const Permutation& perm) {
    const std::array<T,N> temp = a;
    detail::permute_array(perm, temp, a);
    return a;
  }

  template <typename T, typename A>
  inline std::vector<T, A>& operator^=(std::vector<T, A>& orig, const Permutation& perm) {
    const std::vector<T, A> temp = orig;
    detail::permute_array(perm, temp, orig);
    return orig;
  }

} // namespace TiledArray

#endif // TILEDARRAY_PERMUTATION_H__INCLUED
