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

#ifndef TILEDARRAY_PERMUTATION_H__INCLUDED
#define TILEDARRAY_PERMUTATION_H__INCLUDED

#include <algorithm>
#include <array>
#include <numeric>

#include <TiledArray/error.h>
#include <TiledArray/tensor/type_traits.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/util/vector.h>
#include <TiledArray/utility.h>

namespace TiledArray {

// Forward declarations
class Permutation;
bool operator==(const Permutation&, const Permutation&);
std::ostream& operator<<(std::ostream&, const Permutation&);
template <typename T, std::size_t N>
inline std::array<T, N> operator*(const Permutation&, const std::array<T, N>&);
template <typename T, std::size_t N>
inline std::array<T, N>& operator*=(std::array<T, N>&, const Permutation&);
template <typename T, typename A>
inline std::vector<T> operator*(const Permutation&, const std::vector<T, A>&);
template <typename T, typename A>
inline std::vector<T, A>& operator*=(std::vector<T, A>&, const Permutation&);
template <typename T>
inline std::vector<T> operator*(const Permutation&,
                                const T* MADNESS_RESTRICT const);

namespace detail {

/// Create a permuted copy of an array

/// \tparam Perm The permutation type
/// \tparam Arg The input array type
/// \tparam Result The output array type
/// \param[in] perm The permutation
/// \param[in] arg The input array to be permuted
/// \param[out] result The output array that will hold the permuted array
template <typename Perm, typename Arg, typename Result,
          typename = std::enable_if_t<is_permutation_v<Perm>>>
inline void permute_array(const Perm& perm, const Arg& arg, Result& result) {
  using std::size;
  TA_ASSERT(size(result) == size(arg));
  const unsigned int n = size(arg);
  for (unsigned int i = 0u; i < n; ++i) {
    const typename Perm::index_type pi = perm[i];
    TA_ASSERT(i < size(arg));
    TA_ASSERT(pi < size(result));
    result[pi] = arg[i];
  }
}
}  // namespace detail

/**
 * \defgroup symmetry Permutation and Permutation Group Symmetry
 * @{
 */

/// Permutation of a sequence of objects indexed by base-0 indices.

/** \warning Unlike TiledArray::symmetry::Permutation, this fixes domain size.

 Permutation class is used as an argument in all permutation operations on
 other objects. Permutations can be applied to sequences of objects:
 \code
   b = p * a; // apply permutation p to sequence a and assign the result to
 sequence b. a *= p;    // apply permutation p (in-place) to sequence a.
 \endcode
 Permutations can also be composed, e.g. multiplied and inverted:
 \code
   p3 = p1 * p2;      // computes product of permutations of p1 and p2
   p1_inv = p1.inv(); // computes inverse of p1
 \endcode

 \note

 \par
 Permutation is internally represented in one-line (image) form, e.g.
 \f$
   \left(
   \begin{tabular}{ccccc}
     0 & 1 & 2 & 3 & 4 \\
     0 & 2 & 3 & 1 & 4
   \end{tabular}
   \right)
 \f$
 is represented in one-line form as \f$ \{0, 2, 3, 1, 4\} \f$. This means
 that 0th element of a sequence is mapped by this permutation into the 0th
 element of the permuted sequence (hence 0 is referred to as a <em>fixed
 point</em> of this permutation; so is 4); similarly, 1st element of a sequence
 is mapped by this permutation into the 2nd element of the permuted sequence
 (hence 2 is referred as the \em image of 1 under the action of this
 Permutation; similarly, 1 is the image of 3, etc.). Set \f$ \{1, 2, 3\} \f$ is
 referred to as \em domain  (or \em support) of this Permutation. Note that (by
 definition) Permutation maps its domain into itself (i.e. it's a bijection).

 \par
 Note that the one-line representation
 is redundant as multiple distinct one-line representations correspond to the
 same <em>compressed form</em>, e.g. \f$ \{0, 2, 3, 1, 4\} \f$ and \f$ \{0, 2,
 3, 1\} \f$ correspond to the same \f$ \{ 1 \to 2, 2 \to 3, 3 \to 1 \} \f$
 compressed form. For an implementation using compressed form, and without fixed
 domain size, see TiledArray::symmetry::Permutation.
*/
class Permutation {
 public:
  typedef Permutation Permutation_;
  typedef unsigned int index_type;
  template <typename T>
  using vector = container::svector<T>;
  typedef vector<index_type>::const_iterator const_iterator;

 private:
  /// Validate input permutation
  /// \return \c false if each element of [first, last) is non-negative, unique
  /// and less than the size of the domain.
  template <typename InIter>
  bool valid_permutation(InIter first, InIter last) {
    bool result = true;
    using diff_type = typename std::iterator_traits<InIter>::difference_type;
    const diff_type n = std::distance(first, last);
    TA_ASSERT(n >= 0);
    for (; first != last; ++first) {
      const diff_type value = *first;
      result = result && value >= 0 && (value < n) &&
               (std::count(first, last, *first) == 1ul);
    }
    return result;
  }

  // Used to select the correct constructor based on input template types
  struct Enabler {};

 protected:
  /// One-line representation of permutation
  vector<index_type> p_;

 public:
  Permutation() = default;  // constructs an invalid Permutation
  Permutation(const Permutation&) = default;
  Permutation(Permutation&&) = default;
  ~Permutation() = default;
  Permutation& operator=(const Permutation&) = default;
  Permutation& operator=(Permutation&& other) = default;

  /// Construct permutation from a range [first,last)

  /// \tparam InIter An input iterator type
  /// \param first The beginning of the iterator range
  /// \param last The end of the iterator range
  /// \throw TiledArray::Exception If the permutation contains any element
  /// that is greater than the size of the permutation or if there are any
  /// duplicate elements.
  template <typename InIter, typename std::enable_if<detail::is_input_iterator<
                                 InIter>::value>::type* = nullptr>
  Permutation(InIter first, InIter last) : p_(first, last) {
    TA_ASSERT(valid_permutation(first, last));
  }

  /// Array constructor

  /// Construct permutation from an Array
  /// \param a The permutation array to be moved
  template <typename Index,
            typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
  explicit Permutation(const Index& a)
      : Permutation(std::cbegin(a), std::cend(a)) {}

  /// std::vector move constructor

  /// Move the content of the vector into this permutation
  /// \param a The permutation array to be moved
  explicit Permutation(vector<index_type>&& a) : p_(std::move(a)) {
    TA_ASSERT(valid_permutation(p_.begin(), p_.end()));
  }

  /// Construct permutation with an initializer list

  /// \tparam Integer an integral type
  /// \param list An initializer list of integers
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  explicit Permutation(std::initializer_list<Integer> list)
      : Permutation(list.begin(), list.end()) {}

  /// Domain size accessor

  /// \return The domain size
  index_type dim() const { return p_.size(); }

  /// Begin element iterator factory function

  /// \return An iterator that points to the beginning of the element range
  const_iterator begin() const { return p_.begin(); }

  /// Begin element iterator factory function

  /// \return An iterator that points to the beginning of the element range
  const_iterator cbegin() const { return p_.cbegin(); }

  /// End element iterator factory function

  /// \return An iterator that points to the end of the element range
  const_iterator end() const { return p_.end(); }

  /// End element iterator factory function

  /// \return An iterator that points to the end of the element range
  const_iterator cend() const { return p_.cend(); }

  /// Element accessor

  /// \param i The element index
  /// \return The i-th element
  index_type operator[](unsigned int i) const { return p_[i]; }

  /// Cycles decomposition

  /// Certain algorithms are more efficient with permutations represented as a
  /// set of cyclic transpositions. This function returns the set of cycles
  /// that represent this permutation. For example, permutation
  /// \f$ \{3, 2, 1, 0 \} \f$ is represented as the following set of cycles:
  /// \c (0,3)(1,2).
  /// The canonical format for the cycles is:
  /// <ul>
  ///  <li> Cycles of length 1 are skipped.
  ///  <li> Each cycle is in order of increasing elements.
  ///  <li> Cycles are in the order of increasing first elements.
  /// </ul>
  /// \return the set of cycles (in canonical format) that represent this
  /// permutation
  vector<vector<index_type>> cycles() const {
    vector<vector<index_type>> result;

    vector<bool> placed_in_cycle(p_.size(), false);

    // 1. for each i compute its orbit
    // 2. if the orbit is longer than 1, sort and add to the list of cycles
    for (index_type i = 0; i != p_.size(); ++i) {
      if (not placed_in_cycle[i]) {
        vector<index_type> cycle(1, i);
        placed_in_cycle[i] = true;

        index_type next_i = p_[i];
        while (next_i != i) {
          cycle.push_back(next_i);
          placed_in_cycle[next_i] = true;
          next_i = p_[next_i];
        }

        if (cycle.size() != 1) {
          std::sort(cycle.begin(), cycle.end());
          result.emplace_back(cycle);
        }

      }  // this i already in a cycle
    }    // loop over i

    return result;
  }

  /// Identity permutation factory function

  /// \param dim The number of dimensions in the
  /// \return An identity permutation for \c dim elements
  static Permutation identity(const unsigned int dim) {
    Permutation result;
    result.p_.reserve(dim);
    for (unsigned int i = 0u; i < dim; ++i) result.p_.emplace_back(i);
    return result;
  }

  /// Identity permutation factory function

  /// \return An identity permutation
  Permutation identity() const { return identity(p_.size()); }

  /// Product of this permutation by \c other

  /// \param other a Permutation
  /// \return \c other * \c this, i.e. this applied first, then other
  Permutation mult(const Permutation& other) const {
    const unsigned int n = p_.size();
    TA_ASSERT(n == other.p_.size());
    Permutation result;
    result.p_.reserve(n);

    for (unsigned int i = 0u; i < n; ++i) {
      const index_type p_i = p_[i];
      const index_type result_i = other.p_[p_i];
      result.p_.emplace_back(result_i);
    }

    return result;
  }

  /// Construct the inverse of this permutation

  /// The inverse of the permutation is defined as \f$ P \times P^{-1} = I \f$,
  /// where \f$ I \f$ is the identity permutation.
  /// \return The inverse of this permutation
  Permutation inv() const {
    const index_type n = p_.size();
    Permutation result;
    result.p_.resize(n, 0ul);
    for (index_type i = 0ul; i < n; ++i) {
      const index_type pi = p_[i];
      result.p_[pi] = i;
    }
    return result;
  }

  /// Raise this permutation to the n-th power

  /// Constructs the permutation \f$ P^n \f$, where \f$ P \f$ is this
  /// permutation.
  /// \param n Exponent value
  /// \return This permutation raised to the n-th power
  Permutation pow(int n) const {
    // Initialize the algorithm inputs
    int power;
    Permutation value;
    if (n < 0) {
      value = inv();
      power = -n;
    } else {
      value = *this;
      power = n;
    }

    Permutation result = identity(p_.size());

    // Compute the power of value with the exponentiation by squaring.
    while (power) {
      if (power & 1) result = result.mult(value);
      value = value.mult(value);
      power >>= 1;
    }

    return result;
  }

  /// Bool conversion

  /// \return \c true if the permutation is not empty, otherwise \c false.
  operator bool() const { return !p_.empty(); }

  /// Not operator

  /// \return \c true if the permutation is empty, otherwise \c false.
  bool operator!() const { return p_.empty(); }

  /// Permutation data accessor

  /// \return A reference to the array of permutation elements
  const auto& data() const { return p_; }

  /// Serialize permutation

  /// MADNESS compatible serialization function
  /// \tparam Archive The serialization archive type
  /// \param[in,out] ar The serialization archive
  template <typename Archive>
  void serialize(Archive& ar) {
    ar& p_;
  }

};  // class Permutation

/// Permutation equality operator

/// \param p1 The left-hand permutation to be compared
/// \param p2 The right-hand permutation to be compared
/// \return \c true if all elements of \c p1 and \c p2 are equal and in the
/// same order, otherwise \c false.
inline bool operator==(const Permutation& p1, const Permutation& p2) {
  return (p1.dim() == p2.dim()) &&
         std::equal(p1.data().begin(), p1.data().end(), p2.data().begin());
}

/// Permutation inequality operator

/// \param p1 The left-hand permutation to be compared
/// \param p2 The right-hand permutation to be compared
/// \return \c true if any element of \c p1 is not equal to that of \c p2,
/// otherwise \c false.
inline bool operator!=(const Permutation& p1, const Permutation& p2) {
  return !operator==(p1, p2);
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
inline Permutation operator-(const Permutation& perm) { return perm.inv(); }

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
inline Permutation operator^(const Permutation& perm, int n) {
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
inline std::array<T, N> operator*(const Permutation& perm,
                                  const std::array<T, N>& a) {
  TA_ASSERT(perm.dim() == a.size());
  std::array<T, N> result;
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
inline std::array<T, N>& operator*=(std::array<T, N>& a,
                                    const Permutation& perm) {
  TA_ASSERT(perm.dim() == a.size());
  const std::array<T, N> temp = a;
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
inline std::vector<T> operator*(const Permutation& perm,
                                const std::vector<T, A>& v) {
  TA_ASSERT(perm.dim() == v.size());
  std::vector<T> result(perm.dim());
  detail::permute_array(perm, v, result);
  return result;
}

/// In-place permute a \c std::vector

/// \tparam T The element type of the vector
/// \tparam A The allocator type of the vector
/// \param[out] v The vector to be permuted
/// \param[in] perm The permutation
/// \return A reference to \c v
/// \throw TiledArray::Exception When the dimension of the permutation is not
/// equal to the size of \c v.
template <typename T, typename A>
inline std::vector<T, A>& operator*=(std::vector<T, A>& v,
                                     const Permutation& perm) {
  const std::vector<T, A> temp = v;
  detail::permute_array(perm, temp, v);
  return v;
}

/// permute a \c boost::container::small_vector<T>

/// \tparam T The element type of the vector
/// \tparam N The max static size of the vector
/// \param perm The permutation
/// \param v The vector to be permuted
/// \return A permuted copy of \c v
/// \throw TiledArray::Exception When the dimension of the permutation is not
/// equal to the size of \c v.
template <typename T, std::size_t N>
inline boost::container::small_vector<T, N> operator*(
    const Permutation& perm, const boost::container::small_vector<T, N>& v) {
  TA_ASSERT(perm.dim() == v.size());
  boost::container::small_vector<T, N> result(perm.dim());
  detail::permute_array(perm, v, result);
  return result;
}

/// In-place permute a \c boost::container::small_vector

/// \tparam T The element type of the vector
/// \tparam N The max static size of the vector
/// \param[out] v The vector to be permuted
/// \param[in] perm The permutation
/// \return A reference to \c v
/// \throw TiledArray::Exception When the dimension of the permutation is not
/// equal to the size of \c v.
template <typename T, std::size_t N>
inline boost::container::small_vector<T, N>& operator*=(
    boost::container::small_vector<T, N>& v, const Permutation& perm) {
  const boost::container::small_vector<T, N> temp = v;
  detail::permute_array(perm, temp, v);
  return v;
}

/// Permute a memory buffer

/// \tparam T The element type of the memory buffer
/// \param perm The permutation
/// \param ptr A pointer to the memory buffer to be permuted
/// \return A permuted copy of the memory buffer as a \c std::vector
template <typename T>
inline std::vector<T> operator*(const Permutation& perm,
                                const T* MADNESS_RESTRICT const ptr) {
  const unsigned int n = perm.dim();
  std::vector<T> result(n);
  for (unsigned int i = 0u; i < n; ++i) {
    const typename Permutation::index_type perm_i = perm[i];
    const T ptr_i = ptr[i];
    result[perm_i] = ptr_i;
  }
  return result;
}

///////////////////////////////////

/// Permutation of a bipartite set
class BipartitePermutation : public Permutation {
 public:
  BipartitePermutation() = default;
  BipartitePermutation(const BipartitePermutation&) = default;
  BipartitePermutation(BipartitePermutation&&) = default;
  ~BipartitePermutation() = default;
  BipartitePermutation& operator=(const BipartitePermutation&) = default;
  BipartitePermutation& operator=(BipartitePermutation&& other) = default;

  // clang-format off
  /// Construct permutation from a range [first,last)

  /// \tparam InIter An input iterator type
  /// \param first The beginning of the iterator range
  /// \param last The end of the iterator range
  /// \param second_partition_size the size of the second partition; the size of the first is then \c std::distance(first,last)-second_partition_size
  /// \throw TiledArray::Exception If the permutation contains any element
  /// that is greater than the size of the permutation or if there are any
  /// duplicate elements.
  // clang-format on
  template <typename InIter, typename std::enable_if<detail::is_input_iterator<
                                 InIter>::value>::type* = nullptr>
  BipartitePermutation(InIter first, InIter last,
                       index_type second_partition_size = 0)
      : Permutation(first, last), second_size_(second_partition_size) {}

  // clang-format off
  /// Array constructor

  /// Construct permutation from an Array
  /// \param a The permutation array to be moved
  /// \param second_partition_size the size of the second partition; the size of the first is then \c std::distance(first,last)-second_partition_size
  // clang-format on
  template <typename Index,
            typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
  explicit BipartitePermutation(const Index& a,
                                index_type second_partition_size = 0)
      : BipartitePermutation(std::cbegin(a), std::cend(a),
                             second_partition_size) {}

  // clang-format off
  /// std::vector move constructor

  /// Move the content of the vector into this permutation
  /// \param a The permutation array to be moved
  /// \param second_partition_size the size of the second partition; the size of the first is then \c std::distance(first,last)-second_partition_size
  // clang-format on
  explicit BipartitePermutation(vector<index_type>&& a,
                                index_type second_partition_size = 0)
      : Permutation(std::move(a)), second_size_(second_partition_size) {}

  /// \return reference to the first partition
  /// \note the partition object is computed on the first invocation and
  /// memoized
  const Permutation& first() const {
    first_ = Permutation{this->begin(), this->begin() + first_size()};
    return first_;
  }
  /// \return reference to the second partition
  /// \note the partition object is computed on the first invocation and
  /// memoized
  const Permutation& second() const {
    const auto n_first = first_size();
    vector<index_type> temp(second_size());
    for (auto i = n_first; i < dim(); ++i) temp[i - n_first] = p_[i] - n_first;
    second_ = Permutation(temp.begin(), temp.end());
    return second_;
  }

  /// \return the size of the first partition
  index_type first_size() const { return this->dim() - second_size_; }

  /// \return the size of the second partition
  index_type second_size() const { return second_size_; }

  /// Serialize permutation

  /// MADNESS compatible serialization function
  /// \tparam Archive The serialization archive type
  /// \param[in,out] ar The serialization archive
  template <typename Archive>
  void serialize(Archive& ar) {
    ar& static_cast<Permutation&>(*this) & second_size_;
    if constexpr (madness::archive::is_input_archive<Archive>::value) {
      first_ = {};
      second_ = {};
    }
  }

 private:
  /// The size of the second partition
  index_type second_size_ = 0;

  mutable Permutation first_;
  mutable Permutation second_;
};

inline auto inner(const Permutation& p) {
  abort();
  return Permutation{};
}

inline const auto& outer(const Permutation& p) { return p; }

inline auto inner_dim(const Permutation& p) { return 0; }

inline auto outer_dim(const Permutation& p) { return p.dim(); }

inline auto inner(const BipartitePermutation& p) { return p.second(); }

inline auto outer(const BipartitePermutation& p) { return p.first(); }

inline auto inner_dim(const BipartitePermutation& p) { return p.second_size(); }

inline auto outer_dim(const BipartitePermutation& p) { return p.first_size(); }

}  // namespace TiledArray

#endif  // TILEDARRAY_PERMUTATION_H__INCLUED
