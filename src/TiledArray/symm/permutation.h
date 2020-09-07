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
 *  Edward Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  permutation.h
 *  May 13, 2015
 *
 */

#ifndef TILEDARRAY_SYMM_PERMUTATION_H__INCLUDED
#define TILEDARRAY_SYMM_PERMUTATION_H__INCLUDED

#include <array>
#include <map>
#include <set>
#include <vector>

#include <algorithm>

#include <TiledArray/error.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/util/vector.h>

namespace TiledArray {

/**
 * \addtogroup symmetry
 * @{
 */

namespace symmetry {

/// Permutation of a sequence of objects indexed by base-0 indices.

/** \warning Unlike TiledArray::Permutation, this does not fix domain size.

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
 Unlike TiledArray::Permutation, which is internally represented in one-line
 form, TiledArray::symmetry::Permutation is internally represented in compressed
 two-line form. E.g. the following permutation in Cauchy's two-line form, \f$
   \left(
   \begin{tabular}{ccccc}
     0 & 1 & 2 & 3 & 4 \\
     0 & 2 & 3 & 1 & 4
   \end{tabular}
   \right)
 \f$
 , is represented in compressed form as \f$ \{ 1 \to 2, 2 \to 3, 3 \to 1 \} \f$
 . This means that 0th element of a sequence is mapped by this permutation into
 the 0th element of the permuted sequence (hence 0 is referred to as a <em>fixed
 point</em> of this permutation; so is 4); similarly, 1st element of a sequence
 is mapped by this permutation into the 2nd element of the permuted sequence
 (hence 2 is referred as the \em image of 1 under the action of this
 Permutation; similarly, 1 is the image of 3, etc.). Set \f$ \{1, 2, 3\} \f$ is
 referred to as \em domain  (or \em support) of this Permutation. Note that (by
 definition) Permutation maps its domain into itself (i.e. it's a bijection).

 \par
 As a reminder, permutation
 \f$
   \left(
   \begin{tabular}{ccccc}
     0 & 1 & 2 & 3 & 4 \\
     0 & 2 & 3 & 1 & 4
   \end{tabular}
   \right)
 \f$
 is represented in one-line form as \f$ \{0, 2, 3, 1, 4\} \f$. Note that the
 one-line representation is redundant as multiple distinct one-line
 representations correspond to the same compressed form, e.g. \f$ \{0, 2, 3, 1,
 4\} \f$ and \f$ \{0, 2, 3, 1\} \f$ correspond to the same \f$ \{ 1 \to 2, 2 \to
 3, 3 \to 1 \} \f$ compressed form.

 \par
 Another non-redundant representation of Permutation is as a set of cycles. For
 example, permutation \f$ \{0 \to 3, 1 \to 2, 2 \to 1, 0 \to 3 \} \f$ is
 represented uniquely as the following set of cycles: (0,3)(1,2). The canonical
 format for the cycle decomposition used by Permutation class is defined as
 follows: <ul> <li> Cycles of length 1 are skipped. <li> Each cycle is in order
 of increasing elements. <li> Cycles are in the order of increasing first
 elements.
 </ul>
 Cycle representation is convenient for some operations, but is less efficient
 for others. Thus cycle representation can be computed on request, but
 internally the compressed form is used.
*/
class Permutation {
 public:
  typedef Permutation Permutation_;
  typedef unsigned int index_type;
  template <typename T>
  using vector = container::svector<T>;
  typedef std::map<index_type, index_type> Map;
  typedef Map::const_iterator const_iterator;

 private:
  /// Two-line representation of permutation
  Map p_;

  static std::ostream& print_map(std::ostream& output, const Map& p) {
    for (auto i = p.cbegin(); i != p.cend();) {
      output << i->first << "->" << i->second;
      if (++i != p.cend()) output << ", ";
    }
    return output;
  }
  friend inline std::ostream& operator<<(std::ostream& output,
                                         const Permutation& p);

  /// Validate permutation specified in one-line form as an iterator range
  /// \return \c true if each element of \c [first,last) is non-negative and
  /// unique
  template <typename InIter>
  static bool valid_permutation(InIter first, InIter last) {
    bool result = true;
    for (; first != last; ++first) {
      const auto value = *first;
      result = result && value >= 0 && (std::count(first, last, *first) == 1ul);
    }
    return result;
  }

  /// Validate permutation specified in compressed two-line form as an
  /// index->index associative container \note Map can be std::map<index,index>,
  /// std::unordered_map<index,index>, or any similar container.
  template <typename Map>
  static bool valid_permutation(const Map& input) {
    std::set<index_type> keys;
    std::set<index_type> values;
    for (const auto& e : input) {
      const auto& key = e.first;
      const auto& value = e.second;
      if (keys.find(key) == keys.end())
        keys.insert(key);
      else {
        // print_map(std::cout, input);
        return false;  // key is found more than once
      }
      if (values.find(value) == values.end())
        values.insert(value);
      else {
        // print_map(std::cout, input);
        return false;  // value is found more than once
      }
    }
    return keys == values;  // must map domain into itself
  }

  // Used to select the correct constructor based on input template types
  struct Enabler {};

 public:
  Permutation() = default;  // makes an identity permutation
  Permutation(const Permutation&) = default;
  Permutation(Permutation&&) = default;
  ~Permutation() = default;
  Permutation& operator=(const Permutation&) = default;
  Permutation& operator=(Permutation&& other) = default;

  /// Construct permutation using its 1-line form given by range [first,last)

  /// \tparam InIter An input iterator type
  /// \param first The beginning of the iterator range
  /// \param last The end of the iterator range
  /// \throw TiledArray::Exception if invalid input is given. \sa
  /// valid_permutation(first,last)
  template <typename InIter,
            typename std::enable_if<TiledArray::detail::is_input_iterator<
                InIter>::value>::type* = nullptr>
  Permutation(InIter first, InIter last) {
    TA_ASSERT(valid_permutation(first, last));
    size_t i = 0;
    for (auto e = first; e != last; ++e, ++i) {
      auto p_i = *e;
      if (i != p_i) p_[i] = p_i;
    }
  }

  /// Construct permutation using 1-line form given as an integral range

  /// \tparam Index An integral range type
  /// \param a range that specifies permutation in 1-line form
  template <typename Index, typename std::enable_if<
                                TiledArray::detail::is_integral_range_v<Index>,
                                bool>::type* = nullptr>
  explicit Permutation(Index&& a) : Permutation(begin(a), end(a)) {}

  /// Construct permutation with an initializer list

  /// \tparam Integer an integral type
  /// \param list An initializer list of integers
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  explicit Permutation(std::initializer_list<Integer> list)
      : Permutation(list.begin(), list.end()) {}

  /// Construct permutation using its compressed 2-line form given by std::map

  /// \param p the map
  Permutation(Map p) : p_(std::move(p)) { TA_ASSERT(valid_permutation(p_)); }

  /// Permutation domain size

  /// \return The number of elements in the domain of permutation
  unsigned int domain_size() const { return p_.size(); }

  /// @name Iterator accessors
  ///
  /// Permutation iterators dereference into \c
  /// std::pair<index_type,index_type>, where \c first is the domain index, \c
  /// second is its image E.g. \c Permutation::begin() of \f$ \{1->3, 2->1,
  /// 3->2\} \f$ dereferences to \c std::pair {1,3} .
  ///
  /// @{

  /// Begin element iterator factory function

  /// \return An iterator that points to the beginning of the range
  const_iterator begin() const { return p_.begin(); }

  /// Begin element iterator factory function

  /// \return An iterator that points to the beginning of the range
  const_iterator cbegin() const { return p_.cbegin(); }

  /// End element iterator factory function

  /// \return An iterator that points to the end of the range
  const_iterator end() const { return p_.end(); }

  /// End element iterator factory function

  /// \return An iterator that points to the end of the range
  const_iterator cend() const { return p_.cend(); }

  /// @}

  /// Computes image of an element under this permutation

  /// \param e input index
  /// \return the image of element \c e; if \c e is in the domain of this
  /// permutation, returns its image, otherwise returns \c e
  index_type operator[](index_type e) const {
    auto e_image_iter = p_.find(e);
    if (e_image_iter != p_.end())
      return e_image_iter->second;
    else
      return e;
  }

  /// Computes the domain of this permutation

  /// \tparam Set a container type in which the result will be returned
  /// \return the domain of this permutation, as a Set
  template <typename Set>
  Set domain() const {
    Set result;
    for (const auto& e : this->p_) {
      result.insert(result.begin(), e.first);
    }
    // std::sort(result.begin(), result.end());
    return result;
  }

  /// Test if an index is in the domain of this permutation

  /// \tparam Integer an integer type
  /// \param i an index whose presence in domain is tested
  /// \return \c true , if \c i is in the domain, \c false otherwise
  template <typename Integer,
            typename std::enable_if<std::is_integral<Integer>::value>::type* =
                nullptr>
  bool is_in_domain(Integer i) const {
    return p_.find(i) != p_.end();
  }

  /// Cycles decomposition

  /// Certain algorithms are more efficient with permutations represented as a
  /// set of cyclic transpositions. This function returns the set of cycles that
  /// represent this permutation. For example, permutation \f$ \{3, 2, 1, 0 \}
  /// \f$ is represented as the following set of cycles: (0,3)(1,2). The
  /// canonical format for the cycles is: <ul>
  ///  <li> Cycles of length 1 are skipped.
  ///  <li> Each cycle is in order of increasing elements.
  ///  <li> Cycles are in the order of increasing first elements.
  /// </ul>
  /// \return the set of cycles (in canonical format) that represent this
  /// permutation
  vector<vector<index_type>> cycles() const {
    vector<vector<index_type>> result;

    std::set<index_type> placed_in_cycle;

    // safe to use non-const operator[] due to properties of Permutation
    // (domain==image)
    auto& p_nonconst_ = const_cast<Map&>(p_);

    // 1. for each i compute its orbit
    // 2. if the orbit is longer than 1, sort and add to the list of cycles
    for (const auto& e : p_) {
      auto i = e.first;
      if (placed_in_cycle.find(i) ==
          placed_in_cycle.end()) {  // not in a cycle yet?
        vector<index_type> cycle(1, i);
        placed_in_cycle.insert(i);

        index_type next_i = p_nonconst_[i];
        while (next_i != i) {
          cycle.push_back(next_i);
          placed_in_cycle.insert(next_i);
          next_i = p_nonconst_[next_i];
        }

        if (cycle.size() != 1) {
          std::sort(cycle.begin(), cycle.end());
          result.emplace_back(cycle);
        }

      }  // this i already in a cycle
    }    // loop over i

    return result;
  }

  /// Product of this permutation by \c other

  /// \param other a Permutation
  /// \return \c other * \c this, i.e. \c this applied first, then \c other
  Permutation mult(const Permutation& other) const {
    // 1. domain of product = domain of this U domain of other
    using iset = std::set<index_type>;
    auto product_domain = this->domain<iset>();
    auto other_domain = other.domain<iset>();
    product_domain.insert(other_domain.begin(), other_domain.end());

    Map result;
    for (const auto& d : product_domain) {
      const auto d_image = other[(*this)[d]];
      if (d_image != d) result[d] = d_image;
    }

    return Permutation(result);
  }

  /// Construct the inverse of this permutation

  /// The inverse of permutation \f$ P \f$ is defined as \f$ P \times P^{-1} =
  /// P^{-1} \times P = I \f$, where \f$ I \f$ is the identity permutation.
  /// \return The inverse of this permutation
  Permutation inv() const {
    Map result;
    for (const auto& iter : p_) {
      const auto i = iter.first;
      const auto pi = iter.second;
      result.insert(std::make_pair(pi, i));
    }
    return Permutation(std::move(result));
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

    Permutation result;

    while (power) {
      if (power & 1) result = result.mult(value);
      value = value.mult(value);
      power >>= 1;
    }

    return result;
  }

  /// Data accessor

  /// gives direct access to \c std::map that encodes the Permutation
  /// \return \c std::map<index_type,index_type> object encoding the permutation
  /// in compressed two-line form
  const Map& data() const { return p_; }

  /// Idenity permutation factory method

  /// \return the identity permutation
  Permutation identity() const { return Permutation(); }

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
  return (p1.domain_size() == p2.domain_size()) && p1.data() == p2.data();
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
  if (&p1 == &p2) return false;
  return std::lexicographical_compare(p1.data().begin(), p1.data().end(),
                                      p2.data().begin(), p2.data().end());
}

/// Add permutation to an output stream

/// \param[out] output The output stream
/// \param[in] p The permutation to be added to the output stream
/// \return The output stream
inline std::ostream& operator<<(std::ostream& output, const Permutation& p) {
  output << "{";
  Permutation::print_map(output, p.data());
  output << "}";
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

namespace detail {

// clang-format off
/// Create a permuted copy of an array

/// \note a more efficient version of detail::permute_array specialized for
/// TiledArray::symmetry::Permutation
/// \tparam Arg The input array type
/// \tparam Result The output array type
/// \param[in] perm The permutation
/// \param[in] arg The input array to be permuted
/// \param[out] result The output array that will hold the permuted array
// clang-format on
template <typename Arg, typename Result>
inline void permute_array(const TiledArray::symmetry::Permutation& perm,
                          const Arg& arg, Result& result) {
  TA_ASSERT(result.size() == arg.size());
  if (perm.domain_size() < arg.size()) {
    std::copy(arg.begin(), arg.end(),
              result.begin());  // if perm does not map every element of arg,
    // copy arg to result first
  }
  for (const auto& p : perm.data()) {
    TA_ASSERT(result.size() > p.second);
    TA_ASSERT(arg.size() > p.second);
    result[p.second] = arg[p.first];
  }
}
}  // namespace detail

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
  std::array<T, N> result;
  symmetry::detail::permute_array(perm, a, result);
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
  const std::array<T, N> temp = a;
  symmetry::detail::permute_array(perm, temp, a);
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
  std::vector<T> result(v.size());
  symmetry::detail::permute_array(perm, v, result);
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
inline std::vector<T, A>& operator*=(std::vector<T, A>& v,
                                     const Permutation& perm) {
  const std::vector<T, A> temp = v;
  symmetry::detail::permute_array(perm, temp, v);
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
  boost::container::small_vector<T, N> result(v.size());
  symmetry::detail::permute_array(perm, v, result);
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
  symmetry::detail::permute_array(perm, temp, v);
  return v;
}

}  // namespace symmetry

/** @}*/

}  // namespace TiledArray

#endif  // TILEDARRAY_SYMM_PERMUTATION_H__INCLUDED
