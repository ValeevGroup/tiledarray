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

#ifndef TILEDARRAY_EXPRESSIONS_INDEX_LIST_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_INDEX_LIST_H__INCLUDED

#include <algorithm>
#include <iosfwd>
#include <set>
#include <string>
#include "TiledArray/permutation.h"
#include "TiledArray/util/annotation.h"

namespace TiledArray {
namespace expressions {

class IndexList;
IndexList operator*(const ::TiledArray::Permutation&, const IndexList&);
void swap(IndexList&, IndexList&);

class BipartiteIndexList;
BipartiteIndexList operator*(const ::TiledArray::Permutation&,
                             const BipartiteIndexList&);
void swap(BipartiteIndexList&, BipartiteIndexList&);

namespace detail {
/// Finds the range of common elements for two sets of iterators.

/// This function finds the first contiguous set of elements equivalent
/// in two lists. Two pairs of iterators are returned via output parameters
/// \c common1 and \c common2. These two sets of output iterators point to
/// the first range of contiguous, equivalent elements in the two lists.
/// If no common elements far found; then \c common1.first and \c
/// common1.second both are equal to last1, and likewise for common2.
/// \tparam InIter1 The input iterator type for the first range of
/// elements.
/// \tparam InIter2 The input iterator type for the second range of
/// elements.
/// \param[in] first1 An input iterator pointing to the beginning of the
/// first range of elements to be compared.
/// \param[in] last1 An input iterator pointing to one past the end of the
/// first range of elements to be compared.
/// \param[in] first2 An input iterator pointing to the beginning of the
/// second range of elements to be compared.
/// \param[in] last2 An input iterator pointing to one past the end of the
/// second range of elements to be compared.
/// \param[out] common1 A pair of iterators where \c common1.first points
/// to the first common element, and \c common1.second points to one past
/// the last common element in the first list.
/// \param[out] common2 A pair of iterators where \c common1.first points
/// to the first common element, and \c common1.second points to one past
/// the last common element in the second list.
template <typename InIter1, typename InIter2>
void find_common(InIter1 first1, const InIter1 last1, InIter2 first2,
                 const InIter2 last2, std::pair<InIter1, InIter1>& common1,
                 std::pair<InIter2, InIter2>& common2) {
  common1.first = last1;
  common1.second = last1;
  common2.first = last2;
  common2.second = last2;

  // find the first common element in the in the two ranges
  for (; first1 != last1; ++first1) {
    common2.first = std::find(first2, last2, *first1);
    if (common2.first != last2) break;
  }
  common1.first = first1;
  first2 = common2.first;

  // find the last common element in the two ranges
  while ((first1 != last1) && (first2 != last2)) {
    if (*first1 != *first2) break;
    ++first1;
    ++first2;
  }
  common1.second = first1;
  common2.second = first2;
}

inline Permutation var_perm(const IndexList& l, const IndexList& r);
inline BipartitePermutation var_perm(const BipartiteIndexList& l,
                                     const BipartiteIndexList& r);

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////

/// IndexList is a sequence of "indices" (multicharacter labels) used to
/// annotate modes of a miltidimensional tensor/array for the purpose of
/// encoding an operation
class IndexList {
 public:
  using container_type = container::svector<std::string>;
  using const_iterator = typename container_type::const_iterator;

  /// Constructs an empty index list.
  IndexList() : indices_() {}

  /// constructs from a string

  /// \param str a string containing comma-separated index labels.
  /// All whitespaces are discarded, i.e., "a c" will be converted to "ac"
  /// and will be considered a single index.
  explicit IndexList(const std::string& str) {
    if (!str.empty()) init_(str);
  }

  /// constructs from a range of index labels

  /// \tparam InIter an input iterator dereferencing to std::string
  /// \param first the begin iterator
  /// \param last the end iterator
  /// \note All whitespaces are discarded, i.e., "a c" will be converted to "ac"
  /// and will be considered a single index.
  template <typename InIter>
  IndexList(InIter first, InIter last) {
    static_assert(TiledArray::detail::is_input_iterator<InIter>::value,
                  "IndexList constructor requires an input iterator");

    for (; first != last; ++first)
      indices_.push_back(trim_spaces_(first->begin(), first->end()));
  }

  IndexList(const IndexList& other) : indices_(other.indices_) {}

  IndexList& operator=(const IndexList& other) {
    indices_ = other.indices_;

    return *this;
  }

  IndexList& operator=(const std::string& vars) {
    init_(vars);
    return *this;
  }

  IndexList& operator*=(const Permutation& p) {
    TA_ASSERT(p.size() == size());
    indices_ *= p;
    return *this;
  }

  /// Returns an iterator to the first index.
  const_iterator begin() const { return indices_.begin(); }

  /// Returns an iterator to the end of the index list.
  const_iterator end() const { return indices_.end(); }

  /// Returns the n-th index
  const std::string& at(const std::size_t n) const { return indices_.at(n); }

  /// Returns the n-th index
  const std::string& operator[](const std::size_t n) const {
    return indices_[n];
  }

  /// Returns the number of elements in the index list.
  [[deprecated("use IndexList::size()")]] unsigned int dim() const {
    return indices_.size();
  }

  /// Returns the number of elements in the index list.
  unsigned int size() const { return indices_.size(); }

  const auto& data() const { return indices_; }

  /// comma-separated concatenator of indices
  std::string string() const {
    std::string result;
    using std::cbegin;
    auto it = cbegin(indices_);
    if (it == indices_.end()) return result;

    for (result = *it++; it != indices_.end(); ++it) {
      result += "," + *it;
    }

    return result;
  }

  void swap(IndexList& other) { std::swap(indices_, other.indices_); }

  /// Returns the number of times index \p x appears in this instance
  /// \param[in] x The index we are searching for.
  /// \return the number of times \p x appears in this object
  auto count(const std::string& x) const {
    return std::count(begin(), end(), x);
  }

  /// Returns the positions of \p x in this
  ///
  /// \param[in] x The index we are looking for.
  /// \return A random-access container whose length is the number of times
  ///         that \p x appears in the annotation and whose elements are the
  ///         indices whose labels equal \p x.
  auto positions(const std::string& x) const {
    container::svector<std::size_t> rv;
    for (std::size_t i = 0; i < size(); ++i)
      if ((*this)[i] == x) rv.push_back(i);
    return rv;
  }

  /// Computes permutation that converts an index list to this list

  /// The result of this function is a permutation that defines
  /// \c this=p^other .
  /// \tparam V A range type
  /// \param other An array that defines a index list
  /// \return \c p as defined by the above relationship
  template <typename V,
            typename = std::enable_if_t<TiledArray::detail::is_range_v<V>>>
  Permutation permutation(const V& other) const {
    return detail::var_perm(*this, other);
  }

  /// Check that this index list is a permutation of \c other

  /// \return \c true if all indices in this index list are in \c other,
  /// otherwise \c false.
  bool is_permutation(const IndexList& other) const {
    if (indices_.size() != other.indices_.size()) return false;

    for (const_iterator it = begin(); it != end(); ++it) {
      const_iterator other_it = other.begin();
      for (; other_it != other.end(); ++other_it)
        if (*it == *other_it) break;
      if (other_it == other.end()) return false;
    }

    return true;
  }

 private:
  /// Initializes from a comma-separated sequence of indices
  void init_(const std::string& vars) {
    std::string::const_iterator start = vars.begin();
    std::string::const_iterator finish = vars.begin();
    for (; finish != vars.end(); ++finish) {
      if (*finish == ',') {
        indices_.push_back(trim_spaces_(start, finish));
        start = finish + 1;
      }
    }
    indices_.push_back(trim_spaces_(start, finish));
  }

  /// Returns a string with all the spaces ( ' ' ) removed from the string
  /// defined by the start and finish iterators.
  static std::string trim_spaces_(std::string::const_iterator first,
                                  std::string::const_iterator last) {
    TA_ASSERT(first != last);
    std::string result = "";
    for (; first != last; ++first) {
      TA_ASSERT(valid_char_(*first));
      if (*first != ' ' && *first != '\0') result.append(1, *first);
    }

    TA_ASSERT(result.length() != 0);

    return result;
  }

  /// Returns true if all indices contained by the list are unique.
  template <typename InIter>
  bool unique_(InIter first, InIter last) const {
    for (; first != last; ++first) {
      InIter it2 = first;
      for (++it2; it2 != last; ++it2)
        if (first->compare(*it2) == 0) return false;
    }

    return true;
  }

  static bool valid_char_(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || (c == ' ') || (c == ',') || (c == '\0') ||
           (c == '\'') || (c == '_');
  }

  friend void swap(IndexList&, IndexList&);

  container_type indices_;

  friend IndexList operator*(const ::TiledArray::Permutation&,
                             const IndexList&);

};  // class IndexList

/// Exchange the content of the two index lists.
inline void swap(IndexList& v0, IndexList& v1) {
  std::swap(v0.indices_, v1.indices_);
}

inline bool operator==(const IndexList& v0, const IndexList& v1) {
  return (v0.size() == v1.size()) &&
         std::equal(v0.begin(), v0.end(), v1.begin());
}

inline bool operator!=(const IndexList& v0, const IndexList& v1) {
  return !operator==(v0, v1);
}

inline IndexList operator*(const ::TiledArray::Permutation& p,
                           const IndexList& v) {
  TA_ASSERT(p.size() == v.size());
  IndexList result;
  result.indices_ = p * v.indices_;

  return result;
}

/// ostream IndexList output operator.
inline std::ostream& operator<<(std::ostream& out, const IndexList& v) {
  out << "(";
  std::size_t d;
  std::size_t n = v.size() - 1;
  for (d = 0; d < n; ++d) {
    out << v[d] << ", ";
  }
  out << v[d];
  out << ")";
  return out;
}

////////////////////////////////////////////////////////////////////////////////

/// BipartiteIndexList is a set of 2 IndexList objects that can be viewed as a
/// single concatenated range
class BipartiteIndexList {
 private:
  /// Type of container used to hold individual indices
  using container_type = typename IndexList::container_type;

  /// Type of an std::pair of container_type instances
  using container_pair = std::pair<container_type, container_type>;

 public:
  /// Type used to store the individual string indices
  typedef std::string value_type;

  /// A read-only reference to a string index
  typedef const value_type& const_reference;

  /// A read-only random-access iterator
  typedef container_type::const_iterator const_iterator;

  /// Type used for indexing and offsets
  typedef std::size_t size_type;

  /// Constructs an empty index list.
  ///
  /// The default BipartiteIndexList is an empty container. It has no indices
  /// (outer or inner). Iterators to the beginning of the container are the same
  /// as iterators to the end of the container. After instantiation the only way
  /// to add indices to a BipartiteIndexList instance is via copy/move
  /// assignment.
  ///
  /// \throw None No throw guarantee.
  BipartiteIndexList() = default;

  /// Initializes a BipartiteIndexList by tokenizing a string
  ///
  /// This constructor invokes TiledArray::detail::split_index to tokenize
  /// \p str. To label a rank \f$n\f$ tensor, \p str should contain \f$n\f$
  /// substrings delimited by \f$(n-1)\f$ commas, e.g., "i,j,k" labels a rank 3
  /// tensor such that mode 0 is "i", mode 1 is "j", and mode 2 is "k". This
  /// constructor can also be used to label tensors-of-tensors. To label a rank
  /// \f$n\f$ tensor of rank \f$m\f$ tensors, \p str should contain
  /// \f$(n + m)\f$ substrings such that the first \f$n\f$ are delimited from
  /// the last \f$m\f$ by a semicolon and the first \f$n\f$ are delimited from
  /// each other by \f$(n-1)\f$ commas and the last \f$m\f$ are delimited from
  /// each other by \f$(m-1)\f$ commas, e.g. "i,j,k;l,m" labels a rank 3
  /// tensor of rank 2 tensors such that "i,j,k" label the modes of the outer
  /// tensor (mode 0 of the outer tensor is labeled "i", mode 1 "j", and mode 2
  /// "k") and "l,m" label the modes of the inner tensor (mode 0 of the inner
  /// tensor will be labeled "l", mode 1 "m").
  ///
  /// \param[in] str The string to tokenize.
  /// \throw TiledArray::Exception if \p str is not a valid index according to
  ///                              detail::is_valid_index. Strong throw
  ///                              guarantee.
  explicit BipartiteIndexList(const std::string str)
      : BipartiteIndexList(TiledArray::detail::split_index(str)) {}

  /// Creates a new BipartiteIndexList instance by deep-copying \c other
  ///
  /// \param[in] other The BipartiteIndexList to deep copy.
  BipartiteIndexList(const BipartiteIndexList& other) = default;

  /// Modifies the current BipartiteIndexList so it contains a deep copy of
  /// \c other.
  ///
  /// The copy-assignment operator for BipartiteIndexList erases any already
  /// existing state (and in the process invalidating any references to it) and
  /// replaces it with a deep copy of \c other. The current instance, containing
  /// a deep copy of \c other, is then returned to faciliate chaining.
  ///
  /// \param[in] other The BipartiteIndexList instance whose state will be
  /// copied.
  /// \return The current instance containing a deep copy of \c other's
  /// state.
  BipartiteIndexList& operator=(const BipartiteIndexList& other) = default;

  /// Sets the current instance to the BipartiteIndexList constructed from \p
  /// str
  ///
  /// \param[in] str a string to be tokenized
  /// \return Reference to \c *this
  BipartiteIndexList& operator=(const std::string& str) {
    BipartiteIndexList(str).swap(*this);
    return *this;
  }

  /// Applies a permutation, in-place, to the current BipartiteIndexList
  /// instance
  ///
  /// This function applies the Permutation instance, \c p, to the current
  /// BipartiteIndexList instance overwriting the already existing state with
  /// the permuted state.
  ///
  /// \param[in] p The Permutation to apply. \c p should be of rank `size()`.
  /// \return The current instance after applying the permutation.
  BipartiteIndexList& operator*=(const Permutation& p) {
    TA_ASSERT(p.size() == size());
    indices_ *= p;
    return *this;
  }

  /// Determines if two BipartiteIndexList instances are equivalent
  ///
  /// Two BipartiteIndexList instances are equivalent if they contain the same
  /// number of indices, the indices are partitioned into inner and outer
  /// indices identically, and if the \f$i\f$-th index of each instance are
  /// equivalent for all \f$i\f$. In particular this means BipartiteIndexList
  /// instances will compare different if they use different capitalization
  /// and/or are permutations of each other.
  ///
  /// \param[in] other The BipartiteIndexList instance we are comparing to.
  /// \return True if the two instances are equivalent and false otherwise.
  /// \throw None No throw guarantee.
  bool operator==(const BipartiteIndexList& other) const {
    return std::tie(second_size_, indices_) ==
           std::tie(other.second_size_, other.indices_);
  }

  /// Returns a random-access iterator pointing to the first string index.
  ///
  /// For a BipartiteIndexList which describes a normal tensor (i.e., not a
  /// tensor-of- tensors) the indices are stored such that the first index in
  /// the container labels mode 0, the second index labels mode 1, etc. This
  /// function returns an iterator which points to the first index (that
  /// labeling mode 0) in this instance. If this instance is empty than the
  /// resulting iterator is the same as that returned by `end()` and should not
  /// be dreferenced.
  ///
  /// If this instance describes a tensor-of-tensors the indices are stored
  /// flattened such that the outer indices appear before the inner indices. The
  /// iterator returned by `begin()` thus points to the index labeling the 0th
  /// mode of the outer tensor.
  ///
  /// The iterator resulting from this function is valid until the set of
  /// indices managed by this instance is modified.
  ///
  /// \return A read-only iterator pointing to the 0-th mode of the tensor. For
  ///         tensor-of-tensors the iterator points to the 0-th mode of the
  ///         outer tensor and will run over the outer modes followed by the
  ///         modes of the inner tensor.
  /// \throw None No throw guarantee.
  const_iterator begin() const { return indices_.begin(); }

  /// Returns a random-access iterator pointing to just past the last index.
  ///
  /// For a BipartiteIndexList which describes a normal tensor (i.e., not a
  /// tensor-of- tensors) the indices are stored such that the first index in
  /// the container labels mode 0, the second index labels mode 1, etc. This
  /// function returns an iterator which points to just past the last index in
  /// this instance. If this instance is empty than the resulting iterator is
  /// the same as that returned by `begin()`.
  ///
  /// If this instance describes a tensor-of-tensors the indices are stored
  /// flattened such that the outer indices appear before the inner indices. The
  /// iterator returned by `end()` thus points to the just past the last index
  /// of the inner tensor.
  ///
  /// The iterator resulting from this function is valid until the set of
  /// indices managed by this instance is modified. The iterator itself is only
  /// a semaphore and should not be dereferenced.
  ///
  /// \return A read-only iterator pointing to just past the last index. For
  ///         tensor-of-tensors the iterator points to just past the last index
  ///         of the inner tensor.
  /// \throw None No throw guarantee.
  const_iterator end() const { return indices_.end(); }

  /// Returns the n-th string in the index list.
  ///
  /// This member function returns the requested string index, \c n, throwing if
  /// \c n is not in the range [0, dim()). Use operator[](size_type) to avoid
  /// the in-range check.
  /// \param[in] n The index of the requested mode label. \c n should be in the
  ///              range [0, dim()).
  /// \return A read-only reference to the requested string index.
  /// \throw std::out_of_range if \c n is not in the range [0, dim()). Strong
  ///                          throw guarantee.
  const_reference at(const size_type n) const { return indices_.at(n); }

  /// Returns the n-th string in the index list.
  ///
  /// This member function returns the string used to label mode \c n of the
  /// tensor.  Unlike at, no range check is performed and undefined behavior
  /// will occur if \c n is not in the range [0, dims()).
  /// \param[in] n The index of the requested mode label. \c n should be in the
  ///              range [0, dims()) otherwise undefined behavior will occur.
  /// \return A read-only reference to the requested string index.
  /// \throw None No throw guarantee.
  const_reference operator[](const size_type n) const { return indices_[n]; }

  /// Returns the indices of the modes annotated with \p x
  ///
  /// This function can be thought of as the inverse mapping of `at` and
  /// `operator[]` namely given an annotation, \p x, return the modes labeled
  /// with \p x. For example assume that this instance stores `"i,j,k,l"`
  /// calling this function with input `"i"` would return a container whose
  /// only element is `0`, calling this function with `"j"` would return a
  /// container whose only element is `1`, etc. This function returns a
  /// container, and not a single index, in case the annotation labels more
  /// than one mode (e.g., in a trace).
  ///
  /// \param[in] x The annotation we are looking for.
  /// \return A random-access container whose length is the number of times
  ///         that \p x appears in the annotation and whose elements are the
  ///         modes labeled with \p x.
  auto positions(const std::string& x) const {
    container::svector<std::size_t> rv;
    for (size_type i = 0; i < size(); ++i)
      if ((*this)[i] == x) rv.push_back(i);
    return rv;
  }

  /// Returns the number of times annotation \p x appears in this instance
  ///
  /// This function is used to count the number of times an annotation appears
  /// in the set of annotations managed by this instance. A particular
  /// annotation can appear more than once, for example in a trace.
  ///
  /// \param[in] x The annotation we are searching for.
  /// \return An unsigned integer in the range [0, dim()) indicating the number
  ///         of times \p x appears in this instance.
  /// \throw None No throw guarantee.
  size_type count(const std::string& x) const {
    return std::count(begin(), end(), x);
  }

  /// Returns the total number of indices in the index list
  ///
  /// This function is just an alias for the `size()` member. It returns the
  /// the total number of indices in the index list.
  ///
  /// \return The total number of indices in the index list.
  /// \throw None No throw guarantee.
  [[deprecated("use BipartiteIndexList::size()")]] size_type dim() const {
    return size();
  }

  /// \return the size of the first sublist
  size_type first_size() const { return size() - second_size(); }

  /// Returns the number of inner indices in the index list
  ///
  /// BipartiteIndexList is capable of holding a string labeling a
  /// tensor-of-tensors or a normal (non-nested) tensor. For a ToT the indices
  /// are partitioned into outer (those for the outer tensor whose elements are
  /// tensors) and inner (those for the tensors which are elements of the outer
  /// tensor). This function returns the number of inner indices in the provided
  /// index. By convention all indices for a normal tensor are outer indices and
  /// this function will always return zero.
  ///
  /// \return The total number of inner indices in the managed list of labels.
  /// \throw None No throw guarantee.
  size_type second_size() const { return second_size_; }

  IndexList first() const { return IndexList(begin(), begin() + first_size()); }

  IndexList second() const { return IndexList(begin() + first_size(), end()); }

  /// Returns the total number of indices in the index list
  ///
  /// This function returns the total number of indices in the
  /// BipartiteIndexList. For a normal, non-nested, tensor this is simply the
  /// number of indices. For a tensor-of-tensors the total number of indices is
  /// the number of outer indices plus the number of inner indices.
  ///
  /// \return The total number of indices in the index list. For a tensor-of-
  ///         tensors the total number is the sum of the number of outer indices
  ///         plus the number of inner indices.
  /// \throw None No throw guarantee.
  size_type size() const { return indices_.size(); }

  const auto& data() const { return indices_; }

  /// Enables conversion from a BipartiteIndexList to a string
  ///
  /// This function will cast the BipartiteIndexList instance to a string,
  /// such that mode labels are separated by commas
  ///
  /// \return A string representation of the
  explicit operator value_type() const;

  /// Swaps the current instance's state with that of \c other
  ///
  /// \param[in] other The instance to swap state with. After this operation,
  ///           \c other will contain this instance's state and this instance
  ///           will contain other's state.
  void swap(BipartiteIndexList& other) noexcept {
    std::swap(second_size_, other.second_size_);
    std::swap(indices_, other.indices_);
  }

  /// Computes the permutation to go from \c other to this instance

  /// The result of this function is a permutation that defines
  /// \c this=p^other .
  /// \tparam V A container of strings. \c V must minimally be forward iterable.
  /// \param[in] other An array that defines a index list
  /// \return The permutation which can be applied to other to generate this
  ///         instance.
  /// \throw TiledArray::Exception if the \c other does not contain the same
  ///                              number of indices. Strong throw guarantee.
  /// \throw TiledArray::Exception if \c other does not contain the same indices
  ///                              as this instance. Strong throw guarantee.
  /// \throw TiledArray::Exception if \c other does not have the same ToT
  ///                              structure as this tensor. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if \c other is a ToT permutation such that it
  ///                              mixes outer and inner modes. Strong throw
  ///                              guarantee.
  template <typename V,
            typename = std::enable_if_t<TiledArray::detail::is_range_v<V>>>
  BipartitePermutation permutation(const V& other) const {
    return detail::var_perm(*this, other);
  }

  /// Check that this index list is a permutation of \c other

  /// \return \c true if all indices in this index list are in \c other,
  /// otherwise \c false.
  bool is_permutation(const BipartiteIndexList& other) const {
    if (other.size() != size()) return false;
    return std::is_permutation(begin(), end(), other.begin());
  }

  /// Constructor implementing BipartiteIndexList(const value_type&)
  template <typename OuterType, typename InnerType>
  BipartiteIndexList(OuterType&& outer, InnerType&& inner);

 private:
  /// Used to unpack the std::pair resulting from split_index
  template <typename First, typename Second>
  explicit BipartiteIndexList(const std::pair<First, Second>& tot_idx)
      : BipartiteIndexList(tot_idx.first, tot_idx.second) {}

  /// The size of the second sublist
  size_type second_size_ = 0;

  /// The concatenated list of indices
  container_type indices_;

  friend BipartiteIndexList operator*(const ::TiledArray::Permutation&,
                                      const BipartiteIndexList&);

};  // class BipartiteIndexList

/// Returns a set of each annotation found in at least one of the index lists
template <typename T, typename... Args>
auto all_annotations(T&& v, Args&&... args) {
  std::set<std::string> rv;
  if constexpr (sizeof...(Args) > 0) {
    rv = all_annotations(std::forward<Args>(args)...);
  }
  rv.insert(v.begin(), v.end());
  return rv;
}

/// Returns the set of annotations found in all of the index lists
template <typename T, typename... Args>
auto common_annotations(T&& v, Args&&... args) {
  std::set<std::string> rv;
  if constexpr (sizeof...(Args)) {
    rv = common_annotations(std::forward<Args>(args)...);
    // Remove all annotations not found in v
    decltype(rv) buffer(rv);
    for (const auto& x : buffer)
      if (!v.count(x)) rv.erase(x);
  } else {
    // Initialize rv to all annotations in v
    rv.insert(v.begin(), v.end());
  }
  return rv;
}

template <typename IndexList_, typename... Args>
auto bound_annotations(const IndexList_& out, Args&&... args) {
  // Get all indices in the input tensors
  auto rv = all_annotations(std::forward<Args>(args)...);

  // Remove those found in the output tensor
  decltype(rv) buffer(rv);
  for (const auto& x : buffer)
    if (out.count(x)) rv.erase(x);
  return rv;
}

/// Exchange the content of the two index lists.
inline void swap(BipartiteIndexList& v0, BipartiteIndexList& v1) {
  v0.swap(v1);
}

/// Determines if two BipartiteIndexLists are different.
///
/// Two IndexList instances are equivalent if they contain the same number
/// of indices, the indices are partitioned into inner and outer indices
/// identically, and if the \f$i\f$-th index of each instance are equivalent
/// for all \f$i\f$. In particular this means BipartiteIndexList instances
/// will compare different if they use different capitalization and/or are
/// permutations of each other.
///
/// \param[in] other The BipartiteIndexList instance we are comparing to.
/// \return True if the two instances are different and false otherwise.
/// \throw None No throw guarantee.
inline bool operator!=(const BipartiteIndexList& v0,
                       const BipartiteIndexList& v1) {
  return !(v0 == v1);
}

inline BipartiteIndexList operator*(const ::TiledArray::Permutation& p,
                                    const BipartiteIndexList& v) {
  TA_ASSERT(p.size() == v.size());
  BipartiteIndexList result(v);
  return result *= p;
}

/// Prints a BipartiteIndexList instance to a stream
///
/// This function simply casts the IndexList to a string, adds parenthesis to
/// it, and then inserts the resulting string into the stream.
///
/// \param[in,out] out the stream that \c v will be written to.
/// \param[in] v The BipartiteIndexList instance to insert into the stream.
/// \return \c out will be returned after adding \c v to it.
inline std::ostream& operator<<(std::ostream& out,
                                const BipartiteIndexList& v) {
  const std::string str = "(" + static_cast<std::string>(v) + ")";
  return out << str;
}

//------------------------------------------------------------------------------
//                             Implementations
//------------------------------------------------------------------------------

inline BipartiteIndexList::operator value_type() const {
  value_type result;
  for (size_type i = 0; i < size(); ++i) {
    if (i == first_size())
      result += ";";
    else if (i > 0)
      result += ",";
    result += at(i);
  }

  return result;
}

template <typename OuterType, typename InnerType>
inline BipartiteIndexList::BipartiteIndexList(OuterType&& outer,
                                              InnerType&& inner)
    : second_size_(inner.size()), indices_(outer.size() + inner.size()) {
  for (size_type i = 0; i < outer.size(); ++i) indices_[i] = outer[i];
  for (size_type i = 0; i < inner.size(); ++i)
    indices_[i + outer.size()] = inner[i];
}

namespace detail {

inline Permutation var_perm(const IndexList& l, const IndexList& r) {
  using std::size;
  TA_ASSERT(size(l) == size(r));
  container::svector<size_t> a(size(l));
  using std::begin;
  using std::end;
  auto rit = begin(r);
  for (auto it = begin(a); it != end(a); ++it) {
    auto lit = std::find(begin(l), end(l), *rit++);
    TA_ASSERT(lit != end(l));
    *it = std::distance(begin(l), lit);
  }
  return Permutation(std::move(a));
}

inline BipartitePermutation var_perm(const BipartiteIndexList& l,
                                     const BipartiteIndexList& r) {
  using std::size;
  TA_ASSERT(size(l) == size(r));
  TA_ASSERT(l.first_size() == r.first_size());
  container::svector<size_t> a(size(l));
  using std::begin;
  using std::end;
  auto rit = begin(r);
  for (auto it = begin(a); it != end(a); ++it) {
    auto lit = std::find(begin(l), end(l), *rit++);
    TA_ASSERT(lit != end(l));
    *it = std::distance(begin(l), lit);
  }
  // Make sure this permutation doesn't mix outer and inner tensors
  if (l.second_size() > 0) {
    auto outer_size = l.first_size();
    for (decltype(outer_size) i = 0; i < outer_size; ++i)
      TA_ASSERT(a[i] < outer_size);
    for (auto i = outer_size; i < a.size(); ++i) TA_ASSERT(a[i] >= outer_size);
  }
  return BipartitePermutation(std::move(a), l.second_size());
}

}  // namespace detail

/////////////// adaptors for the inner-outer language /////////////////////

inline auto inner(const IndexList& p) {
  abort();
  return IndexList{};
}

// N.B. can't return ref here due to possible dangling ref when p is bound to
// temporary
inline auto outer(const IndexList& p) { return p; }

inline auto inner_size(const IndexList& p) { return 0; }

inline auto outer_size(const IndexList& p) { return p.size(); }

inline auto inner(const BipartiteIndexList& p) { return p.second(); }

inline auto outer(const BipartiteIndexList& p) { return p.first(); }

inline auto inner_size(const BipartiteIndexList& p) { return p.second_size(); }

inline auto outer_size(const BipartiteIndexList& p) { return p.first_size(); }

}  // namespace expressions

}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_INDEX_LIST_H__INCLUDED
