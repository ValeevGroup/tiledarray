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

#ifndef TILEDARRAY_EXPRESSIONS_VARIABLE_LIST_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_VARIABLE_LIST_H__INCLUDED

#include "TiledArray/permutation.h"
#include "TiledArray/util/index_parsing.h"
#include <algorithm>
#include <iosfwd>
#include <string>

namespace TiledArray {
namespace expressions {

class VariableList;
VariableList operator*(const ::TiledArray::Permutation&, const VariableList&);
void swap(VariableList&, VariableList&);

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

template <typename VarLeft, typename VarRight>
inline Permutation var_perm(const VarLeft& l, const VarRight& r) {
  TA_ASSERT(l.size() == r.size());
  std::vector<size_t> a(l.size());
  typename VarRight::const_iterator rit = r.begin();
  for (auto it = a.begin(); it != a.end(); ++it) {
    typename VarLeft::const_iterator lit =
        std::find(l.begin(), l.end(), *rit++);
    TA_ASSERT(lit != l.end());
    *it = std::distance(l.begin(), lit);
  }
  return Permutation(std::move(a));
}
}  // namespace detail

/// Variable list manages the strings used to label an array's modes

/// In Tiled Array's DSL the user provides string labels for each mode of the
/// array. The operators between arrays and the appearance/absence of a label
/// defines the semantics of the exact operation subject to the generalized
/// Einstein summation convention. For the user's convenience the string labels
/// are input as a single string with labels for each tensor mode separated by
/// commas or a semicolon. The difference between the comma and semicolon
/// delimiter is that the semicolon separates outer tensor modes from inner
/// tensor modes (when tiles are tensor-of-tensors), whereas commas separate
/// modes within the same tensor nesting. To simplify parsing, whitespace in a
/// mode is ignored, e.g. the indices "ij,k", "i j, k", "ij ,k", "ij, k", etc.
/// all label the modes of the tensor identically (the zero-th mode is labeled
/// "ij" and the first mode is labeled "k").
///
/// Conceptually this class can be thought of as a container of indices. T
///
/// A ToT index with \f$n\f$ outer indices and \f$m\f$ inner indices is treated
/// like an index with \f$(n + m)\f$ indices; it is the responsibility of the
/// operation to do the right thing for a ToT.
class VariableList {
 private:
  /// Type of container used to hold individual indices
  template<typename T>
  using container_type = std::vector<T>;

  /// Type of an std::pair of container_type<T> instances
  template<typename T>
  using container_pair = std::pair<container_type<T>, container_type<T>>;

 public:
  /// Type used to store the individual string indices
  typedef std::string value_type;

  /// A read-only reference to a string index
  typedef const value_type& const_reference;

  /// A read-only random-access iterator
  typedef container_type<std::string>::const_iterator const_iterator;

  /// Type used for indexing and offsets
  typedef std::size_t size_type;

  /// Constructs an empty variable list.
  ///
  /// The default VariableList is an empty container. It has no indices (outer
  /// or inner). Iterators to the beginning of the container are the same as
  /// iterators to the end of the container. After instantiation the only way to
  /// add indices to a VariableList instance is via copy/move assignment.
  ///
  /// \throw None No throw guarantee.
  VariableList() = default;

  /// Initializes a VariableList by tokenizing a string
  ///
  /// This constructor invokes TiledArray::detail::split_index to tokenize
  /// \c vars. To label a rank \f$n\f$ tensor, \c vars should contain \f$n\f$
  /// substrings delimited by \f$(n-1)\f$ commas, e.g., "i,j,k" labels a rank 3
  /// tensor such that mode 0 is "i", mode 1 is "j", and mode 2 is "k". This
  /// constructor can also be used to label tensors-of-tensors. To label a rank
  /// \f$n\f$ tensor of rank \f$m\f$ tensors, \c vars should contain
  /// \f$(n + m)\f$ substrings such that the first \f$n\f$ are delimited from
  /// the last \f$m\f$ by a semicolon and the first \f$n\f$ are delimited from
  /// each other by \f$(n-1)\f$ commas and the last \f$m\f$ are delimited from
  /// each other by \f$(m-1)\f$ commas, e.g. "i,j,k;l,m" labels a rank 3
  /// tensor of rank 2 tensors such that "i,j,k" label the modes of the outer
  /// tensor (mode 0 of the outer tensor is labeled "i", mode 1 "j", and mode 2
  /// "k") and "l,m" label the modes of the inner tensor (mode 0 of the inner
  /// tensor will be labeled "l", mode 1 "m").
  ///
  /// \param[in] vars The string to tokenize.
  /// \throw TiledArray::Exception if \c vars is not a valid index according to
  ///                              detail::is_valid_index. Strong throw
  ///                              guarantee.
  /// \throw std::bad_alloc if there is insufficient memory to tokenize \c vars
  ///                       and store the result. Strong throw guarantee.
  explicit VariableList(const_reference vars) :
    VariableList(TiledArray::detail::split_index(vars)) {}

  /// Creates a new VariableList instance by deep-copying \c other
  ///
  /// \param[in] other The VariableList to deep copy.
  ///
  /// \throw std::bad_alloc if there is insufficient memory to copy \c other.
  ///                       Strong throw guarantee.
  VariableList(const VariableList& other) = default;

  /// Modifies the current VariableList so it contains a deep copy of \c other.
  ///
  /// The copy-assignment operator for VariableList erases any already existing
  /// state (and in the process invalidating any references to it) and replaces
  /// it with a deep copy of \c other. The current instance, containing a deep
  /// copy of \c other, is then returned to faciliate chaining.
  ///
  /// \param[in] other The VariableList instance whose state will be copied.
  /// \return The current instance containing a deep copy of \c other's state.
  /// \throw std::bad_alloc if there is insufficient memory to copy \c other.
  ///                       Strong throw guarantee.
  VariableList& operator=(const VariableList& other) = default;

  /// Sets the current instance to the provided string indices
  ///
  /// This function can be used to change the indices that are managed by this
  /// instance to those in the provided string. Ultimately the indices will be
  /// parsed the same way as via the string constructor and must satisfy the
  /// same criteria.
  ///
  /// \param[in] vars The new indices that the current instance should manage.
  /// \return The current instance, now storing \c vars.
  /// \throw TiledArray::Exception if \c vars are not valid indices. Strong
  ///                              throw guarantee.
  /// \throw std::bad_alloc if there is insufficient memory to parse and store
  ///                       \c vars. Strong throw guarantee.
  VariableList& operator=(const_reference vars) {
    VariableList(vars).swap(*this);
    return *this;
  }

  VariableList& operator*=(const Permutation& p) {
    TA_ASSERT(p.dim() == dim());
    vars_ *= p;
    return *this;
  }

  /// Determines if two VariableList instances are equivalent
  ///
  /// Two variableList instances are equivalent if they contain the same number
  /// of indices, the indices are partitioned into inner and outer indices
  /// identically, and if the \f$i\f$-th index of each instance are equivalent
  /// for all \f$i\f$. In particular this means VariableList instances will
  /// compare different if they use different capitalization and/or are
  /// permutations of each other.
  ///
  /// \param[in] other The VariableList instance we are comparing to.
  /// \return True if the two instances are equivalent and false otherwise.
  /// \throw None No throw guarantee.
  bool operator==(const VariableList& other) const {
    return std::tie(n_inner_, vars_) == std::tie(other.n_inner_, other.vars_);
  }

  /// Returns a random-access iterator pointing to the first string index.
  ///
  /// For a VariableList which describes a normal tensor (i.e., not a tensor-of-
  /// tensors) the indices are stored such that the first index in the container
  /// labels mode 0, the second index labels mode 1, etc. This function returns
  /// an iterator which points to the first index (that labeling mode 0) in this
  /// instance. If this instance is empty than the resulting iterator is the
  /// same as that returned by `end()` and should not be dreferenced.
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
  const_iterator begin() const { return vars_.begin(); }

  /// Returns a random-access iterator pointing to just past the last index.
  ///
  /// For a VariableList which describes a normal tensor (i.e., not a tensor-of-
  /// tensors) the indices are stored such that the first index in the container
  /// labels mode 0, the second index labels mode 1, etc. This function returns
  /// an iterator which points to just past the last index in this instance. If
  /// this instance is empty than the resulting iterator is the same as that
  /// returned by `begin()`.
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
  const_iterator end() const { return vars_.end(); }

  /// Returns the n-th string in the variable list.
  const_reference at(const size_type n) const { return vars_.at(n); }

  /// Returns the n-th string in the variable list.
  const_reference operator[](const size_type n) const { return vars_[n]; }

  /// Returns the total number of indices in the variable list
  size_type dim() const { return vars_.size(); }

  size_type outer_dim() const { return dim() - inner_dim(); }

  size_type inner_dim() const { return n_inner_; }

  /// Returns the number of strings in the variable list.
  size_type size() const { return vars_.size(); }

  //// Is this instance managing an index for a Tensor-of-Tensors?
  ///
  /// This member function can be used to determine if the managed index should
  /// be interpreted as being for a tensor-of-tensors.
  ///
  /// \return True if the managed index has an inner and outer component and
  ///         false otherwise.
  /// \throw None No throw guarantee.
  bool is_tot() const { return n_inner_ != 0; }

  const std::vector<std::string>& data() const { return vars_; }

  operator value_type() const {
    value_type result;

    size_type counter = 0;
    for(size_type i = 0; i < dim(); ++i){
      if(i == outer_dim()) result += ";";
      else if(i > 0) result += ",";
      result += at(i);
    }

    return result;
  }

  void swap(VariableList& other) {
    std::swap(n_inner_, other.n_inner_);
    std::swap(vars_, other.vars_);
  }

  /// Generate permutation relationship for variable lists

  /// The result of this function is a permutation that defines
  /// \c this=p^other .
  /// \tparam V An array type
  /// \param other An array that defines a variable list
  /// \return \c p as defined by the above relationship
  template <typename V>
  Permutation permutation(const V& other) const {
    return detail::var_perm(*this, other);
  }

  /// Check that this variable list is a permutation of \c other

  /// \return \c true if all variable in this variable list are in \c other,
  /// otherwise \c false.
  bool is_permutation(const VariableList& other) const {
    if(other.dim() != dim()) return false;
    return std::is_permutation(begin(), end(), other.begin());
  }

 private:
  /// Used to unpack the std::pair resulting from split_index
  explicit VariableList(const container_pair<value_type>& tot_idx):
      VariableList(tot_idx.first, tot_idx.second){}

  /// Constructor implementing VariableList(const value_type&)
  VariableList(const container_type<value_type>& outer,
               const container_type<value_type>& inner);


  /// Copies a comma separated list into a vector of strings. All spaces are
  /// removed from the sub-strings.
  void init_(const_reference vars) {
    std::string::const_iterator start = vars.begin();
    std::string::const_iterator finish = vars.begin();
    for (; finish != vars.end(); ++finish) {
      if (*finish == ',') {
        vars_.push_back(trim_spaces_(start, finish));
        start = finish + 1;
      }
    }
    vars_.push_back(trim_spaces_(start, finish));

    TA_ASSERT((unique_(vars_.begin(), vars_.end())));
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

  /// Returns true if all vars contained by the list are unique.
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

  size_type n_inner_ = 0;
  std::vector<value_type> vars_;

  friend VariableList operator*(const ::TiledArray::Permutation&,
                                const VariableList&);

};  // class VariableList

/// Exchange the content of the two variable lists.
inline void swap(VariableList& v0, VariableList& v1) { v0.swap(v1); }

inline bool operator!=(const VariableList& v0, const VariableList& v1) {
  return !(v0 == v1);
}

inline VariableList operator*(const ::TiledArray::Permutation& p,
                              const VariableList& v) {
  TA_ASSERT(p.dim() == v.dim());
  VariableList result;
  result.vars_ = p * v.vars_;

  return result;
}

/// ostream VariableList output operator.
inline std::ostream& operator<<(std::ostream& out, const VariableList& v) {
  return out << "(" << static_cast<std::string>(v)  << ")";
}


//------------------------------------------------------------------------------
//                             Implementations
//------------------------------------------------------------------------------


inline VariableList::VariableList(const container_type<value_type>& outer,
                          const container_type<value_type>& inner):
    n_inner_(inner.size()), vars_(outer.size() + inner.size()) {
  for(size_type i = 0; i < outer.size(); ++i)
    vars_[i] = outer[i];
  for(size_type i = 0; i < inner.size(); ++i)
    vars_[i + outer.size()] = inner[i];
}

}  // namespace expressions

}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_VARIABLE_LIST_H__INCLUDED
