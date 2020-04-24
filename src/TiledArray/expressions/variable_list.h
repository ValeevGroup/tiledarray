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

#include <TiledArray/permutation.h>
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

/// Variable list manages a list variable strings.

/// Each variable is separated by commas. All spaces are ignored and removed
/// from variable list. So, "a c" will be converted to "ac" and will be
/// considered a single variable. All variables must be unique.
class VariableList {
 public:
  typedef std::string value_type;
  typedef const std::string& const_reference;
  typedef std::vector<std::string>::const_iterator const_iterator;
  typedef std::size_t size_type;

  /// Constructs an empty variable list.
  VariableList() : vars_() {}

  /// constructs a variable lists
  explicit VariableList(const_reference vars) {
    if (vars.size() != 0) init_(vars);
  }

  template <typename InIter>
  VariableList(InIter first, InIter last) {
    static_assert(TiledArray::detail::is_input_iterator<InIter>::value,
                  "VariableList constructor requires an input iterator");
    TA_ASSERT(unique_(first, last));

    for (; first != last; ++first)
      vars_.push_back(trim_spaces_(first->begin(), first->end()));
  }

  VariableList(const VariableList& other) = default;

  VariableList& operator=(const VariableList& other) = default;

  VariableList& operator=(const_reference vars) {
    init_(vars);
    return *this;
  }

  VariableList& operator*=(const Permutation& p) {
    TA_ASSERT(p.dim() == dim());
    vars_ *= p;
    return *this;
  }

  /// Returns an iterator to the first variable.
  const_iterator begin() const { return vars_.begin(); }

  /// Returns an iterator to the end of the variable list.
  const_iterator end() const { return vars_.end(); }

  /// Returns the n-th string in the variable list.
  const_reference at(const size_type n) const { return vars_.at(n); }

  /// Returns the n-th string in the variable list.
  const_reference operator[](const size_type n) const { return vars_[n]; }

  /// Returns the number of strings in the variable list.
  size_type dim() const { return vars_.size(); }

  /// Returns the number of strings in the variable list.
  size_type size() const { return vars_.size(); }

  const std::vector<std::string>& data() const { return vars_; }

  value_type string() const {
    std::string result;
    std::vector<std::string>::const_iterator it = vars_.begin();
    if (it == vars_.end()) return result;

    for (result = *it++; it != vars_.end(); ++it) {
      result += "," + *it;
    }

    return result;
  }

  void swap(VariableList& other) { std::swap(vars_, other.vars_); }

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

  friend void swap(VariableList&, VariableList&);

  std::vector<std::string> vars_;

  friend VariableList operator*(const ::TiledArray::Permutation&,
                                const VariableList&);

};  // class VariableList

/// Exchange the content of the two variable lists.
inline void swap(VariableList& v0, VariableList& v1) {
  std::swap(v0.vars_, v1.vars_);
}

inline bool operator==(const VariableList& v0, const VariableList& v1) {
  return (v0.dim() == v1.dim()) && std::equal(v0.begin(), v0.end(), v1.begin());
}

inline bool operator!=(const VariableList& v0, const VariableList& v1) {
  return !operator==(v0, v1);
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
  out << "(";
  std::size_t d;
  std::size_t n = v.dim() - 1;
  for (d = 0; d < n; ++d) {
    out << v[d] << ", ";
  }
  out << v[d];
  out << ")";
  return out;
}

}  // namespace expressions

}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_VARIABLE_LIST_H__INCLUDED
