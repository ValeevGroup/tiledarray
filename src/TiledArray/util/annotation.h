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
 *  tensor.h
 *  Jun 16, 2015
 *
 */

#ifndef TILEDARRAY_ANNOTATION_H__INCLUDED
#define TILEDARRAY_ANNOTATION_H__INCLUDED
#include "TiledArray/error.h"
#include <algorithm>
#include <string>
#include <cstring>

namespace TiledArray::detail {

inline std::string dummy_annotation(unsigned int ndim) {
  std::ostringstream oss;
  if (ndim > 0) oss << "i0";
  for (unsigned int d = 1; d < ndim; ++d) oss << ",i" << d;
  return oss.str();
}

/// This function removes all whitespace characters from a string.
///
/// \param[in] s The string we are removing whitespace from.
/// \return \c s, but without whitespace.
/// \throw none No throw guarantee.
inline auto remove_whitespace(std::string s){
  s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
  return s;
}

/// Splits a sting into tokens based on a character delimiter
///
/// This function assumes that the input string can be considered a series of
/// delimiter separated tokens. It will split the string into tokens and return
/// an std::vector of the tokens. This function does no additional string
/// processing (such as removing spaces).
///
/// It's worth noting several edge cases:
///
/// - If \c s is an empty string the result vector will contain a single empty
///   string.
/// - If \c s starts/ends with a delimiter the result vector will start/end with
///   an empty string
/// - Adjacent delimiters are tokenized as delimiting the empty string
///
/// Downstream error checking relies on this edge case behavior.
///
/// \param[in] s The string we are splitting into tokens.
/// \param[in] delim The character used to delimit tokens
/// \return A vector containing the tokens
/// \throw std::bad_alloc if there is insufficient memory to allocate the vector
///                       which will hold the return. Strong throw guarantee.
/// \throw std::bad_alloc if there is insufficient memory to allocate the
///                       tokens in the return. Strong throw guarantee.
inline auto tokenize_index(const std::string& s, char delim) {
  if(s.size() == 0) return std::vector<std::string>{""};
  std::vector<std::string> tokens;
  std::stringstream ss(s);
  std::string buffer;
  while(std::getline(ss, buffer, delim))
    tokens.emplace_back(std::move(buffer));

  // If the delimiter is the last element, we miss an empty string so add it
  if(s[s.size() -1] == delim)
    tokens.push_back(std::string{});
  return tokens;
}

/// Checks that the provided index is a valid TiledArray index
///
/// Tiled Array defines a string as being a valid index if each character is one
/// of the following:
///
/// - Roman letters A through Z (uppercase and lowercase are allowed)
/// - Base 10 numbers 0 through 9
/// - Whitespace
/// - underscore (_), comma (,), or semicolon (;)
///
/// Additionally the string can not:
///
/// - be only whitespace
/// - contain more than one semicolon
/// - have anonymous index name (i.e. can't have "i,,k" because the middle index
///   has no name).
///
/// \param[in] idx The index whose validity is being questioned.
/// \return True if the string corresponds to a valid index and false otherwise.
/// \note This function only tests that the characters making up the index are
///       valid. The index may still be invalid for a particular tensor. For
///       example if \c idx is an index for a matrix, but the actual tensor is
///       rank 3, then \c idx would be an invalid index for that tensor despite
///       being a valid index.
/// \throw std::bad_alloc if there is insufficient memory to copy \c idx. Strong
///                       throw guarantee.
/// \throw std::bad_alloc if there is insufficient memory to split \c idx into
///                      tokens. Strong throw guarantee.
inline bool is_valid_index(const std::string& idx) {
  const std::string valid_chars =
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "1234567890"
      " _,;";
  // Are valid characters
  for(const auto& c : idx)
    if(valid_chars.find(c) == std::string::npos) return false;

  // Is not only whitespace
  auto no_ws = remove_whitespace(idx);
  if(no_ws.size() == 0) return false;

  // At most one semicolon
  auto split_on_semicolon = tokenize_index(no_ws, ';');
  if(split_on_semicolon.size() > 2) return false;


  for(auto x : split_on_semicolon) {
    auto indices = tokenize_index(x, ',');
    for(const auto& idx : indices)
      if(idx.size() == 0) return false;
  }

  return true;
}


/// Defines what it means for a string index to be for a Tensor-of-Tensors
///
/// Tiled Array defines an index as being for a tensor-of-tensors if:
/// - the index is valid
/// - the index contains a semicolon
///
/// \param[in] idx The index whose tensor-of-tensor-ness is in question.
/// \return True if \c idx is a valid tensor-of-tensor index and false
///        otherwise.
/// \throw std::bad_alloc if is_valid_index throws while copying or splitting
///                       \c idx. Strong throw guarantee.
inline bool is_tot_index(const std::string& idx) {
  if(!is_valid_index(idx)) return false;
  return idx.find(";") != std::string::npos;
}

/// Splits and sanitizes a string labeling a tensor's modes.
///
/// This function encapsulates TiledArray's string index parsing. It is a free
/// function to facilitate usage outside the VariableList class. This function
/// will take a string and separate it into the individual mode labels. The API
/// is designed so that `split_index` can be used with tensors-of-tensors as
/// well as normal, non-nested tensors. By convention, tokenized indices of
/// normal tensors are returned as "outer" indices. The resulting indices will
/// be stripped of all whitespace to facilitate string comparisons.
///
/// \note This function will ensure that \c idx is a valid string label. This
///       entails requiring that `is_valid_index(idx)` is true. It does not take
///       into the rank and/or partitioning of the tensor being labeled, i.e.,
///       it is the caller's responsibility to make sure the index makes sense
///       for the tensor being labeled.
///
/// \param[in] idx The string label that should be tokenized
/// \return An std::pair such that the first element is a vector containing the
///         tokenized outer indices and the second element of the std::pair is a
///         std::vector with the tokenized inner indices. Inner indices will be
///         an empty std::vector if \c idx is not a tensor-of-tensor index.
/// \throw TiledArray::Exception if \c idx is not a valid string labeling.
///                              Strong throw guarantee.
/// \throw std::bad_alloc if there is insufficient memory to copy \c idx or to
///                       create the returns. Strong throw guarantee.
inline auto split_index(const std::string& idx) {
  TA_ASSERT(is_valid_index(idx));
  auto no_ws = remove_whitespace(idx);
  if(!is_tot_index(no_ws)) {
    return std::make_pair(tokenize_index(no_ws, ','), std::vector<std::string>{});
  }
  auto tot_idx = tokenize_index(no_ws, ';');
  return std::make_pair(tokenize_index(tot_idx[0], ','),
                        tokenize_index(tot_idx[1], ','));

}

} // namespace TiledArray::detail

#endif // TILEDARRAY_ANNOTATION_H__INCLUDED
