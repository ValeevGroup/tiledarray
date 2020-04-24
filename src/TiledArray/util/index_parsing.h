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

#ifndef TILEDARRAY_INDEX_PARSING_H__INCLUDED
#define TILEDARRAY_INDEX_PARSING_H__INCLUDED
#include "TiledArray/error.h"
#include <algorithm>
#include <string>
#include <cstring>

namespace TiledArray::detail {

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
/// \param[in] s The string we are splitting into tokens.
/// \param[in] delim The character used to delimit tokens
/// \return A vector containing the tokens
/// \throw std::bad_alloc if there is insufficient memory to allocate the vector
///                       which will hold the return. Strong throw guarantee.
/// \throw std::bad_alloc if there is insufficient memory to allocate the
///                       tokens in the return. Strong throw guarantee.
inline auto tokenize_index(std::string s, char delim) {
  std::vector<std::string> tokens;
  std::stringstream ss(s);
  std::string buffer;
  while(std::getline(ss, buffer, delim))
    tokens.emplace_back(std::move(buffer));
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
  if(split_on_semicolon.size() > 1) return false;


  for(auto x : split_on_semicolon) {
    auto indices = tokenize_index(x, ',');
    for(const auto& idx : indices)
      if(idx.size() == 0) return false;
  }

  return true;
}


/// Defines what it means for a string index to be for a Tensor-of-Tensors
///
/// Tiled Array defines an index
/// @param idx
/// @return
inline bool is_tot_index(const std::string& idx) {
  return idx.find(";") != std::string::npos;
}

inline auto split_tot_index(const std::string& idx) {
  auto temp = tokenize_index(idx, ';');
  TA_ASSERT(temp.size() == 2); // Must have outer and inner separation
  return std::make_pair(tokenize_index(temp[0], ','), tokenize_index(temp[1], ','));
}

} // namespace TiledArray::detail

#endif // TILEDARRAY_INDEX_PARSING_H__INCLUDED
