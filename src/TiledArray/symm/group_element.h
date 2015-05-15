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
 *  group_element.h
 *  May 14, 2015
 *
 */

#ifndef TILEDARRAY_SYMM_GROUP_ELEMENT_H__INCLUDED
#define TILEDARRAY_SYMM_GROUP_ELEMENT_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/permutation.h>

namespace TiledArray {

  /**
   * \addtogroup symmetry
   * @{
   */

  /// Permutation group element

  /// An element contains a permutation as well as the symmetry flag. The
  /// symmetry flag may be 1 or -1.
  class GroupElement {
    Permutation perm_; ///< The permutation of the group element
    int symmetry_;

  public:
    // Compiler generated functions
    GroupElement() = delete;
    GroupElement(const GroupElement&) = default;
    GroupElement(GroupElement&&) = default;
    ~GroupElement() = default;
    GroupElement& operator=(const GroupElement&) = default;
    GroupElement& operator=(GroupElement&&) = default;

    /// Element constructor

    /// \param perm The permutation of the group element
    /// \param symmetry The symmetry flag of the group element
    GroupElement(const Permutation& perm, const int symmetry) :
      perm_(perm), symmetry_(symmetry)
    {
      TA_ASSERT(perm_);
      TA_ASSERT(symmetry == 1 || symmetry == -1);
    }

    /// Element constructor

    /// \param perm The permutation of the group element
    /// \param symmetry The symmetry flag of the group element
    GroupElement(Permutation&& perm, const int symmetry) :
      perm_(std::move(perm)), symmetry_(symmetry)
    {
      TA_ASSERT(perm_);
      TA_ASSERT(symmetry == 1 || symmetry == -1);
    }

    /// Permutation accessor

    /// \return A const reference to the permutation
    const Permutation& permutation() const { return perm_; }

    /// Symmetry flag accessor

    /// \return 1 for symmetric permutation or -1 for antisymmetric permutation
    int symmetry() const { return symmetry_; }

  }; // class GroupElement

  /// Make the product of two group elements

  /// \param g1 The right-hand group element
  /// \param g2 The right-hand group element
  /// \return The product of g1 and g2, where the resulting permutation is equal
  /// to <tt>g1.permutation() * g2.permutation()</tt> and the resulting symmetry
  /// flag is equal to <tt>g1.symmetry() * g2.symmetry()</tt>.
  GroupElement operator*(const GroupElement& g1, const GroupElement& g2) {
    return GroupElement(g1.permutation() * g2.permutation(),
        g1.symmetry() * g2.symmetry());
  }

  /// Group element equality operator

  /// \param g1 The right-hand group element
  /// \param g2 The right-hand group element
  /// \return <tt>g1.permutation() == g2.permutation() and</tt>
  /// <tt>g1.symmetry() == g2.symmetry()</tt>.
  bool operator==(const GroupElement& g1, const GroupElement& g2) {
    return (g1.permutation() == g2.permutation())
        && (g1.symmetry() == g2.symmetry());
  }

  /// Group element inequality operator

  /// \param g1 The right-hand group element
  /// \param g2 The right-hand group element
  /// \return <tt>g1.permutation() != g2.permutation() or</tt>
  /// <tt>g1.symmetry() != g2.symmetry()</tt>.
  bool operator!=(const GroupElement& g1, const GroupElement& g2) {
    return (g1.permutation() != g2.permutation())
        || (g1.symmetry() != g2.symmetry());
  }

  /// Group element less-than operator

  /// \param g1 The right-hand group element
  /// \param g2 The right-hand group element
  /// \return <tt>g1.permutation() < g2.permutation()</tt>.
  bool operator<(const GroupElement& g1, const GroupElement& g2) {
    return g1.permutation() < g2.permutation();
  }


  /** @}*/

} // namespace TiledArray

#endif // TILEDARRAY_SYMM_GROUP_ELEMENT_H__INCLUDED
