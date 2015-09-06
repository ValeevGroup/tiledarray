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
 *  symm_group.h
 *  May 13, 2015
 *
 */

#ifndef TILEDARRAY_SYMM_PERMUTATION_GROUP_H__INCLUDED
#define TILEDARRAY_SYMM_PERMUTATION_GROUP_H__INCLUDED

#include <TiledArray/permutation.h>

namespace TiledArray {

  /**
   * \addtogroup symmetry
   * @{
   */

  /// Permutation group

  /// PermutationGroup is a group of permutations. A permutation group is specified compactly by
  /// a generating set (set of permutations that can multiplicatively generate the entire group).
  class PermutationGroup {
  private:

    /// Group generators
    std::vector<Permutation> generators_;
    /// Group elements
    std::vector<Permutation> elements_;

  public:

    // Compiler generated functions
    PermutationGroup() = delete;
    PermutationGroup(const PermutationGroup&) = default;
    PermutationGroup(PermutationGroup&&) = default;
    PermutationGroup& operator=(const PermutationGroup&) = default;
    PermutationGroup& operator=(PermutationGroup&&) = default;

    /// Symmetric group constructor

    /// This constructs the symmetric group of the given degree, which includes
    /// \em all permutations of \c degree objects
    /// \param degree The number of elements in the set whose symmetry this group describes
    PermutationGroup(unsigned int degree) :
      generators_(),
      elements_(1,Permutation::identity(degree)) // add the permutation
    {
      typedef Permutation::index_type index_type;

      Permutation id = Permutation::identity(degree);
      // Add generators to the list of elements
      if(degree > 2u) {
        for(unsigned int i = 0u; i < degree; ++i) {

          // Construct the generator
          std::vector<index_type> temp = id.data();
          unsigned int i1 = (i + 1u) % degree;
          temp[i] = i1;
          temp[i1] = i;

          // Add the generator to the list
          generators_.emplace_back(std::move(temp));
        }
      } else if(degree == 2u) {
        // Construct the generator
        generators_.emplace_back(std::vector<index_type>({1, 0}));
      }

      init();
    }

    /// General constructor

    /// This constructs a permutation group from a set of generators.
    /// \param degree The number of elements in the set whose symmetry this group describes
    /// \param generators The generating set that defines this group
    PermutationGroup(unsigned int degree, std::vector<Permutation> generators) :
      generators_(std::move(generators)),
      elements_(1,Permutation::identity(degree)) // add the permutation
    {
      init();
    }

    /// Degree accessor

    /// The degree of the group is the number of elements in the set on which the group members act
    /// \return The degree of the group
    unsigned int degree() const { return elements_.front().dim(); }

    /// Order accessor

    /// The order of the group is the number of elements in the group.
    /// For symmetric group \c G the order is factorial of \c G->degree()
    /// \return The order of the group
    unsigned int order() const { return elements_.size(); }

    /// Idenity element accessor

    /// \return the Identity element
    const Permutation& identity() const { return elements_.front(); }

    /// Group element accessor

    /// \param i Index of the group element to be returned, \c 0<=i&&i<order()
    /// \return A const reference to the i-th group element
    const Permutation& operator[](unsigned int i) const {
      TA_ASSERT(i < elements_.size());
      return elements_[i];
    }

    /// Elements vector accessor

    /// \return A const reference to the vector of elements
    const std::vector<Permutation>& elements() const { return elements_; }

    /// Generators vector accessor

    /// \return A const reference to the vector of generators
    const std::vector<Permutation>& generators() const { return generators_; }

    /// forward iterator over the group elements pointing to the first element

    /// \return a std::vector<Permutation>::const_iterator object that points to the first element in the group
    std::vector<Permutation>::const_iterator begin() const {
      return elements_.begin();
    }

    /// forward iterator over the group elements pointing to the first element

    /// \return a std::vector<Permutation>::const_iterator object that points to the first element in the group
    std::vector<Permutation>::const_iterator cbegin() const {
      return elements_.cbegin();
    }

    /// forward iterator over the group elements pointing past the last element

    /// \return a std::vector<Permutation>::const_iterator object that points past the last element in the group
    std::vector<Permutation>::const_iterator end() const {
      return elements_.end();
    }

    /// forward iterator over the group elements pointing past the last element

    /// \return a std::vector<Permutation>::const_iterator object that points past the last element in the group
    std::vector<Permutation>::const_iterator cend() const {
      return elements_.cend();
    }

  private:

    /// uses generators to compute all elements
    void init() {
      using index_type = Permutation::index_type;

      /// add generators to the elements
      for(const auto& g: generators_) {
        elements_.push_back(g);
      }

      // Generate the remaining elements in the group by multiplying by generators
      for(unsigned int g = 1u; g < elements_.size(); ++g) {
        for(const auto& G: generators_) {
          Permutation e = elements_[g] * G;
          if(std::find(elements_.cbegin(), elements_.cend(), e) == elements_.cend()) {
            elements_.emplace_back(std::move(e));
          }
        }
      }
    }

  }; // class PermutationGroup


  /** @}*/

} // namespace TiledArray

#endif // TILEDARRAY_SYMM_SYMM_GROUP_H__INCLUDED
