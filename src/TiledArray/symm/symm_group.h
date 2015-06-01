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

#ifndef TILEDARRAY_SYMM_SYMM_GROUP_H__INCLUDED
#define TILEDARRAY_SYMM_SYMM_GROUP_H__INCLUDED

#include <TiledArray/permutation.h>

namespace TiledArray {

  /**
   * \addtogroup symmetry
   * @{
   */

  /// Permutation symmetry group
  class SymmGroup {
  private:

    std::shared_ptr<std::vector<Permutation> > elements_; ///< A vector of group permutation

  public:

    // Compiler generated functions
    SymmGroup() = delete;
    SymmGroup(const SymmGroup&) = default;
    SymmGroup(SymmGroup&&) = default;
    SymmGroup& operator=(const SymmGroup&) = default;
    SymmGroup& operator=(SymmGroup&&) = default;

    /// Group constructor

    /// \param degree The number of symmetry elements in the group
    SymmGroup(unsigned int degree) :
      elements_(std::make_shared<std::vector<Permutation> >())
    {
      typedef Permutation::index_type index_type;
      std::vector<Permutation>& elements = *elements_;
      // Add identity to list of elements
      Permutation id = Permutation::identity(degree);
      elements.emplace_back(id);

      // Add generators to the list of elements
      if(degree > 2u) {
        for(unsigned int i = 0u; i < degree; ++i) {

          // Construct the generator
          std::vector<index_type> temp = id.data();
          unsigned int i1 = (i + 1u) % degree;
          temp[i] = i1;
          temp[i1] = i;

          // Add the generator to the list
          elements.emplace_back(std::move(temp));
        }
      } else if(degree == 2u) {
        // Construct the generator
        elements.emplace_back(std::vector<index_type>({1, 0}));
      }

      // Get the number of generators
      unsigned int t = elements_->size();

      // Add the remaining elements to the group
      for(unsigned int g = 1u; g < elements.size(); ++g) {
        for(unsigned int s = 1u; s < t; ++s) {
          Permutation e = elements[g] * elements[s];
          if(std::find(elements.cbegin(), elements.cend(), e) == elements.cend()) {
            elements.emplace_back(std::move(e));
          }
        }
      }
    }

    /// Degree accessor

    /// The degree of the group is the number of symmetry elements in the group.
    /// \return The degree of the group
    unsigned int degree() const { return elements_->front().dim(); }

    /// Order accessor

    /// The order of the group is the number of unique permutation in the group,
    /// and is equal to the factorial of the degree.
    /// \return The order of the group (i.e. the number of unique permutations)
    unsigned int order() const { return elements_->size(); }

    const Permutation& identity() const { return elements_->front(); }

    /// Group element accessor

    /// \param i The group element to be returned
    /// \return The a const reference to the i-th group element
    const Permutation& operator[](unsigned int i) const {
      TA_ASSERT(i < elements_->size());
      return (*elements_)[i];
    }


  }; // class SymmGroup


  /** @}*/

} // namespace TiledArray

#endif // TILEDARRAY_SYMM_SYMM_GROUP_H__INCLUDED
