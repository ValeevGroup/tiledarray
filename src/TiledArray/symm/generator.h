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
 *  generator.h
 *  May 13, 2015
 *
 */

#ifndef TILEDARRAY_SYMM_GENERATOR_H__INCLUDED
#define TILEDARRAY_SYMM_GENERATOR_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/symm/group_element.h>

namespace TiledArray {

  /**
   * \addtogroup symmetry
   * @{
   */

  /// Cycle permutation

  /// This permutation contains a list of symmetric or antisymmetric cyclic
  /// permutations. The elements correspond to a row or column of a Young
  /// tableau.
  class CyclePermutation {
  private:
    std::unique_ptr<unsigned int[]> group_; ///< Group of symmetry elements
    unsigned int size_; ///< The number of symmetry elements
    int symmetry_; ///< The symmetry type flag
                   ///< { 1 = symmetric, -1 = antisymmetric }

    template <typename It>
    bool validate(It first, It last) {
      unsigned int prev = *first++;
      bool result = true;
      for(; first != last; ++first) {
        const unsigned int current = *first;
        result = result && (prev < current);
        prev = current;
      }
      return result;
    }

  public:
    CyclePermutation() = delete;
    CyclePermutation(CyclePermutation&& other) = default;
    CyclePermutation& operator=(CyclePermutation&& other) = default;
    ~CyclePermutation() = default;

    /// Cyclic permutation constructor

    /// \param symm_list A list of (anti)symmetric indices
    /// \param symmetry The symmetry type flag (1 = symmetric or
    /// -1 = antisymmetric)  [default = symmetric]
    /// \throw TiledArray::Exception If the \c symmetry is not equal to 1 or -1
    /// \throw TiledArray::Exception If size of the list is less than 2.
    /// \throw TiledArray::Exception If the list is not sorted or if there are
    /// duplicate elements in the list.
    /// \throw std::bad_alloc If memory allocation the generator fails.
    CyclePermutation(const std::initializer_list<unsigned int>& symm_list, const int symmetry = 1) :
      group_(new unsigned int[symm_list.size()]), size_(symm_list.size()),
      symmetry_(symmetry)
    {
      // Verify input
      TA_ASSERT(symmetry == 1 || symmetry == -1);
      TA_ASSERT(symm_list.size() >= 2ul);
      TA_ASSERT(validate(symm_list.begin(), symm_list.end()));
      std::copy(symm_list.begin(), symm_list.end(), group_.get());
    }

    /// Copy constructor

    /// \param other The other generator list to be copied
    /// \throw std::bad_alloc If memory allocation the generator fails.
    CyclePermutation(const CyclePermutation& other) :
      group_(new unsigned int[other.size_]), size_(other.size_),
      symmetry_(other.symmetry_)
    {
      std::copy_n(other.group_.get(), other.size_, group_.get());
    }


    /// Copy assignment operator

    /// \param other The other generator list to be copied
    /// \throw std::bad_alloc If memory allocation the generator fails.
    /// \return A reference to this generator
    CyclePermutation& operator=(const CyclePermutation& other) {
      group_.reset(new unsigned int[other.size_]);
      std::copy_n(other.group_.get(), other.size_, group_.get());
      size_ = other.size_;
      symmetry_ = other.symmetry_;
      return *this;
    }

    /// Element accessor

    /// \param i The element to access
    /// \return The i-th element of the permuation
    /// \throw TiledArray::Exception If \c i is greater than or equal to the
    /// permutation size.
    unsigned int operator[](std::size_t i) const {
      TA_ASSERT(i < size_);
      return group_[i];
    }

    /// Permutation size accessor

    /// \return The number of elements in the permutation
    unsigned int size() const { return size_; }

    /// Symmetry flag accessor

    /// \return The symmetry flag of the generator group, which may be 1 or -1.
    int symmetry() const { return symmetry_; }

    /// Append generator permutations to \c perm_list

    /// This function will append the set of pair-wise permutation generators
    /// from this set of generators to the list of permutations.
    /// \pre The first element of \c perm_list must contain the identity
    /// permutation.
    /// \param[in,out] perm_list The list of group permutations
    /// \throw TiledArray::Exception If the size of \c perm_list is less than 1.
    /// \throw TiledArray::Exception If the first element of \c perm_list is not
    /// an identity permutation.
    /// \throw TiledArray::Exception If the the dimension of the identity
    /// permutation is less than the largest permutation element in this
    /// generator.
    void append_generators(std::vector<GroupElement>& perm_list) const {
      typedef Permutation::index_type index_type;

      TA_ASSERT(perm_list.size() > 0ul);
      TA_ASSERT(perm_list[0].permutation() == \
          Permutation::identity(perm_list[0].permutation().dim()));
      TA_ASSERT(perm_list[0].permutation().dim() > group_[size_ - 1u]);
      TA_ASSERT(perm_list[0].symmetry() == 1);

      for(unsigned int i = 0u; i < size_; ++i) {
        const unsigned int p_i = group_[i];
        for(unsigned int j = i + 1u; j < size_; ++j) {
          const unsigned int p_j = group_[j];

          std::vector<index_type> perm_ij = perm_list[0].permutation().data();

          perm_ij[p_i] = p_j;
          perm_ij[p_j] = p_i;

          perm_list.emplace_back(Permutation(std::move(perm_ij)), symmetry_);
        }
      }
    }

  }; // class CyclePermutation


  /// Symmetric cyclic permutation factory function

  /// \param symm_list A list of permutation indices
  /// \throw TiledArray::Exception If element \f$ e_i \ge e_i+1 \f$ , where
  /// \f$ e_i \f$ is the i-th element of \c symm_list.
  inline CyclePermutation symm(std::initializer_list<unsigned int> symm_list) {
    return CyclePermutation(symm_list, 1);
  }

  /// Antisymmetric cyclic permutation factory function

  /// \param symm_list A list of permutation indices
  /// \throw TiledArray::Exception If element \f$ e_i \ge e_i+1 \f$ , where
  /// \f$ e_i \f$ is the i-th element of \c symm_list.
  inline CyclePermutation antisymm(std::initializer_list<unsigned int> symm_list) {
    return CyclePermutation(symm_list, -1);
  }

  /** @}*/

} // namespace TiledArray

#endif // TILEDARRAY_SYMM_GENERATOR_H__INCLUDED
