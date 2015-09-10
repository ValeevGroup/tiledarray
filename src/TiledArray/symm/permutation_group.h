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
 *  permutation_group.h
 *  May 13, 2015
 *
 */

#ifndef TILEDARRAY_SYMM_PERMUTATION_GROUP_H__INCLUDED
#define TILEDARRAY_SYMM_PERMUTATION_GROUP_H__INCLUDED

#include <TiledArray/symm/permutation.h>

namespace TiledArray {

  /**
   * \addtogroup symmetry
   * @{
   */

  /// Permutation group

  /// PermutationGroup is a group of permutations. A permutation group is specified compactly by
  /// a generating set (set of permutations that can multiplicatively generate the entire group).
  class PermutationGroup {
  public:
    using Permutation = TiledArray::symmetry::Permutation;

  protected:

    /// Group generators
    std::vector<Permutation> generators_;
    /// Group elements
    std::vector<Permutation> elements_;

  public:

    // Compiler generated functions
    PermutationGroup(const PermutationGroup&) = default;
    PermutationGroup(PermutationGroup&&) = default;
    PermutationGroup& operator=(const PermutationGroup&) = default;
    PermutationGroup& operator=(PermutationGroup&&) = default;

    /// General constructor

    /// This constructs a permutation group from a set of generators.
    /// The order of generators does not matter, and repeated generators
    /// will be ignored (internally generators are stored as a sorted sequence).
    /// \param degree The number of elements in the set whose symmetry this group describes
    /// \param generators The generating set that defines this group
    PermutationGroup(std::vector<Permutation> generators) :
      generators_(std::move(generators))
    {
      init();
    }

    /// Group order accessor

    /// The order of the group is the number of elements in the group.
    /// For symmetric group \c G the order is factorial of \c G->degree()
    /// \return The order of the group
    unsigned int order() const { return elements_.size(); }

    /// Idenity element accessor

    /// \return the Identity element
    const Permutation& identity() const { return elements_.front(); }

    /// Group element accessor

    /// \note Elements are ordered lexicograhically.
    /// \param i Index of the group element to be returned, \c 0<=i&&i<order()
    /// \return A const reference to the i-th group element
    const Permutation& operator[](unsigned int i) const {
      TA_ASSERT(i < elements_.size());
      return elements_[i];
    }

    /// Elements vector accessor

    /// \note Elements appear in lexicograhical order.
    /// \return A const reference to the vector of elements
    const std::vector<Permutation>& elements() const { return elements_; }

    /// Generators vector accessor

    /// \note Generators appear in lexicograhical order.
    /// \return A const reference to the vector of generators
    const std::vector<Permutation>& generators() const { return generators_; }

    /// @name Iterator accessors

    /// PermutationGroup iterators dereference to group elements, i.e. Permutation objects.
    /// Iterators can be used to iterate over group elements in lexicographical order. \sa operator[]
    /// @{

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

    /// @}

    /// Computes the domain of this group

    /// \tparam Set a container type in which the result will be returned (e.g. \c std::set )
    /// \return the domain of this permutation, as a sorted sequence
    template <typename Set>
    Set domain() const {
      Set result;
      // sufficient to loop over generators
      for(const auto& e: generators_) {
        const auto e_domain = e.domain<Set>();
        result.insert(e_domain.begin(), e_domain.end());
      }
      return result;
    }


  protected:

    PermutationGroup() {} // makes uninitialized group, all initialization is left to the derived class

    /// uses generators to compute all elements
    /// \note generators that appear more than once are removed; generators are then resorted!
    void init() {

      sort(generators_.begin(), generators_.end());
      auto unique_last = std::unique(generators_.begin(), generators_.end());
      generators_.erase(unique_last, generators_.end());
      { // eliminate identity from the generator list
        auto I_iter = std::find(generators_.begin(), generators_.end(), Permutation());
        if (I_iter != generators_.end())
          generators_.erase(I_iter);
      }

      using index_type = Permutation::index_type;

      // add the identity element first
      elements_.emplace_back();

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

      sort(elements_.begin(), elements_.end());
    }

  }; // class PermutationGroup

  /// PermutationGroup equality operator

  /// \param p1 The left-hand permutation group to be compared
  /// \param p2 The right-hand permutation group to be compared
  /// \return \c true if all elements of \c p1 and \c p2 are equal, otherwise \c false.
  inline bool operator==(const PermutationGroup& p1, const PermutationGroup& p2) {
    return (p1.order() == p2.order())
           && p1.elements() == p2.elements();
  }

  /// PermutationGroup inequality operator

  /// \param p1 The left-hand permutation group to be compared
  /// \param p2 The right-hand permutation group to be compared
  /// \return \c true if any element of \c p1 is not equal to that of \c p2,
  /// otherwise \c false.
  inline bool operator!=(const PermutationGroup& p1, const PermutationGroup& p2) {
    return ! operator==(p1, p2);
  }

  /// PermutationGroup less-than operator

  /// \param p1 The left-hand permutation group to be compared
  /// \param p2 The right-hand permutation group to be compared
  /// \return \c true if the elements of \c p1 are lexicographically less than
  /// that of \c p2, otherwise \c false.
  inline bool operator<(const PermutationGroup& p1, const PermutationGroup& p2) {
    return std::lexicographical_compare(p1.cbegin(), p1.cend(),
        p2.cbegin(), p2.cend());
  }

  /// Symmetric group

  /// Symmetric group of degree \f$ n \f$ is a group of \em all permutations of set \f$ \{x_0, x_1, \dots x_{n-1}\} \f$
  /// where \f$ x_i \f$ are nonnegative integers.
  class SymmetricGroup final: public PermutationGroup {
    public:
      using index_type = Permutation::index_type;

      // Compiler generated functions
      SymmetricGroup() = delete;
      SymmetricGroup(const SymmetricGroup&) = default;
      SymmetricGroup(SymmetricGroup&&) = default;
      SymmetricGroup& operator=(const SymmetricGroup&) = default;
      SymmetricGroup& operator=(SymmetricGroup&&) = default;

      /// Construct symmetric group on domain \f$ \{0, 1, \dots n-1\} \f$, where \f$ n \f$ = \c degree
      /// \param degree the degree of this group
      SymmetricGroup(unsigned int degree) :
        SymmetricGroup(SymmetricGroup::iota_vector(degree))
      {
      }

      /// Construct symmetric group on domain \c [begin,end)
      /// \tparam InputIterator an input iterator type
      /// \param begin iterator pointing to the beginning of the range
      /// \param end iterator pointing to past the end of the range
      template <typename InputIterator,
                typename std::enable_if<detail::is_input_iterator<InputIterator>::value>::type* = nullptr>
      SymmetricGroup(InputIterator begin, InputIterator end) :
        PermutationGroup(), domain_(begin, end)
      {
        for(auto iter=begin; iter!=end; ++iter) {
          TA_ASSERT(*iter >= 0);
        }

        const auto degree = domain_.size();

        // Add generators to the list of elements
        if(degree > 2u) {
          for(unsigned int i = 0u; i < degree; ++i) {
            // Construct the generator and add to the list
            unsigned int i1 = (i + 1u) % degree;
            generators_.emplace_back(Permutation::Map{{domain_[i],domain_[i1]},{domain_[i1],domain_[i]}});
          }
        } else if(degree == 2u) {
          // Construct the generator
          generators_.emplace_back(std::vector<index_type>({domain_[1], domain_[0]}));
        }

        init();
      }

      /// Construct symmetric group using domain as an initializer list

      /// \tparam Integer an integral type
      /// \param list An initializer list of Integer
      template <typename Integer,
                typename std::enable_if<std::is_integral<Integer>::value>::type* = nullptr>
      explicit SymmetricGroup(std::initializer_list<Integer> list) :
          SymmetricGroup(list.begin(), list.end())
      {
      }

      /// Degree accessor

      /// The degree of the group is the number of elements in the set on which the group members act
      /// \return The degree of the group
      unsigned int degree() const { return domain_.size(); }

    private:
      std::vector<index_type> domain_;

      /// make vector {0, 1, ... n-1}
      static std::vector<index_type> iota_vector(size_t n) {
        std::vector<index_type> result(n);
        std::iota(result.begin(), result.end(), 0);
        return result;
      }

      SymmetricGroup(const std::vector<index_type>& domain) :
        SymmetricGroup(domain.begin(), domain.end())
      {
      }


  };

  /// determines whether a given Index is lexicographically smallest
  /// among all indices generated by the action of \c pg.
  /// \tparam Index a sequence type that is directly addressable, i.e. has a fast \c operator[]
  /// \param idx an Index object
  /// \param pg the PermutationGroup
  /// \return \c false if action of a permutation in \c pg can produce
  ///            an Index that is lexicographically smaller than \c idx (i.e. there exists
  ///            \c i such that \c pg[i]*idx is lexicographically less than \c idx), \c true otherwise
  template <typename Index>
  bool is_lexicographically_smallest(const Index& idx,
                                     const PermutationGroup& pg) {
    using index_type = PermutationGroup::Permutation::index_type;
    const auto idx_size = idx.size();
    for(const auto& p: pg) {
      for(size_t i=0; i!=idx_size; ++i) {
        auto idx_i = idx[i];
        auto idx_p_i = idx[p[i]];
        if (idx_p_i < idx_i)
          return false;
        if (idx_p_i > idx_i)
          break;
      }
    }
    return true;
  }

  /** @}*/

} // namespace TiledArray

#endif // TILEDARRAY_SYMM_PERMUTATION_GROUP_H__INCLUDED
