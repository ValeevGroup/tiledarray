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
 *  Edward F Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  representation.h
 *  May 13, 2015
 *
 */

#ifndef TILEDARRAY_SYMM_REPRESENTATION_H__INCLUDED
#define TILEDARRAY_SYMM_REPRESENTATION_H__INCLUDED

#include <map>
#include <vector>
#include <memory>

#include <TiledArray/error.h>

namespace TiledArray {
  namespace symmetry {

    /// identity for group of objects of type T
    template <typename T> T identity();

    /// class Representation is a representation of Group in terms of Representatives (linear operators)
    /// \tparam Group class describing the group of symmetry transformations
    /// \tparam Representative class describing the group representatives; in TiledArray these will
    ///         encode mathematical transformation of tiles under permutations, or symmetry transformations.
    template <typename Group, typename Representative>
    class Representation {
      public:
        using group_type = Group;
        using element_type = typename Group::element_type;
        using representative_type = Representative;

        // Compiler generated functions
        Representation(const Representation&) = default;
        Representation(Representation&&) = default;
        Representation& operator=(const Representation&) = default;
        Representation& operator=(Representation&&) = default;

        /// Construct Representation from a set of {generator,operator} pairs construct operator representation of the permutation group
        Representation(std::map<element_type,representative_type> generator_reps = std::map<element_type,representative_type>{}) :
          generator_representatives_(std::move(generator_reps))
        {
          // N.B. this may mutate generator_reps, e.g. if it has an identity
          init(generator_representatives_, element_representatives_);
        }

        static std::shared_ptr<Representation> trivial() {
          std::map<element_type,representative_type> empty_generator_list;
          return std::make_shared<Representation>(empty_generator_list);
        }

        /// the order of the representation = the order of the group
        size_t order() const {
          return element_representatives_.size();
        }

        /// constructs Group object from this
        /// \note this redoes all the work that the constructor did
        std::shared_ptr<group_type> group() const {

          // extract generators and elements
          std::vector<element_type> generators;
          generators.reserve(generator_representatives_.size());
          for(const auto& g_op_pair: generator_representatives_) {
            generators.emplace_back(g_op_pair.first);
          }

          return std::make_shared<group_type>(std::move(generators));
        }

        const std::map<element_type,representative_type>& representatives() const {
          return element_representatives_;
        }

      private:
        std::shared_ptr<group_type> g_;
        std::map<element_type,representative_type> generator_representatives_;
        std::map<element_type,representative_type> element_representatives_;

        /// uses generators to compute representatives for every element
        static void init(std::map<element_type,representative_type>& generator_reps,
                         std::map<element_type,representative_type>& element_reps) {

          // make sure identity is not among the generators
          TA_ASSERT(generator_reps.end() ==
                    generator_reps.find(group_type::identity()));

          // make element->operator map
          // start with generator->operator map
          element_reps = generator_reps;

          // then add the identity element
          element_reps[group_type::identity()] = identity<representative_type>();

          // Generate the remaining elements in the group by multiplying by generators
          // to keep track of elements already added make an extra vector
          std::vector<element_type> elements;
          for(const auto& eop: element_reps)
            elements.push_back(eop.first);

          for(size_t i = 0; i < elements.size(); ++i) {
            const auto& e = elements[i];
            const auto& e_op = element_reps[e];
            for(const auto& g_op_pair: generator_reps) {
              const auto& g = g_op_pair.first;
              const auto& g_op = g_op_pair.second;
              auto h = e * g;
              if(element_reps.find(h) == element_reps.end()) {
                auto h_op = e_op * g_op;
                element_reps[h] = h_op;
                elements.emplace_back(std::move(h));
              }
            }
          }
        } // init()

    }; // class Representation

    /// determines whether a given MultiIndex is lexicographically smallest
    /// among all indices generated by the action of elements of the group
    /// corresponding to \c rep.
    ///
    /// \tparam MultiIndex a sequence type that is directly addressable, i.e. has a fast \c operator[]
    /// \tparam Group group of elements, each of which can act on MultiIndex using operator[] (e.g. PermutationGroup)
    /// \tparam Representative representative type
    /// \param idx a MultiIndex object
    /// \param rep a Representation of Group formed by Representative objects
    /// \return \c false if action of group operation in \c rep can produce
    ///            an Index that is lexicographically smaller than \c idx (i.e. there exists
    ///            \c i such that \c rep.group[i]*idx is lexicographically less than \c idx),
    ///            \c true otherwise
    template <typename MultiIndex,typename Group, typename Representative>
    bool is_lexicographically_smallest(const MultiIndex& idx,
                                       const Representation<Group,Representative>& rep) {
      const auto idx_size = idx.size();
      for(const auto& g_op_pair: rep.representatives()) {
        for(size_t i=0; i!=idx_size; ++i) {
          const auto& g = g_op_pair.first;
          auto idx_i = idx[i];
          auto idx_g_i = idx[g[i]];
          if (idx_g_i < idx_i)
            return false;
          if (idx_g_i > idx_i)
            break;
        }
      }
      return true;
    }

  } // namespace symmetry
} // namespace TiledArray

#endif /* TILEDARRAY_SYMM_REPRESENTATION_H__INCLUDED */
