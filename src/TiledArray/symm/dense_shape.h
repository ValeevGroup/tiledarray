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
 *  Edward Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  symm/dense_shape.h
 *  Oct 21, 2015
 *
 */

#ifndef TILEDARRAY_SYMM_DENSE_SHAPE_H__INCLUDED
#define TILEDARRAY_SYMM_DENSE_SHAPE_H__INCLUDED

#include <memory>
#include <TiledArray/dense_shape.h>

namespace TiledArray {

  namespace symmetry {

    /// SymmetricDenseShape

    /// SymmetricDenseShape is a refinement of DenseShape that also handles permutational and other symmetries.
    /// Symmetry properties of a tensor are specified by the (irreducible) representation of the
    /// group of symmetry operations.
    template <typename SymmetryGroupRepresentation>
    class SymmetricDenseShape : public DenseShape {
      public:
        typedef typename SymmetryGroupRepresentation::representative_type Op;

        /// the default is no symmetry
        SymmetricDenseShape(std::shared_ptr<SymmetryGroupRepresentation> symm =
                            SymmetryGroupRepresentation::trivial()) : rep_(symm) {
        }

        /// Check that a tile index is symmetry-unique, i.e. it is canonically ordered.

        /// As index is canonically ordered if no group element can reduces its lexicographic rank;
        /// in other words, action of group elements never decreases it.
        /// \tparam Index The type of the index
        /// \return true if \c idx is canonically-ordered
        template <typename Index>
        bool is_unique(const Index& idx) { return is_orbit_minimum(idx,*rep_); }

        /// Check that a tile index is symmetry-unique, i.e. it is canonically ordered. \sa is_unique(const Index&)
        template <typename Token>
        bool is_unique(const std::initializer_list<Token>& idx) { return is_unique<std::initializer_list<Token>>(idx); }

        /// Canonicalizes an index

        /// returns the unique Index + the Operator whose action on UniqueIndex produces the input Index
        template <typename Index>
        std::tuple<Index, Op>
        to_unique(const Index& idx) {
          Index unique_idx;
          typename SymmetryGroupRepresentation::element_type to_unique_op;
          std::tie(unique_idx, to_unique_op) = find_orbit_minimum(idx, *rep_);
          return std::make_tuple(unique_idx, rep_->representatives()[to_unique_op]);
        }

      private:
        std::shared_ptr<SymmetryGroupRepresentation> rep_;
    };

  } // namespace TiledArray::symmetry
} // namespace TiledArray

#endif // TILEDARRAY_DENSE_SHAPE_H__INCLUDED
