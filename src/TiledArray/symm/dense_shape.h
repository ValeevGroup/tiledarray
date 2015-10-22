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
  template <typename SymmetryInfo>
  class SymmetricDenseShape : public DenseShape {
    public:
      /// the default is no symmetry
      SymmetricDenseShape(std::shared_ptr<SymmetryInfo> symm = SymmetryInfo::trivial()) : symm_(symm) {
      }

      /// Check that a tile is symmetry-unique, i.e. its index is canonical

      /// \tparam Index The type of the index
      /// \return false
      template <typename Index>
      bool is_unique(const Index& idx) { return is_lexicographically_smallest(idx,*symm_); }

    private:
      std::shared_ptr<SymmetryInfo> symm_;
  };

  } // namespace TiledArray::symmetry
} // namespace TiledArray

#endif // TILEDARRAY_DENSE_SHAPE_H__INCLUDED
