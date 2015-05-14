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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  generator.h
 *  May 13, 2015
 *
 */

#ifndef TILEDARRAY_GENERATOR_H__INCLUDED
#define TILEDARRAY_GENERATOR_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/type_traits.h>

namespace TiledArray {

  /// (Anti)symmetric group generator

  /// This generator contains a list of symmetric or anti
  class Generator {
  private:
    std::unique_ptr<unsigned int[]> group_; ///< Group of symmetry elements
    unsigned int size_; ///< The number of symmetry elements
    unsigned int symmetry_; ///< The symmetry type flag
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
    Generator() = delete;

    /// Generator constructor

    /// \param symm_list A list of (anti)symmetric indices
    /// \param symmetry The symmetry type flag (1 = symmetric, -1 = antisymmetric)
    /// [default = symmetric]
    Generator(const std::initializer_list<unsigned int>& symm_list, const int symmetry = 1) :
      group_(new unsigned int[symm_list.size()]), size_(symm_list.size()),
      symmetry_(symmetry)
    {
      // Varify input
      TA_ASSERT(symmetry == 1 || symmetry == -1);
      TA_ASSERT(symm_list.size() >= 2ul);
      TA_ASSERT(validate(symm_list.begin(), symm_list.end()));
      std::copy(symm_list.begin(), symm_list.end(), group_.get());
    }

    Generator(const Generator& other) :
      group_(new unsigned int[other.size_]), size_(other.size_),
      symmetry_(other.symmetry_)
    {
      std::copy_n(other.group_.get(), other.size_, group_.get());
    }

    Generator(Generator&& other) = default;

    Generator& operator=(const Generator& other) {
      group_.reset(new unsigned int[other.size_]);
      std::copy_n(other.group_.get(), other.size_, group_.get());
      size_ = other.size_;
      symmetry_ = other.symmetry_;
      return *this;
    }

    Generator& operator=(Generator&& other) = default;

    ~Generator() = default;

    unsigned int operator[](std::size_t i) const { return group_[i]; }

    unsigned int size() const { return size_; }

    int symmetry() const { return symmetry_; }


  }; // class Generator


  /// Symmetric group generator factory function

  /// \param symm_list A list of symmetric indices
  inline Generator symm(std::initializer_list<unsigned int> symm_list) {
    return Generator(symm_list, 1);
  }

  /// Antiymmetric group generator factory function

  /// \param symm_list A list of antisymmetric indices
  inline Generator antisymm(std::initializer_list<unsigned int> symm_list) {
    return Generator(symm_list, -1);
  }



} // namespace TiledArray

#endif // TILEDARRAY_GENERATOR_H__INCLUDED
