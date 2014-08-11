/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied war ranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef EXAMPLES_CCD_INPUT_DATA_H__INCLUDED
#define EXAMPLES_CCD_INPUT_DATA_H__INCLUDED

#include <vector>
#include <iosfwd>
#include <madness/world/stdarray.h>
#include <tiledarray.h>

/// Spin enum type
typedef enum {
  alpha = 0,
  beta = 1
} Spin;

/// Occupied or virtual range type
typedef enum {
  occ = 0,
  vir = 1
} RangeOV;

typedef TiledArray::Array<double, 2, TiledArray::Tensor<double>, TiledArray::SparsePolicy> TArray2s;
typedef TiledArray::Array<double, 4, TiledArray::Tensor<double>, TiledArray::SparsePolicy> TArray4s;

/// Read input file and generate tensors for the algorithm
class InputData {
public:
  typedef std::vector<std::size_t> obs_mosym;
  typedef std::vector<std::pair<std::array<std::size_t, 2>, double> > array2d;
  typedef std::vector<std::pair<std::array<std::size_t, 4>, double> > array4d;

private:
  std::string name_;
  unsigned long nirreps_;
  unsigned long nmo_;
  unsigned long nocc_act_alpha_;
  unsigned long nocc_act_beta_;
  unsigned long nvir_act_alpha_;
  unsigned long nvir_act_beta_;
  obs_mosym obs_mosym_alpha_;
  obs_mosym obs_mosym_beta_;
  array2d f_;
  array4d v_ab_;

  template <typename I>
  struct predicate {
    typedef bool result_type;

    predicate(const I& i) : index_(i) { }

    template <typename V>
    result_type operator()(const std::pair<I,V>& data) const { return data.first == index_; }

  private:
    I index_;
  };

  static TiledArray::TiledRange1 make_trange1(const obs_mosym::const_iterator& begin,
      obs_mosym::const_iterator first, obs_mosym::const_iterator last);

  TiledArray::TiledRange trange(const Spin s, const RangeOV ov1, const RangeOV ov2) const;

  TiledArray::TiledRange trange(const Spin s1, const Spin s2, const RangeOV ov1, const RangeOV ov2,
      const RangeOV ov3, const RangeOV ov4) const;

  template <typename R, typename T>
  TiledArray::SparseShape<float> make_sparse_shape(const R& r, const T& t) const {
    TiledArray::Tensor<float> tile_norms(r.tiles(), 0.0f);

    // Find and store the tile for each element in the tensor.
    for(typename T::const_iterator it = t.begin(); it != t.end(); ++it)
        tile_norms[r.element_to_tile(it->first)] += it->second * it->second;

    tile_norms.inplace_unary(& std::sqrt<float>);


    return TiledArray::SparseShape<float>(tile_norms, r);
  }

public:

  InputData(std::ifstream& input);

  std::string name() const { return name_; }

  TArray2s
  make_f(madness::World& w, const Spin s, const RangeOV ov1, const RangeOV ov2);

  TArray4s
  make_v_ab(madness::World& w, const RangeOV ov1, const RangeOV ov2, const RangeOV ov3, const RangeOV ov4);

  TArray2s::value_type
  make_D_vo_tile(const TiledArray::Array<double, 2>::trange_type::tile_range_type& range) const {
    typedef TiledArray::Array<double, 2>::value_type tile_type;
    typedef tile_type::range_type range_type;

    // computes tiles of  D(v,v,o,o)
    tile_type tile(range, 0.0);
    for(range_type::const_iterator it = tile.range().begin(); it != tile.range().end(); ++it)
      tile[*it] = 1.0 / (- f_[(*it)[0]].second + f_[(*it)[1]].second);

    return tile;
  }

  TArray4s::value_type
  make_D_vvoo_tile(const TiledArray::Array<double, 4>::trange_type::tile_range_type& range) const {
    typedef TiledArray::Array<double, 4>::value_type tile_type;
    typedef tile_type::range_type range_type;

    // computes tiles of  D(v,v,o,o)
    tile_type tile(range, 0.0);
    for(range_type::const_iterator it = tile.range().begin(); it != tile.range().end(); ++it)
      tile[*it] = 1.0 / (- f_[(*it)[0]].second - f_[(*it)[1]].second
          + f_[(*it)[2]].second + f_[(*it)[3]].second);

    return tile;
  }

  /// Release allocated data
  void clear() {
    f_.clear();
    array2d().swap(f_);
    v_ab_.clear();
    array4d().swap(v_ab_);
  }
};


#endif // EXAMPLES_CCD_INPUT_DATA_H__INCLUDED
