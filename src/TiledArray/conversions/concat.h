/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2023  Virginia Tech
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
 *  concat.h
 *  January 26, 2023
 */

#ifndef TILEDARRAY_CONVERSIONS_CONCAT_H
#define TILEDARRAY_CONVERSIONS_CONCAT_H

#include <TiledArray/dist_array.h>

#include <vector>

namespace TiledArray {

/// \brief Concatenates 2 or more DistArrays along one or modes

/// An example of concatenation along single mode think of appending columns
/// of multiple matrices together in a single matrix; clearly, the row ranges
/// (or, rather, tiled ranges in the TA data model)
/// must be identical for all matrices for this operation to make sense.
/// Concatenating multiple matrices as blocks along the diagonal is an
/// example of concatenation along multiple (two, in this case) modes.
/// The notion extends straightforwardly to multidimensional arrays.
/// \param arrays a sequence of DistArray objects to concatenate
/// \param concat_modes specifies whether to concat along a mode or not
/// \param target_world if specified, will use this work, else use
/// `arrays[0].world()` \return the concatenated array
template <typename Tile, typename Policy>
DistArray<Tile, Policy> concat(
    const std::vector<DistArray<Tile, Policy>>& arrays,
    const std::vector<bool>& concat_modes, World* target_world = nullptr) {
  TA_ASSERT(arrays.size() > 1);
  const auto& first_array = arrays[0];
  const std::int64_t r = rank(first_array);
  TA_ASSERT(concat_modes.size() == r);
  for (auto&& arr : arrays) {
    TA_ASSERT(rank(arr) == r);
  }

  if (target_world == nullptr) target_world = &first_array.world();

  std::vector<TiledRange1> tr1s(r);  // result's tr1's
  using index = TiledRange::range_type::index;
  std::vector<std::pair<index, index>>
      tile_begin_end;  // tile coordinates, {beg,end}. for block copy of each
                       // tensor
  tile_begin_end.reserve(arrays.size());
  using std::begin;
  using std::end;

  index b(r), e(r);  // updated for concatted modes only
  std::fill(begin(b), end(b), 0);
  for (auto i = 0ul; i != arrays.size(); ++i) {
    auto& tr = arrays[i].trange();
    if (i == 0) {  // corner case: first array
      for (auto mode = 0; mode != r; ++mode) {
        tr1s[mode] = tr.dim(mode);
        e[mode] = tr1s[mode].tile_extent();
      }
      tile_begin_end.emplace_back(b, e);
    } else {  // for subsequent arrays append tr for concatted dims
      for (auto mode = 0; mode != r; ++mode) {
        if (concat_modes[mode]) {              // concat this mode?
          b[mode] = tr1s[mode].tile_extent();  // end of previous tr1
          tr1s[mode] = concat(tr1s[mode], tr.dim(mode));
          e[mode] = tr1s[mode].tile_extent();
        } else {
          TA_ASSERT(tr1s[mode] == tr.dim(mode));
        }
      }
      tile_begin_end.emplace_back(b, e);
    }
  }

  TiledRange tr(tr1s);
  DistArray<Tile, Policy> result(*target_world, tr);
  const auto annot = detail::dummy_annotation(r);
  for (auto i = 0ul; i != arrays.size(); ++i) {
    if (arrays[i].trange().tiles_range().volume() !=
        0) {  // N.B. empty block range expression bug workaround
      result.make_tsrexpr(annot).block(tile_begin_end[i].first,
                                       tile_begin_end[i].second) =
          arrays[i].make_tsrexpr(annot);
      result.make_tsrexpr(annot).block(tile_begin_end[i].first,
                                       tile_begin_end[i].second) =
          arrays[i].make_tsrexpr(annot);
    }
  }
  result.world().gop.fence();

  return result;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CONVERSIONS_CONCAT_H
