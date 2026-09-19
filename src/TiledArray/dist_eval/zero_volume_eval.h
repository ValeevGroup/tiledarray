/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
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
 */

#ifndef TILEDARRAY_DIST_EVAL_ZERO_VOLUME_EVAL_H__INCLUDED
#define TILEDARRAY_DIST_EVAL_ZERO_VOLUME_EVAL_H__INCLUDED

#include <TiledArray/dist_eval/dist_eval.h>

namespace TiledArray {
namespace detail {

/// A distributed evaluator over a zero-volume tiled range: it owns no tiles,
/// so evaluation produces nothing and no tile can be requested. The general
/// (fused x contracted) product evaluates a zero-volume result through it,
/// since its SUMMA evaluator needs a process grid and ProcGrid requires at
/// least one row and one column (see ContEngine::init_distribution_general).
/// \tparam Tile The output tile type
/// \tparam Policy The tensor policy class
template <typename Tile, typename Policy>
class ZeroVolumeEvalImpl final : public DistEvalImpl<Tile, Policy> {
 public:
  typedef DistEvalImpl<Tile, Policy> DistEvalImpl_;  ///< The base class type
  typedef typename DistEvalImpl_::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename DistEvalImpl_::trange_type trange_type;    ///< Tiled range
  typedef typename DistEvalImpl_::shape_type shape_type;      ///< Shape type
  typedef typename DistEvalImpl_::pmap_interface pmap_interface;  ///< Pmap
  typedef typename DistEvalImpl_::value_type value_type;          ///< Tile

  /// \param world The world of the result
  /// \param trange The result tiled range; its tile range must be empty
  /// \param shape The result shape
  /// \param pmap The result process map
  ZeroVolumeEvalImpl(World& world, const trange_type& trange,
                     const shape_type& shape,
                     const std::shared_ptr<const pmap_interface>& pmap)
      : DistEvalImpl_(world, trange, shape, pmap, Permutation{}) {
    TA_ASSERT(trange.tiles_range().volume() == 0);
  }

  Future<value_type> get_tile(ordinal_type) const override {
    TA_EXCEPTION("ZeroVolumeEvalImpl owns no tiles");
    return Future<value_type>();
  }

  void discard_tile(ordinal_type) const override {}

  int internal_eval() override { return 0; }
};

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_DIST_EVAL_ZERO_VOLUME_EVAL_H__INCLUDED
