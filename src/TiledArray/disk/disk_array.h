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
 *
 *  disk_array.h
 */

#ifndef TILEDARRAY_DISK_DISK_ARRAY_H__INCLUDED
#define TILEDARRAY_DISK_DISK_ARRAY_H__INCLUDED

#include <TiledArray/disk/disk_tile.h>
#include <TiledArray/dist_array.h>

namespace TiledArray {

/// An array whose tiles are kept in scratch files and loaded when needed.

/// A DiskArray can be used as an argument of an expression like any other
/// array; each of its tiles is read from disk when the expression needs it
/// and released when it is done with it. To bring the data back into
/// memory assign it to an ordinary array, whole
/// \code
/// DistArray<Tile, Policy> a;
/// a("i,j") = d("i,j");
/// \endcode
/// or by tile block, which reads only the tiles of the block:
/// \code
/// a("i,j") = d("i,j").block(lo, hi);
/// \endcode
/// A DiskArray is produced by assigning to it, whole or by block,
/// \code
/// DiskArray<Tile, Policy> d;
/// d("i,j") = a("i,j");
/// \endcode
/// or tile by tile, by setting its tiles: a tile goes to disk when it is
/// made an element of the array, on the process that owns it:
/// \code
/// d.set(ord, DiskTile<Tile>(tile));
/// \endcode
/// Its tiles are written once and never modified.
///
/// Every process keeps the tiles it owns in a scratch file on its own node
/// (see set_disk_scratch_dir()); the files are used by that process only,
/// are gone when the array is, and do not outlive the process.
template <typename Tile, typename Policy>
using DiskArray = DistArray<DiskTile<Tile>, Policy>;

}  // namespace TiledArray

#endif  // TILEDARRAY_DISK_DISK_ARRAY_H__INCLUDED
