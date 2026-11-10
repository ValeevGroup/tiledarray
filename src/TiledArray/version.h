/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2016  Virginia Tech
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

#ifndef TILEDARRAY_VERSION_H__INCLUDED
#define TILEDARRAY_VERSION_H__INCLUDED

/** TiledArray version X.Y.Z-id */
#define TILEDARRAY_VERSION "1.1.0"

/** TiledArray major version */
#define TILEDARRAY_MAJOR_VERSION 1

/** TiledArray minor version */
#define TILEDARRAY_MINOR_VERSION 1

/** TiledArray micro version */
#define TILEDARRAY_MICRO_VERSION 0

/** TiledArray prerelease id */
#define TILEDARRAY_PRERELEASE_ID ""

namespace TiledArray {

/** \return a string with the Git SHA1 revision hash tag of TiledArray */
const char* git_revision() noexcept;

/** \return a string with the human-readable description of the current source
 * tree of TiledArray */
const char* git_description() noexcept;

}  // namespace TiledArray

#endif  // TILEDARRAY_VERSION_H__INCLUDED
