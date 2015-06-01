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
 *  eval_trait.h
 *  May 26, 2015
 *
 */

#ifndef TILEDARRAY_EVAL_TRAIT_H__INCLUDED
#define TILEDARRAY_EVAL_TRAIT_H__INCLUDED

namespace TiledArray {

  /**
   * \addtogroup TileInterface
   * @{
   */

  /// Determine the object type used in the evaluation of tensor expressions

  /// This trait class allows user to specify the object type used in an
  /// expression by providing a (partial) template specialization of this class
  /// for a user defined tile class. This allows users to use lazy tile
  /// construction inside tensor expressions. If no evaluation type is
  /// specified, the lazy tile evaluation is disabled.
  /// \tparam T The lazy tile type
  /// \tparam Enabler Internal use only
  template <typename T, typename Enabler = void>
  struct eval_trait {
    typedef T type;
  }; // struct eval_trait

  /// Determine the object type used in the evaluation of tensor expressions

  /// This trait class allows user to specify the object type used in an
  /// expression by providing a member type <tt>T::eval_type</tt>. This allows
  /// users to use lazy tile  construction inside tensor expressions. If no
  /// evaluation type is specified, the lazy tile evaluation is disabled.
  /// \tparam T The lazy tile type
  template <typename T>
  struct eval_trait<T, typename std::enable_if<detail::is_type<typename T::eval_type>::value>::type>  {
    typedef typename T::eval_type type;
  }; // struct eval_trait


  /** @}*/

} // namespace TiledArray

#endif // TILEDARRAY_EVAL_TRAIT_H__INCLUDED
