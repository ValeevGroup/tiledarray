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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  cast.h
 *  Jan 10, 2016
 *
 */

#ifndef TILEDARRAY_TILE_INTERFACE_CAST_H__INCLUDED
#define TILEDARRAY_TILE_INTERFACE_CAST_H__INCLUDED

#include "../type_traits.h"

namespace TiledArray {

  template <typename, typename> class Cast;

  namespace tile_interface {

    /// Internal cast implementation

    /// This class is used to define internal tile cast operations. Users may
    /// specialize the `TiledArray::Cast` class.
    /// \tparam Result The output tile type
    /// \tparam Arg The input tile type
    /// \tparam Enabler Enabler type used to select (partial) specializations
    template <typename Result, typename Arg, typename Enabler = void>
    class Cast {
    public:

      typedef Result result_type; ///< Result tile type
      typedef Arg argument_type; ///< Argument tile type

      result_type operator()(const argument_type& arg) const {
        return static_cast<result_type>(arg);
      }

    }; // class Cast

    /// Internal cast implementation

    /// This class is used to define internal tile cast operations. This
    /// specialization handles casting of lazy tiles to a type other than the
    /// evaluation type.
    /// \tparam Result The output tile type
    /// \tparam Arg The input tile type
    template <typename Result, typename Arg>
    class Cast<Result, Arg,
        typename std::enable_if<
            is_lazy_tile<Arg>::value &&
            ! std::is_same<Result, typename TiledArray::eval_trait<Arg>::type>::value
        >::type> :
        public TiledArray::Cast<Result, typename TiledArray::eval_trait<Arg>::type>
    {
    private:
      typedef Cast<Result, typename TiledArray::eval_trait<Arg>::type>
          Cast_; ///< Base class type
      typedef typename TiledArray::eval_trait<Arg>::type
          eval_type; ///< Lazy tile evaluation type
    public:

      typedef Result result_type; ///< Result tile type
      typedef Arg argument_type; ///< Argument tile type

      /// Tile cast operation

      /// Cast arg to a `result_type` tile
      /// \param arg The tile to be cast
      /// \return A cast copy of `arg`
      result_type operator()(const argument_type& arg) const {
        return Cast_::operator()(static_cast<eval_type>(arg));
      }

    }; // class Cast

  } // namespace tile_interface


  /// Tile cast operation

  /// This class is used to define tile cast operations. Users may specialize
  /// this class for arbitrary tile type conversion operations.
  /// \tparam Result The output tile type
  /// \tparam Arg The input tile type
  template <typename Result, typename Arg>
  class Cast : public TiledArray::tile_interface::Cast<Result, Arg> { };


} // namespace TiledArray

#endif // TILEDARRAY_TILE_INTERFACE_CAST_H__INCLUDED
