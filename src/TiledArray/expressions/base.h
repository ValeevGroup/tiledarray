/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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

#ifndef TILEDARRAY_EXPRESSIONS_BASE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_BASE_H__INCLUDED

namespace TiledArray {
  namespace expressions {

    // Forward declarations
    template <typename> class Base;
    template <typename> class TsrBase;
    template <typename> class Tsr;
    template <typename> class ScalTsr;
    template <typename> class BinaryBase;
    template <typename> class ScalBinaryBase;
    template <typename, typename> class TsrAdd;
    template <typename, typename> class ScalTsrAdd;
    template <typename, typename> class TsrSubt;
    template <typename, typename> class ScalTsrSubt;
    template <typename, typename> class TsrCont;
    template <typename, typename> class ScalTsrCont;
    template <typename> class UnaryBase;
    template <typename> class ScalUnaryBase;
    template <typename> class TsrNeg;
    template <typename> class ScalTsrNeg;


    template <typename Derived>
    class Base {
    public:

      typedef Derived derived_type;

      derived_type& derived() { return *static_cast<derived_type*>(this); }
      const derived_type& derived() const { return *static_cast<const derived_type*>(this); }

      Base<Derived>& operator=(const Base<Derived>& other) {
        derived().eval_to(other.derived());
        return *this;
      }

      template <typename D>
      Base<Derived>& operator=(const Base<D>& other) {
        derived().eval_to(other.derived());
        return *this;
      }

    }; // class Base

  } // namespace expressions
} // namespace TiledArray


#endif // TILEDARRAY_EXPRESSIONS_BASE_H__INCLUDED
