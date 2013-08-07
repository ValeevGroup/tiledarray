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

    // Forward declarations for tensor expression objects
    template <typename> class Base;
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
    template <typename, typename> class TsrMult;
    template <typename, typename> class ScalTsrMult;
    template <typename> class UnaryBase;
    template <typename> class ScalUnaryBase;
    template <typename> class TsrNeg;
    template <typename> class ScalTsrNeg;
    template <typename> class TsrConv;
    template <typename> class ScalTsrConv;

    /// Base class for expression evaluation
    template <typename Derived>
    class Base {
    public:

      typedef Derived derived_type; ///< The derived object type
      typedef typename Derived::disteval_type disteval_type; ///< The distributed evaluator type


      /// Cast this object to it's derived type
      derived_type& derived() { return *static_cast<derived_type*>(this); }

      /// Cast this object to it's derived type
      const derived_type& derived() const { return *static_cast<const derived_type*>(this); }

      /// Evaluate this object and assign it to \c tsr

      /// This expression is evaluated in parallel in distributed environments,
      /// where the content of \c tsr will be replace by the results of the
      /// evaluated tensor expression.
      /// \tparam A The array type
      /// \param tsr The tensor to be assigned
      template <typename A>
      void eval_to(Tsr<A>& tsr) {
        // Get the target world
        madness::World& world = (tsr.array().is_initialized() ?
            tsr.array().world() :
            madness::World::get_default());

        // Get the output process map
        std::shared_ptr<typename Tsr<A>::array_type::pmap_interface> pmap;
        if(tsr.array().is_initialized())
          pmap = tsr.array().get_pmap();

        // Create the distributed evaluator from this expression
        disteval_type disteval = derived().eval(world, vars_, pmap);

        // Create the result array
        typename Tsr<A>::array_type result(disteval.get_world(), disteval.trange(),
            disteval.shape(), disteval.pmap());

        // Move the data from disteval into the result array
        typename Tsr<A>::array_type::pmap_interface::const_iterator it =
            disteval.pmap().begin();
        const typename Tsr<A>::array_type::pmap_interface::const_iterator end =
            disteval.pmap().end();
        for(; it != end; ++it)
          if(! disteval.is_zero(*it))
            result.set(*it, disteval.move(*it));

        // Wait for this expression to finish evaluating
        disteval.wait();

        // Swap the new array with the result array object.
        tsr.array().swap(result);
      }

    }; // class Base

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_BASE_H__INCLUDED
