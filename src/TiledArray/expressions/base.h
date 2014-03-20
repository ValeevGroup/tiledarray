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

#include <TiledArray/expressions/variable_list.h>
#include <TiledArray/expressions/expr_trace.h>

namespace TiledArray {
  namespace expressions {

    /// Base class for expression evaluation

    /// \tparam Derived The derived class type
    template <typename Derived>
    class Base {
    protected:

      VariableList vars_; ///< The variable list for the result of this expression
      bool permute_tiles_; ///< Enables and disables permute

    private:

      // Not allowed
      Base<Derived>& operator=(const Base<Derived>&);

    public:

      typedef Derived derived_type; ///< The derived object type
      typedef typename Derived::dist_eval_type dist_eval_type; ///< The distributed evaluator type

      /// Default constructor
      Base() : vars_(), permute_tiles_(true) { }

      /// Variable list constructor

      /// \param vars The variable list for this expression
      explicit Base(const VariableList& vars) : vars_(vars), permute_tiles_(true) { }

      /// Copy constructor

      /// \param other The expression to be copied
      Base(const Base<Derived>& other) :
        vars_(other.vars_), permute_tiles_(other.permute_tiles_)
      { }

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
        // Set the variable lists for this expression with the variable list of
        // tsr as the target.
        derived().init_vars(tsr.vars());

        // Get the target world
        madness::World& world = (tsr.array().is_initialized() ?
            tsr.array().world() :
            madness::World::get_default());

        // Get the output process map
        std::shared_ptr<typename Tsr<A>::array_type::pmap_interface> pmap;
        if(tsr.array().is_initialized())
          pmap = tsr.array().get_pmap();

        // Create the distributed evaluator from this expression
        dist_eval_type dist_eval = derived().make_dist_eval(world, tsr.vars(), pmap);

        // Create the result array
        typename Tsr<A>::array_type result(dist_eval.get_world(), dist_eval.trange(),
            dist_eval.shape(), dist_eval.pmap());

        // Move the data from disteval into the result array
        typename dist_eval_type::pmap_interface::const_iterator it =
            dist_eval.pmap().begin();
        const typename dist_eval_type::pmap_interface::const_iterator end =
            dist_eval.pmap().end();
        for(; it != end; ++it)
          if(! dist_eval.is_zero(*it))
            result.set(*it, dist_eval.move(*it));

        // Wait for child expressions of dist
        dist_eval.wait();

        // Swap the new array with the result array object.
        tsr.array().swap(result);
      }

      /// Variable list accessor

      /// \return a const reference to the variable list
      const VariableList& vars() const { return vars_; }

      /// Enable or disable the tile permutation

      /// \param permute New state for permute tile flag (true == permute, false == no_permute)
      void permute_tiles(const bool permute) { permute_tiles_ = permute; }

      /// Permute tile accessor
      bool permute_tiles() const { return permute_tiles_; }

      /// Expression print

      /// \param os The output stream
      /// \param target_vars The target variable list for this expression
      void print(ExprOStream& os, const VariableList& target_vars) const {
        if((target_vars != vars_) && permute_tiles_) {
          const Permutation perm = target_vars.permutation(vars_);
          os << "[P" << perm << "] " << derived().print_tag() << " " << Base_::vars_ << "\n";
        } else {
          os << derived().print_tag() << vars_ << "\n";
        }
      }

    private:

      struct ExpressionReduceTag { };

    public:

      template <typename Op>
      madness::Future<typename Op::result_type>
      reduce(const Op& op, madness::World& world = madness::World::get_default()) {
        // Set the variable lists for this expression with the variable list of
        // tsr as the target.
        derived().init_vars();

        // Get the output process map
        std::shared_ptr<typename Tsr<A>::array_type::pmap_interface> pmap;

        // Create the distributed evaluator from this expression
        dist_eval_type dist_eval = derived().make_dist_eval(world,
            derived().vars(), pmap);

        // Create a local reduction task
        TiledArray::detail::ReduceTask<Op> reduce_task(world, op);

        // Move the data from dist_eval into the local reduction task
        typename dist_eval_type::pmap_interface::const_iterator it =
            dist_eval.pmap().begin();
        const typename dist_eval_type::pmap_interface::const_iterator end =
            dist_eval.pmap().end();
        for(; it != end; ++it)
          if(! dist_eval.is_zero(*it))
            reduce_task.add(dist_eval.move(*it));


        typedef madness::TaggedKey<madness::uniqueidT, ExpressionReduceTag> key_type;
        return world.gop.all_reduce(key_type(dist_eval.id()), reduce_task.submit(), op);
      }

    }; // class Base

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_BASE_H__INCLUDED
