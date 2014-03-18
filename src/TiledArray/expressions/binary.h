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

#ifndef TILEDARRAY_EXPRESSIONS_BINARY_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_BINARY_H__INCLUDED

#include <TiledArray/expressions/base.h>

namespace TiledArray {
  namespace expressions {

    template <typename, typename>
    struct BinaryExprPolicyHelper { }; // Different policy types is an error

    template <typename Policy>
    struct BinaryExprPolicyHelper<Policy, Policy> {
      typedef Policy policy;
    };

    template <typename Left, typename Right>
    struct BinaryExprPolicy :
        public BinaryExprPolicyHelper<typename Left::policy, typename Right::policy>
    { };

    template <typename Derived>
    class Binary : public Base<Derived> {
    private:
      typedef Base<Derived> Base_;

    public:
      typedef typename Derived::left_type left_type; ///< The left-hand expression type
      typedef typename Derived::right_type right_type; ///< The right-hand expression type
      typedef typename Derived::dist_eval_type dist_eval_type; ///< This expression's distributed evaluator type

      static const bool consumable = true;
      static const unsigned int leaves = left_type::leaves + right_type::leaves;

    protected:
      left_type left_; ///< The left-hand argument
      right_type right_; ///< The right-hand argument

    private:

      // Not allowed
      Binary<Derived>& operator=(const Binary<Derived>&);

    public:

      /// Binary expression constructor
      Binary(const left_type& left, const right_type& right) const :
        Base_(), left_(left), right_(right)
      { }

      /// Copy constructor
      Binary(const Binary<Derived>& other) :
        Base_(other), left_(other.left_), right_(other.right_)
      { }

      // Pull base class functions into this class.
      using Base_::vars;

      /// Left-hand expression argument accessor

      /// \return A const reference to the left-hand expression object
      const left_type& left() const { return left_; }

      /// Right-hand expression argument accessor

      /// \return A const reference to the right-hand expression object
      const right_type& right() const { return right_; }

      /// Set the variable list for this expression

      /// This function will set the variable list for this expression and its
      /// children such that the number of permutations is minimized. The final
      /// variable list may not be set to target, which indicates that the
      /// result of this expression will be permuted to match \c target_vars.
      /// \param target_vars The target variable list for this expression
      void vars(const VariableList& target_vars) {
        // Determine the equality of the variable lists
        bool left_target = true, right_target = true, left_right = true;
        for(unsigned int i = 0u; i < target_vars.dim(); ++i) {
          left_target = left_target && left_.vars()[i] == target_vars[i];
          right_target = right_target && right_.vars()[i] == target_vars[i];
          left_right = left_right && left_.vars()[i] == right_.vars()[i];
        }

        if(left_right) {
          Base_::vars_ = left_.vars();
        } else {
          // Determine which argument will be permuted
          const bool perm_left = (right_target || ((! (left_target || right_target))
              && (left_type::leaves <= right_type::leaves)));

          if(perm_left) {
            Base_::vars_ = right_.vars();
            left_.vars(right_.target());
          } else {
            Base_::vars_ = left_.vars();
            right_.vars(left_.vars());
          }
        }
      }

      /// Initialize the variable list of this expression

      /// \param target_vars The target variable list for this expression
      void init_vars(const VariableList& target_vars) {
        left_.init_vars(target_vars);
        right_.init_vars(target_vars);
        vars(target_vars);
      }

      /// Initialize the variable list of this expression
      void init_vars() {
        if(left_type::leaves <= right_type::leaves) {
          left_.init_vars();
          Base_::vars_ = left_.vars();
          right_.vars(Base_::vars_);
        } else {
          right_.init_vars();
          Base_::vars_ = right_.vars();
          left_.vars(Base_::vars_);
        }
      }

      /// Construct the distributed evaluator for this expression

      /// \param world The world where this expression will be evaluated
      /// \param vars The variable list for the output data layout
      /// \param pmap The process map for the output
      /// \return The distributed evaluator that will evaluate this expression
      dist_eval_type make_dist_eval(madness::World& world, const VariableList& target_vars,
          const std::shared_ptr<typename dist_eval_type::pmap_interface>& pmap) const
      {
        typedef BinaryEvalImpl<typename left_type::dist_eval_type,
            typename right_type::dist_eval_type, op_type, policy>
          binary_impl_type;
        // Verify input
        TA_ASSERT(pmap->procs() == world.size());

        // Construct left and right distributed evaluators
        const typename left_type::dist_eval_type left =
            left_.make_dist_eval(world, Base_::vars_, pmap);
        if(pmap == NULL)
          pmap = left.pmap();
        const typename right_type::dist_eval_type right =
            right_.make_dist_eval(world, Base_::vars_, pmap);

        // Check that the tiled ranges of the left- and right-hand arguments are equal.
#ifndef NDEBUG
        if(left.trange() != right.trange()) {
          if(left.get_world().rank() == 0) {
            TA_USER_ERROR_MESSAGE( "The left- and right-hand tiled ranges are not equal, "
                << Base_::vars_ << " is not compatible with the expected output, "
                << target_vars << "." );
          }

          TA_EXCEPTION("Incompatible TiledRange objects were given in left- and right-hand expressions.");
        }
#endif // NDEBUG


        // Construct the distributed evaluator type
        std::shared_ptr<typename dist_eval_type::impl_type> pimpl;
        if(Base_::vars_ != target_vars) {
          // Determine the permutation that will be applied to the result, if any.
          Permutation perm = target_vars.permutation(Base_::vars_);

          pimpl.reset(new binary_impl_type(left, right, world, perm ^ left.trange(),
              derived().make_shape(left.shape(), right.shape(), perm), pmap, perm,
              derived().make_tile_op(perm)));
        } else {
          pimpl.reset(new binary_impl_type(left, right, world, left.trange(),
              derived().make_shape(left.shape(), right.shape()), pmap, perm,
              derived().make_tile_op()));
        }

        return dist_eval_type(pimpl);
      }

      /// Expression print

      /// \param os The output stream
      /// \param target_vars The target variable list for this expression
      void print(ExprOStream os, const VariableList& target_vars) const {
        if(target_vars != Base_::vars_) {
          const Permutation perm = target_vars.permutation(Base_::vars());
          os << "[P" << perm << "] " << derived().print_tag() << " " << Base_::vars_ << "\n";
        } else {
          os << derived().print_tag() << Base_::vars_ << "\n";
        }

        left_.print(os, Base_::vars_);
        right_.print(os, Base_::vars_);
      }
    }; // class Binary

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_BINARY_BASE_H__INCLUDED
