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

#ifndef TILEDARRAY_EXPRESSIONS_TSR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_TSR_H__INCLUDED

#include <TiledArray/expressions/base.h>
#include <TiledArray/tile_op/add.h>
#include <TiledArray/tile_op/subt.h>
#include <TiledArray/tile_op/mult.h>
#include <TiledArray/tile_op/no_op.h>
#include <TiledArray/tile_op/scal.h>
#include <TiledArray/tile_op/neg.h>
#include <TiledArray/dist_eval/arary_eval.h>

namespace TiledArray {

  // Forward declaration
  template <typename, unsigned int, typename, typename> class Array;

  namespace expressions {

    template <typename Derived>
    class TsrBase : public Base<Derived>{
    protected:
      typedef Base<Derived> Base_;

    public:
      typedef typename Derived::dist_eval_type dist_eval_type;

      static const bool consumable = false;
      static const unsigned int leaves = 1;

    private:

      // Not allowed
      TsrBase<Derived>& operator=(const TsrBase<Derived>&);

    public:

      /// Constructor

      /// \param vars Array variable list
      TsrBase(const VariableList& vars) : Base_(vars) { }

      /// Copy constructor

      /// \param other The Tensor to be copied
      TsrBase(const TsrBase<Derived>& other) : Base_(other) { }

      // Pull base class functions into this class.
      using Base_::derived;
      using Base_::vars;

      /// Set the variable list for this expression

      /// This function is a noop since the variable list is fixed.
      /// \param target_vars The target variable list for this expression
      void vars(const VariableList&) { }

      /// Initialize the variable list of this expression

      /// This function only checks for valid variable lists.
      /// \param target_vars The target variable list for this expression
      void init_vars(const VariableList& target_vars) {
#ifndef NDEBUG
        if(! target_vars.is_permutation(Base_::vars_)) {
          if(madness::World::get_default().rank() == 0) {
            TA_USER_ERROR_MESSAGE( \
                "The array variable list is not compatible with the expected output:" \
                << "\n    expected = " << target_vars << \
                << "\n    array    = " << Base_::vars_ );
          }

          TA_EXCEPTION("Target variable is not a permutation of the given array variable list.");
        }
#endif // NDEBUG
      }


      /// Initialize the variable list of this expression

      /// This function is a noop since the variable list is fixed.
      void init_vars() { }

      /// Construct the distributed evaluator for array
      dist_eval_type make_dist_eval(madness::World& world, VariableList& target_vars,
          const std::shared_ptr<typename dist_eval_type::pmap_interface>& pmap) const
      {
        // Define the distributed evaluator implementation type
        typedef ArrayEvalImpl<array_type, Derived::op_type, policy> impl_type;

        // Verify input
        TA_ASSERT(target_vars.dim() == derived().array().range().dim());

        if(pmap) {
          pmap = derived().array().get_pmap();
        } else {
          TA_ASSERT(pmap->size() == derived().array().size());
          TA_ASSERT(pmap->procs() == world.size());
        }

        /// Create the pimpl for the distributed evaluator
        std::shared_ptr<typename dist_eval_type::impl_type> pimpl;
        if(Base_::vars_ == target_vars) {
          pimpl.reset(new impl_type(derived().array(), world,
              derived().array().trange(), derived().make_shape(), pmap,
              Permutation(), derived().make_tile_op()));
        } else {
          // Create the permutation that will be applied to this tensor
          const Permutation perm = target_vars.permutation(Base_::vars_);

          pimpl.reset(new impl_type(derived().array(), world,
              perm ^ derived().array().trange(), derived().make_shape().perm(perm),
              pmap, perm, (Base_::permute_tiles_ ? derived().make_tile_op(perm) :
              derived().make_tile_op())));
        }

        return dist_eval_type(pimpl);
      }

      /// Expression print

      /// \param os The output stream
      /// \param target_vars The target variable list for this expression
      void print(ExprOStream os, const VariableList& target_vars) const {
        Base_::print(os, target_vars);
      }

    }; // class TsrBase

    template <typename> class Tsr;

    /// Expression wrapper for array objects

    /// \tparam T The array element type
    /// \tparam DIM The array dimension
    /// \tparam Tile The array tile type
    /// \tparam Policy The array policy type
    template <typename T, unsigned int DIM, typename Tile, typename Policy>
    class Tsr<Array<T, DIM, Tile, Policy> > :
        public TsrBase<Tsr<Array<T, DIM, Tile, Policy> > >
    {
    protected:
      typedef Tsr<Array<T, DIM, Tile, Policy> > Tsr_;
      typedef TsrBase<Tsr_> TsrBase_;
      typedef typename TsrBase_::Base_ Base_;

    public:
      typedef Array<T, DIM, Tile, Policy> array_type; ///< The array type
      typedef TiledArray::math::Noop<typename array_type::eval_type,
          typename array_type::eval_type, false> op_type; ///< The tile operation
      typedef TiledArray::detail::LazyArrayTile<typename array_type::value_type,
          op_type> value_type;  ///< Tile type
      typedef Policy policy; ///< Policy type
      typedef DistEval<value_type, policy> dist_eval_type; ///< Distributed evaluator


    private:

      array_type& array_; ///< The array that this expression

    public:

      /// Constructor

      /// \param array The array object
      /// \param vars The variable list that is associated with this expression
      Tsr(array_type& array, const VariableList& vars) :
        TsrBase(vars), array_(array)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      Tsr(const Tsr_& other) :
        TsrBase(other), array_(other.array_)
      { }

      /// Array accessor

      /// \return A reference to the array
      array_type& array() { return array_; }

      /// Array accessor

      /// \return a const reference to this array
      const array_type& array() const { return array_; }

      /// Array type conversion operator

      /// \return A reference to the array
      operator array_type& () { return array(); }

      /// Const array type conversion operator

      /// \return A const reference to the array
      operator const array_type& () const { return array(); }

      /// Expression assignment operator

      /// \param other The expression that will be assigned to this array
      Tsr_& operator=(Tsr_& other) {
        other.eval_to(*this);
        return *this;
      }

      /// Expression assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be assigned to this array
      template <typename D>
      Tsr_& operator=(const Base<D>& other) {
        other.derived().eval_to(*this);
        return *this;
      }

      /// Expression plus-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be added to this array
      template <typename D>
      Tsr_& operator+=(const Base<D>& other) {
        return operator=(Add<Tsr<const array_type>, D, op_type>(*this, other.derived(), op_type()));
      }

      /// Expression minus-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be subtracted from this array
      template <typename D>
      Tsr_& operator-=(const Base<D>& other) {
        return operator=(Subt<Tsr<const array_type>, D, op_type>(*this, other.derived(), op_type()));
      }

      /// Expression multiply-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will scale this array
      template <typename D>
      Tsr_& operator*=(const Base<D>& other) {
        return operator=(Mult<Tsr<const array_type>, D, op_type>(*this, other.derived(), op_type()));
      }

      /// Non-permuting shape factory function

      /// \return The result shape
      typename dist_eval_type::shape_type
      make_shape() { return array_.get_shape(); }

      /// Permuting shape factory function

      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      typename dist_eval_type::shape_type
      make_shape(const Permutation& perm) { return array_.get_shape().perm(perm); }

      /// Non-permuting tile operation factory function

      /// \return The tile operation
      op_type make_tile_op() const { return op_type(); }

      /// Permuting tile operation factory function

      /// \param perm The permutation to be applied to tiles
      /// \return The tile operation
      op_type make_tile_op(const Permutation& perm) const { return op_type(perm); }

      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      const char* print_tag() const { return ""; }

    }; // class Tsr


    /// Expression wrapper for const array objects

    /// Expression wrapper for array objects

    /// \tparam T The array element type
    /// \tparam DIM The array dimension
    /// \tparam Tile The array tile type
    /// \tparam Policy The array policy type
    template <typename T, unsigned int DIM, typename Tile, typename Policy>
    class Tsr<const Array<T, DIM, Tile, Policy> > :
        public TsrBase<Tsr<const Array<T, DIM, Tile, Policy> > >
    {
    protected:
      typedef Tsr<const Array<T, DIM, Tile, Policy> > Tsr_;
      typedef TsrBase<Tsr_> TsrBase_;
      typedef typename TsrBase_::Base_ Base_;

    public:
      typedef Array<T, DIM, Tile, Policy> array_type; ///< The array type
      typedef TiledArray::math::Noop<typename array_type::eval_type,
          typename array_type::eval_type, false> op_type; ///< The tile operation
      typedef TiledArray::detail::LazyArrayTile<typename array_type::value_type,
          op_type> value_type;  ///< Tile type
      typedef Policy policy; ///< Policy type
      typedef DistEval<value_type, policy> dist_eval_type; ///< Distributed evaluator

    private:

      array_type array_; ///< The array that this expression

      // Not allowed
      Tsr_& operator=(Tsr_&);

    public:

      /// Constructor

      /// \param array The array object
      /// \param vars The variable list that is associated with this expression
      Tsr(const array_type& array, const VariableList& vars) :
        TsrBase_(vars), array_(array)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      Tsr(const Tsr_& other) : TsrBase_(other), array_(other.array_) { }


      /// Copy conversion

      /// \param other The expression to be copied
      Tsr(const Tsr<array_type>& other) : TsrBase_(other.vars()), array_(other.array_) { }

      /// Array accessor

      /// \return a const reference to this array
      const array_type& array() const { return array_; }

      /// Non-permuting shape factory function

      /// \return The result shape
      typename dist_eval_type::shape_type make_shape() { return array_.get_shape(); }

      /// Permuting shape factory function

      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      typename dist_eval::shape_type
      make_shape(const Permutation& perm) { return array_.get_shape().perm(perm); }

      /// Non-permuting tile operation factory function

      /// \return The tile operation
      op_type make_tile_op() const { return op_type(); }

      /// Permuting tile operation factory function

      /// \param perm The permutation to be applied to tiles
      /// \return The tile operation
      op_type make_tile_op(const Permutation& perm) const { return op_type(perm); }


      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      const char* print_tag() const { return ""; }

    }; // class Tsr<const A>

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_TSR_H__INCLUDED
