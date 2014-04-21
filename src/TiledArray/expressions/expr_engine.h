/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  expr_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPR_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPR_ENGINE_H__INCLUDED

#include <TiledArray/madness.h>
#include <TiledArray/expressions/variable_list.h>
#include <TiledArray/expressions/expr_trace.h>

namespace TiledArray {
  namespace expressions {

    // Forward declarations
    template <typename> class Expr;
    template <typename> class ExprEngine;
    template <typename> struct EngineTrait;

    template <typename T>
    struct DerivedType {
      template <template <typename> class Base>
      static T* cast(Base<T>* expr) { return static_cast<T*>(expr); }

      template <template <typename> class Base>
      static const T* cast(const Base<T>* expr) { return static_cast<const T*>(expr); }
    };

    /// Expression engine
    template <typename Derived>
    class ExprEngine : private NO_DEFAULTS {
    public:
      typedef ExprEngine<Derived> ExprEngine_;
      typedef Derived derived_type; ///< The derived object type

      // Operational typedefs
      typedef typename EngineTrait<Derived>::value_type value_type; ///< Tensor value type
      typedef typename EngineTrait<Derived>::op_type op_type; ///< Tile operation type
      typedef typename EngineTrait<Derived>::policy policy; ///< The result policy type
      typedef typename EngineTrait<Derived>::dist_eval_type dist_eval_type; ///< This expression's distributed evaluator type

      // Meta data typedefs
      typedef typename EngineTrait<Derived>::size_type size_type; ///< Size type
      typedef typename EngineTrait<Derived>::trange_type trange_type; ///< Tiled range type type
      typedef typename EngineTrait<Derived>::shape_type shape_type; ///< Tensor shape type
      typedef typename EngineTrait<Derived>::pmap_interface pmap_interface; ///< Process map interface type

    protected:
      // The member variables of this class are protected because derived
      // classes will customize initialization.

      madness::World* world_; ///< The world where this expression will be evaluated
      VariableList vars_; ///< The variable list of this expression
      bool permute_tiles_; ///< Result tile permutation flag (\c true == permute tile)
      Permutation perm_; ///< The permutation that will be applied to the result
      trange_type trange_; ///< The tiled range of the result tensor
      shape_type shape_; ///< The shape of the result tensor
      std::shared_ptr<pmap_interface> pmap_; ///< The process map for the result tensor

    public:

      /// Default constructor

      /// All data members are initialized to NULL values.
      ExprEngine() :
        world_(NULL), vars_(), permute_tiles_(true), perm_(), trange_(), shape_(),
        pmap_()
      { }

      /// Construct and initialize the expression engine

      /// This function will initialize all expression engines in the expression
      /// graph. The <tt>init_vars()</tt>, <tt>init_struct()</tt>, and
      /// <tt>init_distributeion()</tt> will be called for each node and leaf of
      /// the graph in that order.
      /// \param world The world where the expression will be evaluated
      /// \param pmap The process map for the result tensor (may be NULL)
      /// \param target_vars The target variable list of the result tensor
      void init(madness::World& world, const std::shared_ptr<pmap_interface>& pmap,
          const VariableList& target_vars)
      {
        if(target_vars.dim()) {
          derived().init_vars(target_vars);
          derived().init_struct(target_vars);
        } else {
          derived().init_vars();
          derived().init_struct(vars_);
        }

        derived().init_distribution(& world, pmap);
      }

      /// Initialize result tensor structure

      /// This function will initialize the permutation, tiled range, and shape
      /// for the result tensor. These members are initialized with the
      /// <tt>make_perm()</tt>, \c make_trange(), and make_shape() functions.
      /// Derived classes may customize the structure initialization by
      /// providing their own implementation of this function or any of the
      /// above initialization.
      /// functions.
      /// \param target_vars The target variable list for the result tensor
      void init_struct(const VariableList& target_vars) {
        if(target_vars != vars_) {
          perm_ = derived().make_perm(target_vars);
          trange_ = derived().make_trange(perm_);
          shape_ = derived().make_shape(perm_);
        } else {
          trange_ = derived().make_trange();
          shape_ = derived().make_shape();
        }
      }

      /// Initialize result tensor distribution

      /// This function will initialize the world and process map for the result
      /// tensor. Derived classes may customize this function by providing their
      /// own implementation it.
      /// \param world The world were the result will be distributed
      /// \param pmap The process map for the result tensor tiles
      void init_distribution(madness::World* world,
          const std::shared_ptr<pmap_interface>& pmap)
      {
        TA_ASSERT(world);
        TA_ASSERT(pmap);
        TA_ASSERT(pmap->procs() == world->size());
        TA_ASSERT(pmap->size() == trange_.tiles().volume());

        world_ = world;
        pmap_ = pmap;
      }

      /// Permutation factory function

      /// This function will generate the permutation that will be applied to
      /// the result tensor. Derived classes may customize this function by
      /// providing their own implementation it.
      Permutation make_perm(const VariableList& target_vars) const {
        return target_vars.permutation(vars_);
      }

      /// Tile operation factory function

      /// This function will generate the tile operations by calling
      /// \c make_tile_op(). The permuting or non-permuting version of the tile
      /// operation will be selected based on permute_tiles(). Derived classes
      /// may customize this function by providing their own implementation it.
      op_type make_op() const {
        if(perm_ && permute_tiles_)
          return derived().make_tile_op(perm_);
        else
          return derived().make_tile_op();
      }

      /// Cast this object to it's derived type
      derived_type& derived() { return *DerivedType<derived_type>::cast(this); }

      /// Cast this object to it's derived type
      const derived_type& derived() const { return *DerivedType<derived_type>::cast(this); }

      /// World accessor

      /// \return A pointer to world
      madness::World* world() const { return world_; }

      /// Variable list accessor

      /// \return A const reference to the variable list
      const VariableList& vars() const { return vars_; }

      /// Permutation accessor

      /// \return A const reference to the permutation
      const Permutation& perm() const { return perm_; }

      /// Tiled range accessor

      /// \return A const reference to the tiled range
      const trange_type& trange() const { return trange_; }

      /// Shape accessor

      /// \return A const reference to the tiled range
      const shape_type& shape() const { return shape_; }

      /// Process map accessor

      /// \return A const reference to the process map
      const std::shared_ptr<pmap_interface>& pmap() const { return pmap_; }

      /// Permute tile flag accessor

      /// \return The permute tile flag (true == permtue result tiles)
      bool permute_tiles() const { return permute_tiles_; }

      /// Set the permute tiles flag

      /// \param status The new status for permute tiles (true == permtue result tiles)
      void permute_tiles(const bool status) { permute_tiles_ = status; }

      /// Expression print

      /// \param os The output stream
      /// \param target_vars The target variable list for this expression
      void print(ExprOStream& os, const VariableList& target_vars) const {
        if(perm_ && permute_tiles_) {
          os << "[P " << target_vars << "] " << derived().make_tag() << " " << vars_ << "\n";
        } else {
          os << derived().make_tag() << " " << vars_ << "\n";
        }
      }

      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      const char* make_tag() const { return ""; }

    }; // class ExprEngine

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPR_ENGINE_H__INCLUDED
