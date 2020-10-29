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

#ifndef TILEDARRAY_EXPRESSIONS_EXPR_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_EXPR_ENGINE_H__INCLUDED

#include <TiledArray/expressions/expr_trace.h>
#include <TiledArray/external/madness.h>

namespace TiledArray {
namespace expressions {

// Forward declarations
template <typename>
struct EngineParamOverride;
template <typename>
class Expr;
template <typename>
struct EngineTrait;

/// Expression engine
template <typename Derived>
class ExprEngine : private NO_DEFAULTS {
 public:
  typedef ExprEngine<Derived> ExprEngine_;
  typedef Derived derived_type;  ///< The derived object type

  // Operational typedefs
  typedef typename EngineTrait<Derived>::value_type
      value_type;  ///< Tensor value type
  typedef
      typename EngineTrait<Derived>::op_type op_type;  ///< Tile operation type
  typedef
      typename EngineTrait<Derived>::policy policy;  ///< The result policy type
  typedef typename EngineTrait<Derived>::dist_eval_type
      dist_eval_type;  ///< This expression's distributed evaluator type

  // Meta data typedefs
  typedef typename EngineTrait<Derived>::size_type size_type;  ///< Size type
  typedef typename EngineTrait<Derived>::trange_type
      trange_type;  ///< Tiled range type type
  typedef typename EngineTrait<Derived>::shape_type
      shape_type;  ///< Tensor shape type
  typedef typename EngineTrait<Derived>::pmap_interface
      pmap_interface;  ///< Process map interface type

 protected:
  // The member variables of this class are protected because derived
  // classes will customize initialization.

  World* world_;  ///< The world where this expression will be evaluated
  BipartiteIndexList
      indices_;  ///< The index list of this expression; bipartite due to need
                 ///< to support recursive tensors (i.e. Tensor-of-Tensor)
  bool permute_tiles_;  ///< Result tile permutation flag (\c true == permute
                        ///< tile)
  /// The permutation that will be applied to the outer tensor of tensors
  BipartitePermutation perm_;
  trange_type trange_;  ///< The tiled range of the result tensor
  shape_type shape_;    ///< The shape of the result tensor
  std::shared_ptr<pmap_interface>
      pmap_;  ///< The process map for the result tensor
  std::shared_ptr<EngineParamOverride<Derived> >
      override_ptr_;  ///< The engine params overriding the default

 public:
  /// Default constructor

  /// All data members are initialized to NULL values.
  template <typename D>
  ExprEngine(const Expr<D>& expr)
      : world_(NULL),
        indices_(),
        permute_tiles_(true),
        perm_(),
        trange_(),
        shape_(),
        pmap_(),
        override_ptr_(expr.override_ptr_) {}

  /// Construct and initialize the expression engine

  /// This function will initialize all expression engines in the expression
  /// graph. The <tt>init_indices()</tt>, <tt>init_struct()</tt>, and
  /// <tt>init_distribution()</tt> will be called for each node and leaf of
  /// the graph in that order.
  /// \param world The world where the expression will be evaluated
  /// \param pmap The process map for the result tensor (may be NULL)
  /// \param target_indices The target index list of the result tensor
  void init(World& world, std::shared_ptr<pmap_interface> pmap,
            const BipartiteIndexList& target_indices) {
    if (target_indices.size()) {
      derived().init_indices(target_indices);
      derived().init_struct(target_indices);
    } else {
      derived().init_indices();
      derived().init_struct(indices_);
    }

    auto override_world = override_ptr_ != nullptr && override_ptr_->world;
    auto override_pmap = override_ptr_ != nullptr && override_ptr_->pmap;
    world_ = override_world ? override_ptr_->world : &world;
    pmap_ = override_pmap ? override_ptr_->pmap : pmap;

    // Check for a valid process map.
    if (pmap_) {
      // If process map is not valid, use the process map constructed by the
      // expression engine.
      if ((typename pmap_interface::size_type(world_->size()) !=
           pmap_->procs()) ||
          (trange_.tiles_range().volume() != pmap_->size()))
        pmap_.reset();
    }

    derived().init_distribution(world_, pmap_);
  }

  /// Initialize result tensor structure

  /// This function will initialize the permutation, tiled range, and shape
  /// for the result tensor. These members are initialized with the
  /// <tt>make_perm()</tt>, \c make_trange(), and make_shape() functions.
  /// Derived classes may customize the structure initialization by
  /// providing their own implementation of this function or any of the
  /// above initialization.
  /// functions.
  /// \param target_indices The target index list for the result tensor
  void init_struct(const BipartiteIndexList& target_indices) {
    if (target_indices != indices_) {
      perm_ = derived().make_perm(target_indices);
      trange_ = derived().make_trange(outer(perm_));
      shape_ = derived().make_shape(outer(perm_));
    } else {
      trange_ = derived().make_trange();
      shape_ = derived().make_shape();
    }

    if (override_ptr_ && override_ptr_->shape)
      shape_ = shape_.mask(*override_ptr_->shape);
  }

  /// Initialize result tensor distribution

  /// This function will initialize the world and process map for the result
  /// tensor. Derived classes may customize this function by providing their
  /// own implementation it.
  /// \param world The world were the result will be distributed
  /// \param pmap The process map for the result tensor tiles
  void init_distribution(World* world,
                         const std::shared_ptr<pmap_interface>& pmap) {
    TA_ASSERT(world);
    TA_ASSERT(pmap);
    TA_ASSERT(pmap->procs() ==
              typename pmap_interface::size_type(world->size()));
    TA_ASSERT(pmap->size() == trange_.tiles_range().volume());

    world_ = world;
    pmap_ = pmap;
  }

  /// Permutation factory function

  /// This function will generate the permutation that will be applied to
  /// the result tensor. Derived classes may customize this function by
  /// providing their own implementation it.
  BipartitePermutation make_perm(
      const BipartiteIndexList& target_indices) const {
    return target_indices.permutation(indices_);
  }

  /// Tile operation factory function

  /// This function will generate the tile operations by calling
  /// \c make_tile_op(). The permuting or non-permuting version of the tile
  /// operation will be selected based on permute_tiles(). Derived classes
  /// may customize this function by providing their own implementation it.
  op_type make_op() const {
    if (perm_ && permute_tiles_)
      // permutation can only be applied to the tile, not to its element (if
      // tile = tensor-of-tensors)
      return derived().make_tile_op(perm_);
    else
      return derived().make_tile_op();
  }

  /// Cast this object to its derived type
  derived_type& derived() { return *static_cast<derived_type*>(this); }

  /// Cast this object to its derived type
  const derived_type& derived() const {
    return *static_cast<const derived_type*>(this);
  }

  /// World accessor

  /// \return A pointer to world
  World* world() const { return world_; }

  /// Index list accessor

  /// \return A const reference to the index list
  const BipartiteIndexList& indices() const { return indices_; }

  /// Permutation accessor

  /// \return A const reference to the permutation
  const BipartitePermutation& perm() const { return perm_; }

  /// Tiled range accessor

  /// \return A const reference to the tiled range
  const trange_type& trange() const { return trange_; }

  /// Shape accessor

  /// \return A const reference to the tiled range
  const shape_type& shape() const { return shape_; }

  /// Process map accessor

  /// \return A const reference to the process map
  const std::shared_ptr<pmap_interface>& pmap() const { return pmap_; }

  /// Set the permute tiles flag

  /// \param status The new status for permute tiles (true == permute result
  /// tiles)
  void permute_tiles(const bool status) { permute_tiles_ = status; }

  /// Expression print

  /// \param os The output stream
  /// \param target_indices The target index list for this expression
  void print(ExprOStream& os, const BipartiteIndexList& target_indices) const {
    if (perm_) {
      os << "[P " << target_indices << "]"
         << (permute_tiles_ ? " " : " [no permute tiles] ")
         << derived().make_tag() << indices_ << "\n";
    } else {
      os << derived().make_tag() << indices_ << "\n";
    }
  }

  /// Expression identification tag

  /// \return An expression tag used to identify this expression
  const char* make_tag() const { return ""; }

};  // class ExprEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_EXPR_ENGINE_H__INCLUDED
