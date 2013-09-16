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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  array_eval.h
 *  Aug 9, 2013
 *
 */

#ifndef TILEDARRAY_ARRAY_EVAL_H__INCLUDED
#define TILEDARRAY_ARRAY_EVAL_H__INCLUDED

#include <TiledArray/dist_eval/dist_eval.h>

namespace TiledArray {
  namespace detail {



    /// Lazy tile for evaluating array tiles on the fly.

    /// \tparam Result The result tile type for the lazy tile
    /// \tparam
    template <typename Tile, typename Op>
    class LazyArrayTile {
    public:
      typedef LazyArrayTile<Tile, Op> LazyArrayTile_; ///< This class type
      typedef typename Tile::eval_type eval_type; ///< The evaluation type for this tile
      typedef Tile tile_type; ///< The input tile type
      typedef Op op_type; ///< The operation that will modify this tile

    private:
      madness::Future<tile_type> tile_; ///< The input tile
      std::shared_ptr<op_type> op_; ///< The operation that will be applied to argument tiles

    public:
      /// Default constructor
      LazyArrayTile() :
        tile_(madness::Future<tile_type>::default_initializer()), op_()
      { }

      LazyArrayTile(const LazyArrayTile_& other) : tile_(other.tile_), op_(other.op_) { }

      LazyArrayTile(const madness::Future<tile_type>& tile, std::shared_ptr<op_type> op) :
        tile_(tile), op_(op)
      { }

      LazyArrayTile_& operator=(const LazyArrayTile_& other) {
        tile_ = other.tile_;
        op_ = other.op_;
        return *this;
      }

      operator eval_type() const {
        TA_ASSERT(tile_.probe());
        return (*op_)(tile_);
      }

    }; // LazyArrayTile

    /// Distributed evaluator for \c TiledArray::Array objects

    /// This distributed evaluator applies modifications to Array that will be
    /// used as input to other distributed evaluators. Common operations that
    /// may be applied to array objects are scaling, permutation, and lazy tile
    /// evaluation. It also serves as an abstraction layer between
    /// \c TiledArray::Array objects and internal evaluation of expressions. The
    /// main purpose of this evaluator is to do a lazy evaluation of input tiles
    /// so that the resulting data is only evaluated when the tile is needed by
    /// subsequent operations.
    /// \tparam Policy The evaluator policy type
    template <typename Policy>
    class ArrayEvalImpl : public DistEvalImpl<Policy> {
    public:
      typedef ArrayEvalImpl<Policy> ArrayEvalImpl_; ///< This object type
      typedef DistEvalImpl<Policy> DistEvalImpl_; ///< The base class type
      typedef typename DistEvalImpl_::TensorImpl_ TensorImpl_; ///< The base, base class type
      typedef typename Policy::arg_type array_type; ///< The array type
      typedef typename DistEvalImpl_::size_type size_type; ///< Size type
      typedef typename DistEvalImpl_::range_type range_type; ///< Range type
      typedef typename DistEvalImpl_::shape_type shape_type; ///< Shape type
      typedef typename DistEvalImpl_::pmap_interface pmap_interface; ///< Process map interface type
      typedef typename DistEvalImpl_::trange_type trange_type; ///< tiled range type
      typedef typename DistEvalImpl_::value_type value_type; ///< value type
      typedef typename DistEvalImpl_::future future; ///< Future type
      typedef typename Policy::op_type op_type; ///< Tile evaluation operator type

    private:
      array_type array_; ///< The array that will be evaluated
      std::shared_ptr<op_type> op_; ///< The tile operation

    public:

      /// Constructor

      /// \param arg The argument
      /// \param op The element transform operation
      ArrayEvalImpl(const array_type& array, const Permutation& perm, const shape_type& shape,
          const std::shared_ptr<pmap_interface>& pmap, const op_type& op) :
        DistEvalImpl_(array.get_world(), perm, array.trange(), shape, pmap, false),
        array_(array),
        op_(op)
      { }

      /// Virtual destructor
      virtual ~ArrayEvalImpl() { }

    private:

      void eval_tile(const size_type i, const madness::Future<typename array_type::value_type>& tile) {
        DistEvalImpl_::set_tile(i, value_type(tile, op_));
      }

      /// Function for evaluating this tensor's tiles

      /// This function is run inside a task, and will run after \c eval_children
      /// has completed. It should spawn additional tasks that evaluate the
      /// individual result tiles.
      /// \return The number of local tiles
      virtual size_type eval_tiles(const std::shared_ptr<DistEvalImpl_>& pimpl) {
        // Counter for the number of tasks submitted by this object
        size_type task_count = 0ul;

        // Convert pimpl to this object type so it can be used in tasks
        TA_ASSERT(this == pimpl.get());
        std::shared_ptr<ArrayEvalImpl_> self =
            std::static_pointer_cast<ArrayEvalImpl_>(pimpl);

        // Make sure all local tiles are present.
        const typename pmap_interface::const_iterator end = TensorImpl_::pmap()->end();
        typename pmap_interface::const_iterator it = TensorImpl_::pmap()->begin();
        for(; it != end; ++it) {
          if(! array_.is_zero(*it)) {
            madness::Future<typename array_type::value_type> tile =
                array_.find(*it);
            if(tile.probe()) {
              // Skip the task since the tile is ready
              DistEvalImpl_::set_tile(*it, value_type(tile, op_));
            } else {
              // Spawn a task to set the tile the input tile is ready.
              TensorImpl_::get_world().taskq.add(self,
                  & ArrayEvalImpl_::eval_tile, *it, tile);
            }

            ++task_count;
          }
        }

        return task_count;
      }

      /// Function for evaluating child tensors
      virtual void eval_children() { }

      /// Wait for tasks of children to finish
      virtual void wait_children() const { }

    }; // class UnaryEvalImpl

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_EVAL_H__INCLUDED
