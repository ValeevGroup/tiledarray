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
    template <typename Op>
    class LazyArrayTile {
    public:
      typedef LazyArrayTile<Op> LazyArrayTile_; ///< This class type
      typedef Op op_type; ///< The operation that will modify this tile
      typedef typename op_type::result_type eval_type; ///< The evaluation type for this tile
      typedef typename detail::remove_cvr<typename op_type::argument_type>::type
          tile_type; ///< The input tile type
      typedef typename tile_type::value_type value_type; ///< Tile element type
      typedef typename scalar_type<value_type>::type numeric_type;

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

      /// Convert tile to evaluation type
      operator eval_type() const {
        TA_ASSERT(tile_.probe());
        return (*op_)(tile_);
      }

      /// Serialization not implemented
      template <typename Archive>
      void serialize(const Archive& ar) {
        TA_ASSERT(false);
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
    template <typename Array, typename Op, typename Policy>
    class ArrayEvalImpl :
//        public madness::WorldObject<ArrayEvalImpl<Array, Op, Policy> >,
        public DistEvalImpl<LazyArrayTile<Op>, Policy>
    {
    public:
      typedef ArrayEvalImpl<Array, Op, Policy> ArrayEvalImpl_; ///< This object type
      typedef DistEvalImpl<LazyArrayTile<Op>, Policy> DistEvalImpl_; ///< The base class type
      typedef typename DistEvalImpl_::TensorImpl_ TensorImpl_; ///< The base, base class type
      typedef Array array_type; ///< The array type
      typedef typename DistEvalImpl_::size_type size_type; ///< Size type
      typedef typename DistEvalImpl_::range_type range_type; ///< Range type
      typedef typename DistEvalImpl_::shape_type shape_type; ///< Shape type
      typedef typename DistEvalImpl_::pmap_interface pmap_interface; ///< Process map interface type
      typedef typename DistEvalImpl_::trange_type trange_type; ///< tiled range type
      typedef typename DistEvalImpl_::value_type value_type; ///< value type
      typedef Op op_type; ///< Tile evaluation operator type

    private:
      array_type array_; ///< The array that will be evaluated
      std::shared_ptr<op_type> op_; ///< The tile operation
      Permutation inv_perm_;

    public:

      /// Constructor

      /// \param arg The argument
      /// \param op The element transform operation
      ArrayEvalImpl(const array_type& array, const Permutation& perm, const shape_type& shape,
          const std::shared_ptr<pmap_interface>& pmap, const op_type& op) :
        DistEvalImpl_(array.get_world(), perm, array.trange(), shape, pmap),
        array_(array),
        op_(new op_type(op)),
        inv_perm_(-perm)
      { }

      /// Virtual destructor
      virtual ~ArrayEvalImpl() { }

    private:

      void eval_tile(const size_type i, const madness::Future<typename array_type::value_type>& tile) {
        TA_ASSERT(TensorImpl_::is_local(i));
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

        // Create iterator to tiles that are local for this evaluator.
        const typename array_type::pmap_interface::const_iterator end =
            TensorImpl_::pmap()->end();
        typename array_type::pmap_interface::const_iterator it =
            TensorImpl_::pmap()->begin();

        for(; it != end; ++it) {
          if(! TensorImpl_::is_zero(*it)) {

            // Get the tile from array_, which may be located on a remote node.
            madness::Future<typename array_type::value_type> tile =
                array_.find(inv_perm_ ^ TensorImpl_::range().idx(*it));

            // Insert the tile into this evaluator for subsequent processing
            const size_type i = DistEvalImpl_::perm_index(*it);
            if(tile.probe()) {
              // Skip the task since the tile is ready
              DistEvalImpl_::set_tile(i, value_type(tile, op_));
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
