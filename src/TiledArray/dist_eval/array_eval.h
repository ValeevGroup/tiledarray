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

  // Forward declarations
  template <typename, unsigned int, typename, typename> class Array;

  namespace detail {

    /// Lazy tile for on-the-fly evaluation of array tiles.

    /// This tile object is used to hold input array tiles and do on-the-fly
    /// evaluation, type conversion, and/or permutations.
    /// \tparam Result The result tile type for the lazy tile
    /// \tparam Op The operation type
    template <typename Tile, typename Op>
    class LazyArrayTile {
    public:
      typedef LazyArrayTile<Tile, Op> LazyArrayTile_; ///< This class type
      typedef Op op_type; ///< The operation that will modify this tile
      typedef typename op_type::result_type eval_type; ///< The evaluation type for this tile
      typedef Tile tile_type; ///< The input tile type
      typedef typename tile_type::value_type value_type; ///< Tile element type
      typedef typename scalar_type<value_type>::type numeric_type;

    private:
      mutable tile_type tile_; ///< The input tile
      std::shared_ptr<op_type> op_; ///< The operation that will be applied to argument tiles
      bool consume_; ///< If true, \c tile_ is consumable

    public:
      /// Default constructor
      LazyArrayTile() : tile_(), op_(), consume_(false) { }

      /// Copy constructor

      /// \param other The LazyArrayTile object to be copied
      LazyArrayTile(const LazyArrayTile_& other) :
        tile_(other.tile_), op_(other.op_), consume_(other.consume_)
      { }

      /// Construct from tile and operation

      /// \param tile The input tile that will be modified
      /// \param op The operation to be applied to the input tile
      /// \param consume If true, the input tile may be consumed by \c op
      LazyArrayTile(const tile_type& tile, const std::shared_ptr<op_type>& op, const bool consume) :
        tile_(tile), op_(op), consume_(consume)
      { }

      /// Assignment operator

      /// \param other The object to be copied
      /// \param A reference to this object
      LazyArrayTile_& operator=(const LazyArrayTile_& other) {
        tile_ = other.tile_;
        op_ = other.op_;
        consume_ = other.consume_;

        return *this;
      }

      /// Query runtime consumable status

      /// \return \c true if this tile is consumable, otherwise \c false .
      bool is_consumable() const { return consume_ || op_->permutation(); }

      /// Convert tile to evaluation type
      operator eval_type() const { return (*op_)(tile_, consume_); }

      /// return ref to input tile
      const tile_type& tile() const { return tile_; }

      /// Serialization (not implemented)

      /// \tparam Archive The archive type
      template <typename Archive>
      void serialize(const Archive&) {
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
        public DistEvalImpl<LazyArrayTile<typename Array::value_type, Op>, Policy>,
        public std::enable_shared_from_this<ArrayEvalImpl<Array, Op, Policy> >
    {
    public:
      typedef ArrayEvalImpl<Array, Op, Policy> ArrayEvalImpl_; ///< This object type
      typedef DistEvalImpl<LazyArrayTile<typename Array::value_type, Op>, Policy> DistEvalImpl_; ///< The base class type
      typedef typename DistEvalImpl_::TensorImpl_ TensorImpl_; ///< The base, base class type
      typedef Array array_type; ///< The array type
      typedef typename DistEvalImpl_::size_type size_type; ///< Size type
      typedef typename DistEvalImpl_::range_type range_type; ///< Range type
      typedef typename DistEvalImpl_::shape_type shape_type; ///< Shape type
      typedef typename DistEvalImpl_::pmap_interface pmap_interface; ///< Process map interface type
      typedef typename DistEvalImpl_::trange_type trange_type; ///< tiled range type
      typedef typename DistEvalImpl_::value_type value_type; ///< value type
      typedef Op op_type; ///< Tile evaluation operator type

      using std::enable_shared_from_this<ArrayEvalImpl<Array, Op, Policy> >::shared_from_this;

    private:
      array_type array_; ///< The array that will be evaluated
      std::shared_ptr<op_type> op_; ///< The tile operation

    public:

      /// Constructor

      /// \param array The array that will be evaluated
      /// \param world The world where array will be evaluated
      /// \param trange The tiled range of the result tensor
      /// \param shape The shape of the result tensor
      /// \param pmap The process map for the result tensor tiles
      /// \param op The operation that will be used to evaluate the tiles of array
      ArrayEvalImpl(const array_type& array, madness::World& world, const trange_type& trange,
          const shape_type& shape, const std::shared_ptr<pmap_interface>& pmap,
          const Permutation& perm, const op_type& op) :
        DistEvalImpl_(world, trange, shape, pmap, perm),
        array_(array),
        op_(new op_type(op))
      { }

      /// Virtual destructor
      virtual ~ArrayEvalImpl() { }

      virtual madness::Future<value_type> get_tile(size_type i) const {

        // Get the array index that corresponds to the target index
        const size_type array_index = DistEvalImpl_::perm_index_to_source(i);

        // Get the tile from array_, which may be located on a remote node.
        madness::Future<typename array_type::value_type> tile =
            array_.find(array_index);

        const bool consumable_tile = ! array_.is_local(array_index);
        // Insert the tile into this evaluator for subsequent processing
        if(tile.probe()) {
          // Skip the task since the tile is ready
          madness::Future<value_type> result;
          result.set(make_tile(tile, consumable_tile));
          const_cast<ArrayEvalImpl_*>(this)->notify();
          return result;
        } else {
          // Spawn a task to set the tile the input tile is ready.
          madness::Future<value_type> result =
              TensorImpl_::get_world().taskq.add(shared_from_this(),
              & ArrayEvalImpl_::make_tile, tile, consumable_tile,
              madness::TaskAttributes::hipri());

          result.register_callback(const_cast<ArrayEvalImpl_*>(this));
          return result;
        }
      }

    private:

      value_type make_tile(const typename array_type::value_type& tile, const bool consume) const {
        return value_type(tile, op_, consume);
      }

      /// Make an array tile and insert it into the distributed storage container

      /// \param i The tile index
      /// \param tile The array tile that is the basis for lazy tile
      void set_tile(const size_type i, const typename array_type::value_type& tile, const bool consume) {
        DistEvalImpl_::set_tile(i, value_type(tile, op_, consume));
      }

      /// Evaluate the tiles of this tensor

      /// This function will evaluate the children of this distributed evaluator
      /// and evaluate the tiles for this distributed evaluator.
      /// \param pimpl A shared pointer to this object
      /// \return The number of tiles that will be set by this process
      virtual int internal_eval() {
        // Counter for the number of tasks submitted by this object
        int task_count = 0;

        // Get a count of the number of local tiles.
        if(TensorImpl_::shape().is_dense()) {
          task_count = TensorImpl_::pmap()->local_size();
        } else {
          // Create iterator to tiles that are local for this evaluator.
          typename array_type::pmap_interface::const_iterator it =
              TensorImpl_::pmap()->begin();
          const typename array_type::pmap_interface::const_iterator end =
              TensorImpl_::pmap()->end();

          for(; it != end; ++it) {
            if(! TensorImpl_::is_zero(*it))
              ++task_count;
          }
        }

        return task_count;
      }

    }; // class ArrayEvalImpl

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_EVAL_H__INCLUDED
