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
      LazyArrayTile(const tile_type& tile, std::shared_ptr<op_type> op, const bool consume) :
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
      bool is_consumable() const { return consume_; }

    public:

      /// Convert tile to evaluation type
      operator eval_type() const { return (*op_)(tile_, consume_); }

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
        public DistEvalImpl<LazyArrayTile<typename Array::value_type, Op>, Policy>
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

    private:
      array_type array_; ///< The array that will be evaluated
      std::shared_ptr<op_type> op_; ///< The tile operation
      Permutation inv_perm_; ///< The inverse permutation

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
        DistEvalImpl_(world, array.trange(), shape, pmap, perm),
        array_(array),
        op_(new op_type(op)),
        inv_perm_(-perm)
      { }

      /// Virtual destructor
      virtual ~ArrayEvalImpl() { }

    private:

      /// Make an array tile and insert it into the distributed storage container

      /// \param i The tile index
      /// \param tile The array tile that is the basis for lazy tile
      void set_tile(const size_type i, const typename array_type::value_type& tile, const bool consume) {
        DistEvalImpl_::set_tile(i, value_type(tile, op_, consume));
      }

      /// Get the array tile that corresponds to the target tile

      /// This function applies the inverse permutation to the target index to
      /// get the array tile index.
      /// \param inv_perm The inverse permutation
      /// \param i The target tile index
      size_type get_array_index(const Permutation& inv_perm, const size_type i) {
        return array_.range().ord(inv_perm ^ TensorImpl_::range().idx(i));
      }

      /// Get the array tile that corresponds to the target tile

      /// No permutation is applied to the target index.
      /// \param i The target tile index
      size_type get_array_index(const NoPermutation, const size_type i) {
        return i;
      }

      /// Evaluate tiles for this operation

      /// This function will construct the local tiles of this object, which may
      /// be different from that of \c array_ . Array tiles are consumed using a
      /// "pull" algorithm.
      /// \tparam Perm The permutation type (Permutation or NoPermutation)
      /// \param self A shared pointer to this object
      /// \param inv_perm The inverse permutation applied to the target tile index
      template <typename Perm>
      size_type eval_kernel(const std::shared_ptr<ArrayEvalImpl_>& self, Perm inv_perm) {
        // Counter for the number of tasks submitted by this object
        size_type task_count = 0ul;

        // Create iterator to tiles that are local for this evaluator.
        const typename array_type::pmap_interface::const_iterator end =
            TensorImpl_::pmap()->end();
        typename array_type::pmap_interface::const_iterator it =
            TensorImpl_::pmap()->begin();

        for(; it != end; ++it) {
          const size_type i = *it; // The target index
          if(! TensorImpl_::is_zero(i)) {

            // Get the array index that corresponds to the target index
            const size_type array_index = get_array_index(inv_perm, i);

            // Get the tile from array_, which may be located on a remote node.
            const bool consumable_tile = ! array_.is_local(array_index);
            madness::Future<typename array_type::value_type> tile =
                array_.find(array_index);

            // Insert the tile into this evaluator for subsequent processing
            if(tile.probe()) {
              // Skip the task since the tile is ready
              DistEvalImpl_::set_tile(i, value_type(tile, op_, consumable_tile));
            } else {
              // Spawn a task to set the tile the input tile is ready.
              TensorImpl_::get_world().taskq.add(self, & ArrayEvalImpl_::set_tile,
                  i, tile, consumable_tile, madness::TaskAttributes::hipri());
            }

            ++task_count;
          }
        }

        return task_count;
      }

      /// Function for evaluating this tensor's tiles

      /// This function is run inside a task, and will run after \c eval_children
      /// has completed. It should spawn additional tasks that evaluate the
      /// individual result tiles.
      /// \return The number of local tiles
      virtual size_type internal_eval(const std::shared_ptr<DistEvalImpl_>& pimpl) {
        // Convert pimpl to this object type so it can be used in tasks
        TA_ASSERT(this == pimpl.get());
        std::shared_ptr<ArrayEvalImpl_> self =
            std::static_pointer_cast<ArrayEvalImpl_>(pimpl);

        return (DistEvalImpl_::perm().dim() > 1 ?
            eval_kernel(self, -DistEvalImpl_::perm()) :
            eval_kernel(self, NoPermutation()));
      }

    }; // class ArrayEvalImpl

    /// Distrubuted array evaluator factory function

    /// Construct a distributed array evaluator, which wraps an \c Array object
    /// in a distributed evaluator so that it can be used by other distributed
    /// evaluators.
    /// \tparam T The element type of \c array
    /// \tparam DIM The number of dimensions of \c array
    /// \tparam Tile Tile type of \c array
    /// \tparam Policy The policy type of \c array
    /// \tparam Op The unary tile operation type
    /// \param arg Argument to be modified
    /// \param world The world where \c array will be evaluated
    /// \param shape The shape of the evaluated tensor
    /// \param pmap The process map for the evaluated tensor
    /// \param perm The permutation applied to the tensor
    /// \param op The unary tile operation
    template <typename T, unsigned int DIM, typename Tile, typename Policy, typename Op>
    DistEval<LazyArrayTile<typename Array<T, DIM, Tile, Policy>::value_type, Op>, Policy>
    make_array_eval(
        const Array<T, DIM, Tile, Policy>& array,
        madness::World& world,
        const typename DistEval<Tile, Policy>::shape_type& shape,
        const std::shared_ptr<typename DistEval<Tile, Policy>::pmap_interface>& pmap,
        const Permutation& perm,
        const Op& op)
    {
      typedef ArrayEvalImpl<Array<T, DIM, Tile, Policy>, Op, Policy> impl_type;
      typedef typename impl_type::DistEvalImpl_ impl_base_type;
      return DistEval<LazyArrayTile<typename Array<T, DIM, Tile, Policy>::value_type, Op>, Policy>(
          std::shared_ptr<impl_base_type>(new impl_type(array, world,
              (perm.dim() > 1u ? perm ^ array.trange() : array.trange()), shape,
              pmap, perm, op)));
    }

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_EVAL_H__INCLUDED
