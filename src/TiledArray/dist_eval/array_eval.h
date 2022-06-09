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

#ifndef TILEDARRAY_DIST_EVAL_ARRAY_EVAL_H__INCLUDED
#define TILEDARRAY_DIST_EVAL_ARRAY_EVAL_H__INCLUDED

#include <TiledArray/block_range.h>
#include <TiledArray/dist_eval/dist_eval.h>

namespace TiledArray {
namespace detail {

/// Lazy tile for on-the-fly evaluation of array tiles.

/// This tile object is used to hold input array tiles and do on-the-fly
/// evaluation, type conversion, and/or permutations.
/// \tparam Tile Array tile type
/// \tparam Op The operation type
template <typename Tile, typename Op>
class LazyArrayTile {
 public:
  typedef LazyArrayTile<Tile, Op> LazyArrayTile_;  ///< This class type
  typedef Op op_type;  ///< The operation that will modify this tile
  typedef typename op_type::result_type eval_type;
  typedef Tile tile_type;  ///< The input tile type
 private:
  mutable tile_type tile_;  ///< The input tile
  std::shared_ptr<op_type>
      op_;        ///< The operation that will be applied to argument tiles
  bool consume_;  ///< If true, \c tile_ is consumable

  template <typename T>
  using eval_t = typename eval_trait<typename std::decay<T>::type>::type;

 public:
  using conversion_result_type = decltype((
      (!Op::is_consumable) && consume_ ? op_->consume(tile_)
                                       : (*op_)(tile_)));  ///< conversion_type

#ifdef TILEDARRAY_HAS_CUDA
  // TODO need a better design on how to manage the lifetime of converted Tile
  mutable conversion_result_type conversion_tile_;
#endif
 public:
  /// Default constructor
  LazyArrayTile()
      : tile_(),
        op_(),
        consume_(false)
#ifdef TILEDARRAY_HAS_CUDA
        ,
        conversion_tile_()
#endif
  {
  }

  /// Copy constructor

  /// \param other The LazyArrayTile object to be copied
  LazyArrayTile(const LazyArrayTile_& other)
      : tile_(other.tile_),
        op_(other.op_),
        consume_(other.consume_)
#ifdef TILEDARRAY_HAS_CUDA
        ,
        conversion_tile_()
#endif
  {
  }

  /// Construct from tile and operation

  /// \param tile The input tile that will be modified
  /// \param op The operation to be applied to the input tile
  /// \param consume If true, the input tile may be consumed by \c op
  LazyArrayTile(const tile_type& tile, const std::shared_ptr<op_type>& op,
                const bool consume)
      : tile_(tile),
        op_(op),
        consume_(consume)
#ifdef TILEDARRAY_HAS_CUDA
        ,
        conversion_tile_()
#endif
  {
  }

  /// Assignment operator

  /// \param other The object to be copied
  LazyArrayTile_& operator=(const LazyArrayTile_& other) {
    tile_ = other.tile_;
    op_ = other.op_;
    consume_ = other.consume_;
#ifdef TILEDARRAY_HAS_CUDA
    conversion_tile_ = other.conversion_tile_;
#endif
    return *this;
  }

  /// Query runtime consumable status

  /// \return \c true if this tile is consumable, otherwise \c false .
  bool is_consumable() const { return consume_ || op_->permutation(); }

  /// Convert tile to evaluation type using the op object
#ifdef TILEDARRAY_HAS_CUDA

  explicit operator conversion_result_type&() const {
    conversion_tile_ =
        std::move(((!Op::is_consumable) && consume_ ? op_->consume(tile_)
                                                    : (*op_)(tile_)));
    return conversion_tile_;
  }

#else
  explicit operator conversion_result_type() const {
    return ((!Op::is_consumable) && consume_ ? op_->consume(tile_)
                                             : (*op_)(tile_));
  }
#endif

  /// return ref to input tile
  const tile_type& tile() const { return tile_; }

  /// Serialization (not implemented)

  /// \tparam Archive The archive type
  template <typename Archive>
  void serialize(Archive&) {
    TA_ASSERT(false);
  }

};  // LazyArrayTile

/// Distributed evaluator for \c TiledArray::DistArray objects

/// This distributed evaluator applies modifications to Array that will be
/// used as input to other distributed evaluators. Common operations that
/// may be applied to array objects are scaling, permutation, and lazy tile
/// evaluation. It also serves as an abstraction layer between
/// \c TiledArray::DistArray objects and internal evaluation of expressions. The
/// main purpose of this evaluator is to do a lazy evaluation of input tiles
/// so that the resulting data is only evaluated when the tile is needed by
/// subsequent operations.
/// \tparam Policy The evaluator policy type
template <typename Array, typename Op, typename Policy>
class ArrayEvalImpl
    : public DistEvalImpl<LazyArrayTile<typename Array::value_type, Op>,
                          Policy>,
      public std::enable_shared_from_this<ArrayEvalImpl<Array, Op, Policy>> {
 public:
  typedef ArrayEvalImpl<Array, Op, Policy>
      ArrayEvalImpl_;  ///< This object type
  typedef DistEvalImpl<LazyArrayTile<typename Array::value_type, Op>, Policy>
      DistEvalImpl_;  ///< The base class type
  typedef typename DistEvalImpl_::TensorImpl_
      TensorImpl_;           ///< The base, base class type
  typedef Array array_type;  ///< The array type
  typedef typename DistEvalImpl_::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename DistEvalImpl_::range_type range_type;      ///< Range type
  typedef typename DistEvalImpl_::shape_type shape_type;      ///< Shape type
  typedef typename DistEvalImpl_::pmap_interface
      pmap_interface;  ///< Process map interface type
  typedef
      typename DistEvalImpl_::trange_type trange_type;  ///< tiled range type
  typedef typename DistEvalImpl_::value_type
      value_type;      ///< value type = LazyArrayTile
  typedef Op op_type;  ///< Tile evaluation operator type

  using std::enable_shared_from_this<
      ArrayEvalImpl<Array, Op, Policy>>::shared_from_this;

 private:
  array_type array_;             ///< The array that will be evaluated
  std::shared_ptr<op_type> op_;  ///< The tile operation
  BlockRange block_range_;       ///< Sub-block range

 public:
  /// Construct with full array range

  /// \param array The array that will be evaluated
  /// \param world The world where array will be evaluated
  /// \param trange The tiled range of the result tensor
  /// \param shape The shape of the result tensor
  /// \param pmap The process map for the result tensor tiles
  /// \param perm The permutation that is applied to the tile coordinate index
  /// \param op The operation that will be used to evaluate the tiles of array
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  ArrayEvalImpl(const array_type& array, World& world,
                const trange_type& trange, const shape_type& shape,
                const std::shared_ptr<pmap_interface>& pmap, const Perm& perm,
                const op_type& op)
      : DistEvalImpl_(world, trange, shape, pmap, outer(perm)),
        array_(array),
        op_(std::make_shared<op_type>(op)),
        block_range_() {}

  /// Constructor with sub-block range

  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param array The array that will be evaluated
  /// \param world The world where array will be evaluated
  /// \param trange The tiled range of the result tensor
  /// \param shape The shape of the result tensor
  /// \param pmap The process map for the result tensor tiles
  /// \param perm The permutation that is applied to the tile coordinate index
  /// \param op The operation that will be used to evaluate the tiles of array
  /// \param lower_bound The sub-block lower bound
  /// \param upper_bound The sub-block upper bound
  template <typename Index1, typename Index2, typename Perm,
            typename = std::enable_if_t<
                TiledArray::detail::is_integral_range_v<Index1> &&
                TiledArray::detail::is_integral_range_v<Index2> &&
                TiledArray::detail::is_permutation_v<Perm>>>
  ArrayEvalImpl(const array_type& array, World& world,
                const trange_type& trange, const shape_type& shape,
                const std::shared_ptr<pmap_interface>& pmap, const Perm& perm,
                const op_type& op, const Index1& lower_bound,
                const Index2& upper_bound)
      : DistEvalImpl_(world, trange, shape, pmap, outer(perm)),
        array_(array),
        op_(std::make_shared<op_type>(op)),
        block_range_(array.trange().tiles_range(), lower_bound, upper_bound) {}

  /// Virtual destructor
  virtual ~ArrayEvalImpl() {}

  virtual Future<value_type> get_tile(ordinal_type i) const {
    // Get the array index that corresponds to the target index
    auto array_index = DistEvalImpl_::perm_index_to_source(i);

    // If this object only uses a sub-block of the array, shift the tile
    // index to the correct location.
    if (block_range_.rank()) array_index = block_range_.ordinal(array_index);

    // Get the tile from array_, which may be located on a remote node.
    Future<typename array_type::value_type> tile = array_.find(array_index);

    const bool consumable_tile = !array_.is_local(array_index);

    return eval_tile(tile, consumable_tile);
  }

  /// Discard a tile that is not needed

  /// This function handles the cleanup for tiles that are not needed in
  /// subsequent computation.
  virtual void discard_tile(ordinal_type) const {
    const_cast<ArrayEvalImpl_*>(this)->notify();
  }

 private:
  value_type make_tile(const typename array_type::value_type& tile,
                       const bool consume) const {
    return value_type(tile, op_, consume);
  }

  /// Evaluate a single LazyArrayTile
  madness::Future<value_type> eval_tile(
      const madness::Future<typename array_type::value_type>& tile,
      const bool consumable_tile) const {
    // Insert the tile into this evaluator for subsequent processing
    if (tile.probe()) {
      // Skip the task since the tile is ready
      Future<value_type> result;
      result.set(make_tile(tile, consumable_tile));
      const_cast<ArrayEvalImpl_*>(this)->notify();
      return result;
    } else {
      // Spawn a task to set the tile when the input tile is not ready.
      Future<value_type> result = TensorImpl_::world().taskq.add(
          shared_from_this(), &ArrayEvalImpl_::make_tile, tile, consumable_tile,
          madness::TaskAttributes::hipri());
      result.register_callback(const_cast<ArrayEvalImpl_*>(this));
      return result;
    }
  }
  /// Evaluate the tiles of this tensor

  /// This function will evaluate the children of this distributed evaluator
  /// and evaluate the tiles for this distributed evaluator.
  /// \return The number of tiles that will be set by this process
  virtual int internal_eval() {
    // Counter for the number of tasks submitted by this object
    int task_count = 0;

    // Get a count of the number of local tiles.
    if (TensorImpl_::shape().is_dense()) {
      task_count = TensorImpl_::pmap()->local_size();
    } else {
      // Create iterator to tiles that are local for this evaluator.
      typename array_type::pmap_interface::const_iterator it =
          TensorImpl_::pmap()->begin();
      const typename array_type::pmap_interface::const_iterator end =
          TensorImpl_::pmap()->end();

      for (; it != end; ++it) {
        if (!TensorImpl_::is_zero(*it)) ++task_count;
      }
    }

    return task_count;
  }

};  // class ArrayEvalImpl

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_DIST_EVAL_ARRAY_EVAL_H__INCLUDED
