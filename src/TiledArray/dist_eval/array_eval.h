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

#ifdef TILEDARRAY_HAS_DEVICE
  // TODO need a better design on how to manage the lifetime of converted Tile
  mutable conversion_result_type conversion_tile_;
#endif
 public:
  /// Default constructor
  LazyArrayTile()
      : tile_(),
        op_(),
        consume_(false)
#ifdef TILEDARRAY_HAS_DEVICE
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
#ifdef TILEDARRAY_HAS_DEVICE
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
#ifdef TILEDARRAY_HAS_DEVICE
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
#ifdef TILEDARRAY_HAS_DEVICE
    conversion_tile_ = other.conversion_tile_;
#endif
    return *this;
  }

  /// Query runtime consumable status

  /// \return \c true if this tile is consumable, otherwise \c false .
  bool is_consumable() const { return consume_ || op_->permutation(); }

  /// Convert tile to evaluation type using the op object
#ifdef TILEDARRAY_HAS_DEVICE

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

#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
  // tracing artifacts
  using pending_counter_t = std::atomic<std::size_t>[];  // 1 counter per rank
  mutable std::shared_ptr<pending_counter_t>
      ntiles_pending_;  // number of pending tiles from each rank
  mutable std::shared_ptr<pending_counter_t>
      ntasks_pending_;  // number of pending tasks using data from each rank

  struct AtomicCounterDecreaser : public madness::CallbackInterface {
    std::shared_ptr<std::atomic<std::size_t>> counter;

    AtomicCounterDecreaser(std::shared_ptr<std::atomic<std::size_t>> counter)
        : counter(std::move(counter)) {}
    void notify() override {
      --(*counter);
      delete this;
    }
  };
#endif

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
                const std::shared_ptr<const pmap_interface>& pmap,
                const Perm& perm, const op_type& op)
      : DistEvalImpl_(world, trange, shape, pmap, outer(perm)),
        array_(array),
        op_(std::make_shared<op_type>(op)),
        block_range_()
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
        ,
        ntiles_pending_(new std::atomic<std::size_t>[world.size()]),
        ntasks_pending_(new std::atomic<std::size_t>[world.size()])
#endif
  {
#if 0
    std::stringstream ss;
    ss << "ArrayEvalImpl: id=" << this->id();
    if (array_) ss << " array.id()=" << array_.id();
    ss << "\n";
    std::cout << ss.str();
#endif

#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
    for (auto rank = 0; rank != world.size(); ++rank) {
      ntiles_pending_[rank] = 0;
      ntasks_pending_[rank] = 0;
    }
#endif
  }

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
                const std::shared_ptr<const pmap_interface>& pmap,
                const Perm& perm, const op_type& op, const Index1& lower_bound,
                const Index2& upper_bound)
      : DistEvalImpl_(world, trange, shape, pmap, outer(perm)),
        array_(array),
        op_(std::make_shared<op_type>(op)),
        block_range_(array.trange().tiles_range(), lower_bound, upper_bound)
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
        ,
        ntiles_pending_(new std::atomic<std::size_t>[world.size()]),
        ntasks_pending_(new std::atomic<std::size_t>[world.size()])
#endif
  {
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
    for (auto rank = 0; rank != world.size(); ++rank) {
      ntiles_pending_[rank] = 0;
      ntasks_pending_[rank] = 0;
    }
#endif
  }

  /// Virtual destructor
  virtual ~ArrayEvalImpl() {
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
    if (std::find_if(ntiles_pending_.get(),
                     ntiles_pending_.get() + this->world().size(),
                     [](const auto& v) { return v != 0; }) !=
        ntiles_pending_.get() + this->world().size()) {
      madness::print_error(
          "ArrayEvalImpl: pending tiles at destruction! (id=", this->id(), ")");
      abort();
    }
    if (std::find_if(ntasks_pending_.get(),
                     ntasks_pending_.get() + this->world().size(),
                     [](const auto& v) { return v != 0; }) !=
        ntasks_pending_.get() + this->world().size()) {
      madness::print_error(
          "ArrayEvalImpl: pending tasks at destruction! (id=", this->id(), ")");
      abort();
    }
#endif
  }

  Future<value_type> get_tile(ordinal_type i) const override {
    // Get the array index that corresponds to the target index
    auto array_index = DistEvalImpl_::perm_index_to_source(i);

    // If this object only uses a sub-block of the array, shift the tile
    // index to the correct location.
    if (block_range_.rank()) array_index = block_range_.ordinal(array_index);

    const bool arg_tile_is_remote = !array_.is_local(array_index);
    const ProcessID arg_tile_owner = array_.owner(array_index);

    Future<value_type> result;
    bool task_created = false;
    if (arg_tile_is_remote) {
      TA_ASSERT(arg_tile_owner != array_.world().rank());
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
      ntiles_pending_[arg_tile_owner]++;
#endif
      auto arg_tile = array_.find(array_index);
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
      arg_tile.register_callback(
          new AtomicCounterDecreaser(std::shared_ptr<std::atomic<std::size_t>>(
              ntiles_pending_, ntiles_pending_.get() + arg_tile_owner)));
#endif
      std::tie(result, task_created) =
          eval_tile(arg_tile, /* consumable_tile = */ true
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
                    ,
                    arg_tile_owner
#endif
          );
    } else {
      TA_ASSERT(arg_tile_owner == array_.world().rank());
      std::tie(result, task_created) = eval_tile(array_.find_local(array_index),
                                                 /* consumable_tile = */ false
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
                                                 ,
                                                 arg_tile_owner
#endif
      );
    }
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
    TA_ASSERT(ntiles_pending_[this->world().rank()] == 0);
    // even if data is local we may have created a task to evaluate it
    // TA_ASSERT(ntasks_pending_[this->world().rank()] == 0);
#endif
    return result;
  }

  void discard_tile(ordinal_type i) const override {
    TA_ASSERT(this->is_local(i));
    const_cast<ArrayEvalImpl_*>(this)->notify();
  }

 private:
  value_type make_tile(const typename array_type::value_type& tile,
                       const bool consume) const {
    return value_type(tile, op_, consume);
  }

  /// Evaluate a single LazyArrayTile
  /// @return A pair of the future to the tile and a boolean indicating whether
  /// a task was created to produce the tile
  [[nodiscard]] std::pair<madness::Future<value_type>, bool> eval_tile(
      const madness::Future<typename array_type::value_type>& tile,
      const bool consumable_tile
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
      ,
      const ProcessID tile_owner
#endif
  ) const {
    // Insert the tile into this evaluator for subsequent processing
    if (tile.probe()) {
      // Skip the task since the tile is ready
      Future<value_type> result;
      result.set(make_tile(tile, consumable_tile));
      const_cast<ArrayEvalImpl_*>(this)->notify();
      return {result, false};
    } else {
      // Spawn a task to set the tile when the input tile is not ready.
      Future<value_type> result = TensorImpl_::world().taskq.add(
          shared_from_this(), &ArrayEvalImpl_::make_tile, tile, consumable_tile,
          madness::TaskAttributes::hipri());
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
      ntasks_pending_[tile_owner]++;
      result.register_callback(
          new AtomicCounterDecreaser(std::shared_ptr<std::atomic<std::size_t>>(
              ntasks_pending_, ntasks_pending_.get() + tile_owner)));
#endif
      result.register_callback(const_cast<ArrayEvalImpl_*>(this));
      return {result, true};
    }
  }
  /// Evaluate the tiles of this tensor

  /// This function will evaluate the children of this distributed evaluator
  /// and evaluate the tiles for this distributed evaluator.
  /// \return The number of tiles that will be set by this process
  int internal_eval() override { return TensorImpl_::local_nnz(); }

#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
  std::string status() const override {
    std::stringstream ss;
    ss << "ArrayEvalImpl: array.id()=" << array_.id();
    ss << " ntiles_pending=[";
    for (auto rank = 0; rank != this->world().size(); ++rank) {
      ss << " " << ntiles_pending_[rank];
    }
    ss << "] ntasks_pending=[";
    for (auto rank = 0; rank != this->world().size(); ++rank) {
      ss << " " << ntasks_pending_[rank];
    }
    ss << "]\n";
    return ss.str();
  }
#endif
};  // class ArrayEvalImpl

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_DIST_EVAL_ARRAY_EVAL_H__INCLUDED
