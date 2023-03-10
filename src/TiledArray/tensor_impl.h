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

#ifndef TILEDARRAY_TENSOR_IMPL_H__INCLUDED
#define TILEDARRAY_TENSOR_IMPL_H__INCLUDED

#include "TiledArray/external/madness.h"
#include "error.h"
#include "policies/dense_policy.h"
#include "policies/sparse_policy.h"

namespace TiledArray {
namespace detail {

/// Tensor implementation and base for other tensor implementation objects

/// This implementation object holds the meta data for tensor object, which
/// includes tiled range, shape, and process map.
/// \note The process map must be set before data elements can be set.
/// \note It is the users responsibility to ensure the process maps on all
/// nodes are identical.
template <typename Policy>
class TensorImpl : private NO_DEFAULTS {
 public:
  typedef TensorImpl<Policy> TensorImpl_;
  typedef Policy policy_type;                        ///< Policy type
  typedef typename Policy::trange_type trange_type;  ///< Tiled range type
  typedef typename Policy::range_type range_type;  ///< Element/tile range type
  typedef typename Policy::index1_type index1_type;    ///< 1-index type
  typedef typename Policy::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename Policy::shape_type shape_type;      ///< Tensor shape type
  typedef typename Policy::pmap_interface
      pmap_interface;  ///< Process map interface type

 private:
  World& world_;                                ///< World that contains
  const trange_type trange_;                    ///< Tiled range type
  std::shared_ptr<const shape_type> shape_;     ///< Tensor shape
  std::shared_ptr<const pmap_interface> pmap_;  ///< Process map for tiles

 public:
  /// Constructor

  /// The size of shape must be equal to the volume of the tiled range tiles.
  /// \param world The world where this tensor will live
  /// \param trange The tiled range for this tensor
  /// \param shape The shape of this tensor
  /// \param pmap The tile-process map
  /// \param replicate_shape if true, will replicate the shape from rank 0, else
  ///        will use as is, but assert that the data is _bitwise_ identical
  ///        between ranks
  /// \throw TiledArray::Exception When the size of shape is not equal to
  /// zero
  TensorImpl(World& world, const trange_type& trange, const shape_type& shape,
             const std::shared_ptr<const pmap_interface>& pmap,
             bool replicate_shape = true)
      : world_(world),
        trange_(trange),
        shape_(std::make_shared<shape_type>(shape)),
        pmap_(pmap) {
    // ensure that shapes are identical on every rank
    if (replicate_shape && !shape.is_dense())
      world.gop.broadcast_serializable(*shape_, 0);
    else
      TA_ASSERT(is_replicated(world, *shape_));
    // Validate input data.
    TA_ASSERT(pmap_);
    TA_ASSERT(pmap_->size() == trange_.tiles_range().volume());
    TA_ASSERT(pmap_->rank() ==
              typename pmap_interface::size_type(world_.rank()));
    TA_ASSERT(pmap_->procs() ==
              typename pmap_interface::size_type(world_.size()));
    TA_ASSERT(shape_->validate(trange_.tiles_range()));
  }

  /// Virtual destructor
  virtual ~TensorImpl() {}

  /// Tensor process map accessor

  /// \return A shared pointer to the process map of this tensor
  /// \throw nothing
  const std::shared_ptr<const pmap_interface>& pmap() const { return pmap_; }

  /// Tensor process map accessor

  /// \return A shared pointer to the process map of this tensor
  /// \note mirrors the shape_shared , the use of this name is recommended, but
  /// pmap() is not deprecated
  const std::shared_ptr<const pmap_interface>& pmap_shared() const {
    return pmap_;
  }

  /// Tiles range accessor

  /// \return The range of tile indices
  /// \throw nothing
  const range_type& tiles_range() const { return trange_.tiles_range(); }

  /// Tensor tile volume accessor

  /// \return The number of tiles in the tensor
  /// \throw nothing
  ordinal_type size() const { return trange_.tiles_range().volume(); }

  /// Max count of local tiles

  /// This function is primarily available for debugging  purposes. The
  /// returned value is volatile and may change at any time; you should not
  /// rely on it in your algorithms.
  /// \return The max count of local tiles; for dense array this will be equal
  /// to the actual number of local tiles stored, but for a sparse array
  /// the actual number of stored tiles will be less than or equal to this.
  ordinal_type local_size() const {
    return static_cast<ordinal_type>(pmap_->local_size());
  }

  /// Query a tile owner

  /// \tparam Index The sized integral range type
  /// \param i The tile index to query
  /// \return The process ID of the node that owns tile \c i
  /// \throw TiledArray::Exception When \c i is outside the tiled range tile
  /// range
  /// \throw TiledArray::Exception When the process map has not been set
  template <typename Index, typename = std::enable_if_t<
                                detail::is_integral_sized_range_v<Index>>>
  ProcessID owner(const Index& index) const {
    const auto ord = trange_.tiles_range().ordinal(index);
    TA_ASSERT(trange_.tiles_range().includes_ordinal(ord));
    return pmap_->owner(ord);
  }

  /// Query a tile owner

  /// \tparam Integer An integer type
  /// \param i The tile index to query
  /// \return The process ID of the node that owns tile \c i
  /// \throw TiledArray::Exception When \c i is outside the tiled range tile
  /// range
  /// \throw TiledArray::Exception When the process map has not been set
  template <typename Integer,
            typename = std::enable_if_t<std::is_integral_v<Integer>>>
  ProcessID owner(const std::initializer_list<Integer>& index) const {
    const auto ord = trange_.tiles_range().ordinal(index);
    TA_ASSERT(trange_.tiles_range().includes_ordinal(ord));
    return pmap_->owner(ord);
  }

  /// Query a tile owner

  /// \tparam Ordinal An integer type
  /// \param i The tile index to query
  /// \return The process ID of the node that owns tile \c i
  /// \throw TiledArray::Exception When \c i is outside the tiled range tile
  /// range
  /// \throw TiledArray::Exception When the process map has not been set
  template <typename Ordinal>
  std::enable_if_t<std::is_integral_v<Ordinal>, ProcessID> owner(
      const Ordinal& ord) const {
    TA_ASSERT(trange_.tiles_range().includes_ordinal(ord));
    return pmap_->owner(ord);
  }

  /// Query for a locally owned tile

  /// \tparam Index The sized integral range type
  /// \param i The tile index to query
  /// \return \c true if the tile is owned by this node, otherwise \c false
  /// \throw TiledArray::Exception When the process map has not been set
  template <typename Index, typename = std::enable_if_t<
                                detail::is_integral_sized_range_v<Index>>>
  bool is_local(const Index& index) const {
    const auto ord = trange_.tiles_range().ordinal(index);
    TA_ASSERT(trange_.tiles_range().includes_ordinal(ord));
    return pmap_->is_local(ord);
  }

  /// Query for a locally owned tile

  /// \tparam Integer An integer type
  /// \param i The tile index to query
  /// \return \c true if the tile is owned by this node, otherwise \c false
  /// \throw TiledArray::Exception When the process map has not been set
  template <typename Integer,
            std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  bool is_local(const std::initializer_list<Integer>& index) const {
    const auto ord = trange_.tiles_range().ordinal(index);
    TA_ASSERT(trange_.tiles_range().includes_ordinal(ord));
    return pmap_->is_local(ord);
  }

  /// Query for a locally owned tile

  /// \tparam Ordinal An integer type
  /// \param i The tile index to query
  /// \return \c true if the tile is owned by this node, otherwise \c false
  /// \throw TiledArray::Exception When the process map has not been set
  template <typename Ordinal>
  std::enable_if_t<std::is_integral_v<Ordinal>, bool> is_local(
      const Ordinal& ord) const {
    TA_ASSERT(trange_.tiles_range().includes_ordinal(ord));
    return pmap_->is_local(ord);
  }

  /// Query for a zero tile

  /// \tparam Index The sized integral range type
  /// \param i The tile index to query
  /// \return \c true if the tile is zero, otherwise \c false
  /// \throw TiledArray::Exception When \c i is outside the tiled range tile
  /// range
  template <typename Index, typename = std::enable_if_t<
                                detail::is_integral_sized_range_v<Index> &&
                                !std::is_integral_v<Index>>>
  bool is_zero(const Index& index) const {
    const auto ord = trange_.tiles_range().ordinal(index);
    TA_ASSERT(trange_.tiles_range().includes_ordinal(ord));
    return shape_->is_zero(ord);
  }

  /// Query for a zero tile

  /// \tparam Integer An integer type
  /// \param i The tile index to query
  /// \return \c true if the tile is zero, otherwise \c false
  /// \throw TiledArray::Exception When \c i is outside the tiled range tile
  /// range
  template <typename Integer,
            typename = std::enable_if_t<std::is_integral_v<Integer>>>
  bool is_zero(const std::initializer_list<Integer>& index) const {
    const auto ord = trange_.tiles_range().ordinal(index);
    TA_ASSERT(trange_.tiles_range().includes_ordinal(ord));
    return shape_->is_zero(ord);
  }

  /// Query for a zero tile

  /// \tparam Ordinal An integer type
  /// \param i The tile index to query
  /// \return \c true if the tile is zero, otherwise \c false
  /// \throw TiledArray::Exception When \c i is outside the tiled range tile
  /// range
  template <typename Ordinal>
  std::enable_if_t<std::is_integral_v<Ordinal>, bool> is_zero(
      const Ordinal& ord) const {
    TA_ASSERT(trange_.tiles_range().includes_ordinal(ord));
    return shape_->is_zero(ord);
  }

  /// Query the density of the tensor

  /// \return \c true if the tensor is dense, otherwise false
  /// \throw nothing
  bool is_dense() const { return shape_->is_dense(); }

  /// Tensor shape accessor

  /// \return A reference to the tensor shape map
  const shape_type& shape() const { return *shape_; }

  /// Tensor shape accessor

  /// \return A reference to the tensor shape shared_ptr object
  const std::shared_ptr<const shape_type>& shape_shared() const {
    return shape_;
  }

  /// Tiled range accessor

  /// \return The tiled range of the tensor
  const trange_type& trange() const { return trange_; }

  /// \deprecated use TensorImpl::world()
  [[deprecated]] World& get_world() const {
    TA_ASSERT(World::exists(&world_));
    return world_;
  }

  /// World accessor

  /// \return A reference to the world that contains this tensor
  World& world() const {
    TA_ASSERT(World::exists(&world_));
    return world_;
  }

};  // class TensorImpl

#ifndef TILEDARRAY_HEADER_ONLY

extern template class TensorImpl<DensePolicy>;
extern template class TensorImpl<SparsePolicy>;

#endif  // TILEDARRAY_HEADER_ONLY

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_IMPL_H__INCLUDED
