#ifndef TILEDARRAY_TENSOR_IMPL_BASE_H__INCLUDED
#define TILEDARRAY_TENSOR_IMPL_BASE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/distributed_storage.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/bitset.h>
#include <cstddef>

namespace TiledArray {
  namespace detail {

    /// Base class for Tensor implementation objects

    /// This is the basis for other tiled tensor implementation objects. It
    /// provides the basic interface for accessing and setting the tensor's
    /// tiled range, shape, process map, and data. There are some restrictions
    /// on how the tensor may be modified. Most significantly, the volume of the
    /// tiled range's tiles must be constant. This allows the shape to be
    /// permuted, but not resized. Also, the size of shape must be either zero,
    /// which indicates the tensor is dense, or it must be exactly equal to the
    /// volume of the tiled range tiles. It is the responsibility of the derived
    /// class to ensure data consistancy, that is the derived class must
    /// explicitly permute the tensor's tiled range, shape, process map, and
    /// data.
    /// \tparam TRange The tiled range type
    /// \tparam Tile The tile or value_type of this tensor
    /// \note The process map must be set after construction before data elements
    /// can be set.
    template <typename TRange, typename Tile>
    class TensorImplBase : private NO_DEFAULTS{
    public:
      typedef std::size_t size_type; ///< Size type
      typedef TRange trange_type; ///< Tiled range type
      typedef typename trange_type::range_type range_type; ///< Tile range type
      typedef Pmap<size_type> pmap_interface; ///< Process map interface type
      typedef Tile value_type; ///< Tile or data type
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type; ///< The data container type
      typedef typename storage_type::const_iterator const_iterator; ///< Constant iterator type
      typedef typename storage_type::iterator iterator; ///< Iterator type
      typedef typename storage_type::future const_reference; ///< Constant reference type
      typedef typename storage_type::future reference; ///< Reference type

    private:

      trange_type trange_; ///< Tiled range type
      Bitset<> shape_; ///< Tensor shape (zero size == dense)
      storage_type data_; ///< Tile container

    public:

      /// Constructor

      /// The size of shape must be equal to the volume of the tiled range tiles.
      /// Also, the volume of trange must remain constant. This restriction allows
      /// the tiled range to be permuted, but not resized.
      /// \param arg The argument
      /// \param op The element transform operation
      /// \throw TiledArray::Exception When the size of shape is not equal to
      /// zero
      template <typename TR>
      TensorImplBase(madness::World& world, const TiledRange<TR>& trange, const Bitset<>& shape) :
        trange_(trange), shape_(shape), data_(world, trange_.tiles().volume())
      {
        TA_ASSERT((shape_.size() == trange_.tiles().volume()) || (shape_.size() == 0ul));
      }


      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      /// \throw nothing
      const std::shared_ptr<pmap_interface>& pmap() const { return data_.get_pmap(); }

      /// Initialize pmap

      /// \param pmap The process map
      /// \throw TiledArray::Exception When the process map has already been set
      /// \throw TiledArray::Exception When \c pmap is \c NULL
      void pmap(const std::shared_ptr<pmap_interface>& pmap) { data_.init(pmap); }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      /// \throw TiledArray::Exception when the trange of \c dest is not equal
      /// to the trange of this tensor.
      template <typename Dest>
      void eval_to(Dest& dest) {
        TA_ASSERT(trange() == dest.trange());

        // Add result tiles to dest
        typename pmap_interface::const_iterator end = data_.get_pmap()->end();
        typename pmap_interface::const_iterator it = data_.get_pmap()->begin();
        if(is_dense()) {
          for(; it != end; ++it)
            dest.set(*it, move(*it));
        } else {
          for(; it != end; ++it)
            if(! is_zero(*it))
              dest.set(*it, move(*it));
        }
      }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      /// \throw nothing
      const range_type& range() const { return trange_.tiles(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      /// \throw nothing
      size_type size() const { return data_.max_size(); }

      /// Local element count

      /// This function is primarily available for debugging  purposes. The
      /// returned value is volatile and may change at any time; you should not
      /// rely on it in your algorithms.
      /// \return The current number of local tiles stored in the tensor.
      size_type local_size() const { return data_.size(); }

      /// Query a tile owner

      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      /// \throw TiledArray::Exception When \c i is outside the tiled range tile
      /// range
      /// \throw TiledArray::Exception When the process map has not been set
      ProcessID owner(size_type i) const { return data_.owner(i); }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      /// \throw nothing
      /// \throw TiledArray::Exception When the process map has not been set
      bool is_local(size_type i) const { return data_.is_local(i); }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      /// \throw TiledArray::Exception When \c i is outside the tiled range tile
      /// range
      bool is_zero(size_type i) const {
        TA_ASSERT(range().includes(i));
        if(is_dense())
          return false;
        return ! (shape_[i]);
      }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      /// \throw nothing
      bool is_dense() const { return shape_.size() == 0ul; }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      /// \throw TiledArray::Exception When this tensor is dense
      const TiledArray::detail::Bitset<>& shape() const {
        TA_ASSERT(! is_dense());
        return shape_;
      }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      /// \throw TiledArray::Exception When this tensor is dense
      TiledArray::detail::Bitset<>& shape() {
        TA_ASSERT(! is_dense());
        return shape_;
      }

      /// Set the shape

      /// \param s The new shape
      /// \throw TiledArray::Exception When the size of \c s is not equal to the
      /// size of this tensor or zero.
      void shape(TiledArray::detail::Bitset<> s) {
        TA_ASSERT((s.size() == trange_.tiles().volume()) || (s.size() == 0ul));
        s.swap(shape_);
      }

      /// Set shape values

      /// Modify the shape value for tile \c i to \c value
      /// \param i Tile index
      /// \param value The value of the tile
      /// \throw TiledArray::Exception When this tensor is dense
      void shape(size_type i, bool value = true) {
        TA_ASSERT(! is_dense());

        shape_.set(i, value);
      }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const { return trange_; }

      /// Set tiled range

      /// \tparam TR TiledRange type
      /// \param tr Tiled range to set
      template <typename TR>
      void trange(const TiledRange<TR>& tr) {
        TA_ASSERT(tr.tiles().volume() == trange_.tiles().volume());
        trange_ = tr;
      }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      /// \throw TiledArray::Exception When tile \c i is zero
      const_reference operator[](size_type i) const {
        TA_ASSERT(! is_zero(i));
        return data_[i];
      }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      /// \throw TiledArray::Exception When tile \c i is zero
      reference operator[](size_type i) {
        TA_ASSERT(! is_zero(i));
        return data_[i];
      }

      /// Tile move

      /// Tile is removed after it is set.
      /// \param i The tile index
      /// \return Tile \c i
      /// \throw TiledArray::Exception When tile \c i is zero
      const_reference move(size_type i) {
        TA_ASSERT(! is_zero(i));
        return data_.move(i);
      }


      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return data_.begin(); }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      iterator begin() { return data_.begin(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return data_.end(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      iterator end() { return data_.end(); }

      madness::World& get_world() const { return data_.get_world(); }

      template <typename Value>
      void set(size_type i, const Value& value) { data_.set(i, value); }

      /// Clear the tile data

      /// Remove all local tiles from the tensor.
      /// \note: Any tiles will remain in memory until the last reference
      /// is destroyed. This function only removes them from the container.
      void clear() { data_.clear(); }

    }; // class TensorImplBase

  }  // namespace detail
}  // namespace TiledArray


#endif // TILEDARRAY_TENSOR_IMPL_BASE_H__INCLUDED
