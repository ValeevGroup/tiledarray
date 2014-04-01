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

#ifndef TILEDARRAY_ARRAY_H__INCLUDED
#define TILEDARRAY_ARRAY_H__INCLUDED

#include <TiledArray/replicator.h>
#include <TiledArray/pmap/replicated_pmap.h>
#include <TiledArray/tensor.h>
#include <TiledArray/policies/dense_policy.h>
#include <TiledArray/tensor_impl.h>
#include <TiledArray/pmap/blocked_pmap.h>
#include <TiledArray/expressions.h>

namespace TiledArray {

  /// An n-dimensional, tiled array

  /// Array is the local representation of a global object. This means that the
  /// local array object will only contain a portion of the data. It may be used
  /// to construct distributed tensor algebraic operations.
  /// \tparam T The element type of for array tiles
  /// \tparam DIM The number of dimensions for this array object
  /// \tparam Tile The tile type [ Default = \c Tensor<T> ]
  template <typename T, unsigned int DIM, typename Tile = Tensor<T>, typename Policy = DensePolicy >
  class Array {
  public:
    typedef Array<T, DIM, Tile, Policy> Array_; ///< This object's type
    typedef TiledArray::detail::TensorImpl<Tile, Policy> impl_type;
    typedef T element_type; ///< The tile element type
    typedef typename impl_type::trange_type trange_type; ///< Tile range type
    typedef typename impl_type::range_type range_type; ///< Range type for array tiling
    typedef typename impl_type::shape_type shape_type; ///< Shape type for array tiling
    typedef typename impl_type::range_type::index index; ///< Array coordinate index type
    typedef typename impl_type::size_type size_type; ///< Size type
    typedef typename impl_type::value_type value_type; ///< Tile type
    typedef typename impl_type::eval_type eval_type; ///< The tile evaluation type
    typedef typename impl_type::reference future; ///< Future of \c value_type
    typedef typename impl_type::reference reference; ///< \c future type
    typedef typename impl_type::const_reference const_reference; ///< \c future type
    typedef typename impl_type::iterator iterator; ///< Local tile iterator
    typedef typename impl_type::const_iterator const_iterator; ///< Local tile const iterator
    typedef typename impl_type::pmap_interface pmap_interface; ///< Process map interface type


    /// Default constructor
    Array() : pimpl_() { }

    /// Copy constructor

    /// This is a shallow copy, that is no data is copied.
    /// \param other The array to be copied
    Array(const Array_& other) : pimpl_(other.pimpl_) { }

    /// Dense array constructor

    /// \param w The world where the array will live.
    /// \param tr The tiled range object that will be used to set the array tiling.
    /// \param pmap The tile index -> process map
    Array(madness::World& w, const trange_type& tr,
        const std::shared_ptr<pmap_interface>& pmap = std::shared_ptr<pmap_interface>()) :
      pimpl_(new impl_type(w, tr, shape_type(), (pmap ? pmap : Policy::default_pmap(w, tr.tiles().volume()))),
          madness::make_deferred_deleter<impl_type>(w))
    {
      TA_USER_ASSERT(tr.tiles().dim() == DIM,
          "The dimensions of the tiled range do not match that of the array object.");
    }

    /// Sparse array constructor

    /// \param w The world where the array will live.
    /// \param tr The tiled range object that will be used to set the array tiling.
    /// \param shape Bitset of the same length as \c tr. Describes the array shape: bit set (1)
    ///        means tile exists, else tile does not exist.
    /// \param pmap The tile index -> process map
    /// \param collective_shape_init If true then the collective_init method for
    /// shape will be invoced by the constructor
    Array(madness::World& w, const trange_type& tr, const shape_type& shape,
        const std::shared_ptr<pmap_interface>& pmap = std::shared_ptr<pmap_interface>()) :
      pimpl_(new impl_type(w, tr, shape, (pmap ? pmap : Policy::default_pmap(w, tr.tiles().volume()))),
          madness::make_deferred_deleter<impl_type>(w))
    {
      TA_USER_ASSERT(tr.tiles().dim() == DIM,
          "The dimensions of the tiled range do not match that of the array object.");
    }

//    /// Construct Array from a pimpl
//
//    /// \param pimpl The implementation pointer
//    Array(const std::shared_ptr<impl_type>& pimpl) : pimpl_(pimpl) { }

    /// Copy constructor

    /// This is a shallow copy, that is no data is copied.
    /// \param other The array to be copied
    Array_& operator=(const Array_& other) {
      pimpl_ = other.pimpl_;
      return *this;
    }

    /// Evaluate this object

    /// \return A future that is set when the array evaluation is complete
    madness::Future<bool> eval() { return madness::Future<bool>(true); }

    /// Begin iterator factory function

    /// \return An iterator to the first local tile.
    iterator begin() {
      check_pimpl();
      return pimpl_->begin();
    }

    /// Begin const iterator factory function

    /// \return A const iterator to the first local tile.
    const_iterator begin() const {
      check_pimpl();
      return pimpl_->begin();
    }

    /// End iterator factory function

    /// \return An iterator to one past the last local tile.
    iterator end() {
      check_pimpl();
      return pimpl_->end();
    }

    /// End const iterator factory function

    /// \return A const iterator to one past the last local tile.
    const_iterator end() const {
      check_pimpl();
      return pimpl_->end();
    }

    /// Find local or remote tile

    /// \tparam Index The index type
    template <typename Index>
    madness::Future<value_type> find(const Index& i) const {
      check_index(i);
      return pimpl_->operator[](i);
    }

    /// Set the data of tile \c i

    /// \tparam Index \c index or an integral type
    /// \tparam InIter An input iterator
    /// \param i The index of the tile to be set
    /// \param first The iterator that points to the new tile data
    template <typename Index, typename InIter>
    typename madness::enable_if<detail::is_input_iterator<InIter> >::type
    set(const Index& i, InIter first) {
      check_index(i);
      pimpl_->set(i, value_type(pimpl_->trange().make_tile_range(i), first));
    }

  private:

    template <typename Index, typename Value>
    class MakeTile : public madness::TaskInterface {
    private:
      std::shared_ptr<impl_type> pimpl_;
      const Index index_;
      const typename Value::value_type value_;
      madness::Future<Value> result_;

      static const std::shared_ptr<impl_type>&
      checked_pimpl(const std::shared_ptr<impl_type>& pimpl) {
        TA_USER_ASSERT(pimpl,
            "The Array::pimpl has not been initialized, likely reason: it was default constructed and used.");
        return pimpl;
      }

    public:
      MakeTile(const std::shared_ptr<impl_type>& pimpl, const Index& index, const T& value) :
        madness::TaskInterface(),
        pimpl_(pimpl),
        index_(index),
        value_(value),
        result_(pimpl->get(index))
      { }

      virtual void run(madness::World&) {
        result_.set(value_type(pimpl_->trange().make_tile_range(index_), value_));
      }

      const madness::Future<value_type>& result() const { return result_; }

    }; // class MakeTile

  public:

    template <typename Index>
    void set(const Index& i, const T& v = T()) {
      check_index(i);
      pimpl_->get_world().taskq.add(new MakeTile<Index, value_type>(pimpl_, i, v));
    }

    /// Set tile \c i with future \c f

    /// \tparam Index The index type (i.e. index or size_type)
    /// \param i The tile index to be set
    template <typename Index>
    void set(const Index& i, const madness::Future<value_type>& f) {
      check_index(i);
      pimpl_->set(i, f);
    }

    /// Set tile \c i to value \c v

    /// \tparam Index The index type (i.e. index or size_type)
    /// \param i The tile index to be set
    /// \param v The tile value
    template <typename Index>
    void set(const Index& i, const value_type& v) {
      check_index(i);
      pimpl_->set(i, v);
    }

    void set_all_local(const T& v = T()) {
      check_pimpl();
      typename pmap_interface::const_iterator it = pimpl_->pmap()->begin();
      const typename pmap_interface::const_iterator end = pimpl_->pmap()->end();

      if(pimpl_->is_dense()) {
        for(; it != end; ++it)
          set(*it, v);
      } else {
        for(; it != end; ++it)
          if(! pimpl_->is_zero(*it))
            set(*it, v);
      }
    }

    /// Tiled range accessor

    /// \return A const reference to the tiled range object for the array
    /// \throw nothing
    const trange_type& trange() const {
      check_pimpl();
      return pimpl_->trange();
    }

    /// Tile range accessor

    /// \return A const reference to the range object for the array tiles
    /// \throw nothing
    const range_type& range() const {
      check_pimpl();
      return pimpl_->range();
    }

    /// Element range accessor

    /// \return A const reference to the range object for the array elements
    /// \throw nothing
    const typename trange_type::tile_range_type& elements() const {
      check_pimpl();
      return pimpl_->trange().elements();
    }

    size_type size() const {
      check_pimpl();
      return pimpl_->size();
    }

    /// Create a tensor expression

    /// \param v A string with a comma-separated list of variables
    /// \return A const tensor expression object
    TiledArray::expressions::TsrExpr<const Array_>
    operator ()(const std::string& vars) const {
      return TiledArray::expressions::TsrExpr<const Array_>(*this, vars);
    }

    /// Create a tensor expression

    /// \param v A string with a comma-separated list of variables
    /// \return A non-const tensor expression object
    TiledArray::expressions::TsrExpr<Array_>
    operator ()(const std::string& vars) {
      return TiledArray::expressions::TsrExpr<const Array_>(*this, vars);
    }

    /// World accessor

    /// \return A reference to the world that owns this array.
    madness::World& get_world() const {
      check_pimpl();
      return pimpl_->get_world();
    }

    /// Process map accessor

    /// \return A reference to the world that owns this array.
    const std::shared_ptr<pmap_interface>& get_pmap() const {
      check_pimpl();
      return pimpl_->pmap();
    }

    /// Check dense/sparse

    /// \return \c true when \c Array is dense, \c false otherwise.
    bool is_dense() const {
      check_pimpl();
      return pimpl_->is_dense();
      }


    /// Shape map accessor

    /// Bits are \c true when the tile exists, either locally or remotely. No
    /// no communication required.
    /// \return A bitset that maps the existence of tiles.
    /// \throw TiledArray::Exception When the Array is dense.
    const shape_type& get_shape() const {
      TA_USER_ASSERT(! is_dense(),
          "You cannot access the shape of a dense array. Use Array::is_dense() to check for a dense array.");
      return pimpl_->shape();
    }

    /// Tile ownership

    /// \tparam Index An index type
    /// \param i The index of a tile
    /// \return The process ID of the owner of a tile.
    /// \note This does not indicate whether a tile exists or not. Only, who
    /// would own it if it does exist.
    template <typename Index>
    ProcessID owner(const Index& i) const {
      check_index(i);
      return pimpl_->owner(i);
    }

    template <typename Index>
    bool is_local(const Index& i) const {
      check_index(i);
      return pimpl_->is_local(i);
    }

    /// Check for zero tiles

    /// \return \c true if tile at index \c i is zero, false if the tile is
    /// non-zero or remote existence data is not available.
    template <typename Index>
    bool is_zero(const Index& i) const {
      check_index(i);
      return pimpl_->is_zero(i);
    }

    /// Swap this array with \c other

    /// \param other The array to be swapped with this array.
    void swap(Array_& other) { std::swap(pimpl_, other.pimpl_); }

    /// Convert a distributed \c Array into a replicated array
    void make_replicated() {
      check_pimpl();
      if((! pimpl_->pmap()->is_replicated()) && (get_world().size() > 1)) {
        // Construct a replicated array
        std::shared_ptr<pmap_interface> pmap(new detail::ReplicatedPmap(get_world(), size()));
        Array_ result = (is_dense() ? Array_(get_world(), trange(), pmap) : Array_(get_world(), trange(), get_shape(), pmap));

        // Create the replicator object that will do an all-to-all broadcast of
        // the local tile data.
        detail::Replicator<Array_>* replicator = new detail::Replicator<Array_>(*this, result);

        // Put the replicator pointer in the deferred cleanup object so it will
        // be deleted at the end of the next fence.
        madness::DeferredDeleter<detail::Replicator<Array_> > deleter =
            madness::make_deferred_deleter<detail::Replicator<Array_> >(get_world());
        deleter(replicator);

        result.swap(*this);
      }
    }

    bool is_initialized() const { return static_cast<bool>(pimpl_); }

  private:

    template <typename Index>
    typename madness::enable_if<std::is_integral<Index> >::type
    check_index(const Index i) const {
      check_pimpl();
      TA_USER_ASSERT(pimpl_->range().includes(i),
          "The ordinal index used to access an array tile is out of range.");
    }

    template <typename Index>
    typename madness::disable_if<std::is_integral<Index> >::type
    check_index(const Index& i) const {
      check_pimpl();
      TA_USER_ASSERT(pimpl_->range().includes(i),
          "The coordinate index used to access an array tile is out of range.");
      TA_USER_ASSERT(i.size() == DIM,
          "The number of elements in the coordinate index does not match the dimension of the array.");
    }

    /// Makes sure pimpl has been initialized
    void check_pimpl() const {
      TA_USER_ASSERT(pimpl_,
          "The Array has not been initialized, likely reason: it was default constructed and used.");
    }

    std::shared_ptr<impl_type> pimpl_; ///< Array implementation pointer
  }; // class Array

  /// Add the tensor to an output stream

  /// This function will iterate through all tiles on node 0 and print non-zero
  /// tiles. It will wait for each tile to be evaluated (i.e. it is a blocking
  /// function). Tasks will continue to be processed.
  /// \tparam T The element type of Array
  /// \tparam DIM The number of dimensions
  /// \tparam Tile The Tile type
  /// \param os The output stream
  /// \param a The array to be put in the output stream
  /// \return A reference to the output stream
  template <typename T, unsigned int DIM, typename Tile>
  inline std::ostream& operator<<(std::ostream& os, const Array<T, DIM, Tile>& a) {
    if(a.get_world().rank() == 0) {
      for(std::size_t i = 0; i < a.size(); ++i)
        if(! a.is_zero(i)) {
          const typename Array<T, DIM, Tile>::value_type tile = a.find(i).get();
          os << i << ": " << tile  << "\n";
        }
    }
    return os;
  }

} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_H__INCLUDED
