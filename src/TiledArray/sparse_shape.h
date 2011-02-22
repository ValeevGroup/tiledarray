#ifndef TILEDARRAY_SPARSE_SHAPE_H__INCLUDED
#define TILEDARRAY_SPARSE_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/shape.h>
#include <TiledArray/utility.h>
#include <boost/dynamic_bitset.hpp>
#include <boost/scoped_array.hpp>

namespace TiledArray {

  template<typename CS, typename Key>
  class SparseShape :
      public Shape<CS, Key>,
      public madness::WorldObject<SparseShape<CS, Key> >
  {
  protected:
    typedef SparseShape<CS, Key> SparseShape_;
    typedef Shape<CS, Key> Shape_;
    typedef madness::WorldObject<SparseShape_ > WorldObject_;
    typedef Key key_type;

  public:
    typedef typename Shape_::index index;               ///< index type
    typedef typename Shape_::ordinal_index ordinal_index; ///< ordinal index type
    typedef typename Shape_::range_type range_type;     ///< Range type of shape
    typedef madness::WorldDCPmapInterface< key_type > pmap_interface_type;
                                                        ///< Process map interface type

    /// Primary constructor

    /// \tparam InIter Input list input iterator type
    /// \param w The world where this shape lives
    /// \param r The range object associated with this shape
    /// \param pm The process map for this shape
    /// \param first First element of a list of tiles that will be stored locally
    /// \param last Last element of a list of tiles that will be stored locally
    /// \note Tiles in the list that are not owned by this process (according to
    /// the process map) are ignored.
    template <typename InIter>
    SparseShape(madness::World& w, const range_type& r,
        const madness::SharedPtr<pmap_interface_type> pm, InIter first, InIter last) :
      Shape_(r),
      WorldObject_(w),
      pmap_(pm),
      tiles_(make_tiles(first, last))
    {
      WorldObject_::process_pending();
    }

    SparseShape(const SparseShape_& other) :
      Shape_(other),
      WorldObject_(other),
      pmap_(other.pmap_),
      tiles_(other.tiles_)
    { }

    virtual ~SparseShape() { }

    virtual boost::shared_ptr<Shape_> clone() const {
      return boost::dynamic_pointer_cast<Shape_>(
          boost::make_shared<SparseShape_>(*this));
    }

    virtual const std::type_info& type() const { return typeid(SparseShape_); }


  private:

    /// Construct the tile bitset

    /// Generates a local bitset, then shares the bitset with other processes.
    /// \tparam InIter Input list input iterator type
    /// \param first First element of a list of tiles that will be stored locally
    /// \param last Last element of a list of tiles that will be stored locally
    /// \note Tiles in the list that are not owned by this process are ignored.
    template <typename InIter>
    boost::dynamic_bitset<unsigned long> make_tiles(InIter first, InIter last)
    {
      // Construct the bitset for the local data
      boost::dynamic_bitset<unsigned long> local(Shape_::range().volume());
      const std::size_t size = local.num_blocks();

      ordinal_index o = 0;
      for(; first != last; ++first) {
        if(this->is_local(*first)) {
          o = Shape_::ord_(*first);
          local.set(o, SparseShape_::local(o));
        }
      }

      // Construct the bitset for remote data
      boost::scoped_array<unsigned long> data(new unsigned long[size]);
      boost::to_block_range(local, data.get());

      WorldObject_::get_world().gop.reduce(data.get(), size, detail::bit_or<unsigned long>());
      boost::dynamic_bitset<unsigned long> remote(data.get(), data.get() + size);

      local |= remote;

      return local;
    }

    /// Check that a tiles information is stored locally.

    /// \param i The ordinal index to check.
    virtual bool local(ordinal_index i) const {
      return pmap_->owner(i) == WorldObject_::get_world().rank();
    }

    /// Probe for the presence of a tile in the shape

    /// \param i The index to be probed.
    virtual madness::Future<bool> probe(ordinal_index i) const {
      return madness::Future<bool>(tiles_[i]);
    }

    madness::SharedPtr<pmap_interface_type> pmap_;
    const boost::dynamic_bitset<unsigned long> tiles_;
  }; // class SparseShape

  template <typename CS, typename Key>
  inline bool is_sparse(const boost::shared_ptr<Shape<CS, Key> >& s) {
    return s->type() == typeid(SparseShape<CS,Key>);
  }

  template <typename CS, typename Key>
  inline bool is_sparse(const boost::shared_ptr<SparseShape<CS, Key> >&) {
    return true;
  }

} // namespace TiledArray

#endif // TILEDARRAY_SPARSE_SHAPE_H__INCLUDED
