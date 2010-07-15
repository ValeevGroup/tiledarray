#ifndef TILEDARRAY_SPARSE_SHAPE_H__INCLUDED
#define TILEDARRAY_SPARSE_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/shape.h>
#include <boost/shared_ptr.hpp>
#include <algorithm>
#include <utility>
#include <map>

namespace TiledArray {
/*
  template<typename I>
  class SparseShape :
      public Shape<I>,
      public madness::WorldObject<SparseShape<I> >,
      private madness::MutexReaderWriter
  {
  protected:
    typedef SparseShape<I> SparseShape_;
    typedef madness::WorldObject<SparseShape_ > WorldObject_;
    typedef madness::MutexReaderWriter MutexReaderWriter_;
    typedef Shape<I> Shape_;

    typedef madness::RemoteReference< madness::FutureImpl<bool> > remote_ref;

    // Note: Shape has private default constructor, copy constructor, and
    // assignment operator. These operations are not allowed here.

  public:
    typedef typename Shape_::ordinal_type ordinal_type;
    typedef typename Shape_::volume_type volume_type;
    typedef typename Shape_::size_array size_array;
    typedef madness::WorldDCPmapInterface<ordinal_type> pmap_type;

  private:
    static boost::shared_ptr<pmap_type> default_pmap(madness::World& w) {
      boost::shared_ptr<pmap_type> result(new madness::WorldDCDefaultPmap<ordinal_type>(w));
      return result;
    }

    static boost::shared_ptr<pmap_type> null_pmap() {
      boost::shared_ptr<pmap_type> result;
      return result;
    }

  public:

    /// Primary constructor
    template<typename R>
    SparseShape(madness::World& w, const R& r, const boost::shared_ptr<pmap_type> pm = null_pmap()) :
        Shape_(detail::sparse_shape, r),
        WorldObject_(w),
        MutexReaderWriter_(),
        dependancy_(),
        n_(r.volume() / sizeof(unsigned long) + (r.volume() % sizeof(unsigned long) == 0 ? 0 : 1)),
        tiles_(new unsigned long[n_]),
        initialized_(false),
        pmap_(pm.get() != NULL ? pm : default_pmap(w))
    {
      TA_ASSERT(pmap_.get() != NULL, std::runtime_error, "Pmap pointer is NULL.");
      std::fill(tiles_, tiles_ + n_, 0ul);
    }

    /// Constructor defined by an size array.
    template<typename Size>
    SparseShape(madness::World& w, const Size& size, detail::DimensionOrderType o,
      const boost::shared_ptr<pmap_type>& pm) :
        Shape_(detail::sparse_shape, size, o),
        WorldObject_(w),
        MutexReaderWriter_(),
        dependancy_(),
        n_(Shape_::volume() / sizeof(unsigned long) + (Shape_::volume() % sizeof(unsigned long) == 0 ? 0 : 1)),
        tiles_(new unsigned long[n_]),
        initialized_(false),
        pmap_(pm)
    { }

    /// Constructor defined by an upper and lower bound.

    /// All elements of finish must be greater than or equal to those of start.
    template<typename Index>
    SparseShape(madness::World& w, const Index& start, const Index& finish, detail::DimensionOrderType o,
      const boost::shared_ptr<pmap_type>& pm) :
        Shape_(detail::sparse_shape, start, finish, o),
        WorldObject_(w),
        MutexReaderWriter_(),
        dependancy_(),
        n_(Shape_::volume() / sizeof(unsigned long) + (Shape_::volume() % sizeof(unsigned long) == 0 ? 0 : 1)),
        tiles_(new unsigned long[n_]),
        initialized_(false),
        pmap_(pm)
    { }

    virtual ~SparseShape() {
      delete [] tiles_;
    }

    /// Add a tile at index i to the sparse shape.

    /// This will add a single tile to the sparse shape. It CANNOT be called
    /// after set_initialized() (a std::runtime_error exception will be thrown
    /// if this happens).
    template<typename Index>
    void add(const Index& i) {
      TA_ASSERT(range_includes(forward_index(i)), std::out_of_range,
          "Index i is not included by the range.");
      TA_ASSERT(is_local(i), std::runtime_error, "Index i is not be stored locally.");
      if(initialized_)
        TA_EXCEPTION(std::runtime_error,
            "Tiles cannot be added once set_initialized() is called.");

      set_tile(i);
    }


    /// Add a tile at index i to the sparse shape.

    /// This will add a single tile to the sparse shape. It CANNOT be called
    /// after set_initialized() (a std::runtime_error exception will be thrown
    /// if this happens).
    template<typename Compare, typename Index>
    void add(const Index& i, madness::Future<bool>& f1, madness::Future<bool>& f2, Compare logic = Compare()) {
      TA_ASSERT(range_includes(forward_index(i)), std::out_of_range,
          "Index i is not included by the range.");
      TA_ASSERT(is_local(i), std::runtime_error, "Index i is not be stored locally.");
      if(initialized_)
        TA_EXCEPTION(std::runtime_error,
            "Tiles cannot be added once set_initialized() is called.");
      const ordinal_type o = ord(i);

      if(f1.probe() && f2.probe()) {
        // The tile data is ready now, so set it.
        if(logic(f1.get(), f2.get()))
          set_tile(i);
      } else {
        // One or both of the tiles is not ready, so spawn a task.
        add_dep(o, f1, f2, logic);
      }
    }


    /// Set the local tiles as initialized
    void set_initialized() {
      // Todo: In the future we likely need to do something more scalable than an all reduce.
      WorldObject_::get_world().gop.reduce(tiles_, n_, detail::bit_or<unsigned long>());
      WorldObject_::process_pending();

      // We probably do not need a lock here since it is only set once, but we
      // are going to error on the side of caution.
      MutexReaderWriter_::write_lock();
      initialized_ = true;
      MutexReaderWriter_::write_unlock();
    }

    /// Returns true if the tile data is stored locally.
    template<typename Index>
    bool is_local(const Index& i) const {
      // Todo: This will likely need to to be improved.
      return owner(ord(i)) == WorldObject_::get_world().rank();
    }

    /// Returns the owner of a given index.
    ProcessID owner(ordinal_type i) const {
      return pmap_->owner(i);
    }

  private:
    /// Add a tile dependency to the shape.

    /// This function will spawn a task and store a future to the result.
    template<typename Compare>
    madness::Future<bool> add_dep(ordinal_type i, madness::Future<bool> f1, madness::Future<bool> f2, Compare logic) {
      MutexReaderWriter_::write_lock();

      madness::Future<bool> result = WorldObject_::get_world().taskq.add(*this,
          & SparseShape_::future_set_tile<Compare>, i, f1, f2, logic);
      dependancy_.insert(std::make_pair(i, result));
      MutexReaderWriter_::write_unlock();

      return result;
    }

    /// Set the tile flag (no error checking)
    template<typename Index>
    void set_tile(const Index& i) {
      // get the ordinal index.
      const ordinal_type o = ord(i);

      // Calc flag.
      const unsigned int t = 1ul << (o % sizeof(unsigned long));
      // Calc flag offset
      const std::size_t n = o / sizeof(unsigned long);
      // Set tile flag.
      MutexReaderWriter_::write_lock();
      tiles_[n] = tiles_[n] | t;
      MutexReaderWriter_::write_unlock();
    }

    template<typename Compare>
    bool future_set_tile(const ordinal_type i, const bool a, const bool b, Compare logic) {
      const bool include_tile = logic(a, b);
      if(include_tile)
        set_tile(i);

      return include_tile;
    }

    /// Set the tile flag (no error checking)
    bool get_tile(const ordinal_type i) const {
      MutexReaderWriter_::read_lock();
      return tiles_[i / sizeof(unsigned long)] & (1ul << (i % sizeof(unsigned long)));
      MutexReaderWriter_::read_lock();
    }

    /// Handles tile probe messages.
    madness::Void fetch_handler(ProcessID requestor, ordinal_type i, const remote_ref& ref) const {
      madness::Future<bool> t = SparseShape_::tile_includes(i);

      // If the data is ready, send it now, otherwise send it when it is ready.
      if(t.probe())
        WorldObject_::send(requestor, &SparseShape_::fetch_return, t.get(), ref);
      else
        WorldObject_::get_world().taskq.add(this, &delay_handler, requestor, ref, t);

      return madness::None;
    }

    /// Task function used to handle requests for tiles that are not ready.
    madness::Void delay_handler(ProcessID requestor, const remote_ref& ref, bool result) const {
      WorldObject_::send(requestor, &SparseShape_::fetch_return, result, ref);

      return madness::None;
    }

    madness::Void fetch_return(bool val, const remote_ref& ref) const {
      madness::FutureImpl<bool>* f = ref.get();
      f->set(val);

      ref.dec();

      return madness::None;
    }

    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(ordinal_type i) const {
      TA_ASSERT(initialized_, std::runtime_error, "Shape is not full initialized.");

//      if(is_local(i))
      // Everything is local because it is broadcast. When data is distributed
      // this will need to change.
      MutexReaderWriter_::read_lock();
      typename std::map<ordinal_type, madness::Future<bool> >::const_iterator it
          = dependancy_.find(i);
      if(it != dependancy_.end()) {
        if(! it->second.probe())
          return it->second;

        MutexReaderWriter_::convert_read_lock_to_write_lock();
        dependancy_.erase(i);
        MutexReaderWriter_::convert_write_lock_to_read_lock();
      }
      MutexReaderWriter_::read_unlock();

      return madness::Future<bool>(get_tile(i));
    }

    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(size_array i) const {
      return SparseShape_::tile_includes(ord(i));
    }

    /// Returns true if the local data has been fully initialized.
    virtual bool initialized() const { return initialized_; }

    mutable std::map<ordinal_type, madness::Future<bool> > dependancy_;
    const std::size_t n_;
    unsigned long* tiles_; ///< tiling data.
    bool initialized_;
    boost::shared_ptr<pmap_type> pmap_;
  }; // class SparseShape
*/
} // namespace TiledArray

#endif // TILEDARRAY_SPARSE_SHAPE_H__INCLUDED
