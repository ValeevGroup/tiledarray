#ifndef TILEDARRAY_SHAPE_H__INCLUDED
#define TILEDARRAY_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/utility.h>
#include <TiledArray/array_ref.h>
#include <TiledArray/madness_runtime.h>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <algorithm>
#include <numeric>
#include <functional>

namespace TiledArray {

  template <typename I, unsigned int DIM, typename Tag, typename CS>
  class ArrayCoordinate;

  namespace detail {

    enum ShapeType {
      dense_shape,
      sparse_shape,
      predicated_shape
    };

  }  // namespace detail

// =============================================================================
// Shape

  /// Shape is the interface for all array shapes.

  /// Shape is used to determine the presence (or absence) of a tile in an
  /// during the construction of an array and for math operations on arrays.
  /// Math operations will be performed on shapes before any operations for
  /// arrays are placed in the task queue. The Shape class is the public
  /// interface for the all other shape classes. Note: Derived classes must call
  /// WorldObject_::process_pending() once the final shape is fully initialized.
  ///
  /// Template parameters:
  /// \var \c I is the ordinal index type.
  template<typename I>
  class Shape : boost::noncopyable {
  private:
    typedef Shape<I> Shape_;

  public:
    typedef I ordinal_type;
    typedef detail::ArrayRef<const I> size_array;
    typedef I volume_type;

  protected:
    /// Primary constructor

    /// \arg \c w is a madness World object reference
    /// \arg \c t is the derived class shape type
    Shape(detail::ShapeType t) : type_(t) { }

  public:
    /// Returns true if the shape is locally initialized.
    bool is_initialized() const {
      return this->initialized();
    }

    /// Returns true if the element is present.
    template<typename Index>
    madness::Future<bool> includes(const Index& i) const {

      if(! this->range_includes(forward_index(i)))
        return madness::Future<bool>(false);

      return this->tile_includes(forward_index(i));
    }

    /// Returns the shape type (dense_shape, sparse_shape, or predicated_shape).
    detail::ShapeType type() const { return type_; }

  protected:
    /// Returns true if the range includes the ordinal index i.
    virtual bool range_includes(const ordinal_type& i) const = 0;

    /// Returns true if the range includes the coordinate index i.
    virtual bool range_includes(const size_array& i) const = 0;

    /// Forward ordinal index
    ordinal_type forward_index(const ordinal_type& i) const {
      return i;
    }

    /// Forward coordinate index
    template<unsigned int DIM, typename Tag, typename CS>
    size_array forward_index(const ArrayCoordinate<I, DIM, Tag, CS>& i) const {
      return size_array(i.data());
    }

  private:
    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(ordinal_type) const = 0;
    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(size_array) const = 0;
    /// Returns true if the local data has been fully initialized.
    virtual bool initialized() const = 0;
    /// Returns a size_array of the array weight.
    virtual size_array weight() const = 0;
    /// Returns a size_array of the range start index.
    virtual size_array start() const = 0;
    /// Returns a size_array of the range finish index.
    virtual size_array finish() const = 0;
    /// Returns the volume of the range.
    virtual volume_type volume() const = 0;
    /// Returns the array dimension.
    virtual unsigned int dim() const = 0;

    detail::ShapeType type_; ///< Shape type (dense, sparse, or predicated).
  }; // class shape

// =============================================================================
// RangeShape class

  /// Shape range handles the range based functionality.

  /// This class instantiates the virtual functions that use the range object.
  /// The remaining functions are implemented by derived classes. This class is
  /// for internal use only and should not be used as a base pointer.
  ///
  /// Template parameters:
  /// \var \c R is the range object type.
  template<typename R>
  class RangeShape : public Shape<typename R::ordinal_type> {
  protected:
    typedef RangeShape<R> RangeShape_;
    typedef Shape<typename R::ordinal_type> Shape_;
    typedef R range_type;

  protected:
    typedef typename Shape_::ordinal_type ordinal_type;
    typedef typename Shape_::size_array size_array;
    typedef typename Shape_::volume_type volume_type;

    /// Primary constructor

    /// Since all tiles are present in a dense array, the shape is considered
    /// Immediately available.
    RangeShape(const boost::shared_ptr<range_type>& r, detail::ShapeType t) :
        Shape_(t), range_(r)
    { }

  protected:
    /// Returns the ordinal index i.
    ordinal_type ord(ordinal_type i) const { return i; }

    /// Calculates the ordinal index based on i.
    ordinal_type ord(const size_array& i) const {
      TA_ASSERT(i.size() == range_type::dim, std::runtime_error, "Array dimensions do not match range dimensions.");
      return detail::calc_ordinal(i.begin(), i.end(), range_->weight().begin(), range_->start().begin());
    }

    /// Calculates the ordinal index based on i.
    ordinal_type ord(const typename range_type::index_type& i) const {
      return detail::calc_ordinal(i.begin(), i.end(), range_->weight().begin(), range_->start().begin());
    }

    /// Forward the ordinal index
    virtual bool range_includes(const ordinal_type& i) const {
      return range_->includes(i);
    }

    /// Calculate the ordinal index
    virtual bool range_includes(const size_array& i) const {
      typename range_type::index_type ii(i.begin());
      return range_->includes(ii);
    }

  private:
    /// Returns a size_array of the array weight.
    virtual size_array weight() const { return size_array(range_->weight()); }

    /// Returns a size_array of the array start index.
    virtual size_array start() const {
      return size_array(range_->start().data());
    }

    /// Returns a size_array of the array start index.
    virtual size_array finish() const {
      return size_array(range_->finish().data());
    }

    /// Returns the volume of the range.
    virtual volume_type volume() const {
      return range_->volume();
    }

    /// Returns the array dimension.
    virtual unsigned int dim() const { return range_type::dim; }

    boost::shared_ptr<range_type> range_; ///< pointer to range data.
  }; // class RangeShape

// =============================================================================
// DenseShape class

  /// Dense shape used to construct Array objects.

  /// DenseShape is used to represent dense arrays. It is initialized with a
  /// madness world object and a range object. It includes all tiles included by
  /// the range object.
  ///
  /// Template parameters:
  /// \var \c R is the range object type.
  template<typename R>
  class DenseShape : public RangeShape<R> {
  protected:
    typedef RangeShape<R> RangeShape_;
    typedef typename RangeShape_::Shape_ Shape_;
    typedef typename RangeShape_::range_type range_type;

  public:
    typedef typename Shape_::ordinal_type ordinal_type;
    typedef typename Shape_::size_array size_array;

    /// Primary constructor

    /// Since all tiles are present in a dense array, the shape is considered
    /// Immediately available.
    DenseShape(const boost::shared_ptr<range_type>& r) :
        RangeShape_(r, detail::dense_shape)
    { }

  private:
    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(ordinal_type) const {
      return madness::Future<bool>(true);
    }

    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(size_array) const {
      return madness::Future<bool>(true);
    }

    /// Returns true if the local data has been fully initialized.
    virtual bool initialized() const { return true; }

  }; // class DenseShape

// =============================================================================
// SparseShape class

  template<typename R, typename H = madness::Hash_private::defhashT<typename R::ordinal_type> >
  class SparseShape : public RangeShape<R>, public madness::WorldObject<SparseShape<R, H> > {
  protected:
    typedef SparseShape<R, H> SparseShape_;
    typedef RangeShape<R> RangeShape_;
    typedef madness::WorldObject<SparseShape_ > WorldObject_;
    typedef typename RangeShape_::Shape_ Shape_;
    typedef typename RangeShape_::range_type range_type;
    typedef H hasher_type;

    typedef madness::RemoteReference< madness::FutureImpl<bool> > remote_ref;

    // Note: Shape has private default constructor, copy constructor, and
    // assignment operator. These operations are not allowed here.

  public:
    typedef typename Shape_::ordinal_type ordinal_type;
    typedef typename Shape_::volume_type volume_type;
    typedef typename Shape_::size_array size_array;

    /// Primary constructor

    /// This is a world object so all processes must construct SparseShape
    /// on all processes in the same order relative to other world objects.
    /// It is your responsibility to ensure that the range object is identical
    /// on all processes, otherwise process mapping will not be correct.
    SparseShape(madness::World& w, const boost::shared_ptr<range_type>& r,
      const hasher_type h = hasher_type()) :
        RangeShape_(r, detail::sparse_shape),
        WorldObject_(w),
        n_(r->volume() / sizeof(unsigned long) + (r->volume() % sizeof(unsigned long) == 0 ? 0 : 1)),
        tiles_(new unsigned long[n_]),
        initialized_(false),
        hasher_(h),
        pmap_(new madness::WorldDCDefaultPmap<ordinal_type, hasher_type>(w, hasher_)),
        me_(w.rank())
    {
      std::fill(tiles_, tiles_ + n_, 0ul);
    }

    /// Primary constructor
    SparseShape(madness::World& w, const boost::shared_ptr<range_type>& r,
      const boost::shared_ptr<madness::WorldDCDefaultPmap<ordinal_type, hasher_type> > pm) :
        RangeShape_(r, detail::sparse_shape),
        WorldObject_(w),
        n_(r->volume() / sizeof(unsigned long) + (r->volume() % sizeof(unsigned long) == 0 ? 0 : 1)),
        tiles_(new unsigned long[n_]),
        initialized_(false),
        hasher_(),
        pmap_(pm),
        me_(w.rank())
    {
      std::fill(tiles_, tiles_ + n_, 0ul);
    }

    ~SparseShape() {
      delete [] tiles_;
    }

    /// Add a tile at index i to the sparse shape.

    /// This will add a single tile to the sparse shape. It CANNOT be called
    /// after set_initialized() (a std::runtime_error exception will be thrown
    /// if this happens).
    template<typename Index>
    void add(const Index& i) {
      TA_ASSERT(SparseShape_::range_includes(forward_index(i)), std::out_of_range,
          "Index i is not included by the range.");
      TA_ASSERT(is_local(i), std::runtime_error, "Index i is not be stored locally.");
      if(initialized_)
        TA_EXCEPTION(std::runtime_error,
            "Tiles cannot be added once set_initialized() is called.");

      ordinal_type o = ord(i);

      unsigned int e = 1ul << (o % sizeof(unsigned long));
      std::size_t n = o / sizeof(unsigned long);
      tiles_[n] = tiles_[n] | e;

    }

    /// Set the local tiles as initialized
    void set_initialized() {
      // Todo: In the future we likely need to do something more scalable than an all reduce.
      WorldObject_::get_world().gop.reduce(tiles_, n_, detail::bit_or<unsigned long>());
      WorldObject_::process_pending();
      initialized_ = true;
    }

    /// Returns true if the tile data is stored locally.
    template<typename Index>
    bool is_local(const Index& i) const {
      // Todo: This will likely need to to be improved.
      return owner(ord(i)) == me_;
    }

  private:
    /// Handles tile probe messages.
    madness::Void fetch_handler(ProcessID requestor, ordinal_type i, const remote_ref& ref) const {
      WorldObject_::send(requestor, &SparseShape_::fetch_return, tile_includes(i).get(), ref);

      return madness::None;
    }

    madness::Void fetch_return(bool val, const remote_ref& ref) const {
      madness::FutureImpl<bool>* f = ref.get();
      f->set(val);

      ref.dec();

      return madness::None;
    }

    /// Returns the owner of a given index.
    ProcessID owner(ordinal_type i) const {
      return pmap_->owner(i);
    }

    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(ordinal_type i) const {
      TA_ASSERT(initialized_, std::runtime_error, "Shape is not full initialized.");

//      if(is_local(i))
      // Everything is local because it is broadcast. When data is ditributed
      // this will need to change.
        return madness::Future<bool>(tiles_[i / sizeof(unsigned long)] &
            (1ul << (i % sizeof(unsigned long))));

      madness::Future<bool> result;
      WorldObject_::send(owner(i), &SparseShape_::fetch_handler, me_, i,
          result.remote_ref(WorldObject_::get_world()));
      return result;
    }

    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(size_array i) const {
      return SparseShape_::tile_includes(ord(i));
    }

    /// Returns true if the local data has been fully initialized.
    virtual bool initialized() const { return initialized_; }

    std::size_t n_;
    unsigned long* tiles_; ///< tiling data.
    bool initialized_;
    hasher_type hasher_;
    boost::shared_ptr<madness::WorldDCPmapInterface<ordinal_type> > pmap_;
    ProcessID me_;
  }; // class SparseShape

// =============================================================================
// PredShape class

  template<typename R, typename P>
  class PredShape : public RangeShape<R> {
  protected:
    typedef RangeShape<R> RangeShape_;
    typedef typename RangeShape_::Shape_ Shape_;
    typedef typename RangeShape_::range_type range_type;
    typedef P pred_type;

  public:
    typedef typename Shape_::ordinal_type ordinal_type;
    typedef typename Shape_::size_array size_array;

    PredShape(const boost::shared_ptr<range_type>& r, pred_type p = pred_type()) :
        RangeShape_(r, detail::predicated_shape), predicate_(p)
    { }

  private:
    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(ordinal_type i) const {
      return madness::Future<bool>(predicate_(i));
    }

    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(size_array i) const {
      return madness::Future<bool>(predicate_(i));
    }

    /// Returns true if the local data has been fully initialized.
    virtual bool initialized() const { return true; }

    pred_type predicate_;
  }; // class PredShape

}  // namespace TiledArray

#endif // TILEDARRAY_SHAPE_H__INCLUDED
