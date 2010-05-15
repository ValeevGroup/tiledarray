#ifndef TILEDARRAY_SHAPE_H__INCLUDED
#define TILEDARRAY_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/utility.h>
#include <TiledArray/array_ref.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/coordinate_system.h>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/utility.hpp>
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
// RangeShape class

  /// Shape range handles the range based functionality.

  /// This class instantiates the virtual functions that use the range object.
  /// The remaining functions are implemented by derived classes. This class is
  /// for internal use only and should not be used as a base pointer.
  ///
  /// Template parameters:
  /// \var \c R is the range object type.
  template<typename I>
  class RangeInterface {
  public:
    typedef RangeInterface<I> RangeShape_;
    typedef I ordinal_type;
    typedef detail::ArrayRef<const I> size_array;
    typedef I volume_type;

    /// Returns the dimensions of the range
    unsigned int dim() { return this->range_dim(); }

    /// Returns the ordinal index i.
    template<typename Index>
    typename boost::enable_if<boost::is_integral<Index>, ordinal_type>::type
    ord(Index i) const { return i; }

    /// Calculates the ordinal index based on i.
    template<typename Index>
    typename boost::disable_if<boost::is_integral<Index>, ordinal_type>::type
    ord(const Index& i) const {
      TA_ASSERT(std::distance(i.begin(), i.end()) == dim(), std::runtime_error,
          "Array dimensions do not match range dimensions.");
      return detail::calc_ordinal(i.begin(), i.end(), this->range_weight().begin(),
          this->range_start().begin());
    }

    /// Forward the ordinal index
    template<typename Index>
    typename boost::enable_if<boost::is_integral<Index>, bool>::type
    includes(const Index i) const { return i < this->range_volume(); }

    /// Calculate the ordinal index
    template<typename Index>
    typename boost::disable_if<boost::is_integral<Index>, bool>::type
    includes(const Index& i) const {
      TA_ASSERT(std::distance(i.begin(), i.end()), std::runtime_error,
          "Index diemsions do not match range dimensions.");
      return (std::equal(this->range_start().begin(), this->range_start().end(), i.begin(), std::less_equal<I>()) &&
          std::equal(i.begin(), i.end(), this->range_finish().begin(), std::less<I>()));
    }

    unsigned int dim() const { return this->range_dim(); }

    volume_type volume() const { return this->range_volume(); }

    size_array start() const { return this->range_start(); }

    size_array finish() const { return this->range_finish(); }

    size_array size() const { return this->range_size(); }

    size_array weight() const { return this->range_weight(); }

  private:
    /// Returns the dimensions of the range.
    virtual unsigned int range_dim() const = 0;

    /// Returns the volume of the shape's range.
    virtual volume_type range_volume() const = 0;

    /// Returns the start index of the range.
    virtual size_array range_start() const = 0;

    /// Returns the finish index of the range.
    virtual size_array range_finish() const = 0;

    /// Returns the size of the range.
    virtual size_array range_size() const = 0;

    /// Returns the weight of the range.
    virtual size_array range_weight() const = 0;
  }; // class RangeShape

// =============================================================================
// Range Holder class

  /// Shape range handles the range based functionality.

  /// This class instantiates the virtual functions that use the range object.
  ///
  /// Template parameters:
  /// \var \c R is the range object type.
  template<typename R>
  class RangeHolder : public RangeInterface<typename R::ordinal_type> {
  private:
    typedef RangeHolder<R> RangeHolder_;
    typedef RangeInterface<typename R::ordinal_type> RangeInterface_;
  public:
    typedef R range_type;
    typedef typename RangeInterface_::ordinal_type ordinal_type;
    typedef typename RangeInterface_::size_array size_array;
    typedef typename RangeInterface_::volume_type volume_type;

  private:
    RangeHolder();

  public:
    /// Primary constructor

    /// Since all tiles are present in a dense array, the shape is considered
    /// Immediately available.
    RangeHolder(const range_type& r) :
        range_(r)
    { }

  private:
    /// Returns the dimensions of the range.
    virtual unsigned int range_dim() const { return range_type::dim; }

    /// Returns the volume of the shape's range.
    virtual volume_type range_volume() const { return range_.volume(); }

    /// Returns the start index of the range.
    virtual size_array range_start() const { return range_.start().data(); }

    /// Returns the finish index of the range.
    virtual size_array range_finish() const { return range_.finish().data(); }

    /// Returns the size of the range.
    virtual size_array range_size() const { return range_.size(); }

    /// Returns the weight of the range.
    virtual size_array range_weight() const { return range_.weight(); }

    const range_type& range_; ///< range object reference.
  }; // class RangeShape

// =============================================================================
// Range Holder class

  /// Dynamic range handles the range based functionality for ranges at runtime.

  /// This class stores range data based on runtime conditions. It implements a
  /// small subset of the range functionality.
  ///
  /// Template parameters:
  /// \var \c I is the range indexing.
  template<typename I>
  class DynamicRange : public RangeInterface<I> {
  private:
    typedef DynamicRange<I> DynamicRange_;
    typedef RangeInterface<I> RangeInterface_;
  public:
    typedef typename RangeInterface_::ordinal_type ordinal_type;
    typedef typename RangeInterface_::size_array size_array;
    typedef typename RangeInterface_::volume_type volume_type;

  private:
    DynamicRange();

  public:
    /// Constructor defined by an size array.
    template<typename Size>
    DynamicRange(const Size& size, detail::DimensionOrderType o) :
        dim_(std::distance(size.begin(), size.end())),
        start_(dim_, 0),
        finish_(size.begin(), size.end()),
        size_(size.begin(), size.end()),
        weight_(dim_),
        volume_(detail::volume(size_.begin(), size_.end()))
    {
      if(o == detail::decreasing_dimension_order)
        detail::calc_weight(size_.rbegin(), size_.rend(), weight_.rbegin());
      else
        detail::calc_weight(size_.begin(), size_.end(), weight_.begin());
    }

    /// Constructor defined by an upper and lower bound.

    /// All elements of finish must be greater than or equal to those of start.
    template<typename Index>
    DynamicRange(const Index& start, const Index& finish, detail::DimensionOrderType o) :
        dim_(std::distance(start.begin(), start.end())),
        start_(start.begin(), start.end()),
        finish_(finish.begin(), finish.end()),
        size_(dim_),
        weight_(dim_),
        volume_(0)
    {
      // Todo: Make this a little more optimal with transform iterators?
      TA_ASSERT( std::equal(start.begin(), start.end(), finish.begin(), std::less<I>()) ,
          std::runtime_error, "Finish is less than start.");
      std::transform(finish_.begin(), finish_.end(), start_.begin(), size_.begin(), std::minus<I>());
      if(o == detail::decreasing_dimension_order)
        detail::calc_weight(size_.rbegin(), size_.rend(), weight_.rbegin());
      else
        detail::calc_weight(size_.begin(), size_.end(), weight_.begin());
      volume_ = detail::volume(size_.begin(), size_.end());
    }

  private:
    /// Returns the dimensions of the range.
    virtual unsigned int range_dim() const { return dim_; }

    /// Returns the volume of the shape's range.
    virtual volume_type range_volume() const { return volume_; }

    /// Returns the start index of the range.
    virtual size_array range_start() const { return vector_to_array(start_); }

    /// Returns the finish index of the range.
    virtual size_array range_finish() const { return vector_to_array(finish_); }

    /// Returns the size of the range.
    virtual size_array range_size() const { return vector_to_array(size_); }

    /// Returns the weight of the range.
    virtual size_array range_weight() const { return vector_to_array(weight_); }

    /// Convert a vector to an ArrayRef.
    size_array vector_to_array(const std::vector<I>& v) const {
      return size_array(& v.front(), (& v.front()) + dim_);
    }

    unsigned int dim_;
    std::vector<I> start_;
    std::vector<I> finish_;
    std::vector<I> size_;
    std::vector<I> weight_;
    volume_type volume_;
  }; // class RangeShape

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
    template<typename R>
    Shape(detail::ShapeType t, const R& r) :
        range_(boost::dynamic_pointer_cast<RangeInterface<I> >(
            boost::make_shared<RangeHolder<R> >(r))),
        type_(t)
    { }

    /// Constructor defined by an size array.
    template<typename Size>
    Shape(detail::ShapeType t, const Size& size, detail::DimensionOrderType o) :
        range_(boost::dynamic_pointer_cast<RangeInterface<I> >(
            boost::make_shared<DynamicRange<I> >(size, o))),
        type_(t)
    { }

    /// Constructor defined by an upper and lower bound.

    /// All elements of finish must be greater than or equal to those of start.
    template<typename Index>
    Shape(detail::ShapeType t, const Index& start, const Index& finish, detail::DimensionOrderType o) :
        range_(boost::dynamic_pointer_cast<RangeInterface<I> >(
            boost::make_shared<DynamicRange<I> >(start, finish, o))),
        type_(t)
    { }

  public:
    /// Set the range to a new range object.
    template<typename R>
    void set_range(const R& r) {
      range_ = boost::dynamic_pointer_cast<RangeInterface<I> >(
          boost::make_shared<RangeHolder<R> >(r));
    }

    /// Returns true if the shape is locally initialized.
    bool is_initialized() const {
      return this->initialized();
    }

    /// Returns true if the element is present.
    template<typename Index>
    madness::Future<bool> includes(const Index& i) const {
      if(! range_->includes(i))
        return madness::Future<bool>(false);

      return this->tile_includes(forward_index(i));
    }

    /// Returns the shape type (dense_shape, sparse_shape, or predicated_shape).
    detail::ShapeType type() const { return type_; }

    /// Returns the volume of the shape's range.
    volume_type volume() const { return range_->volume(); }

    /// Returns the start index of the range.
    size_array start() const { return range_->start(); }

    /// Returns the finish index of the range.
    size_array finish() const { return range_->finish(); }

    /// Returns the size of the range.
    size_array size() const { return range_->size(); }

    /// Returns the weight of the range.
    size_array weight() const { return range_->weight(); }

  protected:

    /// Returns true if the range includes the ordinal index i.
    bool range_includes(const ordinal_type& i) const { return range_->includes(i); }

    /// Returns true if the range includes the coordinate index i.
    bool range_includes(const size_array& i) const { return range_->includes(i); }

    /// Returns an ordinal index given the integral index i.
    template<typename Index>
    static typename boost::enable_if<boost::is_integral<Index>, ordinal_type>::type
    forward_index(Index i) { return i; }

    /// Returns a size_array given the array i.
    template<typename Index>
    static typename boost::disable_if<boost::is_integral<Index>, size_array>::type
    forward_index(const Index& i) {
      return size_array( &(* i.begin()), &(* i.end()));
    }

    /// Returns the ordinal index of a coordinate index.
    template<typename Index>
    ordinal_type ord(const Index& i) const { return range_->ord(i); }

  private:
    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(ordinal_type) const = 0;

    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(size_array) const = 0;

    /// Returns true if the local data has been fully initialized.
    virtual bool initialized() const = 0;

    boost::shared_ptr<RangeInterface<I> > range_;
    detail::ShapeType type_; ///< Shape type (dense, sparse, or predicated).
  }; // class shape

// =============================================================================
// DenseShape class

  /// Dense shape used to construct Array objects.

  /// DenseShape is used to represent dense arrays. It is initialized with a
  /// madness world object and a range object. It includes all tiles included by
  /// the range object.
  ///
  /// Template parameters:
  /// \var \c R is the range object type.
  template<typename I>
  class DenseShape : public Shape<I> {
  protected:
    typedef DenseShape<I> DenseShape_;
    typedef Shape<I> Shape_;

  public:
    typedef typename Shape_::ordinal_type ordinal_type;
    typedef typename Shape_::size_array size_array;

    /// Primary constructor

    /// Since all tiles are present in a dense array, the shape is considered
    /// Immediately available.
    template<typename R>
    DenseShape(const R& r) :
        Shape_(detail::dense_shape, r)
    { }

    /// Constructor defined by an size array.
    template<typename Size>
    DenseShape(const Size& size, detail::DimensionOrderType o) :
        Shape_(detail::dense_shape, size, o)
    { }

    /// Constructor defined by an upper and lower bound.

    /// All elements of finish must be greater than or equal to those of start.
    template<typename Index>
    DenseShape(const Index& start, const Index& finish, detail::DimensionOrderType o) :
        Shape_(detail::dense_shape, start, finish, o)
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

  template<typename I, typename H = madness::Hash_private::defhashT<I> >
  class SparseShape : public Shape<I>, public madness::WorldObject<SparseShape<I, H> > {
  protected:
    typedef SparseShape<I, H> SparseShape_;
    typedef madness::WorldObject<SparseShape_ > WorldObject_;
    typedef Shape<I> Shape_;
    typedef H hasher_type;

    typedef madness::RemoteReference< madness::FutureImpl<bool> > remote_ref;

    // Note: Shape has private default constructor, copy constructor, and
    // assignment operator. These operations are not allowed here.

  public:
    typedef typename Shape_::ordinal_type ordinal_type;
    typedef typename Shape_::volume_type volume_type;
    typedef typename Shape_::size_array size_array;
    typedef madness::WorldDCPmapInterface<ordinal_type> pmap_type;

    /// Primary constructor

    /// This is a world object so all processes must construct SparseShape
    /// on all processes in the same order relative to other world objects.
    /// It is your responsibility to ensure that the range object is identical
    /// on all processes, otherwise process mapping will not be correct.
    template<typename R>
    SparseShape(madness::World& w, const R& r,
      const hasher_type h = hasher_type()) :
        Shape_(detail::sparse_shape, r),
        WorldObject_(w),
        n_(r.volume() / sizeof(unsigned long) + (r.volume() % sizeof(unsigned long) == 0 ? 0 : 1)),
        tiles_(new unsigned long[n_]),
        initialized_(false),
        hasher_(h),
        pmap_(dynamic_cast<pmap_type*>(
            new madness::WorldDCDefaultPmap<ordinal_type, hasher_type>(w, hasher_)))
    {
      TA_ASSERT(pmap_.get() != NULL, std::runtime_error, "Pmap dynamic cast failed.");
      std::fill(tiles_, tiles_ + n_, 0ul);
    }

    /// Primary constructor
    template<typename R>
    SparseShape(madness::World& w, const R& r,
      const boost::shared_ptr<pmap_type>& pm) :
        Shape_(detail::sparse_shape, r),
        WorldObject_(w),
        n_(r.volume() / sizeof(unsigned long) + (r.volume() % sizeof(unsigned long) == 0 ? 0 : 1)),
        tiles_(new unsigned long[n_]),
        initialized_(false),
        hasher_(),
        pmap_(pm)
    {
      std::fill(tiles_, tiles_ + n_, 0ul);
    }

    /// Constructor defined by an size array.
    template<typename Size>
    SparseShape(madness::World& w, const Size& size, detail::DimensionOrderType o,
      const hasher_type h = hasher_type()) :
        Shape_(detail::sparse_shape, size, o),
        WorldObject_(w),
        n_(Shape_::volume() / sizeof(unsigned long) + (Shape_::volume() % sizeof(unsigned long) == 0 ? 0 : 1)),
        tiles_(new unsigned long[n_]),
        initialized_(false),
        hasher_(h),
        pmap_(dynamic_cast<pmap_type*>(
            new madness::WorldDCDefaultPmap<ordinal_type, hasher_type>(w, hasher_)))

    { }

    /// Constructor defined by an upper and lower bound.

    /// All elements of finish must be greater than or equal to those of start.
    template<typename Index>
    SparseShape(madness::World& w, const Index& start, const Index& finish, detail::DimensionOrderType o,
      const hasher_type h = hasher_type()) :
        Shape_(detail::sparse_shape, start, finish, o),
        WorldObject_(w),
        n_(Shape_::volume() / sizeof(unsigned long) + (Shape_::volume() % sizeof(unsigned long) == 0 ? 0 : 1)),
        tiles_(new unsigned long[n_]),
        initialized_(false),
        hasher_(h),
        pmap_(dynamic_cast<pmap_type*>(
            new madness::WorldDCDefaultPmap<ordinal_type, hasher_type>(w, hasher_)))
    { }

    /// Constructor defined by an size array.
    template<typename Size>
    SparseShape(madness::World& w, const Size& size, detail::DimensionOrderType o,
      const boost::shared_ptr<pmap_type>& pm) :
        Shape_(detail::sparse_shape, size, o),
        WorldObject_(w),
        n_(Shape_::volume() / sizeof(unsigned long) + (Shape_::volume() % sizeof(unsigned long) == 0 ? 0 : 1)),
        tiles_(new unsigned long[n_]),
        initialized_(false),
        hasher_(),
        pmap_(pm)
    { }

    /// Constructor defined by an upper and lower bound.

    /// All elements of finish must be greater than or equal to those of start.
    template<typename Index>
    SparseShape(madness::World& w, const Index& start, const Index& finish, detail::DimensionOrderType o,
      const boost::shared_ptr<pmap_type>& pm) :
        Shape_(detail::sparse_shape, start, finish, o),
        WorldObject_(w),
        n_(Shape_::volume() / sizeof(unsigned long) + (Shape_::volume() % sizeof(unsigned long) == 0 ? 0 : 1)),
        tiles_(new unsigned long[n_]),
        initialized_(false),
        hasher_(),
        pmap_(pm)
    { }

    ~SparseShape() {
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
      return owner(ord(i)) == WorldObject_::get_world().rank();
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
      WorldObject_::send(owner(i), &SparseShape_::fetch_handler,
          WorldObject_::get_world().rank(), i,
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
    boost::shared_ptr<pmap_type> pmap_;
  }; // class SparseShape

// =============================================================================
// PredShape class

  template<typename I, typename P>
  class PredShape : public Shape<I> {
  protected:
    typedef PredShape<I, P> PredShape_;
    typedef Shape<I> Shape_;
    typedef P pred_type;

  public:
    typedef typename Shape_::ordinal_type ordinal_type;
    typedef typename Shape_::size_array size_array;

    template<typename R>
    PredShape(const R& r, pred_type p = pred_type()) :
        Shape_(detail::predicated_shape, r), predicate_(p)
    { }

    /// Constructor defined by an size array.
    template<typename Size>
    PredShape(const Size& size, detail::DimensionOrderType o, pred_type p = pred_type()) :
        Shape_(detail::predicated_shape, size, o), predicate_(p)
    { }

    /// Constructor defined by an upper and lower bound.

    /// All elements of finish must be greater than or equal to those of start.
    template<typename Index>
    PredShape(const Index& start, const Index& finish, detail::DimensionOrderType o, pred_type p = pred_type()) :
        Shape_(detail::predicated_shape, start, finish, o), predicate_(p)
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
