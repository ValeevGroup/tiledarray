#ifndef TILEDARRAY_SHAPE_H__INCLUDED
#define TILEDARRAY_SHAPE_H__INCLUDED
/*
#include <TiledArray/error.h>
#include <TiledArray/utility.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/coordinate_system.h>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/utility.hpp>
#include <algorithm>
#include <numeric>
#include <functional>
#include <map>

namespace TiledArray {

  template <typename, unsigned int, typename>
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

    detail::DimensionOrderType order() const { return this->range_order(); }

    unsigned int dim() const { return this->range_dim(); }

    volume_type volume() const { return this->range_volume(); }

    size_array start() const { return this->range_start(); }

    size_array finish() const { return this->range_finish(); }

    size_array size() const { return this->range_size(); }

    size_array weight() const { return this->range_weight(); }

  private:
    /// Returns the dimensions of the range.
    virtual unsigned int range_dim() const = 0;

    virtual detail::DimensionOrderType range_order() const = 0;

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

    /// Returns the dimension ordering
    virtual detail::DimensionOrderType range_order() const {
      return range_type::coordinate_system::dimension_order;
    }

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
        order_(o),
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
        order_(o),
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

    /// Returns the dimension ordering.
    virtual detail::DimensionOrderType range_order() const { return order_; }

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
    detail::DimensionOrderType order_;
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

    virtual ~Shape() { }

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

    /// Returns the range dimension
    unsigned int dim() const { return range_->dim(); }

    /// Returns the dimension order
    detail::DimensionOrderType order() const { return range_->order(); }

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

} // namespace TiledArray
*/
#endif // TILEDARRAY_SHAPE_H__INCLUDED
