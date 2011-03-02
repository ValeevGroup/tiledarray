#ifndef TILEDARRAY_SHAPE_H__INCLUDED
#define TILEDARRAY_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/range.h>
#include <TiledArray/madness_runtime.h>
#include <world/sharedptr.h>
#include <world/worlddc.h>
#include <typeinfo>

namespace TiledArray {

  template <typename, unsigned int, typename>
  class ArrayCoordinate;

  /// Shape of array tiles

  /// Stores information about the presence or absence of tiles in an array,
  /// and handles interprocess communication when the information is not locally
  /// available.
  /// \tparam CS The \c Shape coordinate system type
  /// \tparam Key The key type used by the process map [ Default =
  /// \c Array::key_type ].
  /// \note This is an interface class only and cannot be constructed directly.
  /// Insted use DenseShape, SparseShape, or PredShape.
  template <typename CS, typename Key>
  class Shape {
  private:
    typedef Shape<CS, Key> Shape_;  ///< This type
    typedef Key key_type;           ///< The pmap key type
    typedef madness::WorldDCPmapInterface< key_type > pmap_interface_type;
                                    ///< The process map interface type

    // assignment not allowed
    Shape_& operator=(const Shape_&);

  public:
    typedef CS coordinate_system;
    typedef typename coordinate_system::index index;
    typedef typename coordinate_system::ordinal_index ordinal_index;
    typedef Range<coordinate_system> range_type; ///< Range object type

  protected:
    /// Shape constructor

    /// Shape base class constructor
    Shape(const range_type& r) : range_(&r) { }

    /// Copy constructor

    /// \param other The shape to be copied
    Shape(const Shape_& other) : range_(other.range_) { }

    /// Range accessor

    /// \return A const reference to the range associated with this shape.
    const range_type& range() const { return *range_; }

  public:

    /// Create a copy of this object

    /// \return A shared pointer to a copy of this object.
    virtual std::shared_ptr<Shape_> clone() const = 0;

    /// Virtual destructor
    virtual ~Shape() { }

    /// Set a new range for the shape
    Shape_ set(const range_type& r) {
      range_ = &r;
      return *this;
    }

    /// Checks that the index, \c i , is stored locally

    /// \tparam Index Ordinal or coordinate index type
    /// \param i The index to test
    /// \return If \c i is included in the shape range and this rank owns \c i
    /// (according to the shape process map), then true. Otherwise false.
    template <typename Index>
    bool is_local(const Index& i) const {
      return range_->includes(i) && this->local(ord_(i));
    }

    template <typename Index>
    madness::Future<bool> includes(const Index& i) const {
      if(! range_->includes(i)) // Check that it is in range
        madness::Future<bool>(false);

      return this->probe(ord_(i));
    }

    /// Check that \c i is locally available and included

    /// \tparam Index The index type (index or ordinal_index).
    /// \param i The index to be checked
    template <typename Index>
    bool is_local_and_includes(const Index& i) const {
      if(is_local(i))
        return this->probe(ord_(i)).get();

      return false;
    }

    /// Type info accessor for derived class
    virtual const std::type_info& type() const = 0;

  protected:

    /// Calculate the ordinal index

    /// \param i The ordinal index to convert
    /// \return The ordinal index of \c i
    /// \note This function is a pass through function. It is only here to make
    /// life easier with templates.
    /// \note No range checking is done in this function.
    ordinal_index ord_(const ordinal_index& i) const { return i; }

    /// Calculate the ordinal index

    /// \param i The coordinate index to convert
    /// \return The ordinal index of \c i
    /// \note No range checking is done in this function.
    ordinal_index ord_(const index& i) const {
      return coordinate_system::calc_ordinal(i, range_->weight(), range_->start());
    }

  private:

    // The virtual function interface

    /// Check that a tiles information is stored locally.

    /// \param i The ordinal index to check.
    virtual bool local(ordinal_index i) const = 0;

    /// Probe for the presence of a tile in the shape

    /// \param i The index to be probed.
    virtual madness::Future<bool> probe(ordinal_index i) const = 0;

    // private data

    const range_type* range_; ///< The range object associated with this shape.
  };

} // namespace TiledArray

#endif // TILEDARRAY_SHAPE_H__INCLUDED
