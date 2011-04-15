#ifndef TILEDARRAY_SHAPE_H__INCLUDED
#define TILEDARRAY_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/range.h>
#include <TiledArray/versioned_pmap.h>
#include <world/sharedptr.h>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <typeinfo>

namespace TiledArray {

  // Forward declarations
  template <typename>
  class DenseShape;
  template <typename>
  class SparseShape;
  template <typename, typename>
  class PredShape;
  template <typename>
  class Shape;

  namespace detail {

    template <typename CS>
    class LocalIndexIterator : public boost::iterator_facade<
        LocalIndexIterator<CS>, std::size_t, std::input_iterator_tag >
    {
    private:
      typedef boost::iterator_facade<LocalIndexIterator<CS>, std::size_t,
          std::input_iterator_tag > iterator_facade_; ///< Base class

    public:

      // Standard iterator typedefs
      typedef typename iterator_facade_::value_type value_type; ///< Iterator value type
      typedef typename iterator_facade_::reference reference; ///< Iterator reference type
      typedef typename iterator_facade_::pointer pointer; ///< Iterator pointer type
      typedef typename iterator_facade_::iterator_category iterator_category; /// Iterator category tag
      typedef typename iterator_facade_::difference_type difference_type; ///< Iterator difference type

      /// Copy constructor

      /// \param other The other iterator to be copied
      LocalIndexIterator(const LocalIndexIterator<CS>& other) :
          shape_(other.shape_), current_(other.current_)
      { }

      /// Construct an index iterator

      /// \param v The initial value of the iterator index
      /// \param c The container that the iterator will reference
      LocalIndexIterator(std::size_t v, const Shape<CS>* c) :
          shape_(c), current_(v)
      { }

      /// Copy constructor

      /// \param other The other iterator to be copied
      /// \return A reference to this object
      LocalIndexIterator<CS>& operator=(const LocalIndexIterator<CS>& other) {
        current_ = other.current_;
        shape_ = other.shape_;

        return *this;
      }

    private:

      /// Compare this iterator with \c other for equality

      /// \param other The other iterator to be checked for equality
      /// \return \c true when the value of the two iterators are the same and
      /// they point to the same container, otherwise \c false
      bool equal(const LocalIndexIterator<CS>& other) const {
        return current_ == other.current_ && shape_ == other.shape_;
      }

      /// Increment the current value

      /// This calls \c Container::increment(Value&) and passes the current
      /// value as the function argument.
      void increment() {
        while(current_ < shape_->volume()) {
          if(shape_->is_local(current_))
            if(shape_->probe(current_))
              break;
        }
      }

      /// Dereference the iterator

      /// \return A const reference to the current value
      reference dereference() const {
        return current_;
      }

      // boost::iterator_core_access requires access to private members for
      // boost::iterator_facade to function correctly.
      friend class boost::iterator_core_access;

      const Shape<CS>* shape_;  ///< The shape that the iterator references
      std::size_t current_;     ///< The current value of the iterator
    }; // class RangeIterator

  }  // namespace detail


  /// Shape of array tiles

  /// Stores information about the presence or absence of tiles in an array,
  /// and handles interprocess communication when the information is not locally
  /// available.
  /// \tparam CS The \c Shape coordinate system type
  /// \note This is an interface class only and cannot be constructed directly.
  /// Insted use DenseShape, SparseShape, or PredShape.
  template <typename CS>
  class Shape {
  private:
    typedef Shape<CS> Shape_;  ///< This type

    // assignment not allowed
    Shape();
    Shape_& operator=(const Shape_&);

  public:
    typedef CS coordinate_system;
    typedef typename coordinate_system::key_type key_type;           ///< The pmap key type
    typedef typename coordinate_system::index index;
    typedef typename coordinate_system::ordinal_index ordinal_index;
    typedef Range<coordinate_system> range_type;  ///< Range object type
    typedef detail::VersionedPmap<key_type> pmap_type; ///< The process map interface type
    typedef detail::LocalIndexIterator<coordinate_system> const_iterator;

  protected:
    /// Shape constructor

    /// Shape base class constructor
    /// \param r The range for the shape
    /// \param m The process map for the shape
    Shape(const range_type& r, const pmap_type& m) : range_(r), pmap_(m) { }

    /// Copy constructor

    /// \param other The shape to be copied
    Shape(const Shape_& other) : range_(other.range_), pmap_(other.pmap_) { }

  public:

    /// Virtual destructor
    virtual ~Shape() { }

    /// Create a copy of this object

    /// \return A shared pointer to a copy of this object.
    virtual std::shared_ptr<Shape_> clone() const = 0;

    /// Type info accessor for derived class
    virtual const std::type_info& type() const = 0;

    const_iterator begin() const { return const_iterator(this, 0); }

    const_iterator end() const { return const_iterator(this, volume()); }

    /// Is shape data for key \c k stored locally.

    /// \param i The key to check
    /// \return \c true when shape data for the given tile is stored on this node,
    /// otherwise \c false.
    template <typename Index>
    bool is_local(const Index& i) const {
      TA_ASSERT(range_.includes(i), std::out_of_range,
          "Cannot check for tiles that are not in the range.");
      return this->local_data(key(i));
    }

    /// Probe for the presence of an element at key
    template <typename Index>
    bool probe(const Index& i) const {
      TA_ASSERT(this->is_local(i), std::runtime_error,
          "You cannot probe data that is not stored locally.");
      return  this->local_probe(key(i));
    }

  protected:

    ordinal_index ord(ordinal_index o) const { return o; }

    ordinal_index ord(const index& i) const {
      return coordinate_system::ord(i, range_.weight(), range_.start());
    }

    ordinal_index ord(const key_type& k) const {
      if(k.keys() & 1)
        return k.key1();
      else
        return coordinate_system::calc_ordinal(k.key2(), range_.weight(), range_.start());
    }

    template <typename Index>
    key_type key(const Index& i) const { return coordinate_system::key(i, range_.weight(), range_.start()); }

    typename range_type::volume_type volume() { return range_.volume(); }

    template <typename Index>
    ProcessID owner(const Index& i) const {
      return pmap_.owner(coordinate_system::key(i));
    }

  private:

    virtual bool local_data(const key_type&) const { return true; }
    virtual bool local_probe(const key_type&) const { return true; }

    const range_type& range_; ///< The range object associated with this shape.
    const pmap_type& pmap_;   ///< The process map for the shape.
  };

  /// Runtime type checking for dense shape

  /// \tparam T Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c DenseShape class return \c true , otherwise \c false .
  template <typename T>
  inline bool is_dense_shape(const std::shared_ptr<T>&) {
    return false;
  }

  /// Runtime type checking for dense shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c DenseShape class return \c true , otherwise \c false .
  template <typename CS>
  inline bool is_dense_shape(const std::shared_ptr<Shape<CS> >& s) {
    return s->type() == typeid(DenseShape<CS>);
  }

  /// Runtime type checking for dense shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c DenseShape class return \c true , otherwise \c false .
  template <typename CS>
  inline bool is_dense_shape(const std::shared_ptr<DenseShape<CS> >&) {
    return true;
  }

  /// Runtime type checking for sparse shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c SparseShape class return \c true , otherwise \c false .
  template <typename T>
  inline bool is_sparse_shape(const std::shared_ptr<T>&) {
    return false;
  }

  /// Runtime type checking for sparse shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c SparseShape class return \c true , otherwise \c false .
  template <typename CS>
  inline bool is_sparse_shape(const std::shared_ptr<Shape<CS> >& s) {
    return s->type() == typeid(SparseShape<CS>);
  }

  /// Runtime type checking for sparse shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c SparseShape class return \c true , otherwise \c false .
  template <typename CS>
  inline bool is_sparse_shape(const std::shared_ptr<SparseShape<CS> >&) {
    return true;
  }

  /// Runtime type checking for predicated shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c PredShape class return \c true , otherwise \c false .
  template <typename T>
  inline bool is_pred_shape(const std::shared_ptr<T>& s) {
    return false;
  }

  /// Runtime type checking for predicated shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c PredShape class return \c true , otherwise \c false .
  template <typename CS>
  inline bool is_pred_shape(const std::shared_ptr<Shape<CS> >& s) {
    return (! is_dense_shape(s)) && (! is_sparse_shape(s));
  }

  /// Runtime type checking for predicated shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c PredShape class return \c true , otherwise \c false .
  template <typename CS, typename Pred>
  inline bool is_pred_shape(const std::shared_ptr<PredShape<CS, Pred> >& s) {
    return (! is_dense_shape(s)) && (! is_sparse_shape(s));
  }

} // namespace TiledArray

#endif // TILEDARRAY_SHAPE_H__INCLUDED
