#ifndef TILEDARRAY_TILE_H__INCLUDED
#define TILEDARRAY_TILE_H__INCLUDED

#include <TiledArray/range.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/tensor_base.h>
#include <world/archive.h>
#include <TiledArray/dense_storage.h>
#include <algorithm>
#include <functional>
#include <iosfwd>

namespace TiledArray {
  namespace expressions {

  // Forward declarations
    template <typename, typename, typename>
    class Tile;

    template <typename T, typename CS, typename A>
    struct TensorTraits<Tile<T,CS,A> > {
      typedef DenseStorage<T,A> storage_type;
      typedef typename CS::size_array size_array;
      typedef typename CS::volume_type size_type;
      typedef typename storage_type::value_type value_type;
      typedef typename storage_type::reference reference;
      typedef typename storage_type::const_reference const_reference;
      typedef typename storage_type::iterator iterator;
      typedef typename storage_type::const_iterator const_iterator;
      typedef typename storage_type::pointer pointer;
      typedef typename storage_type::const_pointer const_pointer;
    }; // struct TensorTraits<Tile<T,CS,A> >

    template <typename T, typename CS, typename A>
    struct Eval<Tile<T,CS,A> > {
      typedef const Tile<T,CS,A>& type;
    }; // struct TensorTraits<Tile<T,CS,A> >


    /// Tile is an N-dimensional, dense array.

    /// \tparam T Tile element type.
    /// \tparam CS A \c CoordinateSystem type
    /// \tparam A A C++ standard library compliant allocator (Default:
    /// \c Eigen::aligned_allocator<T>)
    template <typename T, typename CS, typename A = Eigen::aligned_allocator<T> >
    class Tile : public expressions::DirectWritableTensor<Tile<T,CS,A> > {
    public:
      typedef DenseStorage<T,A> storage_type;
      typedef Tile<T,CS,A> Tile_;                             ///< This object's type

      typedef CS coordinate_system;                           ///< The array coordinate system

      typedef typename CS::volume_type volume_type;           ///< Array volume type
      typedef typename CS::index index;                       ///< Array coordinate index type
      typedef typename CS::ordinal_index ordinal_index;       ///< Array ordinal index type
      typedef typename CS::size_array size_array;             ///< Size array type

      typedef typename storage_type::allocator_type allocator_type;   ///< Allocator type
      typedef typename storage_type::size_type size_type;             ///< Size type
      typedef typename storage_type::difference_type difference_type; ///< difference type
      typedef typename storage_type::value_type value_type;           ///< Array element type
      typedef typename storage_type::reference reference;             ///< Element reference type
      typedef typename storage_type::const_reference const_reference; ///< Element reference type
      typedef typename storage_type::pointer pointer;                 ///< Element pointer type
      typedef typename storage_type::const_pointer const_pointer;     ///< Element const pointer type
      typedef typename storage_type::iterator iterator;               ///< Element iterator type
      typedef typename storage_type::const_iterator const_iterator;   ///< Element const iterator type

      typedef Range<coordinate_system> range_type;            ///< Tile range type

      static unsigned int dim() { return coordinate_system::dim; }
      static TiledArray::detail::DimensionOrderType order() { return coordinate_system::order; }

      /// Default constructor

      /// Constructs a tile with zero size.
      /// \note You must call resize() before attempting to access any elements.
      Tile() :
          range_(), data_()
      { }

      /// Copy constructor

      /// \param other The tile to be copied.
      Tile(const Tile_& other) :
          range_(other.range_), data_(other.data_)
      { }

      /// Copy constructor

      /// \param other The tile to be copied.
      template <typename Derived>
      Tile(const TensorBase<Derived>& other) :
          range_(index(0), index(other.size().begin())), data_()
      {
        other.eval_to(data_);
      }

      /// Copy constructor

      /// \param other The tile to be copied.
      Tile(const range_type& r, const DenseStorage<T,A>& other) :
          range_(r), data_(other)
      { }

      /// Constructs a new tile

      /// The tile will have the dimensions specified by the range object \c r and
      /// the elements of the new tile will be equal to \c v. The provided
      /// allocator \c a will allocate space for only for the tile data.
      /// \param r A shared pointer to the range object that will define the tile
      /// dimensions
      /// \param val The fill value for the new tile elements ( default: value_type() )
      /// \param a The allocator object for the tile data ( default: alloc_type() )
      /// \throw std::bad_alloc There is not enough memory available for the target tile
      /// \throw anything Any exception that can be thrown by \c T type default or
      /// copy constructors
      Tile(const range_type& r, const value_type& val = value_type(), const allocator_type& a = allocator_type()) :
          range_(r), data_(r.volume(), val, a)
      { }


      /// Constructs a new tile

      /// The tile will have the dimensions specified by the range object \c r and
      /// the elements of the new tile will be equal to \c v. The provided
      /// allocator \c a will allocate space for only for the tile data.
      /// \tparam InIter An input iterator type.
      /// \param r A shared pointer to the range object that will define the tile
      /// dimensions
      /// \param first An input iterator to the beginning of the data to copy.
      /// \param last An input iterator to one past the end of the data to copy.
      /// \param a The allocator object for the tile data ( default: alloc_type() )
      /// \throw std::bad_alloc There is not enough memory available for the
      /// target tile
      /// \throw anything Any exceptions that can be thrown by \c T type default
      /// or copy constructors
      template <typename InIter>
      Tile(const range_type& r, InIter first, const allocator_type& a = allocator_type()) :
          range_(r), data_(r.volume(), first, a)
      { }

      /// destructor
      ~Tile() { }

      template <typename Dest>
      void eval_to(Dest& dest) const {
        TA_ASSERT(volume() == dest.volume());
        std::copy(begin(), end(), dest.begin());
      }

      const Tile_& eval() const { return *this; }

      /// Assignment operator

      /// \tparam U The proxy tile element type
      /// \tparam B The proxy tile allocator type
      /// \param other The tile object to be moved
      /// \return A reference to this object
      /// \throw std::bad_alloc There is not enough memory available for the target tile
      template <typename Arg>
      typename madness::disable_if<std::is_same<Tile_, Arg>, Tile_&>::type
      operator=(const Arg& other) {
        other.eval_to(data_);
        return *this;
      }

      /// Assignment operator

      /// \param other The tile object to be moved
      /// \return A reference to this object
      /// \throw std::bad_alloc There is not enough memory available for the target tile
      Tile_& operator=(const Tile_& other) {
        range_ = other.range_;
        data_ = other.data_;
        return *this;
      }

      /// In-place permutation of tile elements.

      /// \param p A permutation object.
      /// \return A reference to this object
      /// \warning This function modifies the shared range object.
      Tile_& operator ^=(const Permutation<coordinate_system::dim>& p) {
        return (*this = p ^ *this);
      }

      template <typename Arg>
      Tile_& operator+=(const Arg& other) {
        TA_ASSERT(volume() == other.volume());
        data_ += other;
        return *this;
      }

      Tile_& operator+=(const value_type& value) {
        data_ += value;
        return *this;
      }

      template <typename Arg>
      Tile_& operator-=(const Arg& other) {
        TA_ASSERT(volume() == other.volume());
        data_ -= other;

        return *this;
      }

      Tile_& operator-=(const value_type& value) {
        data_ -= value;
        return *this;
      }

      template <typename Value>
      Tile_& operator*=(const Value& value) {
        data_ *= value;
        return *this;
      }

      /// Resize the array to the specified dimensions.

      /// \param r The range object that specifies the new size.
      /// \param val The value that will fill any new elements in the array
      /// ( default: value_type() ).
      /// \return A reference to this object.
      /// \note The current data common to both arrays is maintained.
      /// \note This function cannot change the number of tile dimensions.
      Tile_& resize(const range_type& r, value_type val = value_type()) {
        Tile_ temp(r, val);
        if(data_.data()) {
          // replace Range with ArrayDim?
          range_type range_common = r & (range_);

          for(typename range_type::const_iterator it = range_common.begin(); it != range_common.end(); ++it)
            temp[*it] = data_[range_.ord(*it)]; // copy common data.
        }
        swap(temp);
        return *this;
      }

      pointer data() { return data_.data(); }
      const_pointer data() const { return data_.data(); }

      iterator begin() { return data_.begin(); }
      const_iterator begin() const { return data_.begin(); }
      iterator end() { return data_.end(); }
      const_iterator end() const { return data_.end(); }

      /// Returns a reference to element i (range checking is performed).

      /// This function provides element access to the element located at index i.
      /// If i is not included in the range of elements, std::out_of_range will be
      /// thrown. Valid types for Index are ordinal_type and index_type.
      template <typename Index>
      reference at(const Index& i) { return data_.at(range_.ord(i)); }

      /// Returns a constant reference to element i (range checking is performed).

      /// This function provides element access to the element located at index i.
      /// If i is not included in the range of elements, std::out_of_range will be
      /// thrown. Valid types for Index are ordinal_type and index_type.
      template <typename Index>
      const_reference at(const Index& i) const { return data_.at(range_.ord(i)); }

      /// Returns a reference to the element at i.

      /// This No error checking is performed.
      template <typename Index>
      reference operator[](const Index& i) { return data_[range_.ord(i)]; }

      /// Returns a constant reference to element i. No error checking is performed.
      template <typename Index>
      const_reference operator[](const Index& i) const { return data_[range_.ord(i)]; }

      /// Tile range accessor

      /// \return A const reference to the tile range object.
      /// \throw nothing
      const range_type& range() const { return range_; }

      const size_array& size() const { return range_.size(); }

      size_type volume() const { return range_.volume(); }

      /// Exchange the content of this object with other.

      /// \param other The other Tile to swap with this object
      /// \throw nothing
      void swap(Tile_& other) {
        data_.swap(other.data_);
        std::swap(range_, other.range_);
      }

    protected:

      template <typename Archive>
      void load(const Archive& ar) {
        ar & range_;
        data_.load(ar);
      }

      template <typename Archive>
      void store(const Archive& ar) const {
        ar & range_;
        data_.store(ar);
      }

    private:

      template <class, class>
      friend struct madness::archive::ArchiveStoreImpl;
      template <class, class>
      friend struct madness::archive::ArchiveLoadImpl;

      range_type range_;  ///< Range data for this tile
      storage_type data_; ///< Store tile element data
    }; // class Tile




    /// Swap the data of the two arrays.

    /// \tparam T Tile element type
    /// \tparam CS Tile coordinate system type
    /// \tparam A Tile allocator
    /// \param first The first tile to swap
    /// \param second The second tile to swap
    template <typename T, typename CS, typename A>
    void swap(Tile<T, CS, A>& first, Tile<T, CS, A>& second) { // no throw
      first.swap(second);
    }

    /// ostream output operator.

    /// \tparam T Tile element type
    /// \tparam CS Tile coordinate system type
    /// \tparam A Tile allocator
    /// \param out The output stream that will hold the tile output.
    /// \param t The tile to be place in the output stream.
    /// \return The modified \c out .
    template <typename T, typename CS, typename A>
    std::ostream& operator <<(std::ostream& out, const Tile<T, CS, A>& t) {
      typedef Tile<T, CS, A> tile_type;
      typedef typename TiledArray::detail::CoordIterator<const typename tile_type::size_array,
          tile_type::coordinate_system::order>::iterator weight_iterator;

      typename tile_type::ordinal_index i = 0;
      weight_iterator weight_begin_1 = tile_type::coordinate_system::begin(t.range().weight()) + 1;
      weight_iterator weight_end = tile_type::coordinate_system::end(t.range().weight());
      weight_iterator weight_it;

      out << "{";
      for(typename tile_type::const_iterator it = t.begin(); it != t.end(); ++it, ++i) {
        for(weight_it = weight_begin_1; weight_it != weight_end; ++weight_it) {
          if((i % *weight_it) == 0)
            out << "{";
        }

        out << *it << " ";

        for(weight_it = weight_begin_1; weight_it != weight_end; ++weight_it) {
          if(((i + 1) % *weight_it) == 0)
            out << "}";
        }
      }
      out << "}";
      return out;
    }

  } // namespace expressions
} // namespace TiledArray

namespace madness {
  namespace archive {

    template <class Archive, class T>
    struct ArchiveStoreImpl;
    template <class Archive, class T>
    struct ArchiveLoadImpl;

    template <class Archive, typename T, typename CS, typename A>
    struct ArchiveStoreImpl<Archive, TiledArray::expressions::Tile<T, CS, A> > {
      static void store(const Archive& ar, const TiledArray::expressions::Tile<T, CS, A>& t) {
        t.store(ar);
      }
    };

    template <class Archive, typename T, typename CS, typename A>
    struct ArchiveLoadImpl<Archive, TiledArray::expressions::Tile<T, CS, A> > {
      typedef TiledArray::expressions::Tile<T, CS, A> tile_type;

      static void load(const Archive& ar, tile_type& t) {
        t.load(ar);
      }
    };
  }
}
#endif // TILEDARRAY_TILE_H__INCLUDED
