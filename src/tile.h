#ifndef TILEDARRAY_TILE_H__INCLUDED
#define TILEDARRAY_TILE_H__INCLUDED

#include <dense_array.h>
#include <annotated_tile.h>
#include <tile_math.h>
#include <tile_slice.h>
//#include <iosfwd>
//#include <functional>

namespace TiledArray {

  // Forward declaration of TiledArray components.
  template <unsigned int DIM>
  class Permutation;
  template <unsigned int DIM>
  class LevelTag;

  template<typename T, unsigned int DIM, typename CS>
  class Tile;
  template<class T>
  class TileSlice;
  template<typename T, unsigned int DIM, typename CS>
  void swap(Tile<T,DIM,CS>&, Tile<T,DIM,CS>&);
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator ^(const Permutation<DIM>& p, const Tile<T,DIM,CS>& t);
  template<typename T, unsigned int DIM, typename CS>
  std::ostream& operator <<(std::ostream& out, const Tile<T,DIM,CS>& t);
  template<typename T, unsigned int DIM, typename CS, typename InIter, typename Generator>
  Tile<T, DIM, CS>& generate(Tile<T, DIM, CS>&, InIter, InIter, Generator);
  template<typename T, unsigned int DIM, typename CS, typename Generator>
  Tile<T, DIM, CS>& generate(Tile<T, DIM, CS>&, Generator);

  /// Tile is a dense, multi-dimensional array.

  /// Data in the tile is always stored in the order from least significant to
  /// most significant dimension.
  /// \arg \c T is the data type stored in the tile.
  /// \arg \c DIM is the number of dimensions of the tile.
  /// \arg \c CS is the coordinate system used by the tile.
  template<typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Tile
  {
    typedef DenseArray<T, DIM, LevelTag<0>, CS > data_container;
  public:
    typedef Tile<T, DIM, CS> Tile_;
    typedef T value_type;
    typedef T & reference_type;
    typedef const T & const_reference_type;
    typedef CS coordinate_system;
    typedef typename data_container::ordinal_type ordinal_type;
    typedef Range<ordinal_type, DIM, LevelTag<0>, coordinate_system > range_type;
    typedef typename range_type::index_type index_type;
    typedef typename range_type::size_array size_array;
    typedef typename range_type::volume_type volume_type;
    typedef typename range_type::const_iterator index_iterator;
    typedef typename data_container::const_iterator const_iterator;
    typedef typename data_container::iterator iterator;

    static unsigned int dim() { return DIM; }
    static detail::DimensionOrderType  order() { return coordinate_system::order; }

    /// Default constructor

    /// Constructs a tile with a size of zero in each dimension. The start and
    /// finish of the tile is also set to zero in each dimension. No memory is
    /// allocated or initialized.
    Tile() : range_(), data_() { }

    /// Copy constructor
    Tile(const Tile_& t) : range_(t.range_), data_(t.data_) { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Copy constructor
    Tile(Tile_&& t) : range_(std::move(t.range_)), data_(std::move(t.data_)) { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Construct a tile with a specific dimensions and initialize value.

    /// The tile will have the dimensions specified by \c range, and all elements
    /// will be initialized to \c val. If \c val is not specified, the data
    /// elements will be initialized using the default \c value_type constructor.
    /// \arg \c range A \c Range<> object that specifies the tile dimensions.
    /// \arg \c val Specifies the initial value of data elements (optional).
    Tile(const range_type& range, const value_type val = value_type()) :
        range_(range), data_(range.size(), val)
    { }

    /// Construct a tile with a specific dimensions and initialize values.

    /// The tile will have the dimensions specified by \c range, and elements are
    /// initialized with the data contained by [\c first, \c last ). If there
    /// are more elements in the tile than specified by the initializer list,
    /// then the remaining elements will be initialized with the default
    /// constructor. The initializer list must dereference to a type that is
    /// implicitly convertible to \c value_type.
    /// \arg \c range A \c Range<> object that specifies the tile dimensions.
    /// \arg \c first, \c last Input iterators, which point to a list of initial values.
    template <typename InIter>
    Tile(const range_type& range, InIter first, InIter last) :
    	  range_(range), data_(range.size(), first, last)
    {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
    }

    /// Creates a new tile from a TileSlice

    /// A deep copy of the slice data is done here.
    Tile(const TileSlice<Tile_>& s) :
        range_(s.range()), data_(s.size(), s.begin(), s.end())
    { }

    /// AnnotatedTile copy constructor

    /// The constructor will throw when the dimensions of the annotated tile do
    /// not match the dimensions of the tile.
    template<typename U>
    Tile(const expressions::tile::AnnotatedTile<U>& atile) :
        range_(make_size_(atile.size().begin(), atile.size().end())),
        data_(range_.size(), atile.begin(), atile.end())
    {
      TA_ASSERT((atile.dim() == DIM), std::runtime_error,
          "The dimensions of the annotated tile do not match the dimensions of the tile.");
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// AnnotatedTile assignment operator
    template<typename U>
    Tile(expressions::tile::AnnotatedTile<U>&& atile) :
        range_(make_size_(atile.size().begin(), atile.size().end())), data_()
    {
      TA_ASSERT((atile.dim() == DIM), std::runtime_error,
          "The dimensions of the annotated tile do not match the dimensions of the tile.");
      if(atile.owner_) {
        data_.move(range_.size(), atile.data());
        atile.owner_ = false;
      } else {
        data_container temp(range_.size(), atile.begin(), atile.end());
        data_.swap(temp);
      }
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    ~Tile() { }

    /// Assignment operator
    Tile_& operator =(const Tile_& other) {
      range_ = other.range_;
      data_ = other.data_;

      return *this;
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move assignment operator
    Tile_& operator =(Tile_&& other) {
      range_ = std::move(other.range_);
      data_ = std::move(other.data_);

      return *this;
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Returns a raw pointer to the element data.
    value_type* data() { return data_.data(); }
    /// Returns a constant raw pointer to the element data.
    const value_type* data() const { return data_.data(); }

    /// Returns an iterator to the first element of the tile.
    iterator begin() { return data_.begin(); } // no throw
    /// Returns a constant iterator to the first element of the tile.
    const_iterator begin() const { return data_.begin(); } // no throw
    /// Returns an iterator that points to the end of the tile.
    iterator end() { return data_.end(); } // no throw
    /// Returns a constant iterator that points to the end of the tile.
    const_iterator end() const { return data_.end(); } // no throw

    /// return a constant reference to the tile \c Range<> object.
    const range_type& range() const { return range_; } // no throw
    /// Returns the tile range start.
    const index_type& start() const { return range_.start(); } // no throw
    /// Returns the tile range finish.
    const index_type& finish() const { return range_.finish(); } // no throw
    /// Returns the tile range size.
    const size_array& size() const { return range_.size(); } // no throw
    /// Returns the number of elements in the volume.
    const volume_type volume() const { return data_.volume(); } // no throw
    /// Returns the dimension weights.
    const size_array& weight() const { return data_.weight(); } // no throw

    /// Returns true when \i is in the tile range.

    /// \arg \c i Element index.
    bool includes(const index_type& i) const { return range_.includes(i); }

    // The at() functions do error checking, but we do not need to implement it
    // here because the data container already does that. There is no need to do
    // it twice.
    /// Element access with range checking
    reference_type at(const ordinal_type& i) { return data_.at(i); }
    /// Element access with range checking
    const_reference_type at(const ordinal_type& i) const { return data_.at(i); }
    /// Element access with range checking
    reference_type at(const index_type& i) { return data_.at(i); }
    /// Element access with range checking
    const_reference_type at(const index_type& i) const { return data_.at(i); }

    /// Element access without error checking
    reference_type operator [](const ordinal_type& i) { return data_[i]; }
    /// Element access without error checking
    const_reference_type operator [](const ordinal_type& i) const { return data_[i]; }
    /// Element access without error checking
    reference_type operator [](const index_type& i) { return data_[i]; }
    /// Element access without error checking
    const_reference_type operator [](const index_type& i) const { return data_[i]; }

    /// Returns a slice of the tile given a sub-range.

    /// The range \c r must be completely contained by the tile.
    TileSlice<Tile_> slice(const range_type& r) {
      return TileSlice<Tile_>(*this, r); // Note: The range checks are done by the constructor.
    }

    /// The range \c r must be completely contained by the tile.
    TileSlice<const Tile_> slice(const range_type& r) const {
      return TileSlice<const Tile_>(*this, r); // Note: The range checks are done by the constructor.
    }

    /// Resize the tile. Any new elements added to the array will be initialized
    /// with val. If val is not specified, new elements will be initialized with
    /// the default constructor.
    void resize(const size_array& size, const value_type& val = value_type()) {
      range_.resize(size);
      data_.resize(size, val);
    }

    /// Move the origin of the tile to the given index.

    /// The overall size and data of the tile are unaffected by this operation.
    /// \arg \c origin is the new lower bound for the tile.
    void set_origin(const index_type& origin) {
      range_.resize(origin, origin + range_.size());
    }

    /// Permute the tile given a permutation.
    Tile_& operator ^=(const Permutation<DIM>& p) {
      Tile_ temp = p ^ (*this);
      swap(*this, temp);
      return *this;
    }


    expressions::tile::AnnotatedTile<value_type> operator ()(const std::string& v) {
      expressions::tile::AnnotatedTile<value_type> result(*this, expressions::VariableList(v));
      return result;
    }

    expressions::tile::AnnotatedTile<const value_type> operator ()(const std::string& v) const {
      return expressions::tile::AnnotatedTile<const value_type>(*this, expressions::VariableList(v));
    }

    /// Serializes the tile data for communication with other nodes.
    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & range_ & data_;
    }

  private:

    template<typename InIter>
    static size_array make_size_(InIter first, InIter last) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      size_array result;
      std::copy(first, last, result.begin());
      return result;
    }

    friend void swap<>(Tile_&, Tile_&);
    friend std::ostream& operator<< <>(std::ostream& , const Tile_&);
    friend Tile_ operator^ <>(const Permutation<DIM>&, const Tile_&);

    range_type range_;     ///< tile dimension information
    data_container data_;  ///< element data

  }; // class Tile

  /// Exchange the data of the two tiles
  template<typename T, unsigned int DIM, typename CS>
  void swap(Tile<T,DIM,CS>& t0, Tile<T,DIM,CS>& t1) {
    TiledArray::swap(t0.range_, t1.range_);
    TiledArray::swap(t0.data_, t1.data_);
  }

  /// Permute the tile given a permutation.
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator ^(const Permutation<DIM>& p, const Tile<T,DIM,CS>& t) {
    Tile<T,DIM,CS> result(t);
    result.range_ ^= p;
    result.data_ ^= p;

    return result;
  }

  /// Assigns a value to the specified range of element in tile with a generator function.

  /// This function assigns values to \c res for the given index
  /// list ( specified by \c InIter \c first and \c InIter \c last ) with
  /// \c Generator \c gen. /c first and /c last may dereference to
  /// \c Tile<T,DIM,CS>::index_type or \c Tile<T,DIM,CS>::ordinal_type. The
  /// function or functor object \c gen must accept a single argument of
  /// \c InIter::value_type and have a return type of \c Tile<T,DIM,CS>::value_type.
  /// Elements are assigned in the following manner:
  /// \code
  /// for(; first != last; ++first)
  ///   t[ *first ] = gen( *first );
  /// \endcode
  /// Note: Range<> may be used to define a specific of indexes and create the
  /// index iterators.
  /// \arg \c res The results of \c gen()  will be placed in this tile.
  /// \arg \c first, \c last input iterators, which point to a list of indexes.
  /// \arg \c gen Function or functor object used to generate the data.
  template<typename T, unsigned int DIM, typename CS, typename InIter, typename Generator>
  Tile<T, DIM, CS>& generate(Tile<T, DIM, CS>& res, InIter first, InIter last, Generator gen) {
    BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
    for(; first != last; ++first)
      res[ *first ] = gen( *first );

    return res;
  }

  /// Assigns a value to the specified range of element in tile with a generator function.

  /// This function assigns values to all elements of \c res with \c Generator
  /// \c gen.  The function or functor object \c gen must
  /// accept a single argument of \c Tile<T,DIM,CS>::::index_type and have a
  /// return type of \c Tile<T,DIM,CS>::value_type. Elements are assigned in
  /// the following manner:
  /// \code
  /// typename Tile<T, DIM, CS>::range_type::const_iterator r_it = res.range().begin();
  /// for(typename Tile<T, DIM, CS>::iterator it = res.begin(); it != res.end(); ++it, ++r_it)
  ///   *it = gen(*r_it);
  /// \endcode
  /// \arg \c res The results of \c gen()  will be placed in this tile.
  /// \arg \c gen Function or functor object used to generate the data.
  template<typename T, unsigned int DIM, typename CS, typename Generator>
  Tile<T, DIM, CS>& generate(Tile<T, DIM, CS>& res, Generator gen) {
    typename Tile<T, DIM, CS>::range_type::const_iterator r_it = res.range().begin();
    for(typename Tile<T, DIM, CS>::iterator it = res.begin(); it != res.end(); ++it, ++r_it)
      *it = gen(*r_it);

    return res;
  }

  /// ostream output orperator.
  template<typename T, unsigned int DIM, typename CS>
  std::ostream& operator <<(std::ostream& out, const Tile<T,DIM,CS>& t) {
    typedef Tile<T,DIM,CS> tile_type;
    const typename tile_type::size_array& weight = t.data_.weight();

    out << "{";
    typename CS::const_iterator d ;
    typename tile_type::ordinal_type i = 0;
    for(typename tile_type::const_iterator it = t.begin(); it != t.end(); ++it, ++i) {
      for(d =  CS::begin(), ++d; d != CS::end(); ++d) {
        if((i % weight[*d]) == 0)
          out << "{";
      }

      out << *it << " ";


      for(d = CS::begin(), ++d; d != CS::end(); ++d) {
        if(((i + 1) % weight[*d]) == 0)
          out << "}";
      }
    }
    out << "}";
    return out;
  }


} // namespace TiledArray

#endif // TILEDARRAY_TILE_H__INCLUDED
