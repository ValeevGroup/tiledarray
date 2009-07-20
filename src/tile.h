#ifndef TILEDARRAY_TILE_H__INCLUDED
#define TILEDARRAY_TILE_H__INCLUDED

#include <array_storage.h>
#include <iterator.h>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>
#include <iosfwd>
#include <functional>
#include <cstddef>

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#include <tr1/tuple>
#else
#include <tuple>
#endif

extern "C" {
#include <cblas.h>
};

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
  Tile<T,DIM,CS> operator ^(const Permutation<DIM>& p, const Tile<T,DIM,CS>& t);
  template<typename T, unsigned int DIM, typename CS>
  std::ostream& operator <<(std::ostream& out, const Tile<T,DIM,CS>& t);

  /// Tile is a dense, multi-dimensional array.

  /// Data in the tile is always stored in the order from least significant to
  /// most significant dimension.
  /// \arg \c T is the data type stored in the tile.
  /// \arg \c DIM is the number of dimensions of the tile.
  /// \arg \c CS is the coordinate system used by the tile.
  template<typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Tile
  {
    typedef DenseArrayStorage<T, DIM, LevelTag<0>, CS > data_container;
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

    static const unsigned int dim() { return DIM; }

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
    { }

    /// Creates a new tile from a TileSlice

    /// A deep copy of the slice data is done here.
    Tile(const TileSlice<Tile_>& s) :
        range_(s.range()), data_(s.size(), s.begin(), s.end())
    { }

    ~Tile() { }

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
    const range_type& range() const { return range_; }
    /// Returns the tile range start.
    const index_type& start() const { return range_.start(); }
    /// Returns the tile range finish.
    const index_type& finish() const { return range_.finish(); }
    /// Returns the tile range size.
    const size_array& size() const { return range_.size(); }
    /// Returns the number of elements in the volume.
    const volume_type volume() const { return range_.volume(); }
    /// Returns the dimension weights.
    const size_array weight() const { return data_.weight(); }

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
    reference_type at(const index_type& i){ return data_.at(i); }
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
      swap(temp);
      return *this;
    }

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

    /// Exchange calling tile's data with that of \c other.
    void swap(Tile_& other) {
      range_.swap(other.range_);
      data_.swap(other.data_);
    }

    /// Serializes the tile data for communication with other nodes.
    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & range_ & data_;
    }

  private:

    range_type range_;     ///< tile dimension information
    data_container data_;  ///< element data

    friend std::ostream& operator<< <>(std::ostream& , const Tile_&);
    friend Tile_ operator^ <>(const Permutation<DIM>&, const Tile_&);

  }; // class Tile

  namespace detail {
    template <typename T, typename V>
    struct mirror_const {
      typedef T type;
      typedef V value;
      typedef V& reference;
      typedef V* pointer;
    };

    template <typename T, typename V>
    struct mirror_const<const T, V> {
      typedef const T type;
      typedef const V value;
      typedef const V& reference;
      typedef const V* pointer;
    };

  } // namespace detail
  /// Sub-range of a tile

  /// \c TileSlice represents an arbitrary sub-range of a tile. \c TileSlice
  /// does not contain any element data. The primary use of \c TileSlice is to
  /// provide the ability to iterate over a sub section of the referenced tile.
  /// All element access is done via index translation between the slice and the
  /// tile.
  /// Note: Ordinal indexes for the slice are not equivalent to the tile ordinal
  /// indexes.
  /// Note: The memory of the slice may or may not be contiguous, depending on
  /// the slice selected.
  template<class T>
  class TileSlice
  {
  public:
    typedef TileSlice<T> TileSlice_;
    typedef typename boost::remove_const<T>::type tile_type;
    typedef typename tile_type::value_type value_type;
    typedef typename detail::mirror_const<T, value_type>::reference reference_type;
    typedef const value_type& const_reference_type;
    typedef typename tile_type::coordinate_system coordinate_system;
    typedef typename tile_type::range_type range_type;
    typedef typename tile_type::index_type index_type;
    typedef typename tile_type::size_array size_array;
    typedef typename tile_type::volume_type volume_type;
    typedef typename tile_type::index_iterator index_iterator;
    typedef detail::ElementIterator<value_type, typename range_type::const_iterator, TileSlice_> iterator;
    typedef detail::ElementIterator<const value_type, typename range_type::const_iterator, TileSlice_> const_iterator;

    static const unsigned int dim() { return tile_type::dim(); }

    /// Slice constructor.

    /// Constructs a slice of tile \c t given a sub range. The range \c r must
    /// be completely contained by the tile range. You may easily construct a
    /// range that is contained by the original tile range by using the &
    /// operator on the tile range and an arbitrary range (i.e. slice_range =
    /// t.range() & other_range;).
    /// \arg \c t is the tile which the slice will reference.
    /// \arg \c r is the range which defines the slice.
    ///
    /// Warning: Iteration and element access for a slice are more expensive
    /// operations than the equivalent tile operations. If you need to iterate
    /// over a slice in a time critical loop, you may want to copy the slice
    /// into a new tile object.
    TileSlice(T& t, range_type& r) : r_(r),
        w_(detail::calc_weight<coordinate_system>(r.size())), t_(t)
    {
      TA_ASSERT( ( valid_range_(r, t) ) ,
          std::runtime_error("TileSlice<...>::TileSlice(...): Range slice is not contained by the range of the original tile."));
    }

    /// Copy constructor
    TileSlice(TileSlice_& other) : r_(other.r_), w_(other.w_), t_(other.t_) { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move constructor
    TileSlice(TileSlice&& other) : r_(std::move(other.r_)), w_(std::move(other.w_)), t_(other.t_) { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    ~TileSlice() { }

    /// Assignment operator
    TileSlice_& operator =(const TileSlice_& other) {
      r_ = other.r_;
      w_ = other.w_;
      t_ = other.t_;

      return *this;
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move assignment operator
    TileSlice_& operator =(TileSlice_&& other) {
      r_ = std::move(other.r_);
      w_ = std::move(other.w_);
      t_ = other.t_;

      return *this;
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Returns an iterator to the first element of the tile.
    iterator begin() { return iterator(r_.begin(), this); } // no throw

    /// Returns an iterator that points to the end of the tile.
    iterator end() { return iterator(r_.end(), this); } // no throw

    /// Returns a const_iterator to the first element of the tile.
    const_iterator begin() const { return const_iterator(r_.begin(), const_cast<TileSlice_*>(this)); } // no throw

    /// Returns a const_iterator that points to the end of the tile.
    const_iterator end() const { return const_iterator(r_.end(), const_cast<TileSlice_*>(this)); } // no throw

    /// return a constant reference to the tile \c Range<> object.
    const range_type& range() const { return r_; }
    /// Returns the tile range start.
    const index_type& start() const { return r_.start(); }
    /// Returns the tile range finish.
    const index_type& finish() const { return r_.finish(); }
    /// Returns the tile range size.
    const size_array& size() const { return r_.size(); }
    /// Returns the number of elements in the volume.
    const volume_type volume() const { return r_.volume(); }
    /// Returns the dimension weights.
    const size_array weight() const { return w_; }
    /// Returns true when \i is in the tile range.

    /// Returns true if index \c i is included in the tile.

    /// \arg \c i Element index.
    bool includes(const index_type& i) const { return r_.includes(i); }
    /// Returns true if index \c i is included in the tile.

    // The at() functions do error checking, but we do not need to implement it
    // here because the data container already does that. There is no need to do
    // it twice.
    /// Element access with range checking
    reference_type at(const index_type& i) { return t_.at(i); }

    /// Element access with range checking
    const_reference_type at(const index_type& i) const { return t_.at(i); }

    /// Element access without error checking
    reference_type operator [](const index_type& i) { return t_[i]; }

    /// Element access without error checking
    const_reference_type operator [](const index_type& i) const { return t_[i]; }

    /// Exchange calling tile's data with that of \c other.
    void swap(TileSlice_& other) {
      r_.swap(other.r_);
      std::swap(t_, other.t_);
    }

  private:

    TileSlice(); ///< No default construction allowed.

    bool valid_range_(const range_type& r, const tile_type& t) const {
      if(detail::greater_eq(r.start().data(), t.start().data()) &&
          detail::less(r.start().data(), t.finish().data()) &&
          detail::greater(r.finish().data(), t.start().data()) &&
          detail::less_eq(r.finish().data(), t.finish().data()))
        return true;

      return false;
    }

    range_type r_;  ///< tile slice dimension information
    size_array w_;  ///< slice dimension weight
    T& t_;          ///< element data

  }; // class TileSlice

  namespace detail {

    template<typename Arg, typename Op>
    struct Binary2Unary : std::unary_function<boost::tuple<Arg,Arg>, Arg> {
      typedef typename std::unary_function<boost::tuple<Arg,Arg>, Arg>::argument_type argument_type;
      typedef typename std::unary_function<boost::tuple<Arg,Arg>, Arg>::result_type result_type;
      Binary2Unary() : op_(Op()) { }
      Binary2Unary(const Op& op) : op_(op) { }
      result_type operator ()(const argument_type& a) const {
        return op_( boost::get<0>(a) , boost::get<1>(a) );
      }
    private:
      Op op_;
    };

    /// Tile-Tile operation
    template<typename T, unsigned int DIM, typename CS, typename Op>
    struct TileOp : std::binary_function<Tile<T,DIM,CS>, Tile<T,DIM,CS>, Tile<T,DIM,CS> > {
      typedef typename std::binary_function<Tile<T,DIM,CS>, Tile<T,DIM,CS>, Tile<T,DIM,CS> >::result_type result_type;
      typedef typename std::binary_function<Tile<T,DIM,CS>, Tile<T,DIM,CS>, Tile<T,DIM,CS> >::first_argument_type first_argument_type;
      typedef typename std::binary_function<Tile<T,DIM,CS>, Tile<T,DIM,CS>, Tile<T,DIM,CS> >::second_argument_type second_argument_type;

      TileOp() : op_(Op()) { }
      TileOp(Op op) : op_(op) { }
      Tile<T,DIM,CS> operator ()(const Tile<T,DIM,CS>& t1, const Tile<T,DIM,CS>& t2) const {

        return result_type(t1.range(),
            boost::make_transform_iterator(boost::make_zip_iterator(boost::make_tuple(t1.begin(), t2.begin())), op_),
            boost::make_transform_iterator(boost::make_zip_iterator(boost::make_tuple(t1.end(), t2.end())), op_));

      }
    private:
      Op op_;
    };

    /// Tile-Scalar operation
    template<typename S, typename T, unsigned int DIM, typename CS, typename Op>
    struct TileScalarOp : std::binary_function<Tile<T,DIM,CS>, S, Tile<T,DIM,CS> > {
      typedef typename std::binary_function<Tile<T,DIM,CS>, S, Tile<T,DIM,CS> >::result_type result_type;
      typedef typename std::binary_function<Tile<T,DIM,CS>, S, Tile<T,DIM,CS> >::first_argument_type first_argument_type;
      typedef typename std::binary_function<Tile<T,DIM,CS>, S, Tile<T,DIM,CS> >::second_argument_type second_argument_type;

      TileScalarOp() : op_(Op()) { }
      TileScalarOp(const Op& op) : op_(op) { }
      result_type operator ()(const first_argument_type& t, const second_argument_type& s) const {
        return result_type(t.range(),
            boost::make_transform_iterator(t.begin(), std::bind2nd<Op,T>(op_, s)),
            boost::make_transform_iterator(t.end(), std::bind2nd<Op,T>(op_, s)));
      }
    private:
      Op op_;
    };
  }

  /// Permute the tile given a permutation.
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator ^(const Permutation<DIM>& p, const Tile<T,DIM,CS>& t) {
    Tile<T,DIM,CS> result(t);
    result.range_ ^= p;
    result.data_ ^= p;

    return result;
  }

  /// Sum two tiles
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator +(const Tile<T,DIM,CS>& t1, const Tile<T,DIM,CS>& t2) {
    TA_ASSERT( t1.size() == t2.size() ,
        std::runtime_error("operator+(const Tile<T,DIM,CS>&, const Tile<T,DIM,CS>&): Tile dimensions do not match.") );
    detail::TileOp<T,DIM,CS,detail::Binary2Unary<T, std::plus<T> > > p;
    return p(t1,t2);
  }

  /// Sum two tiles (double specialization with blas)
  template<unsigned int DIM, typename CS>
  Tile<double,DIM,CS> operator +(const Tile<double,DIM,CS>& t1, const Tile<double,DIM,CS>& t2) {
    TA_ASSERT( t1.size() == t2.size() ,
        std::runtime_error("operator+(const Tile<T,DIM,CS>&, const Tile<T,DIM,CS>&): Tile dimensions do not match.") );
    Tile<double,DIM,CS> result(t1);
    cblas_daxpy(result.volume(), 1.0, t2.begin(), 1, result.begin(), 1);
    return result;
  }

  /// Subtract two tiles
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator -(const Tile<T,DIM,CS>& t1, const Tile<T,DIM,CS>& t2) {
    TA_ASSERT( t1.size() == t2.size() ,
        std::runtime_error("operator+(const Tile<T,DIM,CS>&, const Tile<T,DIM,CS>&): Tile dimensions do not match.") );
    detail::TileOp<T,DIM,CS,detail::Binary2Unary<T, std::minus<T> > > p;
    return p(t1,t2);
  }

  /// Subtract two tiles (double specialization with blas)
  template<unsigned int DIM, typename CS>
  Tile<double,DIM,CS> operator -(const Tile<double,DIM,CS>& t1, const Tile<double,DIM,CS>& t2) {
    TA_ASSERT( t1.size() == t2.size() ,
        std::runtime_error("operator+(const Tile<T,DIM,CS>&, const Tile<T,DIM,CS>&): Tile dimensions do not match.") );
    Tile<double,DIM,CS> result(t1);
    cblas_daxpy(result.volume(), -1.0, t2.begin(), 1, result.begin(), 1);
    return result;
  }

  /// In-place tile addition
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS>& operator +=(Tile<T,DIM,CS>& tr, const Tile<T,DIM,CS>& ta) {
    TA_ASSERT( (tr.size() == ta.size()) ,
        std::runtime_error("operator+=(Tile<T,DIM,CS>&, const Tile<T,DIM,CS>&): Tile dimensions do not match.") );
    typename Tile<T,DIM,CS>::const_iterator a_it = ta.begin();
    for(typename Tile<T,DIM,CS>::iterator r_it = tr.begin(); r_it != tr.end(); ++r_it, ++a_it)
      *r_it += *a_it;

    return tr;
  }

  /// In-place tile addition (double specialization with blas)
  template<unsigned int DIM, typename CS>
  Tile<double,DIM,CS>& operator +=(Tile<double,DIM,CS>& tr, const Tile<double,DIM,CS>& ta) {
    TA_ASSERT( (tr.size() == ta.size()) ,
        std::runtime_error("operator+=(Tile<T,DIM,CS>&, const Tile<T,DIM,CS>&): Tile dimensions do not match.") );
    cblas_daxpy(tr.volume(), 1.0, ta.begin(), 1, tr.begin(), 1);

    return tr;
  }

  /// In-place tile-scalar addition
  template<typename S, typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS>& operator +=(Tile<T,DIM,CS>& tr, const S& s) {
    for(typename Tile<T,DIM,CS>::iterator r_it = tr.begin(); r_it != tr.end(); ++r_it)
      *r_it += s;

    return tr;
  }

  /// In-place tile subtraction
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS>& operator -=(Tile<T,DIM,CS>& tr, const Tile<T,DIM,CS>& ta) {
    TA_ASSERT( (tr.size() == ta.size()) ,
        std::runtime_error("operator-=(Tile<T,DIM,CS>&, const Tile<T,DIM,CS>&): Tile dimensions do not match.") );
    typename Tile<T,DIM,CS>::const_iterator a_it = ta.begin();
    for(typename Tile<T,DIM,CS>::iterator r_it = tr.begin(); r_it != tr.end(); ++r_it, ++a_it)
      *r_it -= *a_it;

    return tr;
  }

  /// In-place tile subtraction (double specialization with blas)
  template<unsigned int DIM, typename CS>
  Tile<double,DIM,CS>& operator -=(Tile<double,DIM,CS>& tr, const Tile<double,DIM,CS>& ta) {
    TA_ASSERT( (tr.size() == ta.size()) ,
        std::runtime_error("operator-=(Tile<double,DIM,CS>&, const Tile<double,DIM,CS>&): Tile dimensions do not match.") );
    cblas_daxpy(tr.volume(), -1.0, ta.begin(), 1, tr.begin(), 1);

    return tr;
  }

  /// In-place tile-scalar subtraction
  template<typename S, typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS>& operator -=(Tile<T,DIM,CS>& tr, const S& s) {
    for(typename Tile<T,DIM,CS>::iterator r_it = tr.begin(); r_it != tr.end(); ++r_it)
      *r_it -= s;

    return tr;
  }

  /// In-place tile-scalar multiplication
  template<typename S, typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS>& operator *=(Tile<T,DIM,CS>& tr, const S& s) {
    for(typename Tile<T,DIM,CS>::iterator r_it = tr.begin(); r_it != tr.end(); ++r_it)
      *r_it *= s;

    return tr;
  }

  /// In-place tile-scalar multiplication (double specialization with blas)
  template<typename S, unsigned int DIM, typename CS>
  Tile<double,DIM,CS>& operator *=(Tile<double,DIM,CS>& tr, const S& s) {
    cblas_dscal(tr.volume(), s, tr.begin(), 1);

    return tr;
  }

  /// add a scaler to each tile element.
  template<typename T, unsigned int DIM, typename CS, typename S>
  Tile<T,DIM,CS> operator +(const Tile<T,DIM,CS>& t, S s) {
    detail::TileScalarOp<S,T,DIM,CS,std::plus<T> > op;
    return op(t,s);
  }

  /// add a scaler to each tile element.
  template<typename T, unsigned int DIM, typename CS, typename S>
  Tile<T,DIM,CS> operator -(const Tile<T,DIM,CS>& t, S s) {
    detail::TileScalarOp<S,T,DIM,CS,std::minus<T> > op;
    return op(t,s);
  }

  /// add a scaler to each tile element.
  template<typename T, unsigned int DIM, typename CS, typename S>
  Tile<T,DIM,CS> operator *(const Tile<T,DIM,CS>& t, S s) {
    detail::TileScalarOp<S,T,DIM,CS,std::multiplies<T> > op;
    return op(t,s);
  }

  /// add a scaler to each tile element.
  template<typename T, unsigned int DIM, typename CS, typename S>
  Tile<T,DIM,CS> operator *(S s, const Tile<T,DIM,CS>& t) {
    detail::TileScalarOp<S,T,DIM,CS,std::multiplies<T> > op;
    return op(t,s);
  }

  /// add a scaler to each tile element.
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator -(const Tile<T,DIM,CS>& t) {
    detail::TileScalarOp<T,T,DIM,CS,std::multiplies<T> > op;
    return op(t,-1);
  }

  namespace detail {
/*
    template<char V>
    class IndexVar {
      const static char value;
    };

    template<char V>
    const char IndexVar<V>::value = V;

    template<class... V>
    class VarList {
    public:
      typedef std::tuple<V...> var_list;
      V var;

      template<unsigned int NN>
      char get() {
        return next::get<NN>();
      }

      template<>
      char get<N>() {
        return var;
      }
    };

    template<class V>
    class VarList {

    };
*/
    /// Contraction of two rank 3 tensors.
    /// r[a,b,c,d] = t0[a,i,c] * t1[b,i,d]
    template<typename T, detail::DimensionOrderType Order>
    void contract_aic_x_bid(const Tile<T,3,CoordinateSystem<3,Order> >& t0, const Tile<T,3,CoordinateSystem<3,Order> >& t1, Tile<T,4,CoordinateSystem<4,Order> >& tr) {
      typedef Tile<T,3,CoordinateSystem<3,Order> > Tile3;
      typedef Tile<T,4,CoordinateSystem<4,Order> > Tile4;
      typedef Eigen::Matrix< T , Eigen::Dynamic , Eigen::Dynamic, (Order == decreasing_dimension_order ? Eigen::RowMajor : Eigen::ColMajor) | Eigen::AutoAlign > matrix_type;
      TA_ASSERT(t0.size()[1] == t1.size()[1],
          std::runtime_error("void contract(const contraction_pair<T,3>& t0, const contraction_pair<T,3>& t1, contraction_pair<T,4>& tr): t0[1] != t1[1]."));

      const unsigned int i0 = Tile3::coordinate_system::ordering().order2dim(0);
      const unsigned int i1 = Tile3::coordinate_system::ordering().order2dim(1);
      const unsigned int i2 = Tile3::coordinate_system::ordering().order2dim(2);

      typename Tile4::size_array s;
      typename Tile4::coordinate_system::const_reverse_iterator it = Tile4::coordinate_system::rbegin();
      s[*it++] = t0.size()[i2];
      s[*it++] = t1.size()[i2];
      s[*it++] = t0.size()[i0];
      s[*it] = t1.size()[i0];

      if(tr.size() != s)
        tr.resize(s);

      const typename Tile3::ordinal_type step0 = t0.weight()[i2];
      const typename Tile3::ordinal_type step1 = t1.weight()[i2];
      const typename Tile4::ordinal_type stepr = t0.size()[i0] * t1.size()[i0];
      const typename Tile3::value_type* p0_begin = NULL;
      const typename Tile3::value_type* p0_end = t0.data() + step0 * t0.size()[i2];
      const typename Tile3::value_type* p1_begin = NULL;
      const typename Tile3::value_type* p1_end = t1.data() + step1 * t1.size()[i2];
      typename Tile4::value_type* pr = tr.data();

      for(p0_begin = t0.data(); p0_begin != p0_end; p0_begin += step0) {
        Eigen::Map<matrix_type> m0(p0_begin, t0.size()[i1], t0.size()[i0]);
        for(p1_begin = t1.begin(); p1_begin != p1_end; p1_begin += step1, pr += stepr) {
          Eigen::Map<matrix_type> mr(pr, t0.size()[i0], t1.size()[i0]);
          Eigen::Map<matrix_type> m1(p1_begin, t0.size()[i1], t1.size()[i0]);

          mr = m0.transpose() * m1;
        }
      }
    }

    /// Contraction of a rank 3 and rank 2 tensor.
    /// r[a,b,c] = t0[a,i,b] * t1[i,c]
    template<typename T, detail::DimensionOrderType Order>
    void contract_aib_x_ic(const Tile<T,3,CoordinateSystem<3,Order> >& t0, const Tile<T,3,CoordinateSystem<3,Order> >& t1, Tile<T,3,CoordinateSystem<4,Order> >& tr) {
      typedef Tile<T,3,CoordinateSystem<3,Order> > Tile3;
      typedef Tile<T,4,CoordinateSystem<4,Order> > Tile4;
      typedef Eigen::Matrix< T , Eigen::Dynamic , Eigen::Dynamic, (Order == decreasing_dimension_order ? Eigen::RowMajor : Eigen::ColMajor) | Eigen::AutoAlign > matrix_type;
      TA_ASSERT(t0.size()[1] == t1.size()[1],
          std::runtime_error("void contract(const contraction_pair<T,3>& t0, const contraction_pair<T,3>& t1, contraction_pair<T,4>& tr): t0[1] != t1[1]."));

      const unsigned int i0 = Tile3::coordinate_system::ordering().order2dim(0);
      const unsigned int i1 = Tile3::coordinate_system::ordering().order2dim(1);
      const unsigned int i2 = Tile3::coordinate_system::ordering().order2dim(2);

      typename Tile4::size_array s;
      typename Tile4::coordinate_system::const_reverse_iterator it = Tile4::coordinate_system::rbegin();
      s[*it++] = t0.size()[i2];
      s[*it++] = t1.size()[i2];
      s[*it++] = t0.size()[i0];
      s[*it] = t1.size()[i0];

      if(tr.size() != s)
        tr.resize(s);

      const typename Tile3::ordinal_type step0 = t0.weight()[i2];
      const typename Tile3::ordinal_type step1 = t1.weight()[i2];
      const typename Tile4::ordinal_type stepr = t0.size()[i0] * t1.size()[i0];
      const typename Tile3::value_type* p0_begin = NULL;
      const typename Tile3::value_type* p0_end = t0.data() + step0 * t0.size()[i2];
      const typename Tile3::value_type* p1_begin = NULL;
      const typename Tile3::value_type* p1_end = t1.data() + step1 * t1.size()[i2];
      typename Tile4::value_type* pr = tr.data();

      for(p0_begin = t0.data(); p0_begin != p0_end; p0_begin += step0) {
        Eigen::Map<matrix_type> m0(p0_begin, t0.size()[i1], t0.size()[i0]);
        for(p1_begin = t1.begin(); p1_begin != p1_end; p1_begin += step1, pr += stepr) {
          Eigen::Map<matrix_type> mr(pr, t0.size()[i0], t1.size()[i0]);
          Eigen::Map<matrix_type> m1(p1_begin, t0.size()[i1], t1.size()[i0]);

          mr = m0.transpose() * m1;
        }
      }
    }

    /// Contraction of two 3D arrays.
    /// r = t0[i] * t1[i]
    template<typename T, typename CS>
    void contract(const Tile<T,1,CS>& t0, const Tile<T,1,CS>& t1, T& tr) {
      TA_ASSERT(t0.volume() == t1.volume(),
          std::runtime_error("void contract(const contraction_pair<T,1>& t0, const contraction_pair<T,1>& t1, T& tr): t0[0] != t1[0]."));

      tr = 0;
      for(std::size_t i = 0; i < t0.volume(); ++i)
        tr += t0[i] * t1[i];
    }

  } // namespace detail

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

} // namespace TiledArray

#endif // TILEDARRAY_TILE_H__INCLUDED
