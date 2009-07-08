#ifndef TILEDARRAY_TILE_H__INCLUDED
#define TILEDARRAY_TILE_H__INCLUDED

#include <array_storage.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>
//#include <world/archive.h>
#include <iosfwd>
#include <functional>
#include <cstddef>
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

    /// Construct a tile with a specific size and initialize value.

    /// The tile will have the size specified by \c size and the tile lower bound
    /// will be set to \c origin. All elements will be initialized to \c val.
    /// If \c val is not specified, the data elements will be initialized using
    /// the default \c value_type constructor.
    /// \arg \c size Specifies the tile dimension sizes.
    /// \arg \c origin Specifies the start of the tile (optional).
    /// \arg \c val Specifies the initial value of data elements (optional).
    Tile(const size_array& size, const index_type& origin = index_type(), const value_type val = value_type()) :
        range_(size, origin), data_(range_.size(), val)
    { }

    /// Construct a tile with a specific size and initialize values.

    /// The tile will have the size specified by \c size and the tile lower bound
    /// will be set to \c origin. Elements are initialized with the data
    /// contained by [\c first, \c last ). If there are more elements in the
    /// tile than specified by the initializer list, then the remaining elements
    /// will be initialized with the default constructor. The initializer list
    /// must dereference to a type that is implicitly convertible to
    /// \c value_type.
    /// \arg \c size Specifies the tile dimension sizes.
    /// \arg \c origin Specifies the start of the tile.
    /// \arg \c first, \c last Input iterators, which point to a list of initial values.
    template <typename InIter>
    Tile(const size_array& size, const index_type& origin, InIter first, InIter last) :
        range_(size, origin), data_(range_.size(), first, last)
    { }

    /// Construct a tile with a specific start and finish, and initialize value.

    /// The tile will have the dimensions given by \c start and \c finish. All
    /// elements will be initialized to \c val. If \c val is not specified, the
    /// data elements will be initialized using the default \c value_type
    /// constructor.
    /// \arg \c start Specifies the lower bound of the tile.
    /// \arg \c finish Specifies the upper bound of the tile.
    /// \arg \c val Specifies the initial value of data elements (optional).
    Tile(const index_type& start, const index_type& finish, const value_type val = value_type()) :
        range_(start, finish), data_(range_.size(), val)
    { }

    /// Construct a tile with a specific start and finish, and initialize values.

    /// The tile will have the dimensions given by \c start and \c finish.
    /// Elements are initialized with the data contained by [\c first, \c last ).
    /// If there are more elements in the tile than specified by the initializer
    /// list, then the remaining elements will be initialized with the default
    /// constructor. The initializer list must dereference to a type that is
    /// implicitly convertible to \c value_type.
    /// \arg \c start Specifies the lower bound of the tile.
    /// \arg \c finish Specifies the upper bound of the tile.
    /// \arg \c first, \c last Input iterators, which point to a list of initial values.
    template <typename InIter>
    Tile(const index_type& start, const index_type& finish, InIter first, InIter last) :
        range_(start, finish), data_(range_.size(), first, last)
    { }

    ~Tile() { }

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

    /// Serializes the tile data for communication with other nodes.
    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & range_ & data_;
    }

    /// Exchange calling tile's data with that of \c other.
    void swap(Tile_& other) {
      range_.swap(other.range_);
      data_.swap(other.data_);
    }

  private:

    range_type range_;     ///< tile dimension information
    data_container data_;  ///< element data

    friend std::ostream& operator<< <>(std::ostream& , const Tile_&);
    friend Tile_ operator^ <>(const Permutation<DIM>&, const Tile_&);

  }; // class Tile

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

  /// Subtract two tiles
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator -(const Tile<T,DIM,CS>& t1, const Tile<T,DIM,CS>& t2) {
    TA_ASSERT( t1.size() == t2.size() ,
        std::runtime_error("operator+(const Tile<T,DIM,CS>&, const Tile<T,DIM,CS>&): Tile dimensions do not match.") );
    detail::TileOp<T,DIM,CS,detail::Binary2Unary<T, std::minus<T> > > p;
    return p(t1,t2);
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

  /// In-place tile-scalar multiplication
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

  /// ostream output orperator.
  template<typename T, unsigned int DIM, typename CS>
  std::ostream& operator <<(std::ostream& out, const Tile<T,DIM,CS>& t) {
    typedef Tile<T,DIM,CS> tile_type;
    typename tile_type::size_array weight = t.data_.weight();

    out << "{ ";
    typename CS::const_iterator d ;
    typename tile_type::ordinal_type i = 0;
    for(typename tile_type::const_iterator it = t.begin(); it != t.end(); ++it, ++i) {
      for(d =  CS::begin(), ++d; d != CS::end(); ++d) {
        if((i % weight[*d]) == 0)
          out << "{ ";
      }

      out << " " << *it;


      for(d = CS::begin(), ++d; d != CS::end(); ++d) {
        if(((i + 1) % weight[*d]) == 0)
          out << " }";
      }
    }
    out << " }";
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

namespace madness {
  namespace archive {
    template <class Archive, typename T, unsigned int DIM, typename Index>
    struct ArchiveLoadImpl<Archive,TiledArray::Tile<T,DIM,Index>*> {
      typedef TiledArray::Tile<T,DIM,Index> Tile;
      static inline void load(const Archive& ar, Tile*& tileptr) {
        tileptr = new Tile;
        ar & wrap(tileptr,1);
      }
    };

    template <class Archive, typename T, unsigned int DIM, typename Index>
    struct ArchiveStoreImpl<Archive,TiledArray::Tile<T,DIM,Index>*> {
      typedef TiledArray::Tile<T,DIM,Index> Tile;
      static inline void store(const Archive& ar, Tile* const& tileptr) {
        ar & wrap(tileptr,1);
      }
    };

  }
}

#endif // TILEDARRAY_TILE_H__INCLUDED
