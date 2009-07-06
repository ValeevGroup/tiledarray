#ifndef TILEDARRAY_TILE_H__INCLUDED
#define TILEDARRAY_TILE_H__INCLUDED

#include <array_storage.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/iterator/transform_iterator.hpp>
//#include <world/archive.h>
#include <iosfwd>
#include <functional>
#include <cstddef>

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

  /// Tile is a multi-dimensional dense array.
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
    typedef typename range_type::const_iterator index_iterator;
    typedef typename data_container::const_iterator const_iterator;
    typedef typename data_container::iterator iterator;

    static const unsigned int dim() { return DIM; }

    /// Default constructor, constructs an empty array.
    Tile() : range_(), data_() { }

    /// Construct a tile given a range definition and initialize the data to
    /// equal val.
    Tile(const range_type& range) :
        range_(range), data_(range.size(), value_type())
    { }

    /// Construct a tile given a range definition and initialize the data to
    /// equal val.
    Tile(const range_type& range, const value_type val) :
        range_(range), data_(range.size(), val)
    { }

    /// Construct a tile given a range definition and initialize the data to
    /// equal val.
    template<typename U>
    Tile(const range_type& range, const U val) :
        range_(range), data_(range.size(), static_cast<T>(val))
    { }

    /// Construct a tile given a range definition and initialize the data to
    /// the values contained in the range [first, last).
    template <typename InIter>
    Tile(const range_type& range, InIter first, InIter last) :
    	range_(range), data_(range.size(), first, last)
    { }


    /// Constructs a tile given the dimensions of the tile.
    Tile(const size_array& size, const index_type& origin, const value_type val) :
        range_(size, origin), data_(size, val)
    { }

    template <typename InIter>
    Tile(const size_array& size, const index_type& origin, InIter first, InIter last) :
        range_(size, origin), data_(size, first, last)
    { }

    /// Copy constructor
    Tile(const Tile& t) : range_(t.range_), data_(t.data_) { }

    ~Tile() { }

    /// iterator factory functions
    iterator begin() { return data_.begin(); } // no throw
    const_iterator begin() const { return data_.begin(); } // no throw
    iterator end() { return data_.end(); } // no throw
    const_iterator end() const { return data_.end(); } // no throw

    /// Returns the range information about this tile.
    const index_type& start() const { return range_->start(); }
    const index_type& finish() const { return range_->finish(); }
    const size_array& size() const { return range_->size(); }
    const typename range_type::volume_type volume() const { return range_->volume(); }
    bool includes(const index_type& i) const { return range_->includes(i); }

    /// return a constant reference to the tile range.
    const range_type& range() const { return range_; }

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

    /// Move the origin of the tile to the given index. The overall size and
    /// data are unaffected.
    void set_origin(const index_type& origin) {
      range_->set_origin(origin);
    }

    /// Permute the tile given a permutation.
    Tile& operator ^=(const Permutation<DIM>& p) {
      Tile temp = p ^ (*this);
      swap(temp);
      return *this;
    }

    Tile& operator +=(const Tile& other) {
      assert(this->size_ == other.size_);
      const_iterator other_it = other.begin();
      for(iterator it = begin(); it != end(); ++it, ++other_it)
        *it += *other_it;

      return *this;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & range_ & data_;
    }

    void swap(Tile& other) {
      range_.swap(other.range_);
      data_.swap(other.data_);
    }

  private:

    range_type range_;
    data_container data_;  // element data

    friend std::ostream& operator<< <>(std::ostream& , const Tile&);
    friend Tile operator^ <>(const Permutation<DIM>&, const Tile&);

  }; // class Tile

  namespace detail {
    /// Tile-Tile operation
    template<typename T, unsigned int DIM, typename CS, typename Op>
    struct TileOp : std::binary_function<Tile<T,DIM,CS>, Tile<T,DIM,CS>, Tile<T,DIM,CS> > {
      Tile<T,DIM,CS> operator ()(const Tile<T,DIM,CS>& t1, const Tile<T,DIM,CS>& t2) {
        Tile<T,DIM,CS> result(* t1.range());
        std::transform(t1.begin(), t1.end(), t2.begin(), result.begin(), Op());

        return result;
      }
    };

    /// Tile-Tile operation
    template<typename S, typename T, unsigned int DIM, typename CS, typename Op>
    struct TileScalerOp : std::binary_function<Tile<T,DIM,CS>, Tile<T,DIM,CS>, Tile<T,DIM,CS> > {
      Tile<T,DIM,CS> operator ()(const Tile<T,DIM,CS>& t, S s) {
        return Tile<T,DIM,CS>(t.range(),
            boost::make_transform_iterator(t.begin(), std::bind2nd<Op,T>(Op(), s)),
            boost::make_transform_iterator(t.end(), std::bind2nd<Op,T>(Op(), s)));
      }
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

  /// sum two tiles
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator +(const Tile<T,DIM,CS>& t1, const Tile<T,DIM,CS>& t2) {
    TA_ASSERT( t1.size() == t2.size() ,
        std::runtime_error("operator+(const Tile<T,DIM,CS>&, const Tile<T,DIM,CS>&): Tile dimensions do not match.") );
    detail::TileOp<T,DIM,CS,std::plus<T> > p;
    return p(t1,t2);
  }

  /// add a scaler to each tile element.
  template<typename T, unsigned int DIM, typename CS, typename S>
  Tile<T,DIM,CS> operator +(const Tile<T,DIM,CS>& t, S s) {
    detail::TileScalerOp<S,T,DIM,CS,std::plus<T> > op;
    return op(t,s);
  }

  /// add a scaler to each tile element.
  template<typename T, unsigned int DIM, typename CS, typename S>
  Tile<T,DIM,CS> operator -(const Tile<T,DIM,CS>& t, S s) {
    detail::TileScalerOp<S,T,DIM,CS,std::minus<T> > op;
    return op(t,s);
  }

  /// add a scaler to each tile element.
  template<typename T, unsigned int DIM, typename CS, typename S>
  Tile<T,DIM,CS> operator *(const Tile<T,DIM,CS>& t, S s) {
    detail::TileScalerOp<S,T,DIM,CS,std::plus<T> > op;
    return op(t,s);
  }

  /// add a scaler to each tile element.
  template<typename T, unsigned int DIM, typename CS, typename S>
  Tile<T,DIM,CS> operator *(S s, const Tile<T,DIM,CS>& t) {
    detail::TileScalerOp<S,T,DIM,CS,std::plus<T> > op;
    return op(t,s);
  }

  /// add a scaler to each tile element.
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator -(const Tile<T,DIM,CS>& t) {
    detail::TileScalerOp<T,T,DIM,CS,std::multiplies<T> > op;
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

  /// \arg \c res The results of \c gen()  will be placed in this tile.
  /// \arg \c first, \c last input iterators, which point to a list of indexes.
  /// \arg \c gen Function or functor object used to generate the data.
  ///
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
  template<typename T, unsigned int DIM, typename CS, typename InIter, typename Generator>
  Tile<T, DIM, CS>& generate(Tile<T, DIM, CS>& res, InIter first, InIter last, Generator gen) {
    for(; first != last; ++first)
      res[ *first ] = gen( *first );

    return res;
  }

  /// Assigns a value to the specified range of element in tile with a generator function.

  /// \arg \c res The results of \c gen()  will be placed in this tile.
  /// \arg \c gen Function or functor object used to generate the data.
  ///
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
