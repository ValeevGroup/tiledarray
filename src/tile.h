#ifndef TILE_H__INCLUDED
#define TILE_H__INCLUDED

#include <array_storage.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
//#include <world/archive.h>
#include <iosfwd>
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

  /// Tile is a multidimensional dense array, the dimensions of the tile are constant.
  template<typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Tile
  {
  public:
	typedef Tile<T, DIM, CS> Tile_;
    typedef T value_type;
    typedef T& reference_type;
    typedef const T & const_reference_type;
    typedef CS coordinate_system;
	typedef DenseArrayStorage<value_type, DIM, LevelTag<0>, coordinate_system > data_container;
    typedef typename data_container::ordinal_type ordinal_type;
    typedef Range<ordinal_type, DIM, LevelTag<0>, coordinate_system > range_type;
    typedef typename range_type::index_type index_type;
    typedef typename range_type::size_array size_array;
    typedef typename range_type::const_iterator index_iterator;
    typedef boost::shared_ptr<range_type> block_ptr;
    typedef boost::shared_ptr<const range_type> const_range_ptr;
    typedef boost::shared_ptr<data_container> data_ptr;
    typedef typename data_container::const_iterator const_iterator;
    typedef typename data_container::iterator iterator;

    static const unsigned int dim() { return DIM; }

    /// Default constructor, constructs an empty array.
    Tile() : block_(), data_() { }

    /// Construct a tile given a block definition and initialize the data to
    /// equal val.
    Tile(const range_type& block, const value_type val = value_type()) :
        block_(block), data_(block.size(), val)
    { }

    /// Construct a tile given a block definition and initialize the data to
    /// the values contained in the range [first, last).
    template <typename InIter>
    Tile(const range_type& block, InIter first, InIter last) :
    	block_(block), data_(block.size(), first, last)
    { }


    /// Constructs a tile given the dimensions of the tile.
    Tile(const size_array& size, const index_type& origin = index_type(), const value_type val = value_type()) :
        block_(size, origin), data_(size, val)
    { }

    template <typename InIter>
    Tile(const size_array& size, const index_type& origin, InIter first, InIter last) :
        block_(size, origin), data_(size, first, last)
    { }

    /// Copy constructor
    Tile(const Tile& t) : block_(t.block_), data_(t.data_) { }

    ~Tile() { }

    /// iterator factory functions
    iterator begin() { return data_.begin(); } // no throw
    const_iterator begin() const { return data_.begin(); } // no throw
    iterator end() { return data_.end(); } // no throw
    const_iterator end() const { return data_.end(); } // no throw

    /// Returns the block information about this tile.
    const index_type& start() const { return block_->start(); }
    const index_type& finish() const { return block_->finish(); }
    const size_array& size() const { return block_->size(); }
    const typename range_type::volume_type volume() const { return block_->volume(); }
    bool includes(const index_type& i) const { return block_->includes(i); }

    /// Element access with range checking
    reference_type at(const ordinal_type& i) { return data_.at(i); }
    const_reference_type at(const ordinal_type& i) const { return data_.at(i); }
    reference_type at(const index_type& i){ return data_.at(i); }
    const_reference_type at(const index_type& i) const { return data_.at(i); }

    /// Element access without error checking
    reference_type operator [](const ordinal_type& i) {
#ifdef NDEBUG
      return data_[i];
#else
      return data_.at(i);
#endif
    }

    const_reference_type operator [](const ordinal_type& i) const {
#ifdef NDEBUG
      return data_[i];
#else
      return data_.at(i);
#endif
    }

    reference_type operator [](const index_type& i) {
#ifdef NDEBUG
      return data_[i];
#else
      return data_.at(i);
#endif
    }

    /// Element access using the element index without error checking
    const_reference_type operator [](const index_type& i) const {
#ifdef NDEBUG
      retrun data_[i];
#else
      return data_.at(i);
#endif
    }

    /// Assigns a value to the specified range of element in tile.
    /// *iterator = gen(index_type&)
    template <typename Generator>
    Tile& assign(index_iterator first, index_iterator last, Generator gen) {
      for(; first != last; ++first)
        data_[ *first ] = gen( *first );

      return *this;
    }

    /// Assigns a value to each element in tile.
    /// *iterator = gen(index_type&)
    template <typename Generator>
    Tile& assign(Generator gen) {
      typename range_type::const_iterator b_it = block_.begin();
      for(iterator it = begin(); it != end(); ++it, ++b_it)
        *it = gen(*b_it);

      return *this;
    }

    /// Resize the tile.
    void resize(const size_array& sizes, const value_type& val = value_type()) {
      block_->resize(sizes);
      data_->resize(block_->volume(), val);
    }

    void set_origin(const index_type& origin) {
      block_->set_origin(origin);
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
      ar & block_ & data_;
    }

    void swap(Tile& other) {
      block_.swap(other.block_);
      data_.swap(other.data_);
    }

  private:

    range_type block_;
    data_container data_;  // element data

    friend std::ostream& operator<< <>(std::ostream& , const Tile&);
    friend Tile operator^ <>(const Permutation<DIM>&, const Tile&);

  }; // class Tile

  /// Permute the tile given a permutation.
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator ^(const Permutation<DIM>& p, const Tile<T,DIM,CS>& t) {
    Tile<T,DIM,CS> result(t);
    result.block_ ^= p;
    result.data_ ^= p;

    return result;
  }

  /// sum two tiles
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator +(const Tile<T,DIM,CS>& t1, const Tile<T,DIM,CS>& t2) {
    assert( t1.size() == t2.size() );
    Tile<T,DIM,CS> result(* t1.block());
    typename Tile<T,DIM,CS>::const_iterator it1 = t1.begin();
    typename Tile<T,DIM,CS>::const_iterator it2 = t2.begin();
    for(typename Tile<T,DIM,CS>::iterator itr = result.begin(); itr != result.end(); ++itr, ++it1, ++it2) {
      *itr = *it1 + *it2;
    }

    return result;
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

#endif // TILE_H__INCLUDED
