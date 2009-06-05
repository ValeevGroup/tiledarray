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
    typedef std::size_t ordinal_type;
    typedef T value_type;
    typedef T& reference_type;
    typedef const T & const_reference_type;
    typedef CS coordinate_system;
	typedef DenseArrayStorage<value_type, DIM, LevelTag<0>, coordinate_system > data_container;
    typedef Block<ordinal_type, DIM, LevelTag<0>, coordinate_system > block_type;
    typedef typename block_type::index_type index_type;
    typedef typename block_type::size_array size_array;
    typedef typename block_type::const_iterator block_iterator;
    typedef boost::shared_ptr<block_type> block_ptr;
    typedef boost::shared_ptr<const block_type> const_block_ptr;
    typedef boost::shared_ptr<data_container> data_ptr;
    typedef typename data_container::const_iterator const_iterator;
    typedef typename data_container::iterator iterator;

    static const unsigned int dim() { return DIM; }

    /// Default constructor, constructs an empty array.
    Tile() : block_(), data_() {
      block_ = boost::make_shared<block_type>();
      data_ = boost::make_shared<data_container>();
    }

    /// Construct a tile given a block definition and initialize the data to
    /// equal val.
    Tile(const block_ptr& block, const value_type val = value_type()) :
        block_(block), data_()
    {
      data_ = boost::make_shared<data_container>(block_->size(), val);
    }

    /// Construct a tile given a block definition and initialize the data to
    /// the values contained in the range [first, last).
    template <typename InIter>
    Tile(const block_ptr& block, InIter first, InIter last) :
    	block_(block), data_()
    {
      data_ = boost::make_shared<data_container>(block_->size(), first, last);
    }


    /// Constructs a tile given the dimensions of the tile.
    Tile(const size_array& size, const index_type& origin = index_type(), const value_type val = value_type()) :
        block_(), data_()
    {
      block_ = boost::make_shared<block_type>(size, origin);
      data_ = boost::make_shared<data_container>(size, val);
    }

    template <typename InIter>
    Tile(const size_array& size, const index_type& origin, InIter first, InIter last) :
        block_(), data_()
    {
      block_ = boost::make_shared<block_type>(size, origin);
      data_ = boost::make_shared<data_container>(size, first, last);
    }

    /// Copy constructor
    Tile(const Tile& t) : block_(), data_() {
      block_ = boost::make_shared<block_type>(* t.block_);
      data_ = boost::make_shared<data_container>(* t.data_);
    }

    ~Tile() {}

    /// iterator factory functions
    iterator begin() { return data_->begin(); } // no throw
    const_iterator begin() const { return data_->begin(); } // no throw
    iterator end() { return data_->end(); } // no throw
    const_iterator end() const { return data_->end(); } // no throw

    /// Returns the block information about this tile.
    const index_type& start() const { return block_->start(); }
    const index_type& finish() const { return block_->finish(); }
    const size_array& size() const { return block_->size(); }
    const_block_ptr block() const {
      const_block_ptr result = boost::const_pointer_cast<const block_type>(block_);
    }

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
    Tile_& assign(block_iterator first, block_iterator last, Generator gen) {
      for(; first != last; ++first)
        data_[ *first ] = gen( *first );

      return *this;
    }

    /// Assigns a value to each element in tile.
    /// *iterator = gen(index_type&)
    template <typename Generator>
    Tile_& assign(Generator gen) {
      typename block_type::const_iterator b_it = block_->begin();
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
    Tile_& operator ^=(const Permutation<DIM>& p) {
      // copy data needed for iteration.
      const block_type temp_block(*block_);
      const data_container temp_data(*data_);

  	  // Permute support data.
      *block_ ^= p;

      // Permute the tile data.
      const_iterator data_it = temp_data.begin();
      typename block_type::const_iterator index_it = temp_block.begin();
      for(; data_it != temp_data.end(); ++data_it, ++index_it) {
        (*data_)[ p ^ (*index_it)] = *data_it;
      }
      return *this;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & (*block_) & data_;
    }

  private:

    block_ptr block_;
    data_ptr data_;  // element data

    friend std::ostream& operator<< <>(std::ostream& , const Tile&);
    friend Tile_ operator^ <>(const Permutation<DIM>& p, const Tile_& t);

  }; // class Tile

  /// Permute the tile given a permutation.
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator ^(const Permutation<DIM>& p, const Tile<T,DIM,CS>& t) {
    Tile<T,DIM,CS> result;
    return result;
  }

  /// Permute the tile given a permutation.
  template<typename T, unsigned int DIM, typename CS>
  Tile<T,DIM,CS> operator +(const Tile<T,DIM,CS>& t1, const Tile<T,DIM,CS>& t2) {
    assert( t1.size() == t2.size() );
    Tile<T,DIM,CS> result(* t1.block());
    typename Tile<T,DIM,CS>::const_iterator it1 = t1.begin();
    typename Tile<T,DIM,CS>::const_iterator it2 = t2.begin();
    for(typename Tile<T,DIM,CS>::iterator itr = result.begin(); itr != result.end(); ++itr) {
      *itr = *it1 + *it2;
      ++it1;
      ++it2;
    }


    return result;
  }


  /// ostream output orperator.
  template<typename T, unsigned int DIM, typename CS>
  std::ostream& operator <<(std::ostream& out, const Tile<T,DIM,CS>& t) {
    typedef  detail::DimensionOrder<DIM> DimOrder;
    typedef Tile<T,DIM,CS> tile_type;
    DimOrder order = CS::ordering();
    typename tile_type::size_array weight = t.data_->weight();

    out << "{ ";
    typename DimOrder::const_iterator d ;
    typename tile_type::ordinal_type i = 0;
    for(typename tile_type::const_iterator it = t.begin(); it != t.end(); ++it, ++i) {
      for(d = order.begin(), ++d; d != order.end(); ++d) {
        if((i % weight[*d]) == 0)
          out << "{ ";
      }

      out << " " << *it;


      for(d = order.begin(), ++d; d != order.end(); ++d) {
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
