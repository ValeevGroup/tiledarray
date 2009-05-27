#ifndef TILE_H__INCLUDED
#define TILE_H__INCLUDED

#include <coordinate_system.h>
#include <block.h>
#include <madness_runtime.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
//#include <world/archive.h>
#include <vector>
#include <iosfwd>

namespace TiledArray {

  // Forward declaration of TiledArray components.
  template <unsigned int DIM>
  class Permutation;

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
    typedef size_t ordinal_type;
    typedef ArrayCoordinate<ordinal_type, DIM, LevelTag<0>, coordinate_system> index_type;
    typedef typename index_type::Array size_array;
    typedef Block<ordinal_type, DIM, LevelTag<0>, coordinate_system > block_type;
    typedef boost::shared_ptr<block_type> block_ptr;
    typedef boost::shared_ptr<const block_type> const_block_ptr;
    typedef std::vector<value_type> data_container;
    typedef typename data_container::const_iterator const_iterator;
    typedef typename data_container::iterator iterator;

    static const unsigned int dim() { return DIM; }

    /// Default constructor
    Tile() : block_(), data_(0) {
      block_ = boost::make_shared<block_type>();
    }

    /// Primary constructor. The block pointer must point to a properly
    /// initialized Block<>.
    Tile(const block_ptr& block, const value_type val = value_type()) : block_(block), data_(block_->volume(), val)
    { }

    template <typename InIter>
    Tile(const block_ptr& block, InIter first, InIter last) : block_(block), data_(first, last) {
      data_.resize(block->volume());
    }


    /// Constructs a tile given the dimensions of the tile.
    Tile(const size_array& sizes, const index_type& origin = index_type(), const value_type val = value_type()) :
        data_(0) {
      block_ = boost::make_shared<block_type>(sizes, origin);
      data_.resize(block_->volume(), val);
    }

    template <typename InIter>
    Tile(const size_array& sizes, const index_type& origin, InIter first, InIter last) :
        data_(first, last) {
      block_ = boost::make_shared<block_type>(sizes, origin);
      data_.resize(block_->volume());
    }

    /// Copy constructor
    Tile(const Tile& t) : block_(), data_(t.data_) {
      block_ = boost::make_shared<block_type>(* t.block_);
    }

    ~Tile() {}

    // iterator factory functions

    iterator begin() {
      return data_.begin();
    }

    const_iterator begin() const {
      return data_.begin();
    }

    iterator end() {
      return data_.end();
    }

    const_iterator end() const {
      return data_.end();
    }

    /// Element access using the ordinal index with error checking
    reference_type at(const ordinal_type& i) {
      return data_.at(i);
    }

    /// Element access using the ordinal index with error checking
    const_reference_type at(const ordinal_type& i) const {
      return data_.at(i);
    }

    /// Element access using the element index with error checking
    reference_type at(const index_type& i){
      return data_.at(ordinal(i));
    }

    /// Element access using the element index with error checking
    const_reference_type at(const index_type& i) const {
      return data_.at(ordinal(i));
    }

    /// Element access using the ordinal index without error checking
    reference_type operator[](const ordinal_type& i) {
#ifdef NDEBUG
      return data_[i];
#else
      return data_.at(i);
#endif
    }

    /// Element access using the ordinal index without error checking
    const_reference_type operator[](const ordinal_type& i) const {
#ifdef NDEBUG
      return data_[i];
#else
      return data_.at(i);
#endif
    }

    /// Element access using the element index without error checking
    reference_type operator[](const index_type& i) {
#ifdef NDEBUG
      return data_[block_->ordinal(i)];
#else
      return data_.at(block_->ordinal(i));
#endif
    }

    /// Element access using the element index without error checking
    const_reference_type operator[](const index_type& i) const {
#ifdef NDEBUG
      retrun data_[block_->ordinal(i)];
#else
      return data_.at(block_->ordinal(i));
#endif
    }

    /// Returns a constant pointer to the tile block definition.
    block_ptr block() const { return block_; }

    /// Assigns a value to the specified range of element in tile.
    /// *iterator = gen(index_type&)
    template <typename Generator>
    Tile_& assign(const_iterator& first, const_iterator& last, Generator gen) {
      assert(last >= first);
      typename block_type::const_iterator b_it = block_->begin();
      for(iterator it = begin(); it != first; ++it, ++b_it);
      for(iterator it = first; it != last; ++it)
        *it = gen(*b_it);

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
      data_.resize(block_->volume(), val);
    }

    void set_origin(const index_type& origin) {
      block_->set_origin(origin);
    }

    /// Permute the tile given a permutation.
    Tile_& operator ^=(const Permutation<DIM>& p) {
      // copy data needed for iteration.
      const block_type temp_block(*block_);
      const data_container temp_data(data_);

  	  // Permute support data.
      *block_ ^= p;

      // Permute the tile data.
      const_iterator data_it = temp_data.begin();
      typename block_type::const_iterator index_it = temp_block.begin();
      for(; data_it != temp_data.end(); ++data_it, ++index_it) {
        data_[block_->ordinal(p ^ *index_it)] = *data_it;
      }
      return *this;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & (*block_) & data_;
    }

  private:

    block_ptr block_;
    data_container data_;  // element data

    friend std::ostream& operator<< <>(std::ostream& , const Tile&);
    friend Tile_ operator^ <>(const Permutation<DIM>& p, const Tile_& t);

  };

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
    DimOrder order = CS::ordering();
    typedef Tile<T,DIM,CS> tile_type;

    out << "{ ";
    typename DimOrder::const_iterator d ;
    typename tile_type::ordinal_type i = 0;
    for(typename tile_type::const_iterator it = t.begin(); it != t.end(); ++it, ++i) {
      for(d = order.begin(), ++d; d != order.end(); ++d) {
        if((i % t.block()->weights()[*d]) == 0)
          out << "{ ";
      }

      out << " " << *it;


      for(d = order.begin(), ++d; d != order.end(); ++d) {
        if(((i + 1) % t.block()->weights()[*d]) == 0)
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
