#ifndef TILE_H__INCLUDED
#define TILE_H__INCLUDED

#include <vector>
#include <utility>
#include <iostream>

#include <coordinates.h>
#include <permutation.h>
#include <iterator.h>

namespace TiledArray {

  template<typename T, unsigned int DIM, typename Index, typename CS>
  class Tile;
  template<typename T, unsigned int DIM, typename Index, typename CS>
  Tile<T,DIM,Index,CS> operator ^(const Permutation<DIM>& p, const Tile<T,DIM,Index,CS>& t);
  template<typename T, unsigned int DIM, typename Index, typename CS>
  std::ostream& operator <<(std::ostream& out, const Tile<T,DIM,Index,CS>& t);

  /// Tile is a multidimensional dense array, the dimensions of the tile are constant.
  template<typename T, unsigned int DIM, typename Index = ArrayCoordinate<size_t, DIM, LevelTag<0> >, typename CS = CoordinateSystem<DIM> >
  class Tile
  {
  public:
	typedef Tile<T, DIM, Index, CS> Tile_;
    typedef T value_type;
    typedef T& reference_type;
    typedef T const & const_reference_type;
    typedef Index index_type;
    typedef typename index_type::Array size_array;
    typedef CS coordinate_system;
    typedef size_t ordinal_type;

  private:
    typedef detail::IndexIterator<index_type, Tile_> index_iterator;
    INDEX_ITERATOR_FRIENDSHIP(index_type, Tile_);
  public:
    typedef detail::ElementIterator<value_type, index_iterator, Tile_ > iterator;
    typedef detail::ElementIterator<value_type const, index_iterator, const Tile_ > const_iterator;
//    ELEMENT_ITERATOR_FRIENDSHIP( value_type, index_iterator, Tile_ );
//    ELEMENT_ITERATOR_FRIENDSHIP( value_type const, index_iterator, const Tile_ );

    static const unsigned int dim() { return DIM; }

    /// Default constructor
    Tile() : n_(0), sizes_(0ul), start_(0ul), finish_(0ul), data_(0) {
      for(unsigned int i = 0; i < Tile_::dim(); ++i)
        weights_[i] = 0;
    }

    /// Primary constructor
    Tile(const size_array& sizes, const index_type& origin = index_type(), const value_type val = value_type()) :
        n_(volume<typename index_type::volume, DIM>(sizes)), weights_(calc_weights(sizes)),
        sizes_(sizes), start_(origin), finish_(origin + sizes), data_(n_, val)
    {}

    /// Copy constructor
    Tile(const Tile& t) :
        n_(t.n_), weights_(t.weights_), sizes_(t.sizes_),
        start_(t.start_), finish_(t.finish_), data_(t.data_)
    {}

    ~Tile() {}

    // iterator factory functions

    iterator begin() {
      return iterator(index_iterator(start_, this), this);
    }

    const_iterator begin() const {
      return const_iterator(index_iterator(start_, this), this);
    }

    iterator end() {
      return iterator(index_iterator(finish_, this), this);
    }

    const_iterator end() const {
      return const_iterator(index_iterator(finish_, this), this);
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
      return data_[ordinal(i)];
#else
      return data_.at(ordinal(i));
#endif
    }

    /// Element access using the element index without error checking
    const_reference_type operator[](const index_type& i) const {
#ifdef NDEBUG
      retrun data_[ordinal(i)];
#else
      return data_.at(ordinal(i));
#endif
    }

    /// Returns an array with the size of each dimension.
    const size_array& size() const {
      return sizes_.data();
    }

    /// Returns the lower bound of the tile
    const index_type& start() const {
      return start_;
    }

    const index_type& finish() const {
      return finish_;
    }

    /// Returns the number of elements contained in the array.
    ordinal_type nelements() const {
      return data_.size();
    }

    /// Assigns a value to the specified range of element in tile.
    /// *iterator = gen(index_type&)
    template <typename Generator>
    Tile_& assign(const_iterator& first, const_iterator& last, Generator gen) {
      for(iterator it = first; it != last; ++it)
        *it = gen(it.index());

      return *this;
    }

    /// Assigns a value to each element in tile.
    /// *iterator = gen(index_type&)
    template <typename Generator>
    Tile_& assign(Generator gen) {
      for(iterator it = begin(); it != end(); ++it)
        *it = gen(it.index());

      return *this;
    }

    /// Resize the tile.
    void resize(const size_array& sizes, const value_type& val = value_type()) {
      sizes_ = sizes;
      finish_ = start_ + sizes;
      n_ = volume<typename index_type::volume, DIM>(sizes);
      weights_ = calc_weights(sizes);
      data_.resize(n_, val);
    }

    void set_origin(const index_type& origin) {
      start_ = origin;
      finish_ = origin + sizes_;
    }

    /// Permute the tile given a permutation.
    Tile_& operator ^=(const Permutation<DIM>& p) {
      // copy data needed for iteration.
      index_type temp_index(start_);
      const index_type temp_start(start_);
      const index_type temp_finish(finish_);
  	  const std::vector<value_type> temp_data(data_);

  	  // Permute support data.
  	  start_ ^= p;
  	  finish_ ^= p;
  	  sizes_ = p ^ sizes_;
  	  weights_ = calc_weights(sizes_.data());

      // Permute the tile data.
      for(typename std::vector<value_type>::const_iterator it = temp_data.begin(); it != temp_data.end(); ++it) {
        data_[ordinal(p ^ temp_index)] = *it;
        detail::IncrementCoordinate<DIM,index_type,coordinate_system>(temp_index, temp_start, temp_finish);
      }
      return *this;
    }

  private:

    static size_array calc_weights(const size_array& sizes) {
      size_array result;

      // Get dim ordering iterator
      const detail::DimensionOrder<DIM>& dimorder = coordinate_system::ordering();
      typename detail::DimensionOrder<DIM>::const_iterator d;

      ordinal_type weight = 1;
      for(d = dimorder.begin(); d != dimorder.end(); ++d) {
        // calc ordinal weights.
        result[*d] = weight;
        weight *= sizes[*d];
      }

      return result;
    }

    /// Check the coordinate to make sure it is within the tile range
    bool includes(const index_type& i) const{

      return (i >= start()) && (i < finish());
    }

    /// computes an ordinal index for a given index_type
    ordinal_type ordinal(const index_type& i) const {
      assert(includes(i));
      index_type relative_index = i - start_;
      ordinal_type result = dot_product(relative_index.data(), weights_);
      return result;
    }

    void increment(index_type& i) const {
      detail::IncrementCoordinate<DIM,index_type,coordinate_system>(i, start_, finish_);
    }

    ordinal_type n_;                // Number of elements
	size_array weights_;            // Index weights used for calculating ordinal indices
	index_type sizes_;              // Dimension sizes
	index_type start_;              // Tile origin
	index_type finish_;              // Tile upper bound
    std::vector<value_type> data_;  // element data

    friend std::ostream& operator<< <>(std::ostream& , const Tile&);
//    friend Tile_& operator^ <>(const Permutation<DIM>&, const Tile_&);

  };

  /// Permute the tile given a permutation.
  template<typename T, unsigned int DIM, typename Index, typename CS>
  Tile<T,DIM,Index,CS> operator ^(const Permutation<DIM>& p, const Tile<T,DIM,Index,CS>& t) {
    Tile<T,DIM,Index,CS> result(t);

    return result ^= p;
  }

  /// ostream output orperator.
  template<typename T, unsigned int DIM, typename Index, typename CS>
  std::ostream& operator <<(std::ostream& out, const Tile<T,DIM,Index,CS>& t) {
    typedef Tile<T,DIM,Index,CS> tile_type;
    const detail::DimensionOrder<DIM>& dimorder = CS::ordering();
    typename detail::DimensionOrder<DIM>::const_iterator d;

    out << "{ ";
    for(typename tile_type::ordinal_type i = 0; i < t.nelements(); ++i) {
      for(d = dimorder.begin(), ++d; d != dimorder.end(); ++d) {
        if((i % t.weights_[*d]) == 0)
          out << "{ ";
      }

      out << " " << t[i];


      for(d = dimorder.begin(), ++d; d != dimorder.end(); ++d) {
        if(((i + 1) % t.weights_[*d]) == 0)
          out << " }";
      }
    }
    out << " }";
    return out;
  }


} // namespace TiledArray

#endif // TILE_H__INCLUDED
