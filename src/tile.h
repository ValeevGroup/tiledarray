#ifndef TILE_H__INCLUDED
#define TILE_H__INCLUDED

#include <vector>
#include <utility>
#include <iostream>
#include <coordinates.h>
#include <permutation.h>
#include <iterator.h>

namespace TiledArray {

  /// Tile is a multidimensional dense array, the dimensions of the tile are constant.
  template<typename T, unsigned int DIM, typename Index = ArrayCoordinate<size_t, 3, LevelTag<0> >, typename CS = CoordinateSystem<DIM> >
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

    typedef detail::ElementIterator<value_type, index_type, Tile_ > iterator;
    typedef detail::ElementIterator<value_type const, index_type, const Tile_ > const_iterator;
    ELEMENT_ITERATOR_FRIENDSHIP( value_type, index_type, Tile_ );
    ELEMENT_ITERATOR_FRIENDSHIP( value_type const, index_type, const Tile_ );

    static const unsigned int dim() { return DIM; }

    /// Primary constructor
    Tile(const size_array& sizes, const index_type& origin = index_type(), const value_type val = value_type()) :
        n_(volume<typename index_type::volume, DIM>(sizes)), weights_(calc_weights(sizes)),
        sizes_(sizes), origin_(origin), data_(n_, val)
    {}

    /// Copy constructor
    Tile(const Tile& t) :
        n_(t.n_), weights_(t.weights_), sizes_(t.sizes_), origin_(t.origin_), data_(t.data_)
    {}

    ~Tile() {}

    // iterator factory functions

    iterator begin() {
      return iterator(origin_, this);
    }

    const_iterator begin() const {
      return const_iterator(origin_, this);
    }

    iterator end() {
      return iterator(origin_ + sizes_, this);
    }

    const_iterator end() const {
      return const_iterator(origin_ + sizes_, this);
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
      return data_.at(ordinal_(i));
    }

    /// Element access using the element index with error checking
    const_reference_type at(const index_type& i) const {
      return data_.at(ordinal_(i));
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

    /// Returns the number of elements contained in the array.
    ordinal_type nelements() const {
      return data_.size();
    }

    /// Assigns a value to the specified range of element in tile.
    /// *iterator = gen(index_type&)
    template <typename Generator>
    Tile_& assign(const_iterator& first, const_iterator& last, Generator& gen) {
      for(iterator it = first; it != last; ++it)
        *it = gen(it.index());

      return *this;
    }

    /// Assigns a value to each element in tile.
    /// *iterator = gen(index_type&)
    template <typename Generator>
    Tile_& assign(Generator& gen) {
        for(iterator it = begin(); it != end(); ++it)
          *it = gen(it.index());

        return *this;
    }

  private:

    Tile();

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

      return (i >= origin_) && (i < (origin_ + sizes_));
    }

    /// computes an ordinal index for a given index_type
    ordinal_type ordinal(const index_type& i) const {
      assert(includes(i));
      ordinal_type result = dot_product(i.data(), weights_);
      return result;
    }

    void increment(index_type& i) const {
      detail::IncrementCoordinate<DIM,index_type,coordinate_system>(i, origin_, origin_ + sizes_);
    }

    ordinal_type n_;                // Number of elements
	size_array weights_;            // Index weights used for calculating ordinal indices
	index_type sizes_;              // Dimension sizes
	index_type origin_;             // Tile origin
    std::vector<value_type> data_;  // element data

    // ToDo: Why dees this not work in gcc? It worked for shape.
//    friend std::ostream& operator<< <> (std::ostream& , const Tile&);

  };

  /// Permute the tile given a permutation.
  template<typename T, unsigned int DIM, typename Index, typename CS>
  Tile<T,DIM,Index,CS> operator ^(const Permutation<DIM>& p, const Tile<T,DIM,Index,CS>& t) {
    Tile<T,DIM,Index,CS> result(t.size());
    for(typename Tile<T,DIM,Index,CS>::const_iterator it = t.begin(); it != t.end(); ++it)
      result[p ^ it.index()] = *it;

//      weights_ = p ^ weights_;
//      sizes_ = p ^ sizes_;
//      origin_ = p ^ origin_;

    return result;
  }

  /// ostream output orperator.
  template<typename T, unsigned int DIM, typename Index, typename CS>
  std::ostream& operator <<(std::ostream& out, const Tile<T,DIM,Index,CS>& t) {
    typedef Tile<T,DIM,Index,CS> tile_type;
    const detail::DimensionOrder<DIM>& dimorder = CS::ordering();
    typename detail::DimensionOrder<DIM>::const_iterator d;

    // ToDo: remove this code when the function is made a friend of tile,
    // so we don't have to recalculate weights.
    typename tile_type::ordinal_type weight = 1;
    typename tile_type::size_array weights;
    for(d = dimorder.begin(); d != dimorder.end(); ++d) {
      // calc ordinal weights.
      weights[*d] = weight;
      weight *= t.size()[*d];
    }

    out << "{ ";
    for(typename tile_type::ordinal_type i = 0; i < t.nelements(); ++i) {
      for(d = dimorder.begin(), ++d; d != dimorder.end(); ++d) {
        if((i % weights[*d]) == 0)
          out << "{ ";
      }

      out << " " << t[i];


      for(d = dimorder.begin(), ++d; d != dimorder.end(); ++d) {
        if(((i + 1) % weights[*d]) == 0)
          out << " }";
      }
    }
    out << " }";
    return out;
  }


} // namespace TiledArray

#endif // TILE_H__INCLUDED
