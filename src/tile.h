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
  template<typename T, unsigned int DIM, typename Index, typename CS = CoordinateSystem<DIM> >
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

    typedef std::pair<const index_type, reference_type> iterator_value;
    typedef std::pair<const index_type, const_reference_type> iterator_const_value;
    typedef detail::ElementIterator<iterator_value, Tile_ > iterator;
    typedef detail::ElementIterator<iterator_const_value, Tile_ > const_iterator;
    ELEMENT_ITERATOR_FRIENDSHIP( iterator_value, Tile_ );
    ELEMENT_ITERATOR_FRIENDSHIP( iterator_const_value, Tile_ );

    static const unsigned int dim() { return DIM; }

    /// Primary constructor
    Tile(const size_array& sizes, const index_type& origin = index_type(), const value_type val = value_type()) :
        n_(volume<typename index_type::volume, DIM>(sizes)), weights_(calc_weights(sizes)),
        sizes_(sizes), origin_(origin), data_(n_, val)
    {}

    Tile(const Tile& t) :
        n_(t.n_), weights_(t.weights_), sizes_(t.sizes_), origin_(t.origin_), data_(t.data_)
    {}

    ~Tile() {}

    // iterator factory functions

    iterator begin() {
      return iterator(std::make_pair(origin_, * data_.begin()), this);
    }

    const_iterator begin() const {
      return const_iterator(std::make_pair(origin_, * data_.begin()), this);
    }

    iterator end() {
      return iterator(std::make_pair(origin_ + sizes_, * data_.end()), this);
    }

    const_iterator end() const {
      return const_iterator(std::make_pair(origin_ + sizes_, * data_.end()), this);
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
      return data_[ordinal_(i)];
#else
      return data_.at(ordinal_(i));
#endif
    }

    /// Element access using the element index without error checking
    reference_type operator[](const index_type& i) const {
#ifdef NDEBUG
      retrun data_[ordinal_(i)];
#else
      return data_.at(ordinal_(i));
#endif
    }

    const size_array& size() const {
      return sizes_.data();
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

    void increment(iterator_value& val) const {
      detail::IncrementCoordinate<DIM,index_type,coordinate_system>(val.first, origin_, origin_ + sizes_);
      val.second = at(val.first);
    }

    ordinal_type n_;                // Number of elements
	size_array weights_;            // Index weights used for calculating ordinal indices
	index_type sizes_;              // Dimension sizes
	index_type origin_;             // Tile origin
    std::vector<value_type> data_;  // element data
  };

  template<typename T, unsigned int DIM, typename Index, typename CS>
  std::ostream& operator <<(std::ostream& out, const Tile<T,DIM,Index,CS>& t) {
    out << "{ ";
    for(typename Tile<T,DIM,Index,CS>::const_iterator i = t.begin(); i != t.end(); ++i)
      out << ", " << i->second;
    out << " }";
    return out;
  }


} // namespace TiledArray

#endif // TILE_H__INCLUDED
