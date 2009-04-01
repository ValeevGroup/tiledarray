#ifndef ARRAY_H__INCLUDED
#define ARRAY_H__INCLUDED

#include <cassert>
#include <map>
#include <boost/shared_ptr.hpp>
#include <boost/multi_array.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <coordinates.h>
#include <permutation.h>
#include <range.h>
#include <shape.h>

namespace TiledArray {

  /// The main player: Array
  /// Serves as base to various implementations (local, replicatd, distributed)
  ///
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Array {
  public:
    typedef Array this_type;
    typedef Range<DIM, CS> range_type;
    typedef typename range_type::tile_iterator range_iterator;
    typedef Shape<range_type> shape_type;
    typedef typename shape_type::iterator shape_iterator;
    typedef typename range_type::ordinal_index ordinal_index;
    typedef typename range_type::tile_index tile_index;
    typedef typename range_type::element_index element_index;
    typedef T value_type;
    typedef CS coordinate_system;

    /// Tile is implemented in terms of boost::multi_array
    /// it provides reshaping, iterators, etc., and supports direct access to the raw pointer.
    /// array layout must match that given by CoordinateSystem (i.e. both C, or both Fortran)
    typedef boost::multi_array<value_type,DIM> Tile;

    class Iterator : public boost::iterator_facade<Iterator, Tile, std::output_iterator_tag > {
        typedef boost::iterator_facade<Iterator, Tile, std::output_iterator_tag > iterator_facade_;
      public:

        Iterator(const boost::shared_ptr<shape_iterator>& it, const Array& a) :
          current_index_(it), array_ref_(a)
          {}

        Iterator(const Iterator& other) :
          current_index_(other.current_index_), array_ref_(other.array_ref_)
          {}

        Iterator& operator =(const Iterator& other) {
          current_index_ = other.current_index_;
          array_ref_ = other.array_ref_;
          return *this;
        }

      private:
        friend class boost::iterator_core_access;

        Iterator();

        void increment() { ++(*current_index_); }
        bool equal(Iterator const& other) const {
          return (*current_index_ == * other.current_index_) && (array_ref_ == other.array_ref_); }
        value_type& dereference() const { return array_ref_.at( *current_index_ ); }

        boost::shared_ptr<shape_iterator> current_index_;
        const Array& array_ref_;

    };

    Iterator begin() const {
      return Iterator( boost::shared_ptr<shape_iterator>( shape_->begin() ) );
    }

    Iterator end() const {
      return Iterator( boost::shared_ptr<shape_iterator>( shape_->end() ) );
    }

    /// array is defined by its shape
    Array(const boost::shared_ptr<shape_type>& shp) : shape_(shp) {}
    Array(const Array& other) : shape_(other.shape_) {}
    virtual ~Array() {}
    virtual boost::shared_ptr<this_type> clone() const =0;

    const boost::shared_ptr<shape_type>& shape() const { return shape_; }

    /// Returns the number of dimensions in the array.
    unsigned int ndim() const { return DIM; }

    // WARNING I don't think these can be implemented correctly in principle
    // Returns the number of elements contained in the array.
    //ordinal_index nelements() const { return shape_->range()->size(); }
    //ordinal_index ntiles() const { return shape_->range()->ntiles(); }
    //const element_index& origin() const { return shape_->range()->start_element(); }


    /// assign each element to a
    virtual void assign(const value_type& val) =0;
    //virtual void assign(const element_index& e_idx, const value_type& val) =0;

    /// where is tile k
    virtual unsigned int proc(const tile_index& k) const =0;
    virtual bool local(const tile_index& k) const =0;


    /// Low-level interface will only allow permutations and efficient direct contractions
    /// it should be sufficient to use with an optimizing array expression compiler

    /// make new Array by applying permutation P
//    virtual Array transpose(const Permutation<DIM>& P) =0;

    /// Higher-level interface will be be easier to use but necessarily less efficient
    /// since it will allow more complex operations implemented in terms of permutations
    /// and contractions by a runtime

    /// bind a string to this array to make operations look normal
    /// e.g. R("ijcd") += T2("ijab") . V("abcd") or T2new("iajb") = T2("ijab")
    /// runtime then can figure out how to implement operations
    // ArrayExpression operator()(const char*) const;

  protected:

    tile_index get_tile_index(const element_index& e_idx) const {
      assert(includes(e_idx));
      return * this->shape_->range()->find(e_idx);
    }

	bool includes(const tile_index& t_idx) const {
      return this->shape_->includes(t_idx);
	}

	bool includes(const element_index& e_idx) const {
      return this->shape_->range()->includes(e_idx);
	}

	/// given a tile index, create a boost::array object containg its extents in each dimension
	boost::array<typename Tile::index, DIM> tile_extent(const tile_index& t) const {
	  typedef boost::array<typename Tile::index, DIM> result_type;
	  result_type extents;

	  const range_type& rng = *(shape_->range());
	  unsigned int dim=0;
	  for(typename range_type::range_iterator rng1=rng.begin_range();
	      rng1 != rng.end_range();
	      ++rng1, ++dim) {
	    extents[dim] = rng1->size( t[dim] );
	  }

	  return extents;
	}

  private:

    // no default constructor
    Array();

    /// Shape pointer.
    boost::shared_ptr<shape_type> shape_;
  };

};

#endif // TILEDARRAY_H__INCLUDED
