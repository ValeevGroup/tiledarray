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
    typedef Array<T, DIM, CS> Array_;
    typedef Range<DIM, CS> range;
    typedef typename range::tile_iterator range_iterator;
    typedef Shape<range> shape;
    typedef typename shape::iterator shape_iterator;

  public:
    typedef T value_type;
    typedef CS coordinate_system;
    typedef typename range::ordinal_index ordinal_index;
    typedef typename range::tile_index tile_index;
    typedef typename range::element_index element_index;

    /// Tile is implemented in terms of boost::multi_array
    /// it provides reshaping, iterators, etc., and supports direct access to the raw pointer.
    /// array layout must match that given by CoordinateSystem (i.e. both C, or both Fortran)
    class Tile : public boost::multi_array<value_type,DIM> { };

  private:
    typedef std::map<tile_index, Tile> array_map;

  public:
    class Iterator : public boost::iterator_facade<Iterator, Tile, std::output_iterator_tag > {
      typedef boost::iterator_facade<Iterator, Tile, std::output_iterator_tag > iterator_facade_;
    public:

      Iterator(const boost::shared_ptr<shape_iterator>& it, const Array<T,DIM,CS>& a) :
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
    Array(const boost::shared_ptr<shape>& shp) : shape_(shp)
    {}

    virtual ~Array() {}

    /// Returns the number of dimentions in the array.
    unsigned int dim() const { return DIM; }

    /// Returns the number of elements contained in the array.
    ordinal_index nelements() const { return shape_->range()->size(); }
    ordinal_index ntiles() const { return shape_->range()->ntiles(); }
    const element_index& origin() const { return shape_->range()->start_element(); }


    /// assign each element to a
    virtual void assign(const value_type& val) =0;
    virtual void assign(const element_index& e_idx, const value_type& val) =0;

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

    /// Map that stores all local tiles.
    array_map data_;
    /// Shape pointer.
    boost::shared_ptr<shape> shape_;
  };

};

#endif // TILEDARRAY_H__INCLUDED
