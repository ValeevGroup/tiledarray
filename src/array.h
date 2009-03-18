#ifndef ARRAY_H__INCLUDED
#define ARRAY_H__INCLUDED

#include <cassert>
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
    typedef Array<T, DIM, CS> my_type;
    typedef Range<DIM, CS> Range;
    typedef typename Range::tile_iterator RangeIterator;
    typedef Shape<Range> Shape;
    typedef typename Shape::Iterator ShapeIterator;

  public:
    typedef T value_type;
    typedef CS CoordinateSystem;
    typedef typename Range::tile_index index_t;

    /// Tile is implemented in terms of boost::multi_array
    /// it provides reshaping, iterators, etc., and supports direct access to the raw pointer.
    /// array layout must match that given by CoordinateSystem (i.e. both C, or both Fortran)
    class Tile : public boost::multi_array<value_type,DIM> { };

    class Iterator : public boost::iterator_facade<Iterator, Tile, std::output_iterator_tag > {
      typedef boost::iterator_facade<Iterator, Tile, std::output_iterator_tag > iterator_facade_;
    public:

      Iterator(const boost::shared_ptr<ShapeIterator>& it, const Array<T,DIM,CS>& a) :
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

      boost::shared_ptr<ShapeIterator> current_index_;
      const Array& array_ref_;

    };

    Iterator begin() const {
      return Iterator( boost::shared_ptr<ShapeIterator>( shape_->begin() ) );
    }

    Iterator end() const {
      return Iterator( boost::shared_ptr<ShapeIterator>( shape_->end() ) );
    }

    /// array is defined by its shape
    Array(const boost::shared_ptr<Shape>& shp) : shape_(shp)
    {}

    virtual ~Array() {}

    /// access a tile
    virtual Tile& at(const index_t& i) { return dummy_; }

    bool operator ==(const my_type& other) const {
      // TODO: add comparison code.
      return false;
    }

    /// assign each element to a
//    virtual void assign(T a) =0;

    /// where is tile k
//    virtual unsigned int proc(const index& k) const =0;
//    virtual bool local(const index_t& k) const =0;


    /// Low-level interface will only allow permutations and efficient direct contractions
    /// it should be sufficient to use with an optimizing array expression compiler

    /// make new Array by applying permutation P
//    virtual Array transpose(const Permutation<DIM>& P) =0;

    /// Higher-level interface will be be easier to use but necessarily less efficient since it will allow more complex operations
    /// implemented in terms of permutations and contractions by a runtime

    /// bind a string to this array to make operations look normal
    /// e.g. R("ijcd") += T2("ijab") . V("abcd") or T2new("iajb") = T2("ijab")
    /// runtime then can figure out how to implement operations
    // ArrayExpression operator()(const char*) const;

  private:

	// TODO: dummy_ is here for testing purposes, it needs to be removed once tile access has been implemented.
    Tile dummy_;
    boost::shared_ptr<Shape> shape_;
  };

};

#endif // TILEDARRAY_H__INCLUDED
