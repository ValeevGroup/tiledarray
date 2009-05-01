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

  template<typename T, unsigned int DIM, typename INDEX, typename CS = CoordinateSystem<DIM> >
  class Tile
  {
  public:
    typedef T value_type;
    typedef INDEX index_type;
    typedef typename index_type::Array size_array;
    typedef CS coordinate_system;
    typedef size_t ordinal_type;

    static const unsigned int dim() { return DIM; }

    Tile(const size_array& size, const value_type val = 0) {
      size_ = size;

      init_();
      data_ = new value_type[n_];

      for(unsigned int i = 0; i < n_; ++i)
        data_[i] = val;
    }

    ~Tile() {
      delete[] data_;
    }

  private:

    void init_() {


      // Get dim ordering iterator
      const detail::DimensionOrder<DIM>& dimorder = coordinate_system::ordering();
      typename detail::DimensionOrder<DIM>::const_iterator d;

      ordinal_type weight = 1;
      for(d = dimorder.begin(); d != dimorder.end(); ++d) {
        // Init ordinal weights.
        weights_[*d] = weight;
        weight *= size_[*d];
      }
      n_ = weight;

    }

    bool includes_(const index_type& i) const{
      index_type origin(0);
      return (i >= origin) && (i.data() < size_);
    }

    /// computes an ordinal index for a given index_type
    ordinal_type ordinal_(const index_type& i) const {
      assert(includes_(i));
      ordinal_type result = dot_product(i.data(), weights_);
      return result;
    }

    ordinal_type n_;
	size_array weights_;
	size_array size_;
	size_array origin_;
    value_type* data_;
  };

  /// The main player: Array
  /// Serves as base to various implementations (local, replicated, distributed)
  ///
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Array : private boost::noncopyable // we don't allow copy constructors, et al. May need in the future.
  {
   public:
    typedef Array<T,DIM,CS> Array_;
    typedef Range<DIM, CS> range_type;
    typedef typename range_type::tile_iterator range_iterator;
    typedef Shape<DIM, CS> shape_type;
    typedef typename shape_type::iterator shape_iterator;
    typedef typename range_type::ordinal_index ordinal_index;
    typedef typename range_type::tile_index tile_index;
    typedef typename range_type::element_index element_index;
    typedef T value_type;
    typedef CS coordinate_system;

    /// Tile is implemented in terms of boost::multi_array
    /// it provides reshaping, iterators, etc., and supports direct access to the raw pointer.
    /// array layout must match that given by CoordinateSystem (i.e. both C, or both Fortran)
    typedef Tile<value_type, DIM, element_index, coordinate_system> tile;

  public:

    class iterator : public boost::iterator_facade<iterator, tile, std::output_iterator_tag > {

      /// Default construction not allowed. Access to the array is required for proper operation.
      iterator();

    public:

      /// Standard constructor used by Array class to create new iterators.
      iterator(const boost::shared_ptr<shape_iterator>& it, const Array_& a) :
        current_index_(it), array_ref_(a)
      {}

      /// copy constructor for iterator.
      iterator(const iterator& other) :
        current_index_(other.current_index_), array_ref_(other.array_ref_)
      {}

      /// Assignment operator.
      iterator& operator =(const iterator& other) {
        current_index_ = other.current_index_;
        array_ref_ = other.array_ref_;
        return *this;
      }

    private:
      friend class boost::iterator_core_access;

      /// Increment the iterator.
      void increment() { ++(*current_index_); }

      /// Compare with another iterator for equality.
      bool equal(iterator const& other) const {
        return (*current_index_ == * other.current_index_) && (array_ref_ == other.array_ref_);
      }

      /// Dereference tile to an iterator.
      value_type& dereference() const { return array_ref_.at( *current_index_ ); }

      boost::shared_ptr<shape_iterator> current_index_;
      const Array& array_ref_;

    };

    iterator begin() const {
      return iterator( boost::shared_ptr<shape_iterator>( shape_->begin() ) );
    }

    iterator end() const {
      return iterator( boost::shared_ptr<shape_iterator>( shape_->end() ) );
    }

    /// array is defined by its shape
    Array(const boost::shared_ptr<shape_type>& shp) : shape_(shp) {}

    virtual ~Array() {}

    /// Access array shape.
    const boost::shared_ptr<shape_type>& shape() const { return shape_; }

    /// Access array range.
    const boost::shared_ptr<range_type>& range() const { return shape_->range(); }

    /// Returns the number of dimensions in the array.
    unsigned int dim() const { return DIM; }

    /// Returns an element index that contains the lower limit of each dimension.
    const element_index& origin() const { return shape_->range()->start_element(); }

    // Array virtual functions

	/// Clone array
    virtual boost::shared_ptr<Array_> clone() const =0;

    /// assign each element to a
    virtual void assign(const value_type& val) =0;

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

    /// Returns the tile index that contains the element index e_idx.
    tile_index get_tile_index(const element_index& e_idx) const {
      assert(includes(e_idx));
      return * this->shape_->range()->find(e_idx);
    }

    /// Returns true when the tile is included in the shape.
	bool includes(const tile_index& t_idx) const {
      return this->shape_->includes(t_idx);
	}

	/// Returns true when the element is included in the range.
	/// It may or may not be included in the shape.
	bool includes(const element_index& e_idx) const {
      return this->shape_->range()->includes(e_idx);
	}

  private:
    /// Shape pointer to a shape object.
    boost::shared_ptr<shape_type> shape_;

  };

};

#endif // TILEDARRAY_H__INCLUDED
