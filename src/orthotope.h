#ifndef ORTHOTOPE_H__INCLUDED
#define ORTHOTOPE_H__INCLUDED

#include <iostream>
#include <range.h>
#include <tuple.h>
#include <boost/iterator/filter_iterator.hpp>

namespace TiledArray {

// Orthotope stores tile information of a rectilinear array. 
template<unsigned int DIM>
class Orthotope
{
	std::vector<Range> m_ranges;	// Vector of range data for each dimension

public:
	// typedefs
	typedef Range::index_t                      index_t;
	typedef Tuple<DIM> tile_index;
    typedef Tuple<DIM> element_index;
	
    /// used to implement Shape::iterator and Shape::const_iterator
    template <typename Value>
    class Iterator : public boost::iterator_facade<
       Iterator<Value>,
       Value,
       std::input_iterator_tag
      >
    {
      public:
        typedef Orthotope<DIM> Container;
        
        Iterator(const Iterator& other) : container_(other.container_), current_(other.current_) {}
        ~Iterator() {}
        
      private:
        friend class boost::iterator_core_access;
        friend class Orthotope<DIM>;
        Iterator(const Value& cur, const Container* container) : container_(container), current_(cur) {}
        
        bool equal(Iterator<Value> const& other) const
        {
          return current_ == other.current_;
        }

        void increment() {
          // increment least significant
          int lsdim = DIM-1;
          int lsindex = ++(current_[lsdim]);
          // if necessary, carry over
          const Tuple<DIM> low = container_->low();
          const Tuple<DIM> high = container_->high();
          while (lsindex >= high[lsdim]) {
            current_[lsdim] = low[lsdim];
            --lsdim;
            // if ran out of dimensions break out of the loop
            if (lsdim >= 0)
              lsindex = ++(current_[lsdim]);
            else
              break;
          }
        }

        Value& dereference() const { return const_cast<Value&>(current_); }

        Iterator();
        
        const Container* container_;
        Value current_;
    };
    
    // ready to declare iterators
    typedef std::vector<Range>::const_iterator range_iterator;
    typedef Iterator< tile_index > tile_iterator;
    typedef Iterator< element_index > iterator;

	// Default constructor
	Orthotope() : 
		m_ranges(DIM, Range())
	{}

	// Constructed with an array of ranges
	Orthotope(const Range* ranges) :
		m_ranges(ranges, ranges + DIM)
	{
#if (TA_DLEVEL >= 1)
		for(unsigned int dim = 0; dim < DIM; ++dim)
			assert(ranges + dim);

#endif
	}

	// Construct from a set arrays
	Orthotope(const size_t** ranges, const size_t tiles[DIM]) : 
		m_ranges(DIM, Range())
	{
		for(unsigned int dim = 0; dim < DIM; ++dim)
		{
			assert(ranges[dim]);
			m_ranges[dim] = Range(ranges[dim], tiles[dim]);
		}
	}


	/// Constructor from a vector of ranges
	Orthotope(const std::vector<Range>& ranges) :
		m_ranges(ranges)
	{
		assert(ranges.size() == DIM);
	}

	/// Returns an iterator pointing to the first range.
	inline tile_iterator
	begin() const
	{
	  return tile_iterator(tile_index(),this);
	}

	/// Return an iterator pointing one past the last dimension.
	inline tile_iterator
	end() const
	{
	  tile_index _end;
	  _end[DIM-1] = m_ranges[DIM-1].ntiles();
      return tile_iterator(_end,this);
	}

    /// Returns an iterator pointing to the first range.
    inline range_iterator
    begin_range() const
    {
      return m_ranges.begin();
    }

    /// Return an iterator pointing one past the last dimension.
    inline range_iterator
    end_range() const
    {
      return m_ranges.end();
    }

	/// Returns an iterator pointing to the element in the dim range.
	inline Range::const_iterator
	begin(const unsigned int dim) const
	{
		assert(dim < DIM);
		return m_ranges[dim].begin();
	}

	/// Returns an iterator pointing to the end of dim range.
	inline Range::const_iterator
	end(const unsigned int dim) const
	{
		assert(dim < DIM);
		return m_ranges[dim].end();
	}

	// Accessor functions

	/// change one of the dimension ranges.
	inline void
	set_dim(const unsigned int dim, const Range& rng)
	{
		assert(dim < DIM);
		m_ranges[dim] = rng;
	}

	/// Return a reference to one of the dimension ranges.
	inline const Range&
	get_dim(const unsigned int dim) const
	{
		assert(dim < DIM);
		return m_ranges[dim];
	}

	/// Return a reference to one of the dimension ranges.
	inline const Range&
	operator [](const unsigned int dim)
	{
		return get_dim(dim);
	}

	/// Returns true if element_index is within the bounds of the orthotope.
	inline bool
	includes(const Tuple<DIM>& element_index) const 
		{return (element_index >= low()) && (element_index < high());}

#if 0
	/// return tuple with lower bound for each dimension.
	inline Tuple<DIM>
	low() const
	{
		Tuple<DIM> tmp;
		for(unsigned int dim = 0; dim < DIM; ++dim)
			tmp[dim] = m_ranges[dim].low();

		return tmp;
	}

	/// Return tuple with the upper bound for each dimension.
	inline Tuple<DIM>
	high() const
	{
		Tuple<DIM> tmp;
		for(unsigned int dim = 0; dim < DIM; ++dim)
			tmp[dim] = m_ranges[dim].high();

		return tmp;
	}

	/// Returns a tuple with the number of elements in each dimension
	inline Tuple<DIM>
	size() const
	{
		return high() - low();
	}
#endif
	
	/// Returns the number of tiles in each dimension.
	inline Tuple<DIM>
	tile_size() const
	{
		Tuple<DIM> tmp;

		for(unsigned int dim = 0; dim < DIM; ++dim)
			tmp[dim] = m_ranges[dim].ntiles();

		return tmp;
	}

	/// Number of elements contained in the orthotope.
	inline size_t
	nelements() const
	{
		return VectorOps<Tuple<DIM>, DIM>::selfProduct(size());
	}
	
	/// Returns the total number of tiles in the orthotope
	inline size_t
	ntiles() const
	{
		return VectorOps<Tuple<DIM>, DIM>::selfProduct(tile_size());
	}

	/// Return the range of elements in orthotope
	inline std::pair<Tuple<DIM>, Tuple<DIM> >
	range() const
	{
		return std::make_pair(low(), high());
	}

	// Number of elements contained in the orthotope.
	inline size_t
	count() const
	{
		return VectorOps<Tuple<DIM>, DIM>::selfProduct(size());
	}

	/// Return the low index of each dimension of the tile at tile_index.
	Tuple<DIM>
	low(const Tuple<DIM>& tile_index) const
	{
		Tuple<DIM> tmp;
		for(unsigned int dim = 0; dim < DIM; ++dim)
			tmp[dim] = m_ranges[dim].low(tile_index[dim]);

		return tmp;
	}

	/// return the high index of each dimension of the tile at tile_index.
	Tuple<DIM>
	high(const Tuple<DIM>& tile_index) const
	{
		Tuple<DIM> tmp;
		for(unsigned int dim = 0; dim < DIM; ++dim)
			tmp[dim] = m_ranges[dim].high(tile_index[dim]);

		return tmp;
	}

	/// return the number of elements in each dimension of the tile at tile_index.
	inline Tuple<DIM>
	size(const Tuple<DIM>& tile_index) const
	{
		return high(tile_index) - low(tile_index);
	}

	inline std::pair<Tuple<DIM>, Tuple<DIM> >
	range(const Tuple<DIM>& tile_index) const
	{
		return std::pair<Tuple<DIM>, Tuple<DIM> >(low(tile_index), high(tile_index));
	}

	/// Number of elements contained in the tile.
	inline size_t
	nelements(const Tuple<DIM>& tile_index) const
	{
		return VectorOps<Tuple<DIM>, DIM>::selfProduct(size(tile_index));
	}
	
	/// Return the tile that contains an element index.
	inline Tuple<DIM>
	tile(const Tuple<DIM>& element_index) const
	{
		Tuple<DIM> tmp;

		for(unsigned int dim = 0; dim < DIM; ++dim)
			tmp[dim] = m_ranges[dim].tile(element_index[dim]);

		return tmp;
	}

    inline Orthotope<DIM>& permute(const Tuple<DIM>& perm) {
      TiledArray::permute(m_ranges, perm);
      return *this;
    }

	// Equality operator
	inline bool
	operator ==(const Orthotope<DIM>& ortho) const 
	{  
	  if(&ortho == this)
	    return true;
      else 
        return std::equal(begin_range(), end_range(), ortho.begin_range());
	}

	// Inequality operator
	inline bool
	operator!= (const Orthotope<DIM>& ortho) const 
	{  
	  return !(operator ==(ortho));
	}
};

template<unsigned int DIM>  std::ostream&
operator <<(std::ostream& out, const Orthotope<DIM>& ortho) {  
	out << "Orthotope<" << DIM << ">(" 
		<< " @= " << &ortho
		<< " low= " << ortho.low() 
		<< " high= " << ortho.high()
		<< " size= " << ortho.size()
		<< " range= [ " << ortho.range().first << "," << ortho.range().second << " )"
		<< " tile_size= " << ortho.tile_size()
		<< " nelements= " << ortho.nelements()
		<< " ntiles= " << ortho.ntiles()
		<< " )";
	return out;
}

}; // end of namespace TiledArray


#endif // ORTHOTOPE_H__INCLUDED
