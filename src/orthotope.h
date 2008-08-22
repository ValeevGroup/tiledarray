#ifndef ORTHOTOPE_H__INCLUDED
#define ORTHOTOPE_H__INCLUDED

#include <range.h>
#include <tuple.h>

namespace TiledArray {

// Orthotope stores tile information of a rectilinear array. 
template<unsigned int DIM>
class Orthotope
{
	std::vector<Range> m_ranges;	// Vector of range data for each dimention

public:
	// iterator typedefs
	typedef Range::const_iterator				const_range_iterator;
	typedef std::vector<Range>::const_iterator	const_iterator;
	typedef std::vector<Range>::iterator		iterator;
// 	typedef std::pair<Tuple<DIM>, Tuple<DIM> >	tile;

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


	// Constructor from a vector of ranges
	Orthotope(const std::vector<Range>& ranges) :
		m_ranges(ranges)
	{
		assert(ranges.size() == DIM);

	}

	// Returns an iterator pointing to the first range.
	inline iterator
	begin()
	{
		return m_ranges.begin();
	}

	// Return an iterator pointing one past the last dimension.
	inline iterator
	end()
	{
		return m_ranges.end();
	}

	// Returns an iterator pointing to the first range.
	inline const_iterator
	begin() const
	{
		return m_ranges.begin();
	}



	// Return an iterator pointing one past the last dimension.
	inline const_iterator
	end() const
	{
		return m_ranges.end();
	}

	// Returns an iterator pointing to the element in the dim range.
	inline const_range_iterator
	begin(const unsigned int dim) const
	{
		assert(dim < DIM);
		return m_ranges[dim].begin();
	}

	inline const_range_iterator
	end(const unsigned int dim) const
	{
		assert(dim < DIM);
		return m_ranges[dim].end();
	}

	// Accessor functions

	// change one of the dimension ranges.
	inline void
	set_dim(const unsigned int dim, const Range& rng)
	{
		assert(dim < DIM);
		m_ranges[dim] = rng;
	}

	// Return a reference to one of the dimension ranges.
	inline const Range&
	get_dim(const unsigned int dim) const
	{
		assert(dim < DIM);
		return m_ranges[dim];
	}

	// Return a reference to one of the dimension ranges.
	inline const Range&
	operator [](const unsigned int dim)
	{
		return get_dim(dim);
	}

	// Returns true if element_index is within the bounds of the orthotope.
	inline bool
	contains(const Tuple<DIM>& element_index) const 
		{return (element_index >= low()) && (element_index < high());}

	// return tuple with lower bound for each dimension.
	inline Tuple<DIM>
	low() const
	{
		Tuple<DIM> tmp;
		for(unsigned int dim = 0; dim < DIM; ++dim)
			tmp[dim] = m_ranges[dim].low();

		return tmp;
	}

	// Return tuple with the upper bound for each dimension.
	inline Tuple<DIM>
	high() const
	{
		Tuple<DIM> tmp;
		for(unsigned int dim = 0; dim < DIM; ++dim)
			tmp[dim] = m_ranges[dim].high();

		return tmp;
	}

	// Returns a tuple with the number of elements in each dimension
	inline Tuple<DIM>
	size() const
	{
		return high() - low();
	}

	// Return the range of elements in orthotope
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

	// Return the low index of each dimension of the tile at tile_index.
	Tuple<DIM>
	low(const Tuple<DIM>& tile_index) const
	{
		// TODO: add asserts back in once functions have been added to the class.
//		assert(tile_index >= tile_low());
//		assert(tile_index < tile_high());
		Tuple<DIM> tmp;
		for(int dim = 0; dim < DIM; ++dim)
			tmp[dim] = m_ranges[dim].low(tile_index[dim]);

		return tmp;
	}

	// return the high index of each dimension of the tile at tile_index.
	Tuple<DIM>
	high(const Tuple<DIM>& tile_index) const
	{
		// TODO: add asserts back in once the functions have been added to the class.
//		assert(tile_index < tile_high());
		Tuple<DIM> tmp;
		for(int dim = 0; dim < DIM; ++dim)
			tmp[dim] = m_ranges[dim].high(tile_index[dim]);

		return tmp;
	}

	// return the number of elements in each dimension of the tile at tile_index.
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

	// Number of elements contained in the orthotope.
	inline size_t
	count(const Tuple<DIM>& tile_index) const
	{
		return VectorOps<Tuple<DIM>, DIM>::selfProduct(size(tile_index));
	}

	// Return the tile that contains an element index.
	inline Tuple<DIM>
	tile(const Tuple<DIM>& element_index) const
	{
		assert(contains(element_index));
		Tuple<DIM> tmp;

		for(unsigned int dim = 0; dim < DIM; ++dim)
			tmp[dim] = m_ranges[dim].tile(element_index[dim]);

		return tmp;
	}

	// Return a tile at tile_index.
	inline std::pair<Tuple<DIM>, Tuple<DIM> >
	tile_range(const Tuple<DIM>& tile_index) const
	{
		assert(tile_index < count());
		return std::pair<Tuple<DIM>, Tuple<DIM> >(low(tile_index), high(tile_index));
	}

	// Equality operator
	inline bool
	operator ==(const Orthotope<DIM>& ortho) const 
	{  
		if(&ortho == this)
			return true;
		else 
			return std::equal(begin(), end(), ortho.begin());
	}

	// Inequality operator
	inline bool
	operator!= (const Orthotope<DIM>& ortho) const 
	{  
		return !(operator ==(ortho));
	}
};

template<int DIM, class Predicate>  std::ostream&
operator <<(std::ostream& out, const Orthotope<DIM>& ortho) {  
	out << "Orthotope<" << DIM << ">(" 
		<< " @= " << &ortho
		<< " low= " << ortho.low() 
		<< " high= " << ortho.high()
		<< " size= " << ortho.size()
		<< " range= [" << ortho.range().first << "," << ortho.range().second << ")"
		<< " linearStep= " << ortho.linear_step()
		<< " tile_low= " << ortho.tile_low()
		<< " tile_high= " << ortho.tile_high()
		<< " tile_size= " << ortho.tile_size()
		<< " tile_range= [" << ortho.tile_range().first << "," << ortho.tile_range().socond << ")" 
		<< " )";
	return out;
}

}; // end of namespace TiledArray


#endif // ORTHOTOPE_H__INCLUDED
