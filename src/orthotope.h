#ifndef ORTHOTOPE_H__INCLUDED
#define ORTHOTOPE_H__INCLUDED

// Orthotope stores tile information of a rectilinear array. 
template<unsigned int DIM>
class Orthotope
{
	std::vector<Range> m_ranges;	// Vector of range data for each dimention
	Tuple<DIM> m_high;				// highest index in each dimension (not included)
	Tuple<DIM> m_low;				// lowest index in each dimension (included)
	Tuple<DIM> m_size;				// Number of elements in each direction
	Tuple<DIM> m_linear_step;		// the number of elements to skip in the linearized 
									// array until the respective dimension increments

	void
	init()
	{
		this->m_linear_step[DIM - 1] = 1;
		for (int dim = DIM - 1; dim > 0; --dim)
			this->m_linear_step[dim - 1] = (this->m_high[dim] - this->m_low[dim]) * this->m_linear_step[dim];
	}

public:
	// iterator typedefs
	typedef Range::iterator						range_iterator;
	typedef std::vector<Range>::const_iterator	const_iterator;
	typedef std::vector<Range>::iterator		iterator;
// 	typedef std::pair<Tuple<DIM>, Tuple<DIM> >	tile;

	// Default constructor
	Orthotope() : 
		m_ranges(DIM, Range())
	{}

	// Constructed with an array of ranges
	Orthotope(const Range* ranges) :
		m_ranges(ranges, ranges + DIM - 1),
		m_high(),
		m_low(),
		m_size()
	{
		for(int dim = 0; dim < DIM; ++dim)
		{
			assert(ranges + dim);
			this->m_low[dim] = this->m_ranges[dim].low();
			this->m_high[dim] = this->m_ranges[dim].high();
			this->m_size[dim] = this->m_ranges[dim].size();
		}

		this->init();
	}

	// Construct from a set arrays
	Orthotope(const size_t* ranges[DIM], const size_t tiles[DIM]) : 
		m_ranges(DIM, Range()),
		m_high(),
		m_low(),
		m_size()
	{
		for(int dim = 0; dim < DIM; ++dim)
		{
			assert(ranges[dim]);

			this->m_low[dim] = this->m_ranges[dim].low();
			this->m_high[dim] = this->m_ranges[dim].high();
			this->m_size[dim] = this->m_ranges[dim].size();
			this->m_ranges[dim] = Range(ranges[dim], tiles[dim]);
		}
	}

	// Constructor from a vector of ranges
	Orthotope(const std::vector<Range>& ranges) :
		m_ranges(ranges),
		m_high(),
		m_low(),
		m_size()
	{
		assert(ranges.size() == DIM);
		for(int dim = 0; dim < DIM; ++dim)
		{
			this->m_low[dim] = this->m_ranges[dim].low();
			this->m_high[dim] = this->m_ranges[dim].high();
			this->m_size[dim] = this->m_ranges[dim].size();
		}
	}

	// Copy constructor
	Orthotope(const Orthotope& ortho) :
		m_ranges(ortho.m_ranges)
	{}

	// Iterator factory functions

	// Returns an iterator pointing to the first range.
	inline iterator
	begin()
	{
		return this->m_ranges.begin();
	}

	// Return an iterator pointing one past the last dimension.
	inline iterator
	end()
	{
		return this->m_ranges.end();
	}

	// Returns an iterator pointing to the first range.
	inline const_iterator
	begin() const
	{
		return this->m_ranges.begin();
	}

	// Return an iterator pointing one past the last dimension.
	inline const_iterator
	end() const
	{
		return this->m_ranges.end();
	}

	// Returns an iterator pointing to the element in the dim range.
	inline range_iterator
	begin(const unsigned int dim) const
	{
		assert(dim < DIM);
		return this->m_ranges[dim].begin();
	}

	inline range_iterator
	end(const unsigned int dim) const
	{
		assert(dim < DIM);
		return this->m_ranges[dim].end();
	}

	inline Range&
	operator [](const unsigned int dim)
	{
		assert(dim < DIM);
		return this->m_ranges[dim];
	}

	// Accessor functions

	//
	// Test to see if tuple is contained within the shape. No check is done
	// to see if the data is present.
	// 
	// Returns true if element_index is within the bounds of the shape.
	inline bool
	contains(const Tuple<DIM>& element_index) const 
	{
		return (element_index >= m_low) && (element_index < m_high);
	}

	inline const Tuple<DIM>&
	linear_step() const
		{return m_linear_step;}

	// Number of elements contained in the orthotope.
	inline size_t
	count() const
	{
		return VectorOps<Tuple<DIM>, DIM>::selfProduct(this->m_high - this->m_low);
	}

	// return tuple with lower bound for each dimension.
	inline Tuple<DIM>
	low() const
	{
		return this->m_low;
	}

	// Return tuple with the upper bound for each dimension.
	inline Tuple<DIM>
	high() const
	{
		return this->m_high;
	}

	// Returns a tuple with the number of elements in each dimension
	Tuple<DIM>
	size() const
	{
		return this->m_size;
	}

	// Return the low index of each dimension of the tile at tile_index.
	Tuple<DIM>
	low(const Tuple<DIM>& tile_index) const
	{
		assert(tile_index < this->count());
		Tuple<DIM> tmp;
		for(int dim = 0; dim < DIM; ++dim)
			tmp[dim] = this->m_ranges[dim].low(tile_index[dim]);

		return tmp;
	}

	// return the high index of each dimension of the tile at tile_index.
	Tuple<DIM>
	high(const Tuple<DIM>& tile_index) const
	{
		assert(tile_index < this->count());
		Tuple<DIM> tmp;
		for(int dim = 0; dim < DIM; ++dim)
			tmp[dim] = this->m_ranges[dim].high(tile_index[dim]);

		return tmp;
	}

	// return the number of elements in each dimension of the tile at tile_index.
	Tuple<DIM>
	size(const Tuple<DIM>& tile_index) const
	{
		assert(tile_index < this->count());
		Tuple<DIM> tmp;
		for(int dim = 0; dim < DIM; ++dim)
			tmp[dim] = this->m_ranges[dim].size(tile_index[dim]);

		return tmp;
	}

	// Number of elements contained in the orthotope.
	inline size_t
	count(const Tuple<DIM>& tile_index) const
	{
		return VectorOps<Tuple<DIM>, DIM>::selfProduct(this->high(tile_index) - this->low(tile_index));
	}

	// Return the tile that contains an element index.
	inline Tuple<DIM>
	tile(const Tuple<DIM>& element_index) const
	{
		assert(this->contains(element_index));
		Tuple<DIM> tmp;

		for(unsigned int dim = 0; dim < DIM; ++dim)
			tmp[dim] = this->m_ranges.tile(element_index[dim]);

		return tmp;
	}

	// Return a tile at tile_index.
	inline std::pair<Tuple<DIM>, Tuple<DIM> >
	tile_range(const Tuple<DIM>& tile_index) const
	{
		assert(tile_index < this->count());
		return std::pair<Tuple<DIM>, Tuple<DIM> >(this->low(tile_index), this->high(tile_index));
	}


	// Relocates the coordinate system represented by 
	// this shape. The underlying mapping to linearized 
	// representation remains unchanged.
	//
	// @param origin    new low coordinate of the shape
	inline void
	relocate(const Tuple<DIM>& origin) 
	{
		Tuple<DIM> dir(origin - this->m_low);
		this->m_low = origin;
		this->m_high += dir;
		for(unsigned int dim = 0; dim < DIM; ++dim)
			this->m_ranges[dim].relocate(dir[dim]);
	}

	inline void
	relocate_tile(const Tuple<DIM> origin)
	{
		Tuple<DIM> dir(origin - this->m_low);
		for(unsigned int dim = 0; dim < DIM; ++dim)
			this->m_ranges[dim].relocate_tile(dir[dim]);
		
	}

	// Equality operator
	inline bool
	operator ==(const Orthotope<DIM>& ortho) const 
	{  
		if(&ortho == this)
			return true;
		else 
			return std::equal(this->begin(), this->end(), ortho.begin());
	}

	// Inequality operator
	inline bool
	operator!= (const Orthotope<DIM>& ortho) const 
	{  
		return !(this->operator ==(ortho));
	}
};

template<int DIM, class Predicate>  std::ostream&
operator <<(std::ostream& out, const Orthotope<DIM>& ortho) {  
	out << "Orthotope<" << DIM << ">(" 
		<< " @=" << &ortho
		<< " low=" << ortho.low() 
		<< " high=" << ortho.high()
		<< " size=" << ortho.size() 
		<< " linearStep=" << ortho.linear_step()  << ")";
	return out;
}

#endif // ORTHOTOPE_H__INCLUDED
