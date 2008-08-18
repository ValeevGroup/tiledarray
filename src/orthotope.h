#ifndef ORTHOTOPE_H__INCLUDED
#define ORTHOTOPE_H__INCLUDED

template<unsigned int DIM>
class Orthotope
{
	std::vector<Range> m_ranges;

public:
	// iterator typedefs
	typedef Range::iterator						range_iterator;
	typedef std::vector<Range>::const_iterator	iterator;
	typedef std::pair<Tuple<DIM>, Tuple<DIM> >	tile;

	// Default constructor
	Orthotope() : 
		m_ranges(DIM, Range())
	{}

	// Constructed with an array of ranges
	Orthotope(const Range* ranges) :
		m_ranges(ranges, ranges + DIM - 1)
	{
#if TA_DLEVEL >= 1
		for(int index = 0; index < DIM - 1; ++index)
			assert(ranges + index);
#endif
	}

	// Constructor from a vector of ranges
	Orthotope(const std::vector<Range>& ranges) :
		m_ranges(ranges)
	{}

	// Copy constructor
	Orthotope(const Orthotope& ortho) :
		m_ranges(ortho.m_ranges)
	{}

	// Iterator factory functions
	iterator
	begin() const
	{
		return this->m_ranges.begin();
	}

	iterator
	end() const
	{
		return this->m_ranges.end();
	}

	range_iterator
	begin(const unsigned int index) const
	{
		assert(index < DIM);
		return this->m_ranges[index].begin();
	}

	range_iterator
	end(const unsigned int index) const
	{
		assert(index < DIM);
		return this->m_ranges[index].end();
	}

	// Accessor functions

	// Returns a tuple with the number of tiles in each range
	Tuple<DIM>
	count() const
	{
		Tuple<DIM> tmp;
		for(int index = 0; index < DIM; ++index)
			tmp[index] = this->m_ranges[index].count();

		return tmp;
	}

	// return tuple with lower bound for each dimention.
	Tuple<DIM>
	low() const
	{
		Tuple<DIM> tmp;
		for(int index = 0; index < DIM; ++index)
			tmp[index] = this->m_ranges[index].low();

		return tmp;
	}

	// Return tuple with the upper bound for each dimension.
	Tuple<DIM>
	high() const
	{
		Tuple<DIM> tmp;
		for(int index = 0; index < DIM; ++index)
			tmp[index] = this->m_ranges[index].high();

		return tmp;
	}

	// Returns a tuple with the number of elements in each dimention
	Tuple<DIM>
	size() const
	{
		Tuple<DIM> tmp;
		for(int index = 0; index < DIM; ++index)
			tmp[index] = this->m_ranges[index].size();

		return tmp;
	}

	// Returnt the low index of each dimension of the tile at tile_index.
	Tuple<DIM>
	low(const Tuple<DIM>& tile_index)
	{
		assert(tile_index < this->count());
		Tuple<DIM> tmp;
		for(int index = 0; index < DIM; ++index)
			tmp[index] = this->m_ranges[index].low(tile_index[index]);

		return tmp;
	}

	// return the high index of each dimension of the tile at tile_index.
	Tuple<DIM>
	high(const Tuple<DIM>& tile_index)
	{
		assert(tile_index < this->count());
		Tuple<DIM> tmp;
		for(int index = 0; index < DIM; ++index)
			tmp[index] = this->m_ranges[index].high(tile_index[index]);

		return tmp;
	}

	// return the number of elements in each dimension of the tile at tile_index.
	Tuple<DIM>
	size(const Tuple<DIM>& tile_index)
	{
		assert(tile_index < this->count());
		Tuple<DIM> tmp;
		for(int index = 0; index < DIM; ++index)
			tmp[index] = this->m_ranges[index].size(tile_index[index]);

		return tmp;
	}

	// Return a tile at tile_index.
	tile
	get_tile(const Tuple<DIM>& tile_index)
	{
		assert(tile_index < this->count());
		return tile(this->low(tile_index), this->high(tile_index));
	}

};
#endif // ORTHOTOPE_H__INCLUDED
