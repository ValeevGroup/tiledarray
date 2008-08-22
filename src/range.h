#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

/** Range class defines a nonuniformly-tiled one-dimensional range.
    The tiling data is constructed with and stored in an array with
    the format {a, b, c, ...}, where 0 <= a < b < c < ... Each tile is
    defined by [a,b), [b,c), ... The number of tiles in the range is
    defined as one less than the number of elements in the array.
  */
class Range
{
private:

    /// Represents a tile in a Range
	class RangeTile
	{
	    /// the owner
		const Range& m_range;
		/// index in the owner
		size_t m_current;

	public:
		RangeTile(const Range& rng, const size_t current) :
			m_range(rng),
			m_current(current)
		{}

		/// Returns the index of this tile in the containing Range
		inline size_t
		index() const
		{
			assert(this->m_current < this->m_range.tile_size());
			return this->m_current + this->m_range.tile_low();
		}

		inline size_t
		low() const
			{return this->m_range.low(this->index());}

		inline size_t
		high() const
			{return this->m_range.high(this->index());}

		inline size_t
		size() const
			{return this->m_range.size(this->index());}

		inline std::pair<size_t,size_t>
		range() const
			{return this->m_range.range(this->index());}

		inline void
		set_current(const size_t current)
			{this->m_current = current;}
	};

	/// Iterator spec for RangeIterator class.
	class RangeIteratorSpec
	{
	public:
		typedef size_t						iterator_type;
		typedef Range						collection_type;
		typedef std::input_iterator_tag		iterator_category;  
		typedef RangeTile					value;
		typedef value*						pointer;
		typedef const value*				const_pointer;
		typedef value&						reference;
		typedef const value&				const_reference;
	};

	/** RangeIterator is an InputIterator that iterates over the tiles
	    of a Range.
	  */
	class RangeIterator : public Iterator<RangeIteratorSpec>
	{
	public:
		// Iterator typedef's
		typedef Iterator<RangeIteratorSpec>::iterator_type		iterator_type;
		typedef Iterator<RangeIteratorSpec>::collection_type	collection_type;
		typedef Iterator<RangeIteratorSpec>::iterator_category	iterator_catagory;
		typedef Iterator<RangeIteratorSpec>::reference			reference;
		typedef Iterator<RangeIteratorSpec>::const_reference	const_reference;
		typedef Iterator<RangeIteratorSpec>::pointer			pointer;
		typedef Iterator<RangeIteratorSpec>::const_pointer		const_pointer;
		typedef Iterator<RangeIteratorSpec>::value				value;
		typedef Iterator<RangeIteratorSpec>::difference_type	difference_type;

	private:

		const collection_type& m_coll;	// Reference to the collection that will be iterated over
		value m_value;					// used to return

	public:

		/// Main constructor function
		RangeIterator(const collection_type& coll, const iterator_type& cur = 0) : 
			Iterator<RangeIteratorSpec>(cur), 
			m_coll(coll),
			m_value(coll, cur)
		{
			assert(cur <= coll.tile_size());
		}

		/// Copy constructor (required by all iterators)
		RangeIterator(const RangeIterator& it) :
			Iterator<RangeIteratorSpec>(it.m_current), 
			m_coll(it.m_coll),
			m_value(it.m_coll, it.m_current)
		{}

		/// Prefix increment (required by all iterators)
		RangeIterator&
		operator ++() 
		{     
			this->advance();
			return *this;
		}


		/// Postfix increment (required by all iterators)
		RangeIterator
		operator ++(int) 
		{
			assert(this->m_current < this->m_coll.tile_high());
			RangeIterator tmp(*this);
			this->advance();
			return tmp;
		}

		/// Equality operator (required by input iterators)
		inline bool
		operator ==(const RangeIterator& it) const
		{
			return this->m_current == it.m_current;
		}

		/// Inequality operator (required by input iterators)
		inline bool
		operator !=(const RangeIterator& it) const
		{
			return ! (this->operator ==(it));
		}

		/// Dereference operator (required by input iterators)
		inline const_reference
		operator *() const 
		{
			return this->m_value;
		}

		/// Dereference operator (required by input iterators)
		inline const_pointer
		operator ->() const
		{
			return &(this->m_value);
		}

		/** This is for debugging only. Not done in an overload of operator<<
		    because it seems that gcc 3.4 does not manage inner class declarations of 
		    template classes correctly.
		  */
		inline char
		print(std::ostream& ost) const
		{
			ost << "Range::iterator("
				<< " current= " << this->m_current
				<< ", index= " << this->m_value.index()
				<< ", low= " << this->m_value.low()
				<< ", high= " << this->m_value.high()
				<< ", size= " << this->m_value.size()
				<< ", range= [" << this->m_value.low() << "," << this->m_value.high() << ") )";
			return '\0';
		}

	private:

		inline void
		advance(unsigned int n = 1) 
		{
			assert((this->m_current + n) <= this->m_coll.tile_size());

			this->m_current += n;
			this->m_value.set_current(this->m_current);
		}
		
	}; // end of RangeIterator
	
	/// Range data
	std::vector<size_t> m_ranges;
	std::vector<size_t> m_tile_index_map;

	// Initialize the m_tile_index_map
	void
	init_tile_index(const size_t low_tile)
	{
		size_t t = 0;
		for(; t < this->m_ranges.size() - 1; ++t)
			for(size_t i = this->m_ranges[t]; i < this->m_ranges[t + 1]; ++i)
				this->m_tile_index_map[i - this->m_ranges[0]] = t + low_tile;

		this->m_tile_index_map[this->m_tile_index_map.size() - 1] = t + low_tile;
	}

	// Validates the data in rng meets the requirements of Range.
	static bool
	valid(std::vector<size_t>& rng)
	{
		// Check minimum number of elements
		if(rng.size() < 2)
			return false;
		
		// Verify the requirement that a < b < c < ...
		for(size_t index = 1; index < rng.size(); ++index)
			if(rng[index - 1] >= rng[index])
				return false;
		
		return true;
	}

public:
	typedef RangeIterator	iterator;
	
	// Default constructor, range with a single tile [0,1)
	Range() :
		m_ranges(2, 0),
		m_tile_index_map(2, 0)
	{
		m_ranges[1] = 1;
		m_tile_index_map[1] = 1;
	}

	// Constructs range from a vector
	Range(std::vector<size_t> ranges, const size_t low_tile = 0) :
		m_ranges(ranges),
		m_tile_index_map(ranges[ranges.size() - 1] - ranges[0] + 1, low_tile)
	{
		assert(Range::valid(this->m_ranges));
		this->init_tile_index(low_tile);
	}

	// Construct range from a C-style array, ranges must have at least
	// tiles + 1 elements.
	Range(size_t* ranges, size_t tiles, const size_t low_tile = 0) :
		m_ranges(ranges, ranges + tiles + 1),
		m_tile_index_map(ranges[tiles] - ranges[0] + 1, low_tile)
	{
		assert(Range::valid(this->m_ranges));
		this->init_tile_index(low_tile);
	}

	// Copy constructor
	Range(const Range& rng) :
		m_ranges(rng.m_ranges),
		m_tile_index_map(rng.m_tile_index_map)
	{}

	// Assignment operator
	inline Range&
	operator =(const Range& rng)
	{
		this->m_ranges = rng.m_ranges;
		this->m_tile_index_map = rng.m_tile_index_map;

		return *this;
	}

	// Equality operator
	inline bool
	operator ==(const Range& rng) const
	{
		// Check tiling range
		if(this->tile_low() != rng.tile_low() && this->tile_high() != rng.tile_high())
			return false;

		// Check elements range
		if(this->low() != rng.low() && this->high() != rng.high())
			return false;

		// Check tile boundaries
		return std::equal(this->m_ranges.begin(), this->m_ranges.end(), rng.m_ranges.begin());
	}

	// Inequality operator
	inline bool
	operator !=(const Range& rng) const
	{
		return !(this->operator ==(rng));
	}

	// Return tile index associated with element_index
	inline size_t
	tile(const size_t element_index) const
	{
		assert(element_index >= this->low());
		assert(element_index < this->high());

		return this->m_tile_index_map[element_index - this->low()];
	}

	// Return the low tile index
	inline size_t
	tile_low() const
	{
		return this->m_tile_index_map[0];
	}

	// Return the high tile index
	inline size_t
	tile_high() const
	{
		return this->m_tile_index_map[m_tile_index_map.size() - 1];
	}

	// Returns the number of tiles in the range.
	inline size_t
	tile_size() const
	{
		return this->tile_high() - this->tile_low();
	}

	inline std::pair<size_t, size_t>
	tile_range() const
	{
		return std::pair<size_t,size_t>(this->tile_low(), this->tile_high());
	}

	// Returns the low index of the tile
	inline size_t
	low(const size_t tile_index) const
	{
		assert(tile_index >= this->tile_low());
		assert(tile_index < this->tile_high());
		return this->m_ranges[tile_index - this->tile_low()];
	}

	// Returns the high index of the tile.
	inline size_t
	high(const size_t tile_index) const
	{
		assert(tile_index >= this->tile_low());
		assert(tile_index < this->tile_high());
		return this->m_ranges[tile_index - this->tile_low() + 1];
	}

	// Returns the number of elements in a tile.
	inline size_t
	size(const size_t tile_index) const
	{
		assert(tile_index >= this->tile_low());
		assert(tile_index < this->tile_high());
		return (this->high(tile_index) - this->low(tile_index));
	}

	// Returns a pair that contains low and high of the tile.
	inline std::pair<size_t, size_t>
	range(const size_t tile_index) const
	{
		assert(tile_index >= this->tile_low());
		assert(tile_index < this->tile_high());
		return std::pair<size_t, size_t>(this->low(tile_index), this->high(tile_index));
	}

	// Returns the low element index of the range.
	inline size_t
	low() const
	{
		return this->m_ranges[0];
	}

	// Returns the high element index of the range.
	inline size_t
	high() const
	{
		return this->m_ranges[this->m_ranges.size() - 1];
	}

	// Returns the number of elements in the range.
	inline size_t
	size() const
	{
		return (this->high() - this->low());
	}

	inline std::pair<size_t, size_t>
	range() const
	{
		return std::pair<size_t, size_t>(this->low(), this->high());
	}

	// Returns an iterator to the first tile in the range.
	inline iterator
	begin() const
	{
		return iterator(*this, 0);
	}

	// Returns an iterator to the end of the range.
	inline iterator
	end() const
	{
		return iterator(*this, this->tile_size());
	}

	inline void
	relocate(const size_t origin)
	{
		const size_t dir = origin - this->low();
		for(std::vector<size_t>::iterator it = this->m_ranges.begin(); it != this->m_ranges.end(); ++it)
			*it += dir;
	}

	inline void
	relocate_tile(const size_t origin)
	{
		const size_t dir = origin - this->tile_low();
		for(std::vector<size_t>::iterator it = this->m_tile_index_map.begin(); it != this->m_tile_index_map.end(); ++it)
			*it += dir;
	}

}; // Range

inline std::ostream&
operator <<(std::ostream& out, const Range& rng)
{  
	out << "Range(" 
		<< " @=" << &rng
		<< ", low=" << rng.low() 
		<< ", high=" << rng.high()
		<< ", size=" << rng.size() 
		<< ", range= [" << rng.range().first << "," << rng.range().second << ")"
		<< ", tile_low= " << rng.tile_low()
		<< ", tile_high= " << rng.tile_high()
		<< ", tile_size= " << rng.tile_size()
		<< ", tile_range= [" << rng.tile_range().first << "," << rng.tile_range().second << ") )";
	return out;
}

#endif // RANGE_H__INCLUDED
