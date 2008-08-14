/**
 * IBM SOFTWARE DISCLAIMER 
 *
 * This file is part of htalib, a library for hierarchically tiled arrays.
 * Copyright (c) 2005 IBM Corporation.
 *
 * Permission to use, copy, modify and distribute this software for
 * any noncommercial purpose and without fee is hereby granted,
 * provided that this copyright and permission notice appear on all
 * copies of the software. The name of the IBM Corporation may not be
 * used in any advertising or publicity pertaining to the use of the
 * software. IBM makes no warranty or representations about the
 * suitability of the software for any purpose.  It is provided "AS
 * IS" without any express or implied warranty, including the implied
 * warranties of merchantability, fitness for a particular purpose and
 * non-infringement.  IBM shall not be liable for any direct,
 * indirect, special or consequential damages resulting from the loss
 * of use, data or projects, whether in an action of contract or tort,
 * arising out of or in connection with the use or performance of this
 * software.  
 */

/*
 * Version: $Id: Shape.h,v 1.159 2007/02/14 16:09:58 bikshand Exp $
 * Authors: Ganesh Bikshandi, Christoph von Praun
 */

#ifndef SHAPE_H__INCLUDED
#define SHAPE_H__INCLUDED

/**
 * Shape class that defines a multi-diimensional coordinate system and 
 * its mapping to an underlying dense, linearized representation.
 * The mapping to ordinals assumes that 0 is the least signifcant 
 * dimension.
 *
 * Instances of Shape are immutable.
 */

template <unsigned int DIM>
class AbstractShape
{
public:
	
	virtual size_t
	ord(const Tuple<DIM>& index) const = 0;
	
	virtual Tuple<DIM>
	coord(unsigned int index) const = 0;
	
	virtual bool
	included(const Tuple<DIM>& tile_idx) const = 0;

	virtual inline const AbstractShape<DIM>*
	clone() const
	{
		assert (false);
		return NULL;
	}

};


template <unsigned int DIM, class PREDICATE>
class Shape : public AbstractShape<DIM>
{
private:
	
	// iterator class predeclariation
	class ShapeIteratorSpec;
	class ShapeIterator;
	
public:
	// Shape typedef's
	typedef	PREDICATE											predicate;
	typedef boost::filter_iterator<predicate, ShapeIterator>	iterator;
	
protected:
	Tuple<DIM> m_high;					// highest index in each dimension (not included)
	Tuple<DIM> m_low;					// lowest index in each dimension (included)
	Tuple<DIM> m_size;					// Number of elements in each direction
	Tuple<DIM> m_linear_step;			// the number of elements to skip in the linearized 
  										// array until the respective dimension increments
	boost::shared_ptr<PREDICATE> m_pred;// Shared pointer to predicate object, which defines
    									// which elements are present.
    
public:

	/** 
	 * Default constructor
	 */
	Shape() :
		m_high(1),
		m_low(),
		m_size(1),
		m_linear_step(),
		m_pred(new predicate)
	{
		this->init();
	}

	/** 
	 * Convenience constructor.
	 *
	 * @param  t upper coordinates of the shape - not included
	 * @return shape with lower bound (0,0,..), step (1,1,..).
	 */
	Shape(const Tuple<DIM>& tup_high) :
		m_high(tup_high),
		m_low(),
		m_size(tup_high),
		m_linear_step(),
		m_pred(new predicate)
	{
		this->init();
	}

	/** 
	 * Convenience constructor.
	 *
	 * @param  begin lower coordinates of shape
	 * @param  end   upper coordinates of the shape - not included
	 * @return       shape with step (1,1,..).
	 */
	Shape(const Tuple<DIM>& low, const Tuple<DIM> &high) :
		m_high(high),
		m_low(low),
		m_size(high - low),
		m_linear_step(),
		m_pred(new predicate)
	{
		this->init();
	}

	/**
	 * Copy constructor 
	 */
	Shape(const Shape<DIM, PREDICATE>& s) : 
		m_high(s.m_high), 
		m_low(s.m_low), 
		m_size(s.m_size), 
		m_linear_step(s.m_linear_step),
		m_pred(s.m_pred)
	{ 
		TA_DEBUG(2, "Shape::<init> this=" << this << " m_linearStep=" << this->m_linear_step << " high:" << this->m_high);
	}


	virtual
	~Shape()
	{}

private:

	/**
	 * Initializer method
	 * @param t  t[0] is the most significant dimension.
	 */
	void
	init() 
	{
		// memorize values that aid the linearization function
		// DIM-1 is the *least* significant dimension in row-major 
		// (row is in dimension 0)

		// compute cumulative m_linear_step
		this->m_linear_step[DIM - 1] = 1;
		for (int i = DIM - 1; i > 0; --i)
			this->m_linear_step[i - 1] = (this->m_high[i] - this->m_low[i]) * this->m_linear_step[i];
	}

public:

	virtual bool 
	included(const Tuple<DIM>& index) const
	{
		return this->contains(index) && this->m_pred->included(index);
	}

	virtual const Shape<DIM, PREDICATE>*
	clone() const
	{
		assert (false);
		return NULL;
	}

	virtual const Shape<DIM, PREDICATE>&
	operator [](const Tuple<DIM>& i) const
	{
		return *this;
	}

	virtual const Shape<DIM, PREDICATE>&
	shape_at(int i) const
	{
		return *this;
	}
  

	/**
	 * Test to see if tuple is contained within the shape. No check is done
	 * to see if the data is present.
	 * 
	 * @return	true if tup is within the bounds of the shape.
	 */
	inline bool
	contains(const Tuple<DIM>& tup) const 
	{
		return (tup >= m_low) && (tup < m_high);
	}

	/**
	 * Test to see if shape s is contained within the shape. No check is done
	 * to see if the data is present.
	 * 
	 * @return	true if tup is within the bounds of the shape.
	 */
	inline bool
	contains(const Shape<DIM, PREDICATE>* const s) const 
	{
		// current limitation 
		return VectorOps<Tuple<DIM>, DIM>::greatereq(s->m_low, this->m_low) && 
			VectorOps<Tuple<DIM>, DIM>::less(s->m_high, this->m_high);
	}

	/**
	 * @param  s1    shape
	 * @param  s2    shape
	 * @return       true if both shapes are conformable.
	 */
	static inline bool
	conformable(const Shape<DIM, PREDICATE>& s1, const Shape<DIM, PREDICATE>& s2) 
	{
		if(&s1 == &s2)
			return true;
		else 
			return (s1.m_size == s2.m_size);
	}

	/**
	 * @param  s1    shape
	 * @param  s2    shape
	 * @return       true if both shapes can be accessed with the same non-linear iterator.
	 *               the property is reflexive and transitive.
	 */
	static inline bool
	iterator_conformable(const Shape<DIM, PREDICATE>& s1, const Shape<DIM, PREDICATE>& s2) 
	{
		if (&s1 == &s2)
			return true;
		else 
			return (s1.m_linearStep == s2.m_linearStep);
	}



	/**
	 * Equality operator
	 */
	inline bool
	operator ==(const Shape<DIM, PREDICATE>& other) const 
	{  
		if(&other == this)
			return true;
		else 
			return (other.m_low == this->m_low &&
					other.m_high == this->m_high && 
					other.m_step == this->m_step);
	}

	/**
	 * Inequality operator
	 */
	inline bool
	operator!= (const Shape<DIM, PREDICATE>& other) const 
	{  
		return ! (this->operator==(other));
	}
  
	// Constant Accessors

	inline const Tuple<DIM>&
	linear_step() const
		{return m_linear_step;}

	inline const Tuple<DIM>&
	size() const
		{return this->m_size;}

	inline const Tuple<DIM>&
	lows() const
		{return this->m_low;}
  
	inline const Tuple<DIM>&
	highs() const
		{return this->m_high;}

	inline const unsigned int
	count() const
		{return VectorOps<Tuple<DIM>, DIM>::selfProduct(m_size);}

/*
	inline Tuple<DIM>& 
	linear_step()
		{return this->m_linear_step;}

	inline Tuple<DIM>& 
	step()
		{return this->m_step;}

	inline Tuple<DIM>&
	high()
		{return this->m_high;}

	inline Tuple<DIM>&
	low()
		{return m_low;}
*/
	
	/**
	 * Ordinal value of Tuple. Ordinal value does not include offset
	 * and does not consider step. If the shape starts at (5, 3), 
	 * then (5, 4) has ord 1 on row-major. 
	 * The ordinal value of low() is always 0.
	 * The ordinal value of high() is always <= linearCard().
	 */
	virtual inline size_t
	ord(const Tuple<DIM>& coord) const
	{
		assert(coord >= m_low && coord <= m_high);
		return static_cast<size_t>(VectorOps<Tuple<DIM>, DIM>::dotProduct((coord - m_low), m_linear_step));
	}
   
	/* GB: coord is local coordinate */
	virtual inline Tuple<DIM>
	coord(unsigned int index) const 
	{
		assert(index >= 0 && index < this->count());

		Tuple<DIM> ret;

		// start with most significant dimension 0.
		for(unsigned int d = 0; index > 0 && d < DIM; ++d)
		{
			ret[d] = index / this->m_linear_step[d];
			index -= ret[d] * this->m_linear_step[d];
		}

		assert(index == 0); // Sanity check

		// Offset the value so it is inside shape.
		ret += this->m_low;

		return ret;
	}

	
	/**
	 * Iterator factory
	 */
	inline iterator
	begin() const 
	{ 
		return iterator(*(this->m_pred), ShapeIterator(*this), ShapeIterator(*this, -1));
	}
  
	/**
	 * Iterator factory
	 */
	inline iterator
	end() const 
	{
		return iterator(*(this->m_pred), ShapeIterator(*this, -1), ShapeIterator(*this, -1));
	}
  
public:
  
	/**
	 * Relocates the coordinate system represented by 
	 * this shape. The underlying mapping to linearized 
	 * representation remains unchanged.
	 *
	 * @param origin    new low coordinate of the shape
	 */
	inline void
	relocate(const Tuple<DIM>& origin) 
	{
		Tuple<DIM> dir(origin - this->m_low);
		this->m_low = origin;
		this->m_high += dir;
	}
  
	/**
	 * Relocates and remaps.
	 */
	inline Shape<DIM, PREDICATE>
	operator -(const Tuple<DIM>& dir) const
	{
		return (*this) + (-dir);
	}

    
	/**
	 * Relocates and remaps.
	 */
	inline Shape<DIM, PREDICATE>
	operator +(const Tuple<DIM>& dir) const
	{
		Shape<DIM, PREDICATE> ret(*this);    
		Tuple<DIM> new_origin = this->m_low + dir;
		ret.relocate(new_origin);
		return ret;
	}
  
	/**
	 * Assignment-move operator 
	 */
	Shape&
	operator +=(const Tuple<DIM>& dir)
	{
		Tuple<DIM> new_origin = this->m_low + dir;
		relocate(new_origin);
		return *this;
	}
  
	/**
	 * Assignment-move operator 
	 */
	Shape&
	operator-=(const Tuple<DIM>& dir)
	{
		Tuple<DIM> new_origin = this->m_low - dir;
		relocate(new_origin); 
		return *this;
	}

	template <unsigned int D, class P> friend ::std::ostream& operator << (::std::ostream& ost, const Shape<D, P>& s);  

};


// Iterator spec for ShapeIterator class.

template <unsigned int DIM, class PREDICATE>
class Shape<DIM, PREDICATE>::ShapeIteratorSpec
{
public:
	typedef int							iterator_type;
	typedef Shape<DIM, PREDICATE>		collection_type;
	typedef std::input_iterator_tag		iterator_category;  
	typedef Tuple<DIM>					value;
	typedef value*						pointer;
	typedef const value*				const_pointer;
	typedef value&						reference;
	typedef const value&				const_reference;
};


// ShapeIterator is an input iterator that iterates over
// Shape. The iterator assumes row major access and DIM-1 is the least
// significant dimention.

template <unsigned int DIM, class PREDICATE>
class Shape<DIM, PREDICATE>::ShapeIterator : public Iterator<ShapeIteratorSpec>
{
public:
	// Iterator typedef's
	typedef typename Iterator<ShapeIteratorSpec>::iterator_type		iterator_type;
	typedef typename Iterator<ShapeIteratorSpec>::collection_type	collection_type;
	typedef typename Iterator<ShapeIteratorSpec>::iterator_category	iterator_catagory;
	typedef typename Iterator<ShapeIteratorSpec>::reference			reference;
	typedef typename Iterator<ShapeIteratorSpec>::const_reference	const_reference;
	typedef typename Iterator<ShapeIteratorSpec>::pointer			pointer;
	typedef typename Iterator<ShapeIteratorSpec>::const_pointer		const_pointer;
	typedef typename Iterator<ShapeIteratorSpec>::value				value;
	typedef typename Iterator<ShapeIteratorSpec>::difference_type	difference_type;
	
private:

	const collection_type& m_coll;	// Reference to the collection that will be iterated over
	value m_value;					// current value of the iterator

public:

	// Default construction not allowed (required by forward iterators)
	ShapeIterator()
		{assert(false);}

	// Main constructor function
	ShapeIterator(const collection_type& coll, const iterator_type& cur) : 
		Iterator<ShapeIteratorSpec>(cur), 
		m_coll(coll),
		m_value(coll.coord(cur))
	{}

	// Copy constructor (required by all iterators)
	ShapeIterator(const ShapeIterator& it) :
		Iterator<ShapeIteratorSpec>(it.m_current), 
		m_coll(it.m_coll),
		m_value(it.m_value)
	{}

	// Prefix increment (required by all iterators)
	ShapeIterator&
	operator ++() 
	{     
		this->Advance();
		return *this;
	}


	// Postfix increment (required by all iterators)
	ShapeIterator
	operator ++(int) 
	{
		assert(this->m_valid);
		ShapeIterator tmp(*this);
		this->Advance();
		return tmp;
	}

	// Equality operator (required by input iterators)
	inline bool
	operator ==(const ShapeIterator& it) const
	{
		return this->m_current == it.m_current;
	}
	
	// Inequality operator (required by input iterators)
	inline bool
	operator !=(const ShapeIterator& it) const
	{
		return ! this->operator ==(it);
	}

	// Dereference operator (required by input iterators)
	const value&
	operator *() const 
	{
		assert(this->m_current != -1);
		return this->m_value;
	}

	// Dereference operator (required by input iterators)
	const value&
	operator ->() const
	{
		assert(this->m_current != -1);
		return this->m_value;
	}

/*
	// Assignment operator (required by output iterators)
	value&
	operator =(const value& data)
	{
		this->m_value = data;

		return this->m_value;
	}

	// Prefix decrement (required by bidirectional iterators)
	ShapeIterator&
	operator --()
	{
		this->step_back();
		return *this;
	}

	// Postfix decrement (required by bidirectional iterators)
	ShapeIterator
	operator --(int) 
	{
		assert(this->m_valid);
		ShapeIterator tmp(*this);
		this->step_back();
		return tmp;
	}

	// addition assignment operator (required by random access iterators)
	ShapeIterator&
	operator +=(int n)
	{
		this->Advance(n);
		return *this;
	}
	
	// Offset dereference operator (required by random access iterators)
	value
	operator[](int n) const 
	{ 
		assert(this->m_current != -1);
		assert(this->m_current + n < this->m_coll.count());

		return this->m_coll.coord(this->m_current + n) + this->m_coll.low();
	}
*/
	
	const int
	ord() const
	{
		assert(this->m_current != -1);
		return this->m_current;
	}

	/* This is for debugging only. Not doen in an overload of operator<<
	 * because it seems that gcc 3.4 does not manage inner class declarations of 
	 * template classes correctly
	 */
	char
	Print(std::ostream& ost) const
	{
		ost << "Shape<" << DIM << ">::iterator("
			<< "current=" << this->m_current 
			<< " currentTuple=" << this->m_value << ")";
		return '\0';
	}

private:
	
	void
	Advance(int n = 1) 
	{
		assert(this->m_current != -1);	// Don't increment if at the end the end.
		assert(this->m_currnet + n <= this->m_coll.count()); // Don't increment past the end.

		// Precalculate increment
		this->m_current += n;
		this->m_value[DIM - 1] += n;
		
		if(this->m_value[DIM - 1] >= this->m_coll.high()[DIM - 1])
		{
			// The end ofleast signifcant coordinate was exceeded,
			// so recalculate value (the current tuple).
			if(this->m_current < this->m_coll.count())
				this->m_value = this->m_coll.coord(this->m_current);
			else
				this->current = -1;	// The end was reached.
		}

		HTA_DEBUG(3, "Shape::Iterator::advance this=" << this << 
			" current=" << this->m_current << 
			" current tuple=" << this->m_value);
	}

/*
 	// Decrement itertator
	void
	Stepback(int n = 1) 
	{
		assert(this->m_current != 0);		// Don't decrement if at the begining.
		assert(this->m_current - n >= 0);	// Don't decrement past the begining.

		// Precalculate increment
		this->m_current -= n;
		this->m_value[DIM - 1] -= n;
		
		if(this->m_value[DIM - 1] < this->m_coll.low()[DIM - 1])
		{
			// The least signifcant coordinate was exceeded, so recalculate
			// value (the current tuple).
			if(this->m_current < 0)
			{
				// The iterator is before the begining.
				this->m_valid = false;
			}

			this->m_value = this->m_coll.coord(this->m_current);
		}
		
		HTA_DEBUG(3, "Shape::Iterator::advance this=" << this << 
			" current=" << this->m_current << 
			" current tuple=" << this->m_value);
	}
*/
};


template<int DIM, class Predicate>  ::std::ostream&
operator <<(::std::ostream& out, const Shape<DIM, Predicate>& s) {  
	out << "Shape<" << DIM << ">(" 
		<< " @=" << &s
		<< " low=" << s.low() 
		<< " high=" << s.high()
		<< " size=" << s.size() 
		<< " linearStep=" << s.LinearStep()  << ")";
	return out;
}


#endif // SHAPE_H__INCLUDED
