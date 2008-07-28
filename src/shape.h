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

#ifndef SHAPE_H_
#define SHAPE_H_


#include <cassert>
#include <vector>
#include <iostream>

#include "Triplet.h"
#include "Tuple.h"
#include "tracing.h"
#include "iterator.h"
#include "VectorOps.h"


namespace TILED_ARRAY_NAMESPACE
{


/**
 * Shape class that defines a multi-diimensional coordinate system and 
 * its mapping to an underlying dense, linearized representation.
 * The mapping to ordinals assumes that 0 is the least signifcant 
 * dimension.
 *
 * Instances of Shape are immutable.
 */
template <unsigned int DIM>
class Shape
{

private:

	Tuple<DIM> m_size;		// size of each dimension - considering step
  	Tuple<DIM> m_high;		// highest index in each dimension (included)
  	Tuple<DIM> m_low;		// lowest index in each dimension (included)
  	Tuple<DIM> m_step;		// step in each dimension - usually 1
  	Tuple<DIM> m_linearStep;// the number of elements to skip in the linearized 
  							// array until the respective dimension increments
  							// (unlike step, this is cumulative
  	Tuple<DIM> m_localStep;	// the number of elements to be skipped in the local
  							// co-ordinate system
  	Tuple<DIM> m_mod;		// the circshift parameter for each dimension

private:

	class ShapeIterator_spec {
	public:
		typedef int									iterator_type;
		typedef Shape<DIM>							collection_type;
		typedef std::random_access_iterator_tag		iterator_category;  
		typedef Tuple<DIM>							value;
		typedef value*								pointer;
		typedef const value*						const_pointer;
		typedef value&								reference;
		typedef const value&						const_reference;
	};

	class ShapeIterator : public Iterator<ShapeIterator_spec> {
	private:

		const typename ShapeIterator_spec::collection_type& m_coll;
		bool m_valid;
		int m_low;
		Tuple<DIM> m_currentTuple;

	public:

		ShapeIterator (const typename ShapeIterator_spec::collection_type& coll, int cur) : 
		  Iterator<ShapeIterator_spec>(cur), 
			  m_coll(coll), 
			  m_valid(true),
			  m_currentTuple()
		  { 
			  assert(cur >= 0 || cur == -1); 
			  Tuple<DIM> tmp_tuple;
			  int tmp = cur;

			  // compute the current tuple from cur, which  is the index 
			  // within the iteration space, 0 if m_currentTuple == m_coll.m_low.

			  for(int i = 0; i <  DIM - 1 ; i++)
			  {
				  tmp_tuple[i] = tmp / this->m_coll.m_linearStep[i];
				  tmp -= tmp_tuple[i] * this->m_coll.m_linearStep[i];
			  }
			  tmp_tuple[DIM-1] = tmp;

			  this->m_currentTuple = tmp_tuple + coll.m_low;
			  this->m_low = VectorOps<Tuple<DIM>, DIM>::dotProduct (this->m_coll.m_low, this->m_coll.m_linearStep);
		  }

		  ShapeIterator (const typename ShapeIterator_spec::iterator_type& it) :
		  Iterator<ShapeIterator_spec>(it), 
			  m_coll(it.coll_), 
			  m_currentTuple(it.m_currentTuple), 
			  m_valid(it.valid_)
		  {}

		  /**
		  * Prefix increment 
		  */
		  ShapeIterator& operator++() 
		  {     
			  this->Advance(1);
			  return *this;
		  }

		  /**
		  * Postfix increment 
		  */
		  ShapeIterator operator++(int) 
		  {
			  assert(this->m_valid);
			  ShapeIterator tmp(*this);
			  this->Advance(1);
			  return tmp;
		  }


		  /**
		  * Postfix increment 
		  */
		  ShapeIterator
			  operator --(int) 
		  {
			  assert(this->m_valid);
			  ShapeIterator tmp(*this);
			  this->Stepback(1);
			  return tmp;
		  }

		  ShapeIterator&
			  operator +=(int n)
		  {
			  this->Advance(n);
			  return *this;
		  }

		  typename ShapeIterator_spec::value
			  operator *() const 
		  {
			  assert(this->m_valid);
			  return m_currentTuple;
		  }

		  typename ShapeIterator_spec::value
			  operator[](int n) const 
		  { 
			  assert(this->m_valid);
			  ShapeIterator tmp = *this;
			  tmp.Advance(n);
			  return *tmp;
		  }

		  int
			  ord() const
		  {
			  assert(this->m_valid);
			  return this->current_;
		  }

		  /* This is for debugging only. Not doen in an overload of operator<<
		  * because it seems that gcc 3.4 does not manage inner class declarations of 
		  * template classes correctly */
		  char Print(std::ostream& ost) const
		  {
			  ost << "Shape<" << DIM << ">::iterator("
				  << "current=" << this->current_ 
				  << " currentTuple=" << this->m_currentTuple 
				  << " valid=" << this->valid_ << ")";
			  return '\0';
		  }

	private:

		void Advance(int n = 1) 
		{
			int acc = 0;
			bool step_done = false;

			// least significant dimension is at DIM-1
			for (int i = DIM-1; i >= 0; --i)
			{
				if (!step_done) {
					if (this->m_currentTuple[i] + (n * this->m_coll.step_[i]) > this->m_coll_.high_[i])
					{
						m_currentTuple[i] = this->m_coll.m_low[i];
					}
					else
					{
						this->m_currentTuple[i] += n * this->m_coll.m_step[i];
						step_done = true;
					}
				}
				acc += m_currentTuple[i] * m_coll.m_linearStep_[i];
			}

			if (!step_done) {
				// iterator hit the end - make it look like "end()"
				this->current_ = -1;
				this->m_valid = false;
			} else {
				assert (acc > this->m_current);
				this->m_current = acc - this->m_low;
			}
			HTA_DEBUG(3, "Shape::Iterator::advance this=" << this << 
				" acc=" << acc << 
				" current=" << this->current_ << 
				" currentTuple=" << this->m_currentTuple  << 
				" step_done=" << step_done);
		}

		void
			Stepback(int n = 1) 
		{
			int acc = 0;
			bool step_done = false;

			// least significant dimension is at DIM-1
			for (int i = DIM-1; i >= 0; --i) {
				if (!step_done) {
					if (this->m_currentTuple[i] - (n * m_coll.step_[i]) < m_coll.m_low[i]) {
						this->m_currentTuple[i] = m_coll.high_[i];
					} else {
						this->m_currentTuple[i] -= n * m_coll.step_[i];
						step_done = true;
					}
				}
				acc += this->m_currentTuple[i] * m_coll.linearStep_[i];
			}

			if (!step_done) {
				// iterator hit the end - make it look like "end()"
				this->m_current = -1;
				this->m_valid = false;
			} else {
				//cout << acc << " " << this->current_ << endl;
				assert (acc < this->current_);
				this->current_ = acc - m_low;
			}
			HTA_DEBUG(3, "Shape::Iterator::advance this=" << this << 
				" acc=" << acc << 
				" current=" << this->current_ << 
				" currentTuple=" << this->m_currentTuple  << 
				" step_done=" << step_done);
		}
	};



    
public:

	typedef ShapeIterator iterator;

	/** 
	 * Default constructor
	 */
	Shape() :
		m_size(),
		m_high(),
		m_low(),
		m_step(1),
		m_linearStep(),
		m_localStep(),
		m_mod()
	{}

	/** 
	 * Convenience constructor.
	 *
	 * @param  t upper coordinates of the shape - not included
	 * @return shape with lower bound (0,0,..), step (1,1,..).
	 */
	Shape(const Tuple<DIM>& high) :
		m_size(high + 1),
		m_high(high),
		m_low(),
		m_step(1),
		m_linearStep(),
		m_localStep(),
		m_mod()
	{
		this->Init();
	}

	/** 
	 * Convenience constructor.
	 *
	 * @param  begin lower coordinates of shape
	 * @param  end   upper coordinates of the shape - not included
	 * @return       shape with step (1,1,..).
	 */
	Shape(const Tuple<DIM>& low, const Tuple<DIM> &high) :
		m_size(high - low + 1),
		m_high(high),
		m_low(low),
		m_step(1),
		m_linearStep(),
		m_localStep(),
		m_mod()
	{
		this->Init();
	}

	Shape(const Triplet* triplets) :
		m_size(),
		m_high(),
		m_low(),
		m_step(1),
		m_linearStep(),
		m_localStep(),
		m_mod()
	{
		for(int index = 0; index < DIM; ++index)
		{
			assert(triplets + index);

			m_low[index] = triplets[index].Low();
			this->m_high[index] = triplets[index].High();
			this->m_size[index] = triplets[index].Size();
			this->m_step[index] = triplets[index].Step();
			this->m_mod[index] = triplets[index].Mod();
		}
		
		this->Init();
	}

	/**
	 * Copy constructor 
	 */
	Shape(const Shape<DIM>& s) : 
		m_size(s.m_size), 
		m_high(s.m_high), 
		m_low(s.m_low), 
		m_step(s.m_step), 
		m_linearStep (s.m_linearStep),
		m_localStep(s.m_localStep),
		m_mod(s.m_mod)
	{ 
		TA_DEBUG(2, "Shape::<init> this=" << this << " m_linearStep=" << m_linearStep << " high:" << m_high);
	}

	/**
	 * @return a new shape that masks out one dimension
	 */
/*
Shape<DIM>
mask (int dim) const 
{
	assert (dim >=0 && dim < DIM);

	typename Triplet::Seq tmp;
	for (int i = 0; i < DIM; ++i) 
		tmp[i] = (i == dim) ? Triplet(0, 0, 1) : this->m_dims[i];
	return Shape<DIM>(tmp);
}
*/
	virtual
	~Shape()
	{}

private:

	/**
	 * Initializer method
	 * @param t  t[0] is the most significant dimension.
	 */
	void Init() 
	{
		// memorize values that aid the linearization function
		// DIM-1 is the *least* significant dimension in row-major 
		// (row is in dimension 0)

		// compute cumulative m_linearStep and m_localStep
		this->m_linearStep[DIM-1] = 1;
		this->m_localStep[DIM-1] = 1;
		for (int i = DIM - 1; i > 0; --i)
		{
			this->m_linearStep[i - 1] = (this->m_high[i] - this->m_low[i] + 1) * this->m_linearStep[i];
			this->m_localStep[i - 1] = this->m_size[i] * m_localStep[i];
		}

		TA_DEBUG(2, "Shape::Init this=" << this << " m_linearStep=" << m_linearStep << " high:" << m_high);
	}

public:

	Shape<DIM>
	SetLinearStep(const Shape<DIM>& s) const
	{
		Shape<DIM> ret(*this);
		ret.m_linearStep = s.m_linearStep;
		return ret;
	}

	virtual const Shape<DIM>*
	Clone() const
	{
		assert (false);
		return NULL;
	}

	virtual const Shape<DIM>&
	operator [](const Tuple<DIM>& i) const
	{
		return *this;
	}

	virtual const Shape<DIM>&
	ShapeAt (int i) const
	{
		return *this;
	}

	/**
	 * Accessor to dimension 
	 *
	 * @param    dimension index (0 .. DIM-1)
	 * @return   triplet that characterizes dimension i of this shape.
	 *           most significant dimension in row-major layout is found 
	 *           at index 0.
	 */
	inline const Triplet
	operator[] (unsigned int dim) const 
	{
		assert(dim < DIM);
		return Triplet(this->m_low[dim], this->m_high[dim], this->m_size[dim], this->m_mod[dim]); 
	}
  
	/**
	 * @param  s1    shape
	 * @param  s2    shape
	 * @return       true if both shapes are conformable.
	 */
	static inline bool
	Conformable(const Shape<DIM>& s1, const Shape<DIM>& s2) 
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
	IteratorConformable(const Shape<DIM>& s1, const Shape<DIM>& s2) 
	{
		if (&s1 == &s2)
			return true;
		else 
			return (s1.m_linearStep == s2.m_linearStep);
	}

	inline bool
	Contains(const Tuple<DIM>& t) const 
	{
		bool ret = ((t >= m_low) && (t <= m_high));
		for (int i=0; ret && i < DIM; ++i) 
			ret &= (t[i] - m_low[i]) % m_step[i] == 0;

		return ret;
	}

	inline bool
	Contains(const Shape<DIM>& s) const 
	{
		// current limitation 
		return VectorOps<Tuple<DIM>, DIM>::greatereq(s.m_low, m_low) && 
			VectorOps<Tuple<DIM>, DIM>::lesseq(s.m_high, m_high);
	}


	/**
	 * Equality operator
	 */
	inline bool
	operator ==(const Shape<DIM>& other) const 
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
	operator!= (const Shape<DIM>& other) const 
	{  
		return ! (this->operator==(other));
	}
  
	/**
	 * Accessor
	 */
	inline const Tuple<DIM>&
	Size() const // __attribute__((always_inline))
	{
		return this->m_size;
	}

	inline const Tuple<DIM>&
	Lows() const // __attribute__((always_inline))
	{
		return this->m_low;
	}
  
	inline const Tuple<DIM>&
	Highs() const // __attribute__((always_inline))
	{
		return this->m_high;
	}
  
	/**
	 * Accessor
	 *
	 * @return  the number of elements returned by an iteration over this shape
	 */
	inline int
	Count() const
	{
		return VectorOps<Tuple<DIM>, DIM>::selfProduct(m_size);
	}

	/**
	 * Accessor
	 */
	inline Tuple<DIM>& 
	LinearStep() // __attribute__((always_inline))
	{
		return this->m_linearStep;
	}

	inline Tuple<DIM>& 
	Step() // __attribute__((always_inline))
	{
		return this->m_step;
	}

	inline const Tuple<DIM>&
	Step() const // __attribute__((always_inline))
	{
		return m_step;
	}

	inline const Tuple<DIM>&
	LinearStep() const // __attribute__((always_inline))
	{
		return m_linearStep;
	}

	/**
	 * Constant high accessor
	 */
	inline const Tuple<DIM>&
	High () const // __attribute__((always_inline))
	{
		return this->m_high;
	}

	/**
	 * Constant low accessor
	 */
	inline const Tuple<DIM>&
	Low () const // __attribute__((always_inline))
	{
		return m_low;
	}

	/**
	 * Ordinal value of Tuple. Ordinal value does not include offset
	 * and does not consider step. If the shape starts at (5, 3), 
	 * then (5, 4) has ord 1 on row-major. 
	 * The ordinal value of low() is always 0.
	 * The ordinal value of high() is always <= linearCard().
	 */
	inline int
	ord(const Tuple<DIM>& in) const
	{
		//
		// TODO: optimize for m_mod == 0 (common case)
		// 
		// assert (contains(in)) could be too strong because 
		// a call to ord_(5,5) should be allowed in a shape 
		// (0..5:2,0..5:2), where (5,5) is not actually part 
		// of the shape.
		assert (in >= m_low && in <= m_high);

		// assert (m_mod == Tuple<DIM>::zero); // current limitation

		Tuple<DIM> tmp = (in - m_low) % (m_high - m_low + Tuple<DIM>::one);
		int ret =  VectorOps<Tuple<DIM>, DIM>::dotProduct(tmp, m_linearStep); 
		return ret;
	}
  
	/* GB: coord is local coordinate */
	inline Tuple<DIM>
	Coord(int index) const 
	{
		assert(index >= 0 && index < this->card());
		Tuple<DIM> ret;

		/* start with most significant dimension 0 */
		for (int dim = 0; index > 0 && dim < DIM; ++dim)
		{
			ret[dim] = index / m_localStep[dim];
			index -= ret[dim] * m_localStep[dim];
		}

		assert (index == 0);
		return ret;
	}
   
	/**
	 * Iterator factory
	 */
	inline iterator
	Begin() const 
	{ 
		return ShapeIterator(*this, 0);
	}
  
	/**
	 * Iterator factory
	 */
	inline iterator
	End() const 
	{
		return ShapeIterator(*this, -1);
	}

	// TODO: Determine the purpose of this function.
	// Tuple<DIM>::one is no longer implemented.
/*
	inline bool
	IsContiguous() const 
	{
		return this->m_step == Tuple<DIM>::one;
	}
*/
	inline iterator
	Rend() const 
	{
		return ShapeIterator(*this, -1);
	}

	inline iterator
	Rbegin() const 
	{ 
		return ShapeIterator(*this, this->Count() - 1);
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
	Relocate(const Tuple<DIM>& origin) 
	{
		Tuple<DIM> dir(origin - m_low);
		m_low = origin;
		m_high += dir;
	}
  
	/**
	 * Relocates and remaps.
	 */
	inline Shape<DIM>
	operator -(const Tuple<DIM>& dir) const
	{
		return (*this) + (-dir);
	}

    
	/**
	 * Relocates and remaps.
	 */
	inline Shape<DIM>
	operator +(const Tuple<DIM>& dir) const
	{
		Shape<DIM> ret(*this);    
		Tuple<DIM> new_origin = m_low + dir;
		ret.relocate(new_origin);
		return ret;
	}
  
	/**
	 * Assignment-move operator 
	 */
	Shape&
	operator +=(const Tuple<DIM>& dir)
	{
		Tuple<DIM> new_origin = m_low + dir;
		relocate(new_origin);
		return *this;
	}
  
	/**
	 * Assignment-move operator 
	 */
	Shape&
	operator-=(const Tuple<DIM>& dir)
	{
		Tuple<DIM> new_origin = m_low - dir;
		relocate(new_origin); 
		return *this;
	}

	template <int D> friend std::ostream& operator << (std::ostream& ost, const Shape<D>& s);  

};

template<int DIM>  std::ostream&
operator <<(std::ostream& out, const Shape<DIM>& s) {  
	out << "Shape<" << DIM << ">(" 
		<< " @=" << &s
		<< " low=" << s.Low() 
		<< " high=" << s.High()
		<< " step=" << s.Step() 
		<< " size=" << s.Size() 
		<< " linearStep=" << s.LinearStep()  << ")";
	return out;
}

} // TILED_ARRAY_NAMESPACE

#endif /*SHAPE_H_*/
