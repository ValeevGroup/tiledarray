
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
 * Version: $Id: Triplet.h,v 1.30 2006/05/11 21:41:52 vonpraun Exp $
 * Authors: Ganesh Bikshandi, Christoph von Praun
 */

/*
 * Updated: 7/16/2008
 * Author: Justus Calvin
 * 
 * Changes:
 * - Added Tiled Array namespace to header.
 * - Removed seq
 */

#ifndef TRIPLET_H__INCLUDED
#define TRIPLET_H__INCLUDED


class Triplet {

	int m_low;	// Lower bound
	int m_high;	// Upper bound
	int m_step;	// Step size
	int m_mod;	// the circhsift parameter

public:

	// constructors

	Triplet() :
		m_low(0), m_high(0), m_step(1), m_mod(0)
	{}

	Triplet(const Triplet& trip) :
		m_low(trip.m_low), m_high(trip.m_high), m_step(trip.m_step), m_mod(trip.m_mod)
	{} 
  
	Triplet(int h) :
		m_low(0), m_high(h), m_step(1), m_mod(0) 
	{
		assert(h >= 0);
	}

	Triplet(int low, int high, int step = 1, int mod = 0) :
		m_low(low), m_high(high), m_step(step), m_mod(mod)
	{
		assert(((low <= high) && (step > 0)) || ((low > high) && (step < 0 )));
	}

	Triplet&
	operator =(const Triplet& trip) 
	{
		if (&trip != this) {
			this->m_low = trip.m_low;
			this->m_high = trip.m_high;
			this->m_step = trip.m_step;
			this->m_mod = trip.m_mod;
		}

		return (*this);
	}

	bool
	operator ==(const Triplet& rhs) const
	{
		return (&rhs == this) || (rhs.m_low == m_low && 
				rhs.m_high == m_high && 
				rhs.m_step == m_step);
	}
  
	bool
	operator !=(const Triplet& rhs) const
	{
		return !this->operator==(rhs);
	}

	/**
	 * @return the number of elements covered by this triplet
	 */
	inline int
	Size() const 
	{ 
		return std::max(static_cast<int>(ceil(static_cast<double>( abs(m_high - m_low) + 1 ) / static_cast<double>(abs(m_step)))), 0 ); 
	}

	// Accessor functions
	inline int
	Low() const 
	{
		return m_low;
	}

	inline int&
	Low() 
	{
		return m_low;
	}

	inline int
	High() const 
	{
		return m_high;
	}
  
	inline int&
	High() 
	{
		return m_high;
	}

	inline int
	Step() const 
	{
		return m_step;
	}

	inline int&
	Step() 
	{
		return m_step;
	}

	inline int
	Mod() const
	{
		return m_mod;
	}
	
	inline int&
	Mod()
	{
		return m_mod;
	}
	
	// Math operators
	inline Triplet
	operator %(int r) 
	{
		return Triplet(m_low, m_high, m_step, m_mod + r);
	}

	inline Triplet
	operator +(int r) 
	{
		return Triplet(m_low + r, m_high + r, m_step, m_mod);
	}

	inline Triplet
	operator -(int r) 
	{
		return Triplet(m_low - r, m_high - r, m_step, m_mod);
	}

private:

	/** forbid heap allocation */
	void*
	operator new(size_t size) throw ()
	{
		assert(false);
		return NULL;
	}
	
	/**  forbid heap allocation */
	void
	operator delete(void* to_delete)
	{
		assert (0);
	}

};


inline std::ostream& 
operator <<(std::ostream& ost, const Triplet& trip) 
{
	return ost << "<" << trip.Low() << ":" << trip.Step() << ":" << trip.High() << ":" << trip.Mod() << ">";
}


#endif // TRIPLET_H__INCLUDED
