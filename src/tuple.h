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
 * Version: $Id: Tuple.h,v 1.65 2007/01/11 01:11:32 bikshand Exp $
 * Authors: Ganesh Bikshandi, Christoph von Praun
 */

/*
 * Updated: 7/16/2008
 * Author: Justus Calvin
 * 
 * Changes:
 * - Data is now stored in an STL vector.
 * - Modified the default class to handle n dimentions.
 * - Addes STL style iterator functionallity.
 * - Removed seq.
 * - Added Tiled Array namespace
 */

#ifndef TUPLE_H_
#define TUPLE_H_

#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>

#include "tracing.h"
#include "VectorOps.h"
#include "operators.h"

namespace TILED_ARRAY_NAMESPACE
{

template <unsigned int DIM> 
class Tuple
{
	
	typedef	std::vector<int> vectortype;
	
	vectortype m_values;	// vector of DIM-dimentional point
	
public:
	
	// Iterator for dimentions
    typedef     std::vector<int>::iterator          iterator;
    typedef     std::vector<int>::const_iterator    const_iterator;
	
	inline explicit
	Tuple() : m_values(DIM, 0)
	{}

	inline explicit
	Tuple(const int value) : m_values(DIM, value)
	{}

	inline explicit
	Tuple(const int* values) : m_values(values, values + DIM)
	{}

	inline explicit
	Tuple(const std::vector<int>& values) : m_values(DIM, values) 
	{}


	/** 
	 * Copy constructor 
	 */
	inline
	Tuple(const Tuple<DIM>& tup) : m_values(tup.m_values)
	{}

    // STL style coordinate iterator functions
    
    // Returns an interator to the first coordinate
    iterator
    Begin()
    {
    	return this->m_values.begin();
    }
    
    // Returns a constant iterator to the first coordinate. 
    const_iterator
    Begin() const
    {
    	return this->m_values.begin();
    }
    
    // Returns an iterator to one element past the last coordinate.
    iterator
    End()
    {
    	return this->m_values.end();
    }
    
    // Returns a constant iterator to one element past the last coordinate.
    const_iterator
    End() const
    {
    	return this->m_values.end();
    }

	// Arithmatic operators
	inline Tuple<DIM>
	operator +(const Tuple<DIM>& tup) const 
	{    
		Tuple<DIM> ret;
		VectorOps<vectortype, DIM>::add(ret.m_values, this->m_values, tup.m_values);
		return ret;
	}

	inline Tuple<DIM>&
	operator +=(const Tuple<DIM>& tup)  
	{   
		VectorOps<vectortype, DIM>::addIn(this->m_values, tup.m_values);
		
		return (*this);
	}

	inline Tuple<DIM>&
	operator -=(const Tuple<DIM>& tup)  
	{   
		VectorOps<vectortype, DIM>::subIn(this->m_values, tup.m_values);
		
		return (*this);
	}

	inline Tuple<DIM>
	operator +(int d) const 
	{
		Tuple<DIM> ret(d);
		VectorOps<vectortype, DIM>::addIn(this->m_values, ret);
		return ret;
	}

	inline Tuple<DIM>
	operator -(const Tuple<DIM>& other) const 
	{    
		Tuple<DIM> ret;
		VectorOps<vectortype, DIM>::sub(ret.m_values, this->m_values, other.m_values);
		return ret;
	}

	inline Tuple<DIM>
	operator -(int d) const 
	{    
		Tuple<DIM> ret(d);
		VectorOps<vectortype, DIM>::subIn(this->m_values, ret);
		return ret;
	}

	inline Tuple<DIM>
	operator -() const
	{
		Tuple<DIM> ret;
		VectorOps<vectortype, DIM>::uminus(ret.m_values, this->m_values);
		return ret;
	}

	inline Tuple<DIM>
	operator *(const Tuple<DIM>& other) const 
	{    
		Tuple<DIM> ret;
		VectorOps<vectortype, DIM>::mult(ret.m_values, this->m_values, other.m_values);
		return ret;
	}

	inline Tuple<DIM>
	operator /(const Tuple<DIM>& other) const 
	{    
		Tuple<DIM> ret;
		VectorOps<vectortype, DIM>::div(ret.m_values, this->m_values, other.m_values);
		return ret;
	}

	inline Tuple<DIM>
	operator %(const Tuple<DIM>& tup) const 
	{
		Tuple<DIM> ret;
		VectorOps<vectortype, DIM>::mod(ret.m_values, this->m_values, tup.m_values);
		return ret;
	}

	// Comparison Operators
	inline  bool
	operator ==(const Tuple<DIM>& tup) const 
	{
    	return VectorOps<vectortype, DIM>::equal(this->m_values, tup.m_values);
    }

	inline  bool
	operator !=(const Tuple<DIM>& tup) const 
	{
    	return !(this->operator ==(tup));
	}

	inline bool
	operator <(const Tuple<DIM>& tup) const 
	{
		return VectorOps<vectortype, DIM>::less(this->m_values, tup.m_values);
	}
  
	inline bool
	operator <=(const Tuple<DIM>& tup) const 
	{
		return (VectorOps<vectortype, DIM>::less(this->m_values, tup.m_values) ||
				VectorOps<vectortype, DIM>::equal(this->m_values, tup.m_values));
	}

	inline bool
	operator >(const Tuple<DIM>& tup) const 
	{
		return !(this->operator <=(tup));
	}


	inline bool
	operator >=(const Tuple<DIM>& tup) const 
	{
		return !(this->operator <(tup));
	}

	inline Tuple<DIM>&
	operator =(const Tuple<DIM> & tup) 
	{ 
    	std::copy(tup.m_coordinates.begin(), tup.m_coordinates.end(), this->m_coordinates.begin());
    	
    	return (*this);
	}
  
	inline const int
	operator[] (int i) const 
	{ 
		return this->m_values[i]; 
	}
  
	inline int&
	operator[] (int i) 
	{ 
		return this->m_values[i]; 
	}

	/**
	 * mask a value in a tuple with the given value (default 0).
	 *
	 * @param dim    the dimension to mask
	 * @param value  the value to put into the dimension
	 */
	Tuple<DIM> 
	Mask(int dim, int value = 0) const 
	{
		Tuple<DIM> ret = *this;
		ret[dim] = value;
		return ret;
	}
  	
    // Forward permutation of set by one.
    Tuple<DIM>&
    Permute()
    {   	
    	int temp = this->m_values[0];
    	
    	for(unsigned int index = 0; index < this->m_values.size() - 1; ++index)
    		this->m_values[index] = this->m_values[index + 1];
   	
    	this->m_values[this->m_values.size() - 1] = temp;
    		
    	return (*this);
    }
    
    // Reverse permutation of set by one.
    Tuple<DIM>&
    ReversePermute()
    {
    	int index = this->m_values.size() - 1;
    	
    	// Store the value of the last element
    	int temp = this->m_values[index];
    	
    	// shift all elements to the left
    	for(; index > 0; --index)
    		this->m_values[index] = this->m_values[index - 1];
   	
    	// place the value of the last element in the first.
    	this->m_values[0] = temp;
    		
    	return (*this);
    }

	/**
	 * User defined permutation of a tuple.
	 *
	 * @param perm - Tuple must include each index number (0, 1, ..., DIM-1) once and only once.
	 */
    Tuple<DIM>&
    Permute(const Tuple<DIM>& perm)
    {
#if (TA_DLEVEL >= 0)
    	// Ensure each index is present and listed only once.
    	int indexCount = 0;
    	for(unsigned int index = 0; index < this->m_values.size(); ++index)
    		indexCount += std::count(perm.Begin(), perm.End(), index);
    	
    	// Incorrect permutation, do nothing.
    	assert(indexCount == DIM);
#endif
    	
    	Tuple<DIM> temp(*this);
    	for(unsigned int index = 0; index < DIM; ++index)
    		this->m_values[index] = temp.m_values[perm.m_values[index]];
    	
    	return (*this);
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
		assert(false); 
	}

};

template <unsigned int DIM> 
std::ostream&
operator<<(std::ostream& output, const Tuple<DIM>& tup)
{
	output << "(";
	for (unsigned int i=0; i < DIM - 1; i++)
		output << tup[i] << ", ";
	output << tup[DIM-1] << ")";
	return output;
};

} // TILED_ARRAY_NAMESPACE

#endif /*TUPLE_H_*/
