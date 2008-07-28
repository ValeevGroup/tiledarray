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
 * Version: $Id: Iterator.h,v 1.3 2005/12/29 17:26:10 vonpraun Exp $
 * Authors: Christoph von Praun
 */

/*
 * Updated: 7/18/2008
 * Author: Justus Calvin
 * 
 * Changes:
 * - Added Tiled Array namespace
 */

#ifndef ITERATOR_H_
#define ITERATOR_H_

#include <iterator>


namespace TILED_ARRAY_NAMESPACE
{

template <typename _IteratorSpec>
class Iterator :
	public std::iterator<typename _IteratorSpec::iterator_category,
		typename _IteratorSpec::value, typename _IteratorSpec::iterator_type,
		typename _IteratorSpec::pointer, typename _IteratorSpec::reference>
{

public:
	typedef typename _IteratorSpec::iterator_type iterator_type;
	typedef int difference_type;
	typedef typename _IteratorSpec::reference reference;
	typedef typename _IteratorSpec::pointer pointer;
	typedef typename _IteratorSpec::value value;

protected:
	iterator_type current_;

public:
	const iterator_type& base() const {
		return current_; 
	}

	explicit Iterator(iterator_type __x) : current_(__x)
	{}

	Iterator&
	operator ++()
	{
		assert(0); // not implemented
		return *this;
	}
  
	Iterator
	operator ++(int)
	{
		assert(0); // not implemented
		return *this;
	}

	Iterator&
	operator --()
	{
		assert(0); // not implemented
		return *this;
	}
  
	Iterator
	operator --(int)
	{
		assert(0); // not implemented
		return *this;
	}
  
	Iterator
	operator +(difference_type __n) const
	{
		assert(0); // not implemented
		return *this;
	}

	Iterator&
	operator +=(difference_type __n)
	{
		assert(0); // not implemented
		return *this;
	}

	Iterator
	operator -(difference_type __n) const
	{
		assert(0); // not implemented
		return *this;
	}
  
	Iterator&
	operator-=(difference_type __n)
	{
		assert(0); // not implemented
		return *this;
	}
 
	bool
	operator !=(const Iterator& other)
	{
		return current_ != other.current_;
	}

};

// forward iterator requirements
template<typename _IteratorSpec>
inline bool
operator ==(const Iterator<_IteratorSpec>& lhs, const Iterator<_IteratorSpec>& rhs)
{
	return lhs.base() == rhs.base(); 
}

template<typename _IteratorSpec>
inline bool
operator !=(const Iterator<_IteratorSpec>& lhs, const Iterator<_IteratorSpec>& rhs)
{ 
	return lhs.base() != rhs.base(); 
}

// random access iterator requirements
template<typename _IteratorSpec>
inline bool
operator <(const Iterator<_IteratorSpec>& lhs, const Iterator<_IteratorSpec>& rhs)
{
	return lhs.base() < rhs.base(); 
}

template<typename _IteratorSpec>
inline bool
operator >(const Iterator<_IteratorSpec>& __lhs, const Iterator<_IteratorSpec>& __rhs)
{
	return __lhs.base() > __rhs.base(); 
}

template<typename _IteratorSpec>
inline bool
operator <=(const Iterator<_IteratorSpec>& lhs, const Iterator<_IteratorSpec>& rhs)
{ 
	return lhs.base() <= rhs.base(); 
}

template<typename _IteratorSpec>
inline bool
operator >=(const Iterator<_IteratorSpec>& lhs, const Iterator<_IteratorSpec>& rhs)
{ 
	return lhs.base() >= rhs.base(); 
}

/*
template<typename _IteratorSpec>
inline typename Iterator<_IteratorSpec>::difference_type
operator -(const Iterator<_IteratorSpec>& __lhs, const Iterator<_IteratorSpec>& __rhs)
{ 
   assert(0); // not implemented   
}

template<typename _IteratorSpec>
inline Iterator<_IteratorSpec>
operator+(typename Iterator<_IteratorSpec>::difference_type n, const Iterator<_IteratorSpec>& i)
{ 
	assert(0); // not implemented
}
*/

} // TILED_ARRAY_NAMESPACE

#endif /*ITERATOR_H_*/
