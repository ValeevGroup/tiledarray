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

#ifndef ITERATOR_H__INCLUDED
#define ITERATOR_H__INCLUDED

template <class ITERATORSPEC>
class Iterator :
	public ::std::iterator<typename ITERATORSPEC::iterator_category,
		typename ITERATORSPEC::value, typename ITERATORSPEC::iterator_type,
		typename ITERATORSPEC::pointer, typename ITERATORSPEC::reference>
{

public:
	typedef typename ITERATORSPEC::collection_type	collection_type;
	typedef typename ITERATORSPEC::iterator_type	iterator_type;
	typedef int										difference_type;
	typedef typename ITERATORSPEC::reference		reference;
	typedef typename ITERATORSPEC::pointer			pointer;
	typedef typename ITERATORSPEC::value			value;

protected:
	iterator_type m_current;

public:
	Iterator() :
		m_current()
	{}
	
	explicit
	Iterator(const iterator_type& current) :
		m_current(current)
	{}

	explicit
	Iterator(const Iterator& it) :
		m_current(it.m_current)
	{}
	
	const iterator_type&
	base() const
		{return this->m_current;}

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
	operator --(int n)
	{
		assert(0); // not implemented
		return *this;
	}
  
	Iterator
	operator +(difference_type n) const
	{
		assert(0); // not implemented
		return *this;
	}

	Iterator&
	operator +=(difference_type n)
	{
		assert(0); // not implemented
		return *this;
	}

	Iterator
	operator -(difference_type n) const
	{
		assert(0); // not implemented
		return *this;
	}
  
	Iterator&
	operator -=(difference_type n)
	{
		assert(0); // not implemented
		return *this;
	}
 
	bool
	operator !=(const Iterator& other)
	{
		return this->m_current != other.m_current;
	}

};

// forward iterator requirements
template<typename IteratorSpec>
inline bool
operator ==(const Iterator<IteratorSpec>& lhs, const Iterator<IteratorSpec>& rhs)
{
	return lhs.base() == rhs.base(); 
}

template<typename IteratorSpec>
inline bool
operator !=(const Iterator<IteratorSpec>& lhs, const Iterator<IteratorSpec>& rhs)
{ 
	return lhs.base() != rhs.base(); 
}

// random access iterator requirements
template<typename IteratorSpec>
inline bool
operator <(const Iterator<IteratorSpec>& lhs, const Iterator<IteratorSpec>& rhs)
{
	return lhs.base() < rhs.base(); 
}

template<typename IteratorSpec>
inline bool
operator >(const Iterator<IteratorSpec>& lhs, const Iterator<IteratorSpec>& rhs)
{
	return lhs.base() > rhs.base(); 
}

template<typename IteratorSpec>
inline bool
operator <=(const Iterator<IteratorSpec>& lhs, const Iterator<IteratorSpec>& rhs)
{ 
	return lhs.base() <= rhs.base(); 
}

template<typename IteratorSpec>
inline bool
operator >=(const Iterator<IteratorSpec>& lhs, const Iterator<IteratorSpec>& rhs)
{ 
	return lhs.base() >= rhs.base(); 
}


template<typename IteratorSpec>
inline typename Iterator<IteratorSpec>::difference_type
operator -(const Iterator<IteratorSpec>& lhs, const Iterator<IteratorSpec>& rhs)
{ 
//	assert(0); // not implemented
	return (lhs.base() - rhs.base());
}

template<typename IteratorSpec>
inline Iterator<IteratorSpec>
operator+(typename Iterator<IteratorSpec>::difference_type n, const Iterator<IteratorSpec>& i)
{ 
//	assert(0); // not implemented
	Iterator<IteratorSpec> t;
	t.m_current = i.base() + n;
	return  t;
}



#endif // ITERATOR_H__INCLUDED
