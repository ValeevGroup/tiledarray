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
 * Version: $Id: AbstractArray.h,v 1.10 2007/02/14 16:09:58 bikshand Exp $
 * Authors: Christoph von Praun
 */

#ifndef  ARRAY_H__
#define  ARRAY_H__

#include <cassert>
#include <vector>
#include <map>
#include "tuple.h"
#include "shape.h"
#include "VectorOps.h"
#include "iterator.h"


namespace TILED_ARRAY_NAMESPACE
{


template<typename T, unsigned int DIM>
class AbstractArray 
{
public:

	typedef	T			ValueType;
	typedef Tuple<DIM>	IndexType;

protected:
	IndexType m_dim;
	const ValueType m_defaultValue;

public:
	AbstractArray() :
		m_dim(), m_defaultValue(ValueType())
	{}

	AbstractArray(const IndexType dim, const ValueType defaultValue = ValueType()) :
		m_dim(dim), m_defaultValue(defaultValue)
	{}

	virtual inline const ValueType&
	At(const IndexType& index) const = 0;

	virtual inline ValueType&
	At(const IndexType& index) = 0;

	inline const ValueType&
	operator [](const IndexType& index) const
	{
		return this->At(index);
	}

	inline ValueType&
	operator [](const IndexType& index)
	{
		return this->At(index);
	}

	void
	Resize(const IndexType& newDim)
	{
		this->m_dim = newDim;
	}

	size_t
	Size() const
	{
		return VectorOps<IndexType, DIM>::selfProduct(this->m_dim);
	}
};

template<typename T, unsigned int DIM>
class DenseArray : public AbstractArray<T, DIM>
{

	class ArrayIteratorSpec {
	public:
		typedef int									iterator_type;
		typedef DenseArray<T,DIM>					collection_type;
		typedef std::random_access_iterator_tag		iterator_category;  
		typedef T									value;
		typedef value*								pointer;
		typedef const value*						const_pointer;
		typedef value&								reference;
		typedef const value&						const_reference;
	};

	class ArrayIterator : public Iterator<ArrayIteratorSpec>
	{

	};

public:
	typedef	typename AbstractArray<T, DIM>::ValueType	ValueType;
	typedef typename AbstractArray<T, DIM>::IndexType	IndexType;
	typedef std::vector<T>						ArrayType;
	typedef typename std::vector<T>::iterator			IteratorType;

protected:
	
	ArrayType m_data;

public:

	DenseArray() :
		AbstractArray<T, DIM>()
	{}

	DenseArray(const IndexType& dim, const ValueType& val = ValueType()) :
		AbstractArray<T, DIM>(dim, val), m_data(VectorOps<IndexType, DIM>::selfProduct(dim), val)
	{}

	DenseArray(const IndexType& dim, const ValueType& (*func)(const IndexType&)) :
 		AbstractArray<T, DIM>(dim), m_data(VectorOps<IndexType, DIM>::selfProduct(dim))
 	{
 		bool atEnd = false;
		for(IndexType ItDim(0); !atEnd; atEnd = VectorOps<IndexType, DIM>::increment(ItDim))
			this->At(ItDim) = func(ItDim);
 	}

	// Accessor
	virtual inline ValueType&
	At(const IndexType& index)
	{
		return this->m_data[Offset(index)];
	}

	virtual inline const ValueType&
	At(const IndexType& index) const
	{
		return this->m_data[Offset(index)];
	}

	virtual void
	Resize(const IndexType& newDim)
	{
		AbstractArray<T,DIM>::Resize(newDim);
		this->m_data.resize(this->Size(), this->m_defaultValue);
	}

private:

	unsigned int
	Offset(const IndexType& index) const
	{
		unsigned int n = 0;
		unsigned int m = 1;
		for(unsigned int dim = DIM - 1; dim > DIM; --dim)
		{
			assert(index[dim] >= 0 && index[dim] < this->m_dim);
			n += index[dim] * m;
			m *= this->m_dim[dim];
		}

		return n;
	}
};

template<typename T, unsigned int DIM>
class SparceArray : public AbstractArray<T, DIM>
{
public:
	typedef	typename AbstractArray<T, DIM>::ValueType	ValueType;
	typedef typename AbstractArray<T, DIM>::IndexType	IndexType;
	typedef std::map<Tuple<DIM>, T>			ArrayType;
	typedef typename ArrayType::iterator				Iterator;

protected:

	ArrayType m_data;

public:

	SparceArray() :
		AbstractArray<T, DIM>()
	{}

	SparceArray(const IndexType& dim) :
		AbstractArray<T, DIM>(dim), m_data(VectorOps<IndexType, DIM>::selfProduct(dim))
	{}

	SparceArray(const IndexType& dim, const ValueType& val) :
		AbstractArray<T, DIM>(dim), m_data(VectorOps<IndexType, DIM>::selfProduct(dim), val)
	{}

	// Accessor
	virtual inline ValueType&
	At(const IndexType& index)
	{
		typename ArrayType::iterator it = this->m_data.find(index);
		if(it == this->m_data.end())
			return this->m_defaultValue;

		return *it;
	}

	virtual inline const ValueType&
	At(const IndexType& index) const
	{
		typename ArrayType::const_iterator it = this->m_data.find(index);
		if(it == this->m_data.end())
			return this->m_defaultValue;

		return *it;
	}

};

} // TILED_ARRAY_NAMESPACE

#endif // ARRAY_H__
