#ifndef TILEDARRAY_H__INCLUDED
#define TILEDARRAY_H__INCLUDED

#include <cassert>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iterator>

#include <boost/smart_ptr.hpp>
#include <boost/iterator/filter_iterator.hpp>

#ifdef TA_DEBUG_LEVEL

#if (TA_DEBUG_LEVEL > 0)

#ifdef NDEBUG
#undef NDEBUG
#define REDEF_NDEBUG
#endif

#endif // (TA_DEBUG_LEVEL > 0)

#else

#ifndef NDEBUG
#define TA_DEBUG_LEVEL 1
#endif // NDEBUG

#endif // TA_DEBUG_LEVLE

#define TA_FULL_DEBUG 3

namespace TiledArray
{

#include "VectorOps.h"
#include "tracing.h"
#include "tuple.h"
#include "iterator.h"
#include "shape.h"
#include "range.h"
#include "orthotope.h"
#include "operators.h"
#include "array.h"
#include "predicate.h"
#include "allocator.h"
#include "mathkernel.h"
#include "trait.h"


template<typename T, unsigned int DIM, class TRAIT = LocalDenseTrait<T,DIM> >
class TA
{
public:
	typedef Tuple<DIM>										IndexType;
	typedef T												ValueType;
	typedef TRAIT											TraitType;
	typedef typename TRAIT::StructType						StructType;
	typedef typename TRAIT::ShapeType						ShapeType;

	::boost::shared_ptr<AbstractShape<DIM> > m_shape;
	bool m_distributed;

public:
/*
	ElementType&
	operator ()(const IndexType& index);

	const ElementType&
	operator ()(const IndexType& index) const;

	ElementType&
	operator ()(const IndexType* index, unsigned int level);

	const ElementType&
	operator ()(const IndexType* index, unsigned int level) const;
*/
	ValueType&
	operator [](const IndexType& index);

	const ValueType&
	operator [](const IndexType& index) const;

	const ShapeType&
	shape() const
	{return dynamic_cast<ShapeType>(m_shape);}

	bool
	IsDistributed() const;


}; // class TA


} // TILED_ARRAY_NAMESPACE



#ifdef REDEF_NDEBUG
#define NDEBUG
#undef REDEF_DEBUG
#endif // REDEF_NDEBUG


#endif // TILEDARRAY_H__INCLUDED
