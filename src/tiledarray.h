#ifndef TILEDARRAY_H_
#define TILEDARRAY_H_

#include <vector>

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

#define TILED_ARRAY_NAMESPACE TiledArray

// Forward Declaration
#include "tracing.h"
#include "tuple.h"
#include "triplet.h"
#include "shape.h"
#include "operators.h"
#include "array.h"
#include "trait.h"

namespace TILED_ARRAY_NAMESPACE
{

template<typename T, unsigned int DIM, unsigned int LEVEL, class TRAIT = LocalDenseTrait<T,DIM> >
class TA
{
public:
	typedef Tuple<DIM>										IndexType;
	typedef T												ValueType;
	typedef TRAIT											TraitType;
	typedef typename TRAIT::SubrefTrait						SubtraitType;
	typedef TA<ValueType, DIM, LEVEL - 1, SubtraitType>		ElementType;
	typedef typename TRAIT::StructType						StructType;
	typedef typename TRAIT::DataIterator					ElementIterator;

	Shape m_shape;
	bool m_distributed;

public:
	ElementType&
	operator ()(const IndexType& index);

	const ElementType&
	operator ()(const IndexType& index) const;

	ElementType&
	operator ()(const IndexType* index, unsigned int level);

	const ElementType&
	operator ()(const IndexType* index, unsigned int level) const;

	ValueType&
	operator [](const IndexType& index);

	const ValueType&
	operator [](const IndexType& index) const;

	const Shape&
	Shape() const;

	bool
	IsDistributed() const;


}; // class TA

template<typename T, unsigned int DIM, class TRAIT>
class TA<T, DIM, 0, TRAIT>
{
public:
	typedef T		ElementType;

};

} // TILED_ARRAY_NAMESPACE



#ifdef REDEF_NDEBUG
#define NDEBUG
#undef REDEF_DEBUG
#endif // REDEF_NDEBUG


#endif // TILEDARRAY_H_
