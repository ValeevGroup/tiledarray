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

namespace TILED_ARRAY_NAMESPACE
{

template<typename T, unsigned int DIM, unsigned int LEVEL, class TRAIT<T,DIM> = LocalDenseTrait<T,DIM> >
class TA
{
public:
	typedef T												ValueType;
	typedef TRAIT											TraitType;
	typedef typename TraitType::SubrefTrait					SubtraitType;
	typedef TA<ValueType, DIM, LEVEL - 1, SubtraitType>		ElementType;
	typedef TraitType::Structure							StructureType;
	typedef TRAIT::DataIterator								ElementIterator;
	typedef TraitType::SubrefDataIterator					SubElementIterator;

	Shape m_shape;
	bool m_distributed;

public:
	ElementType&
	operator ()(const Tuple& index);

	const ElementType&
	operator ()(const Tuple& index) const;

	ElementType&
	operator ()(const Tuple* index, unsigned int level);

	const ElementType&
	operator ()(const Tuple& index) const;

	ValueType&
	operator [](const Tuple& index);

	ValueType&
	operator ()(const Tuple& index) const;

	const Shape&
	Shape() const;

	bool
	IsDistributed() const;


}; // class TA

template<typename T, unsigned int DIM, class TRAIT = LocalDenseTrait>
class TA<T, DIM, 0, TRAIT>
{
public:
	typedef T												ValueType;
	typedef TRAIT											TraitType;
	typedef typename TraitType::SubrefTrait					SubtraitType;
	typedef TA<ValueType, DIM, LEVEL - 1, SubtraitType>		ElementType;
	typedef TraitType::Structure							StructureType;

private:


public:
	typedef TAIterator								Iterator;


} // TILED_ARRAY_NAMESPACE



#ifdef REDEF_NDEBUG
#define NDEBUG
#undef REDEF_DEBUG
#endif // REDEF_NDEBUG


#endif // TILEDARRAY_H_
