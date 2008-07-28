#ifndef structure_h__
#define structure_h__

#include "iterator.h"

namespace TILED_ARRAY_NAMESPACE
{

template<unsigned int DIM>
class Structure
{

}; // class Structure


template<typename T, unsigned int DIM, unsigned int LEVEL>
class DenseStructure : public Structure<DIM>
{

}; // class DenseStructure

} // TILED_ARRAY_NAMESPACE

#endif // structure_h__