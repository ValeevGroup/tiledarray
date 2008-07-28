#ifndef ALLOCATOR_H__
#define ALLOCATOR_H__

namespace TILED_ARRAY_NAMESPACE
{

template<typename T>
class Allocator
{

}; // class Allocator

template<typename T>
class LocalAllocator : public Allocator<T>
{

}; // class LocalAllocator

} // TILED_ARRAY_NAMESPACE

#endif // ALLOCATOR_H__