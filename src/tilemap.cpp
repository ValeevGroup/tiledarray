#include <iostream>
#include <stdexcept>
#include <tilemap.h>
#include <env.h>

using namespace TiledArray;

//////////////////////

// pointer can't be zero
template<typename Ptr> DistributedMemoryPointer<Ptr> DistributedMemoryPointer<
    Ptr>::invalid(0, 0);

namespace TiledArray {
  template<> std::ostream& operator<<(std::ostream& o,
                                      const DistributedMemoryPointer<size_t>& ptr) {
    o << "DistributedMemoryPointer(proc=" << ptr.proc << " offset="
        << ptr.offset << ")";
    return o;
  }
}

//////////////////////

TileMap::TileMap() {
}

TileMap::~TileMap() {
}

//////////////////////

LocalTileMap::LocalTileMap() :
  me_(RuntimeEnvironment::Instance().me()) {
}

LocalTileMap::~LocalTileMap() {
}

void LocalTileMap::reset() {
  offsets_.clear();
}

size_t LocalTileMap::size() const {
  return local_size();
}

size_t LocalTileMap::local_size() const {
  return offsets_.size();
}

TileMap::MemoryHandle LocalTileMap::find(size_t idx) const {
  const Offsets::const_iterator i = offsets_.find(idx);
  if (i != offsets_.end())
    return MemoryHandle(me_, i->second);
  else
    return MemoryHandle::invalid;
}

void LocalTileMap::register_tile(size_t idx, const MemoryHandle::ptr_t& tileptr) {
  offsets_[idx] = tileptr;
}

//////////////////////

DistributedTileMap::DistributedTileMap() :
  nproc_(RuntimeEnvironment::Instance().nproc()), me_(RuntimeEnvironment::Instance().me()) {
}

DistributedTileMap::~DistributedTileMap() {
}

void DistributedTileMap::reset() {
  localmap_.reset();
}

size_t DistributedTileMap::size() const {
  abort();
  return 0;
}

size_t DistributedTileMap::local_size() const {
  return localmap_.size();
}

TileMap::MemoryHandle DistributedTileMap::find(size_t idx) const {
  // determine the node
  const unsigned int p = proc(idx);
  // if node == me, use localmap_
  if (p == me_)
    return localmap_.find(idx);
  else
    return MemoryHandle(p, 0);
}

void DistributedTileMap::register_tile(size_t idx,
                                       const MemoryHandle::ptr_t& tileptr) {
  // determine the node
  const unsigned int p = proc(idx);
  // if node == me, use localmap_
  if (p == me_)
    localmap_.register_tile(idx, tileptr);
  else
    throw std::runtime_error("DistributedTileMap::register_tile -- cannot register nonlocal tile");
}
