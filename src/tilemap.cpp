#include <iostream>
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

TileMap::MemoryHandle LocalTileMap::find(size_t idx) const {
  const Offsets::const_iterator i = offsets_.find(idx);
  if (i != offsets_.end())
    return MemoryHandle(me_, i->second);
  else
    return MemoryHandle::invalid;
}

void LocalTileMap::register_tile(size_t idx, const MemoryHandle& tilehndl) {
  offsets_[idx] = tilehndl.offset;
}

//////////////////////

DistributedTileMap::DistributedTileMap() :
  me_(RuntimeEnvironment::Instance().me()) {
}

DistributedTileMap::~DistributedTileMap() {
}

void DistributedTileMap::reset() {
  localmap_.reset();
}

TileMap::MemoryHandle DistributedTileMap::find(size_t idx) const {
  // determine the node
  // if node == me, use localmap_
  // else return 0
  abort();
}

void DistributedTileMap::register_tile(size_t idx, const MemoryHandle& tilehndl) {
  // determine the node
  // if node == me, use localmap_
  // else throw
  abort();
}
