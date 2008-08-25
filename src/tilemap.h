#ifndef TILEMAP_H_
#define TILEMAP_H_

#include <cstddef>
#include <map>
#include <boost/shared_ptr.hpp>

namespace TiledArray {

  /// DistributedMemoryPointer = proc + offset
  template <typename Ptr> struct DistributedMemoryPointer {
    /// offset=0 is invalid
    static DistributedMemoryPointer invalid;
    
    unsigned int proc;
    Ptr offset;
    DistributedMemoryPointer(unsigned int n, Ptr o) : proc(n), offset(o) {}
  };
  
  template <typename Ptr> std::ostream& operator <<(std::ostream& o,
                                                    const DistributedMemoryPointer<Ptr>& ptr);

  /** This class maps linearized tile indices to the procs and memory locations
   distribution of tiles to procs is done by hashing
   memory location can be determined for local tiles ONLY
   */
  class TileMap {
    public:
      typedef DistributedMemoryPointer<size_t> MemoryHandle;
      
      // need some data here?
      TileMap();
      virtual ~TileMap();

      /// Empties the tile map
      virtual void reset() =0;
      /// Returns the MemoryHandle corresponding to tile idx
      virtual MemoryHandle find(size_t idx) const =0;
      /// Registers tile idx with this map. Assumes that input tile has not been registered yet.
      virtual void register_tile(size_t idx, const MemoryHandle& tilehndl) =0;
      
  };
  
  /** Maintains information about local tiles, i.e. all tiles are assumed local
      and the proc info is completely ignored.
    */
  class LocalTileMap : public TileMap {
    public:
      typedef TileMap parent_type;
      typedef parent_type::MemoryHandle MemoryHandle;

      LocalTileMap();
      ~LocalTileMap();
      
      /// Implements TileMap::reset()
      void reset();
      /// Implements TileMap::find
      MemoryHandle find(size_t idx) const;
      /// Implements TileMap::register_tile
      void register_tile(size_t idx, const MemoryHandle& tilehndl);
      
    private:
      typedef std::map<size_t, size_t> Offsets;
      /// maps linear_tile_idx to offset for local tiles
      Offsets offsets_;
      
      /// this process rank
      unsigned int me_;
  };

  /// Manages tiles distributed among procs
  class DistributedTileMap : public TileMap {
    public:
      typedef TileMap parent_type;
      typedef parent_type::MemoryHandle MemoryHandle;

      DistributedTileMap();
      ~DistributedTileMap();
      
      /// Implements TileMap::reset()
      void reset();
      /// Implements TileMap::find
      MemoryHandle find(size_t idx) const;
      /// Implements TileMap::register_tile
      void register_tile(size_t idx, const MemoryHandle& tilehndl);
      
    private:
      LocalTileMap localmap_;
      
      /// this process rank
      unsigned int me_;
  };

} // namespace TiledArray

#endif /*TILEMAP_H_*/
