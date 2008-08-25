#include <iostream>
#include <vector>
#include <tilemaptest.h>
#include <tilemap.h>
#include <env.h>

using namespace TiledArray;

void TileMapTest() {
  
  typedef TileMap::MemoryHandle MemoryHandle;

  // test LocalTileMap
  {
    TestRuntimeEnvironment::CreateInstance(1, 0);
    
    LocalTileMap map0;
    std::cout << "created map0<LocalTileMap>" << std::endl;
    MemoryHandle hdl0 = map0.find(127);
    std::cout << "map0.find(127) = " << hdl0 << std::endl;
    map0.register_tile(119, 1024);
    std::cout << "map0: tile 119 added at 1024" << std::endl;
    MemoryHandle hdl1 = map0.find(119);
    std::cout << "map0.find(119) = " << hdl1 << std::endl;
    
    map0.reset();
    std::cout << "map0: reset" << std::endl;
    
    map0.register_tile(69, 1204);
    std::cout << "map0: tile 69 added at 1204" << std::endl;
    MemoryHandle hdl2 = map0.find(69);
    std::cout << "map0.find(69) = " << hdl2 << std::endl;
    MemoryHandle hdl3 = map0.find(119);
    std::cout << "map0.find(119) = " << hdl3 << std::endl;
    
    std::cout << "destroyed map0" << std::endl << std::endl;
    
    TestRuntimeEnvironment::DestroyInstance();
  }
  
  // test DistributedTileMap
  {
    const unsigned int nproc = 4;
    std::vector< boost::shared_ptr<DistributedTileMap> > maps1(nproc);
    for (unsigned int p=0; p<nproc; ++p) {
      TestRuntimeEnvironment::CreateInstance(nproc, p);
      maps1[p] = boost::shared_ptr<DistributedTileMap>(new DistributedTileMap);
      std::cout << "created map1<DistributedTileMap> on proc " << p << std::endl;
      TestRuntimeEnvironment::DestroyInstance();
    }

    const size_t ntiles = 21;
    for(size_t t=0; t<ntiles; ++t) {
      const unsigned int proc = maps1[0]->proc(t);
      maps1[proc]->register_tile(t, 2*t);
      std::cout << "map1[" << proc << "]: tile " << t << " added at " << 2*t << std::endl;
    }
    
    for(unsigned int p=0; p<nproc; ++p) {
      for(size_t t=0; t<ntiles; ++t) {
        const MemoryHandle hdl = maps1[p]->find(t);
        std::cout << "map1[" << p << "].find(" << t << ") = " << hdl << std::endl;
      }
    }
    
    for(unsigned int p=0; p<nproc; ++p) {
      std::cout << "map1[" << p << "].local_size() = " << maps1[p]->local_size() << std::endl;
    }

  }
  
}