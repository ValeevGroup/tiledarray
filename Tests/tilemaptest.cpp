#include <iostream>
#include <tilemaptest.h>
#include <tilemap.h>
#include <env.h>

using namespace TiledArray;

void TileMapTest() {
  
  typedef TileMap::MemoryHandle MemoryHandle;
  
  {
    // create serial runtime environment
    TestRuntimeEnvironment::CreateInstance(1, 0);
    
    LocalTileMap map0;
    std::cout << "created map0<LocalTileMap>" << std::endl;
    MemoryHandle hdl0 = map0.find(127);
    std::cout << "map0.find(127) = " << hdl0 << std::endl;
    MemoryHandle hdl1(0, 1024);
    map0.register_tile(119, hdl1);
    std::cout << "map0: tile 119 added at " << hdl1 << std::endl;
    MemoryHandle hdl2 = map0.find(119);
    std::cout << "map0.find(119) = " << hdl2 << std::endl;

    MemoryHandle hdl3(1, 1204);
    map0.register_tile(69, hdl3);
    std::cout << "map0: tile 69 added at " << hdl3 << std::endl;
    MemoryHandle hdl4 = map0.find(69);
    std::cout << "map0.find(69) = " << hdl4 << std::endl;

    TestRuntimeEnvironment::DestroyInstance();
  }
  
}