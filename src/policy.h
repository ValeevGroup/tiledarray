#ifndef _tiledarray_policy_h_
#define _tiledarray_policy_h_

namespace TiledArray {
  
  template <typename T, unsigned int DIM>
  struct DefaultArrayPolicy {
    
    typedef double* Tile;
    
    template <typename Index> struct Hash {
      typedef unsigned long Key;
      Key operator()(const Index&);
    };
    
  };
  
};

#endif
