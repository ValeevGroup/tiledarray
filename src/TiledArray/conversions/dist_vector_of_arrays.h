//
// Created by Karl Pierce on 9/6/19.
//

#ifndef TILEDARRAY_DIST_VECTOR_OF_ARRAYS_H
#define TILEDARRAY_DIST_VECTOR_OF_ARRAYS_H

#include <tiledarray.h>

namespace TiledArray {

template<typename Array>
class Dist_Vector_of_Arrays : public madness::WorldObject<Dist_Vector_of_Arrays<Array>> {

public:
  using Tile = typename Array::value_type;
  using Policy = typename Array::policy_type;

  Dist_Vector_of_Arrays(madness::World &world, const std::vector<Array> &array) :
          madness::WorldObject<Dist_Vector_of_Arrays<Array>>(world), split_array(array) {
    this->process_pending();
  }

  virtual ~Dist_Vector_of_Arrays(){ }

  template <typename Index>
  madness::Future<Tile> get_tile(int r, Index & i){
    return split_array[r].find(i);
  }

  const std::vector<Array>& array_accessor(){
    return split_array;
  }

  const int size(){
    return split_array.size();
  }
private:
  std::vector<Array> &split_array;
};

} // namespace TiledArray
#endif //TILEDARRAY_DIST_VECTOR_OF_ARRAYS_H
