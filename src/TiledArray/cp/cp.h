//
// Created by Karl Pierce on 3/17/22.
//

#ifndef TILEDARRAY_CP_CP__H
#define TILEDARRAY_CP_CP__H

#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
namespace TiledArray::cp{

template<typename Tile, typename Policy>
class CP {
 public:
  CP() = default;

  CP(const TiledArray::DistArray<Tile, Policy> & ref) : reference(ref), ndim(ref.rank()){

  }

  ~CP() = default;

  double compute_rank(size_t rank, bool build_rank = false){
    double epsilon = 1.0;
    if(build_rank){
      size_t cur_rank = 1;
      do{
        build_guess(cur_rank);
        ALS(cur_rank);
        ++cur_rank;
      }while(cur_rank < rank);
    } else{
      build_guess(rank);
      ALS(rank);
    }
    return epsilon;
  }

  double compute_error(double error, size_t max_rank){
    size_t cur_rank = 1;
    double epsilon = 1.0;
    do{
      build_guess(cur_rank);
      ALS(cur_rank);
      ++cur_rank;
    }while(epsilon > error && cur_rank < max_rank);

    return epsilon;
  }

 protected:
  const TiledArray::DistArray<Tile, Policy> & reference;
  std::vector<TiledArray::DistArray<Tile, Policy> > cp_factors,
      partial_grammian;
  size_t ndim;
  double prev_fit;

  virtual void build_guess(size_t rank);

  virtual void ALS(size_t rank);

  bool check_fit(){

  }
};

/*template<typename Tile, typename Policy>
auto CP_ALS(TA::DistArray<Tile, Policy> target, TA::TiledRange1 tr1_rank,
            double epsilon_stop, int num_als = 1000){
  auto& world = target.world();
  double norm_target = TA::norm2(target);
  auto ndim = TA::rank(target);
  using Array = TA::DistArray<Tile, Policy>;
  std::vector<Array> factors;
  factors.reserve(ndim);
  auto target_trange = target.trange();
  Array A(world, TA::TiledRange{tr1_rank, target_trange.data()[0]});
  A.fill_random();

}*/

} // namespace TiledArray::cp

#endif  // TILEDARRAY_CP_CP__H
