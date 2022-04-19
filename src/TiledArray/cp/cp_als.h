//
// Created by Karl Pierce on 4/17/22.
//

#ifndef TILEDARRAY_CP_CP_ALS__H
#define TILEDARRAY_CP_CP_ALS__H

#include <TiledArray/cp/cp.h>

namespace TiledArray::cp {
template <typename Tile, typename Policy>
class CP_ALS : public CP<Tile, Policy>{
 public:
  using CP<Tile, Policy>::ndim;
  using CP<Tile, Policy>::cp_factors;
  CP_ALS() = default;

  CP_ALS(const DistArray<Tile, Policy> & tref) :
            CP<Tile, Policy>(rank(tref)), reference(tref), world(tref.world()){
  }

 protected:
  const DistArray<Tile, Policy> & reference;
  madness::World & world;

  void build_guess(const size_t rank,
                   const TiledRange1 rank_trange) override{
    if(cp_factors.size() == 0) {
      for (auto i = 0; i < ndim; ++i) {
        cp_factors.emplace_back(this->construct_random_factor(
            world, rank, reference.trange().tiles_range().extent_data()[i],
            rank_trange, reference.trange().data()[i]));
      }
    } else{
      TA_EXCEPTION("Currently no implementation to increase or change rank");
    }

    return;
  }

  /// This function is specified by the CP solver
  /// optimizes the rank @c rank CP approximation
  /// stored in cp_factors.
  /// \param[in] rank rank of the CP approximation
  /// \param[in] max_iter max number of ALS iterations
  /// \param[in] verbose Should ALS print fit information while running?
  void ALS(size_t rank, size_t max_iter, bool verbose = false) override{
    size_t iter = 0;
    bool converged = false;
    // initialize partial grammians
    {
      auto ptr = this->partial_grammian.begin();
      for (auto& i : cp_factors) {
        (*ptr)("r,rp") = i("r,n") * i("rp, n");
        ++ptr;
      }
    }
    auto factor_begin = cp_factors.data();
    do{
      for(auto i = 0; i < ndim; ++i){
        update_factor(i);
        const auto & a = *(factor_begin + i);
        (this->partial_grammian[i])("r,rp") = a("r,n") * a("rp,n");
      }
      converged = this->check_fit(verbose);
      ++iter;
    }while(iter < max_iter && !converged);
  }

  void update_factor(size_t mode){
    std::string ref, contract, final;
    ref.reserve(ndim); contract.reserve(2); final.reserve(ndim);
    auto ptr_ref = ref.data(), ptr_cont = contract.data(),
         ptr_final = final.data();
    *ptr_final = char(ndim); *ptr_cont = char(ndim);
    size_t first_contr = (mode == ndim - 1 ? 0 : ndim - 1);
    *(ptr_cont + 1) = char(first_contr);
    for(size_t i = 0; i < ndim; ++i){
      *(ptr_ref + i) = char(i);
      if(i == first_contr) continue;
      *(ptr_cont) = char(i);
    }
    DistArray<Tile, Policy> An;
    An(final) = this->reference(ref) * cp_factors[first_contr](contract);

    
  }
};
} // namespace TiledArray::cp

#endif  // TILEDARRAY_CP_CP_ALS__H
