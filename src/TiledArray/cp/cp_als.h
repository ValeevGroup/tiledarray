//
// Created by Karl Pierce on 4/17/22.
//

#ifndef TILEDARRAY_CP_CP_ALS__H
#define TILEDARRAY_CP_CP_ALS__H

#include <TiledArray/cp/cp.h>

/**
 * This is a canonical polyadic (CP) optimization class which
 * takes a reference order-N tensor and decomposes it into a
 * set of order-2 tensors all coupled by a hyperdimension called the rank.
 * These factors are optimized using an alternating least squares
 * algorithm. This class is derived form the base CP class
 *
 * @tparam Tile typing for the DistArray tiles
 * @tparam Policy policy of the DistArray
**/
namespace TiledArray::cp {
template <typename Tile, typename Policy>
class CP_ALS : public CP<Tile, Policy>{
 public:
  using CP<Tile, Policy>::ndim;
  using CP<Tile, Policy>::cp_factors;

  /// Default CP_ALS constructor
  CP_ALS() = default;

  /// CP_ALS constructor function
  /// takes, as a constant reference, the tensor to be decomposed
  /// \param[in] tref A constant reference to the tensor to be decomposed.
  CP_ALS(const DistArray<Tile, Policy> & tref) :
            CP<Tile, Policy>(rank(tref)), reference(tref), world(tref.world()){
    for(size_t i = 0; i < ndim; ++i){
      ref_indices += std::to_string(i);
      if(i + 1 != ndim )
        ref_indices += ",";
    }
    std::cout << ref_indices << std::endl;
  }

 protected:
  const DistArray<Tile, Policy> & reference;
  madness::World & world;
  std::string ref_indices;

  /// This function constructs the initial CP facotr matrices
  /// stores them in CP::cp_factors vector.
  /// In general the initial guess is constructed using quasi-random numbers
  /// generated between [-1, 1]
  /// \param[in] rank rank of the CP approximation
  /// \param[in] rank_trange TiledRange1 of the rank dimension.
  void build_guess(const size_t rank,
                   const TiledRange1 rank_trange) override{
    if(cp_factors.size() == 0) {
      for (auto i = 0; i < ndim; ++i) {
        auto factor = this->construct_random_factor(
                              world, rank, reference.trange().elements_range().extent(i),
                              rank_trange, reference.trange().data()[i]);
        std::cout << factor << std::endl;
        cp_factors.emplace_back(factor);
        this->normCol(factor, rank);
        std::cout << factor << std::endl;
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
    auto first_contr = (mode == ndim - 1 ? 0 : ndim - 1);
    std::string contract(char(ndim) + "," + char(first_contr)),
        final = ref_indices;
    auto start_erase = 2 * first_contr,
         end_erase = start_erase + 1;
    final.erase(start_erase, end_erase);
    final = std::to_string(ndim) + "," + final;

    DistArray<Tile, Policy> An;
    An(final) = this->reference(ref_indices) * cp_factors[first_contr](contract);

    
  }
};
} // namespace TiledArray::cp

#endif  // TILEDARRAY_CP_CP_ALS__H
