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
      ref_indices += detail::intToAlphabet(i);
      if(i + 1 != ndim )
        ref_indices += ",";
    }

    first_gemm_dim_one = ref_indices;
    first_gemm_dim_last = ref_indices;

    first_gemm_dim_one.replace(0, 1, 1, detail::intToAlphabet(ndim));
    first_gemm_dim_last = "," + first_gemm_dim_last;
    first_gemm_dim_last.insert(0, 1, detail::intToAlphabet(ndim));
    first_gemm_dim_last.pop_back(); first_gemm_dim_last.pop_back();

    this->norm_reference = norm2(tref);
  }

 protected:
  const DistArray<Tile, Policy> & reference;
  madness::World & world;
  std::string ref_indices, first_gemm_dim_one, first_gemm_dim_last;
  std::vector<typename Tile::value_type> lambda;
  TiledRange1 rank_trange1;

  /// This function constructs the initial CP facotr matrices
  /// stores them in CP::cp_factors vector.
  /// In general the initial guess is constructed using quasi-random numbers
  /// generated between [-1, 1]
  /// \param[in] rank rank of the CP approximation
  /// \param[in] rank_trange TiledRange1 of the rank dimension.
  void build_guess(const size_t rank,
                   const TiledRange1 rank_trange) override{
    rank_trange1 = rank_trange;
    if(cp_factors.size() == 0) {
      for (auto i = 0; i < ndim; ++i) {
        auto factor = this->construct_random_factor(
                              world, rank, reference.trange().elements_range().extent(i),
                              rank_trange, reference.trange().data()[i]);
        lambda = this->normalize_factor(factor, rank,
                                        TiledRange({rank_trange1, rank_trange1}));
        cp_factors.emplace_back(factor);
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
    auto factor_begin = cp_factors.data(),
         gram_begin = this->partial_grammian.data();
    do{
      for(auto i = 0; i < ndim; ++i){
        update_factor(i, rank);
        const auto & a = *(factor_begin + i);
        (*(gram_begin + i))("r,rp") = a("r,n") * a("rp,n");
      }
      converged = this->check_fit(verbose);
      ++iter;
    }while(iter < max_iter && !converged);
  }

  void update_factor(size_t mode, size_t rank){
    auto mode0 = (mode == 0);
    auto & An = cp_factors[mode];
    An = DistArray<Tile, Policy>();
    // Starting to form the Matricized tensor times khatri rao product
    // MTTKRP
    // To do this we, in general, contract the reference with the
    // factor of the first mode unless we are
    // looking to optimize the first mode factor then we contract the
    // last mode.
    auto contracted_index = (mode0 ? ndim - 1 : 0);
    std::string contract({detail::intToAlphabet(ndim), ',', detail::intToAlphabet(contracted_index)}),
        final = (mode == 0 ? first_gemm_dim_last : first_gemm_dim_one);

    TA::DistArray W(this->partial_grammian[contracted_index]);
    An(final) = this->reference(ref_indices) * cp_factors[contracted_index](contract);

    // next we need to contract (einsum) over all modes not including the
    // mode we seek to optimize. We do this by modifying the strings
    std::string mixed_contractions = final;
    // we are going to use this pointer to remove indices from our string
    // but if mode == 0, we want to skip the first mode.
    auto remove_index_start = (mode0 ? 3 : 1), remove_index_end = remove_index_start + 2;
    auto mcont_ptr = mixed_contractions.begin();
    auto end = (mode0 ? ndim - 1 : ndim);
    for(contracted_index = 1; contracted_index < end; ++contracted_index){
      if(contracted_index == mode) {
        remove_index_start += 2; remove_index_end +=2;
        continue;
      }

      contract.replace(2, 1, 1, detail::intToAlphabet(contracted_index));
      mixed_contractions.erase(mcont_ptr + remove_index_start,
                               mcont_ptr + remove_index_end);

      An = einsum(An(final), cp_factors[contracted_index](contract), mixed_contractions);
      //std::cout << An << std::endl;

      final = mixed_contractions;
      W("r,rp") *= this->partial_grammian[contracted_index]("r,rp");;
    }

    if(mode == ndim - 1) this->MTtKRP = An;

    this->cholesky_inverse(An, W);

    if(mode == ndim - 1) this->unNormalized_Factor = An;
    lambda = this->normalize_factor(An, rank,
                                    TiledRange({rank_trange1, rank_trange1}));
  }
};
} // namespace TiledArray::cp

#endif  // TILEDARRAY_CP_CP_ALS__H
