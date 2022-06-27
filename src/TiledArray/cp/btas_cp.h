//
// Created by Karl Pierce on 3/15/22.
//

#ifndef TiledArray_CP_BTAS_CP_H
#define TiledArray_CP_BTAS_CP_H
#include <tiledarray.h>
#include <TiledArray/conversions/btas.h>
#include <btas/btas.h>

namespace TiledArray::cp {
/**
 * REQUIRES BTAS
 * Calculates the canonical product (CP) decomposition of /c Reference to a
 * specific rank using alternating least squares (ALS) method in BTAS
 *
 * @param[in] world Madness world which CP will be computed on.
 * @param[in] Reference  Tensor to be CP decomposed.
 * @param[in] btas_cp_rank The rank of the CP decomposition
 * @param[in, out] rank_trange1 TiledRange1 object used to construct the CP
 * factor matrices from BTAS tensors into TA DistArrays
 * @param[in] decomp_world_rank 0 Processor rank in world which CP will be
 * computed with (CP is implemented using dense serial code in BTAS)
 * @param[in] als_threshold 1e-3 Stopping condition for CP ALS
 * Stops when the change in the loss function \f$ f(x) = | T - \hat{T}(x) |\f$
 * is less than @c als_threshold
 * @param[in] verbose false should ALS print the fit and change in fit for each
 * iteration
 **/
template <typename Tile, typename Policy>
auto btas_cp_als(madness::World& world, const DistArray<Tile, Policy> Reference,
                 long btas_cp_rank, TA::TiledRange1 rank_trange1,
                 std::size_t decomp_world_rank = 0, double als_threshold = 1e-3,
                 bool verbose = false) {
  using Tensor = btas::Tensor<typename Tile::value_type, btas::DEFAULT::range,
               btas::varray<typename Tile::value_type> >;
  auto ref_norm = TA::norm2(Reference);
  Tensor btas_ref;
  auto n_factors = TA::rank(Reference);
  std::vector<Tensor> btas_factors;
  btas_factors.reserve(n_factors + 1);

  if(world.rank() == decomp_world_rank){
    TA_ASSERT(Reference.world().size() > decomp_world_rank &&
              "TiledArray::cp::btas_cp_als(): must compute CP decomposition on a  "
              "single rank.");
    btas_ref = TA::array_to_btas_tensor<Tile, Policy, btas::DEFAULT::range,
    btas::varray<typename Tile::value_type>>(Reference, 0);
    using Tensor = decltype(btas_ref);
    btas::FitCheck<Tensor> fit(als_threshold);
    fit.set_norm(ref_norm);
    fit.verbose(verbose);

    btas::CP_ALS<Tensor, btas::FitCheck<Tensor>> CP_ALS(btas_ref);
    CP_ALS.compute_rank_random(btas_cp_rank, fit);
    btas_factors = CP_ALS.get_factor_matrices();

    // Scale the first factor matrix by the parallel factor, this choice is
    // arbitrary
    for (int i = 0; i < btas_cp_rank; i++) {
      btas::scal(btas_factors[0].extent(0),
                 btas_factors[n_factors](i),
                 std::begin(btas_factors[0]) + i, btas_cp_rank);
    }
    btas_factors.pop_back();
  }
  world.gop.fence();

  // Take each btas factor matrix turn it into a TA factor matrix.
  // Fill the vector which stores the TA factor matrices
  // Share the BTAS tensor vector of factor matrices
  world.gop.broadcast_serializable(btas_factors, decomp_world_rank);
  auto factor_number = 0;
  auto one_node = (world.size() == 1);
  std::vector<TA::DistArray<Tile, Policy> > TA_factors;
  TA_factors.reserve(n_factors);
  for (auto factor : btas_factors) {
    auto row_trange = Reference.trange().data()[factor_number];
    TiledArray::TiledRange trange({row_trange, rank_trange1});

    auto TA_factor = TiledArray::btas_tensor_to_array<TA::DistArray<Tile, Policy>>(
        world, trange, factor, !one_node);
    TA_factors.emplace_back(TA_factor);
    ++factor_number;
  }

  return TA_factors;
}

/**
 * REQUIRES BTAS
 * Calculates the canonical product (CP) decomposition of /c Reference to a
 * specific rank using a regularized alternating least squares (RALS) method in BTAS
 *
 * @param[in] world Madness world which CP will be computed on.
 * @param[in] Reference  Tensor to be CP decomposed.
 * @param[in] btas_cp_rank The rank of the CP decomposition
 * @param[in, out] rank_trange1 TiledRange1 object used to construct the CP
 * factor matrices from BTAS tensors into TA DistArrays
 * @param[in] decomp_world_rank 0 Processor rank in world which CP will be
 * computed with (CP is implemented using dense serial code in BTAS)
 * @param[in] als_threshold 1e-3 Stopping condition for CP ALS
 * Stops when the change in the loss function \f$ f(x) = | T - \hat{T}(x) |\f$
 * is less than @c als_threshold
 * @param[in] verbose false should ALS print the fit and change in fit for each
 * iteration
 **/
template <typename Tile, typename Policy>
auto btas_cp_rals(madness::World& world, DistArray<Tile, Policy> Reference,
                 long btas_cp_rank, TA::TiledRange1 rank_trange1,
                 std::size_t decomp_world_rank = 0, double als_threshold = 1e-3,
                 bool verbose = false) {
  using Tensor = btas::Tensor<typename Tile::value_type, btas::DEFAULT::range,
                              btas::varray<typename Tile::value_type> >;
  auto ref_norm = TA::norm2(Reference);
  Tensor btas_ref;
  auto n_factors = TA::rank(Reference);
  std::vector<Tensor> btas_factors;
  btas_factors.reserve(n_factors + 1);

  if(world.rank() == decomp_world_rank){
    TA_ASSERT(Reference.world().size() > decomp_world_rank &&
              "TiledArray::cp::btas_cp_als(): must compute CP decomposition on a  "
              "single rank.");
    btas_ref = TA::array_to_btas_tensor<Tile, Policy,  btas::DEFAULT::range,
                                        btas::varray<typename Tile::value_type>>(Reference, 0);
    using Tensor = decltype(btas_ref);
    btas::FitCheck<Tensor> fit(als_threshold);
    fit.set_norm(ref_norm);
    fit.verbose(verbose);

    btas::CP_RALS<Tensor, btas::FitCheck<Tensor>> CP_ALS(btas_ref);
    CP_ALS.compute_rank_random(btas_cp_rank, fit);
    btas_factors = CP_ALS.get_factor_matrices();

    // Scale the first factor matrix by the parallel factor, this choice is
    // arbitrary
    for (int i = 0; i < btas_cp_rank; i++) {
      btas::scal(btas_factors[0].extent(0),
                 btas_factors[n_factors](i),
                 std::begin(btas_factors[0]) + i, btas_cp_rank);
    }
    btas_factors.pop_back();
  }
  world.gop.fence();

  // Take each btas factor matrix turn it into a TA factor matrix.
  // Fill the vector which stores the TA factor matrices
  // Share the BTAS tensor vector of factor matrices
  world.gop.broadcast_serializable(btas_factors, decomp_world_rank);
  auto factor_number = 0;
  auto one_node = (world.size() == 1);
  std::vector<TA::DistArray<Tile, Policy> > TA_factors;
  TA_factors.reserve(n_factors);
  for (auto factor : btas_factors) {
    auto row_trange = Reference.trange().data()[factor_number];
    TiledArray::TiledRange trange({row_trange, rank_trange1});

    auto TA_factor = TiledArray::btas_tensor_to_array<TA::DistArray<Tile, Policy>>(
        world, trange, factor, !one_node);
    TA_factors.emplace_back(TA_factor);
    ++factor_number;
  }

  return TA_factors;
}


} // namespace TiledArray
#endif  // TiledArray_CP_BTAS_CP_H
