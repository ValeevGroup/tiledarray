//
// Created by Karl Pierce on 5/23/22.
//

#ifndef TILEDARRAY_CP_TN_CP_ALS__H
#define TILEDARRAY_CP_TN_CP_ALS__H
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
class TN_CP_ALS : public CP<Tile, Policy> {
 public:
  using CP<Tile, Policy>::ndim;
  using CP<Tile, Policy>::cp_factors;

  /// Default CP_ALS constructor
  TN_CP_ALS() = default;

  /// CP_ALS constructor function
  /// takes, as a constant reference, the tensor to be decomposed
  /// \param[in] tref A constant reference to the tensor to be decomposed.
  TN_CP_ALS(const DistArray<Tile, Policy>& trefL, const DistArray<Tile, Policy>& trefR,
            const expressions::TsrExpr<DistArray<Tile, Policy>> expr_L,
            const expressions::TsrExpr<DistArray<Tile, Policy>> expr_R,
            size_t numExternal)
      : CP<Tile, Policy>(numExternal), reference_left(trefL), reference_right(trefR) {
    TA_ASSERT(trefL.world() == trefR.world(), "madness::World of the two tensors must be the same");
    world = trefL.world();
//    for (size_t i = 0; i < ndim; ++i) {
//      ref_indices += detail::intToAlphabet(i);
//      if (i + 1 != ndim) ref_indices += ",";
//    }
//
//    first_gemm_dim_one = ref_indices;
//    first_gemm_dim_last = ref_indices;
//
//    first_gemm_dim_one.replace(0, 1, 1, detail::intToAlphabet(ndim));
//    first_gemm_dim_last = "," + first_gemm_dim_last;
//    first_gemm_dim_last.insert(0, 1, detail::intToAlphabet(ndim));
//    first_gemm_dim_last.pop_back();
//    first_gemm_dim_last.pop_back();
  }

 private:
  const DistArray<Tile, Policy> & reference_left, & reference_right;
  madness::World & world;
  const expressions::TsrExpr<DistArray<Tile, Policy> > expressionL, expressionR;
};
} // namespace TiledArray::cp
#endif  // TILEDARRAY_CP_TN_CP_ALS__H
