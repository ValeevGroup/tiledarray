//
// Created by Karl Pierce on 3/17/22.
//

#ifndef TILEDARRAY_CP_CP__H
#define TILEDARRAY_CP_CP__H

#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
#include <random>
#include <TiledArray/conversions/btas.h>

namespace TiledArray::cp{
namespace detail{
// A seed for the random number generator.
static inline unsigned int& random_seed_accessor(){
  static unsigned int value = 3;
  return value;
}

// given a rank and block size, this computes a
// trange for the rank dimension to be used to make the CP factors.
static inline TiledRange1 compute_trange1(size_t rank, size_t rank_block_size){
  std::size_t nblocks =
      (rank + rank_block_size - 1) / rank_block_size;
  auto dv = std::div((int) (rank + nblocks - 1), (int) nblocks);
  auto avg_block_size = dv.quot - 1, num_avg_plus_one = dv.rem + 1;

  TiledArray::TiledRange1 new_trange1;
  {
    std::vector<std::size_t> new_trange1_v;
    new_trange1_v.reserve(nblocks + 1);
    auto block_counter = 0;
    for(auto i = 0; i < num_avg_plus_one; ++i, block_counter += avg_block_size + 1){
      new_trange1_v.emplace_back(block_counter);
    }
    for (auto i = num_avg_plus_one; i < nblocks; ++i, block_counter+= avg_block_size) {
      new_trange1_v.emplace_back(block_counter);
    }
    new_trange1_v.emplace_back(rank);
    new_trange1 = TiledArray::TiledRange1(new_trange1_v.begin(), new_trange1_v.end());
  }
  return new_trange1;
}

static inline char intToAlphabet( int i ){
  return static_cast<char>('a' + i);
}

} // namespace TiledArray::cp::detail

/**
 * This is a base class for the canonical polyadic (CP)
 * decomposition. The decomposition, in general,
 * represents an order-N tensor as a set of order-2
 * tensors all coupled by a hyperdimension called the rank.
 * In this base class we have functions/variables that are universal to
 * all CP decomposition strategies.
 * In general one must be able to decompose a tensor to a specific rank
 * (this provides a relatively fixed error in the CP loss function)
 * or to a specific error in the loss function (this is accomplished by
 * varying the rank).
 * @tparam Tile typing for the DistArray tiles
 * @tparam Policy policy of the DistArray
 *
 * This base class will be constructed by dependent classes
 * which will be used to actually decompose a specific tensor/tensor-network.
**/
template<typename Tile, typename Policy>
class CP {
  using Array = DistArray<Tile, Policy>;
 public:
  /// Generic construction class
  CP() = default;

  /// This function should be used by dependent classes
  /// @param[in] n_factors The number of factor matrices in
  /// the CP problem.
  CP(size_t n_factors) : ndim(n_factors){
    cp_factors.reserve(n_factors);
    partial_grammian.reserve(n_factors);
    for(auto i = 0; i < ndim; ++i){
      Array W;
      partial_grammian.emplace_back(W);
    }
  }

  /// Generic deconstructor
  ~CP() = default;

  /// This function computes the CP decomposition to rank @c rank
  /// There are 2 options, if @c build_rank the rank starts at 1 and
  /// moving to @c rank else builds an efficient
  /// random guess with rank @c rank
  /// \param[in] rank Rank of the CP deccomposition
  /// \param[in] rank_block_size 0; What is the size of the blocks
  /// in the rank mode's TiledRange, will compute TiledRange1 inline.
  /// if 0 : rank_blocck_size = rank.
  /// \param[in] build_rank should CP approximation be built from rank 1
  /// or set.
  /// \param[in] epsilonALS 1e-3; the stopping condition for the ALS solver
  /// \param[in] verbose false; should check fit print fit information.
  /// \returns the fit: \f$ 1.0 - |T_{\text{exact}} - T_{\text{approx}} | \f$
  double compute_rank(size_t rank,
                      size_t rank_block_size = 0,
                      bool build_rank = false,
                      double epsilonALS= 1e-3,
                      bool verbose = false){
    rank_block_size = (rank_block_size == 0 ? rank: rank_block_size);
    double epsilon = 1.0;
    fit_tol = epsilonALS;
    TiledRange1 rank_trange;
    if(build_rank){
      size_t cur_rank = 1;
      do{
        rank_trange = detail::compute_trange1(cur_rank, rank_block_size);
        build_guess(cur_rank, rank_trange);
        ALS(cur_rank, 100, verbose);
        ++cur_rank;
      }while(cur_rank < rank);
    } else{
      rank_trange = detail::compute_trange1(rank, rank_block_size);
      build_guess(rank, rank_trange);
      ALS(rank, 100, verbose);
    }
    return epsilon;
  }

  /// This function computes the CP decomposition with an
  /// error less than @c error in the CP fit
  /// \f$ |T_{\text{exact}} - T_{\text{approx}} | < error \f$
  /// \param[in] error Acceptable error in the CP decomposition
  /// \param[in] max_rank Maximum acceptable rank.
  /// \param[in] rank_block_size 0; What is the size of the blocks
  /// in the rank mode's TiledRange, will compute TiledRange1 inline.
  /// if 0 : rank_blocck_size = max_rank.
  /// \param[in] epsilonALS 1e-3; the stopping condition for the ALS solver
  /// \param[in] verbose false; should check fit print fit information.
  /// \returns the fit: \f$1.0 - |T_{\text{exact}} - T_{\text{approx}} | \f$
  double compute_error(double error,
                       size_t max_rank,
                       size_t rank_block_size = 0,
                       double epsilonALS = 1e-3,
                       bool verbose = false){
    rank_block_size = (rank_block_size == 0 ? max_rank : rank_block_size);
    size_t cur_rank = 1;
    double epsilon = 1.0;
    fit_tol = epsilonALS;
    do{
      auto rank_trange = detail::compute_trange1(cur_rank, rank_block_size);
      build_guess(cur_rank, rank_trange);
      ALS(cur_rank, 100, verbose);
      ++cur_rank;
    }while(epsilon > error && cur_rank < max_rank);

    return epsilon;
  }

 protected:
  std::vector<Array >
      cp_factors,                   // the CP factor matrices
      partial_grammian;             // square of the factor matrices (r x r)
  Array
      MTtKRP,                      // matricized tensor times
                                   // khatri rao product for check_fit()
      unNormalized_Factor,         // The final factor unnormalized
                                   // so you don't have to
                                   // deal with lambda for check_fit()
      lambda;                      // Column normalizations
  size_t ndim;                     // number of factor matrices
  double prev_fit = 1.0,           // The fit of the previous ALS iteration
         final_fit,                // The final fit of the ALS
                                   // optimization at fixed rank.
         fit_tol,                  // Tolerance for the ALS solver
         converged_num,            // How many times the ALS solver
                                   // has changed less than the tolerance
         norm_reference;           // used in determining the CP fit.

  /// This function is determined by the specific CP solver.
  /// builds the rank @c rank CP approximation and stores
  /// them in cp_factors.
  /// \param[in] rank rank of the CP approximation
  /// \param[in] rank_trange TiledRange1 of the rank dimension.
  virtual void build_guess(size_t rank,
                      TiledRange1 rank_trange) = 0;

  /// This function uses BTAS to construct a random
  /// factor matrix which can be used as an initial guess to any CP
  /// decomposition.
  /// \param[in] world madness::World of the new DistArray.
  /// \param[in] rank size of the CP rank.
  /// \param[in] mode_size size of the mode in the reference tensor(s).
  /// \param[in] trange1_rank TiledRange1 for the rank dimension.
  /// \param[in] trange1_mode TiledRange1 for the mode in the reference tensor.
  /// Should match the TiledRange1 in the reference tensor(s).
  Array
      construct_random_factor(madness::World & world,
                          size_t rank, size_t mode_size,
                          TiledArray::TiledRange1 trange1_rank,
                          TiledArray::TiledRange1 trange1_mode){
    using Tensor = btas::Tensor<typename Array::element_type,
                                btas::DEFAULT::range, btas::varray<typename Tile::value_type>>;
    Tensor factor(rank, mode_size);
    if(world.rank() == 0) {
      std::mt19937 generator(detail::random_seed_accessor());
      std::uniform_real_distribution<> distribution(-1.0, 1.0);
      for (auto ptr = factor.begin(); ptr != factor.end(); ++ptr)
        *ptr = distribution(generator);
    }
    world.gop.broadcast_serializable(factor, 0);

    return TiledArray::btas_tensor_to_array<
        Array >
        (world,
         TiledArray::TiledRange({trange1_rank, trange1_mode}),
         factor, (world.size() != 1));
  }
  /// This function is specified by the CP solver
  /// optimizes the rank @c rank CP approximation
  /// stored in cp_factors.
  /// \param[in] rank rank of the CP approximation
  /// \param[in] max_iter max number of ALS iterations
  /// \param[in] verbose Should ALS print fit information while running?
  virtual void ALS(size_t rank, size_t max_iter, bool verbose = false) = 0;

  /// This function leverages the fact that the grammian (W) is
  /// square and symmetric and solves the least squares (LS) problem Ax = B
  /// where A = \c W and B = \c MtKRP
  /// \param[in,out] MtKRP In: Matricized tensor times KRP Out: The solution to
  /// Ax = B.
  /// \param[in] W The grammian matrixed used to determine LS solution.
  void cholesky_inverse(Array & MtKRP,
                        const Array & W){
    //auto inv = TiledArray::math::linalg::cholesky_lsolve(NoTranspose,W, MtKRP);
    MtKRP = math::linalg::cholesky_solve(W, MtKRP);
  }

  /// Technically the Least squares problem requires doing a pseudoinverse
  /// if Cholesky fails revert to pseudoinverse.
  /// \param[in,out] MtKRP In: Matricized tensor times KRP Out: The solution to
  /// Ax = B.
  /// \param[in] W The grammian matrixed used to determine LS solution.
  /// \param[in] svd_invert_threshold Don't invert
  /// numerical 0 i.e. @c svd_invert_threshold
  void pseudo_inverse(Array & MtKRP,
                      const Array & W,
                      double svd_invert_threshold = 1e-12){
    // compute the SVD of W;
    auto SVD = TiledArray::svd<SVD::Vectors::AllVectors>(W);

    // Grab references to S, U and Vt
    auto& S = std::get<0>(SVD),
        & U = std::get<1>(SVD),
        & Vt = std::get<2>(SVD);

    // Walk through S and invert diagonal elements
    TiledArray::foreach_inplace(S, [&, svd_invert_threshold](auto& tile){
      auto const& range = tile.range();
      auto const lo = range.lobound_data();
      auto const up = range.upbound_data();
      for (auto n = lo[0]; n != up[0]; ++n) {
        auto& val = tile(n,n);
        if(val < svd_invert_threshold) continue;
        val = 1.0 / val;
      }},true);

    MtKRP("r,n") = (U("r,s") * S("s,sp")) * (Vt("sp,rp") * MtKRP("rp,n"));
  }

  /// Computes the grammian using partial_grammian vector.
  /// If the partial_grammian vector is empty, compute using factors.
  /// \returns TiledArray::DistArray <Tile,Policy> with grammian values.
  auto compute_grammian(){
    auto trange_rank = cp_factors[0].trange().data()[0];
    Array W(trange_rank, trange_rank);
    W.fill(1.0);
    if(partial_grammian.empty()){
      for(auto ptr = cp_factors.begin(); ptr != cp_factors.end(); ++ptr){
        W("r,rp") *= (*ptr)("r,n") * (*ptr)("rp, n");
      }
    } else{
      for(auto ptr = partial_grammian.begin(); ptr != partial_grammian.end(); ++ptr){
        W("r,rp") *= (*ptr)("r,rp");
      }
    }
    return W;
  }

  /// Function to column normalize factor matrices
  /// \param[in, out] factor in: un-normalized factor matrix.
  /// out: column normalized factor matrix
  /// \param[in] rank_trange TiledRange for the rank mode
  /// \returns vector of column normalization factors
  std::vector<typename Tile::value_type>
      normalize_factor(Array & factor,
                   size_t cp_rank,
                   TiledRange rank_trange){
    auto & world = factor.world();
    auto lambda = this->temp_normCol(factor, cp_rank);
    std::vector<typename Tile::value_type> inv_lambda(lambda);
    for(auto & i : inv_lambda) i = (i > 1e-12 ? 1 / i : i);
    auto diag_lambda = diagonal_array<Array>(world, rank_trange,
                                                               inv_lambda.begin(), inv_lambda.end());
    factor("rp,n") = diag_lambda("rp,r") * factor("r,n");
    //std::cout << factor << std::endl;
    //cp_factors.emplace_back(factor);
    return lambda;
  }

  /// computes the column normalization of a given factor matrix \c factor
  /// stores the column norms in the lambda vector.
  /// Also normalizes the columns of \c factor
  /// \param[in,out] factor in: unnormalized factor matrix, out: column
  /// normalized factor matrix
  void normCol(Array &factor, size_t rank){
    auto & world = factor.world();
    //lambda = expressions::einsum(factor("r,n"), factor("r,n"), "r");
    //std::vector<typename Tile::value_type> lambda_vector;
    //lambda_vector.reserve(rank);
    auto lambda_vector = temp_normCol(factor, rank);
//    foreach_inplace(lambda, [&](auto& tile){
//      for(auto & i : tile) {
//        i = sqrt((i));
//        lambda_vector.emplace_back(i);
//      }
//    }, true);
    auto & tr1_rank = factor.trange().data()[0];
    auto diag_ = diagonal_array<Array>(world,
                   TiledArray::TiledRange({tr1_rank, tr1_rank}),
                   lambda_vector.begin(), lambda_vector.end());
    //std::cout << factor << std::endl;
    factor("rp,n") =  diag_("rp,r") * factor("r,n");
    //std::cout << factor << std::endl;
  }

  std::vector<double> temp_normCol(Array & factor,
                                   size_t rank){
    std::vector<double> lambda(rank);
    auto & world = factor.world();
    if(world.rank() == 0) {
      auto btas_factor = array_to_btas_tensor(factor, 0);
      TA_ASSERT(rank == btas_factor.extent(0));
      std::fill(lambda.begin(), lambda.end(), 0.0);
      auto lambda_ptr = lambda.begin();
      auto col_dim = btas_factor.extent(1);
      auto bf_ptr = btas_factor.data();
      for(auto r = 0; r < rank; ++r,
                ++lambda_ptr, bf_ptr+=col_dim){
        for(auto n = 0; n < col_dim; ++n) {
          auto val = *(bf_ptr + n);
          *(lambda_ptr) += val * val;
        }
        *(lambda_ptr) = sqrt(*(lambda_ptr));
        auto one_over_lam = (*(lambda_ptr) > 1e-12 ? 1/ *(lambda_ptr) : *(lambda_ptr));
        for(auto n = 0; n < col_dim; ++n){
          *(bf_ptr + n) *= one_over_lam;
        }
      }
    }
    world.gop.broadcast_serializable(lambda, 0);
    world.gop.fence();
    return lambda;
  }

  /// This function checks the fit and change in the
  /// fit for the CP loss function
  /// \f$ fit = 1.0 - |T - T_{CP}| = 1.0 - T^2 + 2 T T_{CP} - T_{CP}^2\f$
  /// must have small change in fit 2 times to return true.
  /// this function can be defined in the derived class if
  /// nonstandard CP loss function
  /// \param[in] verbose false; Should the fit and change
  /// in fit be printed each call?
  /// \returns bool : is the change in fit less than the ALS tolerance?
  virtual bool check_fit(bool verbose = false){
    // Compute the inner product T * T_CP
    double inner_prod = MTtKRP("r,n").dot(unNormalized_Factor("r,n"));
    // compute the square of the CP tensor (can use the grammian)
    auto factor_norm=[&](){
      auto gram_ptr = partial_grammian.begin();
      Array W(*(gram_ptr));
      ++gram_ptr;
      for(size_t i = 0; i < ndim -1; ++i, ++gram_ptr){
        W("r,rp") *= (*gram_ptr)("r,rp");
      }
      return sqrt(W("r,rp").dot((unNormalized_Factor("r,n") * unNormalized_Factor("rp,n"))));
    };
    // compute the error in the loss function and find the fit
    double normFactors = factor_norm(),
           normResidual = sqrt(norm_reference * norm_reference + normFactors * normFactors - 2.0 * inner_prod),
           fit = 1.0 - (normResidual / norm_reference),
           fit_change = abs(prev_fit - fit);
    prev_fit = fit;
    // print fit data if required
    if(verbose){
      std::cout << fit << "\t" << fit_change << std::endl;
    }

    // if the change in fit is less than the tolerance try to return true.
    if(fit_change < fit_tol){
      converged_num++;
      if(converged_num == 2){
        converged_num = 0;
        final_fit = prev_fit;
        prev_fit = 1.0;
        return true;
      }
    }
    return false;
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
