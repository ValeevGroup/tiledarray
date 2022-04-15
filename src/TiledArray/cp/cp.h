//
// Created by Karl Pierce on 3/17/22.
//

#ifndef TILEDARRAY_CP_CP__H
#define TILEDARRAY_CP_CP__H

#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
namespace TiledArray::cp{

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
 public:
  /// Generic construction class
  CP() = default;

  /// This function should be used by dependent classes
  /// @param[in] n_factors The number of factor matrices in
  /// the CP problem.
  CP(size_t n_factors) : ndim(n_factors){
    cp_factors.reserve(n_factors);
    partial_grammian.reserve(n_factors);
  }

  /// Generic deconstructor
  ~CP() = default;

  /// This function computes the CP decomposition to rank @c rank
  /// There are 2 options, if @c build_rank the rank starts at 1 and
  /// moving to @c rank else builds an efficient
  /// random guess with rank @c rank
  /// \param[in] rank Rank of the CP deccomposition
  /// \param[in] build_rank should CP approximation be built from rank 1
  /// or set.
  /// \param[in] epsilonALS 1e-3; the stopping condition for the ALS solver
  /// \returns the fit: \f$ 1.0 - |T_{\text{exact}} - T_{\text{approx}} | \f$
  double compute_rank(size_t rank, bool build_rank = false,
                      double epsilonALS= 1e-3){
    double epsilon = 1.0;
    fit_tol = epsilonALS;
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

  /// This function computes the CP decomposition with an
  /// error less than @c error in the CP fit
  /// \f$ |T_{\text{exact}} - T_{\text{approx}} | < error \f$
  /// \param[in] error Acceptable error in the CP decomposition
  /// \param[in] max_rank Maximum acceptable rank.
  /// \param[in] epsilonALS 1e-3; the stopping condition for the ALS solver
  /// \returns the fit: \f$1.0 - |T_{\text{exact}} - T_{\text{approx}} | \f$
  double compute_error(double error, size_t max_rank,
                       double epsilonALS = 1e-3){
    size_t cur_rank = 1;
    double epsilon = 1.0;
    fit_tol = epsilonALS;
    do{
      build_guess(cur_rank);
      ALS(cur_rank);
      ++cur_rank;
    }while(epsilon > error && cur_rank < max_rank);

    return epsilon;
  }

 protected:
  std::vector<TiledArray::DistArray<Tile, Policy> >
      cp_factors,                   // the CP factor matrices
      partial_grammian;             // square of the factor matrices (r x r)
  TiledArray::DistArray<Tile, Policy>
      MTtKRP,                      // matricized tensor times
                                   // khatri rao product for check_fit()
      unNormalized_Factor;         // The final factor unnormalized
                                   // so you don't have to
                                   // deal with lambda for check_fit()
  std::vector<double> lambda;      // Column normalizations
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
  virtual void build_guess(const size_t rank);

  /// This function is specified by the CP solver
  /// optimizes the rank @c rank CP approximation
  /// stored in cp_factors.
  /// \param[in] rank rank of the CP approximation
  virtual void ALS(size_t rank);

  /// This function leverages the fact that the grammian (W) is
  /// square and symmetric and solves the least squares (LS) problem Ax = B
  /// where A = \c W and B = \c MtKRP
  /// \param[in,out] MtKRP In: Matricized tensor times KRP Out: The solution to
  /// Ax = B.
  /// \param[in] W The grammian matrixed used to determine LS solution.
  void cholesky_inverse(TiledArray::DistArray<Tile, Policy> & MtKRP,
                        const TiledArray::DistArray<Tile, Policy> & W){
    MtKRP = TiledArray::math::linalg::cholesky_lsolve(W, MtKRP);
  }

  /// Technically the Least squares problem requires doing a pseudoinverse
  /// if Cholesky fails revert to pseudoinverse.
  /// \param[in,out] MtKRP In: Matricized tensor times KRP Out: The solution to
  /// Ax = B.
  /// \param[in] W The grammian matrixed used to determine LS solution.
  /// \param[in] svd_invert_threshold Don't invert
  /// numerical 0 i.e. @c svd_invert_threshold
  void pseudo_inverse(TiledArray::DistArray<Tile, Policy> & MtKRP,
                      const TiledArray::DistArray<Tile, Policy> & W,
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
    TiledArray::DistArray<Tile, Policy> W(trange_rank, trange_rank);
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

  /// computes the column normalization of a given factor matrix \c factor
  /// stores the column norms in the lambda vector.
  /// Also normalizes the columns of \c factor
  /// \param[in,out] factor in: unnormalized factor matrix, out: column
  /// normalized factor matrix
  void normCol(TiledArray::DistArray<Tile, Policy> &factor){

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
      TiledArray::DistArray<Tile, Policy> W(*(gram_ptr));
      ++gram_ptr;
      for(size_t i = 0; i < ndim -1; ++i, ++gram_ptr){
        W("r,rp") *= *(gram_ptr)("r,rp");
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
