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

  /// This function computes the CP decomposition with an
  /// error less than @c error in the CP fit
  /// \f$ |T_{\text{exact}} - T_{\text{approx}} | < error \f$
  /// \param[in] error Acceptable error in the CP decomposition
  /// \param[in] max_rank Maximum acceptable rank.
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
  std::vector<TiledArray::DistArray<Tile, Policy> >
      cp_factors,                   // the CP factor matrices
      partial_grammian;             // square of the factor matrices (r x r)
  TiledArray::DistArray<Tile, Policy>
      MTtKRP;                      // matricized tensor times khatri rao product
  std::vector<double> lambda;      // Column normalizations
  size_t ndim;                     // number of factor matrices
  double prev_fit = 1.0,
         norm_reference;          // used in determining the CP fit.

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

  }

  /// Technically the Least squares problem requires doing a pseudoinverse
  /// if Cholesky fails revert to pseudoinverse.
  /// \param[in,out] MtKRP In: Matricized tensor times KRP Out: The solution to
  /// Ax = B.
  /// \param[in] W The grammian matrixed used to determine LS solution.
  void pseudo_inverse(TiledArray::DistArray<Tile, Policy> & MtKRP,
                      const TiledArray::DistArray<Tile, Policy> & W){

  }

  /// computes the column normalization of a given factor matrix \c factor
  /// stores the column norms in the lambda vector.
  /// Also normalizes the columns of \c factor
  /// \param[in,out] factor in: unnormalized factor matrix, out: column
  /// normalized factor matrix
  void normCol(TiledArray::DistArray<Tile, Policy> &factor){

  }

  virtual bool check_fit(){

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
