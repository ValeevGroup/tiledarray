/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2023  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Karl Pierce
 *  Department of Chemistry, Virginia Tech
 *
 *  cp.h
 *  March 17, 2022
 *
 */

#ifndef TILEDARRAY_MATH_SOLVERS_CP_CP__H
#define TILEDARRAY_MATH_SOLVERS_CP_CP__H

#include <TiledArray/conversions/btas.h>
#include <TiledArray/expressions/einsum.h>

namespace TiledArray::math::cp {

namespace detail {

static inline char intToAlphabet(int i) { return static_cast<char>('a' + i); }

}  // namespace detail

/**
 * This is a base class for the canonical polyadic (CP)
 * decomposition solver. The decomposition, in general,
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
template <typename Tile, typename Policy>
class CP {
  using Array = DistArray<Tile, Policy>;

 public:
  /// Generic construction class
  CP() = default;

  /// This function should be used by dependent classes
  /// @param[in] n_factors The number of factor matrices in
  /// the CP problem.
  CP(size_t n_factors) : ndim(n_factors) {
    cp_factors.reserve(n_factors);
    partial_grammian.reserve(n_factors);
    for (auto i = 0; i < ndim; ++i) {
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
  /// \param[in] rank Rank of the CP decomposition
  /// \param[in] rank_block_size The target tile size of
  /// rank mode's range; if 0 will use \p rank as \p rank_block_size .
  /// \param[in] build_rank should CP approximation be built from rank 1
  /// or set.
  /// \param[in] epsilonALS 1e-3; the stopping condition for the ALS solver
  /// \param[in] verbose false; should check fit print fit information.
  /// \returns the fit: \f$ 1.0 - |T_{\text{exact}} - T_{\text{approx}} | \f$
  double compute_rank(size_t rank, size_t rank_block_size = 0,
                      bool build_rank = false, double epsilonALS = 1e-3,
                      bool verbose = false) {
    rank_block_size = (rank_block_size == 0 ? rank : rank_block_size);
    double epsilon = 1.0;
    fit_tol = epsilonALS;
    TiledRange1 rank_trange;
    if (build_rank) {
      size_t cur_rank = 1;
      do {
        rank_trange = TiledRange1::make_uniform(cur_rank, rank_block_size);
        build_guess(cur_rank, rank_trange);
        ALS(cur_rank, 100, verbose);
        ++cur_rank;
      } while (cur_rank < rank);
    } else {
      rank_trange = TiledRange1::make_uniform(rank, rank_block_size);
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
  double compute_error(double error, size_t max_rank,
                       size_t rank_block_size = 0, double epsilonALS = 1e-3,
                       bool verbose = false) {
    rank_block_size = (rank_block_size == 0 ? max_rank : rank_block_size);
    size_t cur_rank = 1;
    double epsilon = 1.0;
    fit_tol = epsilonALS;
    do {
      auto rank_trange = TiledRange1::make_uniform(cur_rank, rank_block_size);
      build_guess(cur_rank, rank_trange);
      ALS(cur_rank, 100, verbose);
      ++cur_rank;
    } while (epsilon > error && cur_rank < max_rank);

    return epsilon;
  }

  std::vector<Array> get_factor_matrices() {
    TA_ASSERT(!cp_factors.empty(),
              "CP factor matrices have not been computed)");
    auto result = cp_factors;
    result.pop_back();
    result.emplace_back(unNormalized_Factor);
    return result;
  }

  Array reconstruct() {
    TA_ASSERT(!cp_factors.empty(),
              "CP factor matrices have not been computed)");
    std::string lhs("r,0"), rhs("r,"), final("r,0");
    Array krp = cp_factors[0];
    for (size_t i = 1; i < ndim - 1; ++i) {
      rhs += std::to_string(i);
      final += "," + std::to_string(i);
      krp = expressions::einsum(krp(lhs), cp_factors[i](rhs), final);
      lhs = final;
      rhs.pop_back();
    }
    rhs += std::to_string(ndim - 1);
    final.erase(final.begin(), final.begin() + 2);
    final += "," + std::to_string(ndim - 1);
    krp(final) = krp(lhs) * unNormalized_Factor(rhs);
    return krp;
  }

 protected:
  std::vector<Array> cp_factors,  // the CP factor matrices
      partial_grammian;           // square of the factor matrices (r x r)
  Array MTtKRP,                   // matricized tensor times
                                  // khatri rao product for check_fit()
      unNormalized_Factor,        // The final factor unnormalized
                                  // so you don't have to
                                  // deal with lambda for check_fit()
      lambda;                     // column normalizations
  // std::vector<typename Tile::value_type>
  //     lambda;                      // Column normalizations

  size_t ndim;            // number of factor matrices
  double prev_fit = 1.0,  // The fit of the previous ALS iteration
      final_fit,          // The final fit of the ALS
                          // optimization at fixed rank.
      fit_tol,            // Tolerance for the ALS solver
      norm_reference;     // used in determining the CP fit.
  std::size_t converged_num =
      0;  // How many times the ALS solver
          // has changed less than the tolerance in a row

  /// This function is determined by the specific CP solver.
  /// builds the rank @c rank CP approximation and stores
  /// them in cp_factors.
  /// \param[in] rank rank of the CP approximation
  /// \param[in] rank_trange TiledRange1 of the rank dimension.
  virtual void build_guess(size_t rank, TiledRange1 rank_trange) = 0;

  /// This function uses BTAS to construct a random
  /// factor matrix which can be used as an initial guess to any CP
  /// decomposition.
  /// \param[in] world madness::World of the new DistArray.
  /// \param[in] rank size of the CP rank.
  /// \param[in] mode_size size of the mode in the reference tensor(s).
  /// \param[in] trange1_rank TiledRange1 for the rank dimension.
  /// \param[in] trange1_mode TiledRange1 for the mode in the reference tensor.
  /// Should match the TiledRange1 in the reference tensor(s).
  Array construct_random_factor(madness::World& world, size_t rank,
                                size_t mode_size,
                                TiledArray::TiledRange1 trange1_rank,
                                TiledArray::TiledRange1 trange1_mode) {
    using Tensor =
        btas::Tensor<typename Array::element_type, btas::DEFAULT::range,
                     btas::varray<typename Tile::value_type>>;
    Tensor factor(rank, mode_size);
    auto lambda = std::vector<typename Tile::value_type>(
        rank, (typename Tile::value_type)0);
    if (world.rank() == 0) {
      auto factor_ptr = factor.data();
      size_t offset = 0;
      for (auto r = 0; r < rank; ++r, offset += mode_size) {
        auto lam_ptr = lambda.data() + r;
        for (auto m = offset; m < offset + mode_size; ++m) {
          auto val = TiledArray::drand() * 2 - 1;  // random number in [-1,1]
          *(factor_ptr + m) = val;
          *lam_ptr += val * val;
        }
        *lam_ptr = sqrt(*lam_ptr);
        auto inv = (*lam_ptr < 1e-12 ? 1.0 : 1.0 / (*lam_ptr));
        for (auto m = 0; m < mode_size; ++m) {
          *(factor_ptr + offset + m) *= inv;
        }
      }
    }
    world.gop.broadcast_serializable(factor, 0);
    world.gop.broadcast_serializable(lambda, 0);

    return TiledArray::btas_tensor_to_array<Array>(
        world, TiledArray::TiledRange({trange1_rank, trange1_mode}), factor,
        (world.size() != 1));
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
  void cholesky_inverse(Array& MtKRP, const Array& W) {
    // auto inv = TiledArray::math::linalg::cholesky_lsolve(NoTranspose,W,
    // MtKRP);
    try {
      MtKRP = math::linalg::cholesky_solve(W, MtKRP);
    } catch (std::runtime_error& ex) {
      // if W is near-singular try LU instead of Cholesky
      if (std::string(ex.what()).find("lapack::posv failed") !=
          std::string::npos) {
        MtKRP = math::linalg::lu_solve(W, MtKRP);
      } else  // rethrow other exceptions
        throw ex;
    }
  }

  /// Technically the Least squares problem requires doing a pseudoinverse
  /// if Cholesky fails revert to pseudoinverse.
  /// \param[in,out] MtKRP In: Matricized tensor times KRP Out: The solution to
  /// Ax = B.
  /// \param[in] W The grammian matrixed used to determine LS solution.
  /// \param[in] svd_invert_threshold Don't invert
  /// numerical 0 i.e. @c svd_invert_threshold
  void pseudo_inverse(Array& MtKRP, const Array& W,
                      double svd_invert_threshold = 1e-12) {
    // compute the SVD of W;
    auto SVD = TiledArray::svd<SVD::Vectors::AllVectors>(W);

    // Grab references to S, U and Vt
    auto &S = std::get<0>(SVD), &U = std::get<1>(SVD), &Vt = std::get<2>(SVD);

    // Walk through S and invert diagonal elements
    TiledArray::foreach_inplace(
        S,
        [&, svd_invert_threshold](auto& tile) {
          auto const& range = tile.range();
          auto const lo = range.lobound_data();
          auto const up = range.upbound_data();
          for (auto n = lo[0]; n != up[0]; ++n) {
            auto& val = tile(n, n);
            if (val < svd_invert_threshold) continue;
            val = 1.0 / val;
          }
        },
        /* fence = */ true);

    MtKRP("r,n") = (U("r,s") * S("s,sp")) * (Vt("sp,rp") * MtKRP("rp,n"));
  }

  /// normalizes "columns" (aka rows) of an updated factor matrix

  /// rows of factor matrices produced by least-squares are not unit
  /// normalized. This takes each row and makes it unit normalized,
  /// with inverse of the normalization factor stored in this->lambda
  /// \param[in,out] factor in: unnormalized factor matrix, out:
  /// normalized factor matrix
  void normalize_factor(Array& factor) {
    auto& world = factor.world();
    // this is what the code should look like, but expressions::einsum seems to
    // be buggy lambda contains squared norms of rows
    lambda = expressions::einsum(factor("r,n"), factor("r,n"), "r");

    // element-wise square root to convert squared norms to norms
    TiledArray::foreach_inplace(
        lambda,
        [](Tile& tile) {
          auto lo = tile.range().lobound_data();
          auto up = tile.range().upbound_data();
          for (auto R = lo[0]; R < up[0]; ++R) {
            const auto norm_squared_RR = tile({R});
            using std::sqrt;
            tile({R}) = sqrt(norm_squared_RR);
          }
        },
        /* fence = */ true);
    lambda.truncate();
    lambda.make_replicated();
    auto lambda_eig = array_to_eigen(lambda);

    TiledArray::foreach_inplace(
        factor,
        [&lambda_eig](Tile& tile) {
          auto lo = tile.range().lobound_data();
          auto up = tile.range().upbound_data();
          for (auto R = lo[0]; R < up[0]; ++R) {
            const auto lambda_R = lambda_eig(R, 0);
            if (lambda_R < 1e-12) continue;
            auto scale_by = 1.0 / lambda_R;
            for (auto N = lo[1]; N < up[1]; ++N) {
              tile(R, N) *= scale_by;
            }
          }
        },
        /* fence = */ true);
    factor.truncate();
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
  virtual bool check_fit(bool verbose = false) {
    // Compute the inner product T * T_CP
    const auto ref_dot_cp = MTtKRP("r,n").dot(unNormalized_Factor("r,n"));
    // compute the square of the CP tensor (can use the grammian)
    auto factor_norm = [&]() {
      auto gram_ptr = partial_grammian.begin();
      Array W(*(gram_ptr));
      ++gram_ptr;
      for (size_t i = 1; i < ndim - 1; ++i, ++gram_ptr) {
        W("r,rp") *= (*gram_ptr)("r,rp");
      }
      auto result = sqrt(W("r,rp").dot(
          (unNormalized_Factor("r,n") * unNormalized_Factor("rp,n"))));
      // not sure why need to fence here, but hang periodically without it
      W.world().gop.fence();
      return result;
    };
    // compute the error in the loss function and find the fit
    const auto norm_cp = factor_norm();  // ||T_CP||_2
    const auto squared_norm_error = norm_reference * norm_reference +
                                    norm_cp * norm_cp -
                                    2.0 * ref_dot_cp;  // ||T - T_CP||_2^2
    // N.B. squared_norm_error is very noisy
    // TA_ASSERT(squared_norm_error >= - 1e-8);
    const auto norm_error = sqrt(abs(squared_norm_error));
    const auto fit = 1.0 - (norm_error / norm_reference);
    const auto fit_change = fit - prev_fit;
    prev_fit = fit;
    // print fit data if required
    if (verbose) {
      std::cout << MTtKRP.world().rank() << ": fit=" << fit
                << " fit_change=" << fit_change << std::endl;
    }

    // if the change in fit is less than the tolerance try to return true.
    if (abs(fit_change) < fit_tol) {
      converged_num++;
      if (converged_num == 2) {
        converged_num = 0;
        final_fit = prev_fit;
        prev_fit = 1.0;
        if (verbose)
          std::cout << MTtKRP.world().rank() << ": converged" << std::endl;
        return true;
      } else {
        TA_ASSERT(converged_num == 1);
        if (verbose)
          std::cout << MTtKRP.world().rank() << ": pre-converged" << std::endl;
      }
    } else {
      converged_num = 0;
    }
    return false;
  }
};

}  // namespace TiledArray::math::cp

#endif  // TILEDARRAY_MATH_SOLVERS_CP_CP__H
