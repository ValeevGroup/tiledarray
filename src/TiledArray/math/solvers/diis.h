/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  diis.h
 *  May 20, 2013
 *
 */

#ifndef TILEDARRAY_MATH_SOLVERS_DIIS_H__INCLUDED
#define TILEDARRAY_MATH_SOLVERS_DIIS_H__INCLUDED

#include <TiledArray/math/linalg/basic.h>
#include "TiledArray/dist_array.h"
#include "TiledArray/external/eigen.h"

#include <Eigen/QR>
#include <deque>

namespace TiledArray::math {

/// DIIS (``direct inversion of iterative subspace'') extrapolation

/// The DIIS class provides DIIS extrapolation to an iterative solver of
/// (systems of) linear or nonlinear equations of the \f$ f(x) = 0 \f$ form,
/// where \f$ f(x) \f$ is a (non-linear) function of \f$ x \f$ (in general,
/// \f$ x \f$ is a set of numeric values). Such equations are usually solved
/// iteratively as follows:
/// \li given a current guess at the solution, \f$ x_i \f$, evaluate the error
///     (``residual'') \f$ e_i = f(x_i) \f$ (NOTE that the dimension of
///     \f$ x \f$ and \f$ e \f$ do not need to coincide);
/// \li use the error to compute an updated guess \f$ x_{i+1} = x_i + g(e_i)
/// \f$; \li proceed until a norm of the error is less than the target precision
///     \f$ \epsilon \f$. Another convergence criterion may include
///     \f$ ||x_{i+1} - x_i|| < \epsilon \f$ .
///
/// For example, in the Hartree-Fock method in the density form, one could
/// choose \f$ x \equiv \mathbf{P} \f$, the one-electron density matrix, and
/// \f$ f(\mathbf{P}) \equiv [\mathbf{F}, \mathbf{P}] \f$ , where
/// \f$ \mathbf{F} = \mathbf{F}(\mathbf{P}) \f$ is the Fock matrix, a linear
/// function of the density. Because \f$ \mathbf{F} \f$ is a linear function
/// of the density and DIIS uses a linear extrapolation, it is possible to
/// just extrapolate the Fock matrix itself, i.e. \f$ x \equiv \mathbf{F} \f$
/// and \f$ f(\mathbf{F}) \equiv [\mathbf{F}, \mathbf{P}] \f$ .
///
/// Similarly, in the Hartree-Fock method in the molecular orbital
/// representation, DIIS is used to extrapolate the Fock matrix, i.e.
/// \f$ x \equiv \mathbf{F} \f$ and \f$ f(\mathbf{F}) \equiv \{ F_i^a \} \f$ ,
/// where \f$ i \f$ and \f$ a \f$ are the occupied and unoccupied orbitals,
/// respectively.
///
/// Here's a short description of the DIIS method. Given a set of solution
/// guess vectors \f$ \{ x_k \}, k=0..i \f$ and the corresponding error
/// vectors \f$ \{ e_k \} \f$ DIIS tries to find a linear combination of
/// \f$ x \f$ that would minimize the error by solving a simple linear system
/// set up from the set of errors. The solution is a vector of coefficients
/// \f$ \{ C_k \} \f$ that can be used to obtain an improved \f$ x \f$:
/// \f$ x_{\mathrm{extrap},i+1} = \sum\limits_{k=0}^i C_{k,i} x_{k} \f$
/// A more complicated version of DIIS introduces mixing:
/// \f$ x_{\mathrm{extrap},i+1} = \sum\limits_{k=0}^i C_{k,i} ( (1-f) x_{k} + f
/// x_{extrap,k} ) \f$ Note that the mixing is not used in the first iteration.
///
/// The original DIIS reference: P. Pulay, Chem. Phys. Lett. 73, 393 (1980).
///
/// \tparam D type of \c x
template <typename D>
class DIIS {
 public:
  typedef typename D::numeric_type value_type;
  typedef typename TiledArray::detail::scalar_t<value_type> scalar_type;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      Matrix;
  typedef Eigen::Matrix<value_type, Eigen::Dynamic, 1> Vector;

  /// Constructor

  /// \param strt The DIIS extrapolation will begin on the iteration given
  ///   by this integer (default = 1).
  /// \param ndi This integer maximum number of data sets to retain (default
  ///   = 5).
  /// \param dmp This nonnegative floating point number is used to dampen
  ///   the DIIS extrapolation (default = 0.0).
  /// \param ngr The number of iterations in a DIIS group. DIIS
  ///   extrapolation is only used for the first \c ngrdiis of these
  ///   iterations (default = 1). If \c ngr is 1 and \c ngrdiis is
  ///   greater than 0, then DIIS will be used on all iterations after and
  ///   including the start iteration.
  /// \param ngrdiis The number of DIIS extrapolations to do at the
  ///   beginning of an iteration group.  See the documentation for \c ngr
  ///   (default = 1).
  /// \param mf This real number in [0,1] is used to dampen the DIIS
  ///   extrapolation by mixing the input data with the output data for each
  ///   iteration (default = 0.0, which performs no mixing). The approach
  ///   described in Kerker, Phys. Rev. B, 23, p3082, 1981.
  /// \param adt This real number controls attenuation of damping_factor;
  ///            if nonzero, once the 2-norm of the error is below this
  ///            attenuate the damping factor by the ratio of the current
  ///            2-norm of the error to this value.
  DIIS(unsigned int strt = 1, unsigned int ndi = 5, scalar_type dmp = 0,
       unsigned int ngr = 1, unsigned int ngrdiis = 1, scalar_type mf = 0,
       scalar_type adt = 0)
      : error_(0),
        errorset_(false),
        start(strt),
        ndiis(ndi),
        iter(0),
        ngroup(ngr),
        ngroupdiis(ngrdiis),
        damping_factor(dmp),
        mixing_fraction(mf),
        attenuated_damping_threshold(adt) {
    init();
  }
  ~DIIS() {
    x_.clear();
    errors_.clear();
    x_extrap_.clear();
  }

  /// \param[in,out] x On input, the most recent solution guess; on output,
  ///   the extrapolated guess
  /// \param[in,out] error On input, the most recent error; on output, the
  ///   if \c extrapolate_error \c == \c true will be the extrapolated
  ///   error, otherwise the value unchanged
  /// \param extrapolate_error whether to extrapolate the error (default =
  ///   false).
  void extrapolate(D& x, D& error, bool extrapolate_error = false) {
    iter++;

    // compute extrapolation coefficients C_ and number of skipped vectors
    // nskip_
    compute_extrapolation_parameters(error);

    // extrapolate x using above computed parameters (C_ and nskip_)
    extrapolate(x, C_, nskip_);

    const unsigned int nvec = errors_.size();

    // sizes of the x set and the error set should equal, otherwise throw
    TA_ASSERT(x_.size() == errors_.size() &&
              "DIIS: numbers of guess and error vectors do not match, "
              "likely due to a programming error");

    // extrapolate the error if needed
    if (extrapolate_error && (mixing_fraction == 0.0 || x_extrap_.empty())) {
      for (unsigned int k = nskip_, kk = 1; k < nvec; ++k, ++kk) {
        axpy(error, C_[kk], errors_[k]);
      }
    }
  }

  /// calling this function performs the extrapolation with provided
  /// coefficients.
  /// \param[in,out] x On input, the most recent solution guess; on output,
  ///   the extrapolated guess
  /// \param c provided coefficients
  /// \param nskip number of old vectors to skip (default = 0)
  /// \param increase_iter whether to increase the diis iteration index
  /// (default = false)
  void extrapolate(D& x, const Vector& c, unsigned int nskip = 0,
                   bool increase_iter = false) {
    if (increase_iter) {
      iter++;
    }

    const bool do_mixing = (mixing_fraction != 0.0);

    // if have ndiis vectors
    if (x_.size() ==
        ndiis) {  // holding max # of vectors already? drop the least recent x
      x_.pop_front();
      if (not x_extrap_.empty()) x_extrap_.pop_front();
    }

    // push x to the set
    x_.push_back(x);

    if (iter == 1) {  // the first iteration
      if (not x_extrap_.empty() && do_mixing) {
        zero(x);
        axpy(x, (1.0 - mixing_fraction), x_[0]);
        axpy(x, mixing_fraction, x_extrap_[0]);
      }
    } else if (iter > start && (((iter - start) % ngroup) <
                                ngroupdiis)) {  // not the first iteration and
                                                // need to extrapolate?

      const unsigned int nvec = x_.size();
      const unsigned int rank = nvec - nskip + 1;  // size of coefficients

      TA_ASSERT(c.size() == rank &&
                "DIIS: numbers of coefficients and x's do not match");
      zero(x);
      for (unsigned int k = nskip, kk = 1; k < nvec; ++k, ++kk) {
        if (not do_mixing || x_extrap_.empty()) {
          // std::cout << "contrib " << k << " c=" << c[kk] << ":" << std::endl
          // << x_[k] << std::endl;
          axpy(x, c[kk], x_[k]);
        } else {
          axpy(x, c[kk] * (1.0 - mixing_fraction), x_[k]);
          axpy(x, c[kk] * mixing_fraction, x_extrap_[k]);
        }
      }

    }  // do DIIS

    // only need to keep extrapolated x if doing mixing
    if (do_mixing) x_extrap_.push_back(x);
  }

  /// calling this function computes extrapolation parameters,
  /// i.e. coefficients \c C_ and number of skipped vectors \c nskip_
  /// \param error the most recent error
  /// \param increase_iter whether to increase the diis iteration index
  /// (default = false)
  void compute_extrapolation_parameters(const D& error,
                                        bool increase_iter = false) {
    if (increase_iter) {
      iter++;
    }

    // if have ndiis vectors
    if (errors_.size() == ndiis) {  // holding max # of vectors already? drop
                                    // the least recent error
      errors_.pop_front();
      Matrix Bcrop = B_.bottomRightCorner(ndiis - 1, ndiis - 1);
      Bcrop.conservativeResize(ndiis, ndiis);
      B_ = Bcrop;
    }

    // push error to the set
    errors_.push_back(error);
    const unsigned int nvec = errors_.size();

    // and compute the most recent elements of B, B(i,j) = <ei|ej>
    for (unsigned int i = 0; i < nvec - 1; i++)
      B_(i, nvec - 1) = B_(nvec - 1, i) =
          inner_product(errors_[i], errors_[nvec - 1]);
    B_(nvec - 1, nvec - 1) =
        inner_product(errors_[nvec - 1], errors_[nvec - 1]);
    using std::abs;
    using std::sqrt;
    const auto current_error_2norm = sqrt(abs(B_(nvec - 1, nvec - 1)));

    const scalar_type zero_determinant = 1.0e-15;
    const scalar_type zero_norm = 1.0e-10;
    const auto current_damping_factor =
        attenuated_damping_threshold > 0 &&
                current_error_2norm < attenuated_damping_threshold
            ? damping_factor *
                  (current_error_2norm / attenuated_damping_threshold)
            : damping_factor;
    const scalar_type scale = 1.0 + current_damping_factor;

    // compute extrapolation coefficients C_ and number of skipped vectors
    // nskip_
    if (iter > start &&
        (((iter - start) % ngroup) <
         ngroupdiis)) {  // not the first iteration and need to extrapolate?

      scalar_type absdetA;
      nskip_ = 0;  // how many oldest vectors to skip for the sake of
                   // conditioning? try zero
      do {
        const unsigned int rank = nvec - nskip_ + 1;  // size of matrix A

        // set up the DIIS linear system: A c = rhs
        Matrix A(rank, rank);
        C_.resize(rank);

        A.col(0).setConstant(-1.0);
        A.row(0).setConstant(-1.0);
        A(0, 0) = 0.0;
        Vector rhs = Vector::Zero(rank);
        rhs[0] = -1.0;

        scalar_type norm = 1.0;
        if (std::abs(B_(nskip_, nskip_)) > zero_norm)
          norm = 1.0 / std::abs(B_(nskip_, nskip_));

        A.block(1, 1, rank - 1, rank - 1) =
            B_.block(nskip_, nskip_, rank - 1, rank - 1) * norm;
        A.diagonal() *= scale;
        // for (unsigned int i=1; i < rank ; i++) {
        //  for (unsigned int j=1; j <= i ; j++) {
        //    A(i, j) = A(j, i) = B_(i+nskip-1, j+nskip-1) * norm;
        //    if (i==j) A(i, j) *= scale;
        //  }
        //}

#if 0
            std::cout << "DIIS: iter=" << iter << " nskip=" << nskip << " nvec=" << nvec << std::endl;
            std::cout << "DIIS: B=" << B_ << std::endl;
            std::cout << "DIIS: A=" << A << std::endl;
            std::cout << "DIIS: rhs=" << rhs << std::endl;
#endif

        // finally, solve the DIIS linear system
        Eigen::ColPivHouseholderQR<Matrix> A_QR = A.colPivHouseholderQr();
        C_ = A_QR.solve(rhs);
        absdetA = A_QR.absDeterminant();

        // std::cout << "DIIS: |A|=" << absdetA << " sol=" << c << std::endl;

        ++nskip_;

      } while (absdetA < zero_determinant &&
               nskip_ < nvec);  // while (system is poorly conditioned)

      // failed?
      if (absdetA < zero_determinant) {
        std::ostringstream oss;
        oss << "DIIS::extrapolate: poorly-conditioned system, |A| = "
            << absdetA;
        throw std::domain_error(oss.str());
      }
      --nskip_;  // undo the last ++ :-(

      parameters_computed_ = true;
    }
  }

  /// calling this function forces the extrapolation to start upon next call
  /// to \c extrapolate() even if this object was initialized with start
  /// value greater than the current iteration index.
  void start_extrapolation() {
    if (start > iter) start = iter + 1;
  }

  void reinitialize(const D* data = 0) {
    iter = 0;
    if (data) {
      const bool do_mixing = (mixing_fraction != 0.0);
      if (do_mixing) x_extrap_.push_front(*data);
    }
  }

  /// calling this function returns extrapolation coefficients
  const Vector& get_coeffs() {
    TA_ASSERT(parameters_computed_ && C_.size() > 0 &&
              "DIIS: empty coefficients, because they have not been computed");
    return C_;
  }

  /// calling this function returns number of skipped vectors in extrapolation
  unsigned int get_nskip() { return nskip_; }

  /// calling this function returns whether diis parameters C_ and nskip_ have
  /// been computed
  bool parameters_computed() { return parameters_computed_; }

 private:
  scalar_type error_;
  bool errorset_;

  unsigned int start;
  unsigned int ndiis;
  unsigned int iter;
  unsigned int ngroup;
  unsigned int ngroupdiis;
  scalar_type damping_factor;                //!< provided initially
  scalar_type mixing_fraction;               //!< provided initially
  scalar_type attenuated_damping_threshold;  //!< if nonzero, will start
                                             //!< decreasing damping factor once
                                             //!< error 2-norm falls below this

  Matrix B_;                  //!< B(i,j) = <ei|ej>
  Vector C_;                  //! DIIS coefficients
  bool parameters_computed_;  //! whether diis parameters C_ and nskip_ have
                              //! been computed
  unsigned int nskip_;        //! number of skipped vectors in extrapolation

  std::deque<D>
      x_;  //!< set of most recent x given as input (i.e. not exrapolated)
  std::deque<D> errors_;    //!< set of most recent errors
  std::deque<D> x_extrap_;  //!< set of most recent extrapolated x

  void set_error(scalar_type e) {
    error_ = e;
    errorset_ = true;
  }
  scalar_type error() { return error_; }

  void init() {
    iter = 0;

    B_ = Matrix::Zero(ndiis, ndiis);
    C_.resize(0);
    parameters_computed_ = false;
    nskip_ = 0;

    x_.clear();
    errors_.clear();
    x_extrap_.clear();
    // x_.resize(ndiis);
    // errors_.resize(ndiis);
    // x_extrap_ is bigger than the other because
    // it must hold data associated with the next iteration
    // x_extrap_.resize(diis+1);
  }

};  // class DIIS

}  // namespace TiledArray::math

namespace TiledArray {
using TiledArray::math::DIIS;
}

#endif  // TILEDARRAY_MATH_SOLVERS_DIIS_H__INCLUDED
