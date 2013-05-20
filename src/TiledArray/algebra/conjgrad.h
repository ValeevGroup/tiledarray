/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_ALGEBRA_CONJGRAD_H__INCLUDED
#define TILEDARRAY_ALGEBRA_CONJGRAD_H__INCLUDED

#include <sstream>
#include <TiledArray/array.h>
#include <TiledArray/algebra/diis.h>
#include <TiledArray/algebra/utils.h>

namespace TiledArray {

  namespace detail {

    inline std::string dummy_annotation(unsigned int DIM) {
      std::ostringstream oss;
      if (DIM > 0) oss << "i0";
      for(auto d=1; d<DIM; ++d)
        oss << ",i" << d;
      return oss.str();
    }

  } // namespace TiledArray::detail

  template <typename T, unsigned int DIM, typename Tile>
  inline size_t size(const TiledArray::Array<T, DIM, Tile>& a) {
    // this is the number of tiles
    if (a.size() > 0) // assuming dense shape
      return a.trange().elements().volume();
    else
      return 0;
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline TiledArray::Array<T,DIM,Tile> clone(const TiledArray::Array<T,DIM,Tile>& a) {
    return a;
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline TiledArray::Array<T,DIM,Tile> copy(const TiledArray::Array<T,DIM,Tile>& a) {
    return a;
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void zero(TiledArray::Array<T,DIM,Tile>& a) {
    a = typename TiledArray::Array<T,DIM,Tile>::element_type(0) * a(detail::dummy_annotation(DIM));
  }

  template <typename T, unsigned int DIM, typename Tile>
  typename TiledArray::Array<T,DIM,Tile>::element_type
  inline minabs_value(const TiledArray::Array<T,DIM,Tile>& a) {
    return minabs(a(detail::dummy_annotation(DIM)));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline typename TiledArray::Array<T,DIM,Tile>::element_type
  maxabs_value(const TiledArray::Array<T,DIM,Tile>& a) {
    return norminf(a(detail::dummy_annotation(DIM)));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void vec_multiply(TiledArray::Array<T,DIM,Tile>& a1, const TiledArray::Array<T,DIM,Tile>& a2) {
    a1 = multiply(a1(detail::dummy_annotation(DIM)), a2(detail::dummy_annotation(DIM)));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline typename TiledArray::Array<T,DIM,Tile>::element_type
  dot_product(const TiledArray::Array<T,DIM,Tile>& a1, const TiledArray::Array<T,DIM,Tile>& a2) {
    return dot(a1(detail::dummy_annotation(DIM)), a2(detail::dummy_annotation(DIM)));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void scale(TiledArray::Array<T,DIM,Tile>& a,
                    typename TiledArray::Array<T,DIM,Tile>::element_type scaling_factor) {
    a = scaling_factor * a(detail::dummy_annotation(DIM));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void axpy(TiledArray::Array<T,DIM,Tile>& y,
                   typename TiledArray::Array<T,DIM,Tile>::element_type a,
                   const TiledArray::Array<T,DIM,Tile>& x) {
    y = y(detail::dummy_annotation(DIM)) + a * x(detail::dummy_annotation(DIM));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void assign(TiledArray::Array<T,DIM,Tile>& m1, const TiledArray::Array<T,DIM,Tile>& m2) {
    m1 = m2;
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline typename TiledArray::Array<T,DIM,Tile>::element_type
  norm2(const TiledArray::Array<T,DIM,Tile>& a) {
    return norm2(a(detail::dummy_annotation(DIM)));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void print(const TiledArray::Array<T,DIM,Tile>& a, const char* label) {
    std::cout << label << ":" << std::endl
              << a << std::endl;
  }

  /**
   * Solves linear system a(x) = b using conjugate gradient solver
   * where a is a linear function of x.
   *
   * @tparam D type of \c x and \c b, as well as the preconditioner; D::element_type must be defined and D must provide
   * the following standalone functions:
   *   - size_t size(const D&)
   *   - D clone(const D&)
   *   - D copy(const D&)
   *   - value_type minabs_value(const D&)
   *   - value_type maxabs_value(const D&)
   *   - void vec_multiply(D& a, const D& b) (element-wise multiply of a by b)
   *   - value_type dot_product(const D& a, const D& b)
   *   - void scale(D&, value_type)
   *   - void axpy(D& y, value_type a, const D& x)
   *   - void assign(D&, const D&)
   *   - double norm2(const D&)
   * @tparam F type that evaluates the LHS, will call F::operator()(x, result), hence must implement operator()(const D&, D&) const
   *
   * @param a object of type F
   * @param b RHS
   * @param x unknown
   * @param preconditioner
   * @param convergence_target
   * @return the 2-norm of the residual, a(x) - b, divided by the number of elements in the residual.
   */
  template <typename D, typename F>
  struct ConjugateGradientSolver {
    typedef typename D::element_type value_type;

    value_type operator()(F& a,
               const D& b,
               D& x,
               const D& preconditioner,
               value_type convergence_target = -1.0) {

      auto n = size(x);
      assert(n == size(preconditioner));

      const bool use_diis = false;
      DIIS<D> diis;

      // solution vector
      D XX_i;
      // residual vector
      D RR_i = clone(b);
      // preconditioned residual vector
      D ZZ_i;
      // direction vector
      D PP_i;
      D APP_i = clone(b);

      // approximate the condition number as the ratio of the min and max elements of the preconditioner
      // assuming that preconditioner is the approximate inverse of A in Ax - b =0
      const value_type precond_min = minabs_value(preconditioner);
      const value_type precond_max = maxabs_value(preconditioner);
      const value_type cond_number = precond_max / precond_min;
      //std::cout << "condition number = " << precond_max << " / " << precond_min << " = " << cond_number << std::endl;
      // if convergence target is given, estimate of how tightly the system can be converged
      if (convergence_target < 0.0) {
        convergence_target = 1e-15 * cond_number;
      }
      else { // else warn if the given system is not sufficiently well conditioned
        if (convergence_target < 1e-15 * cond_number)
          std::cout << "WARNING: ConjugateGradient convergence target (" << convergence_target
                    << ") may be too low for 64-bit precision" << std::endl;
      }

      bool converged = false;
      const unsigned int max_niter = 500;
      value_type rnorm2 = 0.0;
      const auto rhs_size = size(b);

      // starting guess: x_0 = D^-1 . b
      XX_i = copy(b);
      vec_multiply(XX_i, preconditioner);

      // r_0 = b - a(x)
      a(XX_i, RR_i);  // RR_i = a(XX_i)
      scale(RR_i, -1.0);
      axpy(RR_i, 1.0, b); // RR_i = b - a(XX_i)

      if (use_diis)
        diis.extrapolate(XX_i, RR_i, true);

      // z_0 = D^-1 . r_0
      ZZ_i = copy(RR_i);
      vec_multiply(ZZ_i, preconditioner);

      // p_0 = z_0
      PP_i = copy(ZZ_i);

      unsigned int iter = 0;
      while (not converged) {

        // alpha_i = (r_i . z_i) / (p_i . A . p_i)
        value_type rz_norm2 = dot_product(RR_i, ZZ_i);
        a(PP_i,APP_i);

        const value_type pAp_i = dot_product(PP_i, APP_i);
        const value_type alpha_i = rz_norm2 / pAp_i;

        // x_i += alpha_i p_i
        axpy(XX_i, alpha_i, PP_i);

        // r_i -= alpha_i Ap_i
        axpy(RR_i, -alpha_i, APP_i);

        if (use_diis)
          diis.extrapolate(XX_i, RR_i, true);

        const value_type r_ip1_norm = norm2(RR_i) / rhs_size;
        if (r_ip1_norm < convergence_target) {
          converged = true;
          rnorm2 = r_ip1_norm;
        }

        // z_i = D^-1 . r_i
        ZZ_i = copy(RR_i);
        vec_multiply(ZZ_i, preconditioner);

        const value_type rz_ip1_norm2 = dot_product(ZZ_i, RR_i);

        const value_type beta_i = rz_ip1_norm2 / rz_norm2;

        // p_i = z_i+1 + beta_i p_i
        // 1) scale p_i by beta_i
        // 2) add z_i+1 (i.e. current contents of z_i)
        scale(PP_i, beta_i);
        axpy(PP_i, 1.0, ZZ_i);

        ++iter;
        //std::cout << "iter=" << iter << " dnorm=" << r_ip1_norm << std::endl;

        if (iter >= max_niter) {
          assign(x, XX_i);
          throw std::domain_error("ConjugateGradient: max # of iterations exceeded");
        }
      } // solver loop

      assign(x, XX_i);

      return rnorm2;
    }
  };

};

#endif // TILEDARRAY_ALGEBRA_CONJGRAD_H__INCLUDED
