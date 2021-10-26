#ifndef TILEDARRAY_MATH_LINALG_RANK_LOCAL_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_RANK_LOCAL_H__INCLUDED

#include <TiledArray/config.h>

#include <TiledArray/external/eigen.h>
#include <TiledArray/math/linalg/forward.h>

#include <lapack.hh>

#include <vector>

namespace TiledArray::math::linalg::rank_local {

namespace detail {
  template <typename T>
  struct real_type { using type = T; };
  template <typename T>
  struct real_type< std::complex<T> > { using type = T; };
  template <typename T>
  using real_type_t = typename real_type<T>::type;
}

using Job = ::lapack::Job;

template <typename T, int Options = ::Eigen::ColMajor>
using Matrix = ::Eigen::Matrix<T, ::Eigen::Dynamic, ::Eigen::Dynamic, Options>;

// template <typename T>
// using Vector = ::Eigen::Matrix<T, ::Eigen::Dynamic, 1, ::Eigen::ColMajor>;

template <typename T>
void cholesky(Matrix<T> &A);

template <typename T>
void cholesky_linv(Matrix<T> &A);

template <typename T>
void cholesky_solve(Matrix<T> &A, Matrix<T> &X);

template <typename T>
void cholesky_lsolve(Op transpose, Matrix<T> &A, Matrix<T> &X);

template <typename T>
void heig(Matrix<T> &A, std::vector<detail::real_type_t<T>> &W);

template <typename T>
void heig(Matrix<T> &A, Matrix<T> &B, std::vector<detail::real_type_t<T>> &W);

template <typename T>
void svd(Job jobu, Job jobvt, Matrix<T> &A, std::vector<detail::real_type_t<T>> &S, 
         Matrix<T> *U, Matrix<T> *VT);

template <typename T>
void svd(Matrix<T> &A, std::vector<detail::real_type_t<T>> &S, 
         Matrix<T> *U, Matrix<T> *VT) {
  svd( U  ? Job::SomeVec : Job::NoVec, 
       VT ? Job::SomeVec : Job::NoVec,
       A, S, U, VT );
}

template <typename T>
void lu_solve(Matrix<T> &A, Matrix<T> &B);

template <typename T>
void lu_inv(Matrix<T> &A);

template <bool QOnly,typename T>
void householder_qr( Matrix<T> &V, Matrix<T> &R );

}  // namespace TiledArray::math::linalg::rank_local

#endif  // TILEDARRAY_MATH_LINALG_RANK_LOCAL_H__INCLUDED
