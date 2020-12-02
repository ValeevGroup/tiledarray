#ifndef TILEDARRAY_ALGEBRA_RANK_LOCAL_H__INCLUDED
#define TILEDARRAY_ALGEBRA_RANK_LOCAL_H__INCLUDED

#include <TiledArray/config.h>

#include <TiledArray/algebra/types.h>
#include <TiledArray/external/eigen.h>

#include <vector>

namespace TiledArray::algebra::rank_local {

template<typename T, int Options = ::Eigen::ColMajor>
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
void cholesky_lsolve(TransposeFlag transpose,
                     Matrix<T> &A, Matrix<T> &X);

template <typename T>
void heig(Matrix<T> &A, std::vector<T> &W);

template <typename T>
void heig(Matrix<T> &A, Matrix<T> &B, std::vector<T> &W);

template <typename T>
void svd(Matrix<T> &A, std::vector<T> &S,
         Matrix<T> *U, Matrix<T> *VT);

template <typename T>
void lu_solve(Matrix<T> &A, Matrix<T> &B);

template <typename T>
void lu_inv(Matrix<T> &A);

}  // namespace TiledArray::local

#endif  // TILEDARRAY_ALGEBRA_RANK_LOCAL_H__INCLUDED
