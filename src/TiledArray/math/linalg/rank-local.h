#ifndef TILEDARRAY_MATH_LINALG_RANK_LOCAL_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_RANK_LOCAL_H__INCLUDED

#include <TiledArray/config.h>

#include <TiledArray/external/eigen.h>
#include <TiledArray/math/linalg/forward.h>

#if defined(LAPACK_COMPLEX_CPP)
TILEDARRAY_PRAGMA_CLANG(diagnostic push)
TILEDARRAY_PRAGMA_CLANG(diagnostic ignored "-Wreturn-type-c-linkage")
#endif  // defined(LAPACK_COMPLEX_CPP)

#include <lapack.hh>

#if defined(LAPACK_COMPLEX_CPP)
TILEDARRAY_PRAGMA_CLANG(diagnostic pop)
#endif  // defined(LAPACK_COMPLEX_CPP)

#include <vector>

namespace TiledArray::math::linalg::rank_local {

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
void qr_solve(Matrix<T> &A, Matrix<T> &B,
              const TiledArray::detail::real_t<T> cond = 1e8);

template <typename T>
void heig(Matrix<T> &A, std::vector<TiledArray::detail::real_t<T>> &W);

template <typename T>
void heig(Matrix<T> &A, Matrix<T> &B,
          std::vector<TiledArray::detail::real_t<T>> &W);

template <typename T>
void svd(Job jobu, Job jobvt, Matrix<T> &A,
         std::vector<TiledArray::detail::real_t<T>> &S, Matrix<T> *U,
         Matrix<T> *VT);

template <typename T>
void svd(Matrix<T> &A, std::vector<TiledArray::detail::real_t<T>> &S,
         Matrix<T> *U, Matrix<T> *VT) {
  svd(U ? Job::AllVec : Job::NoVec, VT ? Job::AllVec : Job::NoVec, A, S, U, VT);
}

template <typename T>
void lu_solve(Matrix<T> &A, Matrix<T> &B);

template <typename T>
void lu_inv(Matrix<T> &A);

template <bool QOnly, typename T>
void householder_qr(Matrix<T> &V, Matrix<T> &R);

}  // namespace TiledArray::math::linalg::rank_local

namespace madness::archive {

/// Serialize (deserialize) an lapack::Error

/// \tparam Archive The archive type.
template <class Archive>
struct ArchiveSerializeImpl<Archive, lapack::Error> {
  static inline void serialize(const Archive &ar, lapack::Error &e) {
    MAD_ARCHIVE_DEBUG(std::cout << "(de)serialize lapack::Error" << std::endl);
    if constexpr (is_output_archive_v<Archive>) {  // serialize
      const std::string msg = e.what();
      ar & msg;
    } else {
      std::string msg;
      ar & msg;
      e = lapack::Error(msg);
    }
  }
};

}  // namespace madness::archive

/// TA_LAPACK_ON_RANK_ZERO(fn,args...) invokes  linalg::rank_local::fn(args...)
/// on rank 0 and broadcasts/rethrows the exception, if any
#define TA_LAPACK_ON_RANK_ZERO(fn, world, args...) \
  std::optional<lapack::Error> error_opt;          \
  if (world.rank() == 0) {                         \
    try {                                          \
      linalg::rank_local::fn(args);                \
    } catch (lapack::Error & err) {                \
      error_opt = err;                             \
    }                                              \
  }                                                \
  world.gop.broadcast_serializable(error_opt, 0);  \
  if (error_opt) {                                 \
    throw error_opt.value();                       \
  }

#endif  // TILEDARRAY_MATH_LINALG_RANK_LOCAL_H__INCLUDED
