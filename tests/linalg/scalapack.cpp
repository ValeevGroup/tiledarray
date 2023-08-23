#include "heig_tests.h"     // EVP tests
#include "cholesky_tests.h" // Cholesky tests
#include "lu_tests.h"       // LU tests
#include "svd_tests.h"      // SVD tests

// ScaLAPACK linear algebra utilities
#include "TiledArray/math/linalg/scalapack/all.h"

namespace TA = TiledArray;
namespace scalapack = TA::math::linalg::scalapack;

struct ScaLAPACKLinearAlgebraFixture : 
  ReferenceFixture<ScaLAPACKLinearAlgebraFixture> {

  ScaLAPACKLinearAlgebraFixture(int64_t N = 1000) : 
    ReferenceFixture<ScaLAPACKLinearAlgebraFixture>(N) {}

  template <typename... Args>
  static auto heig(Args&&... args) { 
    return scalapack::heig(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto cholesky(Args&&... args) { 
    return scalapack::cholesky(std::forward<Args>(args)...); 
  }

  template <bool RetL, typename... Args>
  static auto cholesky_linv(Args&&... args) { 
    return scalapack::cholesky_linv<RetL>(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto cholesky_solve(Args&&... args) { 
    return scalapack::cholesky_solve(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto cholesky_lsolve(Args&&... args) { 
    return scalapack::cholesky_lsolve(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto lu_solve(Args&&... args) { 
    return scalapack::lu_solve(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto lu_inv(Args&&... args) { 
    return scalapack::lu_inv(std::forward<Args>(args)...); 
  }

  template<TA::SVD::Vectors Vectors, typename... Args>
  static auto svd(Args&&... args) {
    return scalapack::svd<Vectors>(std::forward<Args>(args)...);
  }
};


BOOST_FIXTURE_TEST_SUITE(linear_algebra_suite_scalapack, ScaLAPACKLinearAlgebraFixture)

// HEIG tests
LINALG_TEST_IMPL(heig_same_tiling);
LINALG_TEST_IMPL(heig_diff_tiling);
LINALG_TEST_IMPL(heig_generalized);

// Cholesky tests
LINALG_TEST_IMPL(cholesky);
LINALG_TEST_IMPL(cholesky_linv);
LINALG_TEST_IMPL(cholesky_linv_retl);
LINALG_TEST_IMPL(cholesky_solve);
LINALG_TEST_IMPL(cholesky_lsolve);

// LU tests
LINALG_TEST_IMPL(lu_solve);
LINALG_TEST_IMPL(lu_inv);

// SVD tests
LINALG_TEST_IMPL(svd_values_only);
LINALG_TEST_IMPL(svd_leftvectors);
LINALG_TEST_IMPL(svd_rightvectors);
LINALG_TEST_IMPL(svd_allvectors);

BOOST_AUTO_TEST_SUITE_END()
