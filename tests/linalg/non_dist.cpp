#include "heig_tests.h"     // EVP tests
#include "cholesky_tests.h" // Cholesky tests
#include "lu_tests.h"       // LU tests
#include "svd_tests.h"      // SVD tests
#include "qr_tests.h"       // QR tests

// Non-distributed linear algebra utilities
#include "TiledArray/math/linalg/non-distributed/cholesky.h"
#include "TiledArray/math/linalg/non-distributed/heig.h"
#include "TiledArray/math/linalg/non-distributed/lu.h"
#include "TiledArray/math/linalg/non-distributed/svd.h"

namespace TA = TiledArray;
namespace non_dist = TA::math::linalg::non_distributed;

struct NonDistLinearAlgebraFixture : ReferenceFixture<NonDistLinearAlgebraFixture> {

  NonDistLinearAlgebraFixture(int64_t N = 1000) : 
    ReferenceFixture<NonDistLinearAlgebraFixture>(N) {}

  template <typename... Args>
  static auto heig(Args&&... args) { 
    return non_dist::heig(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto cholesky(Args&&... args) { 
    return non_dist::cholesky(std::forward<Args>(args)...); 
  }

  template <bool RetL, typename... Args>
  static auto cholesky_linv(Args&&... args) { 
    return non_dist::cholesky_linv<RetL>(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto cholesky_solve(Args&&... args) { 
    return non_dist::cholesky_solve(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto cholesky_lsolve(Args&&... args) { 
    return non_dist::cholesky_lsolve(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto lu_solve(Args&&... args) { 
    return non_dist::lu_solve(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto lu_inv(Args&&... args) { 
    return non_dist::lu_inv(std::forward<Args>(args)...); 
  }

  template<TA::SVD::Vectors Vectors, typename... Args>
  static auto svd(Args&&... args) {
    return non_dist::svd<Vectors>(std::forward<Args>(args)...);
  }

  template <bool QOnly, typename... Args>
  static auto householder_qr(Args&&... args) { 
    return non_dist::householder_qr<QOnly>(std::forward<Args>(args)...); 
  }
};


BOOST_FIXTURE_TEST_SUITE(linear_algebra_suite_non_dist, NonDistLinearAlgebraFixture)

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

// QR tests
LINALG_TEST_IMPL(householder_qr_q_only);
LINALG_TEST_IMPL(householder_qr);

BOOST_AUTO_TEST_SUITE_END()