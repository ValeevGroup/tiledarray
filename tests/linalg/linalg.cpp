#include <tiledarray.h>
#include <random>
#include "TiledArray/config.h"
//#include "range_fixture.h"
#include "unit_test_config.h"

#include "linalg_fixture.h"    // ReferenceFixture
#include "compare_utilities.h" // Tensor comparison utilities
#include "gen_trange.h"        // TiledRange generator

#include "TiledArray/math/linalg/non-distributed/cholesky.h"
#include "TiledArray/math/linalg/non-distributed/heig.h"
#include "TiledArray/math/linalg/non-distributed/lu.h"
#include "TiledArray/math/linalg/non-distributed/svd.h"

#include "TiledArray/math/linalg/cholesky.h"
#include "TiledArray/math/linalg/heig.h"
#include "TiledArray/math/linalg/lu.h"
#include "TiledArray/math/linalg/svd.h"

namespace TA = TiledArray;
namespace non_dist = TA::math::linalg::non_distributed;

#if TILEDARRAY_HAS_TTG
#include "TiledArray/math/linalg/ttg/cholesky.h"
#define TILEDARRAY_TTG_TEST(F, E)    \
  GlobalFixture::world->gop.fence(); \
  compare("TiledArray::ttg", non_dist::F, TiledArray::math::linalg::ttg::F, E);
#else
#define TILEDARRAY_TTG_TEST(...)
#endif


struct LinearAlgebraFixture : ReferenceFixture<> { };


BOOST_FIXTURE_TEST_SUITE(linear_algebra_suite, LinearAlgebraFixture)

#if TILEDARRAY_HAS_TTG
BOOST_AUTO_TEST_CASE(cholesky) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  const double epsilon = N * N * std::numeric_limits<double>::epsilon();
  TILEDARRAY_TTG_TEST(cholesky(A), epsilon);
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(cholesky_linv) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  TILEDARRAY_TTG_TEST(cholesky_linv<false>(A), epsilon);
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(cholesky_linv_retl) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  TILEDARRAY_TTG_TEST(cholesky_linv<true>(A), epsilon);
  GlobalFixture::world->gop.fence();
}
#endif

template <typename ArrayT>
void cholesky_qr_q_only_test(const ArrayT& A, double tol) {
  using value_type = typename ArrayT::element_type;

  auto Q = TiledArray::math::linalg::cholesky_qr<true>(A);

  // Make sure the Q is orthogonal at least
  TA::TArray<double> Iden;
  Iden("i,j") = Q("k,i") * Q("k,j");
  Iden.make_replicated();
  auto I_eig = TA::array_to_eigen(Iden);
  const auto N = A.trange().dim(1).extent();
  BOOST_CHECK_SMALL((I_eig - decltype(I_eig)::Identity(N, N)).norm(), tol);
}

template <typename ArrayT>
void cholesky_qr_test(const ArrayT& A, double tol) {
  auto [Q, R] = TiledArray::math::linalg::cholesky_qr<false>(A);

  // Check reconstruction error
  TA::TArray<double> QR_ERROR;
  QR_ERROR("i,j") = A("i,j") - Q("i,k") * R("k,j");
  BOOST_CHECK_SMALL(QR_ERROR("i,j").norm().get(), tol);

  // Check orthonormality of Q
  TA::TArray<double> Iden;
  Iden("i,j") = Q("k,i") * Q("k,j");
  Iden.make_replicated();
  auto I_eig = TA::array_to_eigen(Iden);
  const auto N = A.trange().dim(1).extent();
  BOOST_CHECK_SMALL((I_eig - decltype(I_eig)::Identity(N, N)).norm(), tol);
}

BOOST_AUTO_TEST_CASE(cholesky_qr_q_only) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  double tol = N * N * std::numeric_limits<double>::epsilon();
  cholesky_qr_q_only_test(ref_ta, tol);

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(cholesky_qr) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  double tol = N * N * std::numeric_limits<double>::epsilon();
  cholesky_qr_test(ref_ta, tol);

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_SUITE_END()
