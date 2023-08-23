#include "linalg_fixture.h"    // ReferenceFixture
#include "compare_utilities.h" // Tensor comparison utilities
#include "gen_trange.h"        // TiledRange generator
#include "misc_util.h"         // Misc utilities

// Non-distributed linear algebra utilities
#include "TiledArray/math/linalg/non-distributed/cholesky.h"
#include "TiledArray/math/linalg/non-distributed/heig.h"
#include "TiledArray/math/linalg/non-distributed/lu.h"
#include "TiledArray/math/linalg/non-distributed/svd.h"

namespace TA = TiledArray;
namespace non_dist = TA::math::linalg::non_distributed;

struct NonDistLinearAlgebraFixture : ReferenceFixture {
  NonDistLinearAlgebraFixture(int64_t N = 1000) : ReferenceFixture(N) {}
};

BOOST_FIXTURE_TEST_SUITE(linear_algebra_suite_non_dist, NonDistLinearAlgebraFixture)


// HEIG Test - INPUT/OUTPUT have the same tiling
BOOST_AUTO_TEST_CASE(heig_same_tiling) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  auto [evals, evecs] = non_dist::heig(A);
  BOOST_CHECK(evecs.trange() == A.trange());


  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::non_dist", exact_evals, evals, tol);

  // Check eigenvectors
  TA::TArray<double> tmp;
  tmp("i,j") = A("i,k") * evecs("k,j");
  A("i,j")   = evecs("k,i").conj() * tmp("k,j");
  subtract_diagonal_tensor_inplace(A, evals);

  const auto norm = A("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, tol);

  GlobalFixture::world->gop.fence();
}



// HEIG Test - INPUT/OUTPUT have different tilings
BOOST_AUTO_TEST_CASE(heig_diff_tiling) {
  GlobalFixture::world->gop.fence();
  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  auto new_trange = gen_trange(N, {64ul});
  auto [evals, evecs] = non_dist::heig(A, new_trange);

  BOOST_CHECK(evecs.trange() == new_trange);

  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::non_dist", exact_evals, evals, tol);

  // Check eigenvectors
  auto A_new = generate_ta_reference<array_type>(*GlobalFixture::world, new_trange);

  TA::TArray<double> tmp;
  tmp("i,j")   = A_new("i,k") * evecs("k,j");
  A_new("i,j") = evecs("k,i").conj() * tmp("k,j");
  subtract_diagonal_tensor_inplace(A_new, evals);

  const auto norm = A_new("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, tol);
  

  GlobalFixture::world->gop.fence();
}



// Generalized HEIG Test
BOOST_AUTO_TEST_CASE(heig_generalized) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  // Generate Identity Tensor in TA 
  auto dense_iden = generate_ta_identity<array_type>(*GlobalFixture::world, trange);

  GlobalFixture::world->gop.fence();
  auto [evals, evecs] = non_dist::heig(A, dense_iden);
  BOOST_CHECK(evecs.trange() == A.trange());

  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::non_dist", exact_evals, evals, tol);

  // Check eigenvectors
  TA::TArray<double> tmp;
  tmp("i,j") = A("i,k") * evecs("k,j");
  A("i,j")   = evecs("k,i").conj() * tmp("k,j");
  subtract_diagonal_tensor_inplace(A, evals);

  const auto norm = A("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, tol);


  GlobalFixture::world->gop.fence();
}



// Cholesky (POTRF) Test 
BOOST_AUTO_TEST_CASE(cholesky) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  auto L = non_dist::cholesky(A);

  BOOST_CHECK(L.trange() == A.trange());

  decltype(A) A_minus_LLt;
  A_minus_LLt("i,j") = A("i,j") - L("i,k") * L("j,k").conj();

  const double epsilon = N * N * std::numeric_limits<double>::epsilon();

  BOOST_CHECK_SMALL(A_minus_LLt("i,j").norm().get(), epsilon);

  GlobalFixture::world->gop.fence();
}



// Cholesky LINV (POTRF + TRTRI) Test 
BOOST_AUTO_TEST_CASE(cholesky_linv) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  auto Linv = non_dist::cholesky_linv<false>(A);
  BOOST_CHECK(Linv.trange() == A.trange());

  TA::TArray<double> tmp(*GlobalFixture::world, trange);
  tmp("i,j") = Linv("i,k") * A("k,j");
  A("i,j") = tmp("i,k") * Linv("j,k");
  subtract_identity_inplace(A); // A -= I

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = A("i,j").norm().get();
  BOOST_CHECK_SMALL(norm, epsilon);

  GlobalFixture::world->gop.fence();
}



// Cholesky LINV (POTRF + TRTRI) + L Return Test 
BOOST_AUTO_TEST_CASE(cholesky_linv_retl) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  auto [L, Linv] = non_dist::cholesky_linv<true>(A);

  BOOST_CHECK(Linv.trange() == A.trange());
  BOOST_CHECK(L.trange() == A.trange());

  TA::TArray<double> tmp(*GlobalFixture::world, trange);
  tmp("i,j") = Linv("i,k") * L("k,j");
  subtract_identity_inplace(tmp); // tmp -= I

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = tmp("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, epsilon);

  GlobalFixture::world->gop.fence();
}



// Cholesky Solve (POSV) Test 
BOOST_AUTO_TEST_CASE(cholesky_solve) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  auto iden = non_dist::cholesky_solve(A, A);
  BOOST_CHECK(iden.trange() == A.trange());
  subtract_identity_inplace(iden); // iden -= I

  const auto epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = iden("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, epsilon);

  GlobalFixture::world->gop.fence();
}



// Cholesky L-Solve (POTRF + TRSM) Test 
BOOST_AUTO_TEST_CASE(cholesky_lsolve) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  // Should produce X = L**H
  auto [L, X] = non_dist::cholesky_lsolve(TA::NoTranspose, A, A);
  BOOST_CHECK(X.trange() == A.trange());
  BOOST_CHECK(L.trange() == A.trange());

  X("i,j") -= L("j,i");

  const auto epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = X("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, epsilon);

  GlobalFixture::world->gop.fence();
}



// LU Solve (GESV) Test 
BOOST_AUTO_TEST_CASE(lu_solve) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  auto iden = non_dist::lu_solve(A, A);
  BOOST_CHECK(iden.trange() == A.trange());
  subtract_identity_inplace(iden); // iden -= I

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = iden("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, epsilon);

  GlobalFixture::world->gop.fence();
}



// LU Inverse (GETRF + GETRI) Test 
BOOST_AUTO_TEST_CASE(lu_inv) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  TA::TArray<double> iden(*GlobalFixture::world, trange);

  auto Ainv = non_dist::lu_inv(A);
  iden("i,j") = Ainv("i,k") * A("k,j");

  BOOST_CHECK(iden.trange() == A.trange());
  subtract_identity_inplace(iden); // iden -= I

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = iden("i,j").norm(*GlobalFixture::world).get();

  BOOST_CHECK_SMALL(norm, epsilon);

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(svd_values_only) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  auto S = non_dist::svd<TA::SVD::ValuesOnly>(A, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::non_dist", exact_singular_values, S, tol);

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(svd_leftvectors) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  auto [S, U] = non_dist::svd<TA::SVD::LeftVectors>(A, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::non_dist", exact_singular_values, S, tol);

  // Since A is Hermitian, U is also A's eigenvectors
  // A <- U**H * A * U
  TA::TArray<double> tmp;
  tmp("i,j") = A("i,k") * U("k,j");
  A("i,j")   = U("k,i").conj() * tmp("k,j");
  subtract_diagonal_tensor_inplace(A, S); // A -= SIGMA

  const auto norm = A("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, tol);

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(svd_rightvectors) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  auto [S, VT] = non_dist::svd<TA::SVD::RightVectors>(A, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::non_dist", exact_singular_values, S, tol);


  // Since A is Hermitian, VT is also (the c-transpose) A's eigenvectors
  // A <- VT * A * VT**H
  TA::TArray<double> tmp;
  tmp("i,j") = A("i,k") * VT("j,k").conj();
  A("i,j")   = VT("i,k") * tmp("k,j");
  subtract_diagonal_tensor_inplace(A, S); // A -= SIGMA

  const auto norm = A("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, tol);


  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(svd_allvectors) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);

  auto [S, U, VT] = non_dist::svd<TA::SVD::AllVectors>(A, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::non_dist", exact_singular_values, S, tol);
  
  // Recreate SVD
  // A <- U**H * A * VT**H
  TA::TArray<double> tmp;
  tmp("i,j") = A("i,k") * VT("j,k").conj();
  A("i,j")   = U("k,i").conj() * tmp("k,j");
  subtract_diagonal_tensor_inplace(A, S); // A -= SIGMA

  const auto norm = A("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, tol);


  GlobalFixture::world->gop.fence();
}


BOOST_AUTO_TEST_SUITE_END()
