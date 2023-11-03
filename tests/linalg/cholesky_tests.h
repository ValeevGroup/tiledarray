#pragma once
#include "linalg_fixture.h"
#include "compare_utilities.h" // Tensor comparison utilities
#include "gen_trange.h"        // TiledRange generator
#include "misc_util.h"         // Misc utilities

// Cholesky (POTRF) Test 
template <typename Derived>
void ReferenceFixture<Derived>::cholesky_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  auto L = Derived::cholesky(A);

  BOOST_CHECK(L.trange() == A.trange());

  decltype(A) A_minus_LLt;
  A_minus_LLt("i,j") = A("i,j") - L("i,k") * L("j,k").conj();

  const double epsilon = N * N * std::numeric_limits<double>::epsilon();

  BOOST_CHECK_SMALL(A_minus_LLt("i,j").norm().get(), epsilon);

  world.gop.fence();
}



// Cholesky LINV (POTRF + TRTRI) Test 
template <typename Derived>
void ReferenceFixture<Derived>::cholesky_linv_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  auto Linv = Derived::template cholesky_linv<false>(A);
  BOOST_CHECK(Linv.trange() == A.trange());

  TA::TArray<double> tmp(world, trange);
  tmp("i,j") = Linv("i,k") * A("k,j");
  A("i,j") = tmp("i,k") * Linv("j,k");
  subtract_identity_inplace(A); // A -= I

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = A("i,j").norm().get();
  BOOST_CHECK_SMALL(norm, epsilon);

  world.gop.fence();
}



// Cholesky LINV (POTRF + TRTRI) + L Return Test 
template <typename Derived>
void ReferenceFixture<Derived>::cholesky_linv_retl_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  auto [L, Linv] = Derived::template cholesky_linv<true>(A);

  BOOST_CHECK(Linv.trange() == A.trange());
  BOOST_CHECK(L.trange() == A.trange());

  TA::TArray<double> tmp(world, trange);
  tmp("i,j") = Linv("i,k") * L("k,j");
  subtract_identity_inplace(tmp); // tmp -= I

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = tmp("i,j").norm(world).get();
  BOOST_CHECK_SMALL(norm, epsilon);

  world.gop.fence();
}



// Cholesky Solve (POSV) Test 
template <typename Derived>
void ReferenceFixture<Derived>::cholesky_solve_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  auto iden = Derived::cholesky_solve(A, A);
  BOOST_CHECK(iden.trange() == A.trange());
  subtract_identity_inplace(iden); // iden -= I

  const auto epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = iden("i,j").norm(world).get();
  BOOST_CHECK_SMALL(norm, epsilon);

  world.gop.fence();
}



// Cholesky L-Solve (POTRF + TRSM) Test 
template <typename Derived>
void ReferenceFixture<Derived>::cholesky_lsolve_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  // Should produce X = L**H
  auto [L, X] = Derived::cholesky_lsolve(TA::NoTranspose, A, A);
  BOOST_CHECK(X.trange() == A.trange());
  BOOST_CHECK(L.trange() == A.trange());

  X("i,j") -= L("j,i");

  const auto epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = X("i,j").norm(world).get();
  BOOST_CHECK_SMALL(norm, epsilon);

  world.gop.fence();
}
