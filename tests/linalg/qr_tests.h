
#pragma once
#include "linalg_fixture.h"
#include "compare_utilities.h" // Tensor comparison utilities
#include "gen_trange.h"        // TiledRange generator
#include "misc_util.h"         // Misc utilities


template <typename Derived>
void ReferenceFixture<Derived>::householder_qr_q_only_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);
  
  // Compute Q
  auto Q = Derived::template householder_qr<true>(A);

  // Make sure the Q is orthogonal at least
  double tol = N * N * std::numeric_limits<double>::epsilon();
  TA::TArray<double> Iden;
  Iden("i,j") = Q("k,i") * Q("k,j");
  subtract_identity_inplace(Iden);
  const auto norm = Iden("i,j").norm(world).get();
  BOOST_CHECK_SMALL(norm, tol);

  world.gop.fence();
}

template <typename Derived>
void ReferenceFixture<Derived>::householder_qr_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);
  
  // Compute QR
  auto [Q, R] = Derived::template householder_qr<false>(A);

  double tol = N * N * std::numeric_limits<double>::epsilon();

  // Check reconstruction error
  TA::TArray<double> QR_ERROR;
  QR_ERROR("i,j") = A("i,j") - Q("i,k") * R("k,j");
  BOOST_CHECK_SMALL(QR_ERROR("i,j").norm(world).get(), tol);

  // Check orthonormality of Q
  TA::TArray<double> Iden;
  Iden("i,j") = Q("k,i") * Q("k,j");
  subtract_identity_inplace(Iden);
  const auto norm = Iden("i,j").norm(world).get();
  BOOST_CHECK_SMALL(norm, tol);

  world.gop.fence();
}
