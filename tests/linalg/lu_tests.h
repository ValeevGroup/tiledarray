#pragma once
#include "linalg_fixture.h"
#include "compare_utilities.h" // Tensor comparison utilities
#include "gen_trange.h"        // TiledRange generator
#include "misc_util.h"         // Misc utilities

// LU Solve (GESV) Test 
template <typename Derived>
void ReferenceFixture<Derived>::lu_solve_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  auto iden = Derived::lu_solve(A, A);
  BOOST_CHECK(iden.trange() == A.trange());
  subtract_identity_inplace(iden); // iden -= I

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = iden("i,j").norm(world).get();
  BOOST_CHECK_SMALL(norm, epsilon);

  world.gop.fence();
}



// LU Inverse (GETRF + GETRI) Test 
template <typename Derived>
void ReferenceFixture<Derived>::lu_inv_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  TA::TArray<double> iden(world, trange);

  auto Ainv = Derived::lu_inv(A);
  iden("i,j") = Ainv("i,k") * A("k,j");

  BOOST_CHECK(iden.trange() == A.trange());
  subtract_identity_inplace(iden); // iden -= I

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = iden("i,j").norm(world).get();

  BOOST_CHECK_SMALL(norm, epsilon);

  world.gop.fence();
}
