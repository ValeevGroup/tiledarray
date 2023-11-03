#pragma once
#include "linalg_fixture.h"
#include "compare_utilities.h" // Tensor comparison utilities
#include "gen_trange.h"        // TiledRange generator
#include "misc_util.h"         // Misc utilities

// HEIG Test - INPUT/OUTPUT have the same tiling
template <typename Derived>
void ReferenceFixture<Derived>::heig_same_tiling_test(TA::World& world) {
  world.gop.fence(); // Start epoch

  auto trange = gen_trange(N, {128ul});
  std::cout << "TRANGE = " << trange << std::endl;

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  // Solve EVP
  auto [evals, evecs] = Derived::heig(A);
  BOOST_CHECK(evecs.trange() == A.trange()); // Check for correct trange

  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::non_dist", exact_evals, evals, tol);

  // Check eigenvectors
  TA::TArray<double> tmp;
  tmp("i,j") = A("i,k") * evecs("k,j");
  A("i,j")   = evecs("k,i").conj() * tmp("k,j");
  subtract_diagonal_tensor_inplace(A, evals);

  const auto norm = A("i,j").norm(world).get();
  BOOST_CHECK_SMALL(norm, tol);

  world.gop.fence(); // End epoch
}



// HEIG Test - INPUT/OUTPUT have different tilings
template <typename Derived>
void ReferenceFixture<Derived>::heig_diff_tiling_test(TA::World& world) {
  world.gop.fence();

  auto trange     = gen_trange(N, {128ul});
  auto new_trange = gen_trange(N, {64ul} );

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(*GlobalFixture::world, trange);
  auto A_new = generate_ta_reference<array_type>(*GlobalFixture::world, new_trange);

  // Solve EVP
  auto [evals, evecs] = Derived::heig(A, new_trange);
  BOOST_CHECK(evecs.trange() == new_trange);

  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::non_dist", exact_evals, evals, tol);

  // Check eigenvectors
  TA::TArray<double> tmp;
  tmp("i,j")   = A_new("i,k") * evecs("k,j");
  A_new("i,j") = evecs("k,i").conj() * tmp("k,j");
  subtract_diagonal_tensor_inplace(A_new, evals);

  const auto norm = A_new("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, tol);
  
  GlobalFixture::world->gop.fence();
}



// Generalized HEIG Test
template <typename Derived>
void ReferenceFixture<Derived>::heig_generalized_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  // Generate Identity Tensor in TA 
  auto dense_iden = generate_ta_identity<array_type>(world, trange);

  // Solve EVP
  auto [evals, evecs] = Derived::heig(A, dense_iden);
  BOOST_CHECK(evecs.trange() == A.trange());

  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::non_dist", exact_evals, evals, tol);

  // Check eigenvectors
  TA::TArray<double> tmp;
  tmp("i,j") = A("i,k") * evecs("k,j");
  A("i,j")   = evecs("k,i").conj() * tmp("k,j");
  subtract_diagonal_tensor_inplace(A, evals);

  const auto norm = A("i,j").norm(world).get();
  BOOST_CHECK_SMALL(norm, tol);

  world.gop.fence();
}
