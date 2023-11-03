#pragma once
#include "linalg_fixture.h"
#include "compare_utilities.h" // Tensor comparison utilities
#include "gen_trange.h"        // TiledRange generator
#include "misc_util.h"         // Misc utilities


template <typename Derived>
void ReferenceFixture<Derived>::svd_values_only_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  auto S = Derived::template svd<TA::SVD::ValuesOnly>(A, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::Derived", exact_singular_values, S, tol);

  world.gop.fence();
}

template <typename Derived>
void ReferenceFixture<Derived>::svd_leftvectors_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  auto [S, U] = Derived::template svd<TA::SVD::LeftVectors>(A, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::Derived", exact_singular_values, S, tol);

  // Since A is Hermitian, U is also A's eigenvectors
  // A <- U**H * A * U
  TA::TArray<double> tmp;
  tmp("i,j") = A("i,k") * U("k,j");
  A("i,j")   = U("k,i").conj() * tmp("k,j");
  subtract_diagonal_tensor_inplace(A, S); // A -= SIGMA

  const auto norm = A("i,j").norm(world).get();
  BOOST_CHECK_SMALL(norm, tol);

  world.gop.fence();
}

template <typename Derived>
void ReferenceFixture<Derived>::svd_rightvectors_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  auto [S, VT] = Derived::template svd<TA::SVD::RightVectors>(A, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::Derived", exact_singular_values, S, tol);


  // Since A is Hermitian, VT is also (the c-transpose) A's eigenvectors
  // A <- VT * A * VT**H
  TA::TArray<double> tmp;
  tmp("i,j") = A("i,k") * VT("j,k").conj();
  A("i,j")   = VT("i,k") * tmp("k,j");
  subtract_diagonal_tensor_inplace(A, S); // A -= SIGMA

  const auto norm = A("i,j").norm(world).get();
  BOOST_CHECK_SMALL(norm, tol);


  world.gop.fence();
}

template <typename Derived>
void ReferenceFixture<Derived>::svd_allvectors_test(TA::World& world) {
  world.gop.fence();

  auto trange = gen_trange(N, {128ul});

  // Generate Reference Tensor in TA
  using array_type = TA::TArray<double>;
  auto A = generate_ta_reference<array_type>(world, trange);

  auto [S, U, VT] = Derived::template svd<TA::SVD::AllVectors>(A, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  compare_replicated_vector("TiledArray::Derived", exact_singular_values, S, tol);
  
  // Recreate SVD
  // A <- U**H * A * VT**H
  TA::TArray<double> tmp;
  tmp("i,j") = A("i,k") * VT("j,k").conj();
  A("i,j")   = U("k,i").conj() * tmp("k,j");
  subtract_diagonal_tensor_inplace(A, S); // A -= SIGMA

  const auto norm = A("i,j").norm(world).get();
  BOOST_CHECK_SMALL(norm, tol);


  world.gop.fence();
}
