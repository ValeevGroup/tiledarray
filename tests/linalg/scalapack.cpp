#include "heig_tests.h"     // EVP tests
#include "cholesky_tests.h" // Cholesky tests
#include "lu_tests.h"       // LU tests
#include "svd_tests.h"      // SVD tests
#include "qr_tests.h"       // QR tests

// ScaLAPACK linear algebra utilities
#include "TiledArray/math/linalg/scalapack/all.h"

namespace TA = TiledArray;
namespace scalapack = TA::math::linalg::scalapack;

struct ScaLAPACKLinearAlgebraFixture : 
  ReferenceFixture<ScaLAPACKLinearAlgebraFixture> {

  blacspp::Grid grid;
  scalapack::BlockCyclicMatrix<double> ref_matrix;  // XXX: Just double is fine?

  ScaLAPACKLinearAlgebraFixture(int64_t N = 1000, int64_t NB = 128) : 
    ReferenceFixture<ScaLAPACKLinearAlgebraFixture>(N),
    grid(blacspp::Grid::square_grid(MPI_COMM_WORLD)),  // XXX: Is this safe?
    ref_matrix(*GlobalFixture::world, grid, N, N, NB, NB) {

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        if (ref_matrix.dist().i_own(i, j)) {
          auto [i_local, j_local] = ref_matrix.dist().local_indx(i, j);
          ref_matrix.local_mat()(i_local, j_local) =
              matrix_element_generator(i, j);
        }
      }
    }

  }

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

  template <bool QOnly, typename... Args>
  static auto householder_qr(Args&&... args) { 
    return scalapack::householder_qr<QOnly>(std::forward<Args>(args)...); 
  }




  template <typename Array>
  void block_cyclic_to_tiled_array_test(TA::TiledRange& trange, TA::World& world) {

    world.gop.fence();

    // Generate Reference Tensor
    auto ref_ta = generate_ta_reference<Array>(world, trange);
    world.gop.fence();

    // Convert reference matrix to Tensor
    auto test_ta =
        scalapack::block_cyclic_to_array<Array>(ref_matrix, trange);
    world.gop.fence();

    auto norm_diff =
        (ref_ta("i,j") - test_ta("i,j")).norm(world).get();

    BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

    world.gop.fence();

  }

  template <typename Array>
  void tiled_array_to_block_cyclic_test(TA::TiledRange& trange, TA::World& world) {

    world.gop.fence();

    // Generate Reference Tensor
    auto ref_ta = generate_ta_reference<Array>(world, trange);
    world.gop.fence();

    // Convert reference tensor to matrix
    auto NB = ref_matrix.dist().nb();
    auto test_matrix = scalapack::array_to_block_cyclic(ref_ta, grid, NB, NB);
    world.gop.fence();

    double local_norm_diff =
        (test_matrix.local_mat() - ref_matrix.local_mat()).norm();
    local_norm_diff *= local_norm_diff;

    double norm_diff;
    MPI_Allreduce(&local_norm_diff, &norm_diff, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    norm_diff = std::sqrt(norm_diff);

    BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

    world.gop.fence();

  }

};


BOOST_FIXTURE_TEST_SUITE(linear_algebra_suite_scalapack, ScaLAPACKLinearAlgebraFixture)

using ta_test_types = boost::mpl::list<
    TA::DistArray<TA::Tensor<double>,              TA::DensePolicy>,
    TA::DistArray<btas::Tensor<double, TA::Range>, TA::DensePolicy>,
    TA::DistArray<TA::Tensor<double>,              TA::SparsePolicy>,
    TA::DistArray<btas::Tensor<double, TA::Range>, TA::SparsePolicy>
>;

// ScaLAPACK -> TA, tilings equal
BOOST_AUTO_TEST_CASE_TEMPLATE(block_cyclic_to_tiled_array_equal, array_type, ta_test_types) {
  auto [M, N] = ref_matrix.dims(); 
  auto NB     = ref_matrix.dist().nb();
  BOOST_REQUIRE_EQUAL(M,N); // TiledRangeRange only for square

  auto trange = gen_trange(N, {static_cast<size_t>(NB)});
  block_cyclic_to_tiled_array_test<array_type>(trange, *GlobalFixture::world);
};

// ScaLAPACK -> TA, tiled range smaller than NB
BOOST_AUTO_TEST_CASE_TEMPLATE(block_cyclic_to_tiled_array_all_small, array_type, ta_test_types) {
  auto [M, N] = ref_matrix.dims(); 
  auto NB     = ref_matrix.dist().nb();
  BOOST_REQUIRE_EQUAL(M,N); // TiledRangeRange only for square

  auto trange = gen_trange(N, {static_cast<size_t>(NB/2)});
  block_cyclic_to_tiled_array_test<array_type>(trange, *GlobalFixture::world);
};

// ScaLAPACK -> TA, random tiling
BOOST_AUTO_TEST_CASE_TEMPLATE(block_cyclic_to_tiled_array_random, array_type, ta_test_types) {
  auto [M, N] = ref_matrix.dims(); 
  BOOST_REQUIRE_EQUAL(M,N); // TiledRangeRange only for square

  auto trange = gen_trange(N, {107ul, 113ul, 211ul, 151ul});
  block_cyclic_to_tiled_array_test<array_type>(trange, *GlobalFixture::world);
};



// TA -> ScaLAPACK: tilings equal
BOOST_AUTO_TEST_CASE_TEMPLATE(tiled_array_to_block_cyclic_equal, array_type, ta_test_types) {
  auto [M, N] = ref_matrix.dims(); 
  auto NB     = ref_matrix.dist().nb();
  BOOST_REQUIRE_EQUAL(M,N); // TiledRangeRange only for square

  auto trange = gen_trange(N, {static_cast<size_t>(NB)});
  tiled_array_to_block_cyclic_test<array_type>(trange, *GlobalFixture::world);
};

// TA -> ScaLAPACK, tiled range smaller than NB
BOOST_AUTO_TEST_CASE_TEMPLATE(tiled_array_to_block_cyclic_all_small, array_type, ta_test_types) {
  auto [M, N] = ref_matrix.dims(); 
  auto NB     = ref_matrix.dist().nb();
  BOOST_REQUIRE_EQUAL(M,N); // TiledRangeRange only for square

  auto trange = gen_trange(N, {static_cast<size_t>(NB/2)});
  tiled_array_to_block_cyclic_test<array_type>(trange, *GlobalFixture::world);
};

// TA -> ScaLAPACK, random tiling
BOOST_AUTO_TEST_CASE_TEMPLATE(tiled_array_to_block_cyclic_random, array_type, ta_test_types) {
  auto [M, N] = ref_matrix.dims(); 
  BOOST_REQUIRE_EQUAL(M,N); // TiledRangeRange only for square

  auto trange = gen_trange(N, {107ul, 113ul, 211ul, 151ul});
  tiled_array_to_block_cyclic_test<array_type>(trange, *GlobalFixture::world);
};


BOOST_AUTO_TEST_CASE(const_tiled_array_to_block_cyclic) {
  // Just check that it compiles, meat is tested elsewhere
  using array_type = const TA::TArray<double>;
  using my_t = decltype(scalapack::array_to_block_cyclic(std::declval<array_type>(), std::declval<blacspp::Grid>(), std::declval<int64_t>(), std::declval<int64_t>()));
  constexpr auto my_bool = std::is_same_v<my_t,scalapack::BlockCyclicMatrix<double>>;
  BOOST_REQUIRE(my_bool);
};

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
