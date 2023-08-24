#include "heig_tests.h"     // EVP tests
#include "cholesky_tests.h" // Cholesky tests
#include "lu_tests.h"       // LU tests
#include "svd_tests.h"      // SVD tests
#include "qr_tests.h"       // QR tests

// SLATE linear algebra utilities
#include <TiledArray/conversions/slate.h>
#include <TiledArray/math/linalg/slate/cholesky.h>
#include <TiledArray/math/linalg/slate/lu.h>
#include <TiledArray/math/linalg/slate/heig.h>
#include <TiledArray/math/linalg/slate/svd.h>
#include <TiledArray/math/linalg/slate/qr.h>

namespace TA = TiledArray;
namespace slate_la = TA::math::linalg::slate;

struct SLATELinearAlgebraFixture : 
  ReferenceFixture<SLATELinearAlgebraFixture> {

  SLATELinearAlgebraFixture(int64_t N = 1000) : 
    ReferenceFixture<SLATELinearAlgebraFixture>(N) {}

  slate::Matrix<double> make_ref_slate(int64_t N, TA::SlateFunctors& slate_functors,
    MPI_Comm comm) {

    slate::Matrix<double> A(N, N, slate_functors.tileMb(), slate_functors.tileNb(), 
      slate_functors.tileRank(), slate_functors.tileDevice(), comm);
    
    A.insertLocalTiles();
    int64_t j_off = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {

      int64_t i_off = 0;
      for (int64_t i = 0; i < A.mt(); ++i) {

        if(A.tileIsLocal(i,j)) {
          auto T = A(i,j);
          for(auto jj = 0; jj < T.nb(); ++jj)
          for(auto ii = 0; ii < T.mb(); ++ii) {
            T.at(ii,jj) = matrix_element_generator(i_off+ii,j_off+jj);
          }
        }

        i_off += A.tileMbFunc()(i);
      }

      j_off += A.tileNbFunc()(j);
    }

    return A;
  }

  template <typename... Args>
  static auto heig(Args&&... args) { 
    return slate_la::heig(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto cholesky(Args&&... args) { 
    return slate_la::cholesky(std::forward<Args>(args)...); 
  }

  template <bool RetL, typename... Args>
  static auto cholesky_linv(Args&&... args) { 
    return slate_la::cholesky_linv<RetL>(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto cholesky_solve(Args&&... args) { 
    return slate_la::cholesky_solve(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto cholesky_lsolve(Args&&... args) { 
    return slate_la::cholesky_lsolve(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto lu_solve(Args&&... args) { 
    return slate_la::lu_solve(std::forward<Args>(args)...); 
  }

  template <typename... Args>
  static auto lu_inv(Args&&... args) { 
    return slate_la::lu_inv(std::forward<Args>(args)...); 
  }

  template<TA::SVD::Vectors Vectors, typename... Args>
  static auto svd(Args&&... args) {
    return slate_la::svd<Vectors>(std::forward<Args>(args)...);
  }

  template <bool QOnly, typename... Args>
  static auto householder_qr(Args&&... args) { 
    return slate_la::householder_qr<QOnly>(std::forward<Args>(args)...); 
  }


  template <typename Array>
  void tiled_array_to_slate_test(TA::TiledRange& trange, TA::World& world) {
  
    world.gop.fence();

    auto ref_ta = generate_ta_reference<Array>(world, trange);
    world.gop.fence();

    auto slate_matrix = TA::array_to_slate(ref_ta);
    world.gop.fence();
    BOOST_CHECK( slate_matrix.mt() == trange.dim(0).tile_extent() );
    BOOST_CHECK( slate_matrix.nt() == trange.dim(1).tile_extent() );
    BOOST_CHECK( slate_matrix.m()  == N );
    BOOST_CHECK( slate_matrix.n()  == N );

    TA::SlateFunctors slate_functors( trange, ref_ta.pmap() );
    auto ref_slate = this->make_ref_slate(N, slate_functors, MPI_COMM_WORLD);

    slate::add( 1.0, ref_slate, -1.0, slate_matrix );
    auto norm_diff = slate::norm(slate::Norm::Fro, slate_matrix);
    BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

    world.gop.fence();

  }

  template <typename Array>
  void slate_to_tiled_array_test(TA::TiledRange& trange, TA::World& world) {
  
    world.gop.fence();

    auto ref_ta = generate_ta_reference<Array>(world, trange);
    world.gop.fence();

    TA::SlateFunctors slate_functors( trange, ref_ta.pmap() );
    auto ref_slate = this->make_ref_slate(N, slate_functors, MPI_COMM_WORLD);

    world.gop.fence();
    auto test_ta = TA::slate_to_array<Array>(ref_slate, world);
    world.gop.fence();

    auto norm_diff = (ref_ta("i,j") - test_ta("i,j")).norm(world).get();
    BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

    world.gop.fence();

  }
};


BOOST_FIXTURE_TEST_SUITE(linear_algebra_suite_slate, SLATELinearAlgebraFixture)

using ta_test_types = boost::mpl::list<
    TA::DistArray<TA::Tensor<double>,              TA::DensePolicy>,
    TA::DistArray<btas::Tensor<double, TA::Range>, TA::DensePolicy>,
    TA::DistArray<TA::Tensor<double>,              TA::SparsePolicy>,
    TA::DistArray<btas::Tensor<double, TA::Range>, TA::SparsePolicy>
>;

// SLATE -> TA: tilings equal
BOOST_AUTO_TEST_CASE_TEMPLATE(slate_to_tiled_array_equal, array_type, ta_test_types) {
  auto trange = gen_trange(N, {static_cast<size_t>(128)});
  slate_to_tiled_array_test<array_type>(trange, *GlobalFixture::world);
};

// SLATE -> TA, random tiling
BOOST_AUTO_TEST_CASE_TEMPLATE(slate_to_tiled_array_random, array_type, ta_test_types) {
  auto trange = gen_trange(N, {107ul, 113ul, 211ul, 151ul});
  slate_to_tiled_array_test<array_type>(trange, *GlobalFixture::world);
};


// TA -> SLATE: tilings equal
BOOST_AUTO_TEST_CASE_TEMPLATE(tiled_array_to_slate_equal, array_type, ta_test_types) {
  auto trange = gen_trange(N, {static_cast<size_t>(128)});
  tiled_array_to_slate_test<array_type>(trange, *GlobalFixture::world);
};

// TA -> SLATE, random tiling
BOOST_AUTO_TEST_CASE_TEMPLATE(tiled_array_to_slate_random, array_type, ta_test_types) {
  auto trange = gen_trange(N, {107ul, 113ul, 211ul, 151ul});
  tiled_array_to_slate_test<array_type>(trange, *GlobalFixture::world);
};

// HEIG tests
LINALG_TEST_IMPL(heig_same_tiling);
//LINALG_TEST_IMPL(heig_diff_tiling);
//LINALG_TEST_IMPL(heig_generalized);

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

