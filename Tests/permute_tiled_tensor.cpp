#include "TiledArray/permute_tiled_tensor.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "array_fixture.h"
#include "tensor_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct PermuteTiledTensorFixture : public AnnotatedArrayFixture {
  typedef PermuteTensorFixture::PermN PermN;

  PermuteTiledTensorFixture() {

  }


  static const PermN p;
};


const PermuteTiledTensorFixture::PermN PermuteTiledTensorFixture::p(PermuteTensorFixture::make_perm());

BOOST_FIXTURE_TEST_SUITE( permute_tiled_tensor_suite, PermuteTiledTensorFixture )

BOOST_AUTO_TEST_CASE( constructors )
{
  PermuteTiledTensor<array_annotation, GlobalFixture::coordinate_system::dim> ptt(aa, p);
}


BOOST_AUTO_TEST_SUITE_END()
