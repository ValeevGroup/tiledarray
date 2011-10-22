#include "TiledArray/unary_tiled_tensor.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct UnaryTiledTensorFixture : public AnnotatedArrayFixture {

  UnaryTiledTensorFixture() {

  }

};

BOOST_FIXTURE_TEST_SUITE( unary_tiled_tensor_suite, UnaryTiledTensorFixture )

BOOST_AUTO_TEST_CASE( constructors )
{
  UnaryTiledTensor<array_annotation, std::negate<int> > utt(aa, std::negate<int>());
}


BOOST_AUTO_TEST_SUITE_END()
