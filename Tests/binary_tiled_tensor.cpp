#include "TiledArray/binary_tiled_tensor.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct BinaryTiledTensorFixture : public AnnotatedArrayFixture {

  BinaryTiledTensorFixture() {

  }

};



BOOST_FIXTURE_TEST_SUITE( binary_tiled_tensor_suite, BinaryTiledTensorFixture )

BOOST_AUTO_TEST_CASE( constructors )
{
  BinaryTiledTensor<array_annotation, array_annotation, std::plus<int> > utt(aa, aa, std::plus<int>());
}


BOOST_AUTO_TEST_SUITE_END()
