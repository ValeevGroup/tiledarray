#include "TiledArray/binary_tiled_tensor.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct BinaryTiledTensorFixture : public ArrayFixture {

  BinaryTiledTensorFixture() {

  }

};



BOOST_FIXTURE_TEST_SUITE( binary_tiled_tensor_suite, BinaryTiledTensorFixture )

BOOST_AUTO_TEST_CASE( constructors )
{
  AnnotatedArray<ArrayN> aal(a("a,b,c"));
  AnnotatedArray<ArrayN> aar(a("a,b,c"));
  BinaryTiledTensor<AnnotatedArray<ArrayN>, AnnotatedArray<ArrayN>, std::plus<int> > utt(aal, aar, std::plus<int>());
}


BOOST_AUTO_TEST_SUITE_END()
