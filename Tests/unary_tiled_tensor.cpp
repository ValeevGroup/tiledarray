#include "TiledArray/unary_tiled_tensor.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct UnaryTiledTensorFixture : public AnnotatedArrayFixture {

  UnaryTiledTensorFixture() : utt(aa, std::negate<int>()) {

  }

  UnaryTiledTensor<array_annotation, std::negate<int> > utt;
};

BOOST_FIXTURE_TEST_SUITE( unary_tiled_tensor_suite, UnaryTiledTensorFixture )

BOOST_AUTO_TEST_CASE( range )
{
  BOOST_CHECK_EQUAL(utt.range(), a.range());
  BOOST_CHECK_EQUAL(utt.size(), a.size());
  BOOST_CHECK_EQUAL(utt.trange(), a.trange());
}

BOOST_AUTO_TEST_CASE( vars )
{
  BOOST_CHECK_EQUAL(utt.vars(), aa.vars());
}

BOOST_AUTO_TEST_CASE( shape )
{
  BOOST_CHECK_EQUAL(utt.is_dense(), a.is_dense());
#ifndef NDEBUG
  BOOST_CHECK_THROW(utt.get_shape(), TiledArray::Exception);
#endif
}

BOOST_AUTO_TEST_CASE( location )
{
  BOOST_CHECK((& utt.get_world()) == (& a.get_world()));
  BOOST_CHECK(utt.get_pmap() == a.get_pmap());
  for(std::size_t i = 0; i < utt.size(); ++i) {
    BOOST_CHECK(! utt.is_zero(i));
    BOOST_CHECK_EQUAL(utt.owner(i), a.owner(i));
    BOOST_CHECK_EQUAL(utt.is_local(i), a.is_local(i));
  }
}

BOOST_AUTO_TEST_SUITE_END()
