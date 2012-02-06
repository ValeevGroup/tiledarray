#include "TiledArray/binary_tiled_tensor.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct BinaryTiledTensorFixture : public AnnotatedArrayFixture {
  typedef BinaryTiledTensor<array_annotation, array_annotation, std::plus<int> > BTT;

  BinaryTiledTensorFixture() : btt(a(vars), a(vars), std::plus<int>()) {

  }

  BTT btt;
};



BOOST_FIXTURE_TEST_SUITE( binary_tiled_tensor_suite, BinaryTiledTensorFixture )

BOOST_AUTO_TEST_CASE( range )
{
  BOOST_CHECK_EQUAL(btt.range(), a.range());
  BOOST_CHECK_EQUAL(btt.size(), a.size());
  BOOST_CHECK_EQUAL(btt.trange(), a.trange());
}

BOOST_AUTO_TEST_CASE( vars )
{
  BOOST_CHECK_EQUAL(btt.vars(), aa.vars());
}

BOOST_AUTO_TEST_CASE( shape )
{
  BOOST_CHECK_EQUAL(btt.is_dense(), a.is_dense());
#ifndef NDEBUG
  BOOST_CHECK_THROW(btt.get_shape(), TiledArray::Exception);
#endif
}

BOOST_AUTO_TEST_CASE( location )
{
  BOOST_CHECK((& btt.get_world()) == (& a.get_world()));
  BOOST_CHECK(btt.get_pmap() == a.get_pmap());
  for(std::size_t i = 0; i < btt.size(); ++i) {
    BOOST_CHECK(! btt.is_zero(i));
    BOOST_CHECK_EQUAL(btt.owner(i), a.owner(i));
    BOOST_CHECK_EQUAL(btt.is_local(i), a.is_local(i));
  }
}

BOOST_AUTO_TEST_CASE( result )
{
  btt.eval(btt.vars()).get();
  for(BTT::const_iterator it = btt.begin(); it != btt.end(); ++it) {
    array_annotation::const_reference input = a.find(it.index());

    BOOST_CHECK_EQUAL(it->get().range(), input.get().range());

    array_annotation::value_type::const_iterator input_it = input.get().begin();
    BTT::value_type::const_iterator result_it = it->get().begin();
    for(; result_it != it->get().end(); ++result_it, ++input_it)
      BOOST_CHECK_EQUAL(*result_it, 2 * (*input_it));
  }
}

BOOST_AUTO_TEST_SUITE_END()
