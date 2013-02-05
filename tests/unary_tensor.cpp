#include "TiledArray/unary_tensor.h"
#include "TiledArray/array.h"
#include "TiledArray/functional.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct UnaryTensorFixture : public AnnotatedTensorFixture {
  typedef TensorExpression<array_annotation::value_type> tensor_expression;

  UnaryTensorFixture() : utt(make_unary_tensor(a(vars), make_unary_tile_op(std::negate<int>()))) {
    utt.eval(utt.vars(), std::shared_ptr<tensor_expression::pmap_interface>(
        new TiledArray::detail::BlockedPmap(* GlobalFixture::world, a.size()))).get();
  }

  tensor_expression utt;
};

BOOST_FIXTURE_TEST_SUITE( unary_tensor_suite, UnaryTensorFixture )

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
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(utt.get_shape(), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( location )
{
  BOOST_CHECK((& utt.get_world()) == (& a.get_world()));
  for(std::size_t i = 0; i < utt.size(); ++i) {
    BOOST_CHECK(! utt.is_zero(i));
    BOOST_CHECK_EQUAL(utt.owner(i), a.owner(i));
    BOOST_CHECK_EQUAL(utt.is_local(i), a.is_local(i));
  }
}

BOOST_AUTO_TEST_CASE( result )
{
  for(tensor_expression::const_iterator it = utt.begin(); it != utt.end(); ++it) {
    array_annotation::const_reference input = a.find(it.index());

    BOOST_CHECK_EQUAL(it->get().range(), input.get().range());

    array_annotation::value_type::const_iterator input_it = input.get().begin();
    tensor_expression::value_type::const_iterator result_it = it->get().begin();
    for(; result_it != it->get().end(); ++result_it, ++input_it)
      BOOST_CHECK_EQUAL(*result_it, -(*input_it));
  }
}

BOOST_AUTO_TEST_CASE( result_negate_sparse )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(i % 2)
      odd.set(i);

  // Construct argument tensors
  ArrayN aodd(world, tr, odd);
  aodd.set_all_local(1);

  //Negate the tiled array. The result should be odd tiles are filled.
  ArrayN aresult = make_unary_tensor(aodd(vars), make_unary_tile_op(std::negate<int>()));

  BOOST_CHECK(! aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      ++local_count;
      if(i % 2) {
        BOOST_CHECK(!aresult.is_zero(i));
        madness::Future<ArrayN::value_type> tile = aresult.find(i);
        BOOST_REQUIRE(tile.probe());
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], -1);
      } else
        BOOST_CHECK(aresult.is_zero(i));
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_CASE( result_negate_sparse_with_permute )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(i % 5)
      odd.set(i);

  // Construct argument tensors
  ArrayN aodd(world, tr, odd);
  aodd.set_all_local(1);

  //Negate the tiled array. The result should be odd tiles are filled.
  ArrayN aresult(world, tr, TiledArray::detail::Bitset<>(tr.tiles().volume()));
  aresult("c,b,a") = make_unary_tensor(aodd(vars), make_unary_tile_op(std::negate<int>()));

  BOOST_CHECK(! aresult.is_dense());

  world.gop.fence();


  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      ++local_count;
      if(i < 25) {
        BOOST_CHECK(aresult.is_zero(i));
      } else {
        BOOST_CHECK(!aresult.is_zero(i));
        madness::Future<ArrayN::value_type> tile = aresult.find(i);
        BOOST_REQUIRE(tile.probe());
        BOOST_CHECK_EQUAL(tile.get().range(), aresult.trange().make_tile_range(i));
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], -1);
      }
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_SUITE_END()
