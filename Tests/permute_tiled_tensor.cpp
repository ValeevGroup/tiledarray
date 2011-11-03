#include "TiledArray/permute_tiled_tensor.h"
#include "TiledArray/array.h"
#include "TiledArray/expressions.h"
#include "unit_test_config.h"
#include "array_fixture.h"
#include "tensor_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct PermuteTiledTensorFixture : public AnnotatedArrayFixture {
  typedef PermuteTensorFixture::PermN PermN;
  typedef PermuteTiledTensor<array_annotation, GlobalFixture::coordinate_system::dim> PTT;

  PermuteTiledTensorFixture() : ptt(aa, p) { }


  static const PermN p;


  PTT ptt;
};


const PermuteTiledTensorFixture::PermN PermuteTiledTensorFixture::p(PermuteTensorFixture::make_perm());

BOOST_FIXTURE_TEST_SUITE( permute_tiled_tensor_suite, PermuteTiledTensorFixture )

BOOST_AUTO_TEST_CASE( range )
{
  BOOST_CHECK_EQUAL(ptt.range(), p ^ a.range());
  BOOST_CHECK_EQUAL(ptt.size(), a.size());
  BOOST_CHECK_EQUAL(ptt.trange(), p ^ a.trange());
}

BOOST_AUTO_TEST_CASE( vars )
{
  BOOST_CHECK_EQUAL(ptt.vars(), p ^ aa.vars());
}

BOOST_AUTO_TEST_CASE( shape )
{
  BOOST_CHECK_EQUAL(ptt.is_dense(), a.is_dense());
#ifndef NDEBUG
  BOOST_CHECK_THROW(ptt.get_shape(), TiledArray::Exception);
#endif
}

BOOST_AUTO_TEST_CASE( location )
{
  BOOST_CHECK((& ptt.get_world()) == (& a.get_world()));
  BOOST_CHECK(ptt.get_pmap() == a.get_pmap());
  for(std::size_t i = 0; i < ptt.size(); ++i) {
    BOOST_CHECK(! ptt.is_zero(i));
    BOOST_CHECK_EQUAL(ptt.owner(i), a.owner(i));
    BOOST_CHECK_EQUAL(ptt.is_local(i), a.is_local(i));
  }
}

BOOST_AUTO_TEST_CASE( result )
{
  for(PTT::const_iterator it = ptt.begin(); it != ptt.end(); ++it) {
    // Convert result ordinal index to input ordinal index
    array_annotation::size_type input_ord = aa.range().ord(-p ^ ptt.range().idx(it.index()));

    // Permute the input tile for comparision with the result.
    array_annotation::const_reference input = aa[input_ord];
    array_annotation::value_type ptensor = p ^ input.get();

    // Compare the permuted input with the result.
    BOOST_CHECK_EQUAL(it->get().range(), ptensor.range());

    array_annotation::value_type::const_iterator input_it = ptensor.begin();
    PTT::value_type::const_iterator result_it = it->get().begin();
    for(; result_it != it->get().end(); ++result_it, ++input_it)
      BOOST_CHECK_EQUAL(*result_it, 2 * (*input_it));
  }
}

BOOST_AUTO_TEST_SUITE_END()
