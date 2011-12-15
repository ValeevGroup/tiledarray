#include "TiledArray/contraction_tiled_tensor.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct ContractionTiledTensorFixture : public AnnotatedArrayFixture {
  typedef ContractionTiledTensor<array_annotation, array_annotation> CTT;

  ContractionTiledTensorFixture() : ctt(a(left_var), a(right_var)) { }

  ~ContractionTiledTensorFixture() {
    GlobalFixture::world->gop.fence();
  }


  static const VariableList left_var;
  static const VariableList right_var;
  static const std::shared_ptr<math::Contraction> cont;

  CTT ctt;
};

const VariableList ContractionTiledTensorFixture::left_var(
    AnnotatedArrayFixture::make_var_list(0, GlobalFixture::coordinate_system::dim));
const VariableList ContractionTiledTensorFixture::right_var(
    AnnotatedArrayFixture::make_var_list(1, GlobalFixture::coordinate_system::dim + 1));

const std::shared_ptr<math::Contraction> ContractionTiledTensorFixture::cont(new math::Contraction(
    VariableList(AnnotatedArrayFixture::make_var_list(0, GlobalFixture::coordinate_system::dim)),
    VariableList(AnnotatedArrayFixture::make_var_list(1, GlobalFixture::coordinate_system::dim + 1)),
    GlobalFixture::coordinate_system::get_order()));


BOOST_FIXTURE_TEST_SUITE( contraction_tiled_tensor_suite, ContractionTiledTensorFixture )

BOOST_AUTO_TEST_CASE( range )
{
  DynamicTiledRange dtr = cont->contract_trange(a.trange(), a.trange());
  BOOST_CHECK_EQUAL(ctt.range(), dtr.tiles());
  BOOST_CHECK_EQUAL(ctt.size(), dtr.tiles().volume());
  BOOST_CHECK_EQUAL(ctt.trange(), dtr);
}

BOOST_AUTO_TEST_CASE( vars )
{
  VariableList var2(left_var.data().front() + "," + right_var.data().back());
  BOOST_CHECK_EQUAL(ctt.vars(), var2);
}

BOOST_AUTO_TEST_CASE( shape )
{
  BOOST_CHECK_EQUAL(ctt.is_dense(), a.is_dense());
#ifndef NDEBUG
  BOOST_CHECK_THROW(ctt.get_shape(), TiledArray::Exception);
#endif
}

BOOST_AUTO_TEST_CASE( location )
{
  BOOST_CHECK((& ctt.get_world()) == (& a.get_world()));
  BOOST_CHECK(ctt.get_pmap() == a.get_pmap());
  for(std::size_t i = 0; i < ctt.size(); ++i) {
    BOOST_CHECK(! ctt.is_zero(i));
    BOOST_CHECK_EQUAL(ctt.owner(i), a.owner(i));
    BOOST_CHECK_EQUAL(ctt.is_local(i), a.is_local(i));
  }
}

BOOST_AUTO_TEST_CASE( result )
{
  // Get the dimensions of the contraction
  const array_annotation::size_type A = a.trange().elements().size().front();
  const array_annotation::size_type B = std::accumulate(a.trange().elements().size().begin() + 1,
      a.trange().elements().size().end(), 1, std::multiplies<array_annotation::size_type>());
  const array_annotation::size_type C = a.trange().elements().size().back();

  // Construct equivalent matrix.
  Eigen::MatrixXi left(A, B);
  Eigen::MatrixXi right(B, C);
  Eigen::MatrixXi result(A, C);

  for(std::size_t i = 0; i < a.size(); ++i) {
    array_annotation::const_reference tensor = a.find(i);
    for(array_annotation::value_type::const_iterator it = tensor.get().begin(); it != tensor.get().end(); ++it) {
      left.array()(a.trange().elements().ord(tensor.get().range().idx(it - tensor.get().begin()))) =
          *it;
    }
  }

  for(std::size_t i = 0; i < a.size(); ++i) {
    array_annotation::const_reference tensor = a.find(i);
    for(array_annotation::value_type::const_iterator it = tensor.get().begin(); it != tensor.get().end(); ++it) {
      right.array()(a.trange().elements().ord(tensor.get().range().idx(it - tensor.get().begin()))) =
          *it;
    }
  }

  result = left * right;


  ctt.eval(ctt.vars());

  for(CTT::const_iterator it = ctt.begin(); it != ctt.end(); ++it) {
    CTT::value_type::const_iterator result_it = it->get().begin();
    for(; result_it != it->get().end(); ++result_it)
      BOOST_CHECK_EQUAL(*result_it, result(ctt.trange().elements().ord(it->get().range().idx(result_it - it->get().begin()))));
  }
}


BOOST_AUTO_TEST_SUITE_END()
