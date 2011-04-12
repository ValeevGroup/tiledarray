#include "TiledArray/variable_list_math.h"
#include "unit_test_config.h"
#include "math_fixture.h"

using TiledArray::math::BinaryOp;
using TiledArray::math::UnaryOp;
using TiledArray::expressions::VariableList;

// Variable list math fixture
struct VariableListMathFixture : public MathFixture {

  // Setup function
  VariableListMathFixture() : result() { }

  // variables
  VariableList result;
};


BOOST_FIXTURE_TEST_SUITE( variable_list_math_suite, VariableListMathFixture )

BOOST_AUTO_TEST_CASE( construct )
{
  // Construct a binary VariableList operation object
  BOOST_REQUIRE_NO_THROW((BinaryOp<VariableList, array_annotation, array_annotation, std::plus>()));

  // Construct a binary VariableList contraction operation object
  BOOST_REQUIRE_NO_THROW((BinaryOp<VariableList, array_annotation, array_annotation, std::multiplies>()));

  // Construct a unary VariableList operation object
  BOOST_REQUIRE_NO_THROW((UnaryOp<VariableList, array_annotation, std::negate>()));
}

BOOST_AUTO_TEST_CASE( default_binary_op )
{
  BinaryOp<VariableList, array_annotation, array_annotation, std::plus> plus_op;
  // Check that the result is equal to the argument variable lists.
  plus_op(result, a1, a2);
  BOOST_CHECK_EQUAL(result, a1.vars());
  BOOST_CHECK_EQUAL(result, a2.vars());

#ifdef TA_EXCEPTION_ERROR
  // Check for an exception when the variable lists do not match.
  BOOST_CHECK_THROW(plus_op(result, a1, a3), std::runtime_error);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( contraction_op )
{
  BinaryOp<VariableList, array_annotation, array_annotation, std::multiplies> contraction_op;

  VariableList r(a1.vars()[0] + "," + a3.vars()[GlobalFixture::coordinate_system::dim - 1]);

  // Check the contraction operation.
  contraction_op(result, a1, a3);
  BOOST_CHECK_EQUAL(result, r);
}

BOOST_AUTO_TEST_CASE( default_unary_op )
{
  // Check the default unary operation
  UnaryOp<VariableList, array_annotation, std::negate> negate_op;
  negate_op(result, a1);
  BOOST_CHECK_EQUAL(result, a1.vars());
}

BOOST_AUTO_TEST_SUITE_END()
