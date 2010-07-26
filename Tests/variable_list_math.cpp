#include "TiledArray/math.h"
#include "unit_test_config.h"
#include "TiledArray/variable_list.h"

using TiledArray::math::BinaryOp;
using TiledArray::math::UnaryOp;
using TiledArray::expressions::VariableList;

// Variable list math fixture
struct VariableListMathFixture {
  // static constants
  static const VariableList a;
  static const VariableList b;
  static const VariableList x;

  // Setup function
  VariableListMathFixture() : r() { }

  // variables
  VariableList r;
};

const VariableList VariableListMathFixture::a("a,b,c");
const VariableList VariableListMathFixture::b("a,b,c");
const VariableList VariableListMathFixture::x("x,y,z");

BOOST_FIXTURE_TEST_SUITE( variable_list_math_suite, VariableListMathFixture )

BOOST_AUTO_TEST_CASE( default_binary_op )
{
  BinaryOp<VariableList, VariableList, VariableList, std::plus> plus_op;
  // Check that the result is equal to the argument variable lists.
  BOOST_CHECK_EQUAL(a, b);
  BOOST_CHECK_NE(r, a);
  BOOST_CHECK_NE(r, b);
  plus_op(r, a, b);
  BOOST_CHECK_EQUAL(r, a);
  BOOST_CHECK_EQUAL(r, b);

#ifdef TA_EXCEPTION_ERROR
  // Check for an exception when the variable lists do not match.
  BOOST_CHECK_THROW(plus_op(r, a, x), std::runtime_error);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( contraction_op )
{
  BinaryOp<VariableList, VariableList, VariableList, std::multiplies> contraction_op;

  // Check the contraction operation.
  contraction_op(r, VariableList("a,i,b"), VariableList("x,i,y"));
  BOOST_CHECK_EQUAL(r, VariableList("a,x,b,y"));
  contraction_op(r, VariableList("a,i,b"), VariableList("x,i"));
  BOOST_CHECK_EQUAL(r, VariableList("a,x,b"));
  contraction_op(r, VariableList("a,i,b"), VariableList("i,x"));
  BOOST_CHECK_EQUAL(r, VariableList("a,b,x"));
  contraction_op(r, VariableList("a,i,b"), VariableList("i"));
  BOOST_CHECK_EQUAL(r, VariableList("a,b"));
  contraction_op(r, VariableList("a,i,b"), VariableList());
  BOOST_CHECK_EQUAL(r, VariableList("a,i,b"));
  contraction_op(r, VariableList("a,i"), VariableList("x,i,y"));
  BOOST_CHECK_EQUAL(r, VariableList("x,a,y"));
  contraction_op(r, VariableList("a,i"), VariableList("x,i"));
  BOOST_CHECK_EQUAL(r, VariableList("a,x"));
  contraction_op(r, VariableList("a,i"), VariableList("i,x"));
  BOOST_CHECK_EQUAL(r, VariableList("a,x"));
  contraction_op(r, VariableList("a,i"), VariableList("i"));
  BOOST_CHECK_EQUAL(r, VariableList("a"));
  contraction_op(r, VariableList("i,a"), VariableList("x,i,y"));
  BOOST_CHECK_EQUAL(r, VariableList("x,a,y"));
  contraction_op(r, VariableList("i,a"), VariableList("x,i"));
  BOOST_CHECK_EQUAL(r, VariableList("x,a"));
  contraction_op(r, VariableList("i,a"), VariableList("i,x"));
  BOOST_CHECK_EQUAL(r, VariableList("a,x"));
  contraction_op(r, VariableList("i,a"), VariableList("i"));
  BOOST_CHECK_EQUAL(r, VariableList("a"));
  contraction_op(r, VariableList("i,a"), VariableList());
  BOOST_CHECK_EQUAL(r, VariableList("i,a"));
  contraction_op(r, VariableList("i"), VariableList("x,i,y"));
  BOOST_CHECK_EQUAL(r, VariableList("x,y"));
  contraction_op(r, VariableList("i"), VariableList("x,i"));
  BOOST_CHECK_EQUAL(r, VariableList("x"));
  contraction_op(r, VariableList("i"), VariableList("i,x"));
  BOOST_CHECK_EQUAL(r, VariableList("x"));
  contraction_op(r, VariableList("i"), VariableList("i"));
  BOOST_CHECK_EQUAL(r, VariableList());
}

BOOST_AUTO_TEST_CASE( default_unary_op )
{
  // Check the default unary operation
  UnaryOp<VariableList, VariableList, std::negate> negate_op;
  BOOST_CHECK_NE(r, a);
  negate_op(r, a);
  BOOST_CHECK_EQUAL(r, a);
}

BOOST_AUTO_TEST_SUITE_END()
