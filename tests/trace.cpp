#include "unit_test_config.h"
#include <TiledArray/tensor/trace.h>
#include <TiledArray/tensor/tensor.h> // Defines trace of Tensor<T> (T numeric)
using namespace TiledArray;

// List of types commonly encountered by TA for which trace is NOT defined
using trace_not_defined =
    boost::mpl::list<int, float, double, std::complex<float>,
        std::complex<double>, Tensor<Tensor<int>>, Tensor<Tensor<float>>>;

// List of types commonly encountered by TA for which trace IS defined
using trace_is_defined  =
    boost::mpl::list<Tensor<int>, Tensor<float>, Tensor<double>,
      Tensor<std::complex<float>>, Tensor<std::complex<double>>>;

BOOST_AUTO_TEST_SUITE(trace_is_defined_class)

BOOST_AUTO_TEST_CASE_TEMPLATE(not_defined, TileType, trace_not_defined){
  BOOST_CHECK(detail::TraceIsDefined<TileType>::value == false);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_defined, TileType, trace_is_defined){
  BOOST_CHECK(detail::TraceIsDefined<TileType>::value);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(not_defined_v, TileType, trace_not_defined){
  BOOST_CHECK(detail::trace_is_defined_v<TileType> == false);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_defined_v, TileType, trace_is_defined){
  BOOST_CHECK(detail::trace_is_defined_v<TileType>);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(enable_if_trace_is_defined_t)

/* This should not compile
BOOST_AUTO_TEST_CASE_TEMPLATE(not_defined, TileType, trace_not_defined){
  using type_to_test = detail::enable_if_trace_is_defined_t<TileType>;
  BOOST_CHECK(std::is_void_v<type_to_test> == false);
}*/

BOOST_AUTO_TEST_CASE_TEMPLATE(is_defined, TileType, trace_is_defined){
  BOOST_CHECK(std::is_void_v<detail::enable_if_trace_is_defined_t<TileType>>);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_defined_u, TileType, trace_is_defined){
  using type_to_test = detail::enable_if_trace_is_defined_t<TileType, int>;
  constexpr bool is_same = std::is_same_v<type_to_test, int>;
  BOOST_CHECK(is_same);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(trace_free_fxn)

/* This shouldn't compile
BOOST_AUTO_TEST_CASE_TEMPLATE(not_defined, TileType, trace_not_defined){
  auto r = trace(TileType{});
}*/

/* To test the trace free function we assume that Trace<T>::operator() has been
 * tested and works. Thus all we need to do is make sure that trace forwards
 * arguments correctly. We do this by creating a 10 by 10 matrix and filling it
 * with 2's (as whatever element type the tensor holds). The correct trace of
 * such a matrix is 20.
 */
BOOST_AUTO_TEST_CASE_TEMPLATE(is_defined, TileType, trace_is_defined){
  using value_type = typename TileType::value_type;
  TileType t(Range{10, 10}, value_type{2});
  value_type corr = 20;
  auto tr = trace(t);

  BOOST_CHECK(tr == corr);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(reslt_of_trace_t)

/* This should not compile
BOOST_AUTO_TEST_CASE_TEMPLATE(not_defined, TileType, trace_not_defined){
    using type = result_of_trace_t<TileType>;
}*/

BOOST_AUTO_TEST_CASE_TEMPLATE(is_defined, TileType, trace_is_defined){
  using value_type = typename TileType::value_type;
  using type = result_of_trace_t<TileType>;
  constexpr bool is_same = std::is_same_v<value_type, type>;
  BOOST_CHECK(is_same);
}

BOOST_AUTO_TEST_SUITE_END()
