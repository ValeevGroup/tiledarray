#include "TiledArray/contraction_tensor.h"
#include "TiledArray/tile.h"
#include "TiledArray/contraction.h"
#include <world/tr1/memory.h>
#include "unit_test_config.h"


using namespace TiledArray;
using namespace TiledArray::expressions;

struct ContractionTensorFixture {
  typedef Tile<int, GlobalFixture::coordinate_system> TileN;
  typedef TileN::range_type range_type;
  typedef TileN::index index;
  typedef math::Contraction<std::size_t> cont_op;
  typedef ContractionTensor<TileN,TileN> ContT;

  ContractionTensorFixture() : ct(t2, t3, cont) { }

  // make a tile to be permuted
  static TileN make_tile(TileN::value_type value) {
    index start(0);
    index finish(5);
    range_type r(start, finish);

    return TileN(r, value);
  }

  static std::string make_var_list(std::size_t first, std::size_t last) {
    assert(abs(last - first) <= 24);
    assert(last < 24);

    std::string result;
    result += 'a' + first;
    for(++first; first != last; ++first) {
      result += ",";
      result += 'a' + first;
    }

    return result;
  }

  static const std::shared_ptr<math::Contraction<std::size_t> > cont;
  static const TileN t2;
  static const TileN t3;

  ContT ct;
}; // struct ContractionTensorFixture


const std::shared_ptr<math::Contraction<std::size_t> > ContractionTensorFixture::cont(new math::Contraction<std::size_t>(
    VariableList(ContractionTensorFixture::make_var_list(0, GlobalFixture::coordinate_system::dim)),
    VariableList(ContractionTensorFixture::make_var_list(1, GlobalFixture::coordinate_system::dim + 1))));

const ContractionTensorFixture::TileN ContractionTensorFixture::t2 = make_tile(2);
const ContractionTensorFixture::TileN ContractionTensorFixture::t3 = make_tile(3);


BOOST_FIXTURE_TEST_SUITE( contraction_tensor_suite , ContractionTensorFixture )

BOOST_AUTO_TEST_CASE( dimension_accessor )
{
  BOOST_CHECK_EQUAL(ct.dim(), 2u);
  BOOST_CHECK_EQUAL(ct.size().front(), t2.size().front());
  BOOST_CHECK_EQUAL(ct.size().back(), t3.size().back());

  const int I = std::accumulate(t2.range().finish().begin() + 1,
      t2.range().finish().end(), 1, std::multiplies<int>());
  BOOST_CHECK_EQUAL(ct.volume(), I);
  BOOST_CHECK_EQUAL(ct.order(), t2.order());
}

BOOST_AUTO_TEST_CASE( constructor )
{
  // Test primary constructor
  {
    BOOST_REQUIRE_NO_THROW(ContT x(t2, t3, cont));
    ContT x(t2, t3, cont);
    BOOST_CHECK_EQUAL(x.dim(), 2u);
    BOOST_CHECK_EQUAL(x.size().front(), t2.size().front());
    BOOST_CHECK_EQUAL(x.size().back(), t3.size().back());
    const int I = std::accumulate(t2.range().finish().begin() + 1,
        t2.range().finish().end(), 1, std::multiplies<int>());
    BOOST_CHECK_EQUAL(x.volume(), I);
    BOOST_CHECK_EQUAL(x.order(), t2.order());
  }

  // test copy constructor
  {
    BOOST_REQUIRE_NO_THROW(ContT x(ct));
    ContT x(ct);
    BOOST_CHECK_EQUAL(x.dim(), ct.dim());
    BOOST_CHECK_EQUAL_COLLECTIONS(x.size().begin(), x.size().end(), ct.size().begin(), ct.size().end());
    BOOST_CHECK_EQUAL(x.volume(), ct.volume());
    BOOST_CHECK_EQUAL(x.order(), ct.order());
  }
}


BOOST_AUTO_TEST_CASE( assignment_operator )
{
  TileN t0;
  ContT x(t0, t0, cont);

  // Check initial conditions
  BOOST_CHECK_EQUAL(x.dim(), 2u);
  BOOST_CHECK_EQUAL(x.size().front(), 0ul);
  BOOST_CHECK_EQUAL(x.size().back(), 0ul);
  BOOST_CHECK_EQUAL(x.volume(), 0ul);
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK_EQUAL(x.order(), t0.order());

  x = ct;

  // Check that the tensor was copied.
  BOOST_CHECK_EQUAL(x.dim(), ct.dim());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.size().begin(), x.size().end(), ct.size().begin(), ct.size().end());
  BOOST_CHECK_EQUAL(x.volume(), ct.volume());
  BOOST_CHECK_EQUAL(x.order(), ct.order());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), ct.begin(), ct.end());
}

BOOST_AUTO_TEST_CASE( contraction )
{
  // Calculate the dimensions of the packed contraction
  const int M = t2.size().front();
  const int N = t3.size().back();
  const int I = std::accumulate(t2.size().begin() + 1,
      t2.size().end(), 1, std::multiplies<int>());

  // Construct matrixes that match the packed dimensions of the to tiles.
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> m2(M, I);
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> m3(I, N);

  for(int m = 0; m < M; ++m)
    for(int i = 0; i < I; ++i)
      m2(m, i) = 2;

  for(int i = 0; i < I; ++i)
    for(int n = 0; n < N; ++n)
      m3(i, n) = 3;

  // Do a test contraction.
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> mc = m2 * m3;


  for(ContT::size_type i = 0; i < ct.volume(); ++i) {
    // Check that each element is correct
    BOOST_CHECK_EQUAL(ct[i], mc(0,0));
  }

  TileN::size_type i = 0;
  for(ContT::const_iterator it = ct.begin(); it != ct.end(); ++it, ++i) {
    // Check that iteration works correctly
    BOOST_CHECK_EQUAL(*it, mc(0,0));
  }
}


BOOST_AUTO_TEST_SUITE_END()
