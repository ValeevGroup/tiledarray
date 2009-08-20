#include "tile_math.h"
#include "tile.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/transform_iterator.hpp>

using namespace TiledArray;

struct TileMathFixture {
  typedef Tile<double, 3> Tile3;
  typedef Tile3::index_type index_type;
  typedef Tile3::volume_type volume_type;
  typedef Tile3::size_array size_array;
  typedef Tile3::range_type range_type;
  typedef detail::AnnotatedTile<double, detail::decreasing_dimension_order> ATile;

  TileMathFixture() {

    r.resize(index_type(0,0,0), index_type(5,5,5));

    tr.resize(r.size(), 0.0);
    t1.resize(r.size(), 1.0);
    t2.resize(r.size(), 2.0);
    t3.resize(r.size(), 3.0);

  }

  ~TileMathFixture() { }

  range_type r;
  Tile3 tr;
  Tile3 t1;
  Tile3 t2;
  Tile3 t3;
};

template<typename InIter, typename T>
bool check_val(InIter first, InIter last, const T& v, const T& tol = 0.000001) {
  for(; first != last; ++first)
    if(*first > v + tol || *first < v - tol)
      return false;

  return true;

}

BOOST_FIXTURE_TEST_SUITE( tile_math_test , TileMathFixture )

BOOST_AUTO_TEST_CASE( value_exp )
{
  double d = 1.0;
  BOOST_REQUIRE_NO_THROW(detail::ValueExp<double> ve(d));
  detail::ValueExp<double> ve(d);
  BOOST_CHECK_CLOSE(ve.eval(), 1.0, 0.000001);
}

BOOST_AUTO_TEST_CASE( zip_op )
{
  boost::tuple<const ATile::value_type&, const ATile::value_type&> tup_it =
      boost::make_tuple(t1.at(0), t2.at(0));

  detail::ZipOp<ATile::value_type, ATile::value_type, ATile::value_type,
      std::plus> op;

  BOOST_CHECK_CLOSE(op(tup_it), 3.0, 0.000001);

}

BOOST_AUTO_TEST_CASE( tile_op )
{
  detail::TileOp<ATile, ATile, ATile, std::plus> op;

  ATile result = op(t1("a,b,c"), t2("a,b,c"));
  BOOST_CHECK(check_val(result.begin(), result.end(), 3.0));
}


BOOST_AUTO_TEST_CASE( addition )
{
  tr = t1("a,b,c") + t2("a,b,c");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
}

BOOST_AUTO_TEST_CASE( scalar_addition )
{
  tr = t1("a,b,c") + 2.0;
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
}

BOOST_AUTO_TEST_CASE( subraction )
{
  tr = t3("a,b,c") - t2("a,b,c");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 1.0));
}

BOOST_AUTO_TEST_CASE( scalar_subraction )
{
  tr = t3("a,b,c") - 2.0;
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 1.0));
}

BOOST_AUTO_TEST_CASE( scalar_multiplication )
{
  tr = t2("a,b,c") * 3.0;
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 6.0));
}

BOOST_AUTO_TEST_CASE( negate )
{
  tr = -t2("i,j,k");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), -2.0));
}

BOOST_AUTO_TEST_CASE( contraction )
{
  tr = t2("a,i,b") * t3("x,i,y");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("a,b,i") * t3("x,i,y");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("a,b,i") * t3("x,y,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("a,b,i") * t3("i,c,d");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
  tr = t2("i,j,k") * t3("i,j,k");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 3.0));
}

BOOST_AUTO_TEST_SUITE_END()
