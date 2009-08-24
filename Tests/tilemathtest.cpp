#include "tile_math.h"
#include "tile.h"
#include <Eigen/core>
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

using namespace TiledArray;

struct TileMathFixture {
  typedef Tile<double, 3> Tile3;
  typedef Tile3::index_type index_type;
  typedef Tile3::volume_type volume_type;
  typedef Tile3::size_array size_array;
  typedef Tile3::range_type range_type;
  typedef math::AnnotatedTile<double, detail::decreasing_dimension_order> ATile;

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
    if((*first > (v + tol)) || (*first < (v - tol)))
      return false;

  return true;

}

BOOST_FIXTURE_TEST_SUITE( tile_math_test , TileMathFixture )

BOOST_AUTO_TEST_CASE( contraction_func )
{
  Eigen::aligned_allocator<double> alloc;
  double* a = alloc.allocate(125);
  double* b = alloc.allocate(125);
  for(std::size_t i = 0; i < 125; ++i) {
    alloc.construct(a + i, 3.0);
    alloc.construct(b + i, 2.0);
  }
  double* c = alloc.allocate(625);
  for(std::size_t i = 0; i < 625; ++i) {
    alloc.construct(c + i, 0.0);
  }

  // c[5,5] = a[5,5]T * b[5,5]
  detail::contract<double, detail::decreasing_dimension_order>(5, 5, 5, 5, 5, a, b, c);
  BOOST_CHECK(check_val(c, c + 625, 30.0));
  std::fill(c, c + 625, 0.0);
  // c[25,5] = a[5,25]T * b[5,5]
  detail::contract<double, detail::decreasing_dimension_order>(1, 25, 5, 5, 5, a, b, c);
  BOOST_CHECK(check_val(c, c + 625, 30.0));
  std::fill(c, c + 625, 0.0);
/*
  // c[25,25] = a[5,25]T * b[5,25]
  detail::contract<double, detail::decreasing_dimension_order>(1, 25, 1, 25, 5, a, b, c);
  BOOST_CHECK(check_val(c, c + 625, 30.0));
  std::fill(c, c + 625, 0.0);
  // c[25,1] = a[5,25]T * b[5,1]
  detail::contract<double, detail::decreasing_dimension_order>(1, 25, 25, 1, 5, a, b, c);
  BOOST_CHECK(check_val(c, c + 625, 30.0));
  std::fill(c, c + 625, 0.0);
  // c[1,5] = a[5,1]T * b[5,5]
  detail::contract<double, detail::decreasing_dimension_order>(25, 1, 5, 5, 5, a, b, c);
  BOOST_CHECK(check_val(c, c + 625, 30.0));
  std::fill(c, c + 625, 0.0);
  // c[1,25] = a[5,1]T * b[5,25]
  detail::contract<double, detail::decreasing_dimension_order>(25, 1, 1, 25, 5, a, b, c);
  BOOST_CHECK(check_val(c, c + 625, 30.0));
  std::fill(c, c + 625, 0.0);
  // c[1,1] = a[5,1]T * b[5,1]
  detail::contract<double, detail::decreasing_dimension_order>(25, 1, 25, 1, 5, a, b, c);
  BOOST_CHECK(check_val(c, c + 625, 30.0));
  std::fill(c, c + 625, 0.0);
  // c[1,5] = a[5,1]T * b[5,5]
  detail::contract<double, detail::decreasing_dimension_order>(1, 1, 5, 5, 5, a, b, c);
  BOOST_CHECK(check_val(c, c + 25, 30.0));
  std::fill(c, c + 625, 0.0);
  // c[1,25] = a[5,1]T * b[5,25]
  detail::contract<double, detail::decreasing_dimension_order>(1, 1, 1, 25, 5, a, b, c);
  BOOST_CHECK(check_val(c, c + 25, 30.0));
  std::fill(c, c + 625, 0.0);
  // c[1,25] = a[5,1]T * b[5,25]
  detail::contract<double, detail::decreasing_dimension_order>(1, 1, 25, 1, 5, a, b, c);
  BOOST_CHECK(check_val(c, c + 25, 30.0));
  std::fill(c, c + 625, 0.0);
  // c[1,1] = a[5,1]T * b[5,1]
  detail::contract<double, detail::decreasing_dimension_order>(1, 1, 1, 1, 5, a, b, c);
  BOOST_CHECK(check_val(c, c + 5, 30.0));
  std::fill(c, c + 625, 0.0);
*/

  for(std::size_t i = 0; i < 125; ++i) {
    alloc.destroy(a + i);
    alloc.destroy(b + i);
  }
  for(std::size_t i = 0; i < 625; ++i) {
    alloc.destroy(c + i);
  }
  alloc.deallocate(a, 125);
  alloc.deallocate(b, 125);
  alloc.deallocate(c, 625);
  a = NULL;
  b = NULL;
  c = NULL;
}

BOOST_AUTO_TEST_CASE( value_exp )
{
  double d = 1.0;
  BOOST_REQUIRE_NO_THROW(math::ValueExp<double> ve(d));
  math::ValueExp<double> ve(d);
  BOOST_CHECK_CLOSE(ve.eval(), 1.0, 0.000001);
}

BOOST_AUTO_TEST_CASE( zip_op )
{
  boost::tuple<const ATile::value_type&, const ATile::value_type&> tup_it =
      boost::make_tuple(t1.at(0), t2.at(0));

  math::ZipOp<ATile::value_type, ATile::value_type, ATile::value_type,
      std::plus> op;

  BOOST_CHECK_CLOSE(op(tup_it), 3.0, 0.000001);

}

BOOST_AUTO_TEST_CASE( tile_op )
{
  math::TileOp<ATile, ATile, ATile, std::plus> op;

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
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("a,b,i") * t3("x,i,y");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("a,b,i") * t3("x,y,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("a,b,i") * t3("i,c,d");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("i,a,b") * t3("c,d,i");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
  tr = t2("i,j,k") * t3("i,j,k");
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
}

BOOST_AUTO_TEST_SUITE_END()
