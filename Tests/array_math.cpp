#include "TiledArray/array_math.h"
#include "TiledArray/array.h"
#include <Eigen/Core>
#include "unit_test_config.h"

using namespace TiledArray;
using TiledArray::math::contract;

template <typename T>
struct TiledRangeFixture {
  typedef typename Array<T, 3>::tiled_range_type TRange3;
  typedef typename Array<T, 4>::tiled_range_type TRange4;
  typedef typename TRange3::tiled_range1_type TRange1;

  TiledRangeFixture() {
    const std::size_t d[] = {0, 5, 10, 15, 20, 25};
    const TRange1 dim(d, d + 5);
    const TRange1 dims[4] = {dim, dim, dim, dim};
    trng3.resize(dims, dims + 3);
    trng4.resize(dims, dims + 4);

    const TRange1 dim0(d, d + 5);
    const TRange1 dims0[3] = {dim0, dim0, dim0};
    trng0.resize(dims0, dims + 3);
  }

  ~TiledRangeFixture() { }

  TRange3 trng0;
  TRange3 trng3;
  TRange4 trng4;
}; // struct TiledRangeFixture

struct ArrayMathFixture : public TiledRangeFixture<int> {
  typedef Array<int, 0> Array0;
  typedef Array<int, 1> Array1;
  typedef Array<int, 2> Array2;
  typedef Array<int, 3> Array3;
  typedef Array<int, 4> Array4;
  typedef Array3::tile_type tile_type;
  typedef Array3::index_type index_type;
  typedef Array3::volume_type volume_type;
  typedef Array3::size_array size_array;
  typedef Array3::range_type range_type;
  typedef expressions::array::AnnotatedArray<int> AArray;

  ArrayMathFixture() : TiledRangeFixture<int>(), world(* GlobalFixture::world),
      ar(world, trng3), a1(world, trng3), a2(world, trng3), a3(world, trng3),
      a4(world, trng3)
  {
    world.gop.fence();
    int v;
    for(TRange3::range_type::const_iterator it = trng3.tiles().begin(); it != trng3.tiles().end(); ++it) {
      v = 0;
      tile_type t(trng3.tile(*it));
      ++v;
      std::fill(t.begin(), t.end(), v++);
      ar.insert(*it, t);
      std::fill(t.begin(), t.end(), v++);
      a1.insert(*it, t);
      std::fill(t.begin(), t.end(), v++);
      a2.insert(*it, t);
      std::fill(t.begin(), t.end(), v++);
      a3.insert(*it, t);
      std::fill(t.begin(), t.end(), v++);
      a4.insert(*it, t);
    }
  }

  template<typename T, unsigned int DIM, typename CS>
  static bool check_val(const Array<T, DIM, CS>& a, const T& val) {
    for(typename Array<T, DIM, CS>::const_iterator it = a.begin(); it != a.end(); ++it)
      for(typename Array<T, DIM, CS>::tile_type::const_iterator t_it = it->second.begin(); t_it != it->second.end(); ++t_it)
        if(*t_it == val)
          return false;

    return true;
  }

  template<typename T, unsigned int DIM, typename CS>
  static void fill(Array<T, DIM, CS>& a, const T& val) {
    for(typename Array<T, DIM, CS>::iterator it = a.begin(); it != a.end(); ++it)
      for(typename Array<T, DIM, CS>::tile_type::iterator t_it = it->second.begin();
          t_it != it->second.end(); ++t_it)
        *t_it = val;
  }

  ~ArrayMathFixture() { }

  madness::World& world;
  Array3 ar;
  Array3 a1;
  Array3 a2;
  Array3 a3;
  Array3 a4;
};

BOOST_FIXTURE_TEST_SUITE( array_math_suite , ArrayMathFixture )

BOOST_AUTO_TEST_CASE( contraction_func )
{

}

BOOST_AUTO_TEST_CASE( value_exp )
{
  double d = 1.0;
  BOOST_REQUIRE_NO_THROW(expressions::tile::ValueExp<double> ve(d));
  expressions::tile::ValueExp<double> ve(d);
  BOOST_CHECK_CLOSE(ve.eval(), 1.0, 0.000001);
}


BOOST_AUTO_TEST_CASE( array_op )
{
//  math::BinaryArrayOp<AArray, AArray, AArray, std::plus<double> > op;

}


BOOST_AUTO_TEST_CASE( addition )
{
  ar("a,b,c") = a1("a,b,c") + a2("a,b,c");
  BOOST_CHECK(check_val(ar, 3));
}

BOOST_AUTO_TEST_CASE( scalar_addition )
{
  ar("a,b,c") = a1("a,b,c") + 2.0;
  BOOST_CHECK(check_val(ar, 3));
}

BOOST_AUTO_TEST_CASE( subraction )
{
  ar("a,b,c") = a3("a,b,c") - a2("a,b,c");
  BOOST_CHECK(check_val(ar, 1));
}

BOOST_AUTO_TEST_CASE( scalar_subraction )
{
  ar("a,b,c") = a3("a,b,c") - 2.0;
  BOOST_CHECK(check_val(ar, 1));
}

BOOST_AUTO_TEST_CASE( scalar_multiplication )
{
  ar("a,b,c") = a2("a,b,c") * 3.0;
  BOOST_CHECK(check_val(ar, 6));
}

BOOST_AUTO_TEST_CASE( negate )
{
  ar("i,j,k") = -a2("i,j,k");
  BOOST_CHECK(check_val(ar, -2));
}

BOOST_AUTO_TEST_CASE( contraction )
{
  Array4 ar4(world, trng4);
  Array3 ar0(world, trng0);

  fill(ar4, 0);
  ar4("a,x,b,y") = a2("a,i,b") * a3("x,i,y");
  BOOST_CHECK(check_val(ar, 30));
  fill(a4, 0);
  ar4("a,x,y,b") = a2("a,i,b") * a3("x,y,i");
  BOOST_CHECK(check_val(ar4, 30));
  fill(ar4, 0);
  ar4("a,x,b,y") = a2("a,i,b") * a3("i,x,y");
  BOOST_CHECK(check_val(ar4, 30));
  fill(ar4, 0);
  ar4("a,x,b,y") = a2("a,b,i") * a3("x,i,y");
  BOOST_CHECK(check_val(ar4, 30));
  fill(ar4, 0);
  ar4("a,x,b,y") = a2("a,b,i") * a3("x,y,i");
  BOOST_CHECK(check_val(ar4, 30));
  fill(ar4, 0);
  ar4("a,b,x,y") = a2("a,b,i") * a3("i,x,y");
  BOOST_CHECK(check_val(ar4, 30));
  fill(ar4, 0);
  ar4("x,a,b,y") = a2("i,a,b") * a3("x,i,y");
  BOOST_CHECK(check_val(ar4, 30));
  fill(ar4, 0);
  ar4("x,a,y,b") = a2("i,a,b") * a3("x,y,i");
  BOOST_CHECK(check_val(ar4, 30));
  fill(ar4, 0);
  ar4("a,x,b,y") = a2("i,a,b") * a3("i,x,y");
  BOOST_CHECK(check_val(ar4, 30));
  fill(ar4, 0);
  ar0("") = a2("i,j,k") * a3("i,j,k");
  BOOST_CHECK(check_val(ar0, 750));
}

BOOST_AUTO_TEST_CASE( chain_expressions )
{
  Array4 ar4(world, trng4);

  ar4("a,c,b,d") = 6.0 * a2("a,i,b") * a3("c,i,d") + a3("a,i,b") * a4("c,i,d") - 1.0;
  BOOST_CHECK(check_val(ar4, 239));
}

BOOST_AUTO_TEST_SUITE_END()
