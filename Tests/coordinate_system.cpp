#include "TiledArray/coordinate_system.h"
#include "unit_test_config.h"
#include "math_fixture.h"

using namespace TiledArray;
using namespace TiledArray::detail;

struct CoordinateSystemFixture {
  typedef GlobalFixture::coordinate_system CoordSysN;
  typedef GlobalFixture::element_coordinate_system ECoordSys;
  typedef CoordSysN::index index;
  typedef CoordSysN::ordinal_index ordinal_index;
  typedef CoordSysN::volume_type volume_type;
  typedef CoordSysN::size_array size_array;

  // Used for testing coordinate system sameness.
  template <unsigned int DIM, unsigned int Level, DimensionOrderType O, typename I>
  class CSTester {
    typedef TiledArray::CoordinateSystem<DIM, Level, O, I> coordinate_system;
  };

  CoordinateSystemFixture() : ca(a) {
    // fill test array with incremented values with the 1 in the least significant
    // position.
    ordinal_index i = 0;
    if(CoordSysN::order == increasing_dimension_order) {
      for(size_array::iterator it = a.begin(); it != a.end(); ++it)
        *it = ++i;
    } else {
      for(size_array::reverse_iterator it = a.rbegin(); it != a.rend(); ++it)
        *it = ++i;
    }

  }

  size_array a; // test array
  const size_array& ca;
};


BOOST_FIXTURE_TEST_SUITE( coord_sys_suite , CoordinateSystemFixture )

BOOST_AUTO_TEST_CASE( same_dim )
{
   BOOST_CHECK((same_cs_dim<CoordinateSystem<3>, CoordinateSystem<3> >::value));
   BOOST_CHECK(! (same_cs_dim<CoordinateSystem<3>, CoordinateSystem<2> >::value));

   BOOST_CHECK((same_cs_dim<FakeArray<int,CoordinateSystem<3> >, FakeArray<int,CoordinateSystem<3> > >::value));
   BOOST_CHECK(! (same_cs_dim<FakeArray<int,CoordinateSystem<3> >, FakeArray<int,CoordinateSystem<2> > >::value));
}

BOOST_AUTO_TEST_CASE( same_level )
{
   BOOST_CHECK((same_cs_level<CoordinateSystem<3, 1>, CoordinateSystem<3, 1> >::value));
   BOOST_CHECK(! (same_cs_level<CoordinateSystem<3, 1>, CoordinateSystem<3, 0> >::value));

   BOOST_CHECK((same_cs_level<FakeArray<int,CoordinateSystem<3> >, FakeArray<int,CoordinateSystem<3> > >::value));
   BOOST_CHECK(! (same_cs_level<FakeArray<int,CoordinateSystem<3, 0> >, FakeArray<int,CoordinateSystem<3> > >::value));
}

BOOST_AUTO_TEST_CASE( same_order )
{
   BOOST_CHECK((same_cs_order<CoordinateSystem<3>, CoordinateSystem<3> >::value));
   BOOST_CHECK(! (same_cs_order<CoordinateSystem<3, 1, decreasing_dimension_order>,
       CoordinateSystem<3, 1, increasing_dimension_order> >::value));

   BOOST_CHECK((same_cs_order<FakeArray<int,CoordinateSystem<3> >, FakeArray<int,CoordinateSystem<3> > >::value));
   BOOST_CHECK(! (same_cs_order<FakeArray<int,CoordinateSystem<3, 1, decreasing_dimension_order> >,
       FakeArray<int,CoordinateSystem<3, 1, increasing_dimension_order> > >::value));
}

BOOST_AUTO_TEST_CASE( same_index )
{
   BOOST_CHECK((same_cs_index<CoordinateSystem<3>, CoordinateSystem<3> >::value));
   BOOST_CHECK(! (same_cs_index<CoordinateSystem<3, 1, decreasing_dimension_order>,
       CoordinateSystem<3, 1, decreasing_dimension_order, int> >::value));

   BOOST_CHECK((same_cs_index<FakeArray<int,CoordinateSystem<3> >, FakeArray<int,CoordinateSystem<3> > >::value));
   BOOST_CHECK(! (same_cs_index<FakeArray<int,CoordinateSystem<3, 1, decreasing_dimension_order> >,
       FakeArray<int,CoordinateSystem<3, 1, decreasing_dimension_order, int> > >::value));
}

BOOST_AUTO_TEST_CASE( compatible )
{
 BOOST_CHECK((compatible_coordinate_system<CoordinateSystem<3>, CoordinateSystem<3> >::value));
 BOOST_CHECK((compatible_coordinate_system<CoordinateSystem<3>, CoordinateSystem<2> >::value));
 BOOST_CHECK(! (compatible_coordinate_system<CoordinateSystem<3, 0>, CoordinateSystem<3> >::value));
 BOOST_CHECK(! (compatible_coordinate_system<CoordinateSystem<3, 1, increasing_dimension_order>, CoordinateSystem<3> >::value));
 BOOST_CHECK(! (compatible_coordinate_system<CoordinateSystem<3, 1, decreasing_dimension_order, int>, CoordinateSystem<3> >::value));
}

BOOST_AUTO_TEST_CASE( coord_iterator )
{
  // Check with non-const container and increasing order template parameters
  // Todo: Test CoordIterator<size_array, increasing_dimension_order> member functions

  // Check with const container and increasing order template parameters
  // Todo: Test CoordIterator<const size_array, increasing_dimension_order> member functions

  // Check with non-const container and decreasing order template parameters
  // Todo: Test CoordIterator<size_array, decreasing_dimension_order> member functions

  // Check with const container and decreasing order template parameters
  // Todo: Test CoordIterator<const size_array, decreasing_dimension_order> member functions
}

BOOST_AUTO_TEST_CASE( iterator_selector )
{
  std::array<int, 3> a;
  const std::array<int, 3>& ca = a;
  for(std::size_t i = 0; i != a.size(); ++i)
    a[i] = i;

  // Check CoordSysN::end() returns an iterator to one past the least significant element
  BOOST_CHECK((CoordinateSystem<3,1,increasing_dimension_order>::begin(a) == a.begin()));
  BOOST_CHECK((CoordinateSystem<3,1,increasing_dimension_order>::begin(ca) == ca.begin()));
  BOOST_CHECK((CoordinateSystem<3,1,increasing_dimension_order>::end(a) == a.end()));
  BOOST_CHECK((CoordinateSystem<3,1,increasing_dimension_order>::end(ca) == ca.end()));

  // Check CoordSysN::begin() returns an iterator to the most significant element
  BOOST_CHECK((CoordinateSystem<3,1,increasing_dimension_order>::rbegin(a) == a.rbegin()));
  BOOST_CHECK((CoordinateSystem<3,1,increasing_dimension_order>::rbegin(ca) == ca.rbegin()));
  BOOST_CHECK((CoordinateSystem<3,1,increasing_dimension_order>::rend(a) == a.rend()));
  BOOST_CHECK((CoordinateSystem<3,1,increasing_dimension_order>::rend(ca) == ca.rend()));

  // Check CoordSysN::end() returns an iterator to one past the most significant element
  BOOST_CHECK((CoordinateSystem<3>::begin(a) == a.rbegin()));
  BOOST_CHECK((CoordinateSystem<3>::begin(ca) == ca.rbegin()));
  BOOST_CHECK((CoordinateSystem<3>::end(a) == a.rend()));
  BOOST_CHECK((CoordinateSystem<3>::end(ca) == ca.rend()));

  // Check CoordSysN::begin() returns an iterator to the least significant element
  BOOST_CHECK((CoordinateSystem<3>::rbegin(a) == a.begin()));
  BOOST_CHECK((CoordinateSystem<3>::rbegin(ca) == ca.begin()));
  BOOST_CHECK((CoordinateSystem<3>::rend(a) == a.end()));
  BOOST_CHECK((CoordinateSystem<3>::rend(ca) == ca.end()));
}

BOOST_AUTO_TEST_CASE( calc_weight )
{
  // Check that CoordSysN::calc_weight() calculates the correct weight
  // Todo: Generate the weight for array a and check for the correct values.
}

BOOST_AUTO_TEST_CASE( calc_index )
{
  // Check that CoordSysN::calc_index() calculates the correct index
  // Todo: That the entire range of ordinal indexes for a given size array produce correct coordinate indexes
}

BOOST_AUTO_TEST_CASE( calc_ordinal )
{
  // Check that CoordSysN::calc_ordinal() calculates the correct ordinal index
  // Todo: That the entire range of coordinate indexes for a given size array produce correct coordinate indexes

  // Check that CoordSysN::calc_ordinal() calculates the correct ordinal index
  // when a start index is given
  // Todo: That the entire range of coordinate indexes for a given size array produce correct coordinate indexes
}

BOOST_AUTO_TEST_CASE( key )
{
  // Check that CoordSysN::key() returns a complete key
  // Todo: All three complete key functions work correctly

  // Check that CoordSysN::key() returns a partial key
  // Todo: All three incomplete key functions work correctly
}

BOOST_AUTO_TEST_CASE( calc_volume )
{
  // Check the volume of a given size array.
  // Todo: Test the volume calculated by CoordSysN::calc_volume() for size array a.
}

BOOST_AUTO_TEST_CASE( increment_coordinate )
{
  // Check that coordinate indexes increment in the correctly
  // Todo: Test that all elements are traversed and in the correct order.
}

BOOST_AUTO_TEST_CASE( index_comparison )
{
  // Check less comparison for index
  // Todo: Test less works correctly for all points adjacent to a test point.

  // Check less_eq comparison for index
  // Todo: Test less_eq works correctly for all points adjacent to a test point.


  // Check greater comparison for index
  // Todo: Test greater works correctly for all points adjacent to a test point.


  // Check greater_eq comparison for index
  // Todo: Test greater_eq works correctly for all points adjacent to a test point.
}

BOOST_AUTO_TEST_SUITE_END()
