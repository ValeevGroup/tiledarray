#include "TiledArray/tensor.h"
#include <math.h>
#include <utility>
#include "unit_test_config.h"
#include "range_fixture.h"
#include <world/bufar.h>

//using namespace TiledArray;

// Element Generation object test.
template<typename T, typename Index>
class gen {
public:
  const T operator ()(const Index& i) {
    typedef typename Index::index index_t;
	index_t result = 0;
    index_t e = 0;
    for(unsigned int d = 0; d < Index::dim(); ++d) {
      e = i[d] * static_cast<index_t>(std::pow(10.0, static_cast<int>(Index::dim()-d-1)));
      result += e;
    }

    return result;
  }
};

struct TensorFixture {
  typedef TiledArray::expressions::Tensor<int, StaticRange<GlobalFixture::element_coordinate_system> > TensorN;
  typedef TensorN::value_type value_type;
  typedef TensorN::range_type::index index;
  typedef TensorN::size_type size_type;
  typedef TensorN::range_type::size_array size_array;
  typedef TensorN::range_type range_type;
  typedef Permutation PermN;

  static const range_type r;

  TensorFixture() : t(r, 1) {
  }

  ~TensorFixture() { }

  // get a unique value for the given index
  static value_type get_value(const index i) {
    index::value_type x = 1;
    value_type result = 0;
    for(index::const_iterator it = i.begin(); it != i.end(); ++it, x *= 10)
      result += *it * x;

    return result;
  }

  // make a tile to be permuted
  static TensorN make_tile() {
    index start(0);
    index finish(0);
    index::value_type i = 3;
    for(index::iterator it = finish.begin(); it != finish.end(); ++it, ++i)
      *it = i;

    range_type r(start, finish);
    TensorN result(r);
    for(range_type::const_iterator it = r.begin(); it != r.end(); ++it)
      result[*it] = get_value(*it);

    return result;
  }

  // make permutation definition object
  static PermN make_perm() {
    std::array<std::size_t, GlobalFixture::coordinate_system::dim> temp;
    for(std::size_t i = 0; i < temp.size(); ++i)
      temp[i] = i + 1;

    temp.back() = 0;

    return PermN(temp);
  }

  TensorN t;
};

const TensorFixture::range_type TensorFixture::r = TensorFixture::range_type(index(0), index(5));


template<typename InIter, typename T>
bool check_val(InIter first, InIter last, const T& v, const T& tol) {
  for(; first != last; ++first)
    if(*first > v + tol || *first < v - tol)
      return false;

  return true;

}

BOOST_FIXTURE_TEST_SUITE( tile_suite , TensorFixture )

BOOST_AUTO_TEST_CASE( range_accessor )
{
  BOOST_CHECK_EQUAL_COLLECTIONS(t.range().start().begin(), t.range().start().end(),
      r.start().begin(), r.start().end());  // check start accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(t.range().finish().begin(), t.range().finish().end(),
      r.finish().begin(), r.finish().end());// check finish accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(t.range().size().begin(), t.range().size().end(),
      r.size().begin(), r.size().end());    // check size accessor
  BOOST_CHECK_EQUAL(t.range().volume(), r.volume());// check volume accessor
  BOOST_CHECK_EQUAL(t.range(), r);          // check range accessof
}

BOOST_AUTO_TEST_CASE( element_access )
{
  // check operator[] with array coordinate index
  BOOST_CHECK_EQUAL(t[index(0)], 1);
  BOOST_CHECK_EQUAL(t[index(4)], 1);


  // check operator[] with ordinal index
  BOOST_CHECK_EQUAL(t[0], 1);
  BOOST_CHECK_EQUAL(t[r.volume() - 1], 1);

  // check out of range error
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(t[r.finish()], Exception);
  BOOST_CHECK_THROW(t[r.volume()], Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( iteration )
{
  for(TensorN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_CLOSE(*it, 1.0, 0.000001);

  TensorN t1(t);
  TensorN::iterator it1 = t1.begin();
  *it1 = 2.0;

  // check iterator assignment
  BOOST_CHECK_CLOSE(*it1, 2.0, 0.000001);
  TensorN t2;
  BOOST_CHECK_EQUAL(t2.begin(), t2.end());
}

BOOST_AUTO_TEST_CASE( constructor )
{
  // check default constructor
  BOOST_REQUIRE_NO_THROW(TensorN t0);
  TensorN t0;
  BOOST_CHECK_EQUAL(t0.range().volume(), 0u);
  BOOST_CHECK_EQUAL(t0.begin(), t0.end());

  // check copy constructor
  {
    BOOST_REQUIRE_NO_THROW(TensorN tc(t));
    TensorN tc(t);
    BOOST_CHECK_EQUAL(tc.range(), t.range());
    for(TensorN::const_iterator it = tc.begin(); it != tc.end(); ++it)
      BOOST_CHECK_EQUAL(*it, 1);
  }

  // check constructing with a range
  {
    BOOST_REQUIRE_NO_THROW(TensorN t1(r));
    TensorN t1(r);
    BOOST_CHECK_EQUAL(t1.range(), t.range());
    for(TensorN::const_iterator it = t1.begin(); it != t1.end(); ++it)
      BOOST_CHECK_EQUAL(*it, int());
  }

  // check constructing with a range and initial value.
  {
    BOOST_REQUIRE_NO_THROW(TensorN t2(r, 1));
    TensorN t2(r, 1);
    BOOST_CHECK_EQUAL(t2.range(), t.range());
    for(TensorN::const_iterator it = t2.begin(); it != t2.end(); ++it)
      BOOST_CHECK_EQUAL(*it, 1);
  }

  // check constructing with range and iterators.
  {
    std::vector<int> data;
    int v = r.volume();
    for(int i = 0; i < v; ++i)
      data.push_back(i);

    BOOST_REQUIRE_NO_THROW(TensorN t3(r, data.begin()));
    TensorN t3(r, data.begin());
    BOOST_CHECK_EQUAL(t3.range(), r);
    BOOST_CHECK_EQUAL_COLLECTIONS(t3.begin(), t3.end(), data.begin(), data.end());
  }
}

BOOST_AUTO_TEST_CASE( element_assignment )
{

  // verify preassignment conditions
  BOOST_CHECK_NE(t[1], 2);
  // check that assignment returns itself.
  BOOST_CHECK_EQUAL(t[1] = 2, 2) ;
  // check for correct assignment.
  BOOST_CHECK_EQUAL(t[1], 2);
}

//BOOST_AUTO_TEST_CASE( resize )
//{
//  TensorN t1;
//  BOOST_CHECK_EQUAL(t1.range().volume(), 0u);
//  t1.resize(r);
//  // check new dimensions.
//  BOOST_CHECK_EQUAL(t1.range(), r);
//  // check new element initialization
//  BOOST_CHECK_EQUAL(std::find_if(t1.begin(), t1.end(), std::bind1st(std::not_equal_to<int>(), int())), t1.end());
//
//  TensorN t2;
//  BOOST_CHECK_EQUAL(std::distance(t2.begin(), t2.end()), 0);
//  t2.resize(r, 1);
//  BOOST_CHECK_EQUAL(t2.range(), r);
//  BOOST_CHECK_EQUAL(std::distance(t2.begin(), t2.end()), static_cast<long>(r.volume()));
//  // check for new element initialization
//  BOOST_CHECK_EQUAL(std::find_if(t2.begin(), t2.end(), std::bind1st(std::not_equal_to<int>(), 1)), t2.end());
//
//  // Check that the common elements are maintained in resize operation.
//  range_type r2(index(0), index(6));
//  t2.resize(r2, 2);
//  BOOST_CHECK_EQUAL(t2.range(), r2); // check new dimensions
//  BOOST_CHECK_EQUAL(static_cast<std::size_t>(std::distance(t2.begin(), t2.end())), r2.volume());
//  for(range_type::const_iterator it = r2.begin(); it != r2.end(); ++it) {
//    if(r.includes(*it))
//      BOOST_CHECK_EQUAL(t2[*it], 1);
//    else
//      BOOST_CHECK_EQUAL(t2[*it], 2);
//  }
//}

BOOST_AUTO_TEST_CASE( serialization )
{
  std::size_t buf_size = (t.range().volume() * sizeof(int) + sizeof(TensorN))*2;
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  oar & t;
  std::size_t nbyte = oar.size();
  oar.close();

  TensorN ts;
  madness::archive::BufferInputArchive iar(buf,nbyte);
  iar & ts;
  iar.close();

  delete [] buf;

  BOOST_CHECK_EQUAL(t.range(), ts.range());
  BOOST_CHECK_EQUAL_COLLECTIONS(t.begin(), t.end(), ts.begin(), ts.end());
}

BOOST_AUTO_TEST_CASE( addition )
{
  const TensorN t1(r, 1);
  const TensorN t2(r, 2);

  // Check that += operator
  t += t1;
  for(TensorN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 2);
}

BOOST_AUTO_TEST_CASE( subtraction )
{
  const TensorN t1(r, 1);
  const TensorN t2(r, 2);

  // Check that += operator
  t -= t2;
  for(TensorN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -1);
}

BOOST_AUTO_TEST_SUITE_END()

