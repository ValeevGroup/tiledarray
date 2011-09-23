#include "TiledArray/annotated_array.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct AnnotatedArrayFixture : public ArrayFixture {
  typedef AnnotatedArray<ArrayN> array_annotation;
  typedef AnnotatedArray<const ArrayN> const_array_annotation;

  AnnotatedArrayFixture() : vars(make_var_list()), aa(a, vars) { }


  static std::string make_var_list(std::size_t first = 0,
      std::size_t last = GlobalFixture::element_coordinate_system::dim) {
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

  VariableList vars;
  array_annotation aa;
}; // struct AnnotatedArrayFixture



BOOST_FIXTURE_TEST_SUITE( annotated_array_suite , AnnotatedArrayFixture )

BOOST_AUTO_TEST_CASE( range_accessor )
{
  BOOST_CHECK(aa.range() == a.range());
  BOOST_CHECK_EQUAL(aa.size(), a.size());
}

BOOST_AUTO_TEST_CASE( vars_accessor )
{
  VariableList v(make_var_list());
  BOOST_CHECK_EQUAL(aa.vars(), v);
}

BOOST_AUTO_TEST_CASE( tile_data )
{
  ArrayN::const_iterator a_it = a.begin();
  for(array_annotation::const_iterator aa_it = aa.begin(); aa_it != aa.end(); ++aa_it, ++a_it) {
    BOOST_CHECK(aa_it.base() == a_it);
    ArrayN::value_type::const_iterator a_tile_it = a_it->get().begin();
    aa_it.base()->get();
    for(array_annotation::value_type::const_iterator aa_tile_it = aa_it->begin(); aa_tile_it != aa_it->end(); ++aa_tile_it, ++a_tile_it) {
      BOOST_CHECK(aa_tile_it == a_tile_it);
      BOOST_CHECK(*aa_tile_it == *a_tile_it);
    }
  }
}

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(array_annotation at1(a, vars));
  array_annotation at1(a, vars);
  BOOST_CHECK(at1.begin().base() == a.begin());
  BOOST_CHECK(at1.end().base() == a.end());
  BOOST_CHECK_EQUAL(at1.range(), a.range());
  BOOST_CHECK_EQUAL(at1.size(), r.volume());
  BOOST_CHECK_EQUAL(at1.vars(), vars);

  BOOST_REQUIRE_NO_THROW(array_annotation at2(aa));
  array_annotation at2(aa);
  BOOST_CHECK(at2.begin().base() == a.begin());
  BOOST_CHECK(at2.end().base() == a.end());
  BOOST_CHECK_EQUAL(at2.range(), a.range());
  BOOST_CHECK_EQUAL(at2.size(), a.size());
  BOOST_CHECK_EQUAL(at2.vars(), vars);

}

BOOST_AUTO_TEST_SUITE_END()
