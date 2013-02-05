#include "TiledArray/annotated_tensor.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

std::string AnnotatedTensorFixture::make_var_list(std::size_t first, std::size_t last) {
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


BOOST_FIXTURE_TEST_SUITE( annotated_tensor_suite , AnnotatedTensorFixture )

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
    BOOST_CHECK(aa_it == a_it);
    aa_it->get();
    for(std::size_t it = 0; it != aa_it->get().size(); ++it) {
      const ArrayN::value_type::value_type& ai = a_it->get()[it];
      const array_annotation::value_type::value_type& aai = aa_it->get()[it];
      BOOST_CHECK(&aai == &ai);
      BOOST_CHECK(aai == ai);
    }
  }
}

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(array_annotation at1(expressions::make_annotatied_tensor(a, vars)));
  array_annotation at1(expressions::make_annotatied_tensor(a, vars));
  BOOST_CHECK_EQUAL(at1.range(), a.range());
  BOOST_CHECK_EQUAL(at1.size(), r.volume());
  BOOST_CHECK_EQUAL(at1.vars(), vars);
}

BOOST_AUTO_TEST_CASE( eval )
{
  aa.eval(vars, std::shared_ptr<array_annotation::pmap_interface>(new TiledArray::detail::BlockedPmap(* GlobalFixture::world, a.size()))).get();
  ArrayN::const_iterator a_it = a.begin();

  for(std::size_t i = 0; i < a.size(); ++i) {
    if(a.is_local(i)) {
      madness::Future<ArrayN::value_type> a_tile = a.find(i);
      madness::Future<array_annotation::value_type> aa_tile = aa[i];

      BOOST_CHECK_EQUAL(aa_tile.get().range(), a_tile.get().range());
      BOOST_CHECK_EQUAL_COLLECTIONS(aa_tile.get().begin(), aa_tile.get().end(),
          a_tile.get().begin(), a_tile.get().end());
    }

  }
}

BOOST_AUTO_TEST_SUITE_END()
