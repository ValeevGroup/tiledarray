#include "unit_test_config.h"
#include "global_fixture.h"
#include "TiledArray/bitset.h"

struct BitsetFixture {
  typedef TiledArray::detail::Bitset<> Bitset;

  BitsetFixture() : set(size) { }

  static const std::size_t size;
  static const std::size_t blocks;
  Bitset set;
};

const std::size_t BitsetFixture::size = sizeof(BitsetFixture::Bitset::block_type) * 8 * 10.5;
const std::size_t BitsetFixture::blocks = 11;

// =============================================================================
// Bitset Test Suite


BOOST_FIXTURE_TEST_SUITE( bitset_suite, BitsetFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  // Check for error free construction
  BOOST_REQUIRE_NO_THROW(Bitset b(32));

  Bitset b64(64);
  // Check that the size of the bitset is correct
  BOOST_CHECK_EQUAL(b64.size(), 64);
  BOOST_CHECK_EQUAL(b64.num_blocks(), 1);

  // check that all bits are correctly initialized to false.
  for(std::size_t i = 0; i < b64.size(); ++i)
    BOOST_CHECK(! b64[i]);


  // Check for error free copy construction
  BOOST_REQUIRE_NO_THROW(Bitset bc(set));


  Bitset bc(set);
  // Check that the size of the bitset is correct
  BOOST_CHECK_EQUAL(bc.size(), size);
  BOOST_CHECK_EQUAL(bc.size(), set.size());
  BOOST_CHECK_EQUAL(bc.num_blocks(), blocks);
  BOOST_CHECK_EQUAL(bc.num_blocks(), set.num_blocks());

  // check that all bits are correctly initialized to false.
  for(std::size_t i = 0; i < bc.size(); ++i)
    BOOST_CHECK(! bc[i]);
}

BOOST_AUTO_TEST_SUITE_END()

