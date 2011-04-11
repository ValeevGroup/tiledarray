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
  BOOST_CHECK_EQUAL(b64.size(), 64ul);
  BOOST_CHECK_EQUAL(b64.num_blocks(), 1ul);

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

BOOST_AUTO_TEST_CASE( get )
{
  BOOST_CHECK(set.get() != NULL);
}

BOOST_AUTO_TEST_CASE( set_size )
{
  BOOST_CHECK_EQUAL(set.size(), size);
}

BOOST_AUTO_TEST_CASE( num_blocks )
{
  BOOST_CHECK_EQUAL(set.num_blocks(), blocks);
}

BOOST_AUTO_TEST_CASE( accessor )
{
  // check that all bits are correctly initialized to false.
  for(std::size_t i = 0; i < set.size(); ++i)
    BOOST_CHECK(! set[i]);

  // Check that exceptions are thrown when accessing an element that is out of range.
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(set[set.size()], std::out_of_range);
  BOOST_CHECK_THROW(set[set.size() + 1], std::out_of_range);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( set_bit )
{
  // Check that the bits are not set
  for(std::size_t i = 0; i < set.size(); ++i)
    BOOST_CHECK(! set[i]);

  // Check that we can set all bits
  for(std::size_t i = 0; i < set.size(); ++i) {
    set.set(i);

    for(std::size_t ii = 0; ii < set.size(); ++ii) {
      if(ii <= i)
        BOOST_CHECK(set[ii]);     // Check bits that are set
      else
        BOOST_CHECK(! set[ii]);   // Check bits that are not set
    }

    // Check setting a bit that is already set.
    set.set(i);
    BOOST_CHECK(set[i]);
  }

  // Check that we can unset all bits
  for(std::size_t i = 0; i < set.size(); ++i) {
    set.set(i, false);

    for(std::size_t ii = 0; ii < set.size(); ++ii) {
      if(ii <= i)
        BOOST_CHECK(! set[ii]); // Check bits that are not set
      else
        BOOST_CHECK(set[ii]);   // Check bits that are set
    }

    // Check setting a bit that is already set.
    set.set(i, false);
    BOOST_CHECK(! set[i]);
  }

  // Check that exceptions are thrown when accessing an element that is out of range.
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(set.set(set.size()), std::out_of_range);
  BOOST_CHECK_THROW(set.set(set.size() + 1), std::out_of_range);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( set_all )
{
  // Check that the bits are not set
  for(std::size_t i = 0; i < set.size(); ++i)
    BOOST_CHECK(! set[i]);

  set.set();

  // Check that the bits are set
  for(std::size_t i = 0; i < set.size(); ++i)
    BOOST_CHECK(set[i]);

  set.set();

  // Check that the bits are still set
  for(std::size_t i = 0; i < set.size(); ++i)
    BOOST_CHECK(set[i]);
}

BOOST_AUTO_TEST_CASE( reset_bit )
{
  set.set();

  // Check resetting a bit
  for(std::size_t i = 0; i < set.size(); ++i) {
    set.reset(i);

    for(std::size_t ii = 0; ii < set.size(); ++ii) {
      if(ii <= i)
        BOOST_CHECK(! set[ii]); // Check bits that are not set
      else
        BOOST_CHECK(set[ii]);   // Check bits that are set
    }
  }

  // Check resetting a bit that is not set
  for(std::size_t i = 0; i < set.size(); ++i) {
    set.reset(i);

    for(std::size_t ii = 0; ii < set.size(); ++ii)
      BOOST_CHECK(! set[ii]); // Check bits that are not set

  }

  // Check that exceptions are thrown when accessing an element that is out of range.
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(set.reset(set.size()), std::out_of_range);
  BOOST_CHECK_THROW(set.reset(set.size() + 1), std::out_of_range);
#endif // TA_EXCEPTION_ERROR
}


BOOST_AUTO_TEST_CASE( reset_all )
{
  set.reset();

  // Check that the bits are not set
  for(std::size_t i = 0; i < set.size(); ++i)
    BOOST_CHECK(! set[i]);

  set.set();
  set.reset();

  // Check that the bits are set
  for(std::size_t i = 0; i < set.size(); ++i)
    BOOST_CHECK(! set[i]);
}

BOOST_AUTO_TEST_CASE( bit_flip )
{
  // Check that the bits are not set
  for(std::size_t i = 0; i < set.size(); ++i)
    BOOST_CHECK(! set[i]);

  // Check that we can set all bits
  for(std::size_t i = 0; i < set.size(); ++i) {
    set.flip(i);

    for(std::size_t ii = 0; ii < set.size(); ++ii) {
      if(ii <= i)
        BOOST_CHECK(set[ii]);   // Check bits that are set
      else
        BOOST_CHECK(! set[ii]); // Check bits that are not set
    }
  }

  // Check that we can unset all bits
  for(std::size_t i = 0; i < set.size(); ++i) {
    set.flip(i);

    for(std::size_t ii = 0; ii < set.size(); ++ii) {
      if(ii <= i)
        BOOST_CHECK(! set[ii]); // Check bits that are not set
      else
        BOOST_CHECK(set[ii]);   // Check bits that are set
    }
  }

  // Check that exceptions are thrown when accessing an element that is out of range.
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(set.flip(set.size()), std::out_of_range);
  BOOST_CHECK_THROW(set.flip(set.size() + 1), std::out_of_range);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( flip_all )
{
  set.flip();

  // Check that the bits are set
  for(std::size_t i = 0; i < set.size(); ++i)
    BOOST_CHECK(set[i]);

  set.flip();

  // Check that the bits are not set
  for(std::size_t i = 0; i < set.size(); ++i)
    BOOST_CHECK(! set[i]);

  // Check that we can flip alternating bits
  for(std::size_t i = 0; i < set.size(); ++i)
    if(i % 2)
      set.flip(i);

  set.flip();

  for(std::size_t i = 0; i < set.size(); ++i) {
    if(i % 2)
      BOOST_CHECK(! set[i]);
    else
      BOOST_CHECK(set[i]);
  }


}

BOOST_AUTO_TEST_CASE( assignment )
{
  for(std::size_t i = 0; i < set.size(); ++i)
    set.set(i);

  Bitset b(size);

  // Check that assignment does not throw.
  BOOST_REQUIRE_NO_THROW(b = set);

  // Check that all bits were copied from set.
  for(std::size_t i = 0; i < set.size(); ++i)
    BOOST_CHECK(b[i]);

  // Check that assignment of bitsets with different size throws.
#ifdef TA_EXCEPTION_ERROR
  Bitset bad(size / 2);
  BOOST_CHECK_THROW(bad = set, std::range_error);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( bit_assignment )
{
  Bitset even(size);
  Bitset odd(size);
  for(std::size_t i = 0; i < set.size(); ++i) {
    set.set(i);
    if(i % 2)
      even.set(i);
    else
      odd.set(i);
  }


  // Check that and-assignment does not throw.
  BOOST_REQUIRE_NO_THROW(set &= even);

  // Check for correct and-assignement (evens are set)
  for(std::size_t i = 0; i < set.size(); ++i) {
    if(i % 2)
      BOOST_CHECK(set[i]);
    else
      BOOST_CHECK(! set[i]);
  }

  // Check that or-assignment does not throw.
  BOOST_REQUIRE_NO_THROW(set |= odd);

  // Check for correct or-assignement (all are set)
  for(std::size_t i = 0; i < set.size(); ++i)
      BOOST_CHECK(set[i]);

  BOOST_REQUIRE_NO_THROW(set ^= odd);
  // Check for correct and-assignement (evens are set)
  for(std::size_t i = 0; i < set.size(); ++i) {
    if(i % 2)
      BOOST_CHECK(set[i]);
    else
      BOOST_CHECK(! set[i]);
  }

  // Check that assignment of bitsets with different size throws.
#ifdef TA_EXCEPTION_ERROR
  Bitset bad(size / 2);
  BOOST_CHECK_THROW(bad &= set, std::range_error);
  BOOST_CHECK_THROW(bad |= set, std::range_error);
  BOOST_CHECK_THROW(bad ^= set, std::range_error);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_SUITE_END()

