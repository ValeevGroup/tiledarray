/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "TiledArray/shape.h"
#include "config.h"
#include "range_fixture.h"

using TiledArray::detail::Shape;

struct ShapeFixture : public RangeFixture {

  ShapeFixture() : RangeFixture(), shape(r, 1.0e-8, 0.0) { }

  Shape<float> shape;
};

BOOST_FIXTURE_TEST_SUITE( shape_suite, ShapeFixture )

BOOST_AUTO_TEST_CASE( default_constructor )
{
  BOOST_REQUIRE_NO_THROW(Shape<float> s);

  Shape<float> s;

  // Check that the default constructed shape is dense
  BOOST_CHECK(s.is_dense());

  // Check that all tiles for a default constructed shape are non-zero
  for(Range::const_iterator it = r.begin(); it != r.end(); ++it) {
    BOOST_CHECK(! s.is_zero(*it));
    BOOST_CHECK(! s.is_zero(r.ord(*it)));
  }


#ifdef TA_EXCEPTION_ERROR
  // Check that is_zero does not throw for out of range data
  // NOTE: Default constructed ranges have now range data
  BOOST_CHECK_NO_THROW(s.is_zero(r.finish()));
  BOOST_CHECK_NO_THROW(s.is_zero(r.volume()));
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(Shape<float> s(r, 1.0e-8, 1.0));

  Shape<float> s(r, 1.0e-8, 1.0);

  // Check the tolerance
  BOOST_CHECK_CLOSE(s.threshold(), 1.0e-8, 1.0e-2);

#ifdef TA_EXCEPTION_ERROR
  // Check that data access throws before sharing
  BOOST_CHECK_THROW(s.is_zero(0), TiledArray::Exception);
  // Check that the default constructed shape is dense
  BOOST_CHECK_THROW(s.is_dense(), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR

  // Disable sharing so that the initial values can be checked
  s.no_share();

  // Check that the default constructed shape is dense
  BOOST_CHECK(s.is_dense());

  // Check that all tiles for a default constructed shape are non-zero
  for(Range::const_iterator it = r.begin(); it != r.end(); ++it) {
    BOOST_CHECK(! s.is_zero(*it));
    BOOST_CHECK(! s.is_zero(r.ord(*it)));
  }

#ifdef TA_EXCEPTION_ERROR
  // Check that is_zero throws for out of range data
  BOOST_CHECK_THROW(s.is_zero(r.finish()), TiledArray::Exception);
  BOOST_CHECK_THROW(s.is_zero(r.volume()), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( copy_constructor )
{
  BOOST_REQUIRE_NO_THROW(Shape<float> s(shape));

  Shape<float> s(shape);

  // Check that the tolerances match
  BOOST_CHECK_CLOSE(s.threshold(), shape.threshold(), 1.0e-2);

#ifdef TA_EXCEPTION_ERROR
  // Check that data access throws before sharing
  BOOST_CHECK_THROW(s.is_zero(0), TiledArray::Exception);
  // Check that the default constructed shape is dense
  BOOST_CHECK_THROW(s.is_dense(), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR

  // Disable sharing so that the initial values can be checked
  BOOST_CHECK_NO_THROW(s.no_share());

  // Check that the default constructed shape is dense
  BOOST_CHECK(! s.is_dense());

  // Check that all tiles for a default constructed shape are non-zero
  for(Range::const_iterator it = r.begin(); it != r.end(); ++it) {
    BOOST_CHECK(s.is_zero(*it));
    BOOST_CHECK(s.is_zero(r.ord(*it)));
  }
}


BOOST_AUTO_TEST_CASE( iterator_constructor )
{
  std::vector<std::pair<std::size_t, float> > data;
  for(std::size_t i = 0; i < r.volume(); ++i)
    data.push_back(std::pair<std::size_t, float>(i, GlobalFixture::world->rank() + 1));

  BOOST_REQUIRE_NO_THROW(Shape<float> s(r, 1.0e-8, data.begin(), data.end()));

  Shape<float> s(r, 1.0e-8, data.begin(), data.end());

#ifdef TA_EXCEPTION_ERROR
  // Check that data access throws before sharing
  BOOST_CHECK_THROW(s.is_zero(0), TiledArray::Exception);
  // Check that the default constructed shape is dense
  BOOST_CHECK_THROW(s.is_dense(), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR

  // Disable sharing so that the initial values can be checked
  BOOST_CHECK_NO_THROW(s.no_share());

  // Check that the default constructed shape is dense
  BOOST_CHECK(s.is_dense());

  // Check that all tiles for a default constructed shape are non-zero
  for(Range::const_iterator it = r.begin(); it != r.end(); ++it) {
    BOOST_CHECK(! s.is_zero(*it));
    BOOST_CHECK(! s.is_zero(r.ord(*it)));
  }
}

BOOST_AUTO_TEST_CASE( copy_assignment_before_share )
{
  Shape<float> s;

  BOOST_CHECK_NO_THROW(s = shape);


  // Check that the tolerances match
  BOOST_CHECK_CLOSE(s.threshold(), shape.threshold(), 1.0e-2);

#ifdef TA_EXCEPTION_ERROR
  // Check that data access throws before sharing
  BOOST_CHECK_THROW(s.is_zero(0), TiledArray::Exception);
  // Check that the default constructed shape is dense
  BOOST_CHECK_THROW(s.is_dense(), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR

  // Disable sharing so that the initial values can be checked
  BOOST_CHECK_NO_THROW(s.no_share());

  // Check that the default constructed shape is dense
  BOOST_CHECK(! s.is_dense());

  // Check that all tiles for a default constructed shape are non-zero
  for(Range::const_iterator it = r.begin(); it != r.end(); ++it) {
    BOOST_CHECK(s.is_zero(*it));
    BOOST_CHECK(s.is_zero(r.ord(*it)));
  }
}

BOOST_AUTO_TEST_CASE( share )
{
  Shape<float> s(r, 1.0e-8, (GlobalFixture::world->rank() == 0 ? 1.0 : 0.0));

#ifdef TA_EXCEPTION_ERROR
  // Check that data access throws before sharing
  BOOST_CHECK_THROW(s.is_zero(0), TiledArray::Exception);
  // Check that the default constructed shape is dense
  BOOST_CHECK_THROW(s.is_dense(), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR

  // Disable sharing so that the initial values can be checked
  BOOST_CHECK_NO_THROW(s.share(* GlobalFixture::world));

#ifdef TA_EXCEPTION_ERROR
  // Check that the second call to no_share throws an exception
  BOOST_CHECK_THROW(s.share(* GlobalFixture::world), TiledArray::Exception);

  // Check that data access throws before sharing
  BOOST_CHECK_NO_THROW(s.is_zero(0));
  // Check that the default constructed shape is dense
  BOOST_CHECK_NO_THROW(s.is_dense());
#endif // TA_EXCEPTION_ERROR

  // Check that all tiles for a default constructed shape are non-zero
  for(Range::const_iterator it = r.begin(); it != r.end(); ++it) {
    BOOST_CHECK(! s.is_zero(*it));
    BOOST_CHECK(! s.is_zero(r.ord(*it)));
  }
}

BOOST_AUTO_TEST_CASE( share_with_op )
{
  Shape<float> s(r, 1.0e-8, (GlobalFixture::world->rank() == 0 ? 1.0 : 0.0));

#ifdef TA_EXCEPTION_ERROR
  // Check that data access throws before sharing
  BOOST_CHECK_THROW(s.is_zero(0), TiledArray::Exception);
  // Check that the default constructed shape is dense
  BOOST_CHECK_THROW(s.is_dense(), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR

  // Disable sharing so that the initial values can be checked
  BOOST_CHECK_NO_THROW(s.share(* GlobalFixture::world, std::multiplies<float>()));

#ifdef TA_EXCEPTION_ERROR
  // Check that the second call to no_share throws an exception
  BOOST_CHECK_THROW(s.share(* GlobalFixture::world), TiledArray::Exception);

  // Check that data access throws before sharing
  BOOST_CHECK_NO_THROW(s.is_zero(0));
  // Check that the default constructed shape is dense
  BOOST_CHECK_NO_THROW(s.is_dense());
#endif // TA_EXCEPTION_ERROR

  // Check that all tiles for a default constructed shape are non-zero
  for(Range::const_iterator it = r.begin(); it != r.end(); ++it) {
    if(GlobalFixture::world->size() == 1) {
      BOOST_CHECK(! s.is_zero(*it));
      BOOST_CHECK(! s.is_zero(r.ord(*it)));
    } else {
      BOOST_CHECK(s.is_zero(*it));
      BOOST_CHECK(s.is_zero(r.ord(*it)));
    }
  }
}

BOOST_AUTO_TEST_CASE( copy_assignment_after_share )
{
  Shape<float> s;

  shape.share(* GlobalFixture::world);

  BOOST_CHECK_NO_THROW(s = shape);


  // Check that the tolerances match
  BOOST_CHECK_CLOSE(s.threshold(), shape.threshold(), 1.0e-2);

#ifdef TA_EXCEPTION_ERROR
  // Check that data access throws before sharing
  BOOST_CHECK_THROW(s.share(* GlobalFixture::world), TiledArray::Exception);
  BOOST_CHECK_THROW(s.set(0, 1.0), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR

  // Check that the default constructed shape is dense
  BOOST_CHECK(! s.is_dense());

  // Check that all tiles for a default constructed shape are non-zero
  for(Range::const_iterator it = r.begin(); it != r.end(); ++it) {
    BOOST_CHECK(s.is_zero(*it));
    BOOST_CHECK(s.is_zero(r.ord(*it)));
  }
}

BOOST_AUTO_TEST_CASE( tensor_assignment )
{
  Shape<float>::tensor_type tensor(r, 1.0);

  if(GlobalFixture::world->rank() == 0)
    BOOST_CHECK_NO_THROW(shape = tensor);

  BOOST_CHECK_NO_THROW(shape.share(* GlobalFixture::world));


  // Check that the default constructed shape is dense
  BOOST_CHECK(shape.is_dense());

  // Check that all tiles for a default constructed shape are non-zero
  for(Range::const_iterator it = r.begin(); it != r.end(); ++it) {
    BOOST_CHECK(! shape.is_zero(*it));
    BOOST_CHECK(! shape.is_zero(r.ord(*it)));
  }
}

BOOST_AUTO_TEST_CASE( threshold )
{

  // Check that the tolerances match
  BOOST_CHECK_CLOSE(shape.threshold(), 1.0e-8, 1.0e-2);

  shape.threshold(0.0);

  // Check that the tolerances changed to the correct value
  BOOST_CHECK_CLOSE(shape.threshold(), 0.0, 1.0e-2);
}

BOOST_AUTO_TEST_CASE( set )
{
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(shape.is_zero(r.start()), TiledArray::Exception);
  BOOST_CHECK_THROW(shape.is_zero(1), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR

  if(GlobalFixture::world->rank() == 0) {
    BOOST_CHECK_NO_THROW(shape.set(r.start(), 1.0));
    BOOST_CHECK_NO_THROW(shape.set(1, 1.0));
  }

  // Share the data
  BOOST_CHECK_NO_THROW(shape.share(* GlobalFixture::world));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(shape.set(r.start(), 1.0), TiledArray::Exception);
  BOOST_CHECK_THROW(shape.set(1, 1.0), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( is_not_dense )
{
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(shape.is_zero(r.start()), TiledArray::Exception);
  BOOST_CHECK_THROW(shape.is_zero(1), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR

  // Share the data
  BOOST_CHECK_NO_THROW(shape.share(* GlobalFixture::world));

  BOOST_CHECK(! shape.is_dense());
}

BOOST_AUTO_TEST_CASE( is_dense )
{
  if(GlobalFixture::world->rank() == 0)
    for(std::size_t i = 0ul; i < r.volume(); ++i)
      BOOST_CHECK_NO_THROW(shape.set(i, 1.0));

  // Share the data
  BOOST_CHECK_NO_THROW(shape.share(* GlobalFixture::world));

  // Check that the shape is dense when all tile estimates are greater than threshold
  BOOST_CHECK(shape.is_dense());
}

BOOST_AUTO_TEST_CASE( tensor_conversion )
{
  if(GlobalFixture::world->rank() == 0)
    for(std::size_t i = 0ul; i < r.volume(); ++i)
      BOOST_CHECK_NO_THROW(shape.set(i, 1.0));

  // Share the data
  BOOST_CHECK_NO_THROW(shape.share(* GlobalFixture::world));

  Shape<float>::tensor_type tensor = shape.tensor();

  for(Shape<float>::tensor_type::const_iterator it = tensor.begin(); it != tensor.end(); ++it)
    BOOST_CHECK_CLOSE(*it, 1.0, 1.0e-4);
}

BOOST_AUTO_TEST_CASE( range_accessor )
{
  BOOST_CHECK_EQUAL(shape.range(), r);
}

BOOST_AUTO_TEST_SUITE_END()
