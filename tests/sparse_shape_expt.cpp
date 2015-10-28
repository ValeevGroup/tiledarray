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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  sparse_shape.cpp
 *  Jul 18, 2013
 *
 */

#include "TiledArray/shape/sparse_shape.h"
#include "TiledArray/shape/sparse_shape_expt.h"
#include "tiledarray.h"
#include "unit_test_config.h"
#include "sparse_shape_fixture.h"

using namespace TiledArray;
namespace expt = TiledArray::experimental;

BOOST_FIXTURE_TEST_SUITE( sparse_shape_expt_suite, SparseShapeFixture )

BOOST_AUTO_TEST_CASE( default_constructor )
{
  BOOST_CHECK_NO_THROW(expt::SparseShape<float> x);
  expt::SparseShape<float> x, y;
  Permutation perm;
  math::GemmHelper gemm_helper(madness::cblas::NoTrans, madness::cblas::NoTrans,
      2u, 2u, 2u);

  BOOST_CHECK(x.empty());
  BOOST_CHECK(! x.is_dense());
  BOOST_CHECK(! x.validate(tr.tiles()));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x[0], Exception);

  BOOST_CHECK_THROW(x.perm(perm), Exception);

  BOOST_CHECK_THROW(x.scale(2.0), Exception);
  BOOST_CHECK_THROW(x.scale(2.0, perm), Exception);

  BOOST_CHECK_THROW(x.add(y), Exception);
  BOOST_CHECK_THROW(x.add(y, 2.0), Exception);
  BOOST_CHECK_THROW(x.add(y, perm), Exception);
  BOOST_CHECK_THROW(x.add(y, 2.0, perm), Exception);

  BOOST_CHECK_THROW(x.subt(y), Exception);
  BOOST_CHECK_THROW(x.subt(y, 2.0), Exception);
  BOOST_CHECK_THROW(x.subt(y, perm), Exception);
  BOOST_CHECK_THROW(x.subt(y, 2.0, perm), Exception);

  BOOST_CHECK_THROW(x.mult(y), Exception);
  BOOST_CHECK_THROW(x.mult(y, 2.0), Exception);
  BOOST_CHECK_THROW(x.mult(y, perm), Exception);
  BOOST_CHECK_THROW(x.mult(y, 2.0, perm), Exception);

  BOOST_CHECK_THROW(x.gemm(y, 2.0, gemm_helper), Exception);
  BOOST_CHECK_THROW(x.gemm(y, 2.0, gemm_helper, perm), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( non_comm_constructor )
{
  // Construct test tile norms
  Tensor<float> tile_norms = make_norm_tensor(tr, 1, 42);

  // Construct the shape
  BOOST_CHECK_NO_THROW(SparseShape<float> x(tile_norms, tr));
  SparseShape<float> x(tile_norms, tr);

  // Check that the shape has been initialized
  BOOST_CHECK(! x.empty());
  BOOST_CHECK(! x.is_dense());
  BOOST_CHECK(x.validate(tr.tiles()));

  size_type zero_tile_count = 0ul;

  for(Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i) {
    // Compute the expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = tile_norms[i] / float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    // Check that the tile has been normalized correctly
    BOOST_CHECK_CLOSE(x[i], expected, tolerance);

    // Check zero threshold
    if(x[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(x.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! x.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(x.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_SUITE_END()
