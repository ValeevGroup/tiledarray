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
#include "tot_array_fixture.h"

#include <cstdio>

BOOST_FIXTURE_TEST_SUITE(tot_array_suite2, ToTArrayFixture)
//------------------------------------------------------------------------------
//                       Fill and Initialize
//------------------------------------------------------------------------------

/* fill_local is a thin wrapper around init_tiles. So as long as init_tiles
 * works and fill_local forwards its arguments correctly, fill_local should
 * work too.
 */
BOOST_AUTO_TEST_CASE_TEMPLATE(fill_local, TestParam, test_params) {
  using tensor_type = tensor_type<TestParam>;
  using inner_type = inner_type<TestParam>;
  using except_t = TiledArray::Exception;
  // Throws if PIMPL is empty
  {
    tensor_type t;
    if (m_world.nproc() == 1) {
      BOOST_CHECK_THROW(t.fill_local(inner_type{}), except_t);
    }
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto inner_rank = std::get<1>(tr_t);
    [[maybe_unused]] auto& already_set = std::get<2>(tr_t);

    // Test that it skips filled tiles
    /*{
      auto corr = already_set.clone();
      already_set.fill_local(inner_type{}, true);
      BOOST_CHECK(are_equal(corr, already_set));
    }*/

    // Test that it throws if a tile is already set
    /*{
        if(m_world.nproc() == 1)
          BOOST_CHECK_THROW(already_set.fill_local(inner_type{}), except_t);
    }*/

    // Test we can actually fill tiles
    {
      tensor_type t(m_world, tr);
      tensor_type corr(m_world, tr);
      auto start_idx = tr.tiles_range().lobound();
      auto tile = inner_rank == 1 ? inner_vector_tile<TestParam>(start_idx)
                                  : inner_matrix_tile<TestParam>(start_idx);
      for (auto idx : tr.tiles_range()) {
        if (!t.is_local(idx)) continue;
        tile_type<TestParam> outer_tile(tr.tile(idx), tile);
        corr.set(idx, outer_tile);
      }
      t.fill_local(tile);
      BOOST_CHECK(are_equal(t, corr));
    }
  }
}

// fill is a thin wrapper around fill_local, we just reuse the fill_local test
BOOST_AUTO_TEST_CASE_TEMPLATE(fill, TestParam, test_params) {
  using tensor_type = tensor_type<TestParam>;
  using inner_type = inner_type<TestParam>;
  using except_t = TiledArray::Exception;
  // Throws if PIMPL is empty
  {
    tensor_type t;
    if (m_world.nproc() == 1) {
      BOOST_CHECK_THROW(t.fill(inner_type{}), except_t);
    }
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto inner_rank = std::get<1>(tr_t);
    [[maybe_unused]] auto& already_set = std::get<2>(tr_t);

    // Test that it skips filled tiles
    /*{
      auto corr = already_set.clone();
      m_world.gop.fence();
      already_set.fill(inner_type{}, true);
      BOOST_CHECK(are_equal(corr, already_set));
    }*/

    // Test that it throws if a tile is already set
    /*{
      if(m_world.nproc() == 1)
        BOOST_CHECK_THROW(already_set.fill(inner_type{}), except_t);
    }*/

    // Test we can actually fill tiles
    {
      tensor_type t(m_world, tr);
      tensor_type corr(m_world, tr);
      auto start_idx = tr.tiles_range().lobound();
      auto tile = inner_rank == 1 ? inner_vector_tile<TestParam>(start_idx)
                                  : inner_matrix_tile<TestParam>(start_idx);
      for (auto idx : tr.tiles_range()) {
        if (!t.is_local(idx)) continue;
        tile_type<TestParam> outer_tile(tr.tile(idx), tile);
        corr.set(idx, outer_tile);
      }
      t.fill(tile);
      BOOST_CHECK(are_equal(t, corr));
    }
  }
}

// This should fail to compile if uncommented
/*BOOST_AUTO_TEST_CASE_TEMPLATE(fill_random, TestParam, test_params) {
  for(auto tr_t : run_all<TestParam>())
    std::get<2>(tr_t).fill_random();
}*/

/* The init_tiles function ultimately calls set(idx, future) on all local,
 * non-zero tiles. To test it we recreate the tensors resulting from run_all.
 */
BOOST_AUTO_TEST_CASE_TEMPLATE(init_tiles, TestParam, test_params) {
  using tensor_type = tensor_type<TestParam>;
  using except_t = TiledArray::Exception;

  // Throws if PIMPL is empty
  {
    tensor_type t;
    if (m_world.nproc() == 1) {
      auto l = [](const Range&) { return tile_type<TestParam>{}; };
      BOOST_CHECK_THROW(t.init_tiles(l), except_t);
    }
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto inner_rank = std::get<1>(tr_t);
    auto& corr = std::get<2>(tr_t);

    auto l = [this, inner_rank](const Range& range) {
      tile_type<TestParam> rv(range);
      for (auto idx : range)
        rv(idx) = inner_rank == 1 ? inner_vector_tile<TestParam>(idx)
                                  : inner_matrix_tile<TestParam>(idx);
      return rv;
    };

    // Test that it skips filled tiles
    /*{
      auto corr2 = corr.clone();
      corr.init_tiles(l, true);
      BOOST_CHECK(are_equal(corr, corr2));
    }*/

    // Test that it throws if a tile is already set
    /*{
      if(m_world.nproc() == 1)
        BOOST_CHECK_THROW(corr.init_tiles(l), except_t);
    }*/

    // Test we can actually fill tiles
    {
      tensor_type t(m_world, tr);
      t.init_tiles(l);
      BOOST_CHECK(are_equal(t, corr));
    }
  }
}

/* The init_elements function ultimately just calls init_tiles with a lambda
 * that calls the provided function/functor for each index in the tile. Thus if
 * init_tiles works, this function will work assuming it properly wraps the
 * provided lambda.
 */
BOOST_AUTO_TEST_CASE_TEMPLATE(init_elements, TestParam, test_params) {
  using tensor_type = tensor_type<TestParam>;
  using inner_type = inner_type<TestParam>;
  using except_t = TiledArray::Exception;
  using index_type = typename tensor_type::index;

  // Throws if PIMPL is empty
  {
    tensor_type t;
    auto l = [](const index_type&) { return inner_type{}; };
    if (m_world.nproc() == 1) {
      BOOST_CHECK_THROW(t.init_elements(l), except_t);
    }
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto inner_rank = std::get<1>(tr_t);
    auto& corr = std::get<2>(tr_t);

    auto l = [this, inner_rank](const index_type& idx) -> inner_type {
      if (inner_rank == 1)
        return inner_vector_tile<TestParam>(idx);
      else
        return inner_matrix_tile<TestParam>(idx);
    };

    // Test that it skips filled tiles
    /*{
      auto corr2 = corr.clone();
      corr.init_elements(l, true);
      BOOST_CHECK(are_equal(corr, corr2));
    }*/

    // Test that it throws if a tile is already set
    /*{
      if(m_world.nproc() == 1)
        BOOST_CHECK_THROW(corr.init_elements(l), except_t);
    }*/

    // Test we can actually fill tiles
    {
      tensor_type t(m_world, tr);
      t.init_elements(l);
      BOOST_CHECK(are_equal(t, corr));
    }
  }
}

//------------------------------------------------------------------------------
//                     Accessors
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(trange, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.trange(), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto& corr = std::get<2>(tr_t);
    BOOST_TEST(corr.trange() == tr);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(range, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.tiles_range(), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto& corr = std::get<2>(tr_t);
    bool are_same = corr.tiles_range() == tr.tiles_range();
    BOOST_TEST(are_same);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(elements_range, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.elements_range(), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto& corr = std::get<2>(tr_t);
    bool are_same = corr.elements_range() == tr.elements_range();
    BOOST_TEST(are_same);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(size, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.size(), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto& corr = std::get<2>(tr_t);
    BOOST_TEST(corr.size() == tr.tiles_range().volume());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(world, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.world(), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& corr = std::get<2>(tr_t);
    BOOST_TEST(&corr.world() == &m_world);
  }
}

/// TODO: Check pmap value
BOOST_AUTO_TEST_CASE_TEMPLATE(pmap, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.pmap(), TiledArray::Exception);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(shape, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.shape(), TiledArray::Exception);
  }
  using shape_type = typename tensor_type<TestParam>::shape_type;
  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto& corr = std::get<2>(tr_t);
    bool are_same = corr.shape() == shape_type(1, tr);
    BOOST_TEST(are_same);
  }
}

//------------------------------------------------------------------------------
//                            Call Operators
//------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE_TEMPLATE(call_operator, TestParam, test_params) {
  for (auto tr_t : run_all<TestParam>()) {
    auto inner_rank = std::get<1>(tr_t);
    auto& t = std::get<2>(tr_t);
    auto outer_rank = t.tiles_range().rank();
    std::string outer_idx = (outer_rank == 1 ? "i" : "i,j");
    std::string inner_idx = (inner_rank == 1 ? "k" : "k,l");

    if (m_world.nproc() == 1) {
      using except_t = TiledArray::Exception;
      // Throws if no semicolon
      BOOST_CHECK_THROW(t(outer_idx), except_t);
      // Throws if wrong outer rank
      BOOST_CHECK_THROW(t("i,j,k,l,m;" + inner_idx), except_t);
    }

    auto vars = outer_idx + ";" + inner_idx;
    auto expr = t(vars);
    BOOST_CHECK(are_equal(expr.array(), t));
    BOOST_CHECK(expr.annotation() == vars);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(const_call_operator, TestParam, test_params) {
  for (auto tr_t : run_all<TestParam>()) {
    auto inner_rank = std::get<1>(tr_t);
    const auto& t = std::get<2>(tr_t);
    auto outer_rank = t.tiles_range().rank();
    std::string outer_idx = (outer_rank == 1 ? "i" : "i,j");
    std::string inner_idx = (inner_rank == 1 ? "k" : "k,l");

    if (m_world.nproc() == 1) {
      using except_t = TiledArray::Exception;
      // Throws if no semicolon
      BOOST_CHECK_THROW(t(outer_idx), except_t);
      // Throws if wrong outer rank
      BOOST_CHECK_THROW(t("i,j,k,l,m;" + inner_idx), except_t);
    }

    auto vars = outer_idx + ";" + inner_idx;
    auto expr = t(vars);
    BOOST_CHECK(are_equal(expr.array(), t));
    BOOST_CHECK(expr.annotation() == vars);
  }
}

//------------------------------------------------------------------------------
//                          Tile Properties
//------------------------------------------------------------------------------

/* This is a thin wrapper around calling is_dense on the PIMPL. As long as the
 * PIMPL's is_dense works this function should work. The only additional piece
 * of code to test is ensuring an exception is raised when the PIMPL is not
 * initialized.
 */
BOOST_AUTO_TEST_CASE_TEMPLATE(is_dense, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.is_dense(), TiledArray::Exception);
  }

  using shape_type = typename tensor_type<TestParam>::shape_type;

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto& corr = std::get<2>(tr_t);
    BOOST_TEST(corr.is_dense() == shape_type(1, tr).is_dense());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(owner, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.owner(0), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto& corr = std::get<2>(tr_t);

    if (m_world.nproc() == 1) {
      const auto& upbound = tr.tiles_range().upbound();

      // Test throws if index is out of bounds
      BOOST_CHECK_THROW(corr.owner(upbound), TiledArray::Exception);

      // Throws if index has wrong rank
      std::vector<unsigned int> bad_idx(upbound.size() + 1, 0);
      BOOST_CHECK_THROW(corr.owner(bad_idx), TiledArray::Exception);
    }

    for (auto idx : corr.tiles_range()) {
      const auto ordinal = corr.tiles_range().ordinal(idx);
      BOOST_TEST(corr.owner(idx) == corr.pmap()->owner(ordinal));
      BOOST_TEST(corr.owner(ordinal) == corr.pmap()->owner(ordinal));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(owner_init_list, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.owner({0}), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto rank = tr.rank();
    auto& corr = std::get<2>(tr_t);

    if (m_world.nproc() == 1) {
      const auto& upbound = tr.tiles_range().upbound();
      using except_t = TiledArray::Exception;

      // Test throws if index is out of bounds
      if (rank == 1)
        BOOST_CHECK_THROW(corr.owner({upbound[0]}), except_t);
      else if (rank == 2)
        BOOST_CHECK_THROW(corr.owner({upbound[0], upbound[1]}), except_t);

      // Throws if index has wrong rank
      std::initializer_list<unsigned int> il2{0, 0, 0, 0, 0, 0};
      BOOST_CHECK_THROW(corr.owner(il2), except_t);
    }

    for (auto idx : corr.tiles_range()) {
      const auto ordinal = corr.tiles_range().ordinal(idx);
      const auto owner = corr.pmap()->owner(ordinal);
      if (rank == 1) {
        BOOST_TEST(corr.owner({idx[0]}) == owner);
      } else if (rank == 2) {
        BOOST_TEST(corr.owner({idx[0], idx[1]}) == owner);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_local, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.is_local(0), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto& corr = std::get<2>(tr_t);

    if (m_world.nproc() == 1) {
      const auto& upbound = tr.tiles_range().upbound();

      // Test throws if index is out of bounds
      BOOST_CHECK_THROW(corr.is_local(upbound), TiledArray::Exception);

      // Throws if index has wrong rank
      std::vector<unsigned int> bad_idx(upbound.size() + 1, 0);
      BOOST_CHECK_THROW(corr.is_local(bad_idx), TiledArray::Exception);
    }

    for (auto idx : corr.tiles_range()) {
      const auto ordinal = corr.tiles_range().ordinal(idx);
      BOOST_TEST(corr.is_local(idx) == corr.pmap()->is_local(ordinal));
      BOOST_TEST(corr.is_local(ordinal) == corr.pmap()->is_local(ordinal));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_local_init_list, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.is_local({0}), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto rank = tr.rank();
    auto& corr = std::get<2>(tr_t);

    if (m_world.nproc() == 1) {
      const auto& upbound = tr.tiles_range().upbound();
      using except_t = TiledArray::Exception;

      // Test throws if index is out of bounds
      if (rank == 1)
        BOOST_CHECK_THROW(corr.is_local({upbound[0]}), except_t);
      else if (rank == 2)
        BOOST_CHECK_THROW(corr.is_local({upbound[0], upbound[1]}), except_t);

      // Throws if index has wrong rank
      std::initializer_list<unsigned int> il2{0, 0, 0, 0, 0, 0};
      BOOST_CHECK_THROW(corr.is_local(il2), except_t);
    }

    for (auto idx : corr.tiles_range()) {
      const auto ordinal = corr.tiles_range().ordinal(idx);
      const auto is_local = corr.pmap()->is_local(ordinal);
      if (rank == 1) {
        BOOST_TEST(corr.is_local({idx[0]}) == is_local);
      } else if (rank == 2) {
        BOOST_TEST(corr.is_local({idx[0], idx[1]}) == is_local);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_zero, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.is_zero(0), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto& corr = std::get<2>(tr_t);

    if (m_world.nproc() == 1) {
      const auto& upbound = tr.tiles_range().upbound();

      // Test throws if index is out of bounds
      BOOST_CHECK_THROW(corr.is_zero(upbound), TiledArray::Exception);

      // Throws if index has wrong rank
      std::vector<unsigned int> bad_idx(upbound.size() + 1, 0);
      BOOST_CHECK_THROW(corr.is_zero(bad_idx), TiledArray::Exception);
    }

    for (auto idx : corr.tiles_range()) {
      const auto ordinal = corr.tiles_range().ordinal(idx);
      BOOST_TEST(corr.is_zero(idx) == corr.shape().is_zero(ordinal));
      BOOST_TEST(corr.owner(ordinal) == corr.pmap()->owner(ordinal));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_zero_init_list, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.is_zero({0}), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& tr = std::get<0>(tr_t);
    auto rank = tr.rank();
    auto& corr = std::get<2>(tr_t);

    if (m_world.nproc() == 1) {
      const auto& upbound = tr.tiles_range().upbound();
      using except_t = TiledArray::Exception;

      // Test throws if index is out of bounds
      if (rank == 1)
        BOOST_CHECK_THROW(corr.is_zero({upbound[0]}), except_t);
      else if (rank == 2)
        BOOST_CHECK_THROW(corr.is_zero({upbound[0], upbound[1]}), except_t);

      // Throws if index has wrong rank
      std::initializer_list<unsigned int> il2{0, 0, 0, 0, 0, 0};
      BOOST_CHECK_THROW(corr.is_zero(il2), except_t);
    }

    for (auto idx : corr.tiles_range()) {
      const auto ordinal = corr.tiles_range().ordinal(idx);
      const auto is_zero = corr.shape().is_zero(ordinal);
      if (rank == 1) {
        BOOST_TEST(corr.is_zero({idx[0]}) == is_zero);
      } else if (rank == 2) {
        BOOST_TEST(corr.is_zero({idx[0], idx[1]}) == is_zero);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(swap, TestParam, test_params) {
  for (auto tr_t : run_all<TestParam>()) {
    auto& corr = std::get<2>(tr_t);
    auto copy_corr = corr.clone();
    tensor_type<TestParam> t, t2;
    t.swap(corr);
    BOOST_CHECK(are_equal(t, copy_corr));
    BOOST_CHECK(are_equal(corr, t2));
  }
}

// TODO: Actually check that it makes the array replicated.
BOOST_AUTO_TEST_CASE_TEMPLATE(make_replicated, TestParam, test_params) {
  {
    tensor_type<TestParam> t;
    if (m_world.nproc() == 1)
      BOOST_CHECK_THROW(t.make_replicated(), TiledArray::Exception);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& corr = std::get<2>(tr_t);
    BOOST_CHECK_NO_THROW(corr.make_replicated());
  }
}

// TODO: Actually check that it truncates
BOOST_AUTO_TEST_CASE_TEMPLATE(truncate, TestParam, test_params) {
  for (auto tr_t : run_all<TestParam>()) {
    auto& corr = std::get<2>(tr_t);
    BOOST_CHECK_NO_THROW(corr.truncate());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_initialized, TestParam, test_params) {
  // Not initialized
  {
    tensor_type<TestParam> t1;
    BOOST_TEST(t1.is_initialized() == false);
  }

  for (auto tr_t : run_all<TestParam>()) {
    auto& corr = std::get<2>(tr_t);
    BOOST_TEST(corr.is_initialized());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(serialization, TestParam, test_params) {
  for (auto tr_t : run_all<TestParam>()) {
    auto& corr = std::get<2>(tr_t);
    char file_name[] = "tmp.XXXXXX";
    mktemp(file_name);
    {
      output_archive_type ar_out(file_name);
      corr.serialize(ar_out);
    }

    tensor_type<TestParam> t2;
    {
      input_archive_type ar_in(file_name);
      t2.serialize(ar_in);
      BOOST_TEST(are_equal(t2, corr));
    }
    std::remove(file_name);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(parallel_serialization, TestParam, test_params) {
  for (auto tr_t : run_all<TestParam>()) {
    auto& corr = std::get<2>(tr_t);
    const int nio = 1;  // use 1 rank for I/O
    char file_name[] = "tmp.XXXXXX";
    mktemp(file_name);
    {
      madness::archive::ParallelOutputArchive<> ar_out(m_world, file_name, nio);
      corr.store(ar_out);
    }

    tensor_type<TestParam> t2;
    {
      madness::archive::ParallelInputArchive<> ar_in(m_world, file_name, nio);
      t2.load(m_world, ar_in);
      BOOST_TEST(are_equal(corr, t2));
    }
    std::remove(file_name);
  }
}

/* To test printing we assume that the ToT tiles already print correctly. Then
 * we create the correct string by looping over ordinal indices and prepending
 * them to the string representation of the tile.
 */
BOOST_AUTO_TEST_CASE_TEMPLATE(printing, TestParam, test_params) {
  for (auto tr_t : run_all<TestParam>()) {
    const auto& t = std::get<2>(tr_t);
    std::stringstream corr;
    if (m_world.rank() == 0) {
      for (auto i = 0; i < t.size(); ++i) {
        if (t.is_zero(i)) continue;
        corr << i << ": " << t.find(i).get() << std::endl;
      }
    }
    std::stringstream ss;
    ss << t;
    BOOST_TEST(ss.str() == corr.str());
  }
}

BOOST_AUTO_TEST_SUITE_END()
