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
 *  proc_grid.cpp
 *  Nov 13, 2013
 *
 */

// Enable the testing constructor
#define TILEDARRAY_ENABLE_TEST_PROC_GRID

#include "TiledArray/proc_grid.h"
#include "unit_test_config.h"
#include "TiledArray/sparse_shape.h"

struct ProcGridFixture {

  ProcGridFixture() { }

  ~ProcGridFixture() { }

}; // Fixture

BOOST_FIXTURE_TEST_SUITE( proc_grid_suite, ProcGridFixture )

BOOST_AUTO_TEST_CASE( random_constructor_test )
{
  GlobalFixture::world->srand(time(NULL));

  for(int test = 0; test < 100; ++test) {

    // Generate random process and matrix sizes
    const ProcessID nprocs = GlobalFixture::world->rand() % 4095 + 1;
    const std::size_t rows = GlobalFixture::world->rand() % 1023 + 1;
    const std::size_t cols = GlobalFixture::world->rand() % 1023 + 1;
    const std::size_t size = rows * cols;
    const std::size_t row_size = rows * ((GlobalFixture::world->rand() % 511) + 1);
    const std::size_t col_size = cols * ((GlobalFixture::world->rand() % 512) + 1);

    TiledArray::detail::ProcGrid proc_grid0(*GlobalFixture::world, 0, nprocs,
        rows, cols, row_size, col_size);

    // Check tile dimensions and sizes
    BOOST_CHECK_EQUAL(proc_grid0.rows(), rows);
    BOOST_CHECK_EQUAL(proc_grid0.cols(), cols);
    BOOST_CHECK_EQUAL(proc_grid0.size(), size);

    // Check process grid sizes
    BOOST_CHECK_LE(proc_grid0.proc_size(), nprocs);
    BOOST_CHECK_EQUAL(proc_grid0.proc_size(), proc_grid0.proc_rows() * proc_grid0.proc_cols());

    // Check that process rows are within limits
    BOOST_CHECK_GE(proc_grid0.proc_rows(), 1ul);
    BOOST_CHECK_LE(proc_grid0.proc_rows(), rows);
    BOOST_CHECK_LE(proc_grid0.proc_rows(), nprocs);

    // Check that process columns are within limits
    BOOST_CHECK_GE(proc_grid0.proc_cols(), 1ul);
    BOOST_CHECK_LE(proc_grid0.proc_cols(), cols);
    BOOST_CHECK_LE(proc_grid0.proc_cols(), nprocs);

    // Check process grid rank
    BOOST_CHECK_EQUAL(proc_grid0.rank_row(), 0);
    BOOST_CHECK_EQUAL(proc_grid0.rank_col(), 0);

    // Accumulate the number of local elements on each node
    std::size_t local_rows = proc_grid0.local_rows();
    std::size_t local_cols = proc_grid0.local_cols();
    std::size_t local_size = proc_grid0.local_size();

    // Check that the process grids on other ranks match what was obtained on
    // rank 0.
    ProcessID rank = 0;
    for(ProcessID rank_row = 0; rank_row < proc_grid0.proc_rows(); ++rank_row) {
      for(ProcessID rank_col = 0; rank_col < proc_grid0.proc_cols(); ++rank_col, ++rank) {

        // Skip rank 0
        if(rank == 0)
          continue;

        // Construct the process grid
        TiledArray::detail::ProcGrid proc_grid(*GlobalFixture::world, rank, nprocs,
            rows, cols, row_size, col_size);

        // Check tile dimensions and sizes are equal to that of rank 0
        BOOST_CHECK_EQUAL(proc_grid.rows(), proc_grid0.rows());
        BOOST_CHECK_EQUAL(proc_grid.cols(), proc_grid.cols());
        BOOST_CHECK_EQUAL(proc_grid.size(), proc_grid.size());

        // Check process grid dimensions are equal for all ranks
        BOOST_CHECK_EQUAL(proc_grid0.proc_rows(), proc_grid0.proc_rows());
        BOOST_CHECK_EQUAL(proc_grid0.proc_cols(), proc_grid0.proc_cols());
        BOOST_CHECK_EQUAL(proc_grid0.proc_size(), proc_grid0.proc_size());

        // Check process grid rank
        BOOST_CHECK_EQUAL(proc_grid.rank_row(), rank_row);
        BOOST_CHECK_EQUAL(proc_grid.rank_col(), rank_col);

        // Accumulate the number of local elements on each node
        local_rows += proc_grid.local_rows();
        local_cols += proc_grid.local_cols();
        local_size += proc_grid.local_size();
      }
    }

    // Check that the processes not included in the process grid have
    // appropriate values.
    for(; rank < nprocs; ++rank) {
      TiledArray::detail::ProcGrid proc_grid(*GlobalFixture::world, rank, nprocs,
          rows, cols, row_size, col_size);

      // Check tile dimensions and sizes are equal to that of rank 0
      BOOST_CHECK_EQUAL(proc_grid.rows(), proc_grid0.rows());
      BOOST_CHECK_EQUAL(proc_grid.cols(), proc_grid.cols());
      BOOST_CHECK_EQUAL(proc_grid.size(), proc_grid.size());

      // Check process grid dimensions are equal for all ranks
      BOOST_CHECK_EQUAL(proc_grid.proc_rows(), proc_grid0.proc_rows());
      BOOST_CHECK_EQUAL(proc_grid.proc_cols(), proc_grid0.proc_cols());
      BOOST_CHECK_EQUAL(proc_grid.proc_size(), proc_grid0.proc_size());

      // Check process grid rank
      BOOST_CHECK_EQUAL(proc_grid.rank_row(), -1);
      BOOST_CHECK_EQUAL(proc_grid.rank_col(), -1);

      // Accumulate the number of local elements on each node
      BOOST_CHECK_EQUAL(proc_grid.local_rows(), 0ul);
      BOOST_CHECK_EQUAL(proc_grid.local_cols(), 0ul);
      BOOST_CHECK_EQUAL(proc_grid.local_size(), 0ul);
    }

    // Check that the sum of the local rows and sizes matches the total
    BOOST_CHECK_EQUAL(local_rows / proc_grid0.proc_cols(), rows);
    BOOST_CHECK_EQUAL(local_cols / proc_grid0.proc_rows(), cols);
    BOOST_CHECK_EQUAL(local_size, size);

  }
}

BOOST_AUTO_TEST_CASE( make_groups )
{
  madness::DistributedID did(madness::uniqueidT(), 0);

  // Construct the process grid
  TiledArray::detail::ProcGrid proc_grid(*GlobalFixture::world, 42, 84,
      2048, 1024);

  // Create the row and column group
  madness::Group row_group, col_group;
  BOOST_REQUIRE_NO_THROW(row_group = proc_grid.make_row_group(did));
  BOOST_REQUIRE_NO_THROW(col_group = proc_grid.make_col_group(did));

  // Check group sizes
  BOOST_CHECK_EQUAL(row_group.size(), proc_grid.proc_cols());
  BOOST_CHECK_EQUAL(col_group.size(), proc_grid.proc_rows());

  // Check that the groups contain the correct processes.
  ProcessID rank = 0;
  for(ProcessID rank_row = 0; rank_row < proc_grid.proc_rows(); ++rank_row) {
    for(ProcessID rank_col = 0; rank_col < proc_grid.proc_cols(); ++rank_col, ++rank) {
      // Check that the row group includes ranks in this this processes
      if(rank_row == proc_grid.rank_row()) {
        BOOST_CHECK_NE(row_group.rank(rank), -1);
      } else {
        BOOST_CHECK_EQUAL(row_group.rank(rank), -1);
      }

      // Check that the column group includes the correct ranks
      if(rank_col == proc_grid.rank_col()) {
        BOOST_CHECK_NE(col_group.rank(rank), -1);
      } else {
        BOOST_CHECK_EQUAL(col_group.rank(rank), -1);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( make_sparse_groups )
{
  madness::DistributedID did(madness::uniqueidT(), 0);

  // Construct the process grid
  TiledArray::detail::ProcGrid proc_grid(*GlobalFixture::world, 42, 84,
      2048, 1024);

  // Create two shape with approximately 1% non-zero tiles
  const std::size_t K = 20ul;
  std::array<std::size_t, 2> left_size = {{ proc_grid.rows(), K }};
  TiledArray::Tensor<float> left_shape_data(TiledArray::Range(left_size), 0.0f);
  std::size_t end = left_shape_data.size() / 100ul;
  for(std::size_t n = 0ul; n < end; ++n)
    left_shape_data[GlobalFixture::world->rand() % left_shape_data.size()] = 1.0f;
  TiledArray::SparseShape left_shape(left_shape_data, 0.5f);


  std::array<std::size_t, 2> right_size = {{ K, proc_grid.cols() }};
  TiledArray::Tensor<float> right_shape_data(TiledArray::Range(right_size), 0.0f);
  end = right_shape_data.size() / 100ul;
  for(std::size_t n = 0ul; n < end; ++n)
    right_shape_data[GlobalFixture::world->rand() % right_shape_data.size()] = 1.0f;
  TiledArray::SparseShape right_shape(right_shape_data, 0.5f);

  // Check each inner index
  for(std::size_t k = 0ul; k < K; ++k) {
    {
      // Create the row and column groups
      madness::Group row_group;
      BOOST_REQUIRE_NO_THROW(row_group = proc_grid.make_row_group(did, right_shape, k, right_shape.data().size()));

      std::size_t n_row_procs = 0ul;
      for(std::size_t col_start = 0ul; col_start < proc_grid.proc_cols(); ++col_start) {
        // Compute the rank that will be tested for non-zero tiles
        ProcessID world_rank = proc_grid.rank_row() * proc_grid.proc_cols() + col_start;

        // Count the number of non-zero tiles on world_rank.
        std::size_t n_tiles = 0ul;
        for(std::size_t i[2] = { k, col_start}; i[1] < proc_grid.cols(); i[1] += proc_grid.proc_cols())
          n_tiles += (right_shape.is_zero(i) ? 0ul : 1ul);

        if(n_tiles > 0ul) {
          ++n_row_procs;
          BOOST_CHECK_NE(row_group.rank(world_rank), -1);
        }
      }

      BOOST_CHECK_EQUAL(row_group.empty(), (n_row_procs == 0ul));
    }

    {
      // Create the row and column groups
      madness::Group col_group;
      BOOST_REQUIRE_NO_THROW(col_group = proc_grid.make_col_group(did, left_shape, k, left_shape.data().size()));

      std::size_t n_col_procs = 0ul;
      for(std::size_t row_start = 0ul; row_start < proc_grid.proc_rows(); ++row_start) {
        // Compute the rank that will be tested for non-zero tiles
        ProcessID world_rank = row_start * proc_grid.proc_cols() + proc_grid.rank_col();

        // Count the number of non-zero tiles on world_rank.
        std::size_t n_tiles = 0ul;
        for(std::size_t i[2] = { row_start, k }; i[0] < proc_grid.rows(); i[0] += proc_grid.proc_rows()) {
          n_tiles += (left_shape.is_zero(i) ? 0ul : 1ul);
        }

        // Check that the rank is in the group if it has a non-zero tile.
        if(n_tiles > 0ul) {
          ++n_col_procs;
          BOOST_CHECK_NE(col_group.rank(world_rank), -1);
        }
      }

      BOOST_CHECK_EQUAL(col_group.empty(), (n_col_procs == 0ul));
    }

  }
}

#if 0
BOOST_AUTO_TEST_CASE( statistics )
{
  GlobalFixture::world->srand(time(NULL));

  double total_time = 0.0;
  std::size_t count = 0;
  std::vector<std::size_t> unused_process_distribution(100, 0ul);

  while(count < 100000) {

    const ProcessID nprocs = (GlobalFixture::world->rand() % 16 + 1);
    const std::size_t rows = GlobalFixture::world->rand() % 1023 + 1;
    const std::size_t cols = GlobalFixture::world->rand() % 1023 + 1;
    const std::size_t row_size = rows * ((GlobalFixture::world->rand() % 511) + 1);
    const std::size_t col_size = cols * ((GlobalFixture::world->rand() % 512) + 1);

    if((rows * cols) < nprocs)
      continue;

    const double start = madness::wall_time();
    TiledArray::detail::ProcGrid proc_grid(*GlobalFixture::world, 0, nprocs,
        rows, cols, row_size, col_size);
    total_time += madness::wall_time() - start;
    ++count;

    ++unused_process_distribution[100ul * (nprocs - proc_grid.proc_size()) / nprocs];

  }

  for(std::vector<std::size_t>::const_iterator it = unused_process_distribution.begin();
      it != unused_process_distribution.end(); ++it)
    std::cout << *it << "\n";
  std::cout << "\nAverage build time: " << total_time / double(count) << "\n";
}
#endif

BOOST_AUTO_TEST_SUITE_END()
