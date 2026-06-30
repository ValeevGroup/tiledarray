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
#include "tiledarray.h"
#include "unit_test_config.h"

struct ProcGridFixture {
  ProcGridFixture() {}

  ~ProcGridFixture() {}

};  // Fixture

BOOST_FIXTURE_TEST_SUITE(proc_grid_suite, ProcGridFixture)

BOOST_AUTO_TEST_CASE(random_constructor_test) {
  GlobalFixture::world->srand(time(NULL));

  for (int test = 0; test < 100; ++test) {
    // Generate random process and matrix sizes
    const ProcessID nprocs = GlobalFixture::world->rand() % 4095 + 1;
    const std::size_t rows = GlobalFixture::world->rand() % 1023 + 1;
    const std::size_t cols = GlobalFixture::world->rand() % 1023 + 1;
    const std::size_t size = rows * cols;
    const std::size_t row_size =
        rows * ((GlobalFixture::world->rand() % 511) + 1);
    const std::size_t col_size =
        cols * ((GlobalFixture::world->rand() % 512) + 1);

    TiledArray::detail::ProcGrid proc_grid0(*GlobalFixture::world, 0, nprocs,
                                            rows, cols, row_size, col_size);

    // Check tile dimensions and sizes
    BOOST_CHECK_EQUAL(proc_grid0.rows(), rows);
    BOOST_CHECK_EQUAL(proc_grid0.cols(), cols);
    BOOST_CHECK_EQUAL(proc_grid0.size(), size);

    // Check process grid sizes
    BOOST_CHECK_LE(proc_grid0.proc_size(), nprocs);
    BOOST_CHECK_EQUAL(proc_grid0.proc_size(),
                      proc_grid0.proc_rows() * proc_grid0.proc_cols());

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
    for (std::size_t rank_row = 0; rank_row < proc_grid0.proc_rows();
         ++rank_row) {
      for (std::size_t rank_col = 0; rank_col < proc_grid0.proc_cols();
           ++rank_col, ++rank) {
        // Skip rank 0
        if (rank == 0) continue;

        // Construct the process grid
        TiledArray::detail::ProcGrid proc_grid(*GlobalFixture::world, rank,
                                               nprocs, rows, cols, row_size,
                                               col_size);

        // Check tile dimensions and sizes are equal to that of rank 0
        BOOST_CHECK_EQUAL(proc_grid.rows(), proc_grid0.rows());
        BOOST_CHECK_EQUAL(proc_grid.cols(), proc_grid.cols());
        BOOST_CHECK_EQUAL(proc_grid.size(), proc_grid.size());

        // Check process grid dimensions are equal for all ranks
        BOOST_CHECK_EQUAL(proc_grid0.proc_rows(), proc_grid0.proc_rows());
        BOOST_CHECK_EQUAL(proc_grid0.proc_cols(), proc_grid0.proc_cols());
        BOOST_CHECK_EQUAL(proc_grid0.proc_size(), proc_grid0.proc_size());

        // Check process grid rank
        BOOST_CHECK_EQUAL(proc_grid.rank_row(), ProcessID(rank_row));
        BOOST_CHECK_EQUAL(proc_grid.rank_col(), ProcessID(rank_col));

        // Accumulate the number of local elements on each node
        local_rows += proc_grid.local_rows();
        local_cols += proc_grid.local_cols();
        local_size += proc_grid.local_size();
      }
    }

    // Check that the processes not included in the process grid have
    // appropriate values.
    for (; rank < nprocs; ++rank) {
      TiledArray::detail::ProcGrid proc_grid(
          *GlobalFixture::world, rank, nprocs, rows, cols, row_size, col_size);

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

BOOST_AUTO_TEST_CASE(make_groups) {
  madness::DistributedID did_row(madness::uniqueidT(), 0);
  madness::DistributedID did_col(madness::uniqueidT(), 1);

  // Construct the process grid
  TiledArray::detail::ProcGrid proc_grid(*GlobalFixture::world, 42, 84, 2048,
                                         1024);

  // Create the row and column group
  madness::Group row_group, col_group;
  BOOST_REQUIRE_NO_THROW(row_group = proc_grid.make_row_group(did_row));
  BOOST_REQUIRE_NO_THROW(col_group = proc_grid.make_col_group(did_col));

  // Check group sizes
  BOOST_CHECK_EQUAL(row_group.size(), proc_grid.proc_cols());
  BOOST_CHECK_EQUAL(col_group.size(), proc_grid.proc_rows());

  // Check that the groups contain the correct processes.
  std::size_t rank = 0;
  for (std::size_t rank_row = 0; rank_row < proc_grid.proc_rows(); ++rank_row) {
    for (std::size_t rank_col = 0; rank_col < proc_grid.proc_cols();
         ++rank_col, ++rank) {
      // Check that the row group includes ranks in this this processes
      if (ProcessID(rank_row) == proc_grid.rank_row()) {
        BOOST_CHECK_NE(row_group.rank(rank), -1);
      } else {
        BOOST_CHECK_EQUAL(row_group.rank(rank), -1);
      }

      // Check that the column group includes the correct ranks
      if (ProcessID(rank_col) == proc_grid.rank_col()) {
        BOOST_CHECK_NE(col_group.rank(rank), -1);
      } else {
        BOOST_CHECK_EQUAL(col_group.rank(rank), -1);
      }
    }
  }
}

// Regression test for the SUMMA broadcast-root computation
// (Summa::get_row_group_root / get_col_group_root in
// dist_eval/contraction_eval.h).
//
// When the right (resp. left) operand is block-sparse the SUMMA row (resp.
// column) broadcast group is pruned to the ranks that own nonzero source
// tiles, so its size drops below proc_cols (resp. proc_rows). In that case the
// broadcast root's group-local rank no longer equals its process-column
// (-row) index and must be looked up from its *world* rank. make_group always
// inserts the geometric root into the group via ProcGrid::map_col / map_row,
// whose returned world rank includes the grid's rank_offset_ -- the world rank
// of the grid's first process. rank_offset_ is nonzero for the h-grouped 3-d
// batched SUMMA grid (every slab group beyond group 0). The old root code
// reconstructed the world rank by hand as rank_row*proc_cols + group_root
// (resp. group_root*proc_cols + rank_col), *omitting rank_offset_*, so for an
// offset grid Group::rank() failed to find the root and returned -1, tripping
// MADNESS_ASSERT(group_root >= 0) in WorldGopInterface::bcast and aborting the
// process.
//
// The fix recomputes the root's world rank with ProcGrid::map_col / map_row,
// which is exactly the value make_group inserts. This test validates that
// primitive: on an offset grid map_col / map_row must include the offset, so
// the broadcast root the fix now uses is the value present in the group, while
// the old offset-less formula would have missed it (differing by exactly the
// nonzero offset). A full end-to-end SUMMA repro requires >= 4 ranks (proc_h_
// > 1 needs P >= 2*proc_rows*proc_cols and a pruned group needs proc_cols or
// proc_rows >= 2), beyond this suite's np<=2 runs.
BOOST_AUTO_TEST_CASE(summa_bcast_root_offset) {
  // Baseline (np-independent): on an unoffset grid the offset-aware maps agree
  // with the bare geometric formula the old root code used. The test ctor
  // permits an arbitrary (test_rank, test_nprocs) regardless of world size and
  // leaves rank_offset_ == 0.
  {
    TiledArray::detail::ProcGrid grid(*GlobalFixture::world, /*test_rank=*/5u,
                                      /*test_nprocs=*/12u, /*rows=*/8,
                                      /*cols=*/8,
                                      /*row_size=*/64, /*col_size=*/64);
    BOOST_REQUIRE_GT(grid.proc_cols(), 0u);
    BOOST_REQUIRE_GT(grid.proc_rows(), 0u);
    for (std::size_t c = 0; c < grid.proc_cols(); ++c)
      BOOST_CHECK_EQUAL(
          grid.map_col(c),
          grid.rank_row() * ProcessID(grid.proc_cols()) + ProcessID(c));
    for (std::size_t r = 0; r < grid.proc_rows(); ++r)
      BOOST_CHECK_EQUAL(
          grid.map_row(r),
          grid.rank_col() + ProcessID(r) * ProcessID(grid.proc_cols()));
  }

  // Offset grid: this is the regression. Construct a process grid whose first
  // process is at a nonzero world rank (as the h-grouped batched SUMMA does
  // for slab groups beyond group 0) and verify the maps the fix now uses
  // include the offset -- so the broadcast root is the in-group world rank,
  // not the offset-less one the old code computed (which would be -1 once the
  // group is pruned). Requires the world to be large enough to host an offset
  // sub-grid (rank_subset_t ctor asserts rank_offset + nprocs <= world.size()).
  const auto world_size = GlobalFixture::world->size();
  if (world_size >= 2) {
    const std::size_t nprocs = world_size - (world_size / 2);  // upper ~half
    const ProcessID offset = world_size - ProcessID(nprocs);   // nonzero offset
    BOOST_REQUIRE_GT(offset, 0);
    TiledArray::detail::ProcGrid grid(*GlobalFixture::world,
                                      TiledArray::detail::rank_subset, offset,
                                      nprocs, /*rows=*/8,
                                      /*cols=*/8, /*row_size=*/64,
                                      /*col_size=*/64);
    // Only ranks inside the sub-grid have a valid (rank_row, rank_col).
    if (GlobalFixture::world->rank() >= offset) {
      for (std::size_t c = 0; c < grid.proc_cols(); ++c) {
        const ProcessID offset_less =
            grid.rank_row() * ProcessID(grid.proc_cols()) + ProcessID(c);
        // map_col == the value make_group inserts as the row-broadcast root
        BOOST_CHECK_EQUAL(grid.map_col(c), offset_less + offset);
        // ... which differs from the old offset-less root computation
        BOOST_CHECK_NE(grid.map_col(c), offset_less);
      }
      for (std::size_t r = 0; r < grid.proc_rows(); ++r) {
        const ProcessID offset_less =
            grid.rank_col() + ProcessID(r) * ProcessID(grid.proc_cols());
        // map_row == the value make_group inserts as the col-broadcast root
        BOOST_CHECK_EQUAL(grid.map_row(r), offset_less + offset);
        BOOST_CHECK_NE(grid.map_row(r), offset_less);
      }
    }
  }
}

#if 0
// This test case us used to evaluate distribute statistics. This unit test
// should only be enabled when changes are made to the ProcGrid algorithm, and
// human eyes are examining the output.

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

#if 0
// This test case us used to evaluate quality of ProcGrid output. This unit test
// should only be enabled when changes are made to the ProcGrid algorithm, and
// human eyes are examining the output.

BOOST_AUTO_TEST_CASE( generate )
{
  const std::size_t tile_size = 256;
  const std::size_t row_size = 32768;
  const std::size_t col_size = 32768;
  const std::size_t rows = row_size / tile_size;
  const std::size_t cols = col_size / tile_size;

  std::cout << "\n Matrix size = " << row_size << "x" << col_size << "\n blocksize = "
      << tile_size << "x" << tile_size << "\n";

  for(ProcessID nprocs = 1; nprocs <= 16; ++nprocs) {

    TiledArray::detail::ProcGrid proc_grid(*GlobalFixture::world, 0, nprocs,
        rows, cols, row_size, col_size);

    std::cout << "P = " << nprocs << "  P_rows = " << proc_grid.proc_rows()
        << "  P_cols = " << proc_grid.proc_cols() << "\n";
  }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
