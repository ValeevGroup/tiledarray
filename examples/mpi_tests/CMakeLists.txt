#
#  This file is a part of TiledArray.
#  Copyright (C) 2013  Virginia Tech
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  Justus Calvin
#  Department of Chemistry, Virginia Tech
#
#  CMakeLists.txt
#  Sep 4, 2013
#

# Create the mpi_bandwidth and mpi_latency executable

if(NOT DISABLE_MPI)
  # Add the mpi_bandwidth executable
  add_executable(mpi_bandwidth EXCLUDE_FROM_ALL mpi_bandwidth.c)
  target_link_libraries(mpi_bandwidth ${MPI_C_LIBRARIES})
  add_dependencies(example mpi_bandwidth)

  # Add the mpi_latency executable
  add_executable(mpi_latency EXCLUDE_FROM_ALL mpi_latency.c)
  target_link_libraries(mpi_latency ${MPI_C_LIBRARIES})
  add_dependencies(example mpi_latency)
endif()