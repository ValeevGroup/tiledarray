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
  enable_language(C)
    
  # Add the mpi_bandwidth executable
  add_ta_executable(mpi_bandwidth "mpi_bandwidth.c" "${MADNESS_LIBRARIES}")
  add_dependencies(examples-tiledarray mpi_bandwidth)

  # Add the mpi_latency executable
  add_ta_executable(mpi_latency "mpi_latency.c" "${MADNESS_LIBRARIES}")
  add_dependencies(examples-tiledarray mpi_latency)


  if(CUDA_FOUND)
    add_ta_executable(mpi_cuda "mpi_cuda.cpp" "${MADNESS_LIBRARIES}")
    set_source_files_properties(mpi_cuda.cpp
            PROPERTIES
            LANGUAGE CUDA)
    add_dependencies(examples-tiledarray mpi_cuda)
  endif(CUDA_FOUND)

endif()
