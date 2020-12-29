#
#  This file is a part of TiledArray.
#  Copyright (C) 2019  Virginia Tech
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
#  Edward Valeev
#  Department of Chemistry, Virginia Tech
#
#  FindBLASPPHeaders.cmake
#  Dec 28, 2020
#

# check for MKL
include( CheckFunctionExists )
include(CMakePushCheckState)
cmake_push_check_state( RESET )
find_package( OpenMP QUIET ) #XXX Open LAPACKPP issue for this...
set( CMAKE_REQUIRED_LIBRARIES "${blaspp_libraries}" m )
check_function_exists( mkl_dimatcopy  BLAS_IS_MKL )
cmake_pop_check_state()

add_library(blaspp_headers INTERFACE)

if (BLAS_IS_MKL)
  foreach(_lib ${blaspp_libraries})
    if (EXISTS ${_lib} AND _lib MATCHES libmkl_)
      string(REGEX REPLACE "/lib/(|intel64/)libmkl_.*" "" _mklroot "${_lib}")
    elseif (_lib MATCHES "^-L")
      string(REGEX REPLACE "^-L" "" _mklroot "${_lib}")
      string(REGEX REPLACE "/lib(/|/intel64)(|/)" "" _mklroot "${_mklroot}")
    endif()
    if (_mklroot)
      break()
    endif(_mklroot)
  endforeach()

  target_include_directories(blaspp_headers INTERFACE "${_mklroot}/include")
endif()

install(TARGETS blaspp_headers EXPORT tiledarray COMPONENT blaspp_headers
    LIBRARY DESTINATION "${TILEDARRAY_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${TILEDARRAY_INSTALL_LIBDIR}")
