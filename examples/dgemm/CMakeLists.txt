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

# Create the blas, eigen, ta_band, ta_dense, ta_fock_build, ta_asymm_dense and 
# ta_sparse executable

# Add include directories
include_directories(${TiledArray_INCLUDE_DIRS})

# Add the blas executable
add_executable(blas EXCLUDE_FROM_ALL blas.cpp)
target_link_libraries(blas ${TiledArray_LIBRARIES})
add_dependencies(blas External)
add_dependencies(example blas)

# Add the eigen executable
add_executable(eigen EXCLUDE_FROM_ALL eigen.cpp)
target_link_libraries(eigen ${TiledArray_LIBRARIES})
add_dependencies(eigen External)
add_dependencies(example eigen)

# Add the blas executable
add_executable(ta_band EXCLUDE_FROM_ALL ta_band.cpp)
target_link_libraries(ta_band ${TiledArray_LIBRARIES})
add_dependencies(ta_band External)
add_dependencies(example ta_band)

# Add the ta_dense executable
add_executable(ta_dense EXCLUDE_FROM_ALL ta_dense.cpp)
target_link_libraries(ta_dense ${TiledArray_LIBRARIES})
add_dependencies(ta_dense External)
add_dependencies(example ta_dense)

# Add the ta_dense_nonuniform executable
add_executable(ta_dense_nonuniform EXCLUDE_FROM_ALL ta_dense_nonuniform.cpp)
target_link_libraries(ta_dense_nonuniform ${TiledArray_LIBRARIES})
add_dependencies(ta_dense_nonuniform External)
add_dependencies(example ta_dense_nonuniform)

# Add the ta_asymm_dense executable
add_executable(ta_asymm_dense EXCLUDE_FROM_ALL ta_asymm_dense.cpp)
target_link_libraries(ta_asymm_dense ${TiledArray_LIBRARIES})
add_dependencies(ta_asymm_dense External)
add_dependencies(example ta_asymm_dense)

# Add the blas executable
add_executable(ta_sparse EXCLUDE_FROM_ALL ta_sparse.cpp)
target_link_libraries(ta_sparse ${TiledArray_LIBRARIES})
add_dependencies(ta_sparse External)
add_dependencies(example ta_sparse)

if(ENABLE_ELEMENTAL)
    # Add the ta_elem_dense executable
    add_executable(ta_elem_dense EXCLUDE_FROM_ALL ta_elem_dense.cpp)
    target_link_libraries(ta_elem_dense ${TiledArray_LIBRARIES})
    add_dependencies(ta_elem_dense External)
    add_dependencies(example ta_elem_dense)
endif()
