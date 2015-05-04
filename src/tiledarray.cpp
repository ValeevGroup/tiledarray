/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  tiledarray.cpp
 *  May 2, 2015
 *
 */

#include <tiledarray.h>

#define TA_DEF_TILE( t ) TiledArray::Tensor< t , Eigen::aligned_allocator< t > >
#define TA_DEF_ARRAY( t , d , p ) TiledArray::Array< t , d , TA_DEF_TILE( t ), p >

#define TA_INST_TILE( t ) \
  template class TA_DEF_TILE( t );

#define TA_INST_ARRAY( t , p ) \
  template class TA_DEF_ARRAY( t , 1 , p ); \
  template class TA_DEF_ARRAY( t , 2 , p ); \
  template class TA_DEF_ARRAY( t , 3 , p ); \
  template class TA_DEF_ARRAY( t , 4 , p );

#define TA_INST_TYPE( t ) \
  TA_INST_TILE( t ) \
  TA_INST_ARRAY( t , TiledArray::DensePolicy ) \
  TA_INST_ARRAY( t , TiledArray::SparsePolicy )

TA_INST_TYPE( double )
TA_INST_TYPE( float )

