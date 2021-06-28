/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
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
 *  Created by Chong Peng on 7/19/18.
 *
 */

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <TiledArray/cuda/btas_um_tensor.h>
#include <tiledarray.h>

#include <iostream>

/**
 *  Test cuTT
 */

const std::size_t N = 100;
using namespace TiledArray;

int main(int argc, char* argv[]) {
  TiledArray::initialize(argc, argv);

  std::vector<int> extent{N, N};
  std::vector<int> perm{1, 0};

  TiledArray::finalize();
  return 0;
}

#endif
