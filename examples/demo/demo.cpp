/*
 * This file is a part of TiledArray.
 * Copyright (C) 2016  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#ifndef EXAMPLES_DEMO_DEMO_CPP_
#define EXAMPLES_DEMO_DEMO_CPP_

#include <tiledarray.h>
#include <random>

auto make_tile(const TA::Range &range) {
  // Construct a tile
  TA::TArrayD::value_type tile(range);

  // Fill tile with data
  std::fill(tile.begin(), tile.end(), 0.0);

  return tile;
}

int main(int argc, char *argv[]) {
  using namespace std;

  std::srand(2017);
  TA::World &world = TA::initialize(argc, argv);

  using namespace TA;

  // Range R0({0,0},{10,10});
  Range R0{10, 10};
  cout << R0 << endl;

  Range R1({2, 1}, {7, 8});
  cout << R1 << endl;

  Range R2(array<int, 3>{{1, 2, 3}}, array<int, 3>{{3, 4, 5}});
  cout << R2 << " area=" << R2.area() << endl;

  for (const auto &i : R2) {
    cout << i << " " << R2.ordinal(i) << endl;
  }

  TiledRange1 TR0{0, 3, 8, 10};
  TiledRange1 TR1{0, 4, 7, 10};
  TiledRange TR{TR0, TR1};
  assert((TiledRange{{0, 3, 8, 10}, {0, 4, 7, 10}} == TR));
  cout << TR << endl;
  for (const auto &i : TR.elements_range()) {
    cout << i << endl;
  }
  for (const auto &i : TR.tiles_range()) {
    cout << i << endl;
  }

  cout << TR.tile({0, 1}) << endl;

  auto TRp = Permutation{1, 0} * TR;
  cout << TRp << endl;

  TArray<double> a0(world, TR);
  a0.fill(1.0);
  if (world.rank() == 0) cout << "a0:\n" << a0 << endl;
  world.gop.fence();

  Tensor<float> shape_tensor(TR.tiles_range(), 0.0);
  shape_tensor(0, 0) = 1.0;
  shape_tensor(0, 1) = 1.0;
  shape_tensor(1, 1) = 1.0;
  shape_tensor(2, 2) = 1.0;
  SparseShape<float> shape(shape_tensor, TR);
  TSpArrayD a1(world, TR, shape);
  a1.fill_random();

  if (world.rank() == 0) cout << "a1:\n" << a1 << endl;
  world.gop.fence();

  //  TSpArrayZ a1(world, TR, shape);
  //  a1.fill_random();
  //  if (world.rank() == 0)
  //    cout << a1 << endl;
  //  world.gop.fence();

  TSpArrayD a2;
  a2("i,j") = a1("i,j") * 2.0;
  if (world.rank() == 0) cout << "a2:\n" << a2 << endl;
  world.gop.fence();

  TSpArrayD a3;
  a3("j,i") = a2("i,j");
  if (world.rank() == 0) cout << "a3:\n" << a3 << endl;
  world.gop.fence();

  TSpArrayD a4;
  a4("j,i") = a3("i,j") * 0.5;
  if (world.rank() == 0) cout << "a4:\n" << a4 << endl;
  world.gop.fence();

  TSpArrayD a5;
  a5("i,j") = a4("i,j") + 2.0 * a4("i,j");
  if (world.rank() == 0) cout << "a5:\n" << a5 << endl;
  world.gop.fence();

  TSpArrayD a6;
  a6("i,j") = a4("i,j") - 2.0 * a4("i,j");
  if (world.rank() == 0) cout << "a6:\n" << a6 << endl;
  world.gop.fence();

  TSpArrayD a7;
  a7("i,j") = a6("i,j") * a5("i,j");
  if (world.rank() == 0) cout << "a7:\n" << a7 << endl;
  world.gop.fence();

  TSpArrayD a8;
  a8("i,j") = a1("i,k") * a5("j,k");
  if (world.rank() == 0) cout << "a8:\n" << a8 << endl;
  world.gop.fence();

  auto tile_0_0 = a1.find({0, 0});
  cout << "a1: tile {0,0}: " << tile_0_0 << endl;
  world.taskq.add([](const TensorD &tile) { cout << tile << endl; },
                  a1.find({0, 1}));
  // auto tile_1_0 = a1.find({1,0});

  // Construct a replicated array.
  {
    auto replicated_pmap = std::make_shared<TA::detail::ReplicatedPmap>(
        world, TR.tiles_range().volume());
    TA::TArrayD array(world, TR, replicated_pmap);

    // Construct *all* tiles in each process
    for (TArrayD::index1_type i = 0; i < array.size(); ++i) {
      // Construct a tile using a MADNESS task.
      array.set(i,
                world.taskq.add(&make_tile, array.trange().make_tile_range(i)));
    }

    world.gop.fence();
    std::cout << array << std::endl;
  }

  return 0;
}

#endif /* EXAMPLES_DEMO_DEMO_CPP_ */
