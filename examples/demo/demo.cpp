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

int main(int argc, char* argv[]) {
  using namespace std;
  namespace TA = TiledArray;

  auto& world = madness::initialize(argc, argv);

  //TA::Range R0({0,0},{10,10});
  TA::Range R0{10,10};
  cout << R0 << endl;

  TA::Range R1({2,1},{7,8});
  cout << R1 << endl;

  TA::Range R2(array<int,3>{{1,2,3}},array<int,3>{{3,4,5}});
  cout << R2 << " area=" << R2.area() << endl;

  for(const auto& i: R2) {
    cout << i << " " << R2.ordinal(i) << endl;
  }

  TA::TiledRange1 TR0{0, 3, 8, 10};
  TA::TiledRange1 TR1{0, 4, 7, 10};
  TA::TiledRange TR{TR0, TR1};
  cout << TR << endl;
  for(const auto& i: TR.elements()) {
    cout << i << endl;
  }
  for(const auto& i: TR.tiles()) {
    cout << i << endl;
  }

  cout << TR.tile({0,1}) << endl;

  auto TRp = TA::Permutation{1,0} * TR;
  cout << TRp << endl;

  TA::TArray<double> a0(world, TR);
  a0.fill_local(1.0);
  //std::vector<double> vec(100,0.0); for(auto& i: vec) { i = (double)rand()/RAND_MAX; }
  //a0.set(vec.begin());
  if (world.rank() == 0)
    cout << a0 << endl;
  world.gop.fence();

  TA::Tensor<float> shape_tensor(TR.tiles());
  shape_tensor(0,0) = 1.0;
  shape_tensor(0,1) = 1.0;
  shape_tensor(1,1) = 1.0;
  shape_tensor(2,2) = 1.0;
  TA::SparseShape<float> shape(shape_tensor, TR);
  TA::TSpArray<double> a1(world, TR, shape);
  a1.fill_local(1.0);
  if (world.rank() == 0)
    cout << a1 << endl;
  world.gop.fence();

//  TA::TArray<complex<double>> a1(world, TR);
//  a1.fill_local(complex<double>{1., 2.});
//  if (world.rank() == 0)
//    cout << a1 << endl;
//  world.gop.fence();

  TA::TSpArray<double> a2; a2("i,j") = a1("i,j") * 2.0;
  if (world.rank() == 0)
    cout << a2 << endl;
  world.gop.fence();

  TA::TSpArray<double> a3; a3("j,i") = a2("i,j");
  if (world.rank() == 0)
    cout << a3 << endl;
  world.gop.fence();

  TA::TSpArray<double> a4; a4("j,i") = a3("i,j") * 0.5;
  if (world.rank() == 0)
    cout << a4 << endl;
  world.gop.fence();

  using TSpArrayD = TA::TSpArray<double>;

  TSpArrayD a5; a5("i,j") = a4("i,j") + 2.0 * a4("i,j");
  if (world.rank() == 0)
    cout << a5 << endl;
  world.gop.fence();

  TSpArrayD a6; a6("i,j") = a4("i,j") - 2.0 * a4("i,j");
  if (world.rank() == 0)
    cout << a6 << endl;
  world.gop.fence();

  TSpArrayD a7; a7("i,j") = a6("i,j") * a5("i,j");
  if (world.rank() == 0)
    cout << a7 << endl;
  world.gop.fence();

  TSpArrayD a8; a8("i,j") = a1("i,k") * a5("j,k");
  if (world.rank() == 0)
    cout << a8 << endl;
  world.gop.fence();

  return 0;
}

#endif /* EXAMPLES_DEMO_DEMO_CPP_ */
