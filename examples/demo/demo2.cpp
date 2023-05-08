/*
 * This file is a part of TiledArray.
 * Copyright (C) 2023  Virginia Tech
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

#ifndef EXAMPLES_DEMO_DEMO2_CPP_
#define EXAMPLES_DEMO_DEMO2_CPP_

#include <tiledarray.h>
#include <random>

#include <TiledArray/expressions/einsum.h>
#include <TiledArray/math/solvers/cp.h>

int main(int argc, char* argv[]) {
  using namespace std;

  TA::srand(2017);
  TA::World& world = TA::initialize(argc, argv);

  using namespace TA;

  // requires compiler new enough to support unicode characters in variable
  // names
#ifndef TILEDARRAY_CXX_COMPILER_SUPPORTS_UNICODE_VARIABLES
#ifdef TILEDARRAY_CXX_COMPILER_IS_GCC
#if __GNUC__ >= 10
#define TILEDARRAY_CXX_COMPILER_SUPPORTS_UNICODE_VARIABLES 1
#endif
#elif !defined(TILEDARRAY_CXX_COMPILER_IS_ICC)
#define TILEDARRAY_CXX_COMPILER_SUPPORTS_UNICODE_VARIABLES 1
#endif
#endif  // !defined(TILEDARRAY_CXX_COMPILER_SUPPORT_UNICODE_VARIABLES)

#ifdef TILEDARRAY_CXX_COMPILER_SUPPORTS_UNICODE_VARIABLES

  // $\rho \equiv \mathbb{Z}_{1,11} \otimes \mathbb{Z}_{-1,9}$
  Range ρ{{1, 11}, {-1, 9}};
  // lower bound of $\mathbb{Z}_{1,11} \otimes \mathbb{Z}_{-1,9}$
  assert((ρ.lobound() == Index{1, -1}));
  // upper bound of $\mathbb{Z}_{1,11} \otimes \mathbb{Z}_{-1,9}$
  assert((ρ.upbound() == Index{11, 9}));
  // extent of $\mathbb{Z}_{1,11} \otimes \mathbb{Z}_{-1,9}$
  assert((ρ.extent() == Index{10, 10}));
  // 1st dimension of ρ is $\mathbb{Z}_{1,11}$
  assert((ρ.dim(0) == Range1{1, 11}));
  // 2nd dimension of ρ is $\mathbb{Z}_{-1,9}$
  assert((ρ.dim(1) == Range1{-1, 9}));
  // the number of elements in $\mathbb{Z}_{1,11} \otimes \mathbb{Z}_{-1,9}$
  assert(ρ.volume() == 100);
  // row-major order
  assert((ρ.stride() == Index{10, 1}));
  assert((ρ.ordinal({1, -1}) == 0));
  assert((ρ.ordinal({1, 0}) == 1));
  assert((ρ.ordinal({10, 8}) + 1 == ρ.volume()));
  // prints "[1,-1] [1,0] .. [1,8] [2,-1] .. [10,8] "
  for (auto&& idx : ρ) cout << idx << " ";

  // $\mathbb{Z}_{1,11}$ tiled into $\mathbb{Z}_{1,5}$, $\mathbb{Z}_{5,8}$, and
  // $\mathbb{Z}_{8,11}$
  TiledRange1 τ0{1, 5, 8, 11};  // hashmarks
  assert(τ0.extent() == 10);    // there are 10 elements in τ0
  assert((τ0.elements_range() ==
          Range1{1, 11}));        // elements indexed by $\mathbb{Z}_{1,11}$
  assert(τ0.tile_extent() == 3);  // there are 3 tiles in τ0
  assert((τ0.tiles_range() ==
          Range1{0, 3}));                // tiles indexed by $\mathbb{Z}_{0,3}$
  assert((τ0.tile(1) == Range1{5, 8}));  // 1st tile of τ0 is $\mathbb{Z}_{5,8}$

  // $\mathbb{Z}_{-1,9}$ tiled into $\mathbb{Z}_{-1,5}$ and $\mathbb{Z}_{5,9}$
  TiledRange1 τ1{-1, 5, 9};

  // 2nd tile of $\code{tau0}$ is $\mathbb{Z}_{5,8}$
  assert((τ0.tile(1) == Range1{5, 8}));
  // 1st tile of $\code{tau1}$ is $\mathbb{Z}_{-1,5}$
  assert((τ1.tile(0) == Range1{-1, 5}));

  // prints "-1 0 1 2 3 4 "
  for (auto&& i : τ1.tile(0)) cout << i << " ";

  // tiling of $\mathbb{Z}_{1,11} \otimes \mathbb{Z}_{-1,9}$ by tensor product
  // of
  // $\code{τ0}$ and $\code{τ1}$
  TiledRange τ{τ0, τ1};
  // shortcut
  TiledRange same_as_τ{{1, 5, 8, 11}, {-1, 5, 9}};

  // tile index {0,0} refers to tile $\mathbb{Z}_{1,5} \otimes
  // \mathbb{Z}_{-1,5}$
  auto tile_0_0 = τ.tile({0, 0});
  assert((tile_0_0 == Range{{1, 5}, {-1, 5}}));

  // clang-format off

  // 2-d array of $\code{double}$ 0s, indexed by ρ
  Tensor<double> t0(ρ, 0.);
  // same as $\code{t0}$ but filled with ordinals
  TensorD t1(ρ, [&ρ](auto&& idx) {
    return ρ.ordinal(idx);
  });
  // print out "0 1 .. 99 "
  for (auto&& v : t1) cout << v << " ";
  // same as $\code{t0}$, using existing buffer
  shared_ptr<double[]> v(new double[ρ.volume()]);
  TensorD t2(ρ, v); // t2 and v co-manage buffer lifetime
  v[0] = 1.;
  assert(t2(1, -1) == 1.);
  // same as $\code{t0}$, using existing (unmanaged) buffer
  auto t3 = make_map(v.get(), ρ);
  v[0] = 2.;
  assert(t3(1, -1) == 2.);
  // Tensor has shallow-copy semantics
  auto t4 = t0;
  t0(1, -1) = 3.;
  assert(t4(1, -1) == 3.);

  // clang-format on

  // default instance of $\code{DistArray}$ is
  // a {\em dense} array of $\code{double}$s
  // NB can use TArrayD instead of DistArray<>
  DistArray<> a0(τ);
  a0.fill(1.);  // fill $\code{da}$ with 1s
  // every tile exists in a dense array
  assert(!a0.is_zero({0, 0}));
  // grab a ${\em future}$ to the {0,0} tile
  auto t00 = a0.find({0, 0});

  // shape of a {\em sparse} array over τ
  // tiles with even ordinals ({0,0}, {0,2}, {1,1}) are zero
  SparseShape s(TensorF(τ.tiles_range(), {0, 1, 0, 1, 0, 1}), τ);
  // a sparse array of $\code{double}$s
  // TSpArrayX $\equiv$ DistArray<TensorX, SparsePolicy>
  TSpArrayD a1(τ, s);
  // only some tiles are nonzero in sparse array
  assert(a1.is_zero({0, 0}));
  assert(!a1.is_zero({0, 1}));

#endif  // defined(TILEDARRAY_CXX_COMPILER_SUPPORT_UNICODE_VARIABLES)

  return 0;
}

#endif /* EXAMPLES_DEMO_DEMO2_CPP_ */
