/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
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
 *  Ajay Melekamburath
 *  Department of Chemistry, Virginia Tech
 */

#include <TiledArray/device/tensor.h>

#include <complex>

namespace TiledArray::detail {

// Phase 1 sanity: confirm the is_device_tile specialization fires for the
// allocator alias and propagates through Tile<>.
static_assert(is_device_tile_v<TiledArray::UMTensor<double>>,
              "UMTensor<double> must be tagged as a device tile");
static_assert(is_device_tile_v<TiledArray::UMTensor<float>>,
              "UMTensor<float> must be tagged as a device tile");
static_assert(
    is_device_tile_v<TiledArray::UMTensor<std::complex<double>>>,
    "UMTensor<std::complex<double>> must be tagged as a device tile");
static_assert(is_device_tile_v<TiledArray::Tile<TiledArray::UMTensor<double>>>,
              "Tile<UMTensor<double>> must propagate the device-tile tag");
static_assert(!is_device_tile_v<TiledArray::Tensor<double>>,
              "Plain Tensor<double> must not be tagged as a device tile");

}  // namespace TiledArray::detail

// Phase 2 instantiation probes: force the compiler to type-check the
// device-tile overloads. Real explicit instantiations land in Phase 4.
namespace {

template <typename T>
void compile_test_tier1() {
  using TA::UMTensor;
  using helper_t = TiledArray::math::GemmHelper;
  UMTensor<T> a, b, c;
  helper_t h(TiledArray::math::blas::Op::NoTrans,
             TiledArray::math::blas::Op::NoTrans, 2u, 2u, 2u);

  (void)TiledArray::clone(a);
  (void)TiledArray::scale(a, T(2));
  (void)TiledArray::scale_to(a, T(2));
  (void)TiledArray::neg(a);
  (void)TiledArray::neg_to(a);
  (void)TiledArray::add(a, b);
  (void)TiledArray::add(a, b, T(2));
  (void)TiledArray::add_to(a, b);
  (void)TiledArray::add_to(a, b, T(2));
  (void)TiledArray::subt(a, b);
  (void)TiledArray::subt(a, b, T(2));
  (void)TiledArray::subt_to(a, b);
  (void)TiledArray::subt_to(a, b, T(2));
  (void)TiledArray::dot(a, b);
  (void)TiledArray::squared_norm(a);
  (void)TiledArray::norm(a);
  (void)TiledArray::gemm(a, b, T(1), h);
  TiledArray::gemm(c, a, b, T(1), h);

  // Phase 2b: permute / shift / mult and the perm-variants.
  TiledArray::Permutation perm(std::vector<unsigned>{1, 0});
  TiledArray::BipartitePermutation bperm(perm);
  std::vector<long> shift{0, 0};
  (void)TiledArray::permute(a, perm);
  (void)TiledArray::permute(a, bperm);
  (void)TiledArray::shift(a, shift);
  (void)TiledArray::shift_to(a, shift);
  (void)TiledArray::scale(a, T(2), perm);
  (void)TiledArray::neg(a, perm);
  (void)TiledArray::add(a, b, perm);
  (void)TiledArray::add(a, b, T(2), perm);
  (void)TiledArray::subt(a, b, perm);
  (void)TiledArray::subt(a, b, T(2), perm);
  (void)TiledArray::mult(a, b);
  (void)TiledArray::mult(a, b, T(2));
  (void)TiledArray::mult(a, b, perm);
  (void)TiledArray::mult(a, b, T(2), perm);
  (void)TiledArray::mult_to(a, b);
  (void)TiledArray::mult_to(a, b, T(2));
}

[[maybe_unused]] auto instantiate_tier1_double = &compile_test_tier1<double>;
[[maybe_unused]] auto instantiate_tier1_float = &compile_test_tier1<float>;

}  // namespace

