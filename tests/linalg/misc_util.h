#pragma once
#include <tiledarray.h>

template <typename Array, typename ReplicatedDiag>
void subtract_diagonal_tensor_inplace(Array& A, const ReplicatedDiag& D) {

  TiledArray::foreach_inplace( A, [=](auto& tile) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
      for (auto n = lo[1]; n < up[1]; ++n)
        if (m == n) { tile(m, n) -= D[m]; }
  });

}

template <typename Array>
void subtract_identity_inplace(Array& A) {
  using element_type = typename Array::element_type;
  const auto M = A.trange().dim(0).extent();
  const auto N = A.trange().dim(1).extent();
  BOOST_CHECK(M == N);
  std::vector<element_type> D(N,1.0);
  subtract_diagonal_tensor_inplace(A, D);
}

template <typename Array, typename ReplicatedDiag>
void multiply_tensor_by_diag_inplace(char SIDE, Array& A, const ReplicatedDiag& D) {

  TiledArray::foreach_inplace( A, [=](auto& tile) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    // A(i,j) = D(i,i) * A(i,j)
    if(SIDE == 'L') {
      for (auto m = lo[0]; m < up[0]; ++m) {
        const auto d = D[m];
        for (auto n = lo[1]; n < up[1]; ++n) {
          tile(m, n) *= d;
        }
      }
    // A(i,j) = A(i,j) * D(j,j)
    } else {
      for (auto m = lo[0]; m < up[0]; ++m)
        for (auto n = lo[1]; n < up[1]; ++n) {
          tile(m, n) *= D[n];
        }
    }
  });

}
