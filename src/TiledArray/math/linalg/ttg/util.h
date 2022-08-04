/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020 Virginia Tech
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
 *  Eduard Valeyev
 *
 *  util.h
 *  Created:    26 July, 2022
 *
 */

#ifndef TILEDARRAY_MATH_LINALG_TTG_UTIL_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_TTG_UTIL_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_TTG

#include <TiledArray/error.h>
#include <TiledArray/math/linalg/forward.h>

#include <ttg.h>

#include <ttg/../../examples/matrixtile.h>
#include <ttg/../../examples/potrf/pmw.h>

namespace TiledArray::math::linalg::ttg {

namespace detail {
inline std::size_t& default_block_size_accessor() {
  static std::size_t block_size = 128;
  return block_size;
}
}  // namespace detail

inline std::size_t default_block_size() {
  return detail::default_block_size_accessor();
}

inline void set_default_block_size(std::size_t NB) {
  TA_ASSERT(NB > 0);
  detail::default_block_size_accessor() = NB;
}

template <typename Tile, typename Policy>
class MatrixDescriptor {
 public:
  using element_type = typename Tile::value_type;

  MatrixDescriptor(const DistArray<Tile, Policy>& da)
      : trange_(da.trange()), pmap_(da.pmap()), shape_(da.shape_shared()) {
    TA_ASSERT(trange_.rank() == 2);
  }

  /** Number of tiled rows **/
  auto rows(void) const { return trange_.dim(0).tile_extent(); }

  /** Number of rows in tile */
  //  int rows_in_tile(void) const {
  //    return pm->super.mb;
  //  }

  /** Number of rows in the matrix */
  auto rows_in_matrix(void) const { return trange_.dim(0).extent(); }

  /** Number of tiled columns **/
  auto cols(void) const { return trange_.dim(1).tile_extent(); }

  /** Number of columns in tile */
  //  int cols_in_tile(void) const {
  //    return pm->super.nb;
  //  }

  /** Number of columns in the matrix */
  auto cols_in_matrix(void) const { return trange_.dim(1).extent(); }

  /* The rank storing the tile at {row, col} */
  int rank_of(int row, int col) const {
    const auto ord = trange_.tiles_range().ordinal({row, col});
    return pmap_->owner(ord);
  }

  bool is_local(int row, int col) const {
    return ::ttg::default_execution_context().rank() == rank_of(row, col);
  }

 private:
  TiledRange trange_;
  std::shared_ptr<const Pmap> pmap_;
  std::shared_ptr<const typename Policy::shape_type> shape_;
};

namespace detail {

/// sends ptr to managed data as MatrixTile to a TT
template <lapack::Layout Layout, typename Tile>
struct ForwardMatrixTile : public madness::CallbackInterface {
  using index1_type = TiledRange1::index1_type;
  using tile_type = Tile;
  using element_type = typename tile_type::value_type;

  ForwardMatrixTile(::ttg::TerminalBase* term, index1_type r, index1_type c,
                    index1_type r_extent, index1_type c_extent,
                    const madness::Future<Tile>* rc_fut)
      : in(static_cast<::ttg::In<Key2, MatrixTile<element_type>>*>(term)),
        r(r),
        c(c),
        r_extent(r_extent),
        c_extent(c_extent),
        rc_fut(rc_fut) {}

  void notify() override {
    const auto& tile = rc_fut->get();
    const auto tile_range = tile.range();
    TA_ASSERT(r_extent == tile_range.dim(0).extent());
    TA_ASSERT(c_extent == tile_range.dim(1).extent());

    // currently TTG assumes col-major tiles
    if constexpr (Layout == lapack::Layout::RowMajor) {
      in->send(Key2(r, c),
               MatrixTile<element_type>(
                   r_extent, c_extent,
                   // WARNING: const-smell since all interfaces uses
                   // MatrixTile<nonconst-T>
                   const_cast<element_type*>(tile.data()), c_extent));
    } else {
      auto tile_permuted = permute(tile, Permutation{1, 0});
      in->send(Key2(r, c),
               MatrixTile<element_type>(r_extent, c_extent,
                                        tile_permuted.data_shared(), r_extent));
    }
    delete this;
  }

  ::ttg::In<Key2, MatrixTile<element_type>>* in;
  index1_type r;
  index1_type c;
  index1_type r_extent;
  index1_type c_extent;
  const madness::Future<Tile>* rc_fut;
};

}  // namespace detail

// clang-format off
/// connects a matrix represented by DistArray to the in terminals of a TT/TTG

/// @tparam Layout specifies the layout of the tiles expected by the receiving TT/TTG
/// @tparam Uplo specifies whether to write:
///   - `lapack::Uplo::General`: all data
///   - `lapack::Uplo::Lower`: lower triangle only (upper triangle of the diagonal tiles is untouched)
///   - `lapack::Uplo::Upper`: upper triangle only (lower triangle of the diagonal tiles is untouched)
/// @param A the DistArray object whose data will flow to @p tt
/// @param tt the TT/TTG object that will receive the data
// clang-format on
template <lapack::Layout Layout = lapack::Layout::ColMajor,
          lapack::Uplo Uplo = lapack::Uplo::General, typename Tile,
          typename Policy>
void flow_matrix_to_tt(const DistArray<Tile, Policy>& A, ::ttg::TTBase* tt) {
  using element_type = typename Tile::value_type;
  const auto ntiles_row = TiledArray::extent(A.trange().tiles_range().dim(0));
  const auto ntiles_col = TiledArray::extent(A.trange().tiles_range().dim(1));
  for (Range1::index1_type r = 0; r < ntiles_row; r++) {
    const auto r_extent = TiledArray::extent(A.trange().dim(0).tile(r));
    const Range1::index1_type cbegin =
        Uplo == lapack::Uplo::Upper ? std::min(r + 1, ntiles_col) : 0;
    const Range1::index1_type cfence =
        Uplo == lapack::Uplo::Lower ? std::min(r + 1, ntiles_col) : ntiles_col;
    for (Range1::index1_type c = cbegin; c < cfence; c++) {
      if (A.is_local({r, c})) {
        const auto c_extent = TiledArray::extent(A.trange().dim(1).tile(c));
        if (!A.is_zero({r, c})) {
          auto& rc_fut = A.find_local({r, c});
          const_cast<madness::Future<Tile>&>(rc_fut).register_callback(
              new detail::ForwardMatrixTile<Layout, Tile>(
                  tt->template in<0>(), r, c, r_extent, c_extent, &rc_fut));
        } else {
          static_cast<::ttg::In<Key2, MatrixTile<element_type>>*>(
              tt->template in<0>())
              ->send(
                  Key2(r, c),
                  MatrixTile<element_type>(
                      r_extent, c_extent,
                      Layout == lapack::Layout::RowMajor ? c_extent : r_extent)
                      .fill(0.));
        }
      }
    }
  }
}

// clang-format off
/// creates a TTG that will write MatrixTile's to a DistArray

/// @tparam Layout specifies the layout of the incoming tiles
/// @tparam Uplo specifies whether to write:
///   - `lapack::Uplo::General`: all data
///   - `lapack::Uplo::Lower`: lower triangle only (upper triangle of the diagonal tiles is zeroed out)
///   - `lapack::Uplo::Upper`: upper triangle only (lower triangle of the diagonal tiles is zeroed out)
/// @tparam Tile a tile type
/// @tparam Policy a policy type
/// @param A the DistArray object that will contain the data
/// @param result the Edge that connects the resulting TTG to the source
/// @param defer_write
/// @return a unique_ptr to the TTG object that will write MatrixTile's to @p A
// clang-format on
template <lapack::Layout Layout = lapack::Layout::ColMajor,
          lapack::Uplo Uplo = lapack::Uplo::General, typename Tile,
          typename Policy>
auto make_writer_ttg(
    DistArray<Tile, Policy>& A,
    ::ttg::Edge<Key2, MatrixTile<typename Tile::value_type>>& result,
    bool defer_write) {
  using T = typename Tile::value_type;
  auto keymap2 = [pmap = A.pmap_shared(),
                  range = A.trange().tiles_range()](const Key2& key) {
    const auto IJ = range.ordinal({key.I, key.J});
    return pmap->owner(IJ);
  };

  auto f = [&A](const Key2& key, MatrixTile<T>&& tile, std::tuple<>& out) {
    TA_ASSERT(
        tile.lda() ==
        (Layout == lapack::Layout::ColMajor
             ? tile.rows()
             : tile.cols()));  // the code below only works if tile's LD == rows
    const int I = key.I;
    const int J = key.J;
    auto rng = A.trange().make_tile_range({I, J});
    if constexpr (Uplo != lapack::Uplo::General) {
      if (I != J) {                                   // zero tile
        if constexpr (Uplo == lapack::Uplo::Lower) {  // zero out upper
          TA_ASSERT(I > J);
          A.set({J, I}, Tile(A.trange().make_tile_range({J, I}), 0.0));
        }
        if constexpr (Uplo == lapack::Uplo::Upper) {  // zero out lower
          TA_ASSERT(I < J);
          A.set({I, J}, Tile(rng, 0.0));
        }
      }
    }

    // incoming data is moved if RowMajor, else need to permute
    auto tile_IJ = Layout == lapack::Layout::ColMajor
                       ? permute(Tile(A.trange().make_tile_range({J, I}), 1,
                                      std::move(std::move(tile).yield_data())),
                                 Permutation{1, 0})
                       : Tile(rng, 1, std::move(std::move(tile).yield_data()));
    // zero out the lower/upper triangle of the diagonal
    // tiles
    if constexpr (Uplo != lapack::Uplo::General) {
      if (I == J) {
        const auto lo = rng.lobound_data()[0];
        const auto up = rng.upbound_data()[0];
        if constexpr (Uplo == lapack::Uplo::Lower) {  // zero out upper
          Range1::index1_type ij = 1;
          auto* tile_IJ_data = tile_IJ.data();
          for (auto i = lo; i != up; ++i, ij += (1 + i - lo)) {
            for (auto j = i + 1; j != up; ++j, ++ij) {
              tile_IJ_data[ij] = 0.0;
            }
          }
        }
        if constexpr (Uplo == lapack::Uplo::Upper) {  // zero out lower
          Range1::index1_type ij = 0;
          auto* tile_IJ_data = tile_IJ.data();
          for (auto i = lo; i != up; ++i) {
            for (auto j = lo; j != i; ++j, ++ij) {
              tile_IJ_data[ij] = 0.0;
            }
            ij += up - i;
          }
        }
      }
    }
    A.set({I, J}, std::move(tile_IJ));
    if (::ttg::tracing()) ::ttg::print("WRITE2TA(", key, ")");
  };

  auto result_tt = ::ttg::make_tt(f, ::ttg::edges(result), ::ttg::edges(),
                                  "Final Output", {"result"}, {});
  result_tt->set_keymap(keymap2);
  result_tt->set_defer_writer(defer_write);

  auto ins = std::make_tuple(result_tt->template in<0>());
  std::vector<std::unique_ptr<::ttg::TTBase>> ops(1);
  ops[0] = std::move(result_tt);

  return make_ttg(std::move(ops), ins, std::make_tuple(), "Result Writer");
}

}  // namespace TiledArray::math::linalg::ttg

#endif  // TILEDARRAY_HAS_TTG

#endif  // TILEDARRAY_MATH_LINALG_TTG_UTIL_H__INCLUDED
