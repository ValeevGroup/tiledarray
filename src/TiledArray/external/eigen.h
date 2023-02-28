/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2019  Virginia Tech
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
 *  Edward Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  eigen.h
 *  Oct 1, 2019
 *
 */

#ifndef TILEDARRAY_EXTERNAL_EIGEN_H__INCLUDED
#define TILEDARRAY_EXTERNAL_EIGEN_H__INCLUDED

//
// Configure Eigen and include Eigen/Core, all dependencies should include this
// before including any Eigen headers
//

#include <TiledArray/config.h>
#include <madness/config.h>
#include <madness/world/archive.h>

TILEDARRAY_PRAGMA_GCC(diagnostic push)
TILEDARRAY_PRAGMA_GCC(system_header)

// If EIGEN_USE_LAPACKE_STRICT is defined, Eigen doesn't check if
// EIGEN_USE_LAPACKE is defined before setting it, leading to a warning when it
// is already set, so we unset here to avoid that warning
#if defined(EIGEN_USE_LAPACKE_STRICT) && defined(EIGEN_USE_LAPACKE)
#undef EIGEN_USE_LAPACKE
#endif

#include <Eigen/Core>

// disable warnings re: ignored attributes on template argument
// Eigen::PacketType<int, Eigen::DefaultDevice>::type
// {aka __vector(2) long long int}
TILEDARRAY_PRAGMA_GCC(diagnostic push)
TILEDARRAY_PRAGMA_GCC(diagnostic ignored "-Wignored-attributes")
#include <unsupported/Eigen/CXX11/Tensor>
TILEDARRAY_PRAGMA_GCC(diagnostic pop)

#if defined(EIGEN_USE_LAPACKE) || defined(EIGEN_USE_LAPACKE_STRICT)
#if !EIGEN_VERSION_AT_LEAST(3, 3, 7)
#error "Eigen3 < 3.3.7 with LAPACKE enabled may give wrong eigenvalue results"
#error \
    "Either turn off EIGEN_USE_LAPACKE/EIGEN_USE_LAPACKE_STRICT or use Eigen3 3.3.7"
#endif
#endif  // EIGEN_USE_LAPACKE || EIGEN_USE_LAPACKE_STRICT

TILEDARRAY_PRAGMA_GCC(diagnostic pop)

namespace madness {
namespace archive {

template <class Archive, typename Scalar, int RowsAtCompileTime,
          int ColsAtCompileTime, int Options, int MaxRowsAtCompileTime,
          int MaxColsAtCompileTime>
struct ArchiveStoreImpl<
    Archive,
    Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options,
                  MaxRowsAtCompileTime, MaxColsAtCompileTime>> {
  static inline void store(
      const Archive& ar,
      const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options,
                          MaxRowsAtCompileTime, MaxColsAtCompileTime>& t) {
    ar& t.rows() & t.cols();
    if (t.size()) ar& madness::archive::wrap(t.data(), t.size());
  }
};

template <class Archive, typename Scalar, int RowsAtCompileTime,
          int ColsAtCompileTime, int Options, int MaxRowsAtCompileTime,
          int MaxColsAtCompileTime>
struct ArchiveLoadImpl<
    Archive,
    Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options,
                  MaxRowsAtCompileTime, MaxColsAtCompileTime>> {
  static inline void load(
      const Archive& ar,
      Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options,
                    MaxRowsAtCompileTime, MaxColsAtCompileTime>& t) {
    typename Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime,
                           Options, MaxRowsAtCompileTime,
                           MaxColsAtCompileTime>::Index nrows(0),
        ncols(0);
    ar& nrows& ncols;
    t.resize(nrows, ncols);
    if (t.size()) ar& madness::archive::wrap(t.data(), t.size());
  }
};

template <class Archive, typename Scalar_, int NumIndices_, int Options_,
          typename IndexType_>
struct ArchiveStoreImpl<
    Archive, Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_>> {
  static inline void store(
      const Archive& ar,
      const Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_>& t) {
    using idx_t = typename Eigen::Tensor<Scalar_, NumIndices_, Options_,
                                         IndexType_>::Index;
    std::array<idx_t, NumIndices_> extents;
    for (int d = 0; d != NumIndices_; ++d) extents[d] = t.dimension(d);
    ar& extents;
    if (t.size()) ar& madness::archive::wrap(t.data(), t.size());
  }
};

template <class Archive, typename Scalar_, int NumIndices_, int Options_,
          typename IndexType_>
struct ArchiveLoadImpl<
    Archive, Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_>> {
  static inline void load(
      const Archive& ar,
      Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_>& t) {
    using idx_t = typename Eigen::Tensor<Scalar_, NumIndices_, Options_,
                                         IndexType_>::Index;
    std::array<idx_t, NumIndices_> extents;
    ar& extents;
    t.resize(extents);
    if (t.size()) ar& madness::archive::wrap(t.data(), t.size());
  }
};

}  // namespace archive
}  // namespace madness

#endif  // TILEDARRAY_EXTERNAL_EIGEN_H__INCLUDED
