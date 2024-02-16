/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 */

#ifndef TILEDARRAY_DIST_EVAL_CONTRACTION_EVAL_H__INCLUDED
#define TILEDARRAY_DIST_EVAL_CONTRACTION_EVAL_H__INCLUDED

#include <vector>

#include <TiledArray/config.h>
#include <TiledArray/dist_eval/dist_eval.h>
#include <TiledArray/proc_grid.h>
#include <TiledArray/reduce_task.h>
#include <TiledArray/shape.h>
#include <TiledArray/type_traits.h>

#include <TiledArray/tensor/type_traits.h>

// #define TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL 1
// #define TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE 1
// #define TILEDARRAY_ENABLE_SUMMA_TRACE_STEP 1
// #define TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST 1
// #define TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE 1

namespace TiledArray {
namespace detail {

/// \brief Distributed contraction evaluator implementation

/// \tparam Left The left-hand argument evaluator type
/// \tparam Right The right-hand argument evaluator type
/// \tparam Op The contraction/reduction operation type
/// \tparam Policy The tensor policy class
/// \note The algorithms in this class assume that the arguments have a two-
/// dimensional cyclic distribution, and that the row phase of the left-hand
/// argument and the column phase of the right-hand argument are equal to
/// the number of rows and columns, respectively, in the \c ProcGrid object
/// passed to the constructor.
template <typename Left, typename Right, typename Op, typename Policy>
class Summa
    : public DistEvalImpl<typename Op::result_type, Policy>,
      public std::enable_shared_from_this<Summa<Left, Right, Op, Policy>> {
 public:
  typedef Summa<Left, Right, Op, Policy> Summa_;  ///< This object type
  typedef DistEvalImpl<typename Op::result_type, Policy>
      DistEvalImpl_;  ///< The base class type
  typedef typename DistEvalImpl_::TensorImpl_
      TensorImpl_;           ///< The base, base class type
  typedef Left left_type;    ///< The left-hand argument type
  typedef Right right_type;  ///< The right-hand argument type
  typedef typename DistEvalImpl_::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename DistEvalImpl_::range_type range_type;      ///< Range type
  typedef typename DistEvalImpl_::shape_type shape_type;      ///< Shape type
  typedef typename DistEvalImpl_::pmap_interface
      pmap_interface;  ///< Process map interface type
  typedef
      typename DistEvalImpl_::trange_type trange_type;    ///< Tiled range type
  typedef typename DistEvalImpl_::value_type value_type;  ///< Tile type
  typedef
      typename DistEvalImpl_::eval_type eval_type;  ///< Tile evaluation type
  typedef Op op_type;  ///< Tile evaluation operator type

 private:
  static ordinal_type max_memory_;  ///< Maximum memory used per node
  static ordinal_type
      max_depth_;  ///< Maximum number of concurrent SUMMA iterations

  // Arguments and operation
  left_type left_;    ///< The left-hand argument
  right_type right_;  /// < The right-hand argument
  op_type op_;  /// < The operation used to evaluate tile-tile contractions

  // Broadcast groups for dense arguments (empty for non-dense arguments)
  madness::Group row_group_;  ///< The row process group for this rank
  madness::Group col_group_;  ///< The column process group for this rank

  // Dimension information
  const ordinal_type k_;      ///< Number of tiles in the inner dimension
  const ProcGrid proc_grid_;  ///< Process grid for this contraction

  // Contraction results
  ReducePairTask<op_type>* reduce_tasks_;  ///< A pointer to the reduction tasks

  // Constants used to iterate over columns and rows of left_ and right_,
  // respectively.
  const ordinal_type
      left_start_local_;  ///< The starting point of left column iterator ranges
                          ///< (just add k for specific columns)
  const ordinal_type left_end_;  ///< The end of the left column iterator ranges
  const ordinal_type left_stride_;  ///< Stride for left column iterators
  const ordinal_type
      left_stride_local_;            ///< Stride for local left column iterators
  const ordinal_type right_stride_;  ///< Stride for right row iterators
  const ordinal_type
      right_stride_local_;  ///< stride for local right row iterators

  typedef Future<typename right_type::eval_type>
      right_future;  ///< Future to a right-hand argument tile
  typedef Future<typename left_type::eval_type>
      left_future;  ///< Future to a left-hand argument tile
  typedef std::pair<ordinal_type, right_future>
      row_datum;  ///< Datum element type for a right-hand argument row
  typedef std::pair<ordinal_type, left_future>
      col_datum;  ///< Datum element type for a left-hand argument column

  // various tracing/debugging artifacts
  static constexpr const bool trace_tasks =
#ifdef TILEDARRAY_ENABLE_TASK_DEBUG_TRACE
      true
#else
      false
#endif
      ;
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
  mutable std::atomic<ordinal_type>
      left_ntiles_used_;  // # of tiles used from left_
  mutable std::atomic<ordinal_type>
      right_ntiles_used_;  // # of tiles used from right_
  mutable std::atomic<ordinal_type>
      left_ntiles_discarded_;  // # of tiles discarded from left_
  mutable std::atomic<ordinal_type>
      right_ntiles_discarded_;  // # of tiles discarded from right_
#endif

 protected:
  // Import base class functions
  using std::enable_shared_from_this<Summa_>::shared_from_this;

 private:
  // Static variable initialization ----------------------------------------

  /// Initialize max_memory_ limit for SUMMA
  static ordinal_type init_max_memory() {
    const char* max_memory = getenv("TA_SUMMA_MAX_MEMORY");
    if (max_memory) {
      // Convert the string into bytes
      std::stringstream ss(max_memory);
      double memory = 0.0;
      if (ss >> memory) {
        if (memory > 0.0) {
          std::string unit;
          if (ss >> unit) {  // Failure == assume bytes
            if (unit == "KB" || unit == "kB") {
              memory *= 1000.0;
            } else if (unit == "KiB" || unit == "kiB") {
              memory *= 1024.0;
            } else if (unit == "MB") {
              memory *= 1000000.0;
            } else if (unit == "MiB") {
              memory *= 1048576.0;
            } else if (unit == "GB") {
              memory *= 1000000000.0;
            } else if (unit == "GiB") {
              memory *= 1073741824.0;
            }
          }
        }
      }

      memory = std::max(memory, 104857600.0);  // Minimum 100 MiB
      return memory;
    }

    return 0ul;
  }

  static ordinal_type init_max_depth() {
    const char* max_depth = getenv("TA_SUMMA_MAX_DEPTH");
    if (max_depth) return std::stoul(max_depth);
    return 0ul;
  }

  // Process groups --------------------------------------------------------

  /// Process group factory function

  /// This function generates a sparse process group.
  /// \tparam Shape The shape type
  /// \tparam ProcMap The process map operation type
  /// \param shape The shape that will be used to select processes that are
  /// included in the process group
  /// \param process_mask the process mask, if
  ///        \code process_mask[p] == true \endcode,
  ///        process \c p will not be included in the result (p is row/col index
  ///        in this process's row/column)
  /// \param index The first index of the row or column range
  /// \param end The end of the row or column range
  /// \param stride The row or column index stride
  /// \param k The broadcast group index
  /// \param max_group_size The maximum number of processes in the result
  /// group, which is equal to the number of process in this process row or
  /// column as defined by \c proc_grid_.
  /// \param key_offset The key that will be used to identify the process group
  /// \param proc_map The operator that will convert a process row/column
  /// index into the absolute process index (ProcessID)
  /// \return A sparse process group that includes process in the row or
  /// column of this process as defined by \c proc_grid_.
  template <typename Shape, typename ProcMap>
  madness::Group make_group(const Shape& shape,
                            const std::vector<bool>& process_mask,
                            ordinal_type index, const ordinal_type end,
                            const ordinal_type stride,
                            const ordinal_type max_group_size,
                            const ordinal_type k, const ordinal_type key_offset,
                            const ProcMap& proc_map) const {
    // Generate the list of processes in rank_row
    std::vector<ProcessID> proc_list(max_group_size, -1);

    // Flag the root processes of the broadcast, which may not be included
    // by shape.
    ordinal_type p = k % max_group_size;
    proc_list[p] = proc_map(p);
    ordinal_type count = 1ul;

    // Flag all processes that have non-zero tiles
    for (p = 0ul; (index < end) && (count < max_group_size);
         index += stride, p = (p + 1u) % max_group_size) {
      if ((proc_list[p] != -1) || (shape.is_zero(index)) || !process_mask.at(p))
        continue;

      proc_list[p] = proc_map(p);
      ++count;
    }

    // Remove processes from the list that will not be in the group
    for (ordinal_type x = 0ul, p = 0ul; x < count; ++p) {
      if (proc_list[p] == -1) continue;
      proc_list[x++] = proc_list[p];
    }

    // Truncate invalid process id's
    proc_list.resize(count);

    return madness::Group(
        TensorImpl_::world(), proc_list,
        madness::DistributedID(DistEvalImpl_::id(), k + key_offset));
  }

  /// Row process group factory function

  /// \param k The broadcast group index
  /// \return A row process group
  madness::Group make_row_group(const ordinal_type k) const {
    // Construct the sparse broadcast group
    const ordinal_type right_begin_k = k * proc_grid_.cols();
    const ordinal_type right_end_k = right_begin_k + proc_grid_.cols();
    // make the row mask; using the same mask for all tiles avoids having to
    // compute mask for every tile and use of masked broadcasts
    auto result_row_mask_k = make_row_mask(k);

    // return empty group if I am not in this group, otherwise make a group
    if (result_row_mask_k[proc_grid_.rank_col()])
      return make_group(right_.shape(), result_row_mask_k, right_begin_k,
                        right_end_k, right_stride_, proc_grid_.proc_cols(), k,
                        k_, [&](const ProcGrid::size_type col) {
                          return proc_grid_.map_col(col);
                        });
    else
      return madness::Group();
  }

  /// Column process group factory function

  /// \param k The broadcast group index
  /// \return A column process group
  madness::Group make_col_group(const ordinal_type k) const {
    // make the column mask; using the same mask for all tiles avoids having to
    // compute mask for every tile and use of masked broadcasts
    auto result_col_mask_k = make_col_mask(k);

    // return empty group if I am not in this group, otherwise make a group
    if (result_col_mask_k[proc_grid_.rank_row()])
      return make_group(
          left_.shape(), result_col_mask_k, k, left_end_, left_stride_,
          proc_grid_.proc_rows(), k, 0ul,
          [&](const ordinal_type row) { return proc_grid_.map_row(row); });
    else
      return madness::Group();
  }

  /// Makes the row result mask

  /// \param k The SUMMA iteration (i.e. contraction tile) index
  /// \return a set object, if \code result[p] == true \endcode the process
  ///         in column \c p of this row has at least 1 result tile for this \c
  ///         k
  std::vector<bool> make_row_mask(const ordinal_type k) const {
    // "local" A[i][k] (i.e. for all i assigned to my row of processes) will
    // produce C[i][*] for each process in my row of the process grid determine
    // whether there are any nonzero C[i][*] located on that node

    const auto nproc_cols = proc_grid_.proc_cols();
    const auto my_proc_row = proc_grid_.rank_row();

    // result shape
    const auto& result_shape = TensorImpl_::shape();

    // if result is dense, include all processors
    if (result_shape.is_dense()) return std::vector<bool>(nproc_cols, true);

    // initialize the mask
    std::vector<bool> mask(nproc_cols, false);

    // number of tiles in the col dimension of the result
    const auto nj = proc_grid_.cols();
    // number of tiles in contraction dim
    const auto nk = k_;

    // for each i assigned to my row of processes ...
    ordinal_type i_start, i_fence, i_stride;
    std::tie(i_start, i_fence, i_stride) = result_row_range(my_proc_row);
    const auto ik_stride = i_stride * nk;
    for (ordinal_type i = i_start, ik = i_start * nk + k; i < i_fence;
         i += i_stride, ik += ik_stride) {
      // ... such that A[i][k] exists ...
      if (!left_.shape().is_zero(ik)) {
        // ... the owner of А[i][k] is always in the group ...
        const auto k_proc_col = k % nproc_cols;
        mask[k_proc_col] = true;
        // ... loop over processes in my row ...
        for (ordinal_type proc_col = 0; proc_col != nproc_cols; ++proc_col) {
          // ... that are not the owner of A[i][k] ...
          if (proc_col != k_proc_col) {
            // ... loop over all C[i][j] tiles that belong to this process ...
            ordinal_type j_start, j_fence, j_stride;
            std::tie(j_start, j_fence, j_stride) = result_col_range(proc_col);
            const auto ij_stride = j_stride;
            for (ordinal_type j = j_start, ij = i * nj + j_start; j < j_fence;
                 j += j_stride, ij += ij_stride) {
              // ... if any such C[i][j] exists, update the mask, and move
              // on to next process
              if (!result_shape.is_zero(
                      DistEvalImpl_::perm_index_to_target(ij))) {
                mask[proc_col] = true;
                break;
              }
            }
          }
        }
      }
    }

    return mask;
  }

  /// Makes the column result mask

  /// \param k The SUMMA iteration (i.e. contraction tile) index
  /// \return a set object, if \code result[p] == true \endcode the process
  ///         in row \c p of this column has at least 1 result tile for this
  ///         \c k
  std::vector<bool> make_col_mask(const ordinal_type k) const {
    // "local" B[k][j] (i.e. for all j assigned to my column of processes)
    // will produce C[*][j]
    // for each process in my column of the process grid determine whether
    // there are any
    // nonzero C[*][j] located on that node

    const auto nproc_rows = proc_grid_.proc_rows();
    const auto my_proc_col = proc_grid_.rank_col();

    // result shape
    const auto& result_shape = TensorImpl_::shape();

    // if result is dense, include all processors
    if (result_shape.is_dense()) return std::vector<bool>(nproc_rows, true);

    // initialize the mask
    std::vector<bool> mask(nproc_rows, false);

    // number of tiles in col dim of the result
    const auto nj = proc_grid_.cols();

    // for each j assigned to my column of processes ...
    ordinal_type j_start, j_fence, j_stride;
    std::tie(j_start, j_fence, j_stride) = result_col_range(my_proc_col);
    const auto kj_stride = j_stride;
    for (ordinal_type j = j_start, kj = k * nj + j_start; j < j_fence;
         j += j_stride, kj += kj_stride) {
      // ... such that B[k][j] exists ...
      if (!right_.shape().is_zero(kj)) {
        // ... the owner of B[k][j] is always in the group ...
        auto k_proc_row = k % nproc_rows;
        mask[k_proc_row] = true;
        // ... loop over processes in my col ...
        for (ordinal_type proc_row = 0; proc_row != nproc_rows; ++proc_row) {
          // ... that are not the owner of B[k][j] ...
          if (proc_row != k_proc_row) {
            // ... loop over all C[i][j] tiles that belong to this process
            ordinal_type i_start, i_fence, i_stride;
            std::tie(i_start, i_fence, i_stride) = result_row_range(proc_row);
            const auto ij_stride = i_stride * nj;
            for (ordinal_type i = i_start, ij = i_start * nj + j; i < i_fence;
                 i += i_stride, ij += ij_stride) {
              // ... if any such C[i][j] exists, update the mask, and move
              // on to next process
              if (!result_shape.is_zero(
                      DistEvalImpl_::perm_index_to_target(ij))) {
                mask[proc_row] = true;
                break;
              }
            }
          }
        }
      }
    }

    return mask;
  }

  /// computes the result row iteration range for a particular processor

  /// \param proc_row the process row in \c this->proc_grid_
  /// \return the {start,fence,stride} tuple which defines the iteration
  ///         range for the row indices of the result tiles residing on
  ///         process in row \c proc_row
  inline std::tuple<ordinal_type, ordinal_type, ordinal_type> result_row_range(
      ordinal_type proc_row) const {
    const ordinal_type start = proc_row;
    const ordinal_type fence = proc_grid_.rows();
    const ordinal_type stride = proc_grid_.proc_rows();
    return std::make_tuple(start, fence, stride);
  }

  /// computes the result column iteration range for a particular processor

  /// \param proc_col the process column in \c this->proc_grid_
  /// \return the {start,fence,stride} tuple which defines the iteration
  ///         range for the column indices of the result tiles residing on
  ///         process in column \c proc_col
  std::tuple<ordinal_type, ordinal_type, ordinal_type> result_col_range(
      ordinal_type proc_col) const {
    const ordinal_type start = proc_col;
    const ordinal_type fence = proc_grid_.cols();
    const ordinal_type stride = proc_grid_.proc_cols();
    return std::make_tuple(start, fence, stride);
  }

  // Broadcast kernels -----------------------------------------------------

  /// Tile conversion task function

  /// \tparam Tile The input tile type
  /// \param tile The input tile
  /// \return The evaluated version of the lazy tile
  template <typename Tile>
  static auto convert_tile(const Tile& tile) {
    TiledArray::Cast<typename eval_trait<Tile>::type, Tile> cast;
    return cast(tile);
  }

  /// Conversion function

  /// This function does nothing since tile is not a lazy tile.
  /// \tparam Arg The type of the argument that holds the input tiles
  /// \param arg The argument that holds the tiles
  /// \param index The tile index of arg
  /// \return \c tile
  template <typename Arg>
  static typename std::enable_if<!is_lazy_tile<typename Arg::value_type>::value,
                                 Future<typename Arg::eval_type>>::type
  get_tile(Arg& arg, const typename Arg::ordinal_type index) {
    return arg.get(index);
  }

  /// Conversion function

  /// This function spawns a task that will convert a lazy tile from the
  /// tile type to the evaluated tile type.
  /// \tparam Arg The type of the argument that holds the input tiles
  /// \param arg The argument that holds the tiles
  /// \param index The tile index of arg
  /// \return A future to the evaluated tile
  template <typename Arg>
  static typename std::enable_if<
      is_lazy_tile<typename Arg::value_type>::value
#ifdef TILEDARRAY_HAS_DEVICE
          && !detail::is_device_tile_v<typename Arg::value_type>
#endif
      ,
      Future<typename Arg::eval_type>>::type
  get_tile(Arg& arg, const typename Arg::ordinal_type index) {
    auto convert_tile_fn =
        &Summa_::template convert_tile<typename Arg::value_type>;
    return arg.world().taskq.add(convert_tile_fn, arg.get(index),
                                 madness::TaskAttributes::hipri());
  }

#ifdef TILEDARRAY_HAS_DEVICE
  /// Conversion function

  /// This function spawns a task that will convert a lazy tile from the
  /// tile type to the evaluated tile type.
  /// \tparam Arg The type of the argument that holds the input tiles
  /// \param arg The argument that holds the tiles
  /// \param index The tile index of arg
  /// \return A future to the evaluated tile
  template <typename Arg>
  static typename std::enable_if<
      is_lazy_tile<typename Arg::value_type>::value &&
          detail::is_device_tile_v<typename Arg::value_type>,
      Future<typename Arg::eval_type>>::type
  get_tile(Arg& arg, const typename Arg::ordinal_type index) {
    auto convert_tile_fn =
        &Summa_::template convert_tile<typename Arg::value_type>;
    return madness::add_device_task(arg.world(), convert_tile_fn,
                                    arg.get(index),
                                    madness::TaskAttributes::hipri());
  }
#endif

  /// Collect non-zero tiles from \c arg

  /// \tparam Arg The argument type
  /// \tparam Datum The vector datum type
  /// \param[in] arg The owner of the input tiles
  /// \param[in] index The index of the first tile to be broadcast
  /// \param[in] end The end of the range of tiles to be broadcast
  /// \param[in] stride The stride between tile indices to be broadcast
  /// \param[out] vec The vector that will hold broadcast tiles
  template <typename Arg, typename Datum>
  void get_vector(Arg& arg, ordinal_type index, const ordinal_type end,
                  const ordinal_type stride, std::vector<Datum>& vec) const {
    TA_ASSERT(vec.size() == 0ul);

    // Iterate over vector of tiles
    if (arg.is_local(index)) {
      for (ordinal_type i = 0ul; index < end; ++i, index += stride) {
        if (arg.shape().is_zero(index)) continue;
        vec.emplace_back(i, get_tile(arg, index));
      }
    } else {
      for (ordinal_type i = 0ul; index < end; ++i, index += stride) {
        if (arg.shape().is_zero(index)) continue;
        vec.emplace_back(i, Future<typename Arg::eval_type>());
      }
    }

    TA_ASSERT(vec.size() > 0ul);
  }

  /// Collect non-zero tiles from column \c k of \c left_

  /// \param[in] k The column to be retrieved
  /// \param[out] col The column vector that will hold the tiles
  void get_col(const ordinal_type k, std::vector<col_datum>& col) const {
    col.reserve(proc_grid_.local_rows());
    get_vector(left_, left_start_local_ + k, left_end_, left_stride_local_,
               col);
  }

  /// Collect non-zero tiles from row \c k of \c right_

  /// \param[in] k The row to be retrieved
  /// \param[out] row The row vector that will hold the tiles
  void get_row(const ordinal_type k, std::vector<row_datum>& row) const {
    row.reserve(proc_grid_.local_cols());

    // Compute local iteration limits for row k of right_.
    ordinal_type begin = k * proc_grid_.cols();
    const ordinal_type end = begin + proc_grid_.cols();
    begin += proc_grid_.rank_col();

    get_vector(right_, begin, end, right_stride_local_, row);
  }

  /// Broadcast tiles from \c arg

  /// \param[in] start The index of the first tile to be broadcast
  /// \param[in] stride The stride between tile indices to be broadcast
  /// \param[in] group The process group where the tiles will be broadcast
  /// \param[in] group_root The root process of the broadcast
  /// \param[in] key_offset The broadcast key offset value
  /// \param[out] vec The vector that will hold broadcast tiles
  template <typename Datum>
  void bcast(const ordinal_type start, const ordinal_type stride,
             const madness::Group& group, const ProcessID group_root,
             const ordinal_type key_offset, std::vector<Datum>& vec) const {
    TA_ASSERT(vec.size() != 0ul);
    TA_ASSERT(group.size() > 0);
    TA_ASSERT(group_root < group.size());

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST
    std::stringstream ss;
    ss << "bcast: rank=" << TensorImpl_::world().rank()
       << " root=" << group.world_rank(group_root) << " groupid=("
       << group.id().first << "," << group.id().second
       << ") keyoffset=" << key_offset << " group={ ";
    for (ProcessID group_proc = 0; group_proc < group.size(); ++group_proc)
      ss << group.world_rank(group_proc) << " ";
    ss << "} tiles={ ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST

    // Iterate over tiles to be broadcast
    for (typename std::vector<Datum>::iterator it = vec.begin();
         it != vec.end(); ++it) {
      const ordinal_type index = it->first * stride + start;

      // Broadcast the tile
      const madness::DistributedID key(DistEvalImpl_::id(), index + key_offset);
      TensorImpl_::world().gop.bcast(key, it->second, group_root, group);

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST
      ss << index << " ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST
    }

    TA_ASSERT(vec.size() > 0ul);

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST
    ss << "}\n";
    printf(ss.str().c_str());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_BCAST
  }

  // Broadcast specialization for left and right arguments -----------------

  ProcessID get_row_group_root(const ordinal_type k,
                               const madness::Group& row_group) const {
    ProcessID group_root = k % proc_grid_.proc_cols();
    if (!right_.shape().is_dense() &&
        row_group.size() < static_cast<ProcessID>(proc_grid_.proc_cols())) {
      const ProcessID world_root =
          proc_grid_.rank_row() * proc_grid_.proc_cols() + group_root;
      group_root = row_group.rank(world_root);
    }
    return group_root;
  }

  ProcessID get_col_group_root(const ordinal_type k,
                               const madness::Group& col_group) const {
    ProcessID group_root = k % proc_grid_.proc_rows();
    if (!left_.shape().is_dense() &&
        col_group.size() < static_cast<ProcessID>(proc_grid_.proc_rows())) {
      const ProcessID world_root =
          group_root * proc_grid_.proc_cols() + proc_grid_.rank_col();
      group_root = col_group.rank(world_root);
    }
    return group_root;
  }

  /// Broadcast column \c k of \c left_ with a dense right-hand argument

  /// \param[in] k The column of \c left_ to be broadcast
  /// \param[out] col The vector that will hold the results of the broadcast
  void bcast_col(const ordinal_type k, std::vector<col_datum>& col,
                 const madness::Group& row_group) const {
    // broadcast if I'm part of the broadcast group
    if (!row_group.empty()) {
      // Broadcast column k of left_.
      ProcessID group_root = get_row_group_root(k, row_group);
      bcast(left_start_local_ + k, left_stride_local_, row_group, group_root,
            0ul, col);
    }
  }

  /// Broadcast row \c k of \c right_ with a dense left-hand argument

  /// \param[in] k The row of \c right to be broadcast
  /// \param[out] row The vector that will hold the results of the broadcast
  void bcast_row(const ordinal_type k, std::vector<row_datum>& row,
                 const madness::Group& col_group) const {
    // broadcast if I'm part of the broadcast group
    if (!col_group.empty()) {
      // Compute the group root process.
      ProcessID group_root = get_col_group_root(k, col_group);

      // Broadcast row k of right_.
      bcast(k * proc_grid_.cols() + proc_grid_.rank_col(), right_stride_local_,
            col_group, group_root, left_.size(), row);
    }
  }

  void bcast_col_range_task(ordinal_type k, const ordinal_type end) const {
    // Compute the first local row of right
    const ordinal_type Pcols = proc_grid_.proc_cols();
    k += (Pcols - ((k + Pcols - proc_grid_.rank_col()) % Pcols)) % Pcols;

    for (; k < end; k += Pcols) {
      // Compute local iteration limits for column k of left_.
      ordinal_type index = left_start_local_ + k;

      // will create broadcast group only if needed
      bool have_group = false;
      madness::Group row_group;
      ProcessID group_root;
      bool do_broadcast;

      // Search column k of left for non-zero tiles
      for (; index < left_end_; index += left_stride_local_) {
        if (left_.shape().is_zero(index)) continue;

        // Construct broadcast group, if needed
        if (!have_group) {
          have_group = true;
          row_group = make_row_group(k);
          // broadcast if I am in this group and this group has others
          do_broadcast = !row_group.empty() && row_group.size() > 1;
          if (do_broadcast) group_root = get_row_group_root(k, row_group);
        }

        if (do_broadcast) {
          // Broadcast the tile
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
          ++left_ntiles_used_;
#endif
          const madness::DistributedID key(DistEvalImpl_::id(), index);
          auto tile = get_tile(left_, index);
          TensorImpl_::world().gop.bcast(key, tile, group_root, row_group);
        } else {
          // Discard the tile
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
          ++left_ntiles_discarded_;
#endif
          left_.discard(index);
        }
      }
    }
  }

  void bcast_row_range_task(ordinal_type k, const ordinal_type end) const {
    // Compute the first local row of right
    const ordinal_type Prows = proc_grid_.proc_rows();
    k += (Prows - ((k + Prows - proc_grid_.rank_row()) % Prows)) % Prows;

    for (; k < end; k += Prows) {
      // Compute local iteration limits for row k of right_.
      ordinal_type index = k * proc_grid_.cols();
      const ordinal_type row_end = index + proc_grid_.cols();
      index += proc_grid_.rank_col();

      // will create broadcast group only if needed
      bool have_group = false;
      madness::Group col_group;
      ProcessID group_root;
      bool do_broadcast;

      // Search for and broadcast non-zero row
      for (; index < row_end; index += right_stride_local_) {
        if (right_.shape().is_zero(index)) continue;

        // Construct broadcast group
        if (!have_group) {
          have_group = true;
          col_group = make_col_group(k);
          // broadcast if I am in this group and this group has others
          do_broadcast = !col_group.empty() && col_group.size() > 1;
          if (do_broadcast) group_root = get_col_group_root(k, col_group);
        }

        if (do_broadcast) {
          // Broadcast the tile
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
          ++right_ntiles_used_;
#endif
          const madness::DistributedID key(DistEvalImpl_::id(),
                                           index + left_.size());
          auto tile = get_tile(right_, index);
          TensorImpl_::world().gop.bcast(key, tile, group_root, col_group);
        } else {
          // Discard the tile
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
          ++right_ntiles_discarded_;
#endif
          right_.discard(index);
        }
      }
    }
  }

  // Row and column iteration functions ------------------------------------

  /// Find next non-zero row of \c right_ for a sparse shape

  /// Starting at the k-th row of the right-hand argument, find the next row
  /// that contains at least one non-zero tile. This search only checks for
  /// non-zero tiles in this processes column.
  /// \param k The first row to search
  /// \return The first row, greater than or equal to \c k with non-zero
  /// tiles, or \c k_ if none is found.
  ordinal_type iterate_row(ordinal_type k) const {
    // Iterate over k's until a non-zero tile is found or the end of the
    // matrix is reached.
    ordinal_type end = k * proc_grid_.cols();
    for (; k < k_; ++k) {
      // Search for non-zero tiles in row k of right
      ordinal_type i = end + proc_grid_.rank_col();
      end += proc_grid_.cols();
      for (; i < end; i += right_stride_local_)
        if (!right_.shape().is_zero(i)) return k;
    }

    return k;
  }

  /// Find the next non-zero column of \c left_ for an arbitrary shape type

  /// Starting at the k-th column of the left-hand argument, find the next
  /// column that contains at least one non-zero tile. This search only
  /// checks for non-zero tiles in this process's row.
  /// \param k The first column to test for non-zero tiles
  /// \return The first column, greater than or equal to \c k, that contains
  /// a non-zero tile. If no non-zero tile is not found, return \c k_.
  ordinal_type iterate_col(ordinal_type k) const {
    // Iterate over k's until a non-zero tile is found or the end of the
    // matrix is reached.
    for (; k < k_; ++k)
      // Search row k for non-zero tiles
      for (ordinal_type i = left_start_local_ + k; i < left_end_;
           i += left_stride_local_)
        if (!left_.shape().is_zero(i)) return k;

    return k;
  }

  /// Find the next k where the left- and right-hand argument have non-zero
  /// tiles

  /// Search for the next k-th column and row of the left- and right-hand
  /// arguments, respectively, that both contain non-zero tiles. This search
  /// only checks for non-zero tiles in this process's row or column. If a
  /// non-zero, local tile is found that does not contribute to local
  /// contractions, the tiles will be immediately broadcast.
  /// \param k The first row/column to check
  /// \return The next k-th column and row of the left- and right-hand
  /// arguments, respectively, that both have non-zero tiles
  ordinal_type iterate_sparse(const ordinal_type k) const {
    // Initial step for k_col and k_row.
    ordinal_type k_col = iterate_col(k);
    ordinal_type k_row = iterate_row(k_col);

    // Search for a row and column that both have non-zero tiles
    while (k_col != k_row) {
      if (k_col < k_row) {
        k_col = iterate_col(k_row);
      } else {
        k_row = iterate_row(k_col);
      }
    }

    if (k < k_row) {
      // Spawn a task to broadcast any local columns of left that were skipped
      TensorImpl_::world().taskq.add(shared_from_this(),
                                     &Summa_::bcast_col_range_task, k, k_row,
                                     madness::TaskAttributes::hipri());

      // Spawn a task to broadcast any local rows of right that were skipped
      TensorImpl_::world().taskq.add(shared_from_this(),
                                     &Summa_::bcast_row_range_task, k, k_col,
                                     madness::TaskAttributes::hipri());
    }

    return k_col;
  }

  /// Find the next k where the left- and right-hand argument have non-zero
  /// tiles

  /// Search for the next k-th column and row of the left- and right-hand
  /// arguments, respectively, that both contain non-zero tiles. This search
  /// only checks for non-zero tiles in this process's row or column. If a
  /// non-zero, local tile is found that does not contribute to local
  /// contractions, the tiles will be immediately broadcast.
  /// \param k The first row/column to check
  /// \return The next k-th column and row of the left- and right-hand
  /// arguments, respectively, that both have non-zero tiles
  ordinal_type iterate(const ordinal_type k) const {
    return (left_.shape().is_dense() && right_.shape().is_dense()
                ? k
                : iterate_sparse(k));
  }

  // Initialization functions ----------------------------------------------

  /// Initialize reduce tasks and construct broadcast groups
  ordinal_type initialize(const DenseShape&) {
    // if contraction is over zero-volume range just initialize tiles to zero
    if (k_ == 0) {
      ordinal_type tile_count = 0;
      const auto& tiles_range = this->trange().tiles_range();
      for (auto&& tile_idx : tiles_range) {
        auto tile_ord = tiles_range.ordinal(tile_idx);
        if (this->is_local(tile_ord)) {
          this->world().taskq.add([this, tile_ord, tile_idx]() {
            this->set_tile(tile_ord,
                           value_type(this->trange().tile(tile_idx),
                                      typename value_type::value_type{}));
          });
          ++tile_count;
        }
      }
      return tile_count;
    } else {
      // Construct static broadcast groups for dense arguments
      const madness::DistributedID col_did(DistEvalImpl_::id(), 0ul);
      col_group_ = proc_grid_.make_col_group(col_did);
      const madness::DistributedID row_did(DistEvalImpl_::id(), k_);
      row_group_ = proc_grid_.make_row_group(row_did);

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
      std::stringstream ss;
      ss << "init: rank=" << TensorImpl_::world().rank() << "\n    col_group_=("
         << col_did.first << ", " << col_did.second << ") { ";
      for (ProcessID gproc = 0ul; gproc < col_group_.size(); ++gproc)
        ss << col_group_.world_rank(gproc) << " ";
      ss << "}\n    row_group_=(" << row_did.first << ", " << row_did.second
         << ") { ";
      for (ProcessID gproc = 0ul; gproc < row_group_.size(); ++gproc)
        ss << row_group_.world_rank(gproc) << " ";
      ss << "}\n";
      printf(ss.str().c_str());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

      // Allocate memory for the reduce pair tasks.
      std::allocator<ReducePairTask<op_type>> alloc;
      reduce_tasks_ = alloc.allocate(proc_grid_.local_size());

      // Iterate over all local tiles
      const ordinal_type n = proc_grid_.local_size();
      for (ordinal_type t = 0ul; t < n; ++t) {
        // Initialize the reduction task
        ReducePairTask<op_type>* MADNESS_RESTRICT const reduce_task =
            reduce_tasks_ + t;
        new (reduce_task) ReducePairTask<op_type>(TensorImpl_::world(), op_
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
                                                  ,
                                                  nullptr, t
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
        );
      }

      return proc_grid_.local_size();
    }
  }

  /// Initialize reduce tasks
  template <typename Shape>
  ordinal_type initialize(const Shape& shape) {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
    std::stringstream ss;
    ss << "    initialize rank=" << TensorImpl_::world().rank() << " tiles={ ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

    // fast return if there is no work to do
    if (k_ == 0) return 0;

    // Allocate memory for the reduce pair tasks.
    std::allocator<ReducePairTask<op_type>> alloc;
    reduce_tasks_ = alloc.allocate(proc_grid_.local_size());

    // Initialize iteration variables
    ordinal_type row_start = proc_grid_.rank_row() * proc_grid_.cols();
    ordinal_type row_end = row_start + proc_grid_.cols();
    row_start += proc_grid_.rank_col();
    const ordinal_type col_stride =  // The stride to iterate down a column
        proc_grid_.proc_rows() * proc_grid_.cols();
    const ordinal_type row_stride =  // The stride to iterate across a row
        proc_grid_.proc_cols();
    const ordinal_type end = TensorImpl_::size();

    // Iterate over all local tiles
    ordinal_type tile_count = 0ul;
    ReducePairTask<op_type>* MADNESS_RESTRICT reduce_task = reduce_tasks_;
    // this loops over result tiles arranged in block-cyclic order
    // index = tile index (row major)
    for (; row_start < end; row_start += col_stride, row_end += col_stride) {
      for (ordinal_type index = row_start; index < row_end;
           index += row_stride, ++reduce_task) {
        // Initialize the reduction task

        // Skip zero tiles
        if (!shape.is_zero(DistEvalImpl_::perm_index_to_target(index))) {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
          ss << index << " ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

          new (reduce_task) ReducePairTask<op_type>(TensorImpl_::world(), op_
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
                                                    ,
                                                    nullptr, index
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
          );
          ++tile_count;
        } else {
          // Construct an empty task to represent zero tiles.
          new (reduce_task) ReducePairTask<op_type>();
        }
      }
    }

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
    ss << "}\n";
    printf(ss.str().c_str());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

    return tile_count;
  }

  ordinal_type initialize() {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
    printf("init: start rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

    const ordinal_type result = initialize(TensorImpl_::shape());

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE
    printf("init: finish rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_INITIALIZE

    return result;
  }

  // Finalize functions ----------------------------------------------------

  /// Set the result tiles, destroy reduce tasks, and destroy broadcast groups
  void finalize(const DenseShape&) {
    // Initialize iteration variables
    ordinal_type row_start = proc_grid_.rank_row() * proc_grid_.cols();
    ordinal_type row_end = row_start + proc_grid_.cols();
    row_start += proc_grid_.rank_col();
    const ordinal_type col_stride =  // The stride to iterate down a column
        proc_grid_.proc_rows() * proc_grid_.cols();
    const ordinal_type row_stride =  // The stride to iterate across a row
        proc_grid_.proc_cols();
    const ordinal_type end = TensorImpl_::size();

    // Iterate over all local tiles
    for (ReducePairTask<op_type>* reduce_task = reduce_tasks_; row_start < end;
         row_start += col_stride, row_end += col_stride) {
      for (ordinal_type index = row_start; index < row_end;
           index += row_stride, ++reduce_task) {
        // Set the result tile
        DistEvalImpl_::set_tile(DistEvalImpl_::perm_index_to_target(index),
                                reduce_task->submit());

        // Destroy the reduce task
        reduce_task->~ReducePairTask<op_type>();
      }
    }

    // Deallocate the memory for the reduce pair tasks.
    std::allocator<ReducePairTask<op_type>>().deallocate(
        reduce_tasks_, proc_grid_.local_size());
  }

  /// Set the result tiles and destroy reduce tasks
  template <typename Shape>
  void finalize(const Shape& shape) {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
    std::stringstream ss;
    ss << "    finalize rank=" << TensorImpl_::world().rank() << " tiles={ ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE

    // Initialize iteration variables
    ordinal_type row_start = proc_grid_.rank_row() * proc_grid_.cols();
    ordinal_type row_end = row_start + proc_grid_.cols();
    row_start += proc_grid_.rank_col();
    const ordinal_type col_stride =  // The stride to iterate down a column
        proc_grid_.proc_rows() * proc_grid_.cols();
    const ordinal_type row_stride =  // The stride to iterate across a row
        proc_grid_.proc_cols();
    const ordinal_type end = TensorImpl_::size();

    // Iterate over all local tiles
    for (ReducePairTask<op_type>* reduce_task = reduce_tasks_; row_start < end;
         row_start += col_stride, row_end += col_stride) {
      for (ordinal_type index = row_start; index < row_end;
           index += row_stride, ++reduce_task) {
        // Compute the permuted index
        const ordinal_type perm_index =
            DistEvalImpl_::perm_index_to_target(index);

        // Skip zero tiles
        if (!shape.is_zero(perm_index)) {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
          ss << index << " ";
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE

          // Set the result tile
          DistEvalImpl_::set_tile(perm_index, reduce_task->submit());
        }

        // Destroy the reduce task
        reduce_task->~ReducePairTask<op_type>();
      }
    }
    // Deallocate the memory for the reduce pair tasks.
    std::allocator<ReducePairTask<op_type>>().deallocate(
        reduce_tasks_, proc_grid_.local_size());

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
    ss << "}\n";
    printf(ss.str().c_str());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
  }

  void finalize() {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
    printf("finalize: start rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE

    finalize(TensorImpl_::shape());

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
    printf("finalize: finish rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_FINALIZE
  }

  /// SUMMA finalization task

  /// This task will set the tiles and do cleanup.
  class FinalizeTask : public madness::TaskInterface {
   private:
    std::shared_ptr<Summa_> owner_;  ///< The parent object for this task

   public:
    FinalizeTask(const std::shared_ptr<Summa_>& owner, const int ndep)
        : madness::TaskInterface(ndep, madness::TaskAttributes::hipri()),
          owner_(owner) {}

    virtual ~FinalizeTask() {}

    virtual void run(const madness::TaskThreadEnv&) { owner_->finalize(); }

  };  // class FinalizeTask

  // Contraction functions -------------------------------------------------

  /// Schedule local contraction tasks for \c col and \c row tile pairs

  /// Schedule tile contractions for each tile pair of \c row and \c col. A
  /// callback to \c task will be registered with each tile contraction
  /// task.
  /// \param col A column of tiles from the left-hand argument
  /// \param row A row of tiles from the right-hand argument
  /// \param task The task that depends on tile contraction tasks
  void contract(const DenseShape&, const ordinal_type,
                const std::vector<col_datum>& col,
                const std::vector<row_datum>& row,
                madness::TaskInterface* const task) {
    // Iterate over the row
    for (ordinal_type i = 0ul; i < col.size(); ++i) {
      // Compute the local, result-tile offset
      const ordinal_type reduce_task_offset =
          col[i].first * proc_grid_.local_cols();

      // Iterate over columns
      for (ordinal_type j = 0ul; j < row.size(); ++j) {
        const ordinal_type reduce_task_index =
            reduce_task_offset + row[j].first;

        // Schedule task for contraction pairs
        if (task) task->inc();
        const left_future left = col[i].second;
        const right_future right = row[j].second;
        reduce_tasks_[reduce_task_index].add(left, right, task);
      }
    }
  }

  /// Schedule local contraction tasks for \c col and \c row tile pairs

  /// Schedule tile contractions for each tile pair of \c row and \c col. A
  /// callback to \c task will be registered with each tile contraction
  /// task.
  /// \param col A column of tiles from the left-hand argument
  /// \param row A row of tiles from the right-hand argument
  /// \param task The task that depends on tile contraction tasks
  template <typename Shape>
  void contract(const Shape&, const ordinal_type,
                const std::vector<col_datum>& col,
                const std::vector<row_datum>& row,
                madness::TaskInterface* const task) {
    // Iterate over the row
    for (ordinal_type i = 0ul; i < col.size(); ++i) {
      // Compute the local, result-tile offset
      const ordinal_type reduce_task_offset =
          col[i].first * proc_grid_.local_cols();

      // Iterate over columns
      for (ordinal_type j = 0ul; j < row.size(); ++j) {
        const ordinal_type reduce_task_index =
            reduce_task_offset + row[j].first;

        // Skip zero tiles
        if (!reduce_tasks_[reduce_task_index]) continue;

        // Schedule task for contraction pairs
        if (task) {
          if (trace_tasks)
            task->inc_debug("destroy(*ReduceObject)");
          else
            task->inc();
        }
        const left_future left = col[i].second;
        const right_future right = row[j].second;
        reduce_tasks_[reduce_task_index].add(left, right, task);
      }
    }
  }

#define TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER
#ifndef TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER
  /// Schedule local contraction tasks for \c col and \c row tile pairs

  /// Schedule tile contractions for each tile pair of \c row and \c col. A
  /// callback to \c task will be registered with each tile contraction
  /// task. This version of contract is used when shape_type is
  /// \c SparseShape. It skips tile contractions that have a negligible
  /// contribution to the result tile.
  /// \tparam T The shape value type
  /// \param k The k step for this contraction set
  /// \param col A column of tiles from the left-hand argument
  /// \param row A row of tiles from the right-hand argument
  /// \param task The task that depends on the tile contraction tasks
  template <typename T>
  typename std::enable_if<std::is_floating_point<T>::value>::type contract(
      const SparseShape<T>&, const ordinal_type k,
      const std::vector<col_datum>& col, const std::vector<row_datum>& row,
      madness::TaskInterface* const task) {
    // Cache row shape data.
    std::vector<typename SparseShape<T>::value_type> row_shape_values;
    row_shape_values.reserve(row.size());
    const ordinal_type row_start =
        k * proc_grid_.cols() + proc_grid_.rank_col();
    for (ordinal_type j = 0ul; j < row.size(); ++j)
      row_shape_values.push_back(
          right_.shape()[row_start + (row[j].first * right_stride_local_)]);

    const ordinal_type col_start = left_start_local_ + k;
    const float threshold_k = TensorImpl_::shape().threshold() /
                              typename SparseShape<T>::value_type(k_);
    // Iterate over the row
    for (ordinal_type i = 0ul; i != col.size(); ++i) {
      // Compute the local, result-tile offset
      const ordinal_type offset = col[i].first * proc_grid_.local_cols();

      // Get the shape data for col_it tile
      const typename SparseShape<T>::value_type col_shape_value =
          left_.shape()[col_start + (col[i].first * left_stride_local_)];

      // Iterate over columns
      for (ordinal_type j = 0ul; j < row.size(); ++j) {
        if ((col_shape_value * row_shape_values[j]) < threshold_k) continue;

        const ordinal_type reduce_task_index = offset + row[j].first;

        // Skip zero tiles
        if (!reduce_tasks_[reduce_task_index]) continue;

        if (task) task->inc();
        reduce_tasks_[reduce_task_index].add(col[i].second, row[j].second,
                                             task);
      }
    }
  }
#endif  // TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER

  void contract(const ordinal_type k, const std::vector<col_datum>& col,
                const std::vector<row_datum>& row,
                madness::TaskInterface* const task) {
    contract(TensorImpl_::shape(), k, col, row, task);
  }

  // SUMMA step task -------------------------------------------------------

  /// SUMMA step task

  /// This task will perform a single SUMMA iteration, and start the next
  /// step task.
  class StepTask : public madness::TaskInterface {
   protected:
    // Member variables
    std::shared_ptr<Summa_> owner_;  ///< The owner of this task
    World& world_;
    std::vector<col_datum> col_{};
    std::vector<row_datum> row_{};
    FinalizeTask* finalize_task_;         ///< The SUMMA finalization task
    StepTask* next_step_task_ = nullptr;  ///< The next SUMMA step task
    StepTask* tail_step_task_ =
        nullptr;  ///< The last SUMMA step task that currently exists

    void get_col(const ordinal_type k) {
      owner_->get_col(k, col_);
      if (trace_tasks)
        this->notify_debug("StepTask::spawn_col");
      else
        this->notify();
    }

    void get_row(const ordinal_type k) {
      owner_->get_row(k, row_);
      if (trace_tasks)
        this->notify_debug("StepTask::spawn_row");
      else
        this->notify();
    }

   public:
    StepTask(const std::shared_ptr<Summa_>& owner, int finalize_ndep)
        :
#ifdef TILEDARRAY_ENABLE_TASK_DEBUG_TRACE
          madness::TaskInterface(0ul, "StepTask 1st ctor",
                                 madness::TaskAttributes::hipri()),
#else
          madness::TaskInterface(0ul, madness::TaskAttributes::hipri()),
#endif
          owner_(owner),
          world_(owner->world()),
          finalize_task_(new FinalizeTask(owner, finalize_ndep)) {
      TA_ASSERT(owner_);
      owner_->world().taskq.add(finalize_task_);
    }

    /// Construct the task for the next step

    /// \param parent The previous SUMMA step task
    /// \param ndep The number of dependencies for this task
    StepTask(StepTask* const parent, const int ndep)
        :
#ifdef TILEDARRAY_ENABLE_TASK_DEBUG_TRACE
          madness::TaskInterface(ndep, "StepTask nth ctor",
                                 madness::TaskAttributes::hipri()),
#else
          madness::TaskInterface(ndep, madness::TaskAttributes::hipri()),
#endif
          owner_(parent->owner_),
          world_(parent->world_),
          finalize_task_(parent->finalize_task_) {
      TA_ASSERT(parent);
      parent->next_step_task_ = this;
    }

    virtual ~StepTask() {}

    void spawn_get_row_col_tasks(const ordinal_type k) {
      // Submit the task to collect column tiles of left for iteration k
      if (trace_tasks)
        madness::DependencyInterface::inc_debug("StepTask::spawn_col");
      else
        madness::DependencyInterface::inc();
      world_.taskq.add(this, &StepTask::get_col, k,
                       madness::TaskAttributes::hipri());

      // Submit the task to collect row tiles of right for iteration k
      if (trace_tasks)
        madness::DependencyInterface::inc_debug("StepTask::spawn_row");
      else
        madness::DependencyInterface::inc();
      world_.taskq.add(this, &StepTask::get_row, k,
                       madness::TaskAttributes::hipri());
    }

    template <typename Derived>
    void make_next_step_tasks(Derived* task, ordinal_type depth) {
      // Set the depth to be no greater than the maximum number steps
      if (depth > owner_->k_) depth = owner_->k_;

      // Spawn n=depth step tasks
      for (; depth > 0ul; --depth) {
        // Set dep count of the tail task to 1, it will not start until this
        // task commands
        Derived* const next = new Derived(task, depth == 1 ? 1 : 0);
        task = next;
      }

      // Keep track of the tail ptr
      tail_step_task_ = task;
    }

    template <typename Derived, typename GroupType>
    void run(const ordinal_type k, const GroupType& row_group,
             const GroupType& col_group) {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_STEP
      printf("step:  start rank=%i k=%lu\n", owner_->world().rank(), k);
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_STEP

      if (k < owner_->k_) {
        // Initialize next tail task and submit next task
        TA_ASSERT(next_step_task_);
        next_step_task_->tail_step_task_ = new Derived(
            static_cast<Derived*>(tail_step_task_),
            1);  // <- ndep=1, will control its scheduling by this task
        // submit next step task ... even if it's same as tail_step_task_ it is
        // safe to submit because its ndep > 0 (see
        // StepTask::make_next_step_tasks)
        TA_ASSERT(tail_step_task_->ndep() > 0);
        world_.taskq.add(next_step_task_);
        next_step_task_ = nullptr;

        // Start broadcast of column and row tiles for this step
        world_.taskq.add(owner_, &Summa_::bcast_col, k, col_, row_group,
                         madness::TaskAttributes::hipri());
        world_.taskq.add(owner_, &Summa_::bcast_row, k, row_, col_group,
                         madness::TaskAttributes::hipri());

        // Submit tasks for the contraction of col and row tiles.
        owner_->contract(k, col_, row_, tail_step_task_);

        // Notify task dependencies
        TA_ASSERT(tail_step_task_);
        if (trace_tasks)
          tail_step_task_->notify_debug("StepTask nth ctor");
        else
          tail_step_task_->notify();
        finalize_task_->notify();

      } else if (finalize_task_) {
        // Signal the finalize task so it can run after all non-zero step
        // tasks have completed.
        finalize_task_->notify();

        // Cleanup any remaining step tasks
        StepTask* step_task = next_step_task_;
        while (step_task) {
          StepTask* const next_step_task = step_task->next_step_task_;
          step_task->next_step_task_ = nullptr;
          step_task->finalize_task_ = nullptr;
          world_.taskq.add(step_task);
          step_task = next_step_task;
        }

        if (trace_tasks)
          tail_step_task_->notify_debug("StepTask nth ctor");
        else
          tail_step_task_->notify();
      }

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_STEP
      printf("step: finish rank=%i k=%lu\n", owner_->world().rank(), k);
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_STEP
    }

  };  // class StepTask

  class DenseStepTask : public StepTask {
   protected:
    const ordinal_type k_;
    using StepTask::owner_;

   public:
    DenseStepTask(const std::shared_ptr<Summa_>& owner,
                  const ordinal_type depth)
        : StepTask(owner, owner->k_ + 1ul), k_(0) {
      StepTask::make_next_step_tasks(this, depth);
      StepTask::spawn_get_row_col_tasks(k_);
    }

    DenseStepTask(DenseStepTask* const parent, const int ndep)
        : StepTask(parent, ndep), k_(parent->k_ + 1ul) {
      // Spawn tasks to get k-th row and column tiles
      if (k_ < owner_->k_) StepTask::spawn_get_row_col_tasks(k_);
    }

    virtual ~DenseStepTask() {}

    virtual void run(const madness::TaskThreadEnv&) {
      StepTask::template run<DenseStepTask>(k_, owner_->row_group_,
                                            owner_->col_group_);
    }
  };  // class DenseStepTask

  class SparseStepTask : public StepTask {
   protected:
    Future<ordinal_type> k_{};
    Future<madness::Group> row_group_{};
    Future<madness::Group> col_group_{};
    using StepTask::finalize_task_;
    using StepTask::next_step_task_;
    using StepTask::owner_;
    using StepTask::world_;

   private:
    /// Spawn task to construct process groups and get tiles.
    void iterate_task(ordinal_type k, const ordinal_type offset) {
      // Search for the next non-zero row and column
      k = owner_->iterate_sparse(k + offset);
      k_.set(k);

      if (k < owner_->k_) {
        // NOTE: The order of task submissions is dependent on the order in
        // which we want the tasks to complete.

        // Spawn tasks to get k-th row and column tiles
        StepTask::spawn_get_row_col_tasks(k);

        // Spawn tasks to construct the row and column broadcast group
        row_group_ = world_.taskq.add(owner_, &Summa_::make_row_group, k,
                                      madness::TaskAttributes::hipri());
        col_group_ = world_.taskq.add(owner_, &Summa_::make_col_group, k,
                                      madness::TaskAttributes::hipri());

        // Increment the finalize task dependency counter, which indicates
        // that this task is not the terminating step task.
        TA_ASSERT(finalize_task_);
        finalize_task_->inc();
      }

      if (trace_tasks)
        madness::DependencyInterface::notify_debug("SparseStepTask ctor");
      else
        madness::DependencyInterface::notify();
    }

   public:
    SparseStepTask(const std::shared_ptr<Summa_>& owner, ordinal_type depth)
        : StepTask(owner, 1ul) {
      StepTask::make_next_step_tasks(this, depth);

      // Spawn a task to find the next non-zero iteration
      if (trace_tasks)
        madness::DependencyInterface::inc_debug("SparseStepTask ctor");
      else
        madness::DependencyInterface::inc();
      world_.taskq.add(this, &SparseStepTask::iterate_task, 0ul, 0ul,
                       madness::TaskAttributes::hipri());
    }

    SparseStepTask(SparseStepTask* const parent, const int ndep)
        : StepTask(parent, ndep) {
      if (parent->k_.probe() && (parent->k_.get() >= owner_->k_)) {
        // Avoid running extra tasks if not needed.
        k_.set(parent->k_.get());
        TA_ASSERT(ndep ==
                  1);  // ensure that this does not get executed immediately
      } else {
        // Spawn a task to find the next non-zero iteration
        if (trace_tasks)
          madness::DependencyInterface::inc_debug("SparseStepTask ctor");
        else
          madness::DependencyInterface::inc();
        world_.taskq.add(this, &SparseStepTask::iterate_task, parent->k_, 1ul,
                         madness::TaskAttributes::hipri());
      }
    }

    virtual ~SparseStepTask() {}

    virtual void run(const madness::TaskThreadEnv&) {
      StepTask::template run<SparseStepTask>(k_, row_group_, col_group_);
    }
  };  // class SparseStepTask

 public:
  /// Constructor

  /// \param left The left-hand argument evaluator
  /// \param right The right-hand argument evaluator
  /// \param world The world where the result lives
  /// \param trange The tiled range object for the result
  /// \param shape The tensor shape object for the result
  /// \param pmap The tile-process map for the result
  /// \param perm The permutation that is applied to result tile indices
  /// \param op The tile transform operation
  /// \param k The number of tiles in the inner dimension
  /// \param proc_grid The process grid that defines the layout of the tiles
  ///                  during the contraction evaluation
  /// \note The trange, shape, and pmap refer to the final,
  ///       permuted, state for the result, NOT to the result during
  ///       the SUMMA evaluation.
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  Summa(const left_type& left, const right_type& right, World& world,
        const trange_type trange, const shape_type& shape,
        const std::shared_ptr<const pmap_interface>& pmap, const Perm& perm,
        const op_type& op, const ordinal_type k, const ProcGrid& proc_grid)
      : DistEvalImpl_(world, trange, shape, pmap, outer(perm)),
        left_(left),
        right_(right),
        op_(op),
        row_group_(),
        col_group_(),
        k_(k),
        proc_grid_(proc_grid),
        reduce_tasks_(NULL),
        left_start_local_(proc_grid_.rank_row() * k),
        left_end_(left.size()),
        left_stride_(k),
        left_stride_local_(proc_grid.proc_rows() * k),
        right_stride_(1ul),
        right_stride_local_(proc_grid.proc_cols())
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
        ,
        left_ntiles_used_(0),
        right_ntiles_used_(0),
        left_ntiles_discarded_(0),
        right_ntiles_discarded_(0)
#endif
  {
  }

  virtual ~Summa() {}

  /// Get tile at index \c i

  /// \param i The index of the tile
  /// \return A \c Future to the tile at index i
  /// \throw TiledArray::Exception When tile \c i is owned by a remote node.
  /// \throw TiledArray::Exception When tile \c i a zero tile.
  Future<value_type> get_tile(ordinal_type i) const override {
    TA_ASSERT(TensorImpl_::is_local(i));
    TA_ASSERT(!TensorImpl_::is_zero(i));

    const ordinal_type source_index = DistEvalImpl_::perm_index_to_source(i);

    // Compute tile coordinate in tile grid
    const ordinal_type tile_row = source_index / proc_grid_.cols();
    const ordinal_type tile_col = source_index % proc_grid_.cols();
    // Compute process coordinate of tile in the process grid
    const ordinal_type proc_row = tile_row % proc_grid_.proc_rows();
    const ordinal_type proc_col = tile_col % proc_grid_.proc_cols();
    // Compute the process that owns tile
    const ProcessID source = proc_row * proc_grid_.proc_cols() + proc_col;

    const madness::DistributedID key(DistEvalImpl_::id(), i);
    return TensorImpl_::world().gop.template recv<value_type>(source, key);
  }

  /// Discard a tile that is not needed

  /// This function handles the cleanup for tiles that are not needed in
  /// subsequent computation.
  /// \param i The index of the tile
  void discard_tile(ordinal_type i) const override { get_tile(i); }

 private:
  /// Adjust iteration depth based on memory constraints

  /// \param depth The unbounded iteration depth
  /// \param left_sparsity The fraction of zero tiles in the left-hand matrix
  /// \param right_sparsity The fraction of zero tiles in the right-hand matrix
  /// \return The memory bounded iteration depth
  /// \thorw TiledArray::Exception When the memory bounded iteration depth
  /// is less than 1.
  ordinal_type mem_bound_depth(ordinal_type depth, const float left_sparsity,
                               const float right_sparsity) {
    // Check if a memory bound has been set
    const ordinal_type available_memory = max_memory_;
    if (available_memory) {
      // Compute the average memory requirement per iteration of this process
      const std::size_t local_memory_per_iter_left =
          (left_.trange().elements_range().volume() /
           left_.trange().tiles_range().volume()) *
          sizeof(typename numeric_type<typename left_type::eval_type>::type) *
          proc_grid_.local_rows() * (1.0f - left_sparsity);
      const std::size_t local_memory_per_iter_right =
          (right_.trange().elements_range().volume() /
           right_.trange().tiles_range().volume()) *
          sizeof(typename numeric_type<typename right_type::eval_type>::type) *
          proc_grid_.local_cols() * (1.0f - right_sparsity);

      // Compute the maximum number of iterations based on available memory
      const ordinal_type mem_bound_depth =
          ((local_memory_per_iter_left + local_memory_per_iter_right) /
           available_memory);

      // Check if the memory bounded depth is less than the optimal depth
      if (depth > mem_bound_depth) {
        // Adjust the depth based on the available memory
        switch (mem_bound_depth) {
          case 0:
            // When memory bound depth is
            TA_EXCEPTION("Insufficient memory available for SUMMA");
            break;
          case 1:
            if (TensorImpl_::world().rank() == 0)
              printf(
                  "!! WARNING TiledArray: Memory constraints limit the SUMMA "
                  "depth depth to 1.\n"
                  "!! WARNING TiledArray: Performance may be slow.\n");
          default:
            depth = mem_bound_depth;
        }
      }
    }

    return depth;
  }

  /// Evaluate the tiles of this tensor

  /// This function will evaluate the children of this distributed evaluator
  /// and evaluate the tiles for this distributed evaluator. It will block
  /// until the tasks for the children are evaluated (not for the tasks of
  /// this object).
  /// \return The number of tiles that will be set by this process
  int internal_eval() override {
#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL
    printf("eval: start eval children rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL

    // Start evaluate child tensors
    left_.eval();
    right_.eval();

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL
    printf("eval: finished eval children rank=%i\n",
           TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL

    ordinal_type tile_count = 0ul;
    if (proc_grid_.local_size() > 0ul) {
      tile_count = initialize();

      // depth controls the number of simultaneous SUMMA iterations
      // that are scheduled.

      // The optimal depth is equal to the smallest dimension of the process
      // grid, but no less than 2
      ordinal_type depth =
          std::max(ProcGrid::size_type(2),
                   std::min(proc_grid_.proc_rows(), proc_grid_.proc_cols()));

      // watch out for the corner case: contraction over zero-volume range
      // producing nonzero-volume result ... in that case there is nothing to do
      // the appropriate initialization was performed in the initialize() method
      if (k_ != 0) {
        // Construct the first SUMMA iteration task
        if (TensorImpl_::shape().is_dense()) {
          // We cannot have more iterations than there are blocks in the k
          // dimension
          if (depth > k_) depth = k_;

          // Modify the number of concurrent iterations based on the available
          // memory.
          depth = mem_bound_depth(depth, 0.0f, 0.0f);

          // Enforce user defined depth bound
          if (max_depth_) depth = std::min(depth, max_depth_);

          TensorImpl_::world().taskq.add(
              new DenseStepTask(shared_from_this(), depth));
        } else {
          // Increase the depth based on the amount of sparsity in an iteration.

          // Get the sparsity fractions for the left- and right-hand arguments.
          const float left_sparsity = left_.shape().sparsity();
          const float right_sparsity = right_.shape().sparsity();

          // Compute the fraction of non-zero result tiles in a single SUMMA
          // iteration.
          const float frac_non_zero = (1.0f - std::min(left_sparsity, 0.9f)) *
                                      (1.0f - std::min(right_sparsity, 0.9f));

          // Compute the new depth based on sparsity of the arguments
          depth = float(depth) * (1.0f - 1.35638f * std::log2(frac_non_zero)) +
                  0.5f;

          // We cannot have more iterations than there are blocks in the k
          // dimension
          if (depth > k_) depth = k_;

          // Modify the number of concurrent iterations based on the available
          // memory and sparsity of the argument tensors.
          depth = mem_bound_depth(depth, left_sparsity, right_sparsity);

          // Enforce user defined depth bound
          if (max_depth_) depth = std::min(depth, max_depth_);

          TensorImpl_::world().taskq.add(
              new SparseStepTask(shared_from_this(), depth));
        }
      }  // k_ != 0
    }

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL
    printf("eval: start wait children rank=%i\n", TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL

    // corner case: if left or right are zero-volume no tasks were scheduled, so
    // need to discard all of their tiles manually
    if (left_.range().volume() == 0) {
      for (auto&& tile_idx : right_.range()) {
        auto tile_ord = right_.range().ordinal(tile_idx);
        if (right_.is_local(tile_ord) && !right_.is_zero(tile_ord))
          right_.discard(tile_ord);
      }
    }
    if (right_.range().volume() == 0) {
      for (auto&& tile_idx : left_.range()) {
        auto tile_ord = left_.range().ordinal(tile_idx);
        if (left_.is_local(tile_ord) && !left_.is_zero(tile_ord))
          left_.discard(tile_ord);
      }
    }

    // Wait for child tensors to be evaluated, and process tasks while waiting.
    left_.wait();
    right_.wait();
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
    // values of left_ntiles_used_ etc. are not available until all broadcasts
    // have been completed ...
//    TA_ASSERT(left_.local_nnz() == left_ntiles_used_ +
//    left_ntiles_discarded_); TA_ASSERT(right_.local_nnz() ==
//    right_ntiles_used_ + right_ntiles_discarded_);
//    TA_ASSERT(left_.task_count() >= left_ntiles_used_ +
//    left_ntiles_discarded_); TA_ASSERT(right_.task_count() >=
//    right_ntiles_used_ + right_ntiles_discarded_);
#endif

#ifdef TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL
    printf("eval: finished wait children rank=%i\n",
           TensorImpl_::world().rank());
#endif  // TILEDARRAY_ENABLE_SUMMA_TRACE_EVAL

    return tile_count;
  }

};  // class Summa

// Initialize static member variables for Summa

template <typename Left, typename Right, typename Op, typename Policy>
typename Summa<Left, Right, Op, Policy>::ordinal_type
    Summa<Left, Right, Op, Policy>::max_depth_ =
        Summa<Left, Right, Op, Policy>::init_max_depth();

template <typename Left, typename Right, typename Op, typename Policy>
typename Summa<Left, Right, Op, Policy>::ordinal_type
    Summa<Left, Right, Op, Policy>::max_memory_ =
        Summa<Left, Right, Op, Policy>::init_max_memory();
}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_DIST_EVAL_CONTRACTION_EVAL_H__INCLUDED
