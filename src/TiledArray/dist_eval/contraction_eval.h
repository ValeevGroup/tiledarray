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

#include <TiledArray/dist_eval/dist_eval.h>
#include <TiledArray/proc_grid.h>
#include <TiledArray/reduce_task.h>
#include <TiledArray/tile_op/type_traits.h>
#include <TiledArray/shape.h>

namespace TiledArray {
  namespace detail {

    /// Distributed contraction evaluator implementation

    /// \param Left The left-hand argument evaluator type
    /// \param Right The right-hand argument evaluator type
    /// \param Op The contraction/reduction operation type
    /// \param Policy The tensor policy class
    /// \note The algorithms in this class assume that the arguments have a two-
    /// dimensional cyclic distribution, and that the row phase of the left-hand
    /// argument and the column phase of the right-hand argument are equal to
    /// the number of rows and columns, respectively, in the \c ProcGrid object
    /// passed to the constructor.
    template <typename Left, typename Right, typename Op, typename Policy>
    class ContractionEvalImpl : public DistEvalImpl<typename Op::result_type, Policy> {
    public:
      typedef ContractionEvalImpl<Left, Right, Op, Policy> ContractionEvalImpl_; ///< This object type
      typedef DistEvalImpl<typename Op::result_type, Policy> DistEvalImpl_; ///< The base class type
      typedef typename DistEvalImpl_::TensorImpl_ TensorImpl_; ///< The base, base class type
      typedef Left left_type; ///< The left-hand argument type
      typedef Right right_type; ///< The right-hand argument type
      typedef typename DistEvalImpl_::size_type size_type; ///< Size type
      typedef typename DistEvalImpl_::range_type range_type; ///< Range type
      typedef typename DistEvalImpl_::shape_type shape_type; ///< Shape type
      typedef typename DistEvalImpl_::pmap_interface pmap_interface; ///< Process map interface type
      typedef typename DistEvalImpl_::trange_type trange_type; ///< Tiled range type
      typedef typename DistEvalImpl_::value_type value_type; ///< Tile type
      typedef typename DistEvalImpl_::eval_type eval_type; ///< Tile evaluation type
      typedef Op op_type; ///< Tile evaluation operator type

    private:
      // Arguments and operation
      left_type left_; ///< The left-hand argument
      right_type right_; /// < The right-hand argument
      op_type op_; /// < The operation used to evaluate tile-tile contractions

      // Broadcast groups for dense arguments (empty for non-dense arguments)
      madness::Group row_group_; ///< The row process group for this rank
      madness::Group col_group_; ///< The column process group for this rank

      // Dimension information
      size_type k_; ///< Number of tiles in the inner dimension
      const ProcGrid proc_grid_; ///< Process grid for this contraction

      // Contraction results
      ReducePairTask<op_type>* reduce_tasks_; ///< A pointer to the reduction tasks

      // Constant used to iterate over columns and rows of left_ and right_, respectively.
      const size_type left_start_local_; ///< The starting point of left column iterator ranges (just add k for specific columns)
      const size_type left_end_; ///< The end of the left column iterator ranges
      const size_type left_stride_; ///< Stride for left column iterators
      const size_type left_stride_local_; ///< Stride for local left column iterators
      const size_type right_stride_; ///< Stride for right row iterators
      const size_type right_stride_local_; ///< stride for local right row iterators


      typedef madness::Future<typename right_type::eval_type> right_future; ///< Future to a right-hand argument tile
      typedef madness::Future<typename left_type::eval_type> left_future; ///< Future to a left-hand argument tile
      typedef std::pair<size_type, right_future> row_datum; ///< Datum element type for a right-hand argument row
      typedef std::pair<size_type, left_future> col_datum; ///< Datum element type for a left-hand argument column


      //------------------------------------------------------------------------
      // Process groups


      /// Process group factory function

      /// This function generates a sparse process group.
      /// \tparam Shape The shape type
      /// \tparam ProcMap The process map operation type
      /// \param shape The shape that will be used to select processes that are
      /// included in the process group
      /// \param index The first index of the row or column range
      /// \param end The end of the row or column range
      /// \param stride The row or column index stride
      /// \param max_group_size The maximum number of processes in the result
      /// group, which is equal to the number of process in this process row or
      /// column as defined by \c proc_grid_.
      /// \param proc_map The operator that will convert a process row/column
      /// into a process
      /// \param key The key that will be used to identify the process group
      /// \return A sparse process group that includes process in the row or
      /// column of this process as defined by \c proc_grid_.
      template <typename Shape, typename ProcMap>
      madness::Group make_group(const Shape& shape, size_type index,
          const size_type end, const size_type stride, const size_type max_group_size,
          const size_type k, const size_type key_offset, const ProcMap& proc_map) const
      {
        // Generate the list of processes in rank_row
        std::vector<ProcessID> proc_list(max_group_size, -1);

        // Flag the root processes of the broadcast, which may not be included
        // by shape.
        size_type p = k % max_group_size;
        proc_list[p] = proc_map(p);

        // Flag all process that have non-zero tiles
        size_type count = 1ul;
        for(p = 0ul; (index < end) && (count < max_group_size); index += stride,
            p = (p + 1u) % max_group_size)
        {
          if(proc_list[p] != -1) continue;
          if(shape.is_zero(index)) continue;

          proc_list[p] = proc_map(p);
          ++count;
        }

        // Remove processes from the list that will not be in the group
        for(size_type x = 0ul, p = 0ul; x < count; ++p) {
          if(proc_list[p] == -1) continue;
          proc_list[x++] = proc_list[p];
        }

        // Truncate invalid process id's
        proc_list.resize(count);

        return madness::Group(TensorImpl_::get_world(), proc_list,
            madness::DistributedID(TensorImpl_::id(), k + key_offset));
      }


      //------------------------------------------------------------------------
      // Broadcast kernels


      /// Broadcast a dense row or column

      /// \tparam Datum The result vector datum type
      /// \tparam GenTile Functor type that will generate the tiles to be broadcast
      /// \param[in] index The index of the first tile to be broadcast
      /// \param[in] end The end of the tile index range
      /// \param[in] stride The stride between tile indices
      /// \param[in] group_root The group root process of the broadcast
      /// \param[in] group The broadcast group
      /// \param[in] key_offset The offset that is applied to broadcast keys
      /// \param[in] gen_tile The tile generation functor
      /// \param[out] vec The vector that will return the broadcast tiles
      template <typename Datum, typename GenTile>
      void bcast(const DenseShape&, size_type index, const size_type end,
          const size_type stride, const ProcessID group_root, const madness::Group& group,
          const size_type key_offset, const GenTile& gen_tile, std::vector<Datum>& vec) const
      {
        // Iterate over tiles to be broadcast
        for(size_type i = 0ul; index < end; ++i, index += stride) {
          // Get tile
          vec.push_back(Datum(i, gen_tile(index)));

          // Broadcast the tile
          const madness::DistributedID key(TensorImpl_::id(), index + key_offset);
          TensorImpl_::get_world().gop.bcast(key, vec.back().second, group_root, group);
        }
      }

      /// Broadcast a sparse row or column

      /// \tparam Shape The sparse shape type
      /// \tparam Datum The result vector datum type
      /// \tparam GenTile Tile generation functor type
      /// \param[in] shape The shape that will be used to filter zero tiles from
      /// the broadcast
      /// \param[in] index The index of the first tile to be broadcast
      /// \param[in] end The end of the tile index range
      /// \param[in] stride The stride between tile indices
      /// \param[in] group_root The group root process of the broadcast
      /// \param[in] group The broadcast group
      /// \param[in] key_offset The offset that is applied to broadcast keys
      /// \param[in] gen_tile The tile generation functor
      /// \param[out] vec The vector that will return the broadcast tiles
      template <typename Shape, typename Datum, typename GenTile>
      void bcast(const Shape& shape, size_type index, const size_type end,
          const size_type stride, const ProcessID group_root, const madness::Group& group,
          const size_type key_offset, const GenTile& gen_tile, std::vector<Datum>& vec) const
      {
        // Iterate over tiles to be broadcast
        for(size_type i = 0ul; index < end; ++i, index += stride) {
          if(shape.is_zero(index)) continue;

          // Get tile
          vec.push_back(Datum(i, gen_tile(index)));

          // Broadcast the tile
          const madness::DistributedID key(TensorImpl_::id(), index + key_offset);
          TensorImpl_::get_world().gop.bcast(key, vec.back().second, group_root, group);
        }
      }

      /// Tile generator for the root process (owner of the data)

      /// This functor is used by the broadcast function to move tiles from the
      /// owner on the root process. If the tiles are lazy tiles, a conversion
      /// task is spawned.
      /// \param Arg The type of the argument that holds the input tiles
      template <typename Arg>
      class GenRootTile {
        Arg& arg_; ///< The argument object that owns the input tiles

        /// Tile conversion task function

        /// \tparam Tile The input tile type
        /// \param tile The input tile
        /// \return The evaluated version of the lazy tile
        template <typename Tile>
        static typename Tile::eval_type convert_task(const Tile& tile) { return tile; }


        /// Conversion function

        /// This function does nothing since tile is not a lazy tile.
        /// \tparam Tile The input tile type
        /// \param tile A future to the tile
        /// \return \c tile
        template <typename Tile>
        typename madness::disable_if<TiledArray::math::is_lazy_tile<Tile>,
            const madness::Future<Tile>& >::type
        convert(const madness::Future<Tile>& tile) { return tile; }


        /// Conversion function

        /// This function spawns a task that will convert a lazy tile from the
        /// tile type to the evaluated tile type.
        /// \tparam Tile The input tile type
        /// \param tile A future to the lazy tile
        /// \return A future to the evaluated tile
        template <typename Tile>
        typename madness::enable_if< TiledArray::math::is_lazy_tile<Tile>,
            madness::Future<typename Tile::eval_type> >::type
        convert(const madness::Future<Tile>& tile) const {
          return arg_.get_world().taskq.add(
              & GenRootTile<Arg>::template convert_task<Tile>,
              tile, madness::TaskAttributes::hipri());
        }

      public:
        GenRootTile(Arg& arg) : arg_(arg) { }

        madness::Future<typename Arg::eval_type> operator()(const size_type index) const {
          TA_ASSERT(arg_.is_local(index));
          return convert(arg_.move(index));
        }
      }; // class GenRootTile

      /// Tile generator object that generates new futures for tiles
      template <typename Arg>
      class GenEmptyTile {
      public:

        madness::Future<typename Arg::eval_type> operator()(const size_type) const {
          return madness::Future<typename Arg::eval_type>();
        }
      }; // class GenEmptyTile

      /// Broadcast tiles from \c arg

      /// \param[in] arg The owner of the input tiles
      /// \param[in] index The index of the first tile to be broadcast
      /// \param[in] end The end of the range of tiles to be broadcast
      /// \param[in] stride The stride between tile indices to be broadcast
      /// \param[in] group The process group where the tiles will be broadcast
      /// \param[in] key_offset The broadcast key offset value
      /// \param[in] vec The vector that will hold broadcast tiles
      template <typename Arg, typename Datum>
      void bcast(Arg& arg, size_type index, const size_type end, const size_type stride,
          const madness::Group& group, const size_type key_offset, std::vector<Datum>& vec) const
      {
        TA_ASSERT(vec.size() == 0ul);

        // Get the root process of the group
        const ProcessID group_root = group.rank(arg.owner(index));
        TA_ASSERT(group_root < group.size());

        if(group_root == group.rank()) {
          // Broadcast data from root process
          bcast(arg.shape(), index, end, stride, group_root, group,
              key_offset, GenRootTile<Arg>(arg), vec);
        } else {
          // Receive broadcast data on non-root processes
          bcast(arg.shape(), index, end, stride, group_root, group,
              key_offset, GenEmptyTile<Arg>(), vec);
        }
      }


      //------------------------------------------------------------------------
      // Broadcast specialization for left and right arguments


      /// Broadcast column \c k of \c left_ with a dense right-hand argument

      /// \param[in] k The column of \c left_ to be broadcast
      /// \param[out] col The vector that will hold the results of the broadcast
      void bcast_col(const DenseShape&, const size_type k, std::vector<col_datum>& col) const {
        // Broadcast column k of left_.
        bcast(left_, left_start_local_ + k, left_end_, left_stride_local_, row_group_, 0ul, col);
      }

      /// Map a process column to a process
      class MapCol {
        const ProcGrid& proc_grid_;  ///< Process grid that will be used to map columns
      public:
        MapCol(const ProcGrid& proc_grid) : proc_grid_(proc_grid) { }

        ProcessID operator()(const ProcGrid::size_type col) const
        { return proc_grid_.map_col(col); }
      }; // class MapCol

      /// Broadcast column \c k of \c left_ with a sparse right-hand argument

      /// \tparam RightShape The shape type of the left-hand argument
      /// \param[in] right_shape The shape of the right-hand argument
      /// \param[in] k The column of \c left_ to be broadcast
      /// \param[out] col The vector that will hold the results of the broadcast
      template <typename RightShape>
      void bcast_col(const RightShape& right_shape, const size_type k, std::vector<col_datum>& col) const {
        // Construct the sparse broadcast group
        const size_type right_begin_k = k * proc_grid_.cols();
        const size_type right_end_k = right_begin_k + proc_grid_.cols();
        madness::Group group = make_group(right_shape, right_begin_k, right_end_k,
            right_stride_, proc_grid_.proc_cols(), k, k_, MapCol(proc_grid_));

        // Broadcast column k of left_.
        bcast(left_, left_start_local_ + k, left_end_, left_stride_local_, group, 0ul, col);
      }

      /// Broadcast column \c k of \c left_

      /// \param[in] k The column of \c left_ to be broadcast
      /// \param[out] col The vector that will hold the results of the broadcast
      void bcast_col(const size_type k, std::vector<col_datum>& col) const {
        col.reserve(proc_grid_.local_rows());
        bcast_col(right_.shape(), k, col);
      }

      /// Broadcast row \c k of \c right_ with a dense left-hand argument

      /// \param[in] k The row of \c right to be broadcast
      /// \param[out] row The vector that will hold the results of the broadcast
      void bcast_row(const DenseShape&, const size_type k, std::vector<row_datum>& row) const {
        // Compute local iteration limits for row k of right_.
        size_type begin = k * proc_grid_.cols();
        const size_type end = begin + proc_grid_.cols();
        begin += proc_grid_.rank_col();

        // Broadcast row k of right_.
        bcast(right_, begin, end, right_stride_local_, col_group_, left_.size(), row);
      }

      /// Map a process row to a process
      class MapRow {
        const ProcGrid& proc_grid_; ///< Process grid that will be used to map rows
      public:
        MapRow(const ProcGrid& proc_grid) : proc_grid_(proc_grid) { }

        ProcessID operator()(const ProcGrid::size_type row) const
        { return proc_grid_.map_row(row); }
      }; // class MapRow


      /// Broadcast row \c k of \c right_ with a sparse left-hand argument

      /// \tparam LeftShape The shape type of the left-hand argument
      /// \param[in] left_shape The shape of the left-hand argument
      /// \param[in] k The row of \c right to be broadcast
      /// \param[out] row The vector that will hold the results of the broadcast
      template <typename LeftShape>
      void bcast_row(const LeftShape& left_shape, const size_type k, std::vector<row_datum>& row) const {
        // Construct the sparse broadcast group
        madness::Group group = make_group(left_shape, k, left_end_, left_stride_,
            proc_grid_.proc_rows(), k, 0ul, MapRow(proc_grid_));

        // Compute local iteration limits for row k of right_.
        size_type begin = k * proc_grid_.cols();
        const size_type end = begin + proc_grid_.cols();
        begin += proc_grid_.rank_col();

        // Broadcast row k of right_.
        bcast(right_, begin, end, right_stride_local_, group, left_.size(), row);
      }

      /// Broadcast row \c k of \c right_

      /// \param[in] k The row of \c right to be broadcast
      /// \param[out] row The vector that will hold the results of the broadcast
      void bcast_row(const size_type k, std::vector<row_datum>& row) const {
        row.reserve(proc_grid_.local_cols());
        bcast_row(left_.shape(), k, row);
      }

      void bcast_col_range_task(size_type k, const size_type end) const {
        // Compute the first local row of right
        const size_type Pcols = proc_grid_.proc_cols();
        k += (Pcols - ((k + Pcols - proc_grid_.rank_col()) % Pcols)) % Pcols;

        // Broadcast local row k of right.
        std::vector<col_datum> col;
        const col_datum null_value(0ul, left_future::default_initializer());
        for(; k < end; k += Pcols) {
          // Search column k of left for non-zero tiles
          for(size_type i = left_start_local_ + k; i < left_end_; i += left_stride_local_) {
            if(! left_.shape().is_zero(i)) {
              bcast_col(right_.shape(), k, col);
              col.resize(0ul, null_value);
              break;
            }
          }
        }
      }

      void bcast_row_range_task(size_type k, const size_type end) const {
        // Compute the first local row of right
        const size_type Prows = proc_grid_.proc_rows();
        k += (Prows - ((k + Prows - proc_grid_.rank_row()) % Prows)) % Prows;

        // Broadcast local row k of right.
        std::vector<row_datum> row;
        const row_datum null_value(0ul, right_future::default_initializer());

        for(size_type row_k_end = k * proc_grid_.cols(); k < end; k += Prows) {
          // Compute the iterator range for row k

          // Search column k for non-zero tiles
          size_type i = row_k_end;
          row_k_end += proc_grid_.cols();
          for(i += proc_grid_.rank_col(); i < row_k_end; i += right_stride_local_) {
            if(! right_.shape().is_zero(i)) {
              bcast_row(left_.shape(), k, row);
              row.resize(0ul, null_value);
              break;
            }
          }
        }
      }


      //------------------------------------------------------------------------
      // Row and column iteration functions


      /// Find next non-zero row of \c right_ for a sparse shape

      /// Starting at the k-th row of the right-hand argument, find the next row
      /// that contains at least one non-zero tile. This search only checks for
      /// non-zero tiles in this processes column.
      /// \param k The first row to search
      /// \return The first row, greater than or equal to \c k with non-zero
      /// tiles, or \c k_ if none is found.
      size_type iterate_row(size_type k) const {

        // Iterate over k's until a non-zero tile is found or the end of the
        // matrix is reached.
        size_type end = k * proc_grid_.cols();
        for(; k < k_; ++k) {
          // Search for non-zero tiles in row k of right
          size_type i = end;
          end += proc_grid_.cols();
          for(i += proc_grid_.rank_col(); i < end; i += right_stride_local_)
            if(! right_.shape().is_zero(i))
              return k;
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
      size_type iterate_col(size_type k) const {
        // Iterate over k's until a non-zero tile is found or the end of the
        // matrix is reached.
        for(; k < k_; ++k)
          // Search row k for non-zero tiles
          for(size_type i = left_start_local_ + k; i < left_end_; i += left_stride_local_)
            if(! left_.shape().is_zero(i))
              return k;

        return k;
      }

      /// Find the next k where the left- and right-hand argument have non-zero tiles

      /// Search for the next k-th column and row of the left- and right-hand
      /// arguments, respectively, that both contain non-zero tiles. This search
      /// only checks for non-zero tiles in this process's row or column. If a
      /// non-zero, local tile is found that does not contribute to local
      /// contractions, the tiles will be immediately broadcast.
      /// \param k The first row/column to check
      /// \return The next k-th column and row of the left- and right-hand
      /// arguments, respectively, that both have non-zero tiles
      size_type iterate(const DenseShape&, const DenseShape&,
          const std::shared_ptr<ContractionEvalImpl_>&, const size_type k) const
      {
        return k;
      }

      /// Find the next k where the left- and right-hand argument have non-zero tiles

      /// Search for the next k-th column and row of the left- and right-hand
      /// arguments, respectively, that both contain non-zero tiles. This search
      /// only checks for non-zero tiles in this process's row or column. If a
      /// non-zero, local tile is found that does not contribute to local
      /// contractions, the tiles will be immediately broadcast.
      /// \param k The first row/column to check
      /// \return The next k-th column and row of the left- and right-hand
      /// arguments, respectively, that both have non-zero tiles
      template <typename LeftShape, typename RightShape>
      size_type iterate(const LeftShape&, const RightShape&,
          const std::shared_ptr<ContractionEvalImpl_>& self, const size_type k) const
      {
        // Initial step for k_col and k_row.
        size_type k_col = iterate_col(k);
        size_type k_row = iterate_row(k);

        // Search for a row and column that both have non-zero tiles
        while(k_col != k_row) {
          if(k_col < k_row) {
            k_col = iterate_col(k_row);
          } else {
            k_row = iterate_row(k_col);
          }
        }

        if(k < k_row) {
          // Spawn a task to broadcast any local columns of left that were skipped
          TensorImpl_::get_world().taskq.add(self,
              & ContractionEvalImpl_::bcast_col_range_task, k, k_row,
              madness::TaskAttributes::hipri());

          // Spawn a task to broadcast any local rows of right that were skipped
          TensorImpl_::get_world().taskq.add(self,
              & ContractionEvalImpl_::bcast_row_range_task, k, k_col,
              madness::TaskAttributes::hipri());
        }

        return k_col;
      }


      /// Find the next k where the left- and right-hand argument have non-zero tiles

      /// Search for the next k-th column and row of the left- and right-hand
      /// arguments, respectively, that both contain non-zero tiles. This search
      /// only checks for non-zero tiles in this process's row or column. If a
      /// non-zero, local tile is found that does not contribute to local
      /// contractions, the tiles will be immediately broadcast.
      /// \param k The first row/column to check
      /// \return The next k-th column and row of the left- and right-hand
      /// arguments, respectively, that both have non-zero tiles
      size_type iterate(const std::shared_ptr<ContractionEvalImpl_>& self, const size_type k) const {
        return iterate(left_.shape(), right_.shape(), self, k);
      }



      //------------------------------------------------------------------------
      // Initialization functions

      /// Initialize reduce tasks and construct broadcast groups
      size_type initialize(const DenseShape&) {
        // Construct static broadcast groups for dense arguments
        col_group_ = proc_grid_.make_col_group(madness::DistributedID(TensorImpl_::id(), 0ul));
        row_group_ = proc_grid_.make_row_group(madness::DistributedID(TensorImpl_::id(), k_));

        // Allocate memory for the reduce pair tasks.
        std::allocator<ReducePairTask<op_type> > alloc;
        reduce_tasks_ = alloc.allocate(proc_grid_.local_size());

        // Iterate over all local tiles
        const size_type n = proc_grid_.local_size();
        for(size_type t = 0ul; t < n; ++t) {
          // Initialize the reduction task
          ReducePairTask<op_type>* restrict const reduce_task = reduce_tasks_ + t;
          new(reduce_task) ReducePairTask<op_type>(TensorImpl_::get_world(), op_);
        }

        return proc_grid_.local_size();
      }

      /// Initialize reduce tasks
      template <typename Shape>
      size_type initialize(const Shape& shape) {
        // Allocate memory for the reduce pair tasks.
        std::allocator<ReducePairTask<op_type> > alloc;
        reduce_tasks_ = alloc.allocate(proc_grid_.local_size());

        // Initialize iteration variables
        size_type row_start = proc_grid_.rank_row() * proc_grid_.cols();
        size_type row_end = row_start + proc_grid_.cols();
        row_start += proc_grid_.rank_col();
        const size_type col_stride = // The stride to iterate down a column
            proc_grid_.proc_rows() * proc_grid_.cols();
        const size_type row_stride = // The stride to iterate across a row
            proc_grid_.proc_cols();
        const size_type end = TensorImpl_::size();

        // Iterate over all local tiles
        size_type tile_count = 0ul;
        for(size_type t = 0ul; row_start < end; row_start += col_stride, row_end += col_stride) {
          for(size_type index = row_start; index < row_end; index += row_stride, ++t) {
            // Skip zero tiles
            if(shape.is_zero(DistEvalImpl_::perm_index(index))) continue;

            // Initialize the reduction task
            ReducePairTask<op_type>* restrict const reduce_task = reduce_tasks_ + t;
            new(reduce_task) ReducePairTask<op_type>(TensorImpl_::get_world(), op_);
            ++tile_count;
          }
        }

        return tile_count;
      }

      size_type initialize() { return initialize(TensorImpl_::shape()); }


      //------------------------------------------------------------------------
      // Finalize functions


      /// Set the result tiles, destroy reduce tasks, and destroy broadcast groups
      void finalize(const DenseShape&) {
        // Initialize iteration variables
        size_type row_start = proc_grid_.rank_row() * proc_grid_.cols();
        size_type row_end = row_start + proc_grid_.cols();
        row_start += proc_grid_.rank_col();
        const size_type col_stride = // The stride to iterate down a column
            proc_grid_.proc_rows() * proc_grid_.cols();
        const size_type row_stride = // The stride to iterate across a row
            proc_grid_.proc_cols();
        const size_type end = TensorImpl_::size();

        // Iterate over all local tiles
        for(ReducePairTask<op_type>* reduce_task = reduce_tasks_;
            row_start < end; row_start += col_stride, row_end += col_stride) {
          for(size_type index = row_start; index < row_end; index += row_stride, ++reduce_task) {


            // Set the result tile
            DistEvalImpl_::set_tile(DistEvalImpl_::perm_index(index),
                reduce_task->submit());

            // Destroy the the reduce task
            reduce_task->~ReducePairTask<op_type>();
          }
        }

        // Deallocate the memory for the reduce pair tasks.
        std::allocator<ReducePairTask<op_type> >().deallocate(reduce_tasks_,
            proc_grid_.local_size());
      }

      /// Set the result tiles and destroy reduce tasks
      template <typename Shape>
      void finalize(const Shape& shape) {
        // Initialize iteration variables
        size_type row_start = proc_grid_.rank_row() * proc_grid_.cols();
        size_type row_end = row_start + proc_grid_.cols();
        row_start += proc_grid_.rank_col();
        const size_type col_stride = // The stride to iterate down a column
            proc_grid_.proc_rows() * proc_grid_.cols();
        const size_type row_stride = // The stride to iterate across a row
            proc_grid_.proc_cols();
        const size_type end = TensorImpl_::size();

        // Iterate over all local tiles
        for(ReducePairTask<op_type>* reduce_task = reduce_tasks_;
            row_start < end; row_start += col_stride, row_end += col_stride) {
          for(size_type index = row_start; index < row_end; index += row_stride, ++reduce_task) {
            // Compute the permuted index
            const size_type perm_index = DistEvalImpl_::perm_index(index);

            // Skip zero tiles
            if(shape.is_zero(perm_index)) continue;


            // Set the result tile
            DistEvalImpl_::set_tile(perm_index, reduce_task->submit());

            // Destroy the the reduce task
            reduce_task->~ReducePairTask<op_type>();
          }
        }

        // Deallocate the memory for the reduce pair tasks.
        std::allocator<ReducePairTask<op_type> >().deallocate(reduce_tasks_,
            proc_grid_.local_size());
      }

      void finalize() { finalize(TensorImpl_::shape()); }

      /// SUMMA finalization task

      /// This task will set the tiles and do cleanup.
      class FinalizeTask : public madness::TaskInterface {
      private:
        std::shared_ptr<ContractionEvalImpl_> owner_; ///< The parent object for this task

      public:
        FinalizeTask(const std::shared_ptr<ContractionEvalImpl_>& owner) :
          madness::TaskInterface(1, madness::TaskAttributes::hipri()),
          owner_(owner)
        { }

        virtual ~FinalizeTask() { }

        virtual void run(const madness::TaskThreadEnv&) { owner_->finalize(); }

      }; // class FinalizeTask


      //------------------------------------------------------------------------
      // Contraction functions


      /// Schedule local contraction tasks for \c col and \c row tile pairs

      /// Schedule tile contractions for each tile pair of \c row and \c col. A
      /// callback to \c task will be registered with each tile contraction
      /// task.
      /// \param col A column of tiles from the left-hand argument
      /// \param row A row of tiles from the right-hand argument
      /// \param callback The callback that will be invoked after each tile-pair
      /// has been contracted
      template <typename Shape>
      void contract(const Shape&, const size_type,
          const std::vector<col_datum>& col, const std::vector<row_datum>& row,
          madness::TaskInterface* const task)
      {
        // Iterate over the row
        for(size_type i = 0ul; i < col.size(); ++i) {
          // Compute the local, result-tile offset
          const size_type offset = col[i].first * proc_grid_.local_cols();

          // Iterate over columns
          for(size_type j = 0ul; j < row.size(); ++j) {
            if(task)
              task->inc();
            reduce_tasks_[offset + row[j].first].add(col[i].second, row[j].second, task);
          }
        }
      }

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
      typename madness::enable_if<std::is_floating_point<T> >::type
      contract(const SparseShape<T>&, const size_type k,
          const std::vector<col_datum>& col, const std::vector<row_datum>& row,
          madness::TaskInterface* const task)
      {
        // Cache row shape data.
        std::vector<typename SparseShape<T>::value_type> row_shape_values;
        row_shape_values.reserve(row.size());
        const size_type row_start = k * proc_grid_.cols() + proc_grid_.rank_col();
        for(size_type j = 0ul; j < row.size(); ++j)
          row_shape_values.push_back(right_.shape()[row_start + (row[j].first * right_stride_local_)]);

        const size_type col_start = left_start_local_ + k;
        const float threshold_k = TensorImpl_::shape().threshold() / typename SparseShape<T>::value_type(k_);
        // Iterate over the row
        for(size_type i = 0ul; i != col.size(); ++i) {
          // Compute the local, result-tile offset
          const size_type offset = col[i].first * proc_grid_.local_cols();

          // Get the shape data for col_it tile
          const typename SparseShape<T>::value_type col_shape_value =
              left_.shape()[col_start + (col[i].first * left_stride_local_)];

          // Iterate over columns
          for(size_type j = 0ul; j < row.size(); ++j) {
            if((col_shape_value * row_shape_values[j]) < threshold_k)
              continue;

            if(task)
              task->inc();
            reduce_tasks_[offset + row[j].first].add(col[i].second, row[j].second, task);
          }
        }
      }
#endif // TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER

      void contract(const size_type k, const std::vector<col_datum>& col,
          const std::vector<row_datum>& row, madness::TaskInterface* const task)
      { contract(TensorImpl_::shape(), k, col, row, task); }


      //------------------------------------------------------------------------
      // SUMMA step task


      /// SUMMA step task

      /// This task will perform a single SUMMA iteration, and start the next
      /// step task.
      class StepTask : public madness::TaskInterface {
      private:
        // Member variables
        std::shared_ptr<ContractionEvalImpl_> owner_; ///< The owner of this task
        size_type k_; ///< The SUMMA that will be processed by this task
        FinalizeTask* finalize_task_; ///< The SUMMA finalization task
        StepTask* next_step_task_; ///< The next SUMMA step task

        /// Construct the task for the next step

        /// \param parent The previous SUMMA step task
        /// \param ndep The number of dependencies for this task
        StepTask(StepTask* const parent, const int ndep) :
          madness::TaskInterface(ndep, madness::TaskAttributes::hipri()),
          owner_(parent->owner_),
          k_(0ul),
          finalize_task_(parent->finalize_task_),
          next_step_task_(NULL)
        { }

      public:

        StepTask(const std::shared_ptr<ContractionEvalImpl_>& owner) :
          madness::TaskInterface(madness::TaskAttributes::hipri()),
          owner_(owner),
          k_(0ul),
          finalize_task_(new FinalizeTask(owner)),
          next_step_task_(new StepTask(this, 1))
        {
          owner_->get_world().taskq.add(next_step_task_);
          owner_->get_world().taskq.add(finalize_task_);
        }

        virtual ~StepTask() { }

        // Initialize member variables
        StepTask* initialize(const size_type k) {
          k_ = k;
          StepTask* step_task = NULL;
          if(k < owner_->k_) {
            next_step_task_ = step_task = new StepTask(this, 2);
            owner_->get_world().taskq.add(step_task);
          } else {
            finalize_task_->notify();
          }
          this->notify();
          return step_task;
        }

        virtual void run(const madness::TaskThreadEnv&) {
          // Search for the next k to be processed
          if(k_ < owner_->k_) {

            k_ = owner_->iterate(owner_, k_);

            if(k_ < owner_->k_) {
              // Add a dependency to the finalize task
              finalize_task_->inc();

              // Initialize and submit next task
              StepTask* const next_next_step_task = next_step_task_->initialize(k_ + 1ul);
              next_step_task_ = NULL; // The next step task can start running after this point

              // Broadcast row and column
              std::vector<col_datum> col;
              owner_->bcast_col(k_, col);
              std::vector<row_datum> row;
              owner_->bcast_row(k_, row);

              // Submit tasks for the contraction of col and row tiles.
              owner_->contract(k_, col, row, next_next_step_task);

              // Notify task dependencies
              if(next_next_step_task)
                next_next_step_task->notify();
              finalize_task_->notify();

            } else {
              finalize_task_->notify();
              if(next_step_task_) {
                next_step_task_->k_ = std::numeric_limits<size_type>::max();
                next_step_task_->notify();
              }
            }
          }
        }

      }; // class StepTask

    public:

      /// Constructor

      /// \param left The left-hand argument evaluator
      /// \param right The right-hand argument evaluator
      /// \param world The world where this evaluator will live
      /// \param trange The tiled range of the result tensor
      /// \param shape The shape of the result tensor
      /// \param pmap The process map for the result tensor
      /// \param perm The permutation that will be applied to tiles and the
      /// coordinate index after contraction of the result tile
      /// \param op The operation that will be used to contract tile pairs
      /// \param k The number of tiles in the inner dimension
      /// \param proc_grid The process grid that defines the layout of the tiles
      /// during the contraction evaluation
      /// \note The trange, shape, and pmap are assumed to be in the final,
      /// permuted, state for the result.
      ContractionEvalImpl(const left_type& left, const right_type& right,
          madness::World& world, const trange_type trange, const shape_type& shape,
          const std::shared_ptr<pmap_interface>& pmap, const Permutation& perm,
          const op_type& op, const size_type k, const ProcGrid& proc_grid) :
        DistEvalImpl_(world, trange, shape, pmap, perm),
        left_(left), right_(right), op_(op),
        row_group_(), col_group_(),
        k_(k), proc_grid_(proc_grid),
        reduce_tasks_(NULL),
        left_start_local_(proc_grid_.rank_row() * k),
        left_end_(left.size()),
        left_stride_(k),
        left_stride_local_(proc_grid.proc_rows() * k),
        right_stride_(1ul),
        right_stride_local_(proc_grid.proc_cols())
      { }

      virtual ~ContractionEvalImpl() { }

    private:

      /// Evaluate the tiles of this tensor

      /// This function will evaluate the children of this distributed evaluator
      /// and evaluate the tiles for this distributed evaluator. It will block
      /// until the tasks for the children are evaluated (not for the tasks of
      /// this object).
      /// \param pimpl A shared pointer to this object
      /// \return The number of tiles that will be set by this process
      virtual int internal_eval(const std::shared_ptr<DistEvalImpl_>& pimpl) {
        // Convert pimpl to this object type so it can be used in tasks
        std::shared_ptr<ContractionEvalImpl_> self =
            std::static_pointer_cast<ContractionEvalImpl_>(pimpl);

        // Start evaluate child tensors
        left_.eval();
        right_.eval();

        size_type tile_count = 0ul;
        if(proc_grid_.local_size() > 0ul) {
          tile_count = initialize();

          // Construct the first SUMMA iteration task
          TensorImpl_::get_world().taskq.add(new StepTask(self));
        }

        // Wait for child tensors to be evaluated, and process tasks while waiting.
        left_.wait();
        right_.wait();

        return tile_count;
      }

    }; // class ContractionEvalImpl


    /// Distributed contraction evaluator factory function

    /// Construct a distributed contraction evaluator, which constructs a new
    /// tensor by applying \c op to tiles of \c left and \c right.
    /// \tparam LeftTile Tile type of the left-hand argument
    /// \tparam RightTile Tile type of the right-hand argument
    /// \tparam Policy The policy type of the argument
    /// \tparam Op The unary tile operation
    /// \param left The left-hand argument
    /// \param right The right-hand argument
    /// \param world The world where the argument will be evaluated
    /// \param shape The shape of the evaluated tensor
    /// \param pmap The process map for the evaluated tensor
    /// \param perm The permutation applied to the tensor
    /// \param op The contraction/reduction tile operation
    template <typename LeftTile, typename RightTile, typename Policy, typename Op>
    DistEval<typename Op::result_type, Policy> make_contract_eval(
        const DistEval<LeftTile, Policy>& left,
        const DistEval<RightTile, Policy>& right,
        madness::World& world,
        const typename DistEval<typename Op::result_type, Policy>::shape_type& shape,
        const std::shared_ptr<typename DistEval<typename Op::result_type, Policy>::pmap_interface>& pmap,
        const Permutation& perm,
        const Op& op)
    {
      TA_ASSERT(left.range().dim() == op.left_rank());
      TA_ASSERT(right.range().dim() == op.right_rank());
      TA_ASSERT((perm.dim() == op.result_rank()) || !perm);

      // Define the impl type
      typedef ContractionEvalImpl<DistEval<LeftTile, Policy>, DistEval<RightTile,
          Policy>, Op, Policy> impl_type;

      // Precompute iteration range data
      const unsigned int num_contract_ranks = op.num_contract_ranks();
      const unsigned int left_end = op.left_rank();
      const unsigned int left_middle = left_end - num_contract_ranks;
      const unsigned int right_end = op.right_rank();

      // Construct a vector TiledRange1 objects from the left- and right-hand
      // arguments that will be used to construct the result TiledTange. Also,
      // compute the fused outer dimension sizes, number of tiles and elements,
      // for the contraction.
      typename impl_type::trange_type::Ranges ranges(op.result_rank());
      std::size_t M = 1ul, m = 1ul, N = 1ul, n = 1ul;
      std::size_t pi = 0ul;
      for(unsigned int i = 0ul; i < left_middle; ++i) {
        ranges[(perm ? perm[pi++] : pi++)] = left.trange().data()[i];
        M *= left.range().size()[i];
        m *= left.trange().elements().size()[i];
      }
      for(std::size_t i = num_contract_ranks; i < right_end; ++i) {
        ranges[(perm ? perm[pi++] : pi++)] = right.trange().data()[i];
        N *= right.range().size()[i];
        n *= right.trange().elements().size()[i];
      }

      // Compute the number of tiles in the inner dimension.
      std::size_t K = 1ul;
      for(std::size_t i = left_middle; i < left_end; ++i)
        K *= left.range().size()[i];

      // Construct the result range
      typename impl_type::trange_type trange(ranges.begin(), ranges.end());

      // Construct the process grid
      ProcGrid proc_grid(world, M, N, m, n);

      return DistEval<typename Op::result_type, Policy>(
          std::shared_ptr<typename impl_type::DistEvalImpl_>(new impl_type(left,
              right, world, trange, shape, pmap, perm, op, K, proc_grid)));
    }

  } // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_DIST_EVAL_CONTRACTION_EVAL_H__INCLUDED
