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
      left_type left_; ///< The left argument tensor
      right_type right_; /// < The right argument tensor
      op_type op_; /// < The contraction/reduction operation

      // Broadcast groups for dense arguments (empty for non-dense arguments)
      madness::Group row_group_; ///< The row process group for this rank
      madness::Group col_group_; ///< The column process group for this rank

      // Dimension information
      size_type k_; ///< Number of tiles in the inner dimension
      const ProcGrid proc_grid_; ///< Process grid for this contraction

      // Contraction results
      ReducePairTask<op_type>* reduce_tasks_; ///< A pointer to the reduction tasks

      // Constant left and right index iterator bounds
      const size_type left_start_local_;
      const size_type left_end_;
      const size_type left_stride_;
      const size_type left_stride_local_;
      const size_type right_stride_;
      const size_type right_stride_local_;


      typedef madness::Future<typename right_type::eval_type> right_future; ///< Future to a right-hand argument tile
      typedef madness::Future<typename left_type::eval_type> left_future; ///< Future to a left-hand argument tile
      typedef std::pair<size_type, right_future> row_datum; ///< Datum element type for a right-hand argument row
      typedef std::pair<size_type, left_future> col_datum; ///< Datum element type for a left-hand argument column


      //------------------------------------------------------------------------
      // Iterator functions

      size_type left_begin(const size_type k) const { return k; }
      size_type left_begin_local(const size_type k) const { return left_start_local_ + k; }

      size_type right_begin(const size_type k) const { return k * proc_grid_.cols(); }
      size_type right_begin_local(const size_type k) const { return k * proc_grid_.cols() + proc_grid_.rank_col(); }
      size_type right_end(const size_type k) const { return (k + 1ul) * proc_grid_.cols(); }


      //------------------------------------------------------------------------
      // Process group factor functions


      /// Process group factory function

      /// This function generates a sparse process group. Processes are included
      /// \tparam Shape The shape type
      /// \tparam ProcMap The process map operation type
      /// \param shape The shape that will be used to select processes included
      /// in the result group
      /// \param index The first shape index
      /// \param end The last shape index
      /// \param strid The stride of the shape index
      /// \param max_group_size The maximum number of processes in the result group
      /// \param proc_map The operator that will convert a process row/column
      /// into a process
      /// \param key The key that will be used to identify the result group
      /// \return A sparse group
      template <typename Shape, tyepname ProcMap>
      madness::Group make_group(const Shape& shape, size_type index,
          const size_type end, const size_type stride, const size_type max_group_size,
          const ProcMap& proc_map, std::size_t key) const
      {
        // Generate the list of processes in rank_row
        std::vector<ProcessID> proc_list(max_group_size, -1);

        // Flag all process that have non-zero tiles
        size_type count = 0ul;
        for(size_type p = 0u; (index < end) && (count < max_group_size); index += stride,
            p = (p + 1u) % max_group_size)
        {
          if(proc_list[p] != -1) continue;
          if(shape.is_zero(index)) continue;

          proc_list[p] = proc_map(p);
          ++count;
        }

        // Remove processes from the list that will not be in the group
        for(size_type p = 0ul, x = 0ul; (p < max_group_size) && (x < count); ++p) {
          if(proc_list[p] == -1) continue;
          proc_list[x++] = proc_list[p];
        }

        // Truncate invalid process id's
        proc_list.resize(count);

        return madness::Group(TensorImpl_::get_world(), proc_list,
            madness::DistributedID(TensorImpl_::id(), key));
      }

      /// Map a process row to a process
      class MapRow {
        const ProcGrid& proc_grid_; ///< Process grid that will be used to map rows
      public:
        MapRow(const ProcGrid& proc_grid) : proc_grid_(proc_grid) { }

        ProcessID operator()(const ProcGrid::size_type row) const
        { return proc_grid_.map_row(row); }
      }; // class MapRow

      /// Sparse column group factor function

      /// \param k The column of left shape that will be used to generate the group
      /// \return A sparse group for column \c k of \c left_
      madness::Group make_col_group(const size_type k) const {
        return make_group(left_.shape(), k, left_end_, left_stride_,
            proc_grid_.proc_rows(), MapRow(proc_grid_), k);
      }

      /// Map a process column to a process
      class MapCol {
        const ProcGrid& proc_grid_;  ///< Process grid that will be used to map columns
      public:
        MapCol(const ProcGrid& proc_grid) : proc_grid_(proc_grid) { }

        ProcessID operator()(const ProcGrid::size_type col) const
        { return proc_grid_.map_col(col); }
      }; // class MapCol

      /// Sparse column group factor function

      /// \param k The column of left shape that will be used to generate the group
      /// \return A sparse group for column \c k of \c left_
      madness::Group make_row_group(const size_type k) const {
        const size_type begin = k * proc_grid_.cols();
        const size_type end = begin + proc_grid_.cols();
        return make_group(right_.shape(), begin, end, right_stride_,
            proc_grid_.proc_cols(), MapCol(proc_grid_), k + k_);
      }


      //------------------------------------------------------------------------
      // General broadcast function


      template <typename ArgShape, typename Datum, typename GenTile>
      void bcast(const ArgShape& shape, std::vector<Datum>& vec,
          size_type index, const size_type end, const size_type stride,
          const ProcessID group_root, const madness::Group& group,
          const GenTile& gen_tile, const size_type key_offset) const
      {
        TA_ASSERT(vec.size() == 0ul);

        if(shape.is_dense()) {
          // Broadcast and store non-zero tiles in the k-th column of the left-hand argument.
          for(size_type i = 0ul; index < end; ++i, index += stride) {
            // Get column tile
            vec.push_back(Datum(i, gen_tile(index)));

            const madness::DistributedID key(TensorImpl_::id(), index + key_offset);
            TensorImpl_::get_world().gop.bcast(key, vec.back().second, group_root, group);
          }
        } else {
          // Broadcast and store non-zero tiles in the k-th column of the left-hand argument.
          for(size_type i = 0ul; index < end; ++i, index += stride) {
            if(shape.is_zero(index)) continue;

            // Get column tile
            vec.push_back(Datum(i, gen_tile(index)));

            const madness::DistributedID key(TensorImpl_::id(), index + key_offset);
            TensorImpl_::get_world().gop.bcast(key, vec.back().second, group_root, group);
          }
        }
      }

      template <typename Datum, typename GenTile>
      void bcast(const SparseShape& shape, std::vector<Datum>& vec,
          size_type index, const size_type end, const size_type stride,
          const ProcessID group_root, const madness::Group& group,
          const GenTile& gen_tile, const size_type key_offset) const
      {
        TA_ASSERT(vec.size() == 0ul);

        // Broadcast and store non-zero tiles in the k-th column of the left-hand argument.
        for(size_type i = 0ul; index < end; ++i, index += stride) {
          if(shape.is_zero(index)) continue;

          // Get column tile
          vec.push_back(Datum(i, gen_tile(index)));

          const madness::DistributedID key(TensorImpl_::id(), index + key_offset);
          TensorImpl_::get_world().gop.bcast(key, vec.back().second, group_root, group);
        }
      }

      template <typename Datum, typename GenTile>
      void bcast(const DenseShape&, std::vector<Datum>& vec,
          size_type index, const size_type end, const size_type stride,
          const ProcessID group_root, const madness::Group& group,
          const size_type key_offset, const GenTile& gen_tile) const
      {
        TA_ASSERT(vec.size() == 0ul);

        // Broadcast and store non-zero tiles in the k-th column of the left-hand argument.
        for(size_type i = 0ul; index < end; ++i, index += stride) {
          // Get column tile
          vec.push_back(Datum(i, gen_tile(index)));

          const madness::DistributedID key(TensorImpl_::id(), index + key_offset);
          TensorImpl_::get_world().gop.bcast(key, vec.back().second, group_root, group);
        }
      }

      template <typename Arg>
      class GenRootTile {
        Arg& arg_;

        template <typename Tile>
        static typename Tile::eval_type convert_task(const Tile& tile) { return tile; }

        template <typename Tile>
        typename madness::disable_if<TiledArray::math::is_lazy_tile<Tile>,
            const madness::Future<Tile>& >::type
        move(const madness::Future<Tile>& tile) { return tile; }

        template <typename Tile>
        typename madness::enable_if< TiledArray::math::is_lazy_tile<Tile>,
            madness::Future<typename Tile::eval_type> >::type
        move(const madness::Future<Tile>& tile) const {
          return arg_.get_world().taskq.add(
              & GenRootTile<Arg>::template convert_task<Tile>,
              tile, madness::TaskAttributes::hipri());
        }

      public:
        GenRootTile(Arg& arg) : arg_(arg) { }

        madness::Future<typename Arg::eval_type> operator()(const size_type index) const {
          madness::Future<typename Arg::eval_type> tile = convert(arg_.move(index));
        }
      };

      template <typename Arg>
      class GenTile {

      public:
        GenTile() { }

        madness::Future<typename Arg::eval_type> operator()(const size_type) const {
          return madness::Future<typename Arg::eval_type>();
        }

      };

      /// Broadcast column \c k of the left-hand argument

      /// \param k The column of \c left_ to be broadcast
      /// \param[out] col The vector that will hold the tiles in column \c k
      template <typename Arg, typename Datum>
      void bcast(Arg& arg, std::vector<Datum>& vec,
          size_type index, const size_type end, const size_type stride,
          const madness::Group& group, const size_type key_offset) const
      {
        TA_ASSERT(vec.size() == 0ul);

        // Get the root process
        const ProcessID group_root = group.rank(arg.owner(index));

        if(group_root == group.rank()) {
          bcast(arg.shape(), vec, index, end, stride, group_root, group,
              key_offset, GenRootTile<Arg>(arg));
        } else {
          bcast(arg.shape(), vec, index, end, stride, group_root, group,
              key_offset, GenTile<Arg>());
        }
      }


      //------------------------------------------------------------------------
      // Broadcast specialization functions

      /// Broadcast column \c k of \c left_

      /// This is a wrapper function around the call to more specialized column
      /// broadcast functions. The possible choices are \c DenseShape,
      /// \c SparseShape, and an arbitrary shape type.
      /// \param[in] k The column of \c left_ to be broadcast
      /// \param[out] col The vector that will hold the results of the broadcast
      void bcast_col(const size_type k, std::vector<col_datum>& col) const {
        // Allocate memory for the column
        col.reserve(proc_grid_.local_rows());

        // Compute local iteration limits for column k of left_.
        const size_type begin = left_begin_local(k);

        // Broadcast column k of left_.
        if(right_.shape().is_dense()) {
          bcast(left_, col, begin, left_end_, left_stride_local_, row_group_, 0ul);
        } else  {
          madness::Group group = make_row_group(k);
          group.register_group();
          bcast(left_, col, begin, left_end_, left_stride_local_, group, 0ul);
          group.unregister_group();
        }
      }

      /// Broadcast row \c k of \c right_

      /// This is a wrapper function around the call to more specialized row
      /// broadcast functions. The possible choices are \c DenseShape,
      /// \c SparseShape, and an arbitrary shape type.
      /// \param[in] k The row of \c right to be broadcast
      /// \param[out] row The vector that will hold the results of the broadcast
      void bcast_row(const size_type k, std::vector<row_datum>& row) const {
        // Allocate memory for the row
        row.reserve(proc_grid_.local_cols());

        // Compute local iteration limits for row k of right_.
        size_type begin = k * proc_grid_.cols();
        const size_type end = begin + proc_grid_.cols();
        begin += proc_grid_.rank_col();

        // Broadcast row k of right_.
        if(left_.shape().is_dense()) {
          bcast(right_, row, begin, end, right_stride_local_, col_group_, left_.size());
        } else {
          madness::Group group = make_col_group(k);
          group.register_group();
          bcast(right_, row, begin, end, right_stride_local_, group, left_.size());
          group.unregister_group();
        }
      }

      /// \note This task while only be called when left and right are sparse.
      void bcast_col_task(size_type k) const {
        // Get the broadcast process group and root process
        madness::Group group = make_sparse_row_group(k);
        group.register_group();

        // Broadcast and store non-zero tiles in the k-th column of the left-hand argument.
        for(size_type index = left_begin_local(k); index < left_end_; index += left_stride_local_) {
          if(left_.is_zero(index)) continue;

          const madness::DistributedID key(TensorImpl_::id(), index);
          TensorImpl_::get_world().gop.bcast(key, move_tile(left_, index), group.rank(), group);
        }

        // Cleanup group
        group.unregister_group();
      }


      void bcast_row_task(size_type k) const {
        TA_ASSERT(right_.is_local(k));

        // Broadcast row k of right.
        std::vector<row_datum> row;
        bcast_row(right_.shape(), k, row);
      }

      void bcast_row_range_task(size_type k, const size_type end) const {
        // Compute the first local row of right
        const size_type nrows = proc_grid_.proc_rows();
        k += (nrows - ((k - proc_grid_.rank_row()) % nrows)) % nrows;

        TA_ASSERT(right_.is_local(k));

        // Broadcast local row k of right.
        std::vector<row_datum> row;
        const col_datum null_value(0ul, left_future::default_initializer());
        for(; k < end; k += nrows) {
          bcast_row(right_.shape(), k, row);
          col.resize(0ul, null_value);
        }
      }

      /// Find next non-zero row of \c right_ for an arbitrary shape type

      /// Starting at the k-th row of the right-hand argument, find the next row
      /// that contains at least one non-zero tile. This search only checks for
      /// non-zero tiles in this processes column.
      /// \tparam S The shape type
      /// \param shape The shape of \c right_
      /// \param k The first row to search
      /// \return The first row, greater than or equal to \c k with non-zero
      /// tiles, or \c k_ if none is found.
      template <typename S>
      size_type iterate_row(const S& shape, size_type k) const {
        if(! shape.is_dense()) {
          // Compute the iterator range for row k
          size_type begin = k * proc_grid_.cols();
          size_type end = begin + proc_grid_.cols();
          begin += proc_grid_.rank_col();

          // Iterate over k's until a non-zero tile is found or the end of the
          // matrix is reached.
          for(; k < k_; ++k, begin += proc_grid_.cols(), end += proc_grid_.cols()) {
            // Search for non-zero tiles in row k of right
            for(size_type i = begin; i < end; i += right_stride_local_)
              if(! shape.is_zero(i))
                return k;
          }
        }

        return k;
      }

      /// Find next non-zero row of \c right_ for a sparse shape

      /// Starting at the k-th row of the right-hand argument, find the next row
      /// that contains at least one non-zero tile. This search only checks for
      /// non-zero tiles in this processes column.
      /// \param shape The shape of \c right_
      /// \param k The first row to search
      /// \return The first row, greater than or equal to \c k with non-zero
      /// tiles, or \c k_ if none is found.
      size_type iterate_row(const SparseShape& shape, size_type k) const {
        // Compute the iterator range for row k
        size_type begin = k * proc_grid_.cols();
        size_type end = begin + proc_grid_.cols();
        begin += proc_grid_.rank_col();

        // Iterate over k's until a non-zero tile is found or the end of the
        // matrix is reached.
        for(; k < k_; ++k, begin += proc_grid_.cols(), end += proc_grid_.cols()) {
          // Search for non-zero tiles in row k of right
          for(size_type i = begin; i < end; i += right_stride_local_)
            if(! shape.is_zero(i))
              return k;
        }

        return k;
      }


      /// Find the next non-zero column of \c left_

      /// Starting at the k-th column of the left-hand argument, find the next
      /// column that contains at least one non-zero tile. This search only
      /// checks for non-zero tiles in this process's row.
      /// \param k The first column to test for non-zero tiles
      /// \return The first column, greater than or equal to \c k, that contains
      /// a non-zero tile. If no non-zero tile is not found, return \c k_.
      template <typename S>
      size_type iterate_col(const S& shape, size_type k) const {
        if(! shape.is_dense()) {
          // Iterate over k's until a non-zero tile is found or the end of the
          // matrix is reached.
          for(; k < k_; ++k) {
            // Search row k for non-zero tiles
            for(size_type i = left_begin_local(k); i < left_end_; i += left_stride_local_)
              if(! shape.is_zero(i))
                return k;
          }
        }

        return k;
      }

      /// Find the next non-zero column of \c left_ for an arbitrary shape type

      /// Starting at the k-th column of the left-hand argument, find the next
      /// column that contains at least one non-zero tile. This search only
      /// checks for non-zero tiles in this process's row.
      /// \param shape The shape of \c left_
      /// \param k The first column to test for non-zero tiles
      /// \return The first column, greater than or equal to \c k, that contains
      /// a non-zero tile. If no non-zero tile is not found, return \c k_.
      size_type iterate_col(const SparseShape& shape, size_type k) const {
        // Iterate over k's until a non-zero tile is found or the end of the
        // matrix is reached.
        for(; k < k_; ++k) {
          // Search row k for non-zero tiles
          for(size_type i = left_begin_local(k); i < left_end_; i += left_stride_local_)
            if(! shape.is_zero(i))
              return k;
        }

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
      size_type iterate(const std::shared_ptr<ContractionEvalImpl_>&,
          const DenseShape&, const DenseShape&, const size_type k) const
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
      template <typename RightShape>
      typename madness::disable_if<std::is_same<RightShape, DenseShape>, size_type>::type
      iterate(const std::shared_ptr<ContractionEvalImpl_>& self, const DenseShape&,
          const RightShape& right_shape, const size_type k) const
      {
        // Find the next non-zero row of right
        const size_type k_row = iterate_row(right_shape, k);

        // Broadcast any local columns of left
        if(k < k_row)
          TensorImpl_::get_world().taskq.add(self,
              & ContractionEvalImpl_::bcast_col_range_task, k, k_row,
              madness::TaskAttributes::hipri());

        return k_row;
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
      template <typename LeftShape>
      typename madness::disable_if<std::is_same<LeftShape, DenseShape>, size_type>::type
      iterate(const std::shared_ptr<ContractionEvalImpl_>& self,
          const LeftShape& left_shape, const DenseShape&, size_type k) const
      {
        // Find the next non-zero column of left
        const size_type k_col = iterate_col(left_shape, k);

        // Broadcast any local columns of left
        if(k < k_col)
          TensorImpl_::get_world().taskq.add(self,
              & ContractionEvalImpl_::bcast_row_range_task, k, k_col,
              madness::TaskAttributes::hipri());

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
      template <typename LeftShape, typename RightShape>
      typename madness::disable_if_c<
          std::is_same<LeftShape, DenseShape>::value ||
          std::is_same<RightShape, DenseShape>::value, size_type>::type
      iterate(const std::shared_ptr<ContractionEvalImpl_>& self,
          const LeftShape& left_shape, const RightShape& right_shape,
          const size_type k) const
      {
        // Initial step for k_col and k_row.
        size_type k_col = iterate_col(left_shape, k);
        size_type k_row = iterate_row(right_shape, k);

        // Search for a row and column that both have non-zero tiles
        while(k_col != k_row) {
          if(k_col < k_row) {
            // If the tiles of k_col are owned by this node, broadcast the tiles.
            if(((k_col - proc_grid_.rank_col()) % proc_grid_.proc_cols()) == 0ul)
              TensorImpl_::get_world().taskq.add(self, & ContractionEvalImpl_::bcast_col_task,
                  k_col, madness::TaskAttributes::hipri());

            // Find the next non-zero column of the left-hand argument
            k_col = iterate_col(left_shape, k_col + 1ul);
          } else {
            // If the tiles of k_row are owned by this node, broadcast the tiles.
            if(((k_row - proc_grid_.rank_row()) % proc_grid_.proc_rows()) == 0ul)
              TensorImpl_::get_world().taskq.add(self, & ContractionEvalImpl_::bcast_row_task,
                  k_row, madness::TaskAttributes::hipri());

            k_row = iterate_row(right_shape, k_row + 1ul);
          }
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
        return iterate(self, left_.shape(), right_.shape(), k);
      }

      /// Destroy reduce tasks and set the result tiles
      void finalize() {
        // Iterate over all local rows and columns
        ReducePairTask<op_type>* reduce_task = reduce_tasks_;
        for(size_type row = proc_grid_.rank_row(); row < proc_grid_.rows(); row += proc_grid_.proc_rows()) {
          const size_type row_start = row * proc_grid_.cols();
          for(size_type col = proc_grid_.rank_col(); col < proc_grid_.cols(); col += proc_grid_.proc_cols(), ++reduce_task) {

            // Compute convert the working ordinal index to a
            const std::size_t index = DistEvalImpl_::perm_index(row_start + col);

            // Construct non-zero reduce tasks
            if(! TensorImpl_::is_zero(index)) {
              // Set the result tile
              DistEvalImpl_::set_tile(index, reduce_task->submit());

              // Destroy the the reduce task
              reduce_task->~ReducePairTask<op_type>();
            }
          }
        }

        // Unregister groups if used.
        if(left_.is_dense())
          col_group_.unregister_group();
        if(right_.is_dense())
          row_group_.unregister_group();

        // Deallocate the memory for the reduce pair tasks.
        std::allocator<ReducePairTask<op_type> >().deallocate(reduce_tasks_, proc_grid_.local_size());
      }

      /// Schedule local contraction tasks for \c col and \c row tile pairs

      /// Schedule tile contractions for each tile pair of \c row and \c col. A
      /// callback to \c task will be registered with each tile contraction
      /// task.
      /// \param col A column of tiles from the left-hand argument
      /// \param row A row of tiles from the right-hand argument
      /// \param callback The callback that will be invoked after each tile-pair
      /// has been contracted
      template <typename S>
#ifndef TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER
      typename madness::disable_if<std::is_same<S, SparseShape> >::type
#else
      void
#endif // TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER
      contract(const size_type, const std::vector<col_datum>& col,
          const std::vector<row_datum>& row, madness::TaskInterface* const task)
      {
        // Iterate over the row
        for(typename std::vector<col_datum>::const_iterator col_it = col.begin(); col_it != col.end(); ++col_it) {
          // Compute the local, result-tile offset
          const size_type offset = col_it->first * proc_grid_.local_cols();

          for(typename std::vector<row_datum>::const_iterator row_it = row.begin(); row_it != row.end(); ++row_it) {
            if(task)
              task->inc();
            reduce_tasks_[offset + row_it->first].add(col_it->second, row_it->second, task);
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
      /// \param k The k step for this contraction set
      /// \param col A column of tiles from the left-hand argument
      /// \param row A row of tiles from the right-hand argument
      /// \param task The task that depends on the tile contraction tasks
      template <typename S>
      typename madness::enable_if<std::is_same<S, SparseShape> >::type
      contract(const size_type k, const std::vector<col_datum>& col,
          const std::vector<row_datum>& row, madness::TaskInterface* const task)
      {
        // Cache row shape data.
        std::vector<float> row_shape_values;
        row_shape_values.reserve(row.size());
        const size_type right_index_base = right_begin_local(k);
        for(typename std::vector<row_datum>::const_iterator row_it = row.begin(); row_it != row.end(); ++row_it)
          row_shape_values.push_back(right_.shape().data()[right_index_base + (row_it->first * right_stride_local_)]);

        // Iterate over the left-hand argument column (rows of the result)
        const size_type left_index_base = left_begin_local(k);
        const float threshold_k = TensorImpl_::shape().threshold() / float(k_);
        for(typename std::vector<col_datum>::const_iterator col_it = col.begin(); col_it != col.end(); ++col_it) {
          // Compute the local, result-tile offset for the current result row.
          const size_type result_offset = col_it->first * proc_grid_.local_cols();

          // Get the shape data for col_it tile
          const float col_shape_value =
              left_.shape().data()[left_index_base + (col_it->first * left_stride_local_)];

          // Iterate over the right-hand argument row (columns of the result)
          for(typename std::vector<row_datum>::const_iterator row_it = row.begin(); row_it != row.end(); ++row_it) {
            // Filter trivial results
            if((col_shape_value * row_shape_values[row_it - row.begin()]) < threshold_k)
              continue;

            if(task)
              task->inc();
            reduce_tasks_[result_offset + row_it->first].add(col_it->second, row_it->second, task);
          }
        }
      }
#endif // TILEDARRAY_DISABLE_TILE_CONTRACTION_FILTER

      class FinalizeTask : public madness::TaskInterface {
      private:
        std::shared_ptr<ContractionEvalImpl_> owner_; ///< The parent object for this task

      public:
        FinalizeTask(const std::shared_ptr<ContractionEvalImpl_>& owner) :
          madness::TaskInterface(1, madness::TaskAttributes::hipri()),
          owner_(owner)
        { }

        virtual ~FinalizeTask() { }

        virtual void run(const madness::TaskThreadEnv&) {
          owner_->finalize();
        }

      }; // class FinalizeTask

      class StepTask : public madness::TaskInterface {
      private:
        // Member variables
        std::shared_ptr<ContractionEvalImpl_> owner_;
        size_type k_;
        FinalizeTask* finalize_task_;
        StepTask* next_step_task_;

        /// Construct the task for the next step
        StepTask(StepTask* const previous, const int ndep) :
          madness::TaskInterface(ndep, madness::TaskAttributes::hipri()),
          owner_(previous->owner_),
          k_(0ul),
          finalize_task_(previous->finalize_task_),
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
              finalize_task_->inc();

              // Initialize and submit next task
              StepTask* const next_next_step_task = next_step_task_->initialize(k_ + 1ul);
              next_step_task_ = NULL;

              // Broadcast row and column
              std::vector<col_datum> col;
              owner_->bcast_col(k_, col);
              std::vector<row_datum> row;
              owner_->bcast_row(k_, row);

              // Submit tasks for the contraction of col and row tiles.
              owner_->template contract<shape_type>(k_, col, row, next_next_step_task);

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
        if(proc_grid_.local_size() != 0ul) {
          // Construct static broadcast groups for dense arguments
          if(left_.is_dense()) {
            col_group_ = proc_grid_.make_col_group(madness::DistributedID(TensorImpl_::id(), 0ul));
            col_group_.register_group();
          }
          if(right_.is_dense()) {
            row_group_ = proc_grid_.make_row_group(madness::DistributedID(TensorImpl_::id(), k_));
            row_group_.register_group();
          }

          // Allocate memory for the reduce pair tasks.
          std::allocator<ReducePairTask<op_type> > alloc;
          reduce_tasks_ = alloc.allocate(proc_grid_.local_size());

          // Iterate over all local rows and columns
          ReducePairTask<op_type>* reduce_task = reduce_tasks_;
          for(size_type row = proc_grid_.rank_row(); row < proc_grid_.rows(); row += proc_grid_.proc_rows()) {
            const size_type row_start = row * proc_grid_.cols();
            for(size_type col = proc_grid_.rank_col(); col < proc_grid_.cols(); col += proc_grid_.proc_cols(), ++reduce_task) {
              // Construct non-zero reduce tasks
              if(! TensorImpl_::is_zero(DistEvalImpl_::perm_index(row_start + col))) {
                new(reduce_task) ReducePairTask<op_type>(TensorImpl_::get_world(), op_);
                ++tile_count;
              }
            }
          }

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
      TA_ASSERT((perm.dim() == op.result_rank()) || (perm.dim() == 0u));

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
        ranges[(perm.dim() > 0ul ? perm[pi++] : pi++)] = left.trange().data()[i];
        M *= left.range().size()[i];
        m *= left.trange().elements().size()[i];
      }
      for(std::size_t i = num_contract_ranks; i < right_end; ++i) {
        ranges[(perm.dim() > 0ul ? perm[pi++] : pi++)] = right.trange().data()[i];
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
