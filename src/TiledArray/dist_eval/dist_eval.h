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

#ifndef TILEDARRAY_DIST_EVAL_DIST_EVAL_BASE_H__INCLUDED
#define TILEDARRAY_DIST_EVAL_DIST_EVAL_BASE_H__INCLUDED

#include <TiledArray/tensor_impl.h>
#include <TiledArray/counter_probe.h>
#include <TiledArray/tile_op/permute.h>

namespace TiledArray {
  namespace detail {

    /// Distributed evaluator implementation object

    /// This class is used as the base class for other distributed evaluation
    /// implementation classes. It has several pure virtual function that are
    /// used by derived classes to implement the distributed evaluate. This
    /// class can also handles permutation of result tiles if necessary.
    /// \tparam Policy The policy type
    template <typename Tile, typename Policy>
    class DistEvalImpl : public TensorImpl<Tile, Policy> {
    public:
      typedef DistEvalImpl<Tile, Policy> DistEvalImpl_; ///< This object type
      typedef TiledArray::detail::TensorImpl<Tile, Policy> TensorImpl_;
                                          ///< Tensor implementation base class

      typedef typename TensorImpl_::size_type size_type; ///< Size type
      typedef typename TensorImpl_::trange_type trange_type; ///< Tiled range type for this object
      typedef typename TensorImpl_::range_type range_type; ///< Range type this tensor
      typedef typename TensorImpl_::shape_type shape_type; ///< Shape type
      typedef typename TensorImpl_::pmap_interface pmap_interface; ///< process map interface type
      typedef typename TensorImpl_::value_type value_type; ///< Tile type

    private:
      const Permutation perm_; ///< The permutation to be applied to this tensor
      const typename TensorImpl_::range_type range_; ///< The original tiled range for this tensor

      // The following variables are used to track the total number of tasks run
      // on the local node, task_count_, and the number of tiles set on this
      // node, set_counter_. They are used to track the progress of work done by
      // this node, which allows us to wait for the completion of these tasks
      // without waiting for all tasks.

      volatile int task_count_; ///< Total number of local tasks
      madness::AtomicInt set_counter_; ///< The number of tiles set by this node

    protected:

      /// Map an index value from the unpermuted index space to the permuted index space

      /// \param i The index in the unpermuted index space
      /// \return The corresponding index in the permuted index space
      size_type perm_index(size_type i) const {
        return (perm_.dim() ? TensorImpl_::range().ord(perm_ ^ range_.idx(i)) : i);
      }

      /// Permutation accessor

      /// \return A const reference to the permutation
      const Permutation& perm() const { return perm_; }

    public:
      /// Constructor

      /// \param world The world where the tensor lives
      /// \param perm The permutation that is applied to the result tensor
      /// \param trange The tiled range object
      /// \param shape The tensor shape bitset [ Default = 0 size bitset ]
      /// \note \c trange and \c shape will be permuted by \c perm before
      /// storing the data.
      DistEvalImpl(madness::World& world, const Permutation& perm,
          const trange_type& trange, const shape_type& shape,
          const std::shared_ptr<pmap_interface>& pmap) :
        TensorImpl_(world, (perm.dim() ? perm ^ trange : trange), shape, pmap),
        perm_(perm),
        range_(trange.tiles()),
        task_count_(0),
        set_counter_()
      {
        set_counter_ = 0;
      }

      virtual ~DistEvalImpl() { }

      /// Set tensor value

      /// This will store \c value at ordinal index \c i . Typically, this
      /// function should be called by a task function.
      /// \param i The index where value will be stored.
      /// \param value The value or future value to be stored at index \c i
      /// \note The index \c i and \c value may be permuted by this function
      /// before storing the value.
      void set_tile(size_type i, const value_type& value) {
        // Store value
        TensorImpl_::set(i, value);

        // Record the assignment of a tile
        set_counter_++;
      }

      /// Wait for all tiles to be assigned
      void wait() const {
        CounterProbe probe(set_counter_, task_count_);
        TensorImpl_::get_world().await(probe);
      }

    private:

      /// Function for evaluating child tensors
      virtual void eval_children() = 0;

      /// Function for evaluating this tensor's tiles

      /// This function is run inside a task, and will run after \c eval_children
      /// has completed. It should spawn additional tasks that evaluate the
      /// individual result tiles.
      /// \param pimpl A shared pointer to this object
      virtual size_type eval_tiles(const std::shared_ptr<DistEvalImpl>& pimpl) = 0;

      /// Wait for tasks of children to finish
      virtual void wait_children() const = 0;

    public:

      /// Evaluate this tensor expression object

      /// This function will evaluate the children of this distributed evaluator
      /// and evaluate the tiles for this distributed evaluator. It will block
      /// until the tasks for the children are evaluated (not for the tasks of
      /// this object).
      /// \param pimpl A shared pointer to this object
      void eval(std::shared_ptr<DistEvalImpl> pimpl) {
        // Evaluate children
        this->eval_children();

        // Evaluate tiles for this object
        task_count_ = this->eval_tiles(pimpl);

        // Wait until the children tasks are complete. Tasks will be processed
        // by this thread while waiting. We block here to throttle the number
        // of simultaneous tasks and evaluations.
        this->wait_children();
      }

    }; // class DistEvalImpl


    /// Tensor expression object

    /// This object holds a tensor expression. It is used to store various type
    /// of tensor expressions that depend on the pimpl used to construct the
    /// expression.
    /// \tparam Policy The policy type
    template <typename Tile, typename Policy>
    class DistEval {
    public:
      typedef DistEval<Tile, Policy> DistEval_; ///< This class type
      typedef DistEvalImpl<Tile, Policy> impl_type; ///< Implementation base class type
      typedef typename impl_type::size_type size_type; ///< Size type
      typedef typename impl_type::trange_type trange_type; ///< Tiled range type for this object
      typedef typename impl_type::range_type range_type; ///< Range type this tensor
      typedef typename impl_type::shape_type shape_type; ///< Tensor shape type
      typedef typename impl_type::pmap_interface pmap_interface; ///< Process map interface type
      typedef typename impl_type::value_type value_type; ///< Tile type
      typedef typename impl_type::future future; ///< Future of tile type

      /// Constructor

      /// \param pimpl A pointer to the expression implementation object
      DistEval(const std::shared_ptr<impl_type>& pimpl) : pimpl_(pimpl) { }

      /// Copy constructor

      /// Create a shallow copy of \c other .
      /// \param other The object to be copied.
      DistEval(const DistEval_& other) :
          pimpl_(other.pimpl_)
      { }

      /// Assignment operator

      /// Create a shallow copy of \c other
      /// \param other The object to be copied
      /// \return A reference to this object
      DistEval_& operator=(const DistEval_& other) {
        pimpl_ = other.pimpl_;
        return *this;
      }

      /// Evaluate this tensor object with the given result variable list

      /// \c v is the dimension ordering that the parent expression expects.
      /// The returned future will be evaluated once the tensor has been evaluated.
      void eval() {
        TA_ASSERT(pimpl_);
        return pimpl_->eval(pimpl_);
      }


      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const range_type& range() const {
        TA_ASSERT(pimpl_);
        return pimpl_->range();
      }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const {
        TA_ASSERT(pimpl_);
        return pimpl_->size();
      }

      /// Query a tile owner

      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      ProcessID owner(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->owner(i);
      }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      bool is_local(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->is_local(i);
      }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      bool is_zero(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->is_zero(i);
      }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      const std::shared_ptr<pmap_interface>& pmap() const {
        TA_ASSERT(pimpl_);
        return pimpl_->pmap();
      }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const {
        TA_ASSERT(pimpl_);
        return pimpl_->is_dense();
      }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const shape_type& shape() const {
        TA_ASSERT(pimpl_);
        return pimpl_->shape();
      }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const {
        TA_ASSERT(pimpl_);
        return pimpl_->trange();
      }

      /// Tile move

      /// Tile is removed after it is set.
      /// \param i The tile index
      /// \return Tile \c i
      future move(size_type i) const {
        TA_ASSERT(pimpl_);
        return pimpl_->move(i);
      }

      /// World object accessor

      /// \return A reference to the world object
      madness::World& get_world() const {
        TA_ASSERT(pimpl_);
        return pimpl_->get_world();
      }

      /// Release tensor data

      /// Clear all tensor data from memory. This is equivalent to
      /// \c UnaryTiledTensor().swap(*this) .
      void release() { pimpl_.reset(); }

    protected:
      std::shared_ptr<impl_type> pimpl_; ///< pointer to the implementation object
    }; // class DistEval

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_DIST_EVAL_DIST_EVAL_BASE_H__INCLUDED
