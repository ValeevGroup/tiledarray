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
    /// \tparam Tile The output tile type
    /// \param Policy The tensor policy class
    template <typename Tile, typename Policy>
    class DistEvalImpl : public TensorImpl<Tile, Policy>, public madness::CallbackInterface {
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
      typedef typename value_type::eval_type eval_type; ///< Tile evaluation type

    private:
      const Permutation perm_; ///< The permutation to be applied to this tensor
      typename TensorImpl_::range_type range_; ///< The original tiled range for this tensor
      std::vector<size_type> ip_weight_; ///< The inverse permuted weight of the result range

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
      size_type perm_index(size_type index) const {
        size_type result_index;
        if(perm_) {
          result_index = 0ul;
          // Permute the index
          for(size_type i = 0ul; i < TensorImpl_::range().dim(); ++i) {
            result_index += (index / range_.weight()[i]) * ip_weight_[i];
            index %= range_.weight()[i];
          }
        } else {
          // Return the unmodified index if no permutation needs to be applied
          result_index = index;
        }
        return result_index;
      }

      /// Permutation accessor

      /// \return A const reference to the permutation
      const Permutation& perm() const { return perm_; }

    public:
      /// Constructor

      /// \param world The world where the tensor lives
      /// \param perm The permutation that is applied to the result tensor
      /// \param trange The unpermuted tiled range object
      /// \param shape The tensor shape bitset [ Default = 0 size bitset ]
      /// \note \c trange and \c shape will be permuted by \c perm before
      /// storing the data.
      DistEvalImpl(madness::World& world, const trange_type& trange,
          const shape_type& shape, const std::shared_ptr<pmap_interface>& pmap,
          const Permutation& perm) :
        TensorImpl_(world, trange, shape, pmap),
        perm_(perm),
        range_(),
        ip_weight_(),
        task_count_(-1),
        set_counter_()
      {
        set_counter_ = 0;

        if(perm) {
          Permutation inv_perm(-perm);
          range_ = inv_perm ^ trange.tiles();
          ip_weight_ = inv_perm ^ TensorImpl_::range().weight();
        }
      }

      virtual ~DistEvalImpl() { }

      /// Set tensor value

      /// This will store \c value at ordinal index \c i . Typically, this
      /// function should be called by a task function.
      /// \param i The index in the result space where value will be stored
      /// \param value The value to be stored at index \c i
      void set_tile(size_type i, const value_type& value) {
        // Store value
        TensorImpl_::set(i, value);

        // Record the assignment of a tile
        DistEvalImpl::notify();
      }

      /// Set tensor value with a future

      /// This will store \c value at ordinal index \c i . Typically, this
      /// function should be called by a task function.
      /// \param i The index in the result space where value will be stored
      /// \param value The future value to be stored at index \c i
      void set_tile(size_type i, madness::Future<value_type> f) {
        // Store value
        TensorImpl_::set(i, f);

        // Record the assignment of a tile
        f.register_callback(this);
      }

      /// Tile set notification
      virtual void notify() { set_counter_++; }

      /// Wait for all tiles to be assigned
      void wait() const {
        TA_ASSERT(task_count_ >= 0);
        CounterProbe probe(set_counter_, task_count_);
        TensorImpl_::get_world().await(probe);
      }

    private:

      /// Evaluate the tiles of this tensor

      /// This function will evaluate the children of this distributed evaluator
      /// and evaluate the tiles for this distributed evaluator. It will block
      /// until the tasks for the children are evaluated (not for the tasks of
      /// this object).
      /// \param pimpl A shared pointer to this object
      /// \return The number of tiles that will be set by this process
      virtual int internal_eval(const std::shared_ptr<DistEvalImpl_>& pimpl) = 0;

    public:

      /// Evaluate this tensor expression object

      /// This function will evaluate the children of this distributed evaluator
      /// and evaluate the tiles for this distributed evaluator. It will block
      /// until the tasks for the children are evaluated (not for the tasks of
      /// this object).
      /// \param pimpl A shared pointer to this object
      void eval(const std::shared_ptr<DistEvalImpl_>& pimpl) {
        TA_ASSERT(task_count_ == -1);
        TA_ASSERT(this == pimpl.get());
        task_count_ = this->internal_eval(pimpl);
        TA_ASSERT(task_count_ >= 0);
      }

    }; // class DistEvalImpl


    /// Tensor expression object

    /// This object holds a tensor expression. It is used to store various type
    /// of tensor expressions that depend on the pimpl used to construct the
    /// expression.
    /// \tparam Tile The output tile type
    /// \tparam Policy The tensor policy class
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
      typedef typename impl_type::eval_type eval_type; ///< Tile evaluation type
      typedef typename impl_type::future future; ///< Future of tile type

    private:
      std::shared_ptr<impl_type> pimpl_; ///< pointer to the implementation object

    public:
      /// Constructor

      /// \param pimpl A pointer to the expression implementation object
      DistEval(const std::shared_ptr<impl_type>& pimpl) :
        pimpl_(pimpl)
      {
        TA_ASSERT(pimpl_);
      }

      /// Copy constructor

      /// Create a shallow copy of \c other .
      /// \param other The object to be copied.
      DistEval(const DistEval_& other) : pimpl_(other.pimpl_) { }

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
      void eval() { return pimpl_->eval(pimpl_); }


      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      const range_type& range() const { return pimpl_->range(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      size_type size() const { return pimpl_->size(); }

      /// Query a tile owner

      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      ProcessID owner(size_type i) const { return pimpl_->owner(i); }

      /// Query for a locally owned tile

      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      bool is_local(size_type i) const { return pimpl_->is_local(i); }

      /// Query for a zero tile

      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      bool is_zero(size_type i) const { return pimpl_->is_zero(i); }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      const std::shared_ptr<pmap_interface>& pmap() const { return pimpl_->pmap(); }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      bool is_dense() const { return pimpl_->is_dense(); }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      const shape_type& shape() const { return pimpl_->shape(); }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const { return pimpl_->trange(); }

      /// Tile move

      /// Tile is removed after it is set.
      /// \param i The tile index
      /// \return Tile \c i
      future move(size_type i) const { return pimpl_->move(i); }

      /// World object accessor

      /// \return A reference to the world object
      madness::World& get_world() const { return pimpl_->get_world(); }

      /// Unique object id

      /// \return The unique id for this object
      madness::uniqueidT id() const { return pimpl_->id(); }

      /// Wait for all local tiles to be evaluated
      void wait() const { pimpl_->wait(); }

    }; // class DistEval

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_DIST_EVAL_DIST_EVAL_BASE_H__INCLUDED
