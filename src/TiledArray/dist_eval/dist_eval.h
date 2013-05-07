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

namespace TiledArray {
  namespace detail {

    /// Distributed evaluator implementation object

    /// This class is used as the base class for other distributed evaluation
    /// implementation classes. It has several pure virtual function that are
    /// used by derived classes to implement the distributed evaluate. This
    /// class can also handles permutation of result tiles if necessary.
    /// \tparam Tile The result tile type for the expression.
    template <typename Tile>
    class DistEvalImpl : public TensorImpl<Tile> {
    public:
      typedef DistEvalImpl<Tile> DistEvalImpl_; ///< This object type
      typedef TiledArray::detail::TensorImpl<Tile> TensorImpl_; ///< Tensor implementation base class

      typedef typename TensorImpl_::size_type size_type; ///< Size type
      typedef typename TensorImpl_::trange_type trange_type; ///< Tiled range type for this object
      typedef typename TensorImpl_::range_type range_type; ///< Range type this tensor
      typedef typename TensorImpl_::shape_type shape_type; ///< Shape type
      typedef typename TensorImpl_::pmap_interface pmap_interface; ///< process map interface type
      typedef typename TensorImpl_::value_type value_type; ///< Tile type
      typedef typename TensorImpl_::numeric_type numeric_type;  ///< the numeric type that supports Tile

    private:
      const Permutation perm_; ///< The permutation to be applied to this tensor
      const typename TensorImpl_::range_type range_; ///< The original tiled range for this tensor
      const bool permute_tiles_;

      /// Task function for permuting result tensor

      /// \param value The unpermuted result tile
      /// \return The permuted result tile
      void permute_and_set_with_value(const size_type index, const value_type& value) {
        // Create tensor to hold the result
        value_type result(perm_ ^ value.range());

        // Construct the inverse permuted weight and size for this tensor
        std::vector<std::size_t> ip_weight = (-perm_) ^ result.range().weight();
        const typename value_type::range_type::size_array& start = value.range().start();

        // Coordinated iterator for the value range
        typename value_type::range_type::const_iterator value_range_it =
            value.range().begin();

        // permute the data
        for(typename value_type::const_iterator value_it = value.begin(); value_it != value.end(); ++value_it, ++value_range_it)
          result[TiledArray::detail::calc_ordinal(*value_range_it, ip_weight, start)] = *value_it;

        // Store the permuted tensor
        TensorImpl_::set(permute_index(index)), result);
      }

      /// Permute and set tile \c i with \c value

      /// If the \c value has been set, then the tensor is permuted and set
      /// immediately. Otherwise a task is spawned that will permute and set it.
      /// \param i The unpermuted index of the tile
      /// \param value The future that holds the unpermuted result tile
      void permute_and_set(size_type i, const value_type& value) {
        permute_and_set_with_value(i, value);
      }

      /// Permute and set tile \c i with \c value

      /// If the \c value has been set, then the tensor is permuted and set
      /// immediately. Otherwise a task is spawn that will permute and set it.
      /// \param i The unpermuted index of the tile
      /// \param value The future that holds the unpermuted result tile
      void permute_and_set(size_type i, const madness::Future<value_type>& value) {
        if(value.probe())
          permute_and_set_with_value(i, value.get());
        else
          TensorImpl_::get_world().taskq.add(*this,
              & DistEvalImpl_::permute_and_set_with_value, i, value);
      }

    protected:

      /// Map an index value from the unpermuted index space to the permuted index space

      /// \param i The index in the unpermuted index space
      /// \return The corresponding index in the permuted index space
      size_type perm_index(size_type i) const {
        return (perm_.dim() ? TensorImpl_::range().ord(perm_ ^ range_.idx(i)) : i);
      }

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
          const std::shared_ptr<pmap_interface>& pmap, const bool permute_tiles) :
        TensorImpl_(world, (perm.dim() ? perm ^ trange : trange), 0),
        range_(trange.tiles()),
        perm_(perm),
        permute_tiles_(perm_tiles)
      {
        if(shape.size()) {
          if(perm.dim()) {
            // Set the shape with a permuted shape.

            // Construct the inverse permuted weight and size for this tensor
            std::vector<std::size_t> ip_weight =
                (-perm_) ^ TensorImpl_::trange().tiles().weight();
            const typename range_type::size_array& start = range_.start();

            // Construct a shape
            TensorImpl_::shape(shape_type(range_.size()));

            // Set the permuted shape
            typename range_type::const_iterator range_it = range_.begin();
            for(size_type i = 0ul; i < size; ++i, ++range_it)
              if(shape[i])
                TensorImpl_::shape(TiledArray::detail::calc_ordinal(*range_it,
                    ip_weight, start), true);
          } else {
            TensorImpl_::shape(shape);
          }
        }

        TensorImpl_::pmap(pmap);
      }

      virtual ~DistEvalImpl() { }

      /// Set tensor value

      /// This will store \c value at ordinal index \c i . The tile will be
      /// permuted if necessary. Typically this function should be called by
      /// \c eval_tiles() or there in.
      /// \tparam Value The value type, either \c value_type or \c madness::Future<value_type>
      /// \param i The index where value will be stored.
      /// \param value The value or future value to be stored at index \c i
      /// \param perm If true, value will be permuted. [default = true]
      /// \note The index \c i and \c value will be permuted by this function
      /// before storing the value.
      template <typename Value>
      void set(size_type i, const Value& value) {
        if(perm_.dim()) {
          if(permute_tiles_)
            permute_and_set(i, value);
          else
            TensorImpl_::set(perm_index(i), value);
        } else
          TensorImpl_::set(i, value);
      }

      /// Permute a tensor

      /// \param result The tensor that will hold the permuted result
      /// \param value The unpermuted tensor
      void permute(value_type& result, const value_type& value) const {
        if(perm_.dim()) {
        }
      }

    private:

      /// Function for evaluating this tensor's tiles

      /// This function is run inside a task, and will run after \c eval_children
      /// has completed. It should spawn additional tasks that evaluate the
      /// individual result tiles.
      /// \param counter An atomic counter that to track the number of completed
      /// tasks.
      /// \param task_counter Counter for the total number of tasks generated
      /// by this evaluator.
      virtual void eval_tiles(const std::shared_ptr<DistEvalImpl>& pimpl,
          madness::AtomicInt& counter, int& task_count) = 0;

      /// Function for evaluating child tensors

      /// This function should return true when the child

      /// This function should evaluate all child tensors.
      /// \param counter An atomic counter that to track the number of completed
      /// tasks.
      /// \param task_counter Counter for the total number of tasks generated
      /// by children evaluators.
      virtual void eval_children(madness::AtomicInt& counter, int& task_count) = 0;

    public:

      /// Evaluate this tensor expression object

      /// This function will evaluate the children of this distributed evaluator
      /// and evaluate the tiles for this distributed evaluator. It will block
      /// until the tasks for the children are evaluated (not for the tasks of
      /// this object).
      /// \param counter An atomic counter that to track the number of completed
      /// tasks.
      /// \param task_counter Counter for the total number of tasks generated
      /// by this evaluator.
      void eval(std::shared_ptr<DistEvalImpl> pimpl, madness::AtomicInt& counter,
          int& task_count)
      {
        // Children eval counter
        madness::AtomicInt children_counter;
        children_counter = 0;
        int children_task_count = 0;

        // Evaluate children
        this->eval_children(children_counter, children_task_count);

        // Evaluate tiles for this object
        this->eval_tiles(pimpl, counter, task_count);

        // Wait until the children tasks are complete. Tasks will be processed
        // by this thread while waiting. We block here to throttle the number
        // of simultaneous tasks and evaluations.
        CounterProbe probe(children_counter, children_task_count);
        TensorImpl_::get_world().await(probe);
      }

    }; // class DistEvalImpl


    /// Tensor expression object

    /// This object holds a tensor expression. It is used to store various type
    /// of tensor expressions that depend on the pimpl used to construct the
    /// expression.
    /// \tparam Tile The expression tile type
    template <typename Tile>
    class DistEval {
    public:
      typedef DistEval<Tile> DistEval_;
      typedef DistEvalImpl<Tile> impl_type;
      typedef typename impl_type::size_type size_type;
      typedef typename impl_type::range_type range_type;
      typedef typename impl_type::shape_type shape_type;
      typedef typename impl_type::pmap_interface pmap_interface;
      typedef typename impl_type::trange_type trange_type;
      typedef typename impl_type::value_type value_type;
      typedef typename impl_type::numeric_type numeric_type;
      typedef typename impl_type::const_reference const_reference;
      typedef typename impl_type::const_iterator const_iterator;

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
      /// \param v The expected data layout of this tensor.
      /// \return A Future bool that will be assigned once this tensor has been
      /// evaluated.
      madness::Future<bool> eval(madness::AtomicInt& counter, int& task_count) {
        TA_ASSERT(pimpl_);
        return pimpl_->eval(pimpl_, counter, task_count);
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
      const shape_type shape() const {
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
      const_reference move(size_type i) const {
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
