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

#include <TiledArray/config.h>
#include <TiledArray/perm_index.h>
#include <TiledArray/permutation.h>
#include <TiledArray/tensor_impl.h>
#include <TiledArray/type_traits.h>
#ifdef TILEDARRAY_HAS_DEVICE
#include <TiledArray/device/device_task_fn.h>
#include <TiledArray/external/device.h>
#endif

namespace TiledArray {
namespace detail {

/// Distributed evaluator implementation object

/// This class is used as the base class for other distributed evaluation
/// implementation classes. It has several pure virtual function that are
/// used by derived classes to implement the distributed evaluate. This
/// class can also handles permutation of result tiles if necessary.
/// \tparam Tile The output tile type
/// \tparam Policy The tensor policy class
template <typename Tile, typename Policy>
class DistEvalImpl : public TensorImpl<Policy>,
                     public madness::CallbackInterface {
 public:
  typedef DistEvalImpl<Tile, Policy> DistEvalImpl_;  ///< This object type
  typedef TiledArray::detail::TensorImpl<Policy> TensorImpl_;
  ///< Tensor implementation base class

  typedef typename TensorImpl_::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename TensorImpl_::trange_type
      trange_type;  ///< Tiled range type for this object
  typedef
      typename TensorImpl_::range_type range_type;  ///< Range type this tensor
  typedef typename TensorImpl_::shape_type shape_type;  ///< Shape type
  typedef typename TensorImpl_::pmap_interface
      pmap_interface;       ///< process map interface type
  typedef Tile value_type;  ///< Tile type
  typedef typename eval_trait<value_type>::type
      eval_type;  ///< Tile evaluation type

 private:
  madness::uniqueidT id_;       ///< Globally unique object identifier.
  PermIndex source_to_target_;  ///< Functor used to permute a source index to a
                                ///< target index.
  PermIndex target_to_source_;  ///< Functor used to permute a target index to a
                                ///< source index.

  // The following variables are used to track the total number of tasks run
  // on the local node, task_count_, and the number of tiles set on this
  // node, set_counter_. They are used to track the progress of work done by
  // this node, which allows us to wait for the completion of these tasks
  // without waiting for all tasks.

  volatile int task_count_;         ///< Total number of local tasks
  madness::AtomicInt set_counter_;  ///< The number of tiles set by this node

 protected:
  /// Permute \c index from a source index to a target index

  /// \param index An ordinal index in the source index space
  /// \return The ordinal index in the target index space
  ordinal_type perm_index_to_target(ordinal_type index) const {
    TA_ASSERT(index < TensorImpl_::trange().tiles_range().volume());
    return (source_to_target_ ? source_to_target_(index) : index);
  }

  /// Permute \c index from a target index to a source index

  /// \param index An ordinal index in the target index space
  /// \return The ordinal index in the source index space
  ordinal_type perm_index_to_source(ordinal_type index) const {
    TA_ASSERT(index < TensorImpl_::trange().tiles_range().volume());
    return (target_to_source_ ? target_to_source_(index) : index);
  }

 public:
  /// Constructor

  /// \param world The world where the tensor lives
  /// \param trange The tiled range object
  /// \param shape The tensor shape object
  /// \param pmap The tile-process map
  /// \param perm The permutation that is applied to tile indices
  /// \note \c trange and \c shape will be permuted by \c perm before
  /// storing the data.
  DistEvalImpl(World& world, const trange_type& trange, const shape_type& shape,
               const std::shared_ptr<const pmap_interface>& pmap,
               const Permutation& perm)
      : TensorImpl_(world, trange, shape, pmap),
        id_(world.make_unique_obj_id()),
        source_to_target_(),
        target_to_source_(),
        task_count_(-1),
        set_counter_() {
    set_counter_ = 0;

    if (perm) {
      Permutation inv_perm(-perm);
      range_type source_range = inv_perm * trange.tiles_range();
      source_to_target_ = PermIndex(source_range, perm);
      target_to_source_ = PermIndex(trange.tiles_range(), inv_perm);
    }

#if 0
    {
      // print out expected number of tiles on each rank
      std::vector<size_t> ntiles_per_rank(world.size(), 0);
      for (auto& i : trange.tiles_range()) {
        if (!TensorImpl_::is_zero(i)) {
          ntiles_per_rank[TensorImpl_::owner(i)]++;
        }
      }
      std::stringstream ss;
      ss << "DistEvalImpl: id=" << id_;
      if (perm)
        ss << " perm=" << perm;
      ss << " ntiles=[";
      for (auto& i : ntiles_per_rank) {
        ss << i << " ";
      }
      ss << "]";
      std::cout << ss.str() << std::endl;
    }
#endif
  }

  virtual ~DistEvalImpl() {}

  /// Unique object id accessor

  /// \return This object's unique identifier
  const madness::uniqueidT& id() const { return id_; }

  /// Get tile at index \c i

  /// \param i The index of the tile
  /// \return Tile at index i
  virtual Future<value_type> get_tile(ordinal_type i) const = 0;

  /// Discard a tile that is not needed

  /// This function handles the cleanup for tiles that are not needed in
  /// subsequent computation.
  /// \param i The index of the local tile to discard
  /// \pre `this->is_local(i)`
  virtual void discard_tile(ordinal_type i) const = 0;

  /// Set tensor value

  /// This will store \c value at ordinal index \c i . Typically, this
  /// function should be called by a task function.
  /// \param i The index in the result space where value will be stored
  /// \param value The value to be stored at index \c i
  void set_tile(ordinal_type i, const value_type& value) {
    // Store value
    madness::DistributedID id(id_, i);
    TensorImpl_::world().gop.send(TensorImpl_::owner(i), id, value);

    // Record the assignment of a tile
    DistEvalImpl_::notify();
  }

  /// Set tensor value with a future

  /// This will store \c value at ordinal index \c i . Typically, this
  /// function should be called by a task function.
  /// \param i The index in the result space where value will be stored
  /// \param f The future value to be stored at index \c i
  void set_tile(ordinal_type i, Future<value_type> f) {
    // Store value
    madness::DistributedID id(id_, i);
    TensorImpl_::world().gop.send(TensorImpl_::owner(i), id, f);

    // Record the assignment of a tile
    f.register_callback(this);
  }

  /// Tile set notification
  void notify() override { set_counter_++; }

  /// Wait for all tiles to be assigned
  void wait() const {
    const int task_count = task_count_;
    if (task_count > 0) {
      auto report_and_abort = [&, this](const char* type,
                                        const char* what = nullptr) {
        std::stringstream ss;
        ss << "!! ERROR TiledArray: Aborting due to " << type << " exception.\n"
           << (what != nullptr ? "!! ERROR TiledArray: " : "")
           << (what != nullptr ? what : "") << (what != nullptr ? "\n" : "")
           << "!! ERROR TiledArray: rank=" << TensorImpl_::world().rank()
           << " id=" << id_ << " " << set_counter_ << " of " << task_count
           << " tiles set" << std::endl;
        std::cerr << ss.str().c_str();
        abort();
      };
      try {
        TensorImpl_::world().await(
            [this, task_count]() { return this->set_counter_ == task_count; });
      } catch (TiledArray::Exception& e) {
        report_and_abort("TiledArray", e.what());
      } catch (madness::MadnessException& e) {
        report_and_abort("MADNESS", e.what());
      } catch (SafeMPI::Exception& e) {
        report_and_abort("SafeMPI", e.what());
      } catch (std::exception& e) {
        report_and_abort("std", e.what());
      } catch (...) {
        report_and_abort("", nullptr);
      }
    }
  }

 private:
  /// Evaluate the tiles of this tensor

  /// This function will evaluate the children of this distributed evaluator
  /// and evaluate the tiles for this distributed evaluator. It will block
  /// until the tasks for the children are evaluated (not for the tasks of
  /// this object).
  /// \return The number of tiles that will be set by this process
  virtual int internal_eval() = 0;

 public:
  /// Evaluate this tensor expression object

  /// This function will evaluate the children of this distributed evaluator
  /// and evaluate the tiles for this distributed evaluator. It will block
  /// until the tasks for the children are evaluated (not for the tasks of
  /// this object).
  void eval() {
    TA_ASSERT(task_count_ == -1);
    task_count_ = this->internal_eval();
    TA_ASSERT(task_count_ >= 0);
  }

  /// \return The number of tasks spawned on this rank (after invoking eval()
  /// this should be equal to local_nnz() for simple evaluators like
  /// unary/binary, or greater than that for more complex evaluators like SUMMA
  ordinal_type task_count() const {
    if (task_count_ == -1)
      return 0;
    else
      return task_count_;
  }

#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
  /// reports evaluator status

  /// intended for debugging purposes
  /// @return string containing log of the current status of evaluator (empty
  /// string, unless overridden in the specialization)
  [[nodiscard]] virtual std::string status() const { return {}; }
#endif
};  // class DistEvalImpl

/// Tensor expression evaluator wrapper

/// This object holds a tensor expression evaluator (DistEvalImpl).
///
/// \note Tensor expression evaluators (DistEval and DistEvalImpl)
/// are similar to DistArray in that they has tensorial structure
/// (TensorImpl), with shape and policy, but their semantics that
/// differs from DistArray (e.g., data is not stored
/// persistently).
///
/// \tparam Tile The output tile type
/// \tparam Policy The tensor policy class
template <typename Tile, typename Policy>
class DistEval {
 public:
  typedef DistEval<Tile, Policy> DistEval_;  ///< This class type
  typedef DistEvalImpl<Tile, Policy>
      impl_type;  ///< Implementation base class type
  typedef typename impl_type::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename impl_type::trange_type
      trange_type;  ///< Tiled range type for this object
  typedef
      typename impl_type::range_type range_type;  ///< Range type this tensor
  typedef typename impl_type::shape_type shape_type;  ///< Tensor shape type
  typedef typename impl_type::pmap_interface
      pmap_interface;  ///< Process map interface type
  typedef typename impl_type::value_type value_type;  ///< Tile type
  typedef typename impl_type::eval_type eval_type;    ///< Tile evaluation type
  typedef Future<value_type> future;                  ///< Future of tile type

 private:
  std::shared_ptr<impl_type> pimpl_;  ///< pointer to the implementation object

 public:
  /// Constructor

  /// \param pimpl A pointer to the expression implementation object
  template <typename Impl>
  DistEval(const std::shared_ptr<Impl>& pimpl)
      : pimpl_(std::static_pointer_cast<impl_type>(pimpl)) {
    TA_ASSERT(pimpl_);
  }

  /// Copy constructor

  /// Create a shallow copy of \c other .
  /// \param other The object to be copied.
  DistEval(const DistEval_& other) : pimpl_(other.pimpl_) {}

  /// Assignment operator

  /// Create a shallow copy of \c other
  /// \param other The object to be copied
  /// \return A reference to this object
  DistEval_& operator=(const DistEval_& other) {
    pimpl_ = other.pimpl_;
    return *this;
  }

  /// Evaluate this object

  /// This function will evaluate the children of this distributed evaluator
  /// and evaluate the tiles for this distributed evaluator. It will block
  /// until the tasks for the children are evaluated (not for the tasks of
  /// this object).
  void eval() { pimpl_->eval(); }

  /// Tensor tile size array accessor

  /// \return The size array of the tensor tiles
  const range_type& range() const { return pimpl_->tiles_range(); }

  /// Tensor tile volume accessor

  /// \return The number of tiles in the tensor
  ordinal_type size() const { return pimpl_->size(); }

  /// Query a tile owner

  /// \param i The tile index to query
  /// \return The process ID of the node that owns tile \c i
  ProcessID owner(ordinal_type i) const { return pimpl_->owner(i); }

  /// Query for a locally owned tile

  /// \param i The tile index to query
  /// \return \c true if the tile is owned by this node, otherwise \c false
  bool is_local(ordinal_type i) const { return pimpl_->is_local(i); }

  /// Query for a zero tile

  /// \param i The tile index to query
  /// \return \c true if the tile is zero, otherwise \c false
  bool is_zero(ordinal_type i) const { return pimpl_->is_zero(i); }

  /// Tensor process map accessor

  /// \return A shared pointer to the process map of this tensor
  const std::shared_ptr<const pmap_interface>& pmap() const {
    return pimpl_->pmap();
  }

  /// Query if the tensor is dense

  /// \return \c true if the tensor is dense, otherwise false
  bool is_dense() const { return pimpl_->is_dense(); }

  /// Tensor shape accessor

  /// \return A reference to the tensor shape map
  const shape_type& shape() const { return pimpl_->shape(); }

  /// Tiled range accessor

  /// \return The tiled range of the tensor
  const trange_type& trange() const { return pimpl_->trange(); }

  /// Tile accessor

  /// Tile is removed after it is set.
  /// \param i The tile index
  /// \return Tile \c i
  future get(ordinal_type i) const { return pimpl_->get_tile(i); }

  /// Discard a tile that is not needed

  /// This function handles the cleanup for tiles that are not needed in
  /// subsequent computation.
  /// \param i The index of a local tile to discard
  /// \pre `this->is_local(i)`
  virtual void discard(ordinal_type i) const {
    TA_ASSERT(this->is_local(i));
    pimpl_->discard_tile(i);
  }

  /// World object accessor

  /// \return A reference to the world object
  World& world() const { return pimpl_->world(); }

  /// Unique object id

  /// \return The unique id for this object
  madness::uniqueidT id() const { return pimpl_->id(); }

  /// \return Number of nonzero tiles on this rank
  /// \sa TensorImpl::local_nnz()
  ordinal_type local_nnz() const { return pimpl_->local_nnz(); }

  /// \return The number of tasks spawned on this rank (after invoking eval()
  /// this should be same as the value returned by local_nnz(), if everything is
  /// well)
  ordinal_type task_count() const { return pimpl_->task_count(); }

  /// Wait for all local tiles to be evaluated
  void wait() const { pimpl_->wait(); }

#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
  /// reports evaluator status

  /// intended for debugging purposes
  /// @return string containing log of the current status of evaluator (empty
  /// string, unless overridden in the specialization)
  std::string status() const {
    std::ostringstream oss;
    oss << "DistEval status: id=" << id()
        << " impl_type_name=" << typeid(*(pimpl_.get())).name()
        << "                 ";
    oss << pimpl_->status();
    oss << "\n";
    return oss.str();
  }
#endif

};  // class DistEval

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_DIST_EVAL_DIST_EVAL_BASE_H__INCLUDED
