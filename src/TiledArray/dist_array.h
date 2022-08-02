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

#ifndef TILEDARRAY_ARRAY_H__INCLUDED
#define TILEDARRAY_ARRAY_H__INCLUDED

#include "TiledArray/expressions/fwd.h"

#include "TiledArray/array_impl.h"
#include "TiledArray/conversions/clone.h"
#include "TiledArray/conversions/truncate.h"
#include "TiledArray/pmap/replicated_pmap.h"
#include "TiledArray/policies/dense_policy.h"
#include "TiledArray/replicator.h"
#include "TiledArray/tile_interface/cast.h"
#include "TiledArray/util/annotation.h"
#include "TiledArray/util/initializer_list.h"
#include "TiledArray/util/random.h"

#include <madness/world/parallel_archive.h>
#include <cstdlib>
#include <tuple>

namespace TiledArray {

// Forward declarations
template <typename, typename>
class Tensor;

/// A (multidimensional) tiled array

/// DistArray is the local representation of a global object. This means that
/// the local array object will only contain a portion of the data. It may be
/// used to construct distributed tensor algebraic operations.
/// \tparam T The element type of for array tiles
/// \tparam Tile The tile type [ Default = \c Tensor<T> ]
template <typename Tile = Tensor<double>, typename Policy = DensePolicy>
class DistArray : public madness::archive::ParallelSerializableObject {
 public:
  typedef TiledArray::detail::ArrayImpl<Tile, Policy>
      impl_type;  ///< The type of the PIMPL
  typedef typename impl_type::policy_type policy_type;  ///< Policy type

  /// Type used to hold the components of the tensors in the array. For
  /// DistArray<Tensor<double>> this will be double. Similarly for
  /// DistArray<Tensor<Tensor<double>> this will also be double. Notably for
  /// complex elements of type std::complex<T> numeric_type will be
  /// std::complex<T> and scalar_type will be T.
  typedef typename detail::numeric_type<Tile>::type numeric_type;

  /// Type used to hold the scalar components of numeric_type. For real-valued
  /// arrays scalar_type will be the same as numeric_type; however, for arrays
  /// with elements of type std::complex<T> scalar_type will be T.
  typedef typename detail::scalar_type<Tile>::type scalar_type;

  typedef typename impl_type::trange_type trange_type;  ///< Tile range type
  typedef
      typename impl_type::range_type range_type;  ///< Elements/tiles range type
  typedef typename impl_type::shape_type
      shape_type;  ///< Shape type for array tiling
  typedef typename impl_type::range_type::index1_type
      index1_type;  ///< 1-index type
  typedef typename impl_type::range_type::index
      index;  ///< Array coordinate index type
  typedef typename impl_type::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename impl_type::value_type value_type;      ///< Tile type
  typedef
      typename impl_type::eval_type eval_type;   ///< The tile evaluation type
  typedef typename impl_type::reference future;  ///< Future of \c value_type
  typedef typename impl_type::reference reference;  ///< \c future type
  typedef
      typename impl_type::const_reference const_reference;  ///< \c future type
  typedef typename impl_type::iterator iterator;  ///< Local tile iterator
  typedef typename impl_type::const_iterator
      const_iterator;  ///< Local tile const iterator
  typedef typename impl_type::pmap_interface
      pmap_interface;  ///< Process map interface type

  /// The type of the elements in the tile. For a tile of type Tensor<T> this is
  /// T. Note that if the tile type is Tensor<Tensor<double>> this typedef will
  /// be Tensor<double> and NOT double like numeric_type.
  typedef typename value_type::value_type element_type;

  ///
  /// These are some of the types and compile-time values needed for SFINAE.
  /// They probably should be factored out into a header file so they can be
  /// used elsewhere too.
  ///
  template <typename OtherTile>
  using is_my_type = std::is_same<DistArray, DistArray<OtherTile, Policy>>;

  template <typename OtherTile>
  using enable_if_not_my_type =
      std::enable_if_t<not is_my_type<OtherTile>::value>;

  template <typename Index>
  static constexpr bool is_integral_or_integral_range_v =
      std::is_integral_v<Index> || detail::is_integral_range_v<Index>;

  template <typename Index>
  using enable_if_is_integral_or_integral_range =
      std::enable_if_t<is_integral_or_integral_range_v<Index>>;

  template <typename Index>
  using enable_if_is_integral = std::enable_if_t<std::is_integral_v<Index>>;

  template <typename Value>
  static constexpr bool is_value_or_future_to_value_v =
      std::is_same_v<std::decay_t<Value>, Future<value_type>> ||
      std::is_same_v<std::decay_t<Value>, value_type>;

 private:
  std::shared_ptr<impl_type> pimpl_;  ///< Array implementation pointer
  bool defer_deleter_to_next_fence_ =
      false;  ///< if true, the impl object is scheduled to be destroyed in the
              ///< next fence

  static madness::AtomicInt cleanup_counter_;

  /// Array deleter function

  /// This function schedules a task for lazy cleanup. Array objects are
  /// deleted only after the object has been deleted in all processes.
  /// \param pimpl The implementation pointer to be deleted.
  static void lazy_deleter(const impl_type* const pimpl) {
    if (pimpl) {
      if (madness::initialized()) {
        World& world = pimpl->world();
        const madness::uniqueidT id = pimpl->id();
        cleanup_counter_++;

        // wait for all DelayedSet's to vanish
        world.await([&]() { return (pimpl->num_live_ds() == 0); }, true);

        try {
          world.gop.lazy_sync(id, [pimpl]() {
            delete pimpl;
            DistArray::cleanup_counter_--;
          });
        } catch (madness::MadnessException& e) {
          fprintf(stderr,
                  "!! ERROR TiledArray: madness::MadnessException thrown in "
                  "Array::lazy_deleter().\n"
                  "%s\n"
                  "!! ERROR TiledArray: The exception has been absorbed.\n"
                  "!! ERROR TiledArray: rank=%i\n",
                  e.what(), world.rank());

          cleanup_counter_--;
          delete pimpl;
        } catch (std::exception& e) {
          fprintf(stderr,
                  "!! ERROR TiledArray: std::exception thrown in "
                  "Array::lazy_deleter().\n"
                  "%s\n"
                  "!! ERROR TiledArray: The exception has been absorbed.\n"
                  "!! ERROR TiledArray: rank=%i\n",
                  e.what(), world.rank());

          cleanup_counter_--;
          delete pimpl;
        } catch (...) {
          fprintf(stderr,
                  "!! ERROR TiledArray: An unknown exception was thrown in "
                  "Array::lazy_deleter().\n"
                  "!! ERROR TiledArray: The exception has been absorbed.\n"
                  "!! ERROR TiledArray: rank=%i\n",
                  world.rank());

          cleanup_counter_--;
          delete pimpl;
        }
      } else {
        delete pimpl;
      }
    }
  }

  /// Sparse array initialization

  /// \param world The world where the array will live.
  /// \param trange The tiled range object that will be used to set the array
  /// tiling. \param shape The array shape that defines zero and non-zero tiles
  /// \param pmap The tile index -> process map
  static std::shared_ptr<impl_type> init(World& world,
                                         const trange_type& trange,
                                         const shape_type& shape,
                                         std::shared_ptr<pmap_interface> pmap) {
    // User level validation of input

    if (!pmap) {
      // Construct a default process map
      pmap = Policy::default_pmap(world, trange.tiles_range().volume());
    } else {
      // Validate the process map
      TA_ASSERT(pmap->size() == trange.tiles_range().volume() &&
                "TiledArray::DistArray::DistArray() -- The size of the process "
                "map is not "
                "equal to the number of tiles in the TiledRange object.");
      TA_ASSERT(pmap->rank() ==
                    typename pmap_interface::size_type(world.rank()) &&
                "TiledArray::DistArray::DistArray() -- The rank of the process "
                "map is not equal to that "
                "of the world object.");
      TA_ASSERT(pmap->procs() ==
                    typename pmap_interface::size_type(world.size()) &&
                "TiledArray::DistArray::DistArray() -- The number of processes "
                "in the process map is not "
                "equal to that of the world object.");
    }

    // Validate the shape
    TA_ASSERT(
        !shape.empty() &&
        "TiledArray::DistArray::DistArray() -- The shape is not initialized.");
    TA_ASSERT(shape.validate(trange.tiles_range()) &&
              "TiledArray::DistArray::DistArray() -- The range of the shape is "
              "not equal to "
              "the tiles range.");

    return std::shared_ptr<impl_type>(new impl_type(world, trange, shape, pmap),
                                      lazy_deleter);
  }

 public:
  /// Default constructor

  /// Constructs an uninitialized array object. Uninitialized arrays contain
  /// no tile or meta data. Most of the functions are not available when the
  /// array is uninitialized, but these arrays may be assign via a tensor
  /// expression assignment or the copy construction.

  DistArray() : pimpl_() {}

  /// Copy constructor

  /// This is a shallow copy, that is no data is copied.
  /// \param other The array to be copied
  DistArray(const DistArray& other) : pimpl_(other.pimpl_) {}

  /// Dense array constructor

  /// Constructs an array with the given meta data. This constructor only
  /// initializes the array meta data; the array tiles are empty and must be
  /// assigned by the user.
  /// \param world The world where the array will live.
  /// \param trange The tiled range object that will be used to set the array
  /// tiling. \param pmap The tile index -> process map
  DistArray(World& world, const trange_type& trange,
            const std::shared_ptr<pmap_interface>& pmap =
                std::shared_ptr<pmap_interface>())
      : pimpl_(init(world, trange, shape_type(1, trange), pmap)) {}

  /// Sparse array constructor

  /// Constructs an array with the given meta data. This constructor only
  /// initializes the array meta data; the array tiles are empty and must be
  /// assigned by the user.
  /// \param world The world where the array will live.
  /// \param trange The tiled range object that will be used to set the array
  /// tiling. \param shape The array shape that defines zero and non-zero tiles
  /// \param pmap The tile index -> process map
  DistArray(World& world, const trange_type& trange, const shape_type& shape,
            const std::shared_ptr<pmap_interface>& pmap =
                std::shared_ptr<pmap_interface>())
      : pimpl_(init(world, trange, shape, pmap)) {}

  /// \name Initializer list constructors
  /// \brief Creates a new tensor containing the elements in the provided
  ///         `std::initializer_list`.
  ///@{

  ///  This ctor will create an array comprised of a single tile. The array
  ///  will have a rank equal to the nesting of \p il and the elements will be
  ///  those in the provided `std::initializer_list`. This ctor can not be used
  ///  to create an empty tensor (attempts to do so will raise an error).
  ///
  /// \tparam T The types of the elements in the `std::initializer_list`. Must
  ///           be implicitly convertible to numeric_type.
  ///
  /// \param[in] world The world where the resulting array will live.
  /// \param[in] il The initial values for the elements in the array. The
  ///               elements are assumed to be listed in row-major order.
  ///
  /// \throw TiledArray::Exception if \p il contains no elements. If an
  ///                              exception is raised \p world and \p il are
  ///                              unchanged (strong throw guarantee).
  /// \throw TiledArray::Exception If the provided `std::initializer_list` is
  ///                              not rectangular (*e.g.*, attempting to
  ///                              initialize a matrix with the value
  ///                              `{{1, 2}, {3, 4, 5}}`). If an exception is
  ///                              raised \p world and \p il are unchanged.
  template <typename T>
  DistArray(World& world,
            std::initializer_list<T>
                il)  // N.B. clang does not like detail::vector_il<T> here
      : DistArray(array_from_il<DistArray>(world, il)) {}

  template <typename T>
  DistArray(World& world, std::initializer_list<std::initializer_list<T>> il)
      : DistArray(array_from_il<DistArray>(world, il)) {}

  template <typename T>
  DistArray(
      World& world,
      std::initializer_list<std::initializer_list<std::initializer_list<T>>> il)
      : DistArray(array_from_il<DistArray>(world, il)) {}

  template <typename T>
  DistArray(World& world, std::initializer_list<std::initializer_list<
                              std::initializer_list<std::initializer_list<T>>>>
                              il)
      : DistArray(array_from_il<DistArray>(world, il)) {}

  template <typename T>
  DistArray(World& world,
            std::initializer_list<std::initializer_list<std::initializer_list<
                std::initializer_list<std::initializer_list<T>>>>>
                il)
      : DistArray(array_from_il<DistArray>(world, il)) {}

  template <typename T>
  DistArray(
      World& world,
      std::initializer_list<
          std::initializer_list<std::initializer_list<std::initializer_list<
              std::initializer_list<std::initializer_list<T>>>>>>
          il)
      : DistArray(array_from_il<DistArray>(world, il)) {}
  ///@}

  /// \name Tiling initializer list constructors
  /// \brief Constructs a new tensor containing the elements in the provided
  ///         `std::initializer_list`.
  /// @{

  ///  This ctor will create an array using the provided TiledRange instance
  ///  whose values will be initialized from the provided
  ///  `std::initializer_list` \p il. This ctor can not be used to create an
  ///  empty tensor (attempts to do so will raise an error).
  ///
  /// \tparam T The types of the elements in the `std::initializer_list`. Must
  ///           be implicitly convertible to numeric_type.
  ///
  /// \param[in] world The world where the resulting array will live.
  /// \param[in] trange The tiling to use for the resulting array.
  /// \param[in] il The initial values for the elements in the array. The
  ///               elements are assumed to be listed in row-major order.
  ///
  /// \throw TiledArray::Exception if \p il contains no elements. If an
  ///                              exception is raised \p world and \p il are
  ///                              unchanged (strong throw guarantee).
  /// \throw TiledArray::Exception If the provided `std::initializer_list` is
  ///                              not rectangular (*e.g.*, attempting to
  ///                              initialize a matrix with the value
  ///                              `{{1, 2}, {3, 4, 5}}`). If an exception is
  ///                              raised \p world and \p il are unchanged.
  template <typename T>
  DistArray(World& world, const trange_type& trange,
            std::initializer_list<T> il)
      : DistArray(array_from_il<DistArray>(world, trange, il)) {}

  template <typename T>
  DistArray(World& world, const trange_type& trange,
            std::initializer_list<std::initializer_list<T>> il)
      : DistArray(array_from_il<DistArray>(world, trange, il)) {}

  template <typename T>
  DistArray(
      World& world, const trange_type& trange,
      std::initializer_list<std::initializer_list<std::initializer_list<T>>> il)
      : DistArray(array_from_il<DistArray>(world, trange, il)) {}

  template <typename T>
  DistArray(World& world, const trange_type& trange,
            std::initializer_list<std::initializer_list<
                std::initializer_list<std::initializer_list<T>>>>
                il)
      : DistArray(array_from_il<DistArray>(world, trange, il)) {}

  template <typename T>
  DistArray(World& world, const trange_type& trange,
            std::initializer_list<std::initializer_list<std::initializer_list<
                std::initializer_list<std::initializer_list<T>>>>>
                il)
      : DistArray(array_from_il<DistArray>(world, trange, il)) {}

  template <typename T>
  DistArray(
      World& world, const trange_type& trange,
      std::initializer_list<
          std::initializer_list<std::initializer_list<std::initializer_list<
              std::initializer_list<std::initializer_list<T>>>>>>
          il)
      : DistArray(array_from_il<DistArray>(world, trange, il)) {}
  /// @}

  /// converting copy constructor

  /// This constructor uses the meta data of `other` to initialize the meta
  /// data of the new array. In addition, the tiles of the new array are also
  /// initialized using TiledArray::Cast<Tile,OtherTile>
  /// \param other The array to be copied
  template <typename OtherTile, typename = enable_if_not_my_type<OtherTile>>
  explicit DistArray(const DistArray<OtherTile, Policy>& other) : pimpl_() {
    *this = foreach<Tile>(other, [](Tile& result, const OtherTile& source) {
      result = TiledArray::Cast<Tile, OtherTile>{}(source);
    });
  }

  /// Unary transform constructor

  /// This constructor uses the meta data of `other` to initialize the meta
  /// data of the new array. In addition, the tiles of the new array are also
  /// initialized using the `op` function/functor, which transforms
  /// each tile in `other` using `op`
  /// \param other The array to be copied
  template <typename OtherTile, typename Op>
  DistArray(const DistArray<OtherTile, Policy>& other, Op&& op) : pimpl_() {
    *this = foreach<Tile>(other, std::forward<Op>(op));
  }

  /// Destructor

  /// This is a distributed lazy destructor. The object will only be deleted
  /// after the last reference to the world object on all nodes has been
  /// destroyed and there are no outstanding references to the object's data.
  /// Use defer_deleter_to_next_fence() to defer the deletion of the destructor
  /// to the next fence.
  /// \sa defer_deleter_to_next_fence
  ~DistArray() {
    if (defer_deleter_to_next_fence_) {
      madness::detail::deferred_cleanup(
          this->world(), pimpl_,
          /* do_not_check_that_pimpl_is_unique = */ true);
    }
  }

  /// Defers deletion to the next fene

  /// By default the destruction of the object's data occurs lazily, when
  /// all local references to the object are gone and all _remote_ references
  /// to the local object's data are gone. This is not always sufficient;
  /// call this at any point during object's lifetime to ensure that the
  /// lifetime of the object lasts to (just past)the next fence.
  void defer_deleter_to_next_fence() { defer_deleter_to_next_fence_ = true; }

  /// Create a deep copy of this array

  /// \return An array that is equal to this array
  DistArray clone() const { return TiledArray::clone(*this); }

  /// Accessor for the (shared_ptr to) implementation object

  /// \return std::shared_ptr to the const implementation object
  std::shared_ptr<const impl_type> pimpl() const { return pimpl_; }

  /// Accessor for the (shared_ptr to) implementation object

  /// \return std::shared_ptr to the nonconst implementation object
  std::shared_ptr<impl_type> pimpl() { return pimpl_; }

  /// Accessor for the (weak_ptr to) implementation object

  /// \return std::weak_ptr to the const implementation object
  std::weak_ptr<const impl_type> weak_pimpl() const { return pimpl_; }

  /// Accessor for the (shared_ptr to) implementation object

  /// \return std::weak_ptr to the nonconst implementation object
  std::weak_ptr<impl_type> weak_pimpl() { return pimpl_; }

  /// Checks if this is a unique handle to the implementation object

  /// \return true if this is a unique handle to the implementation object
  bool is_unique() const { return pimpl_.unique(); }

  /// Wait for lazy tile cleanup

  /// This function will wait for cleanup of tile data that has been
  /// scheduled for lazy deletion. Ready tasks will be executed by this
  /// function while waiting for cleanup. This function will timeout if
  /// the wait time exceeds the timeout specified in the `MAD_WAIT_TIMEOUT`
  /// environment variable. The default timeout is 900 seconds.
  /// \param world The world that to be used to execute ready tasks.
  /// \throw madness::MadnessException When timeout has been exceeded.
  static void wait_for_lazy_cleanup(World& world, const double = 60.0) {
    try {
      world.await([&]() { return (cleanup_counter_ == 0); }, true);
    } catch (...) {
      printf("%i: Array lazy cleanup timeout with %i pending cleanup(s)\n",
             world.rank(), int(cleanup_counter_));
      throw;
    }
  }

  /// Wait for lazy tile cleanup

  /// The member version of the static DistArray::wait_for_lazy_cleanup
  void wait_for_lazy_cleanup() const {
    DistArray::wait_for_lazy_cleanup(world());
  }

  /// Copy assignment

  /// This is a shallow copy, that is no data is copied.
  /// \param other The array to be copied
  DistArray& operator=(const DistArray& other) {
    pimpl_ = other.pimpl_;

    return *this;
  }

  /// Global object id

  /// \return A globally unique identifier.
  /// \throw TiledArray::Exception if the PIMPL has not been initialized.
  /// \note This function is primarily used for debugging purposes. Users
  /// should not rely on this function.
  madness::uniqueidT id() const { return impl_ref().id(); }

  /// Begin iterator factory function

  /// \return An iterator to the first local tile.
  /// \throw TiledArray::Exception if the PIMPL has not been initialized. Strong
  ///                              throw guarantee.
  iterator begin() { return impl_ref().begin(); }

  /// Begin const iterator factory function

  /// \return A const iterator to the first local tile.
  /// \throw Tiledarray::Exception if the PIMPL has not been initialized. Strong
  ///                              throw guarantee.
  const_iterator begin() const { return impl_ref().cbegin(); }

  /// End iterator factory function

  /// \return An iterator to one past the last local tile.
  /// \throw TiledArray::Exception if the PIMPL has not been initialized. Strong
  ///                              throw guarantee.
  iterator end() { return impl_ref().end(); }

  /// End const iterator factory function

  /// \return A const iterator to one past the last local tile.
  /// \throw TiledArray::Exception if the PIMPL has not been initialized. Strong
  ///                              throw guarantee.
  const_iterator end() const { return impl_ref().cend(); }

  /// Find local or remote tile by index

  /// \tparam Index The type of the index. Should be an integral type for an
  ///               ordinal index, a type satisfying container of integral
  ///               instances for a coordinate index, or an integral range type.
  /// \param[in] i The ordinal or coordinate index of the desired tile or a
  ///              range or indices.
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero
  /// \throw TiledArray::Exception If PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i is out of bounds. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i is a coordinate index with the
  ///                              wrong rank. Strong throw guarantee.
  template <typename Index,
            typename = enable_if_is_integral_or_integral_range<Index>>
  Future<value_type> find(const Index& i) const {
    check_index(i);
    return pimpl_->get(i);
  }

  /// Find local or remote tile

  /// \tparam Integer An integer type
  /// \param i An \c std::initializer_list<Integer> indicating the coordinate
  ///          index of the requested tile.
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero
  /// \throw TiledArray::Exception If PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i is out of bounds. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i is a coordinate index with the
  ///                              wrong rank. Strong throw guarantee.
  template <typename Integer, typename = enable_if_is_integral<Integer>>
  Future<value_type> find(const std::initializer_list<Integer>& i) const {
    return find<std::initializer_list<Integer>>(i);
  }

  /// Find local tile

  /// \tparam Index An integral or integral range type
  /// \param i The tile index
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero or not local
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index> ||
                                        detail::is_integral_range_v<Index>>>
  const Future<value_type>& find_local(const Index& i) const {
    check_local_index(i);
    return pimpl_->get_local(i);
  }

  /// Find local tile

  /// \tparam Integer An integral type
  /// \param i The tile index, as an \c std::initializer_list<Integer>
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero or not local
  template <typename Integer,
            typename = std::enable_if_t<(std::is_integral_v<Integer>)>>
  const Future<value_type>& find_local(
      const std::initializer_list<Integer>& i) const {
    return find_local<std::initializer_list<Integer>>(i);
  }

  /// Find local tile

  /// \tparam Index An integral or integral range type
  /// \param i The tile index
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero or not local
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index> ||
                                        detail::is_integral_range_v<Index>>>
  Future<value_type>& find_local(const Index& i) {
    check_local_index(i);
    return pimpl_->get_local(i);
  }

  /// Find local tile

  /// \tparam Integer An integral type
  /// \param i The tile index, as an \c std::initializer_list<Integer>
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero or not local
  template <typename Integer,
            typename = std::enable_if_t<(std::is_integral_v<Integer>)>>
  Future<value_type>& find_local(const std::initializer_list<Integer>& i) {
    return find_local<std::initializer_list<Integer>>(i);
  }

  /// Set a tile and fill it using a sequence
  ///
  /// This function will set an uninitialized tile to the provided value. The
  /// tile may be specified either by ordinal or coordinate index.
  ///
  /// \tparam Index Either an integral type, a container type with integral
  ///               objects, or an integral range type.
  /// \tparam InIter Type satisfying input iterator
  /// \param[in] i The index or the ordinal of the tile to be set. i may be
  ///              either an ordinal or coordinate index.
  /// \param[in] first The iterator that points to the start of the input
  ///                  sequence. It is assumed that the container pointed to by
  ///                  first minimally contains the same number of elements as
  ///                  the tile.
  /// \throw TiledArray::Exception if the tile is already initialized.
  template <
      typename Index, typename InIter,
      typename = std::enable_if_t<is_integral_or_integral_range_v<Index> &&
                                  detail::is_input_iterator<InIter>::value>>
  void set(const Index& i, InIter first) {
    check_index(i);
    pimpl_->set(i, value_type(pimpl_->trange().make_tile_range(i), first));
  }

  /// Set a tile and fill it using a sequence
  ///
  /// This function will set an uninitialized tile to the provided value. This
  /// overload allows the user to specify the coordinate index with an
  /// initializer list.
  ///
  /// \tparam Integer An integral type
  /// \tparam InIter Type satisfying input iterator
  /// \param[in] i The tile's coordinate index, as an
  ///            \c std::initializer_list<Integer>
  /// \param[in] first The iterator that points to the start of the input
  ///                  sequence. It is assumed that the container pointed to by
  ///                  first minimally contains the same number of elements as
  ///                  the tile.
  /// \throw TiledArray::Exception if the tile is already initialized.
  template <typename Integer, typename InIter,
            typename = std::enable_if_t<(std::is_integral_v<Integer>)&&detail::
                                            is_input_iterator<InIter>::value>>
  typename std::enable_if<detail::is_input_iterator<InIter>::value>::type set(
      const std::initializer_list<Integer>& i, InIter first) {
    set<std::initializer_list<Integer>>(i, first);
  }

  /// Set a tile and fill it using a value

  /// This function sets each element of a tile to the specified value. For
  /// normal, non-nested, tiles this amounts to setting each scalar component of
  /// the tile to the provided value. For nested tile types this function sets
  /// the elements of the outer most tile (so the input value would be of type
  /// Tensor<T>, assuming a tile type of Tensor<Tensor<T>>).
  ///
  /// \tparam Index An integral type, a type satisfying the concept of a
  ///               container of integral types, or an integral range type.
  /// \param[in] i Either the ordinal or coordinate index of the tile to be set.
  /// \param[in] value What each element of the tile will be set to.
  /// \throw TiledArray::Exception If PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i is out of bounds. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i is a coordinate index with the
  ///                              wrong rank. Strong throw guarantee.
  /// \throw TiledArray::Exception if tile \c i is already set.
  template <typename Index,
            typename = enable_if_is_integral_or_integral_range<Index>>
  void set(const Index& i, const element_type& value = element_type()) {
    check_index(i);
    pimpl_->set(i, value_type(pimpl_->trange().make_tile_range(i), value));
  }

  /// Set every element of a tile to a specified value

  /// This function sets each element of a tile to the specified value. For
  /// normal, non-nested, tiles this amounts to setting each scalar component of
  /// the tile to the provided value. For nested tile types this function sets
  /// the elements of the outer most tile (so the input value would be of type
  /// Tensor<T>, assuming a tile type of Tensor<Tensor<T>>).
  ///
  /// \tparam Integer An integral type
  /// \param[in] i The coordinate index, as an \c std::initializer_list<Integer>
  ///              for the tile.
  /// \param[in] value What the tile elements should be set to.
  /// \throw TiledArray::Exception If PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i is out of bounds. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i has the wrong rank. Strong
  ///                              throw guarantee.
  /// \throw TiledArray::Exception if tile \c i is already set.
  template <typename Integer, typename = enable_if_is_integral<Integer>>
  void set(const std::initializer_list<Integer>& i,
           const element_type& value = element_type()) {
    set<std::initializer_list<Integer>>(i, value);
  }

  /// Set a tile directly using a future to a tile

  /// \tparam Index For an ordinal index should be an integral type and for a
  ///               coordinate index should be a type satisfying container of
  ///               integral types. Can also be a range type.
  /// \param[in] i The ordinal or coordinate index of the tile to set
  /// \param[in] v The tile's new value
  /// \throw TiledArray::Exception If PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i is out of bounds. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i is a coordinate index with the
  ///                              wrong rank. Strong throw guarantee.
  /// \throw TiledArray::Exception if tile \c i is already set.
  template <
      typename Index, typename Value,
      typename = std::enable_if_t<(is_integral_or_integral_range_v<Index> &&
                                   is_value_or_future_to_value_v<Value>)>>
  void set(const Index& i, Value&& v) {
    check_index(i);
    pimpl_->set(i, std::forward<Value>(v));
  }

  /// Set a tile directly using a future to a tile

  /// \tparam Integer An integral type
  /// \param[in] i The coordinate index of the tile to set, as an
  ///          \c std::initializer_list<Integer>
  /// \param[in] f A future to the tile's new value
  /// \throw TiledArray::Exception If PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i is out of bounds. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if index \c i has the wrong rank. Strong
  ///                              throw guarantee.
  /// \throw TiledArray::Exception if tile \c i is already set.
  template <
      typename Index, typename Value,
      typename = std::enable_if_t<
          (std::is_integral_v<Index>)&&is_value_or_future_to_value_v<Value>>>
  void set(const std::initializer_list<Index>& i, Value&& v) {
    set<std::initializer_list<Index>>(i, std::forward<Value>(v));
  }

  /// Fill all local tiles with the specified value

  /// \param[in] value What each local tile should be filled with.
  /// \param[in] skip_set If false, will throw if any tiles are already set
  /// \throw TiledArray::Exception if the PIMPL is uninitialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if skip_set is false and a local tile is
  ///                              already set. Weak throw guarantee.
  void fill_local(const element_type& value = element_type(),
                  bool skip_set = false) {
    init_tiles(
        [value](const range_type& range) { return value_type(range, value); },
        skip_set);
  }

  /// Fill all local tiles with the specified value

  /// \param[in] value What each local tile should be filled with.
  /// \param[in] skip_set If false, will throw if any tiles are already set
  /// \throw TiledArray::Exception if the PIMPL is uninitialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if skip_set is false and a local tile is
  ///                              already set. Weak throw guarantee.
  void fill(const element_type& value = numeric_type(), bool skip_set = false) {
    fill_local(value, skip_set);
  }

  /// Fill all local tiles with random values
  ///
  /// This function will fill all local tiles with random values. The random
  /// values are generated by calling TiledArray::detail::MakeRandom, which can
  /// be specialized to determine how the random values for a given type are
  /// generated. It should be noted that if MakeRandom does not know how to
  /// generate random values of type T this function will be disabled via SFINAE
  /// and attempting to use it will lead to a compile-time error.
  ///
  /// \tparam T The type of random value to generate. Defaults to
  ///           element_type.
  /// \tparam <anonymous> A template type parameter which will be deduced as
  ///                     void only if MakeRandom knows how to generate random
  ///                     values of type T. If MakeRandom does not know how to
  ///                     generate random values of type T, SFINAE will disable
  ///                     this function.
  /// \param[in] skip_set If false, will throw if any tiles are already set
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong
  ///                              throw guarantee.
  /// \throw TiledArray::Exception if skip_set is false and a local tile is
  ///                              already initialized. Weak throw guarantee.
  template <typename T = element_type,
            typename = detail::enable_if_can_make_random_t<T>>
  void fill_random(bool skip_set = false) {
    init_elements(
        [](const auto&) { return detail::MakeRandom<T>::generate_value(); });
  }

  /// Initialize (local) tiles with a user provided functor

  /// This function is used to initialize the local, non-zero tiles of the array
  /// via a function (or functor). The work is done in parallel, therefore \c op
  /// must be a thread safe function/functor. The signature of the functor
  /// should be:
  /// \code
  /// value_type op(const range_type&)
  /// \endcode
  /// For example, in the following code, the array tiles are initialized with
  /// random numbers from 0 to 1:
  /// \code
  /// array.init_tiles([] (const TiledArray::Range& range) ->
  /// TiledArray::Tensor<double>
  ///     {
  ///        // Initialize the tile with the given range object
  ///        TiledArray::Tensor<double> tile(range);
  ///
  ///        // Initialize the random number generator
  ///        std::default_random_engine generator;
  ///        std::uniform_real_distribution<double> distribution(0.0,1.0);
  ///
  ///        // Fill the tile with random numbers
  ///        for(auto& value : tile)
  ///           value = distribution(generator);
  ///
  ///        return tile;
  ///     });
  /// \endcode
  /// \tparam Op The type of the functor/function
  /// \param[in] op The operation used to generate tiles
  /// \param[in] skip_set If false, will throw if any tiles are already set
  /// \throw TiledArray::Exception if the PIMPL is not set. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if a tile is already set and skip_set is
  ///                              false. Weak throw guarantee.
  template <typename Op>
  void init_tiles(Op&& op, bool skip_set = false) {
    // lifetime management of op depends on whether it is a lvalue ref (i.e. has
    // an external owner) or an rvalue ref
    // - if op is an lvalue ref: pass op to tasks
    // - if op is an rvalue ref pass make_shared_function(op) to tasks
    auto op_shared_handle = make_op_shared_handle(std::forward<Op>(op));

    auto it = impl_ref().pmap()->begin();
    const auto end = pimpl_->pmap()->end();
    for (; it != end; ++it) {
      const auto& index = *it;
      if (!pimpl_->is_zero(index)) {
        if (skip_set) {
          auto fut = find(index);
          if (fut.probe()) continue;
        }
        Future<value_type> tile = pimpl_->world().taskq.add(
            [pimpl = this->weak_pimpl(), index = ordinal_type(index),
             op_shared_handle]() -> value_type {
              auto pimpl_ptr = pimpl.lock();
              if (pimpl_ptr)
                return op_shared_handle(
                    pimpl_ptr->trange().make_tile_range(index));
              else
                return {};
            });
        set(index, tile);
      }
    }
  }

  /// Initialize elements of local, non-zero tiles with a user provided functor

  /// This function is used to initialize the elements of the local, non-zero
  /// tiles via a function (or functor). The work is done in parallel, therefore
  /// \c op must be a thread safe function/functor. The signature of the functor
  /// should be:
  /// \code
  /// element_type op(const index&)
  /// \endcode
  /// For example, in the following code, the array elements are initialized
  /// with random numbers from 0 to 1:
  /// \code
  /// array.init_elements([] (const auto&) {
  ///        return (double)std::rand() / RAND_MAX;
  /// });
  /// \endcode
  /// \tparam Op Type of the function/functor which will generate the elements.
  /// \param[in] op The operation used to generate elements
  /// \param[in] skip_set If false, will throw if any tiles are already set
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong
  ///                              throw guarnatee.
  /// \throw TiledArray::Exception if skip_set is false and a local, non-zero
  ///                              tile is already initialized. Weak throw
  ///                              guarantee.
  template <typename Op>
  void init_elements(Op&& op, bool skip_set = false) {
    auto op_shared_handle = make_op_shared_handle(std::forward<Op>(op));
    init_tiles(
        [op = std::move(op_shared_handle)](
            const TiledArray::Range& range) -> value_type {
          // Initialize the tile with the given range object
          Tile tile(range);

          // Initialize tile elements
          for (auto& idx : range) tile[idx] = op(idx);

          return tile;
        },
        skip_set);
  }

  /// Tiled range accessor

  /// This function returns an object containing the tiling information of the
  /// array (e.g., the tile boundaries, or which elements go to which tile).
  /// This should not be confused with `range` which returns a Range object for
  /// iterating over the tile indices or `elements_range` which returns a Range
  /// object for iterating over the elements.
  ///
  /// \return A const reference to the tiled range object for the array
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  const trange_type& trange() const { return impl_ref().trange(); }

  /// Tile range accessor

  /// This function returns a range object which can be used to iterate over the
  /// indices of the tiles. This should be contrasted with `trange()` which
  /// gives an object holding the tiling information (number of tiles, which
  /// elements belong to which tiles, etc.) and `elements_range()` which returns
  /// a range object that can be used to iterate over element indices.
  ///
  /// \return A const reference to the range object for the array's tiles
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \deprecated use DistArray::tiles_range()
  [[deprecated("use DistArray::tiles_range()")]] const range_type& range()
      const {
    return impl_ref().tiles_range();
  }

  /// Tile range accessor

  /// \return A const reference to the range object for the array tiles
  const range_type& tiles_range() const { return impl_ref().tiles_range(); }

  /// \deprecated use DistArray::elements_range()
  [[deprecated("use DistArray::elements_range()")]] const typename trange_type::
      range_type&
      elements() const {
    return elements_range();
  }

  /// Element range accessor

  /// This function returns a Range object which can be used to iterate over
  /// the indices of the tiles' elements. This should not be confused with
  /// `trange()` which returns an object containing the tiling information
  /// (e.g., where the tile boundaries are, or which element is in which tile)
  /// or `range()` which returns an object for iterating over the indices of the
  /// tiles.
  ///
  /// \return A const reference to the range object for the array's elements
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  const typename trange_type::range_type& elements_range() const {
    return impl_ref().trange().elements_range();
  }

  /// Returns the number of tiles in the tensor

  /// This function returns the number of tiles in the tensor. This is usually
  /// not the same as the volume of the tensor (i.e., the number of elements in
  /// the tensor; they are the same only if each tile contains a single
  /// element).
  ///
  /// \return The number of tiles in the tensor.
  /// \throw TiledArray::Exception if the PIMPL has not been set. Strong throw
  ///                              guarantee.
  auto size() const { return impl_ref().size(); }

  /// Create a tensor expression

  /// \param vars A string with a comma-separated list of variables
  /// \return A const tensor expression object
  /// \note size and contents of \p vars are validated using
  ///   DistArray::check_str_index()
  auto operator()(const std::string& vars) const {
    check_str_index(vars);
    return TiledArray::expressions::TsrExpr<const DistArray>(*this, vars);
  }

  /// Create a tensor expression

  /// \param vars A string with a comma-separated list of variables
  /// \return A non-const tensor expression object
  /// \note size and contents of \p vars are validated using
  ///   DistArray::check_str_index()
  auto operator()(const std::string& vars) {
    check_str_index(vars);
    return TiledArray::expressions::TsrExpr<DistArray>(*this, vars);
  }

  /// \deprecated use DistArray::world()
  [[deprecated]] World& get_world() const { return world(); }

  /// World accessor

  /// \return A reference to the world that owns this array.
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  World& world() const { return impl_ref().world(); }

  /// \deprecated use DistArray::pmap()
  [[deprecated]] const std::shared_ptr<pmap_interface>& get_pmap() const {
    return pmap();
  }

  /// Process map accessor

  /// \return A reference to the process map that owns this array.
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong
  ///                              throw guarantee.
  const std::shared_ptr<pmap_interface>& pmap() const {
    return impl_ref().pmap();
  }

  /// Check dense/sparse

  /// \return \c true when \c Array is dense, \c false otherwise.
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  bool is_dense() const { return impl_ref().is_dense(); }

  /// \deprecated use DistArray::shape()
  [[deprecated]] const shape_type& get_shape() const { return shape(); }

  /// Shape accessor

  /// Returns shape object. No communication is required.
  /// \return reference to the shape object.
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  inline const shape_type& shape() const { return impl_ref().shape(); }

  /// Tile ownership

  /// \tparam Index An integral type, a type satisfying container of
  ///               integral types, or an integral range type.
  /// \param[in] i The coordinate or ordinal index of the tile or a range of
  ///              tiles whose ownership is in question.
  /// \return The process ID of the owner of a tile.
  /// \note This does not indicate whether a tile exists or not. Only, the rank
  ///       of the process that would own it if it does exist.
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if the index is out of bounds. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if \c i is a coordinate index with the wrong
  ///                              rank. Strong throw guarantee.
  template <typename Index,
            typename = enable_if_is_integral_or_integral_range<Index>>
  ProcessID owner(const Index& i) const {
    check_index(i);
    return pimpl_->owner(i);
  }

  /// Tile ownership

  /// \tparam Index An integral type
  /// \param[in] i The coordinate index of the tile whose ownership is in
  ///              question.
  /// \return The process ID of the owner of a tile.
  /// \note This does not indicate whether a tile exists or not. Only, the rank
  ///       of the process that would own it if it does exist.
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if the index is out of bounds. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if \c i has the wrong rank. Strong throw
  ///                              guarantee.
  template <typename Index, typename = enable_if_is_integral<Index>>
  ProcessID owner(const std::initializer_list<Index>& i) const {
    return owner<std::initializer_list<Index>>(i);
  }

  /// Check if the tile at index \c i is stored locally

  /// \tparam Index An integral type, a type satisfying container of
  ///               integral types, or an integral range type.
  /// \param[in] i The index of the tile whose locality is being questioned. The
  ///              index may be either a coordinate or ordinal index.
  /// \return \c true if \c owner(i) is equal to the MPI process rank,
  ///         otherwise \c false.
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if the index is out of bounds. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if \c i is a coordinate index with the wrong
  ///                              rank. Strong throw guarantee.
  template <typename Index,
            typename = enable_if_is_integral_or_integral_range<Index>>
  bool is_local(const Index& i) const {
    check_index(i);
    return pimpl_->is_local(i);
  }

  /// Check if the tile at index \c i is stored locally

  /// \tparam Index An integral type
  /// \param[in] i The coordinate index of the tile whose locality is being
  ///              questioned.
  /// \return \c true if \c owner(i) is equal to the MPI process rank,
  ///         otherwise \c false.
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if the index is out of bounds. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if \c i has the wrong rank. Strong throw
  ///                              guarantee.
  template <typename Index, typename = enable_if_is_integral<Index>>
  bool is_local(const std::initializer_list<Index>& i) const {
    return is_local<std::initializer_list<Index>>(i);
  }

  /// Check for zero tiles

  /// \tparam Index An integral type, a type satisfying container of
  ///               integral types, or an integral range type.
  /// \param[in] i The index of the tile whose nothingness is being
  ///              contemplated. Can be either an ordinal or coordinate index.
  /// \return \c true if tile at index \c i is zero, false if the tile is
  ///         non-zero or remote existence data is not available.
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if \c i is not in the range of valid tile
  ///                              indices. Strong throw guarantee.
  /// \throw TiledArray::Exception if \c i is a coordinate index with the wrong
  ///                              rank. Strong throw guarantee.
  template <typename Index,
            typename = enable_if_is_integral_or_integral_range<Index>>
  bool is_zero(const Index& i) const {
    check_index(i);
    return pimpl_->is_zero(i);
  }

  /// Check for zero tiles

  /// \tparam Index An integral type
  /// \param[in] i The coordinate index of the tile whose nothingness is being
  ///              contemplated.
  /// \return \c true if tile at index \c i is zero, false if the tile is
  ///         non-zero or remote existence data is not available.
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if \c i is not in the range of valid tile
  ///                              indices. Strong throw guarantee.
  /// \throw TiledArray::Exception if \c i has the wrong rank. Strong throw
  ///                              guarantee.
  template <typename Index, typename = enable_if_is_integral<Index>>
  bool is_zero(const std::initializer_list<Index>& i) const {
    return is_zero<std::initializer_list<Index>>(i);
  }

  /// Swap this array with \c other

  /// \param other The array to be swapped with this array.
  /// \throw None no throw guarantee.
  void swap(DistArray& other) { std::swap(pimpl_, other.pimpl_); }

  /// Convert a distributed array into a replicated array
  /// \throw TiledArray::Exception if the PIMPL is not initialized. Strong throw
  ///                              guarantee.
  void make_replicated() {
    if ((!impl_ref().pmap()->is_replicated()) && (world().size() > 1)) {
      // Construct a replicated array
      auto pmap = std::make_shared<detail::ReplicatedPmap>(world(), size());
      DistArray result = DistArray(world(), trange(), shape(), pmap);

      // Create the replicator object that will do an all-to-all broadcast of
      // the local tile data.
      auto replicator =
          std::make_shared<detail::Replicator<DistArray>>(*this, result);

      // Put the replicator pointer in the deferred cleanup object so it will
      // be deleted at the end of the next fence.
      TA_ASSERT(replicator.unique());  // Required for deferred_cleanup
      madness::detail::deferred_cleanup(world(), replicator);

      DistArray::operator=(result);
    }
  }

  /// Update shape data and remove tiles that are below the zero threshold
  /// \param[in] thresh the threshold below which the tiles are considered
  ///        to be zero (only for sparse arrays will such tiles be discarded)
  /// \sa SparseShape::is_zero

  /// \note This is a collective operation
  /// \note This function is a no-op for dense arrays.
  void truncate(
      typename shape_type::value_type thresh = shape_type::threshold()) {
    TiledArray::truncate(*this, thresh);
  }

  /// Check if the array is initialized

  /// \return \c false if the array has been default initialized, otherwise
  /// \c true.
  /// \throw None No throw guarantee.
  bool is_initialized() const { return static_cast<bool>(pimpl_); }

  /// Conversion to bool is equivalent to DistArray::is_initialized()

  /// \return \c false if the array has been default initialized, otherwise
  /// \c true.
  /// \throw None No throw guarantee.
  explicit operator bool() const { return is_initialized(); }

  /// serialize local contents of a DistArray to an Archive object

  /// @note use Parallel{Input,Output}Archive for parallel serialization
  /// @tparam Archive an Archive type
  /// @warning this does not fence; it is user's responsibility to do that
  template <typename Archive,
            typename = std::enable_if_t<
                !Archive::is_parallel_archive &&
                madness::is_output_archive_v<std::decay_t<Archive>>>>
  void serialize(Archive& ar) const {
    // serialize array type, world size, rank, and pmap type to be able
    // to ensure same data type and same data distribution expected
    ar& typeid(*this).hash_code() & world().size() & world().rank() & trange() &
        shape() & typeid(pmap().get()).hash_code();
    int64_t count = 0;
    for (auto it = begin(); it != end(); ++it) ++count;
    ar& count;
    for (auto it = begin(); it != end(); ++it) ar & it->get();
  }

  /// deserialize local contents of a DistArray from an Archive object

  /// @note use Parallel{Input,Output}Archive for parallel serialization
  /// @tparam Archive an Archive type
  /// @warning this does not fence; it is user's responsibility to do that
  template <typename Archive,
            typename = std::enable_if_t<
                !Archive::is_parallel_archive &&
                madness::is_input_archive_v<std::decay_t<Archive>>>>
  void serialize(Archive& ar) {
    auto& world = TiledArray::get_default_world();

    std::size_t typeid_hash = 0l;
    ar& typeid_hash;
    if (typeid_hash != typeid(*this).hash_code())
      TA_EXCEPTION(
          "DistArray::serialize: source DistArray type != this DistArray type");

    ProcessID world_size = -1;
    ProcessID world_rank = -1;
    ar& world_size& world_rank;
    if (world_size != world.size() || world_rank != world.rank())
      TA_EXCEPTION(
          "DistArray::serialize: source DistArray world != this DistArray "
          "world");

    trange_type trange;
    shape_type shape;
    ar& trange& shape;

    // use default pmap, ensure it's the same pmap used to serialize
    auto volume = trange.tiles_range().volume();
    auto pmap = detail::policy_t<DistArray>::default_pmap(world, volume);
    size_t pmap_hash_code = 0;
    ar& pmap_hash_code;
    if (pmap_hash_code != typeid(pmap.get()).hash_code())
      TA_EXCEPTION(
          "DistArray::serialize: source DistArray pmap != this DistArray pmap");
    pimpl_.reset(
        new impl_type(world, std::move(trange), std::move(shape), pmap));

    int64_t count = 0;
    ar& count;
    for (auto it = begin(); it != end(); ++it, --count) {
      Tile tile;
      ar& tile;
      this->set(it.ordinal(), std::move(tile));
    }
    if (count != 0)
      TA_EXCEPTION(
          "DistArray::serialize: # of tiles in archive != # of tiles expected");
  }

  /// Replaces this array with one loaded from an Archive object using the
  /// default processor map

  /// @tparam Archive a parallel MADWorld Archive type
  /// @param world a World object with which this object will be associated
  /// @param ar an Archive object from which this object's data will be read
  ///
  /// @note The & operator for serializing will only work with parallel
  ///       MADWorld archives.
  /// @note This is a collective operation that fences before and after
  ///       completion, if @c ar.dofence() is true
  template <typename Archive>
  void load(World& world, Archive& ar) {
    auto me = world.rank();
    const Tag tag = world.mpi.unique_tag();  // for broadcasting metadata

    if (ar.dofence()) world.gop.fence();

    if (ar.is_io_node()) {  // on each io node ...

      auto& localar = ar.local_archive();

      // make sure source data matches the expected type
      // TODO would be nice to be able to convert the data upon reading
      std::size_t typeid_hash = 0l;
      localar& typeid_hash;
      if (typeid_hash != typeid(*this).hash_code())
        TA_EXCEPTION(
            "DistArray::load: source DistArray type != this DistArray type");

      // make sure same number of clients for every I/O node
      int num_io_clients = 0;
      localar& num_io_clients;
      if (num_io_clients != ar.num_io_clients())
        TA_EXCEPTION("DistArray::load: invalid parallel archive");

      trange_type trange;
      shape_type shape;
      localar& trange& shape;

      // send trange and shape to every client
      for (ProcessID p = 0; p < world.size(); ++p) {
        if (p != me && ar.io_node(p) == me) {
          world.mpi.Send(int(1), p, tag);  // Tell client to expect the data
          madness::archive::MPIOutputArchive dest(world, p);
          dest& trange& shape;
          dest.flush();
        }
      }

      // use default pmap
      auto volume = trange.tiles_range().volume();
      auto pmap = detail::policy_t<DistArray>::default_pmap(world, volume);
      pimpl_.reset(
          new impl_type(world, std::move(trange), std::move(shape), pmap));

      int64_t count = 0;
      localar& count;
      for (size_t ord = 0; ord != volume; ++ord) {
        if (!is_zero(ord)) {
          auto owner_rank = pmap->owner(ord);
          if (ar.io_node(owner_rank) == me) {
            Tile tile;
            localar& tile;
            this->set(ord, std::move(tile));
            --count;
          }
        }
      }
      if (count != 0)
        TA_EXCEPTION(
            "DistArray::load: # of tiles in archive != # of tiles expected");
    } else {  // non-I/O node still needs to initialize metadata

      trange_type trange;
      shape_type shape;

      ProcessID p = ar.my_io_node();
      int flag;
      world.mpi.Recv(flag, p, tag);
      TA_ASSERT(flag == 1);
      madness::archive::MPIInputArchive source(world, p);
      source& trange& shape;

      // use default pmap
      auto volume = trange.tiles_range().volume();
      auto pmap = detail::policy_t<DistArray>::default_pmap(world, volume);
      pimpl_.reset(
          new impl_type(world, std::move(trange), std::move(shape), pmap));
    }

    if (ar.dofence()) world.gop.fence();
  }

  /// Stores this array to an Archive object

  /// @tparam Archive a parallel MADWorld Archive type
  /// @param ar an Archive object that will contain this object's data
  ///
  /// @note The & operator for serializing will only work with parallel
  ///       MADWorld archives.
  /// @note this is a collective operation that fences before and after
  ///       completion if @c ar.dofence() is true
  template <typename Archive>
  void store(Archive& ar) const {
    auto me = world().rank();

    if (ar.dofence()) world().gop.fence();

    if (ar.is_io_node()) {  // on each io node ...
      auto& localar = ar.local_archive();
      // ... store metadata first ...
      localar& typeid(*this).hash_code() & ar.num_io_clients() & trange() &
          shape();
      // ... then loop over tiles and dump the data from ranks
      // assigned to this I/O node in order ...
      // for sanity check dump tile count assigned to this I/O node
      const auto volume = trange().tiles_range().volume();
      int64_t count = 0;
      for (size_t ord = 0; ord != volume; ++ord) {
        if (!is_zero(ord)) {
          const auto owner_rank = pmap()->owner(ord);
          if (ar.io_node(owner_rank) == me) {
            ++count;
          }
        }
      }
      localar& count;
      for (size_t ord = 0; ord != volume; ++ord) {
        if (!is_zero(ord)) {
          auto owner_rank = pmap()->owner(ord);
          if (ar.io_node(owner_rank) == me) localar& find(ord).get();
        }
      }
    }  // am I an I/O node?
    if (ar.dofence()) world().gop.fence();
  }

  /// Debugging/tracing instrumentation

  /// registers notifier for set() calls
  /// @param notifier the notifier callable that accepts ref to the
  /// implementation object that received set call
  ///        and the corresponding ordinal index
  static void register_set_notifier(
      std::function<void(const impl_type&, int64_t)> notifier = {}) {
    impl_type::set_notifier_accessor() = notifier;
  }

 private:
  template <typename Index>
  std::enable_if_t<std::is_integral_v<Index>, void> check_index(
      const Index i) const {
    TA_ASSERT(
        impl_ref().tiles_range().includes(i) &&
        "The ordinal index used to access an array tile is out of range.");
  }

  template <typename Index>
  std::enable_if_t<detail::is_integral_range_v<Index>, void> check_index(
      const Index& i) const {
    TA_ASSERT(
        impl_ref().tiles_range().includes(i) &&
        "The coordinate index used to access an array tile is out of range.");
  }

  template <typename Index1>
  void check_index(const std::initializer_list<Index1>& i) const {
    check_index<std::initializer_list<Index1>>(i);
  }

  template <typename Index>
  std::enable_if_t<std::is_integral_v<Index>, void> check_local_index(
      const Index i) const {
    check_index(i);
    TA_ASSERT(pimpl_->is_local(i)  // pimpl_ already checked
              &&
              "The ordinal index used to access an array tile is not local.");
  }

  template <typename Index>
  std::enable_if_t<detail::is_integral_range_v<Index>, void> check_local_index(
      const Index& i) const {
    check_index(i);
    TA_ASSERT(
        pimpl_->is_local(i)  // pimpl_ already checked
        && "The coordinate index used to access an array tile is not local.");
  }

  template <typename Index1>
  void check_local_index(const std::initializer_list<Index1>& i) const {
    check_local_index<std::initializer_list<Index1>>(i);
  }

  /// Ensures that the string indices are consistent with the tensor
  ///
  /// This function checks that PIMPL has been
  /// @param[in] vars The string indices, such as `"i, j"`, the user provided to
  ///                 label the modes.
  /// @warning this tests contents of @p vars using #TA_ASSERT() only if
  /// preprocessor macro @c NDEBUG is not defined
  void check_str_index(const std::string& vars) const {
#if (TA_ASSERT_POLICY != TA_ASSERT_IGNORE)
    // Only check indices if the PIMPL is initialized (okay to not initialize
    // the RHS of an equation)
    if (!is_initialized()) return;

    constexpr bool is_tot = detail::is_tensor_of_tensor_v<value_type>;
    const auto rank = tiles_range().rank();
    // TODO: Make constexpr and use structured bindings when CUDA supports C++17
    if (is_tot) {
      // Make sure the index is capable of being interpreted as a ToT index
      TA_ASSERT(detail::is_tot_index(vars));

      // Rank of outer tiles must match number of outer indices
      // is_tot_index(vars) implies vars.find(';') < vars.size()
      TA_ASSERT(std::count(vars.begin(), vars.begin() + vars.find(';'), ',') +
                    1ul ==
                rank);

      // Check inner index rank?
    } else {
      // Better not be a ToT index
      TA_ASSERT(!detail::is_tot_index(vars));

      // Number of indices must match rank
      TA_ASSERT(std::count(vars.begin(), vars.end(), ',') + 1ul == rank);
    }
#endif  // NDEBUG
  }

  /// Code factorization of the actual assert for the other overloads
  void assert_pimpl() const {
    TA_ASSERT(pimpl_ &&
              "The Array has not been initialized, likely reason: it was "
              "default constructed and used.");
  }

  /// If this is in an initialized state this returns a const
  /// reference to implementation object.
  /// This makes the common scenario of: check-pimpl then use *pimpl, into a
  /// one-liner.
  ///
  /// \return A const reference to the implementation object.
  /// \throw TiledArray::Exception if this is in an uninitialized state
  /// and the NDEBUG preprocessor macro is not defined. Strong
  /// throw guarantee.
  auto& impl_ref() const {
    assert_pimpl();
    return *pimpl_;
  }

  /// If this is in an initialized state this returns a non-const
  /// reference to implementation object.
  /// This makes the common scenario of: check-pimpl then use *pimpl, into a
  /// one-liner.
  ///
  /// \return A reference to the implementation object.
  /// \throw TiledArray::Exception if this is in an uninitialized state
  /// and the NDEBUG preprocessor macro is not defined. Strong
  /// throw guarantee.
  auto& impl_ref() {
    assert_pimpl();
    return *pimpl_;
  }

};  // class DistArray

template <typename Tile, typename Policy>
madness::AtomicInt DistArray<Tile, Policy>::cleanup_counter_;

#ifndef TILEDARRAY_HEADER_ONLY

extern template class DistArray<Tensor<double>, DensePolicy>;
extern template class DistArray<Tensor<float>, DensePolicy>;
// extern template class DistArray<Tensor<int>,
//                                DensePolicy>;
// extern template class DistArray<Tensor<long>,
//                                DensePolicy>;
extern template class DistArray<Tensor<std::complex<double>>, DensePolicy>;
extern template class DistArray<Tensor<std::complex<float>>, DensePolicy>;

extern template class DistArray<Tensor<double>, SparsePolicy>;
extern template class DistArray<Tensor<float>, SparsePolicy>;
// extern template class DistArray<Tensor<int>,
//                                SparsePolicy>;
// extern template class DistArray<Tensor<long>,
//                                SparsePolicy>;
extern template class DistArray<Tensor<std::complex<double>>, SparsePolicy>;
extern template class DistArray<Tensor<std::complex<float>>, SparsePolicy>;

#endif  // TILEDARRAY_HEADER_ONLY

/// Add the tensor to an output stream

/// This function will iterate through all tiles on node 0 and print non-zero
/// tiles. It will wait for each tile to be evaluated (i.e. it is a blocking
/// function). Tasks will continue to be processed.
/// \tparam T The element type of Array
/// \tparam Tile The Tile type
/// \param os The output stream
/// \param a The array to be put in the output stream
/// \return A reference to the output stream
template <typename Tile, typename Policy>
inline std::ostream& operator<<(std::ostream& os,
                                const DistArray<Tile, Policy>& a) {
  if (a.world().rank() == 0) {
    for (std::size_t i = 0; i < a.size(); ++i)
      if (!a.is_zero(i)) {
        const typename DistArray<Tile, Policy>::value_type tile =
            a.find(i).get();
        os << i << ": " << tile << "\n";
      }
  }
  a.world().gop.fence();
  return os;
}

template <typename Tile, typename Policy>
auto rank(const DistArray<Tile, Policy>& a) {
  return a.trange().tiles_range().rank();
}

template <typename Tile, typename Policy>
size_t volume(const DistArray<Tile, Policy>& a) {
  // this is the number of tiles
  if (a.size() > 0)  // assuming dense shape
    return a.trange().elements_range().volume();
  return 0;
}

template <typename Tile, typename Policy>
auto abs_min(const DistArray<Tile, Policy>& a) {
  return a(detail::dummy_annotation(rank(a))).abs_min();
}

template <typename Tile, typename Policy>
auto abs_max(const DistArray<Tile, Policy>& a) {
  return a(detail::dummy_annotation(rank(a))).abs_max();
}

template <typename Tile, typename Policy>
auto dot(const DistArray<Tile, Policy>& a, const DistArray<Tile, Policy>& b) {
  return (a(detail::dummy_annotation(rank(a)))
              .dot(b(detail::dummy_annotation(rank(b)))))
      .get();
}

template <typename Tile, typename Policy>
auto inner_product(const DistArray<Tile, Policy>& a,
                   const DistArray<Tile, Policy>& b) {
  return (a(detail::dummy_annotation(rank(a)))
              .inner_product(b(detail::dummy_annotation(rank(b)))))
      .get();
}

template <typename Tile, typename Policy>
auto squared_norm(const DistArray<Tile, Policy>& a) {
  return a(detail::dummy_annotation(rank(a))).squared_norm();
}

template <typename Tile, typename Policy>
auto norm2(const DistArray<Tile, Policy>& a) {
  return std::sqrt(squared_norm(a));
}

template<typename Array, typename Tiles>
Array make_array(
  World &world,
  const detail::trange_t<Array> &tiled_range,
  Tiles begin, Tiles end)
{
  Array array;
  using Tuple = std::remove_reference_t<decltype(*begin)>;
  using Index = std::tuple_element_t<0,Tuple>;
  using shape_type = typename Array::shape_type;
  if constexpr (shape_type::is_dense()) {
    array = Array(world, tiled_range);
  }
  else {
    std::vector< std::pair<Index,float> > tile_norms;
    for (Tiles it = begin; it != end; ++it) {
      auto [index,tile] = *it;
      tile_norms.push_back({index,tile.norm()});
    }
    shape_type shape(world, tile_norms, tiled_range);
    array = Array(world, tiled_range, shape);
  }
  for (Tiles it = begin; it != end; ++it) {
    auto [index,tile] = *it;
    if (array.is_zero(index)) continue;
    array.set(index,tile);
  }
  return array;
}



}  // namespace TiledArray

// serialization
namespace madness {
namespace archive {
template <class Tile, class Policy>
struct ArchiveLoadImpl<ParallelInputArchive<>,
                       TiledArray::DistArray<Tile, Policy>> {
  static inline void load(const ParallelInputArchive<>& ar,
                          TiledArray::DistArray<Tile, Policy>& x) {
    x.load(*ar.get_world(), ar);
  }
};

template <class Tile, class Policy>
struct ArchiveStoreImpl<ParallelOutputArchive<>,
                        TiledArray::DistArray<Tile, Policy>> {
  static inline void store(const ParallelOutputArchive<>& ar,
                           const TiledArray::DistArray<Tile, Policy>& x) {
    x.store(ar);
  }
};
}  // namespace archive

template <class Tile, class Policy>
void save(const TiledArray::DistArray<Tile, Policy>& x,
          const std::string name) {
  archive::ParallelOutputArchive<> ar2(x.world(), name.c_str(), 1);
  ar2& x;
}

template <class Tile, class Policy>
void load(TiledArray::DistArray<Tile, Policy>& x, const std::string name) {
  archive::ParallelInputArchive<> ar2(x.world(), name.c_str(), 1);
  ar2& x;
}

}  // namespace madness

#endif  // TILEDARRAY_ARRAY_H__INCLUDED
