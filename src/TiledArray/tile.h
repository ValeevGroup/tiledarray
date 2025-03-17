/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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

#ifndef TILEDARRAY_TILE_H__INCLUDED
#define TILEDARRAY_TILE_H__INCLUDED

#include <TiledArray/tensor/tensor_interface.h>
#include <TiledArray/tile_interface/cast.h>
#include <TiledArray/tile_interface/trace.h>
#include <memory>

namespace TiledArray {

/**
 * \defgroup TileInterface Tile interface for user defined tensor types
 * @{
 */

/// An N-dimensional shallow-copy wrapper for Tensor-like types that, unlike
/// Tensor, have deep-copy semantics. Like Tensor, Tile is
/// default-constructible. The default constructor produced a Tile in
/// null state (not referring to any tensor object). The name refers to its
/// intended use as a tile of DistArray.
///
/// \tparam T a tensor type. It may provide a subset of the full operation
/// set of Tensor, since only those operations that are actually used
/// need to be defined. For full equivalence to Tensor \p T must define the
/// following functions, either as members or as non-member functions (see the
/// \ref NonIntrusiveTileInterface "non-intrusive tile interface"
/// documentation for more details on the latter):
/// \li \c add
/// \li \c add_to (in-place add)
/// \li \c subt
/// \li \c subt_to  (in-place subt)
/// \li \c mult
/// \li \c mult_to (in-place mult)
/// \li \c scale
/// \li \c scale_to  (in-place scale)
/// \li \c gemm
/// \li \c neg
/// \li \c permute
/// \li \c empty
/// \li \c shift
/// \li \c shift_to  (in-place shift)
/// \li \c trace
/// \li \c sum
/// \li \c product
/// \li \c squared_norm
/// \li \c norm
/// \li \c min
/// \li \c max
/// \li \c abs_min
/// \li \c abs_max
/// \li \c dot
///
template <typename T>
class Tile {
 public:
  /// This object type
  typedef Tile<T> Tile_;
  /// Tensor type used to represent tile data
  typedef T tensor_type;
  // import types from T
  using value_type = typename tensor_type::value_type;    ///< value type
  using range_type = typename tensor_type::range_type;    ///< Tensor range type
  using index1_type = typename tensor_type::index1_type;  ///< 1-index type
  using size_type =
      typename tensor_type::ordinal_type;  ///< Size type (to meet the container
                                           ///< concept)
  using reference =
      typename tensor_type::reference;  ///< Element reference type
  using const_reference =
      typename tensor_type::const_reference;        ///< Element reference type
  using iterator = typename tensor_type::iterator;  ///< Element iterator type
  using const_iterator =
      typename tensor_type::const_iterator;  ///< Element const iterator type
  using pointer = typename tensor_type::pointer;  ///< Element pointer type
  using const_pointer =
      typename tensor_type::const_pointer;  ///< Element const pointer type
  using numeric_type = typename TiledArray::detail::numeric_type<
      tensor_type>::type;  ///< the numeric type that supports T
  using scalar_type = typename TiledArray::detail::scalar_type<
      tensor_type>::type;  ///< the scalar type that supports T

 private:
  template <typename Element, typename = void>
  struct rebind;
  template <typename Element>
  struct rebind<Element, std::enable_if_t<detail::has_rebind_v<T, Element>>> {
    using type = Tile<typename T::template rebind_t<Element>>;
  };

  template <typename Numeric, typename = void>
  struct rebind_numeric;
  template <typename Numeric>
  struct rebind_numeric<
      Numeric, std::enable_if_t<detail::has_rebind_numeric_v<T, Numeric>>> {
    using type = Tile<typename T::template rebind_numeric_t<Numeric>>;
  };

 public:
  /// compute type of Tile<T> with different element type
  template <typename ElementType>
  using rebind_t = typename rebind<ElementType>::type;

  /// compute type of Tile<T> with different numeric type
  template <typename NumericType>
  using rebind_numeric_t = typename rebind_numeric<NumericType>::type;

 private:
  std::shared_ptr<tensor_type> pimpl_;

 public:
  // Constructors and destructor ---------------------------------------------

  Tile() = default;
  Tile(const Tile_&) = default;
  Tile(Tile_&&) = default;

  /// Forwarding ctor

  /// To simplify construction, Tile provides ctors that all forward their args
  /// to T. To avoid clashing with copy and move ctors need conditional
  /// instantiation -- e.g. see
  /// http://ericniebler.com/2013/08/07/universal-references-and-the-copy-constructo/
  /// NB For Arg that can be converted to Tile also use the copy/move ctors.
  template <typename Arg,
            typename = typename std::enable_if<
                not detail::is_same_or_derived<Tile_, Arg>::value &&
                not std::is_convertible<Arg, Tile_>::value &&
                not TiledArray::detail::is_explicitly_convertible<
                    Arg, Tile_>::value>::type>
  explicit Tile(Arg&& arg)
      : pimpl_(std::make_shared<tensor_type>(std::forward<Arg>(arg))) {}

  template <typename Arg1, typename Arg2, typename... Args>
  Tile(Arg1&& arg1, Arg2&& arg2, Args&&... args)
      : pimpl_(std::make_shared<tensor_type>(std::forward<Arg1>(arg1),
                                             std::forward<Arg2>(arg2),
                                             std::forward<Args>(args)...)) {}

  ~Tile() = default;

  // Assignment operators ----------------------------------------------------

  Tile_& operator=(Tile_&&) = default;
  Tile_& operator=(const Tile_&) = default;

  Tile_& operator=(const tensor_type& tensor) {
    *pimpl_ = tensor;
    return *this;
  }

  Tile_& operator=(tensor_type&& tensor) {
    *pimpl_ = std::move(tensor);
    return *this;
  }

  // State accessor ----------------------------------------------------------

  /// \return true if this is null (default-constructed or
  /// after reset()) OR if the referred object is in null state (i.e. if
  /// `tensor().empty()` is true.
  /// \note use use_count() to check if this is in a null state
  bool empty() const { return pimpl_ ? pimpl_->empty() : true; }

  /// \return the number of Tile objects that refer to the same tensor
  /// as this (if any); `0` is returned if this is in a null state
  /// (default-constructed or
  /// after reset()).
  long use_count() const { return pimpl_.use_count(); }

  // State operations --------------------------------------------------------

  /// release the reference to the managed tensor, and delete it
  /// if this is the last Tile object that refers to it.
  /// \post this object is in a null state
  void reset() { pimpl_.reset(); }

  // Tile accessor -----------------------------------------------------------

  tensor_type& tensor() { return *pimpl_; }

  const tensor_type& tensor() const { return *pimpl_; }

  // Iterator accessor -------------------------------------------------------

  /// Iterator factory

  /// \return An iterator to the first data element
  decltype(auto) begin() { return std::begin(tensor()); }

  /// Iterator factory

  /// \return A const iterator to the first data element
  decltype(auto) begin() const { return std::begin(tensor()); }

  /// Iterator factory

  /// \return An iterator to the last data element
  decltype(auto) end() { return std::end(tensor()); }

  /// Iterator factory

  /// \return A const iterator to the last data element
  decltype(auto) end() const { return std::end(tensor()); }

  /// Iterator factory

  /// \return A const iterator to the first data element
  decltype(auto) cbegin() { return std::cbegin(tensor()); }

  /// Iterator factory

  /// \return A const iterator to the first data element
  decltype(auto) cbegin() const { return std::cbegin(tensor()); }

  /// Iterator factory

  /// \return A const iterator to the last data element
  decltype(auto) cend() { return std::cend(tensor()); }

  /// Iterator factory

  /// \return A const iterator to the last data element
  decltype(auto) cend() const { return std::cend(tensor()); }

  // Data accessor -------------------------------------------------------

  /// Data direct access

  /// \return A pointer to the tensor data
  decltype(auto) data() { return tensor().data(); }

  /// Data direct access

  /// \return A const pointer to the tensor data
  decltype(auto) data() const { return tensor().data(); }

  // Dimension information accessors -----------------------------------------

  /// Size accessor

  /// \return The number of elements in the tensor
  decltype(auto) size() const { return tensor().size(); }

  /// Total size accessor

  /// \return The number of elements in the tensor, tallied across batches (if
  /// any)
  decltype(auto) total_size() const {
    if constexpr (detail::has_member_function_total_size_anyreturn_v<
                      tensor_type>) {
      return tensor().total_size();
    } else
      return size();
  }

  /// Range accessor

  /// \return An object describes the upper and lower bounds of  the tensor data
  decltype(auto) range() const { return tensor().range(); }

  // Element accessors -------------------------------------------------------

  /// Const element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Const reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  const_reference operator[](const Ordinal ord) const {
    TA_ASSERT(pimpl_);
    // can't distinguish between operator[](Index...) and operator[](ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range().rank() != 1 &&
              "use Tile::operator[](index) or "
              "Tile::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(tensor().range().includes_ordinal(ord));
    return tensor().data()[ord];
  }

  /// Element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  reference operator[](const Ordinal ord) {
    TA_ASSERT(pimpl_);
    // can't distinguish between operator[](Index...) and operator[](ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range().rank() != 1 &&
              "use Tile::operator[](index) or "
              "Tile::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(tensor().range().includes_ordinal(ord));
    return tensor().data()[ord];
  }

  /// Const element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Const reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  const_reference at_ordinal(const Ordinal ord) const {
    TA_ASSERT(pimpl_);
    TA_ASSERT(tensor().range().includes_ordinal(ord));
    return tensor().data()[ord];
  }

  /// Element accessor

  /// \tparam Ordinal an integer type that represents an ordinal
  /// \param[in] ord an ordinal index
  /// \return Reference to the element at position \c ord .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Ordinal,
            std::enable_if_t<std::is_integral<Ordinal>::value>* = nullptr>
  reference at_ordinal(const Ordinal ord) {
    TA_ASSERT(pimpl_);
    TA_ASSERT(tensor().range().includes_ordinal(ord));
    return tensor().data()[ord];
  }

  /// Const element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  const_reference operator[](const Index& i) const {
    TA_ASSERT(pimpl_);
    TA_ASSERT(tensor().range().includes(i));
    return tensor().data()[tensor().range().ordinal(i)];
  }

  /// Element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  reference operator[](const Index& i) {
    TA_ASSERT(pimpl_);
    TA_ASSERT(tensor().range().includes(i));
    return tensor().data()[tensor().range().ordinal(i)];
  }

  /// Const element accessor

  /// \tparam Index An integral type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  const_reference operator[](const std::initializer_list<Index>& i) const {
    TA_ASSERT(pimpl_);
    TA_ASSERT(tensor().range().includes(i));
    return tensor().data()[tensor().range().ordinal(i)];
  }

  /// Element accessor

  /// \tparam Index An integral type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  reference operator[](const std::initializer_list<Index>& i) {
    TA_ASSERT(pimpl_);
    TA_ASSERT(tensor().range().includes(i));
    return tensor().data()[tensor().range().ordinal(i)];
  }

  /// Const element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  const_reference operator()(const Index& i) const {
    TA_ASSERT(pimpl_);
    TA_ASSERT(tensor().range().includes(i));
    return tensor().data()[tensor().range().ordinal(i)];
  }

  /// Element accessor

  /// \tparam Index An integral range type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            std::enable_if_t<detail::is_integral_range_v<Index>>* = nullptr>
  reference operator()(const Index& i) {
    TA_ASSERT(pimpl_);
    TA_ASSERT(tensor().range().includes(i));
    return tensor().data()[tensor().range().ordinal(i)];
  }

  /// Const element accessor

  /// \tparam Index An integral type
  /// \param[in] i an index
  /// \return Const reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  const_reference operator()(const std::initializer_list<Index>& i) const {
    TA_ASSERT(pimpl_);
    TA_ASSERT(tensor().range().includes(i));
    return tensor().data()[tensor().range().ordinal(i)];
  }

  /// Element accessor

  /// \tparam Index An integral type
  /// \param[in] i an index
  /// \return Reference to the element at position \c i .
  /// \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  reference operator()(const std::initializer_list<Index>& i) {
    TA_ASSERT(pimpl_);
    TA_ASSERT(tensor().range().includes(i));
    return tensor().data()[tensor().range().ordinal(i)];
  }

  /// Const element accessor

  /// \tparam Index an integral list ( see TiledArray::detail::is_integral_list
  /// ) \param[in] i an index \return Const reference to the element at position
  /// \c i . \note This asserts (using TA_ASSERT) that this is not empty and ord
  /// is included in the range
  template <
      typename... Index,
      std::enable_if_t<(sizeof...(Index) > 1ul) &&
                       detail::is_integral_list<Index...>::value>* = nullptr>
  const_reference operator()(const Index&... i) const {
    TA_ASSERT(pimpl_);
    TA_ASSERT(this->range().rank() == sizeof...(Index));
    // can't distinguish between operator()(Index...) and operator()(ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range().rank() != 1 &&
              "use Tile::operator()(index) or "
              "Tile::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(tensor().range().includes(i...));
    return tensor().data()[tensor().range().ordinal(i...)];
  }

  /// Element accessor

  /// \tparam Index an integral list ( see TiledArray::detail::is_integral_list
  /// ) \param[in] i an index \return Reference to the element at position \c i
  /// . \note This asserts (using TA_ASSERT) that this is not empty and ord is
  /// included in the range
  template <
      typename... Index,
      std::enable_if_t<(sizeof...(Index) > 1ul) &&
                       detail::is_integral_list<Index...>::value>* = nullptr>
  reference operator()(const Index&... i) {
    TA_ASSERT(pimpl_);
    TA_ASSERT(this->range().rank() == sizeof...(Index));
    // can't distinguish between operator()(Index...) and operator()(ordinal)
    // thus insist on at_ordinal() if this->rank()==1
    TA_ASSERT(this->range().rank() != 1 &&
              "use Tile::operator()(index) or "
              "Tile::at_ordinal(index_ordinal) if this->range().rank()==1");
    TA_ASSERT(tensor().range().includes(i...));
    return tensor().data()[tensor().range().ordinal(i...)];
  }

  // Block accessors -----------------------------------------------------------

  // clang-format off
  /// Constructs a view of the block defined by \p lower_bound and \p upper_bound.

  /// Examples of using this:
  /// \code
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///   auto tview = t.block(lobounds, upbounds);
  ///   assert(tview.range().includes(lobounds));
  ///   assert(tview(lobounds) == t(lobounds));
  /// \endcode
  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param lower_bound The lower bound
  /// \param upper_bound The upper bound
  /// \return a {const,mutable} view of the block defined by \p lower_bound and \p upper_bound
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `lower_bound[i] >= upper_bound[i]`
  // clang-format on
  /// @{
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  decltype(auto) block(const Index1& lower_bound, const Index2& upper_bound) {
    TA_ASSERT(pimpl_);
    return detail::TensorInterface<value_type, BlockRange, tensor_type>(
        BlockRange(tensor().range(), lower_bound, upper_bound),
        tensor().data());
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  decltype(auto) block(const Index1& lower_bound,
                       const Index2& upper_bound) const {
    TA_ASSERT(pimpl_);
    return detail::TensorInterface<const value_type, BlockRange, tensor_type>(
        BlockRange(tensor().range(), lower_bound, upper_bound),
        tensor().data());
  }
  /// @}

  // clang-format off
  /// Constructs a view of the block defined by \p lower_bound and \p upper_bound.

  /// Examples of using this:
  /// \code
  ///   auto tview = t.block({0, 1, 2}, {4, 6, 8});
  ///   assert(tview.range().includes(lobounds));
  ///   assert(tview(lobounds) == t(lobounds));
  /// \endcode
  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \param lower_bound The lower bound
  /// \param upper_bound The upper bound
  /// \return a {const,mutable} view of the block defined by \p lower_bound and \p upper_bound
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `lower_bound[i] >= upper_bound[i]`
  // clang-format on
  /// @{
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  decltype(auto) block(const std::initializer_list<Index1>& lower_bound,
                       const std::initializer_list<Index2>& upper_bound) {
    TA_ASSERT(pimpl_);
    return detail::TensorInterface<value_type, BlockRange, tensor_type>(
        BlockRange(tensor().range(), lower_bound, upper_bound),
        tensor().data());
  }

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  decltype(auto) block(const std::initializer_list<Index1>& lower_bound,
                       const std::initializer_list<Index2>& upper_bound) const {
    TA_ASSERT(pimpl_);
    return detail::TensorInterface<const value_type, BlockRange, tensor_type>(
        BlockRange(tensor().range(), lower_bound, upper_bound),
        tensor().data());
  }
  /// @}

  // clang-format off
  /// Constructs a view of the block defined by its \p bounds .

  /// Examples of using this:
  /// \code
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///
  ///   // using vector of pairs
  ///   std::vector<std::pair<size_t,size_t>> vpbounds{{0,4}, {1,6}, {2,8}};
  ///   auto tview0 = t.block(vpbounds);
  ///   // using vector of tuples
  ///   std::vector<std::tuple<size_t,size_t>> vtbounds{{0,4}, {1,6}, {2,8}};
  ///   auto tview1 = t.block(vtbounds);
  ///   assert(tview0 == tview1);
  ///
  ///   // using zipped ranges of bounds (using Boost.Range)
  ///   // need to #include <boost/range/combine.hpp>
  ///   auto tview2 = t.block(boost::combine(lobounds, upbounds));
  ///   assert(tview0 == tview2);
  ///
  ///   // using zipped ranges of bounds (using Ranges-V3)
  ///   // need to #include <range/v3/view/zip.hpp>
  ///   auto tview3 = t.block(ranges::views::zip(lobounds, upbounds));
  ///   assert(tview0 == tview3);
  /// \endcode
  /// \tparam PairRange Type representing a range of generalized pairs (see TiledArray::detail::is_gpair_v )
  /// \param bounds The block bounds
  /// \return a {const,mutable} view of the block defined by its \p bounds
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `get<0>(bounds[i]) >= get<1>(bounds[i])`
  // clang-format on
  /// @{
  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  decltype(auto) block(const PairRange& bounds) {
    TA_ASSERT(pimpl_);
    return detail::TensorInterface<value_type, BlockRange, tensor_type>(
        BlockRange(tensor().range(), bounds), tensor().data());
  }

  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  decltype(auto) block(const PairRange& bounds) const {
    TA_ASSERT(pimpl_);
    return detail::TensorInterface<const value_type, BlockRange, tensor_type>(
        BlockRange(tensor().range(), bounds), tensor().data());
  }
  /// @}

  // clang-format off
  /// Constructs a view of the block defined by its \p bounds .

  /// Examples of using this:
  /// \code
  ///   auto tview0 = t.block({{0,4}, {1,6}, {2,8}});
  /// \endcode
  /// \tparam Index An integral type
  /// \param bounds The block bounds
  /// \return a {const,mutable} view of the block defined by its \p bounds
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `get<0>(bounds[i]) >= get<1>(bounds[i])`
  // clang-format on
  /// @{
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  decltype(auto) block(
      const std::initializer_list<std::initializer_list<Index>>& bounds) {
    TA_ASSERT(pimpl_);
    return detail::TensorInterface<value_type, BlockRange, tensor_type>(
        BlockRange(tensor().range(), bounds), tensor().data());
  }

  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  decltype(auto) block(
      const std::initializer_list<std::initializer_list<Index>>& bounds) const {
    TA_ASSERT(pimpl_);
    return detail::TensorInterface<const value_type, BlockRange, tensor_type>(
        BlockRange(tensor().range(), bounds), tensor().data());
  }
  /// @}

  // Serialization -----------------------------------------------------------

  template <typename Archive,
            typename std::enable_if<
                madness::is_output_archive_v<Archive>>::type* = nullptr>
  void serialize(Archive& ar) const {
    // Serialize data for empty tile check
    bool empty = !static_cast<bool>(pimpl_);
    ar & empty;
    if (!empty) {
      // Serialize tile data
      ar&* pimpl_;
    }
  }

  template <typename Archive,
            typename std::enable_if<
                madness::is_input_archive_v<Archive>>::type* = nullptr>
  void serialize(Archive& ar) {
    // Check for empty tile
    bool empty = false;
    ar & empty;

    if (!empty) {
      // Deserialize tile data
      tensor_type tensor;
      ar & tensor;

      // construct a new pimpl
      pimpl_ = std::make_shared<T>(std::move(tensor));
    } else {
      // Set pimpl to an empty tile
      pimpl_.reset();
    }
  }

  constexpr static std::size_t nbatch() { return 1; }

  const auto& batch(std::size_t idx) const {
    TA_ASSERT(idx < this->nbatch());
    return *this;
  }

};  // class Tile

// The following functions define the non-intrusive interface used to apply
// math operations to Tiles. These functions in turn use the non-intrusive
// interface functions to evaluate tiles.

namespace detail {

/// Factory function for tiles

/// \tparam T A tensor type
/// \param t A tensor object
/// \return A tile that wraps a copy of t.
template <typename T>
Tile<T> make_tile(T&& t) {
  return Tile<T>(std::forward<T>(t));
}

}  // namespace detail

// Clone operations ----------------------------------------------------------

/// Create a copy of \c arg

/// \tparam Arg The tile argument type
/// \param arg The tile argument to be permuted
/// \return A (deep) copy of \c arg
template <typename Arg>
inline Tile<Arg> clone(const Tile<Arg>& arg) {
  return Tile<Arg>(clone(arg.tensor()));
}

#if __cplusplus <= 201402L
// Empty operations ----------------------------------------------------------

/// Check that \c arg is empty (no data)

/// \tparam Arg The tile argument type
/// \param arg The tile argument to be permuted
/// \return \c true if \c arg is empty, otherwise \c false.
template <typename Arg>
inline bool empty(const Tile<Arg>& arg) {
  return arg.empty() || empty(arg.tensor());
}
#endif

// Permutation operations ----------------------------------------------------

/// Create a permuted copy of \c arg

/// \tparam Arg The tile argument type
/// \tparam Perm A permutation tile
/// \param arg The tile argument to be permuted
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ arg</tt>
template <typename Arg, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline decltype(auto) permute(const Tile<Arg>& arg, const Perm& perm) {
  return Tile<Arg>(permute(arg.tensor(), perm));
}

// Shift operations ----------------------------------------------------------

/// Shift the range of \c arg

/// \tparam Arg The tensor argument type
/// \tparam Index An integral range type
/// \param arg The tile argument to be shifted
/// \param range_shift The offset to be applied to the argument range
/// \return A copy of the tile with a new range
template <typename Arg, typename Index,
          typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
inline decltype(auto) shift(const Tile<Arg>& arg, const Index& range_shift) {
  return detail::make_tile(shift(arg.tensor(), range_shift));
}

/// Shift the range of \c arg

/// \tparam Arg The tensor argument type
/// \tparam Index An integral type
/// \param arg The tile argument to be shifted
/// \param range_shift The offset to be applied to the argument range
/// \return A copy of the tile with a new range
template <typename Arg, typename Index,
          typename = std::enable_if_t<std::is_integral_v<Index>>>
inline decltype(auto) shift(const Tile<Arg>& arg,
                            const std::initializer_list<Index>& range_shift) {
  return detail::make_tile(shift(arg.tensor(), range_shift));
}

/// Shift the range of \c arg in place

/// \tparam Arg The tensor argument type
/// \tparam Index An integral range type
/// \param arg The tile argument to be shifted
/// \param range_shift The offset to be applied to the argument range
/// \return A copy of the tile with a new range
template <typename Arg, typename Index,
          typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
inline Tile<Arg>& shift_to(Tile<Arg>& arg, const Index& range_shift) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(arg.use_count() <= 1);
#endif
  shift_to(arg.tensor(), range_shift);
  return arg;
}

/// Shift the range of \c arg in place

/// \tparam Arg The tensor argument type
/// \tparam Index An integral type
/// \param arg The tile argument to be shifted
/// \param range_shift The offset to be applied to the argument range
/// \return A copy of the tile with a new range
template <typename Arg, typename Index,
          typename = std::enable_if_t<std::is_integral_v<Index>>>
inline Tile<Arg>& shift_to(Tile<Arg>& arg,
                           const std::initializer_list<Index>& range_shift) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(arg.use_count() <= 1);
#endif
  shift_to(arg.tensor(), range_shift);
  return arg;
}

// Addition operations -------------------------------------------------------

/// Add tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \param left The left-hand argument to be added
/// \param right The right-hand argument to be added
/// \return A tile that is equal to <tt>(left + right)</tt>
template <typename Left, typename Right>
inline decltype(auto) add(const Tile<Left>& left, const Tile<Right>& right) {
  return detail::make_tile(add(left.tensor(), right.tensor()));
}

/// Add and scale tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar A scalar type
/// \param left The left-hand argument to be added
/// \param right The right-hand argument to be added
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(left + right) * factor</tt>
template <
    typename Left, typename Right, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline decltype(auto) add(const Tile<Left>& left, const Tile<Right>& right,
                          const Scalar factor) {
  return detail::make_tile(add(left.tensor(), right.tensor(), factor));
}

/// Add and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Perm A permutation type
/// \param left The left-hand argument to be added
/// \param right The right-hand argument to be added
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm * (left + right)</tt>
template <typename Left, typename Right, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline decltype(auto) add(const Tile<Left>& left, const Tile<Right>& right,
                          const Perm& perm) {
  return detail::make_tile(add(left.tensor(), right.tensor(), perm));
}

/// Add, scale, and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar A scalar type
/// \tparam Perm A permutation tile
/// \param left The left-hand argument to be added
/// \param right The right-hand argument to be added
/// \param factor The scaling factor
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (left + right) * factor</tt>
template <
    typename Left, typename Right, typename Scalar, typename Perm,
    typename std::enable_if<detail::is_numeric_v<Scalar> &&
                            detail::is_permutation_v<Perm>>::type* = nullptr>
inline decltype(auto) add(const Tile<Left>& left, const Tile<Right>& right,
                          const Scalar factor, const Perm& perm) {
  return detail::make_tile(add(left.tensor(), right.tensor(), factor, perm));
}

/// Add a constant scalar to tile argument

/// \tparam Arg The tile argument type
/// \tparam Scalar A scalar type
/// \param arg The left-hand argument to be added
/// \param value The constant scalar to be added
/// \return A tile that is equal to <tt>arg + value</tt>
template <
    typename Arg, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline decltype(auto) add(const Tile<Arg>& arg, const Scalar value) {
  return detail::make_tile(add(arg.tensor(), value));
}

/// Add a constant scalar and permute tile argument

/// \tparam Arg The tile argument type
/// \tparam Scalar A scalar type
/// \tparam Perm A permutation tile
/// \param arg The left-hand argument to be added
/// \param value The constant scalar value to be added
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (arg + value)</tt>
template <
    typename Arg, typename Scalar, typename Perm,
    typename std::enable_if<detail::is_numeric_v<Scalar> &&
                            detail::is_permutation_v<Perm>>::type* = nullptr>
inline decltype(auto) add(const Tile<Arg>& arg, const Scalar value,
                          const Perm& perm) {
  return detail::make_tile(add(arg.tensor(), value, perm));
}

/// Add to the result tile

/// \tparam Result The result tile type
/// \tparam Arg The argument tile type
/// \param result The result tile
/// \param arg The argument to be added to the result
/// \return A tile that is equal to <tt>result[i] += arg[i]</tt>
template <typename Result, typename Arg>
inline Tile<Result>& add_to(Tile<Result>& result, const Tile<Arg>& arg) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  add_to(result.tensor(), arg.tensor());
  return result;
}

/// Add and scale to the result tile

/// \tparam Result The result tile type
/// \tparam Arg The argument tile type
/// \tparam Scalar A scalar type
/// \param result The result tile
/// \param arg The argument to be added to \c result
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(result[i] += arg[i]) * factor</tt>
template <
    typename Result, typename Arg, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline Tile<Result>& add_to(Tile<Result>& result, const Tile<Arg>& arg,
                            const Scalar factor) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  add_to(result.tensor(), arg.tensor(), factor);
  return result;
}

/// Add constant scalar to the result tile

/// \tparam Result The result tile type
/// \tparam Scalar A scalar type
/// \param result The result tile
/// \param value The constant scalar to be added to \c result
/// \return A tile that is equal to <tt>(result[i] += arg[i]) *= factor</tt>
template <
    typename Result, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline Tile<Result>& add_to(Tile<Result>& result, const Scalar value) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  add_to(result.tensor(), value);
  return result;
}

// Subtraction ---------------------------------------------------------------

/// Subtract tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \param left The left-hand argument to be subtracted
/// \param right The right-hand argument to be subtracted
/// \return A tile that is equal to <tt>(left - right)</tt>
template <typename Left, typename Right>
inline decltype(auto) subt(const Tile<Left>& left, const Tile<Right>& right) {
  return detail::make_tile(subt(left.tensor(), right.tensor()));
}

/// Subtract and scale tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \param left The left-hand argument to be subtracted
/// \param right The right-hand argument to be subtracted
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(left - right) * factor</tt>
template <
    typename Left, typename Right, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline decltype(auto) subt(const Tile<Left>& left, const Tile<Right>& right,
                           const Scalar factor) {
  return detail::make_tile(subt(left.tensor(), right.tensor(), factor));
}

/// Subtract and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Perm A permutation tile
/// \param left The left-hand argument to be subtracted
/// \param right The right-hand argument to be subtracted
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (left - right)</tt>
template <typename Left, typename Right, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline decltype(auto) subt(const Tile<Left>& left, const Tile<Right>& right,
                           const Perm& perm) {
  return detail::make_tile(subt(left.tensor(), right.tensor(), perm));
}

/// Subtract, scale, and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Perm A permutation tile
/// \param left The left-hand argument to be subtracted
/// \param right The right-hand argument to be subtracted
/// \param factor The scaling factor
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (left - right) * factor</tt>
template <
    typename Left, typename Right, typename Scalar, typename Perm,
    typename std::enable_if<detail::is_numeric_v<Scalar> &&
                            detail::is_permutation_v<Perm>>::type* = nullptr>
inline decltype(auto) subt(const Tile<Left>& left, const Tile<Right>& right,
                           const Scalar factor, const Perm& perm) {
  return detail::make_tile(subt(left.tensor(), right.tensor(), factor, perm));
}

/// Subtract a scalar constant from the tile argument

/// \tparam Arg The tile argument type
/// \param arg The left-hand argument to be subtracted
/// \param value The constant scalar to be subtracted
/// \return A tile that is equal to <tt>arg - value</tt>
template <
    typename Arg, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline decltype(auto) subt(const Tile<Arg>& arg, const Scalar value) {
  return detail::make_tile(subt(arg.tensor(), value));
}

/// Subtract a constant scalar and permute tile argument

/// \tparam Arg The tile argument type
/// \tparam Perm A permutation tile
/// \param arg The left-hand argument to be subtracted
/// \param value The constant scalar value to be subtracted
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (arg - value)</tt>
template <
    typename Arg, typename Scalar, typename Perm,
    typename std::enable_if<detail::is_numeric_v<Scalar> &&
                            detail::is_permutation_v<Perm>>::type* = nullptr>
inline decltype(auto) subt(const Tile<Arg>& arg, const Scalar value,
                           const Perm& perm) {
  return detail::make_tile(subt(arg.tensor(), value, perm));
}

/// Subtract from the result tile

/// \tparam Result The result tile type
/// \tparam Arg The argument tile type
/// \param result The result tile
/// \param arg The argument to be subtracted from the result
/// \return A tile that is equal to <tt>result[i] -= arg[i]</tt>
template <typename Result, typename Arg>
inline Tile<Result>& subt_to(Tile<Result>& result, const Tile<Arg>& arg) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  subt_to(result.tensor(), arg.tensor());
  return result;
}

/// Subtract and scale from the result tile

/// \tparam Result The result tile type
/// \tparam Arg The argument tile type
/// \param result The result tile
/// \param arg The argument to be subtracted from \c result
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
template <
    typename Result, typename Arg, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline Tile<Result>& subt_to(Tile<Result>& result, const Tile<Arg>& arg,
                             const Scalar factor) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  subt_to(result.tensor(), arg.tensor(), factor);
  return result;
}

/// Subtract constant scalar from the result tile

/// \tparam Result The result tile type
/// \param result The result tile
/// \param value The constant scalar to be subtracted from \c result
/// \return A tile that is equal to <tt>(result -= arg) *= factor</tt>
template <
    typename Result, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline Tile<Result>& subt_to(Tile<Result>& result, const Scalar value) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  subt_to(result.tensor(), value);
  return result;
}

// Multiplication operations -------------------------------------------------

/// Multiplication tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \param left The left-hand argument to be multiplied
/// \param right The right-hand argument to be multiplied
/// \return A tile that is equal to <tt>(left * right)</tt>
template <typename Left, typename Right>
inline decltype(auto) mult(const Tile<Left>& left, const Tile<Right>& right) {
  return detail::make_tile(mult(left.tensor(), right.tensor()));
}

/// Multiplication and scale tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \param left The left-hand argument to be multiplied
/// \param right The right-hand argument to be multiplied
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(left * right) * factor</tt>
template <
    typename Left, typename Right, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline decltype(auto) mult(const Tile<Left>& left, const Tile<Right>& right,
                           const Scalar factor) {
  return detail::make_tile(mult(left.tensor(), right.tensor(), factor));
}

/// Multiplication and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Perm A permutation tile
/// \param left The left-hand argument to be multiplied
/// \param right The right-hand argument to be multiplied
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (left * right)</tt>
template <typename Left, typename Right, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline decltype(auto) mult(const Tile<Left>& left, const Tile<Right>& right,
                           const Perm& perm) {
  return detail::make_tile(mult(left.tensor(), right.tensor(), perm));
}

/// Multiplication, scale, and permute tile arguments

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Perm A permutation tile
/// \param left The left-hand argument to be multiplied
/// \param right The right-hand argument to be multiplied
/// \param factor The scaling factor
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (left * right) * factor</tt>
template <
    typename Left, typename Right, typename Scalar, typename Perm,
    typename std::enable_if<detail::is_numeric_v<Scalar> &&
                            detail::is_permutation_v<Perm>>::type* = nullptr>
inline decltype(auto) mult(const Tile<Left>& left, const Tile<Right>& right,
                           const Scalar factor, const Perm& perm) {
  return detail::make_tile(mult(left.tensor(), right.tensor(), factor, perm));
}

/// Multiply to the result tile

/// \tparam Result The result tile type
/// \tparam Arg The argument tile type
/// \param result The result tile  to be multiplied
/// \param arg The argument to be multiplied by the result
/// \return A tile that is equal to <tt>result *= arg</tt>
template <typename Result, typename Arg>
inline Tile<Result>& mult_to(Tile<Result>& result, const Tile<Arg>& arg) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  mult_to(result.tensor(), arg.tensor());
  return result;
}

/// Multiply and scale to the result tile

/// \tparam Result The result tile type
/// \tparam Arg The argument tile type
/// \param result The result tile to be multiplied
/// \param arg The argument to be multiplied by \c result
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>(result *= arg) *= factor</tt>
template <
    typename Result, typename Arg, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline Tile<Result>& mult_to(Tile<Result>& result, const Tile<Arg>& arg,
                             const Scalar factor) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  mult_to(result.tensor(), arg.tensor(), factor);
  return result;
}

// Generic element-wise binary operations
// ---------------------------------------------

// clang-format off
/// Binary element-wise transform producing a new tile

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Op An element-wise operation type
/// \param[in] left The left-hand argument to the transform
/// \param[in] right The right-hand argument to the transform
/// \param op An element-wise operation
/// \return \c result where for each \c i in \c left.range() \c result[i]==op(left[i],right[i])
// clang-format on
template <typename Left, typename Right, typename Op>
inline decltype(auto) binary(const Tile<Left>& left, const Tile<Right>& right,
                             Op&& op) {
  return detail::make_tile(
      binary(left.tensor(), right.tensor(), std::forward<Op>(op)));
}

// clang-format off
/// Binary element-wise transform producing a new tile

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Op An element-wise operation type
/// \tparam Perm A permutation type
/// \param[in] left The left-hand argument to the transform
/// \param[in] right The right-hand argument to the transform
/// \param op An element-wise operation
/// \param perm The permutation to be applied to the result
/// \return \c perm^result where for each \c i in \c left.range() \c result[i]==op(left[i],right[i])
// clang-format on
template <typename Left, typename Right, typename Op, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline decltype(auto) binary(const Tile<Left>& left, const Tile<Right>& right,
                             Op&& op, const Perm& perm) {
  return detail::make_tile(
      binary(left.tensor(), right.tensor(), std::forward<Op>(op), perm));
}

// clang-format off
/// Binary element-wise in-place transform

/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Op An element-wise operation type
/// \param[in,out] left The left-hand argument to the transform; output contains the result of \c binary(left,right,op)
/// \param[in] right The right-hand argument to the transform
/// \param op An element-wise operation
/// \return reference to \p left
// clang-format on
template <typename Left, typename Right, typename Op>
inline Tile<Left>& inplace_binary(Tile<Left>& left, const Tile<Right>& right,
                                  Op&& op) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(left.use_count() <= 1);
#endif
  inplace_binary(left.tensor(), right.tensor(), std::forward<Op>(op));
  return left;
}

// Scaling operations --------------------------------------------------------

/// Scale the tile argument

/// \tparam Arg The tile argument type
/// \param arg The left-hand argument to be scaled
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>arg * factor</tt>
template <
    typename Arg, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline decltype(auto) scale(const Tile<Arg>& arg, const Scalar factor) {
  return detail::make_tile(scale(arg.tensor(), factor));
}

/// Scale and permute tile argument

/// \tparam Arg The tile argument type
/// \tparam Perm A permutation tile
/// \param arg The left-hand argument to be scaled
/// \param factor The scaling factor
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ (arg * factor)</tt>
template <
    typename Arg, typename Scalar, typename Perm,
    typename std::enable_if<detail::is_numeric_v<Scalar> &&
                            detail::is_permutation_v<Perm>>::type* = nullptr>
inline decltype(auto) scale(const Tile<Arg>& arg, const Scalar factor,
                            const Perm& perm) {
  return detail::make_tile(scale(arg.tensor(), factor, perm));
}

/// Scale to the result tile

/// \tparam Result The result tile type
/// \param result The result tile to be scaled
/// \param factor The scaling factor
/// \return A tile that is equal to <tt>result *= factor</tt>
template <
    typename Result, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline Tile<Result>& scale_to(Tile<Result>& result, const Scalar factor) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  scale_to(result.tensor(), factor);
  return result;
}

// Negation operations -------------------------------------------------------

/// Negate the tile argument

/// \tparam Arg The tile argument type
/// \param arg The argument to be negated
/// \return A tile that is equal to <tt>-arg</tt>
/// \note equivalent to @c scale(arg,-1)
template <typename Arg>
inline decltype(auto) neg(const Tile<Arg>& arg) {
  return detail::make_tile(neg(arg.tensor()));
}

/// Negate and permute tile argument

/// \tparam Arg The tile argument type
/// \tparam Perm A permutation tile
/// \param arg The argument to be negated
/// \param perm The permutation to be applied to the result
/// \return A tile that is equal to <tt>perm ^ -arg</tt>
/// \note equivalent to @c scale(arg,-1,perm)
template <typename Arg, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline decltype(auto) neg(const Tile<Arg>& arg, const Perm& perm) {
  return detail::make_tile(neg(arg.tensor(), perm));
}

/// In-place negate tile

/// \tparam Result The result tile type
/// \param result The result tile to be negated
/// \return negated <tt>result</tt>
/// \note equivalent to @c scale_to(arg,-1)
template <typename Result>
inline Tile<Result>& neg_to(Tile<Result>& result) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  neg_to(result.tensor());
  return result;
}

// Complex conjugate operations ---------------------------------------------

/// Create a complex conjugated copy of a tile

/// \tparam Arg The tile argument type
/// \param arg The tile to be conjugated
/// \return A complex conjugated copy of `arg`
template <typename Arg>
inline decltype(auto) conj(const Tile<Arg>& arg) {
  return detail::make_tile(conj(arg.tensor()));
}

/// Create a complex conjugated and scaled copy of a tile

/// \tparam Arg The tile argument type
/// \tparam Scalar A scalar type
/// \param arg The tile to be conjugated
/// \param factor The scaling factor
/// \return A complex conjugated and scaled copy of `arg`
template <typename Arg, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar>>::type* = nullptr>
inline decltype(auto) conj(const Tile<Arg>& arg, const Scalar factor) {
  return detail::make_tile(conj(arg.tensor(), factor));
}

/// Create a complex conjugated and permuted copy of a tile

/// \tparam Arg The tile argument type
/// \tparam Perm A permutation tile
/// \param arg The tile to be conjugated
/// \param perm The permutation to be applied to `arg`
/// \return A complex conjugated and permuted copy of `arg`
template <typename Arg, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline decltype(auto) conj(const Tile<Arg>& arg, const Perm& perm) {
  return detail::make_tile(conj(arg.tensor(), perm));
}

/// Create a complex conjugated, scaled, and permuted copy of a tile

/// \tparam Arg The tile argument type
/// \tparam Scalar A scalar type
/// \tparam Perm A permutation tile
/// \param arg The argument to be conjugated
/// \param factor The scaling factor
/// \param perm The permutation to be applied to `arg`
/// \return A complex conjugated, scaled, and permuted copy of `arg`
template <
    typename Arg, typename Scalar, typename Perm,
    typename std::enable_if<TiledArray::detail::is_numeric_v<Scalar> &&
                            detail::is_permutation_v<Perm>>::type* = nullptr>
inline decltype(auto) conj(const Tile<Arg>& arg, const Scalar factor,
                           const Perm& perm) {
  return detail::make_tile(conj(arg.tensor(), factor, perm));
}

/// In-place complex conjugate a tile

/// \tparam Result The tile type
/// \param result The tile to be conjugated
/// \return A reference to `result`
template <typename Result>
inline Tile<Result>& conj_to(Tile<Result>& result) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  conj_to(result.tensor());
  return result;
}

/// In-place complex conjugate and scale a tile

/// \tparam Result The tile type
/// \tparam Scalar A scalar type
/// \param result The tile to be conjugated
/// \param factor The scaling factor
/// \return A reference to `result`
template <typename Result, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar>>::type* = nullptr>
inline Tile<Result>& conj_to(Tile<Result>& result, const Scalar factor) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(result.use_count() <= 1);
#endif
  conj_to(result.tensor(), factor);
  return result;
}

// Generic element-wise unary operations
// ---------------------------------------------

// clang-format off
/// Unary element-wise transform producing a new tile

/// \tparam Arg The tile argument type
/// \tparam Op An element-wise operation type
/// \param[in] arg The tile to be transformed
/// \param op An element-wise operation
/// \return \c result where for each \c i in \c arg.range() \c result[i]==op(arg[i])
// clang-format on
template <typename Arg, typename Op>
inline decltype(auto) unary(const Tile<Arg>& arg, Op&& op) {
  return detail::make_tile(unary(arg.tensor(), std::forward<Op>(op)));
}

// clang-format off
/// Unary element-wise transform producing a new tile

/// \tparam Arg The tile argument type
/// \tparam Op An element-wise operation type
/// \param[in] arg The tile to be transformed
/// \param op An element-wise operation
/// \param perm The permutation to be applied to the result of the transform
/// \return \c perm^result where for each \c i in \c arg.range() \c result[i]==op(arg[i])
// clang-format on
template <typename Arg, typename Op, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline decltype(auto) unary(const Tile<Arg>& arg, Op&& op, const Perm& perm) {
  return detail::make_tile(unary(arg.tensor(), std::forward<Op>(op), perm));
}

// clang-format off
/// Unary element-wise in-place transform

/// \tparam Arg The tile argument type
/// \tparam Op An element-wise operation type
/// \param[in,out] arg The tile to be transformed, on output for each \c i in \c arg.range() \c arg[i] contains \c op(arg[i])
/// \param op An element-wise operation
/// \return \c reference to \p arg
// clang-format on
template <typename Result, typename Op>
inline Tile<Result>& inplace_unary(Tile<Result>& arg, Op&& op) {
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(arg.use_count() <= 1);
#endif
  inplace_unary(arg.tensor(), std::forward<Op>(op));
  return arg;
}

// Contraction operations ----------------------------------------------------

/// Contract 2 tensors over head/tail modes and scale the product

/// The contraction is done via a GEMM operation with fused indices as defined
/// by \c gemm_config.
/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar A numeric type
/// \param left The left-hand argument to be contracted
/// \param right The right-hand argument to be contracted
/// \param factor The scaling factor
/// \param gemm_config A helper object used to simplify gemm operations
/// \return A tile that is equal to <tt>(left * right) * factor</tt>
template <
    typename Left, typename Right, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline decltype(auto) gemm(const Tile<Left>& left, const Tile<Right>& right,
                           const Scalar factor,
                           const math::GemmHelper& gemm_config) {
  return detail::make_tile(
      gemm(left.tensor(), right.tensor(), factor, gemm_config));
}

/// Contract 2 tensors over head/tail modes, scale the product, and add
/// to \c result

/// The contraction is done via a GEMM operation with fused indices as defined
/// by \c gemm_config.
/// \tparam Result The result tile type
/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar A numeric type
/// \param result The contracted result
/// \param left The left-hand argument to be contracted
/// \param right The right-hand argument to be contracted
/// \param factor The scaling factor
/// \param gemm_config A helper object used to simplify gemm operations
/// \return A tile that is equal to <tt>result + (left * right) * factor</tt>
template <
    typename Result, typename Left, typename Right, typename Scalar,
    typename std::enable_if<detail::is_numeric_v<Scalar>>::type* = nullptr>
inline Tile<Result>& gemm(Tile<Result>& result, const Tile<Left>& left,
                          const Tile<Right>& right, const Scalar factor,
                          const math::GemmHelper& gemm_config) {
  gemm(result.tensor(), left.tensor(), right.tensor(), factor, gemm_config);
  return result;
}

/// Contract 2 tensors over head/tail modes and accumulate into \c result
/// using a custom element-wise multiply-add op

/// The contraction is done via a GEMM operation with fused indices as defined
/// by \c gemm_config.
/// \tparam Result The result tile type
/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam ElementMultiplyAddOp a callable type with signature
///   \code
///     void (Result::value_type& result, Left::value_type const& left,
///     Right::value_type const& right)
///   \endcode
///   that implements custom multiply-add operation:
///   \code
///     result = (result) ? result add left mult right : left mult add
///   \endcode
/// \param result The contracted result
/// \param left The left-hand argument to be contracted
/// \param right The right-hand argument to be contracted
/// \param factor The scaling factor
/// \param gemm_config A helper object used to simplify gemm operations
/// \param element_multiplyadd_op a custom multiply op operation for tensor
/// elements \return A tile whose element <tt>result[i,j]</tt> obtained by
/// executing
///      `foreach k: element_multiplyadd_op(result[i,j], left[i,k], right[k,j])`
template <typename Result, typename Left, typename Right,
          typename ElementMultiplyAddOp,
          typename std::enable_if<std::is_invocable_r_v<
              void, std::remove_reference_t<ElementMultiplyAddOp>,
              typename Result::value_type&, const typename Left::value_type&,
              const typename Right::value_type&>>::type* = nullptr>
inline Tile<Result>& gemm(Tile<Result>& result, const Tile<Left>& left,
                          const Tile<Right>& right,
                          const math::GemmHelper& gemm_config,
                          ElementMultiplyAddOp&& element_multiplyadd_op) {
  gemm(result.tensor(), left.tensor(), right.tensor(), gemm_config,
       std::forward<ElementMultiplyAddOp>(element_multiplyadd_op));
  return result;
}

// Reduction operations ------------------------------------------------------

/// Sum the hyper-diagonal elements a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to be summed
/// \return The sum of the hyper-diagonal elements of \c arg
template <typename Arg>
inline decltype(auto) trace(const Tile<Arg>& arg) {
  return trace(arg.tensor());
}

namespace detail {

/// Signals that we can take the trace of a \c Tile<Arg> if can trace \c Arg
template <typename Arg>
struct TraceIsDefined<Tile<Arg>, enable_if_trace_is_defined_t<Arg>>
    : std::true_type {};

}  // namespace detail

/// Sum the elements of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to be summed
/// \return A scalar that is equal to <tt>sum_i arg[i]</tt>
template <typename Arg>
inline decltype(auto) sum(const Tile<Arg>& arg) {
  return sum(arg.tensor());
}

/// Multiply the elements of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to be multiplied
/// \return A scalar that is equal to <tt>prod_i arg[i]</tt>
template <typename Arg>
inline decltype(auto) product(const Tile<Arg>& arg) {
  return product(arg.tensor());
}

/// Squared vector 2-norm of the elements of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to be multiplied and summed
/// \return The sum of the squared elements of \c arg
/// \return A scalar that is equal to <tt>sum_i arg[i] * arg[i]</tt>
template <typename Arg>
inline decltype(auto) squared_norm(const Tile<Arg>& arg) {
  return squared_norm(arg.tensor());
}

/// Vector 2-norm of a tile

/// \tparam Arg The tile argument type
/// \param[in] arg The argument to be multiplied and summed
/// \return A scalar that is equal to <tt>sqrt(sum_i arg[i] * arg[i])</tt>
template <typename Arg>
inline decltype(auto) norm(const Tile<Arg>& arg) {
  return norm(arg.tensor());
}

/// Vector 2-norm of a tile

/// \tparam Arg The tile argument type
/// \tparam ResultType The result type
/// \param[in,out] arg The argument to be multiplied and summed; on output will
/// contain the vector 2-norm of \c arg , i.e. <tt>sqrt(sum_i arg[i] *
/// arg[i])</tt>
template <typename Arg, typename ResultType>
inline void norm(const Tile<Arg>& arg, ResultType& result) {
  norm(arg.tensor(), result);
}

/// Maximum element of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to find the maximum
/// \return A scalar that is equal to <tt>max(arg)</tt>
template <typename Arg>
inline decltype(auto) max(const Tile<Arg>& arg) {
  return max(arg.tensor());
}

/// Minimum element of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to find the minimum
/// \return A scalar that is equal to <tt>min(arg)</tt>
template <typename Arg>
inline decltype(auto) min(const Tile<Arg>& arg) {
  return min(arg.tensor());
}

/// Absolute maximum element of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to find the maximum
/// \return A scalar that is equal to <tt>abs(max(arg))</tt>
template <typename Arg>
inline decltype(auto) abs_max(const Tile<Arg>& arg) {
  return abs_max(arg.tensor());
}

/// Absolute mainimum element of a tile

/// \tparam Arg The tile argument type
/// \param arg The argument to find the minimum
/// \return A scalar that is equal to <tt>abs(min(arg))</tt>
template <typename Arg>
inline decltype(auto) abs_min(const Tile<Arg>& arg) {
  return abs_min(arg.tensor());
}

/// Vector dot product of a tile

/// \tparam Left The left-hand argument type
/// \tparam Right The right-hand argument type
/// \param left The left-hand argument tile to be contracted
/// \param right The right-hand argument tile to be contracted
/// \return A scalar that is equal to <tt>sum_i left[i] * right[i]</tt>
template <typename Left, typename Right>
inline decltype(auto) dot(const Tile<Left>& left, const Tile<Right>& right) {
  return dot(left.tensor(), right.tensor());
}

/// Vector inner product of a tile

/// \tparam Left The left-hand argument type
/// \tparam Right The right-hand argument type
/// \param left The left-hand argument tile to be contracted
/// \param right The right-hand argument tile to be contracted
template <typename Left, typename Right>
inline decltype(auto) inner_product(const Tile<Left>& left,
                                    const Tile<Right>& right) {
  return inner_product(left.tensor(), right.tensor());
}

// Tile arithmetic operators -------------------------------------------------

// see operators.h

/// Tile output stream operator
/// -------------------------------------------------

/// \tparam T The tensor type
/// \param os The output stream
/// \param tile The tile to be printed
/// \return The modified output stream
template <typename Char, typename CharTraits, typename T>
inline std::basic_ostream<Char, CharTraits>& operator<<(
    std::basic_ostream<Char, CharTraits>& os, const Tile<T>& tile) {
  os << tile.tensor();
  return os;
}

/// implement conversions from Tile<T> to
/// TiledArray::Tensor<T::value_type,Allocator>
template <typename Allocator, typename T>
struct Cast<
    TiledArray::Tensor<typename T::value_type, Allocator>, Tile<T>,
    std::void_t<
        decltype(std::declval<TiledArray::Cast<
                     TiledArray::Tensor<typename T::value_type, Allocator>,
                     T>>()(std::declval<const T&>()))>> {
  auto operator()(const Tile<T>& arg) const {
    return TiledArray::Cast<
        TiledArray::Tensor<typename T::value_type, Allocator>, T>{}(
        arg.tensor());
  }
};

/** @}*/

/// Tile equality comparison
template <typename T1, typename T2>
bool operator==(const Tile<T1>& t1, const Tile<T2>& t2) {
  return t1.tensor() == t2.tensor();
}

/// Tile inequality comparison
template <typename T1, typename T2>
bool operator!=(const Tile<T1>& t1, const Tile<T2>& t2) {
  return !(t1 == t2);
}

namespace detail {

template <typename T>
struct real_t_impl<Tile<T>> {
  using type = typename Tile<T>::template rebind_numeric_t<
      typename Tile<T>::scalar_type>;
};

template <typename T>
struct complex_t_impl<Tile<T>> {
  using type = typename Tile<T>::template rebind_numeric_t<
      std::complex<typename Tile<T>::scalar_type>>;
};

}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_H__INCLUDED
