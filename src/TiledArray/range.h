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

#ifndef TILEDARRAY_RANGE_H__INCLUDED
#define TILEDARRAY_RANGE_H__INCLUDED

#include <TiledArray/permutation.h>
#include <TiledArray/range1.h>
#include <TiledArray/range_iterator.h>
#include <TiledArray/size_array.h>
#include <TiledArray/util/vector.h>

namespace TiledArray {

/// \brief A (hyperrectangular) interval on \f$ Z^n \f$, space of integer
/// \f$ n \f$-indices

/// This object is a range of integers on a \f$ n \f$-dimensional,
/// hyperrectangular domain, where _rank_ (aka order, number of dimensions)
/// \f$ n>0 \f$. It is fully specified by its lower and upper bounds. It also
/// provides zero-cost access to
///  _extent_ (size) and _stride_ for
/// each _mode_ (dimension). Range is a _regular_ type with null default state.
/// \warning Range is with rank 0 is _null_, i.e. invalid. There are many
/// reasons that rank-0 Range is not supported; summarily, rank-0 Range is not a
/// special case of rank-\f$ n \f$ Range as many invariants of nonzero-rank
/// Range are not observed by it. E.g. for any nonempty nonzero-rank Range the
/// lower bound differs from its upper bound. To define the 0-dimensional limit
/// of array/tensor to be a scalar, the volume of rank-0 Range should be 1, but
/// clearly its lower and upper bounds are equal.
class Range {
 public:
  typedef Range Range_;                ///< This object type
  typedef TA_1INDEX_TYPE index1_type;  ///< 1-index type, to conform to
                                       ///< Tensor Working Group (TWG) spec
  typedef std::make_signed_t<TA_1INDEX_TYPE>
      index1_difference_type;  ///< type representing difference of 1-indices
  typedef container::svector<index1_type>
      index_type;  ///< Coordinate index type, to conform to
                   ///< TWG spec
  typedef container::svector<index1_difference_type> index_difference_type;
  typedef index_type index;  ///< Coordinate index type (deprecated)
  typedef detail::SizeArray<const index1_type>
      index_view_type;  ///< Non-owning variant of index_type
  typedef index_type
      extent_type;  ///< Range extent type, to conform to TWG spec
  typedef std::size_t ordinal_type;  ///< Ordinal type, to conform to TWG spec
  typedef std::make_signed_t<ordinal_type> distance_type;  ///< Distance type
  typedef ordinal_type size_type;  ///< Size type (deprecated)
  typedef detail::RangeIterator<index1_type, Range_>
      const_iterator;  ///< Coordinate iterator
  friend class detail::RangeIterator<index1_type, Range_>;

  static_assert(detail::is_range_v<index_type>);  // index is a Range

 protected:
  /// A vector that holds the dimension information of the
  /// range. Its layout:
  /// \code
  /// { lobound[0], ..., lobound[rank_ - 1],
  ///   upbound[0], ..., upbound[rank_ - 1],
  ///   extent[0],  ..., extent[rank_ - 1],
  ///   stride[0],  ..., stride[rank_ - 1] }
  /// \endcode
  container::svector<index1_type, 4 * TA_MAX_SOO_RANK_METADATA> datavec_;
  distance_type offset_ =
      0l;  ///< Ordinal index offset correction to support nonzero lobound
  ordinal_type volume_ = 0ul;  ///< Total number of elements
  unsigned int rank_ = 0u;  ///< The rank (or number of dimensions) in the range

  void init_datavec(unsigned int rank) { datavec_.resize(rank << 2); }
  const index1_type* data() const { return datavec_.data(); }
  index1_type* data() { return datavec_.data(); }

  index1_type* lobound_data_nc() { return data(); }
  index1_type* upbound_data_nc() { return data() + rank_; }
  index1_type* extent_data_nc() { return data() + (rank_ << 1); }
  index1_type* stride_data_nc() { return extent_data_nc() + rank_; }

 private:
  /// Initialize range data from sequences of lower and upper bounds

  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param lower_bound The lower bound of the range
  /// \param upper_bound The upper bound of the range
  /// \pre Assume \c rank_ is initialized to the rank of the range and
  /// \c datavec_ has been allocated to hold 4*rank_ elements
  /// \post \c datavec_ and \c volume_ are initialized with range dimension
  /// information.
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  void init_range_data(const Index1& lower_bound, const Index2& upper_bound) {
    // Construct temp pointers
    auto* MADNESS_RESTRICT const lower = data();
    auto* MADNESS_RESTRICT const upper = lower + rank_;
    auto* MADNESS_RESTRICT const extent = upper + rank_;
    auto* MADNESS_RESTRICT const stride = extent + rank_;

    // Set the volume seed
    volume_ = 1ul;
    offset_ = 0l;

    // initialize bounds and extents
    auto lower_it = std::begin(lower_bound);
    auto upper_it = std::begin(upper_bound);
    auto lower_end = std::end(lower_bound);
    auto upper_end = std::end(upper_bound);
    for (int d = 0; lower_it != lower_end && upper_it != upper_end;
         ++lower_it, ++upper_it, ++d) {
      // Compute data for element d of lower, upper, and extent
      const auto lower_bound_d = *lower_it;
      const auto upper_bound_d = *upper_it;
      lower[d] = lower_bound_d;
      upper[d] = upper_bound_d;
      // Check input dimensions
      TA_ASSERT(lower[d] <= upper[d]);
      extent[d] = upper[d] - lower[d];
      TA_ASSERT(extent[d] ==
                static_cast<index1_type>(upper_bound_d - lower_bound_d));
    }

    // Set the volume seed
    volume_ = 1ul;
    offset_ = 0l;

    // Compute strides, volume, and offset, starting with last (least
    // significant) dimension
    for (int d = int(rank_) - 1; d >= 0; --d) {
      stride[d] = volume_;
      offset_ += lower[d] * stride[d];
      volume_ *= extent[d];
    }
  }

  // clang-format off
  /// Initialize range data from a sequence of {lower,upper} bound pairs

  /// \tparam PairRange Type representing a range of generalized pairs (see TiledArray::detail::is_gpair_v )
  /// \param bounds The {lower,upper} bound of the range for each dimension
  /// \pre Assume \c rank_ is initialized to the rank of the range and
  /// \c datavec_ has been allocated to hold 4*rank_ elements
  /// \post \c datavec_ and \c volume_ are initialized with range dimension
  /// information.
  // clang-format on
  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  void init_range_data(const PairRange& bounds) {
    // Construct temp pointers
    auto* MADNESS_RESTRICT const lower = data();
    auto* MADNESS_RESTRICT const upper = lower + rank_;
    auto* MADNESS_RESTRICT const extent = upper + rank_;
    auto* MADNESS_RESTRICT const stride = extent + rank_;

    // Compute range data
    int d = 0;
    for (auto&& bound_d : bounds) {
      // Compute data for element i of lower, upper, and extent
      const auto lower_bound_d = detail::at(bound_d, 0);
      const auto upper_bound_d = detail::at(bound_d, 1);
      lower[d] = lower_bound_d;
      upper[d] = upper_bound_d;
      // Check input dimensions
      TA_ASSERT(lower[d] <= upper[d]);
      extent[d] = upper[d] - lower[d];
      ++d;
    }
    // Compute strides, volume, and offset, starting with last (least
    // significant) dimension
    volume_ = 1ul;
    offset_ = 0l;
    for (int d = int(rank_) - 1; d >= 0; --d) {
      stride[d] = volume_;
      offset_ += lower[d] * stride[d];
      volume_ *= extent[d];
    }
  }

  /// Initialize range data from a sequence of extents

  /// \tparam Index An integral range type
  /// \param extents A sequence of extents for each dimension
  /// \pre Assume \c rank_ is initialized to the rank of the range and
  /// \c datavec_ has been allocated to hold 4*rank_ elements
  /// \post \c datavec_ and \c volume_ are initialized with range dimension
  /// information.
  template <typename Index, typename std::enable_if_t<
                                detail::is_integral_range_v<Index>>* = nullptr>
  void init_range_data(const Index& extents) {
    // Construct temp pointers
    auto* MADNESS_RESTRICT const lower = data();
    auto* MADNESS_RESTRICT const upper = lower + rank_;
    auto* MADNESS_RESTRICT const extent = upper + rank_;
    auto* MADNESS_RESTRICT const stride = extent + rank_;

    // Initialize extents and bounds
    auto it = std::begin(extents);
    auto end = std::end(extents);
    for (int d = 0; it != end; ++it, ++d) {
      const auto extent_d = *it;
      lower[d] = 0ul;
      upper[d] = extent_d;
      extent[d] = extent_d;
      // Check bounds of the input extent
      TA_ASSERT(extent[d] >= 0);
    }

    // Compute strides and volume, starting with last (least significant)
    // dimension
    volume_ = 1ul;
    offset_ = 0ul;
    for (int d = int(rank_) - 1; d >= 0; --d) {
      stride[d] = volume_;
      volume_ *= extent[d];
    }
  }

  /// Initialize range data from a tuple of extents

  /// \tparam Indices A pack of integral types
  /// \param extents A tuple of extents for each dimension
  /// \pre Assume \c rank_ is initialized to the rank of the range and
  /// \c datavec_ has been allocated to hold 4*rank_ elements
  /// \post \c datavec_ and \c volume_ are initialized with range dimension
  /// information.
  template <typename... Indices,
            typename std::enable_if<
                detail::is_integral_list<Indices...>::value>::type* = nullptr>
  void init_range_data(const std::tuple<Indices...>& extents) {
    const constexpr std::size_t rank =
        std::tuple_size<std::tuple<Indices...>>::value;
    TA_ASSERT(rank_ == rank);

    // Set the offset and volume initial values
    volume_ = 1ul;
    offset_ = 0ul;

    // initialize by recursion
    init_range_data_helper(extents, std::make_index_sequence<rank>{});
  }

  template <typename... Indices, std::size_t... Is>
  void init_range_data_helper(const std::tuple<Indices...>& extents,
                              std::index_sequence<Is...>) {
    int workers[] = {0, (init_range_data_helper_iter<Is>(extents), 0)...};
    ++workers[0];
  }

  template <std::size_t I, typename... Indices>
  void init_range_data_helper_iter(const std::tuple<Indices...>& extents) {
    // Get extent i
    const auto extent_i = std::get<I>(extents);

    auto* MADNESS_RESTRICT const lower = data();
    auto* MADNESS_RESTRICT const upper = lower + rank_;
    auto* MADNESS_RESTRICT const extent = upper + rank_;
    auto* MADNESS_RESTRICT const stride = extent + rank_;

    lower[I] = 0ul;
    upper[I] = extent_i;
    extent[I] = extent_i;
    // Check bounds of the input extent
    TA_ASSERT(extent[I] >= 0ul);
    stride[I] = volume_;
    volume_ *= extent[I];
  }

  /// Initialize permuted range data from lower and upper bounds

  /// \param other_lower_bound The lower bound of the unpermuted range
  /// \param other_upper_bound The upper bound of the unpermuted range
  /// \pre Assume \c rank_ is initialized to the rank of the range and
  /// \c datavec_ has been allocated to hold 4*rank_ elements
  /// \post \c datavec_, \c offset_, and \c volume_ are initialized with the
  /// permuted range dimension information from \c other_lower_bound and
  /// \c other_upper_bound.
  void init_range_data(
      const Permutation& perm,
      const index1_type* MADNESS_RESTRICT const other_lower_bound,
      const index1_type* MADNESS_RESTRICT const other_upper_bound) {
    // Create temporary pointers to this range data
    auto* MADNESS_RESTRICT const lower = data();
    auto* MADNESS_RESTRICT const upper = lower + rank_;
    auto* MADNESS_RESTRICT const extent = upper + rank_;
    auto* MADNESS_RESTRICT const stride = extent + rank_;

    // Copy the permuted lower, upper, and extent into this range.
    for (unsigned int i = 0u; i < rank_; ++i) {
      const auto perm_i = perm[i];

      // Get the lower bound, upper bound, and extent from other for rank i.
      const auto other_lower_bound_i = other_lower_bound[i];
      const auto other_upper_bound_i = other_upper_bound[i];
      const auto other_extent_i = other_upper_bound_i - other_lower_bound_i;

      // Store the permuted lower bound, upper bound, and extent
      lower[perm_i] = other_lower_bound_i;
      upper[perm_i] = other_upper_bound_i;
      extent[perm_i] = other_extent_i;
    }

    // Recompute stride, offset, and volume
    volume_ = 1ul;
    offset_ = 0ul;
    for (int i = int(rank_) - 1; i >= 0; --i) {
      const auto lower_i = lower[i];
      const auto extent_i = extent[i];
      stride[i] = volume_;
      offset_ += lower_i * volume_;
      volume_ *= extent_i;
    }
  }

 public:
  /// Default constructor

  /// Constructs a null range, i.e., it has zero volume and rank.
  Range() {}

  /// Construct range defined by upper and lower bound ranges

  /// Construct a range defined by \c lower_bound and \c upper_bound.
  /// Examples of using this constructor:
  /// \code
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///   Range r(lobounds, upbounds);
  ///   // or using in-place ctors
  ///   Range r2(std::vector<size_t>{0, 1, 2}, std::vector<size_t>{4, 6, 8});
  ///   assert(r == r2);
  /// \endcode
  /// \tparam Index1 An integral sized range type
  /// \tparam Index2 An integral sized range type
  /// \param lower_bound A sequence of lower bounds for each dimension
  /// \param upper_bound A sequence of upper bounds for each dimension
  /// \throw TiledArray::Exception When the size of \c lower_bound is not
  /// equal to that of \c upper_bound.
  /// \throw TiledArray::Exception When lower_bound[i] >= upper_bound[i]
  template <typename Index1, typename Index2,
            typename std::enable_if_t<
                detail::is_integral_sized_range_v<Index1> &&
                detail::is_integral_sized_range_v<Index2>>* = nullptr>
  Range(const Index1& lower_bound, const Index2& upper_bound) {
    using std::size;
    const auto n = size(lower_bound);
    TA_ASSERT(n == size(upper_bound));
    if (n) {
      // Initialize array memory
      init_datavec(n);
      rank_ = n;
      init_range_data(lower_bound, upper_bound);
    }  // rank-0 Range is null
  }

  // clang-format off
  /// Construct range defined by the upper and lower bound ranges

  /// Construct a range defined by \c lower_bound and \c upper_bound.
  /// Examples of using this constructor:
  /// \code
  ///   Range r({0, 1, 2}, {4, 6, 8});
  ///   // WARNING: mind the parens! With braces another ctor is called
  ///   Range r2{{0, 1, 2}, {4, 6, 8}};
  /// \endcode
  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \param lower_bound An initializer list of lower bounds for each dimension
  /// \param upper_bound An initializer list of upper bounds for each dimension
  /// \warning do not use uniform initialization syntax ("curly braces") to invoke this
  /// \throw TiledArray::Exception When the size of \c lower_bound is not
  /// equal to that of \c upper_bound.
  /// \throw TiledArray::Exception When lower_bound[i] >= upper_bound[i]
  // clang-format on
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  Range(const std::initializer_list<Index1>& lower_bound,
        const std::initializer_list<Index2>& upper_bound) {
    using std::size;
    const auto n = size(lower_bound);
    TA_ASSERT(n == size(upper_bound));
    init_datavec(n);
    rank_ = n;
    if (n) {
      init_range_data(lower_bound, upper_bound);
    }  // rank-0 Range is null
  }

  /// Range constructor from a range of extents

  /// Construct a range with a lower bound of zero and an upper bound equal to
  /// \c extents.
  /// Examples of using this constructor:
  /// \code
  ///   Range r(std::vector<size_t>{4, 5, 6});
  /// \endcode
  /// \tparam Index A vector type
  /// \param extent A vector that defines the size of each dimension
  template <typename Index,
            typename std::enable_if_t<
                detail::is_integral_sized_range_v<Index>>* = nullptr>
  explicit Range(const Index& extent) {
    using std::size;
    const auto n = size(extent);
    if (n) {
      // Initialize array memory
      init_datavec(n);
      rank_ = n;
      init_range_data(extent);
    }  // rank-0 Range is null
  }

  /// Range constructor from an initializer list of extents

  /// Construct a range with a lower bound of zero and an upper bound equal to
  /// \c extent.
  /// Examples of using this constructor:
  /// \code
  ///   Range r{4, 5, 6};
  /// \endcode
  /// \tparam Index An integral type
  /// \param extent An initializer list that defines the size of each dimension
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  explicit Range(const std::initializer_list<Index>& extent) {
    using std::size;
    const auto n = size(extent);
    if (n) {
      // Initialize array memory
      init_datavec(n);
      rank_ = n;
      init_range_data(extent);
    }  // rank-0 Range is null
  }

  // clang-format off
  /// Construct Range defined by a range of {lower,upper} bound pairs

  /// Examples of using this constructor:
  /// \code
  ///   // using vector of pairs
  ///   std::vector<std::pair<size_t,size_t>> vpbounds{{0,4}, {1,6}, {2,8}};
  ///   Range r0(vpbounds);
  ///   // using vector of tuples
  ///   std::vector<std::tuple<size_t,size_t>> vtbounds{{0,4}, {1,6}, {2,8}};
  ///   Range r1(vtbounds);
  ///   assert(r0 == r1);
  ///
  ///   // using zipped ranges of bounds (using Boost.Range)
  ///   // need to #include <boost/range/combine.hpp>
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///   Range r2(boost::combine(lobounds, upbounds));
  ///   assert(r0 == r2);
  ///
  ///   // using zipped ranges of bounds (using Range-V3)
  ///   // need to #include <range/v3/view/zip.hpp>
  ///   Range r3(ranges::views::zip(lobounds, upbounds));
  ///   assert(r0 == r3);
  /// \endcode
  /// \tparam PairRange Type representing a sized range of generalized pairs (see TiledArray::detail::is_gpair_v )
  /// \param bound A range of {lower,upper} bounds for each dimension
  /// \throw TiledArray::Exception When `bound[i].lower>=bound[i].upper` for any \c i .
  // clang-format on
  template <typename PairRange,
            typename = std::enable_if_t<detail::is_sized_range_v<PairRange> &&
                                        detail::is_gpair_range_v<PairRange>>>
  explicit Range(const PairRange& bounds) {
    const auto n = std::size(bounds);
    if (n) {
      // Initialize array memory
      init_datavec(n);
      rank_ = n;
      init_range_data(bounds);
    }  // rank-0 Range is null
  }

  // clang-format off
  /// Construct range defined by an initializer_list of {lower,upper} bounds for each dimension given as a generalized pair

  /// Examples of using this constructor:
  /// \code
  ///   Range r{std::pair{0,4}, std::pair{1,6}, std::pair{2,8}};
  /// \endcode
  /// \tparam GPair a generalized pair of integral types
  /// \param bound A sequence of {lower,upper} bounds for each dimension
  /// \throw TiledArray::Exception When \c bound[i].lower>=bound[i].upper for any \c i .
  // clang-format on
  template <typename GPair>
  explicit Range(const std::initializer_list<GPair>& bounds,
                 std::enable_if_t<detail::is_gpair_v<GPair>>* = nullptr) {
    using std::size;
#ifndef NDEBUG
    if constexpr (detail::is_contiguous_range_v<GPair>) {
      for (auto&& bound_d : bounds) {
        TA_ASSERT(size(bound_d) == 2);
      }
    }
#endif
    const auto n = size(bounds);
    if (n) {
      // Initialize array memory
      init_datavec(n);
      rank_ = n;
      init_range_data(bounds);
    }  // rank-0 Range is null
  }

  // clang-format off
  /// Construct range defined by an initializer_list of std::initializer_list{lower,upper} bounds

  /// Examples of using this constructor:
  /// \code
  ///   Range r{{0,4}, {1,6}, {2,8}};
  ///   // or can add extra parens
  ///   Range r2({{0,4}, {1,6}, {2,8}});
  ///   assert(r == r2);
  /// \endcode
  /// \tparam Index An integral type
  /// \param bound A sequence of {lower,upper} bounds for each dimension
  /// \throw TiledArray::Exception When `bound[i].lower>=bound[i].upper` for any \c i .
  // clang-format on
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  explicit Range(
      const std::initializer_list<std::initializer_list<Index>>& bounds) {
    using std::size;
    const auto n = size(bounds);
    if (n) {
#ifndef NDEBUG
      for (auto&& bound_d : bounds) {
        TA_ASSERT(size(bound_d) == 2);
      }
#endif
      // Initialize array memory
      init_datavec(n);
      rank_ = n;
      init_range_data(bounds);
    }  // rank-0 Range is null
  }

  /// Range constructor from a pack of extents for each dimension

  /// Examples of using this constructor:
  /// \code
  ///   Range r(4, 5, 6);
  /// \endcode
  /// \tparam Index Pack of integer types
  /// \param extents A pack of extents for each dimension
  /// \post Range has a lower bound of 0, and an upper bound of \c (extents...).
  template <typename... Index, typename std::enable_if<detail::is_integral_list<
                                   Index...>::value>::type* = nullptr>
  explicit Range(const Index... extents)
      : Range(std::array<size_t, sizeof...(Index)>{
            {static_cast<std::size_t>(extents)...}}) {}

  /// Range constructor from a pack of std::pair{lo,up} bounds for each
  /// dimension

  /// Examples of using this constructor:
  /// \code
  ///   Range r(std::pair{0,4}, std::pair{1,6}, std::pair{2,8});
  /// \endcode
  /// \tparam IndexPairs Pack of std::pair's of integer types
  /// \param extents A pack of pairs of lobound and upbound for each dimension
  template <typename... IndexPairs,
            std::enable_if_t<detail::is_integral_pair_list_v<IndexPairs...>>* =
                nullptr>
  explicit Range(const IndexPairs... bounds)
      : Range(std::array<std::pair<std::size_t, std::size_t>,
                         sizeof...(IndexPairs)>{
            {static_cast<std::pair<std::size_t, std::size_t>>(bounds)...}}) {}

  /// Copy Constructor

  /// \param other The range to be copied
  Range(const Range_& other) = default;

  /// Move Constructor

  /// \param[in,out] other The range to be copied; set to default (null) state
  ///                on return
  /// \post `other == Range{}`
  Range(Range_&& other)
      : datavec_(std::move(other.datavec_)),
        offset_(other.offset_),
        volume_(other.volume_),
        rank_(other.rank_) {
    // put other into null state
    other.datavec_.clear();
    other.datavec_.shrink_to_fit();
    other.offset_ = 0ul;
    other.volume_ = 0ul;
    other.rank_ = 0u;
  }

  /// Permuting copy constructor

  /// \param perm The permutation applied to other; if `!perm` then no
  /// permutation is applied \param other The range to be permuted and copied
  Range(const Permutation& perm, const Range_& other) {
    TA_ASSERT(perm.size() == other.rank_ || !perm);

    if (other.rank_ > 0ul) {
      rank_ = other.rank_;

      if (perm) {
        init_datavec(other.rank_);
        init_range_data(perm, other.lobound_data(), other.upbound_data());
      } else {
        // Simple copy will do
        datavec_ = other.datavec_;
        offset_ = other.offset_;
        volume_ = other.volume_;
      }
    } else  // handle null and rank-0 case
      volume_ = other.volume_;
  }

  /// Destructor
  ~Range() = default;

  /// Copy assignment operator

  /// \param other The range to be copied
  /// \return A reference to this object
  Range_& operator=(const Range_& other) = default;

  /// Move assignment operator

  /// \param[in,out] other The range to be copied; set to default (null) state
  ///                on return
  /// \return A reference to this object
  /// \throw nothing
  Range_& operator=(Range_&& other) {
    datavec_ = std::move(other.datavec_);
    offset_ = other.offset_;
    volume_ = other.volume_;
    rank_ = other.rank_;

    // put other into null state
    other.datavec_.clear();
    other.datavec_.shrink_to_fit();
    other.offset_ = 0l;
    other.volume_ = 0ul;
    other.rank_ = 0u;

    return *this;
  }

  /// Conversion to bool

  /// \return false if is null state, i.e. \c this->rank()==0
  explicit operator bool() const { return rank() != 0; }

  /// Rank accessor

  /// \return The rank (number of dimensions) of this range
  /// \throw nothing
  unsigned int rank() const { return rank_; }

  /// Accessor of the \c d-th dimension of the range

  /// \param d the dimension index, a nonnegative integer less than rank()
  /// \return the pair of {lower,upper} bounds for dimension \c d
  Range1 dim(std::size_t d) const {
    TA_ASSERT(d < rank());
    return Range1(lobound(d), upbound(d));
  }

  /// Range lower bound data accessor

  /// \return A pointer to the lower bound data (see Range::lobound() )
  /// \note Not necessarily nullptr for rank-0 or null Range
  /// \throw nothing
  const index1_type* lobound_data() const { return data(); }

  /// Range lower bound accessor

  /// \return A \c index view that contains the lower bounds for each
  /// dimension of the block range.
  /// \throw nothing
  index_view_type lobound() const {
    return index_view_type(lobound_data(), rank_);
  }

  /// Range lower bound element accessor

  /// \return The lower bound of dimension \c dim.
  /// \throw nothing
  index1_type lobound(size_t dim) const {
    TA_ASSERT(dim < rank_);
    return *(lobound_data() + dim);
  }

  /// Range upper bound data accessor

  /// \return A pointer to the upper bound data (see Range::upbound() )
  /// \note Not necessarily nullptr for rank-0 or null Range
  /// \throw nothing
  const index1_type* upbound_data() const { return data() + rank_; }

  /// Range upper bound accessor

  /// \return An index view that contains the upper bounds for each
  /// dimension of the block range.
  /// \throw nothing
  index_view_type upbound() const {
    return index_view_type(upbound_data(), rank_);
  }

  /// Range upped bound element accessor

  /// \return The upper bound of dimension \c dim.
  /// \throw nothing
  index1_type upbound(size_t dim) const {
    TA_ASSERT(dim < rank_);
    return *(upbound_data() + dim);
  }

  /// Range extent data accessor

  /// \return A pointer to the extent data (see Range::extent() )
  /// \note Not necessarily nullptr for rank-0 or null Range
  /// \throw nothing
  const index1_type* extent_data() const { return data() + (rank_ << 1); }

  /// Range extent accessor

  /// \return A range that contains the extent for each dimension.
  /// \throw nothing
  index_view_type extent() const {
    return index_view_type(extent_data(), rank_);
  }

  /// Range extent element accessor

  /// \return The extent of dimension \c dim.
  /// \throw nothing
  index1_type extent(size_t dim) const {
    TA_ASSERT(dim < rank_);
    return *(extent_data() + dim);
  }

  /// Range stride data accessor

  /// \return A pointer to the stride data (see Range::stride() )
  /// \note Not necessarily nullptr for rank-0 or null Range
  /// \throw nothing
  const index1_type* stride_data() const { return extent_data() + rank_; }

  /// Range stride accessor

  /// \return An index view that contains the stride for each dimension of
  /// the block range.
  /// \throw nothing
  index_view_type stride() const {
    return index_view_type(stride_data(), rank_);
  }

  /// Range stride element accessor

  /// \return The stride of dimension \c dim.
  /// \throw nothing
  index1_type stride(size_t dim) const {
    TA_ASSERT(dim < rank_);
    return *(stride_data() + dim);
  }

  /// Range volume accessor

  /// \return The total number of elements in the range, or 0 if this is a null
  /// Range \throw nothing
  ordinal_type volume() const { return volume_; }

  /// alias to volume() to conform to the TWG specification
  /// \return The total number of elements in the range.
  /// \throw nothing
  ordinal_type area() const { return volume_; }

  /// Range offset

  /// The range ordinal offset is equal to the dot product of the lower bound
  /// and stride vector. It is used internally to compute ordinal offsets.
  /// \return The ordinal index offset
  distance_type offset() const { return offset_; }

  /// Index iterator factory

  /// The iterator dereferences to an index. The order of iteration matches
  /// the data layout of a row-major tensor.
  /// \return An iterator that holds the lower bound index of a tensor (unless
  /// it has zero volume, then it returns same result as end()) \throw nothing
  const_iterator begin() const {
    return (volume_ > 0) ? const_iterator(lobound_data(), this) : end();
  }

  /// Index iterator factory

  /// The iterator dereferences to an index. The order of iteration matches
  /// the data layout of a row-major tensor.
  /// \return An iterator that holds the upper bound element index of a tensor
  /// \throw nothing
  const_iterator end() const { return const_iterator(upbound_data(), this); }

  /// Check the coordinate to make sure it is within the range.

  /// \tparam Index An integral range type
  /// \param index The coordinate index to check for inclusion in the range
  /// \return \c true when `i >= lobound` and `i < upbound`,
  /// otherwise \c false
  /// \throw TiledArray::Exception When the rank of this range is not
  /// equal to the size of the index.
  template <typename Index,
            typename std::enable_if<detail::is_integral_sized_range_v<Index>,
                                    bool>::type* = nullptr>
  bool includes(const Index& index) const {
    TA_ASSERT(*this);
    const auto* MADNESS_RESTRICT const lower = lobound_data();
    const auto* MADNESS_RESTRICT const upper = upbound_data();

    bool result = (rank_ > 0u);
    unsigned int d = 0;
    for (auto&& index_d : index) {
      TA_ASSERT(d < rank_);
      const auto lower_d = lower[d];
      const auto upper_d = upper[d];
      result = result && (index_d >= lower_d) && (index_d < upper_d);
#ifdef NDEBUG
      if (!result) {
        d = rank_;
        break;
      }
#endif
      ++d;
    }
    TA_ASSERT(d == rank_);

    return result;
  }

  /// Check the coordinate to make sure it is within the range.

  /// \tparam Index An integral type
  /// \param index The element index whose presence in the range is queried
  /// \return \c true when `i >= lobound` and `i < upbound`,
  /// otherwise \c false
  /// \throw TiledArray::Exception When the rank of this range is not
  /// equal to the size of the index.
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  bool includes(const std::initializer_list<Index>& index) const {
    return includes<std::initializer_list<Index>>(index);
  }

  /// Check the ordinal index to make sure it is within the range.

  /// \param i The ordinal index to check for inclusion in the range
  /// \return \c true when \c i \c >= \c 0 and \c i \c < \c volume
  /// \warning if this->order()==1 this
  /// \throw nothing
  template <typename Ordinal>
  typename std::enable_if<std::is_integral_v<Ordinal>, bool>::type includes(
      Ordinal i) const {
    TA_ASSERT(*this);
    // can't distinguish between includes(Index...) and includes(ordinal)
    // thus assume includes_ordinal() if this->rank()==1
    TA_ASSERT(this->rank() != 1 &&
              "use Range::includes(index) or "
              "Range::includes_ordinal(index_ordinal) if this->rank()==1");
    return include_ordinal_(i);
  }

  /// Check the ordinal index to make sure it is within the range.

  /// \param i The ordinal index to check for inclusion in the range
  /// \return \c true when \c i \c >= \c 0 and \c i \c < \c volume
  /// \throw nothing
  template <typename Ordinal>
  typename std::enable_if<std::is_integral_v<Ordinal>, bool>::type
  includes_ordinal(Ordinal i) const {
    TA_ASSERT(*this);
    return include_ordinal_(i);
  }

  template <typename... Index>
  std::enable_if_t<
      (sizeof...(Index) > 1ul) && (std::is_integral_v<Index> && ...), bool>
  includes(const Index&... index) const {
    const index1_type i[sizeof...(Index)] = {
        static_cast<index1_type>(index)...};
    return includes(i);
  }

  /// Permute this range

  /// \param perm The permutation to be applied to this range
  /// \return A reference to this range
  /// \throw TiledArray::Exception When the rank of this range is not
  /// equal to the rank of the permutation.
  Range_& operator*=(const Permutation& perm);

  /// Resize range to a new upper and lower bound

  /// \tparam Index1 An integral sized range type
  /// \tparam Index2 An integral sized range type
  /// \param lower_bound The lower bounds of the N-dimensional range
  /// \param upper_bound The upper bound of the N-dimensional range
  /// \return A reference to this range
  /// \throw TiledArray::Exception When the size of \c lower_bound is not
  /// equal to that of \c upper_bound.
  /// \throw TiledArray::Exception When `lower_bound[i] >= upper_bound[i]`
  template <
      typename Index1, typename Index2,
      typename = std::enable_if_t<detail::is_integral_sized_range_v<Index1> &&
                                  detail::is_integral_sized_range_v<Index2>>>
  Range_& resize(const Index1& lower_bound, const Index2& upper_bound) {
    using std::size;
    const auto n = size(lower_bound);
    TA_ASSERT(n == size(upper_bound));

    // Reallocate memory for range arrays
    if (rank_ != n) {
      init_datavec(n);
      rank_ = n;
    }
    if (n > 0ul)
      init_range_data(lower_bound, upper_bound);
    else
      volume_ = 0ul;

    return *this;
  }

  /// Shift the lower and upper bound of this range

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the range
  /// \return A reference to this range
  template <typename Index,
            typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
  Range_& inplace_shift(const Index& bound_shift) {
    auto* MADNESS_RESTRICT const lower = lobound_data_nc();
    auto* MADNESS_RESTRICT const upper = upbound_data_nc();
    const auto* MADNESS_RESTRICT const stride = upper + rank_ + rank_;

    // update the data
    offset_ = 0ul;
    unsigned int d = 0;
    for (auto&& bound_shift_d : bound_shift) {
      TA_ASSERT(d < rank_);
      // Load range data
      auto lower_d = lower[d];
      auto upper_d = upper[d];
      const auto stride_d = stride[d];

      // Compute new range bounds
      lower_d += bound_shift_d;
      upper_d += bound_shift_d;

      // Update range data
      offset_ += lower_d * stride_d;
      lower[d] = lower_d;
      upper[d] = upper_d;

      ++d;
    }
    TA_ASSERT(d == rank_);

    return *this;
  }

  /// Shift the lower and upper bound of this range

  /// \tparam Index An integral type
  /// \param bound_shift The shift to be applied to the range
  /// \return A reference to this range
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  Range_& inplace_shift(const std::initializer_list<Index>& bound_shift) {
    return inplace_shift<std::initializer_list<Index>>(bound_shift);
  }

  /// Create a Range with shiften lower and upper bounds

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the range
  /// \return A shifted copy of this range
  template <typename Index,
            typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
  Range_ shift(const Index& bound_shift) {
    Range_ result(*this);
    result.inplace_shift(bound_shift);
    return result;
  }

  /// Create a Range with shiften lower and upper bounds

  /// \tparam Index An integral type
  /// \param bound_shift The shift to be applied to the range
  /// \return A shifted copy of this range
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  Range_ shift(const std::initializer_list<Index>& bound_shift) {
    Range_ result(*this);
    result.inplace_shift(bound_shift);
    return result;
  }

  /// calculate the ordinal index of \c i

  /// This function is just a pass-through so the user can call \c ordinal() on
  /// a template parameter that can be a coordinate index or an integral.
  /// \param index Ordinal index
  /// \return \c index (unchanged)
  /// \throw When \c index is not included in this range
  ordinal_type ordinal(const ordinal_type index) const { return index; }

  /// calculate the ordinal index of \p index

  /// Convert a coordinate index to an ordinal index.
  /// \tparam Index An integral range type
  /// \param index The index to be converted to an ordinal index
  /// \return The ordinal index of \c index
  /// \throw When \c index is not included in this range.
  template <typename Index, typename std::enable_if_t<
                                detail::is_integral_range_v<Index>>* = nullptr>
  ordinal_type ordinal(const Index& index) const {
    auto* MADNESS_RESTRICT const stride = stride_data();

    ordinal_type result = 0ul;
    unsigned int d = 0;
    for (auto&& index_d : index) {
      TA_ASSERT(d < rank_);
      const auto stride_d = stride[d];
      result += index_d * stride_d;
      ++d;
    }
    TA_ASSERT(d == rank_);

    return result - offset_;
  }

  /// calculate the ordinal index of \p index

  /// Convert a coordinate index to an ordinal index.
  /// \tparam Index An integral type
  /// \param index The index to be converted to an ordinal index
  /// \return The ordinal index of \c index
  /// \throw When \c index is not included in this range.
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  ordinal_type ordinal(const std::initializer_list<Index>& index) const {
    return this->ordinal<std::initializer_list<Index>>(index);
  }

  /// calculate the ordinal index of \c index

  /// Convert a coordinate index to an ordinal index.
  /// \tparam Index A pack of integral types
  /// \param index The index to be converted to an ordinal index
  /// \return The ordinal index of \c index
  /// \throw When \c index is not included in this range.
  template <
      typename... Index,
      typename std::enable_if_t<(sizeof...(Index) > 1ul) &&
                                (std::is_integral_v<Index> && ...)>* = nullptr>
  ordinal_type ordinal(const Index&... index) const {
    const index1_type temp_index[sizeof...(Index)] = {
        static_cast<index1_type>(index)...};
    return ordinal(temp_index);
  }

  /// calculate the coordinate index of the ordinal index, \c ord.

  /// Convert an ordinal index to a coordinate index.
  /// \param ord Ordinal index
  /// \return The index of the ordinal index
  /// \throw TiledArray::Exception When \p ord is not included in this range
  index_type idx(ordinal_type ord) const {
    // Check that index is contained by range.
    // N.B. this will fail if any extent is zero
    TA_ASSERT(includes_ordinal(ord));

    // Construct result coordinate index object and allocate its memory.
    Range_::index result(rank_, 0);

    // Get pointers to the data
    auto* MADNESS_RESTRICT const result_data = result.data();
    const auto* MADNESS_RESTRICT const lower = lobound_data();
    const auto* MADNESS_RESTRICT const size = extent_data();

    // Compute the coordinate index of index in range.
    for (int i = int(rank_) - 1; i >= 0; --i) {
      const auto lower_i = lower[i];
      const auto size_i = size[i];

      // Compute result index element i
      const auto result_i = (ord % size_i) + lower_i;
      ord /= size_i;

      // Store result
      result_data[i] = result_i;
    }

    return result;
  }

  /// calculate the index of \c i

  /// This function is just a pass-through so the user can call \c idx() on
  /// a template parameter that can be an index or an ordinal_type.
  /// \tparam Index An integral range type
  /// \param i The index
  /// \return \c i (unchanged)
  template <typename Index, typename std::enable_if_t<
                                detail::is_integral_range_v<Index>>* = nullptr>
  const Index& idx(const Index& i) const {
    TA_ASSERT(includes(i));
    return i;
  }

  template <typename Archive>
  void serialize(Archive& ar) {
    ar & rank_;
    const auto four_x_rank = rank_ << 2;
    // read via madness::archive::wrap to be able to
    // - avoid having to serialize datavec_'s size
    // - read old archives that represented datavec_ by bare ptr
    if constexpr (madness::is_input_archive_v<Archive>) {
      datavec_.resize(four_x_rank);
      ar >> madness::archive::wrap(datavec_.data(), four_x_rank);
    } else if constexpr (madness::is_output_archive_v<Archive>) {
      ar << madness::archive::wrap(datavec_.data(), four_x_rank);
    } else
      abort();  // unreachable
    ar & offset_ & volume_;
  }

  void swap(Range_& other) {
    // Get temp data
    std::swap(datavec_, other.datavec_);
    std::swap(offset_, other.offset_);
    std::swap(volume_, other.volume_);
    std::swap(rank_, other.rank_);
  }

 private:
  /// Check that a signed integral value is included in this range

  /// \tparam Index A signed integral type
  /// \param i The ordinal index to check
  /// \return \c true when `i >= 0` and `i < volume_`, otherwise
  /// \c false.
  template <typename Index>
  typename std::enable_if<std::is_integral_v<Index> && std::is_signed_v<Index>,
                          bool>::type
  include_ordinal_(Index i) const {
    return (i >= Index(0)) && (i < Index(volume_));
  }

  /// Check that an unsigned integral value is include in this range

  /// \tparam Index An unsigned integral type
  /// \param i The ordinal index to check
  /// \return \c true when  `i < volume_`, otherwise \c false.
  template <typename Index>
  typename std::enable_if<
      std::is_integral_v<Index> && !std::is_signed<Index>::value, bool>::type
  include_ordinal_(Index i) const {
    return i < volume_;
  }

  /// Increment the coordinate index \c i in this range

  /// \param[in,out] i The coordinate index to be incremented
  /// \throw TiledArray::Exception When the rank of i is not equal to
  /// the rank of this range
  /// \throw TiledArray::Exception When \c i or \c i+n is outside this range
  void increment(index_type& i) const {
    TA_ASSERT(includes(i));

    const auto* MADNESS_RESTRICT const lower = lobound_data();
    const auto* MADNESS_RESTRICT const upper = upbound_data();

    for (int d = int(rank_) - 1; d >= 0; --d) {
      // increment coordinate
      ++i[d];

      // break if done
      if (i[d] < upper[d]) return;

      // Reset current index to lower bound.
      i[d] = lower[d];
    }

    // if the current location was set to lower then it was at the end and
    // needs to be reset to equal upper.
    std::copy(upper, upper + rank_, i.begin());
  }

  /// Advance the coordinate index \c i by \c n in this range

  /// \param[in,out] i The coordinate index to be advanced
  /// \param n The distance to advance \c i
  /// \throw TiledArray::Exception When the rank of i is not equal to
  /// the rank of this range
  /// \throw TiledArray::Exception When \c i or \c i+n is outside this range
  void advance(index_type& i, std::ptrdiff_t n) const {
    TA_ASSERT(includes(i));
    const auto o = ordinal(i) + n;
    TA_ASSERT(includes(o));
    i = idx(o);
  }

  /// Compute the distance between the coordinate indices \c first and \c last

  /// \param first The starting position in the range
  /// \param last The ending position in the range
  /// \return The difference between first and last, in terms of range positions
  /// \throw TiledArray::Exception When the size of \c first or \c last
  /// is not equal to the rank of this range
  /// \throw TiledArray::Exception When \c first or \c last is outside this
  /// range
  std::ptrdiff_t distance_to(const index_type& first,
                             const index_type& last) const {
    TA_ASSERT(includes(first));
    TA_ASSERT(includes(last));
    return ordinal(last) - ordinal(first);
  }

};  // class Range

// lift Range::index_type and Range::index_view_type into user-land
using Index = Range::index_type;
using IndexView = Range::index_view_type;

inline Range& Range::operator*=(const Permutation& perm) {
  TA_ASSERT(perm.size() == rank_);
  if (rank_ > 1ul) {
    // Copy the lower and upper bound data into a temporary array
    container::svector<index1_type, 2 * TA_MAX_SOO_RANK_METADATA> temp_lower(
        rank_ << 1);
    const auto* MADNESS_RESTRICT const temp_upper = temp_lower.data() + rank_;
    std::copy(lobound_data(), lobound_data() + (rank_ << 1), temp_lower.data());

    init_range_data(perm, temp_lower.data(), temp_upper);
  }
  return *this;
}

/// Exchange the values of the give two ranges.
inline void swap(Range& r0, Range& r1) {  // no throw
  r0.swap(r1);
}

/// Create a permuted range

/// \param perm The permutation to be applied to the range
/// \param r The range to be permuted
/// \return A permuted copy of \c r.
inline Range operator*(const Permutation& perm, const Range& r) {
  return Range(perm, r);
}

/// Create a permuted range

/// \param r The range to be permuted
/// \param perm The permutation to be applied to the range
/// \return A permuted copy of \c r.
/// \note this is an adaptor to BTAS' permute
template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
Range permute(const Range& r, std::initializer_list<I> perm) {
  return Permutation(perm) * r;
}

/// Range equality comparison

/// \param r1 The first range to be compared
/// \param r2 The second range to be compared
/// \return \c true when \p r1 represents the same range as \p r2, otherwise
/// \c false.
inline bool operator==(const Range& r1, const Range& r2) {
  return (r1.rank() == r2.rank()) &&
         !std::memcmp(r1.lobound_data(), r2.lobound_data(),
                      r1.rank() * (2u * sizeof(Range::index1_type)));
}

/// Range inequality comparison

/// \param r1 The first range to be compared
/// \param r2 The second range to be compared
/// \return \c true when \c r1 does not represent the same range as \c r2,
/// otherwise \c false.
inline bool operator!=(const Range& r1, const Range& r2) {
  return (r1.rank() != r2.rank()) ||
         std::memcmp(r1.lobound_data(), r2.lobound_data(),
                     r1.rank() * (2u * sizeof(Range::index1_type)));
}

/// Range output operator

/// \param os The output stream that will be used to print \c r
/// \param r The range to be printed
/// \return A reference to the output stream
inline std::ostream& operator<<(std::ostream& os, const Range& r) {
  os << "[ ";
  detail::print_array(os, r.lobound_data(), r.rank());
  os << ", ";
  detail::print_array(os, r.upbound_data(), r.rank());
  os << " )";
  return os;
}

/// Test the two ranges are congruent

/// This function tests that the rank and extent of
/// \c r1 are equal to those of \c r2.
/// \param r1 The first Range to compare
/// \param r2 The second Range to compare
inline bool is_congruent(const Range& r1, const Range& r2) {
  return (r1.rank() == r2.rank()) &&
         std::equal(r1.extent_data(), r1.extent_data() + r1.rank(),
                    r2.extent_data());
}

/// Tests whether a range is contiguous, i.e. whether its ordinal values form a
/// contiguous range

/// \param range a Range
/// \return true since TiledArray::Range is contiguous by definition
inline bool is_contiguous(const Range& range) { return true; }

namespace detail {

// TiledArray::detail::make_ta_range(rng) converts to its TA equivalent

/// "converts" TiledArray::Range into TiledArray::Range
inline const TiledArray::Range& make_ta_range(const TiledArray::Range& range) {
  return range;
}

}  // namespace detail

}  // namespace TiledArray
#endif  // TILEDARRAY_RANGE_H__INCLUDED
