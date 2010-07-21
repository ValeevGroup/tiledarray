#ifndef TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED
#define TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/config.h>
#include <boost/array.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/bool.hpp>
#include <numeric>

namespace TiledArray {

  template <unsigned int>
  class LevelTag;
  template <typename I, unsigned int DIM, typename Tag>
  class ArrayCoordinate;

  namespace detail {

    /// Coordinate system level tag.

    /// This class is used to differentiate coordinate system types between
    /// tile coordinates (Level = 1) and element coordinate systems (Level = 0).
    template<unsigned int Level>
    struct LevelTag { };

    /// Dimension order types
    typedef enum {
      decreasing_dimension_order, ///< c-style dimension ordering
      increasing_dimension_order  ///< fortran dimension ordering
    } DimensionOrderType;

    /// Select the correct iterator factory function based on dimension ordering

    /// This non-specialized version handles non-const containers and increasing
    /// dimension orders.
    /// \tparam C The container object type
    /// \tparam O The dimension ordering
    template<typename C, DimensionOrderType O>
    struct CoordIterator {
      typedef typename C::iterator iterator;                  ///< Least to most significant iterator type
      typedef typename C::reverse_iterator reverse_iterator;  ///< Most to least significant iterator type

      /// Least to most significant begin iterator factory

      /// \return  An iterator pointing to the least significant element.
      static iterator begin(C& c) { return c.begin(); }

      /// Least to most significant end iterator factory

      /// \return  An iterator pointing to one element past the most significant
      /// element.
      static iterator end(C& c) { return c.end(); }

      /// Most to least significant begin iterator factory

      /// \return  An iterator pointing to the most significant element.
      static reverse_iterator rbegin(C& c) { return c.rbegin(); }

      /// Most to least significant end iterator factory

      /// \return  An iterator pointing to one element past the least significant
      /// element.
      static reverse_iterator rend(C& c) { return c.rend(); }
    }; // struct CoordIterator

    /// Select the correct iterator factory function based on dimension ordering

    /// This specialized version handles const containers and increasing
    /// dimension orders.
    /// \tparam C The container object type
    /// \tparam O The dimension ordering
    template<typename C, DimensionOrderType O>
    struct CoordIterator<const C, O> {
      typedef typename C::const_iterator iterator;                  ///< Least to most significant const iterator type
      typedef typename C::const_reverse_iterator reverse_iterator;  ///< Most to least significant const iterator type

      /// Least to most significant begin iterator factory

      /// \return  An iterator pointing to the least significant element.
      static iterator begin(const C& c) { return c.begin(); }

      /// Least to most significant end iterator factory

      /// \return  An iterator pointing to one element past the most significant
      /// element.
      static iterator end(const C& c) { return c.end(); }

      /// Most to least significant begin iterator factory

      /// \return  An iterator pointing to the most significant element.
      static reverse_iterator rbegin(const C& c) { return c.rbegin(); }

      /// Most to least significant end iterator factory

      /// \return  An iterator pointing to one element past the least significant
      /// element.
      static reverse_iterator rend(const C& c) { return c.rend(); }
    }; // CoordIterator<const C, O>

    /// Select the correct iterator factory function based on dimension ordering

    /// This non-specialized version handles non-const containers and decreasing
    /// dimension orders.
    /// \tparam C The container object type
    template<typename C>
    struct CoordIterator<C, decreasing_dimension_order> {
      typedef typename C::reverse_iterator iterator;  ///< Least to most significant iterator type
      typedef typename C::iterator reverse_iterator;  ///< Most to least significant iterator type

      /// Least to most significant begin iterator factory

      /// \return  An iterator pointing to the least significant element.
      static iterator begin(C& c) { return c.rbegin(); }

      /// Least to most significant end iterator factory

      /// \return  An iterator pointing to one element past the most significant
      /// element.
      static iterator end(C& c) { return c.rend(); }

      /// Most to least significant begin iterator factory

      /// \return  An iterator pointing to the most significant element.
      static reverse_iterator rbegin(C& c) { return c.begin(); }

      /// Most to least significant end iterator factory

      /// \return  An iterator pointing to one element past the least significant
      /// element.
      static reverse_iterator rend(C& c) { return c.end(); }
    }; // CoordIterator<C, decreasing_dimension_order>

    /// Select the correct iterator factory function based on dimension ordering

    /// This non-specialized version handles const containers and decreasing
    /// dimension orders.
    /// \tparam C The container object type
    template<typename C>
    struct CoordIterator<const C, decreasing_dimension_order> {
      typedef typename C::const_reverse_iterator iterator;  ///< Least to most significant const iterator type
      typedef typename C::const_iterator reverse_iterator;  ///< Most to least significant const iterator type

      /// Least to most significant begin iterator factory

      /// \return  An iterator pointing to the least significant element.
      static iterator begin(const C& c) { return c.rbegin(); }

      /// Least to most significant end iterator factory

      /// \return  An iterator pointing to one element past the most significant
      /// element.
      static iterator end(const C& c) { return c.rend(); }

      /// Most to least significant begin iterator factory

      /// \return  An iterator pointing to the most significant element.
      static reverse_iterator rbegin(const C& c) { return c.begin(); }

      /// Most to least significant end iterator factory

      /// \return  An iterator pointing to one element past the least significant
      /// element.
      static reverse_iterator rend(const C& c) { return c.end(); }
    }; // struct CoordIterator<const C, decreasing_dimension_order>

  } // namespace detail

  /// CoordinateSystem is a policy class that specifies e.g. the order of significance of dimension.
  /// This allows to, for example, to define order of iteration to be compatible with C or Fortran arrays.
  /// Specifies the details of a D-dimensional coordinate system.
  /// The default is for the last dimension to be least significant.
  template <unsigned int DIM, unsigned int Level = 1u, detail::DimensionOrderType O = detail::decreasing_dimension_order, typename I = std::size_t>
  class CoordinateSystem {
    // Static asserts
    BOOST_STATIC_ASSERT(boost::is_integral<I>::value);

  public:
    typedef detail::LevelTag<Level> level_tag;  ///< Coordinate system level (Elements: Level = 0, Tiles: Level = 1)

    typedef I volume_type;
    typedef I ordinal_index;
    typedef ArrayCoordinate<I, DIM, level_tag > index;
    typedef boost::array<I, DIM> size_array;

    static const unsigned int dim = DIM;
    static const unsigned int level = Level;
    static const detail::DimensionOrderType order = O;

    /// Calculate the weighted dimension values.
    static size_array calc_weight(const size_array& size) { // no throw
      size_array result;
      calc_weight_(begin(size), end(size), begin(result));
      return result;
    }

    /// Calculate the index of an ordinal.

    /// \arg \c o is the ordinal value.
    /// \arg \c [first, last) is the iterator range that contains the array
    /// weights from most significant to least significant.
    /// \arg \c result is an iterator to the index, which points to the most
    /// significant element.
    static index calc_index(ordinal_index i, const size_array& weight) {
      index result;
      calc_index_(i, rbegin(weight), rend(weight), rbegin(result));
      return result;
    }

    /// Calculate the ordinal index of an array.

    /// \var \c [index_first, \c index_last) is a pair of iterators to the coordinate index.
    /// \var \c weight_first is an iterator to the array weights.
    static ordinal_index calc_ordinal(const index& i, const size_array& weight) {
      return calc_ordinal_(begin(i), end(i), begin(weight));
    }

    /// Calculate the ordinal index of an array.

    /// \var \c [index_first, \c index_last) is a pair of iterators to the coordinate index.
    /// \var \c weight_first is an iterator to the array weights.
    static ordinal_index calc_ordinal(const index& i, const size_array& weight, const index& start) {
      return calc_ordinal_(begin(i), end(i), begin(weight), begin(start));
    }

    /// Calculate the volume of an N-dimensional orthogonal.
    static volume_type calc_volume(const size_array& size) { // no throw
      return std::accumulate(size.begin(), size.end(), volume_type(1), std::multiplies<volume_type>());
    }

    /// Increment a coordinate index within a range.

    /// This will increment a coordinate within the range \c [start, \c finish).
    /// If the end of the range is reached, current will be equal to finish.
    /// \param[out] current The current coordinate index to be incremented.
    /// \param[in] start The start index of the coordinate range.
    /// \param[in] finish The finish index of the coordinate range.
    /// \throw std::runtime_error When current is not bounded by \c [start,
    /// \c finish)
    static void increment_coordinate(index& current, const index& start, const index& finish) {
      TA_ASSERT(start <= current, std::runtime_error,
          "Current coordinate is less than start coordinate.");
      TA_ASSERT(current < finish, std::runtime_error,
          "Current coordinate is less than start coordinate.");

      increment_coordinate_(begin(current), end(current), begin(start), begin(finish));

      // if the current location was set to start then it was at the end and
      // needs to be reset to equal finish.
      if(std::equal(current.begin(), current.end(), start.begin()))
        std::copy(finish.begin(), finish.end(), current.begin());
    }

    /// Index lexicographical less-than comparison

    /// \param i1 The left index to be compared
    /// \param i2 The right index to be compared
    /// \return \c true when all elements of i1 are lexicographically less-than
    /// all elements of i2.
    static bool less(const index& i1, const index& i2) {
      return std::lexicographical_compare(begin(i1), end(i1), begin(i2), end(i2));
    }

    /// Index lexicographical less-than-or-equal-to comparison

    /// \param i1 The left index to be compared
    /// \param i2 The right index to be compared
    /// \return \c true when all elements of i1 are lexicographically
    /// less-than-or-equal-to all elements of i2.
    static bool less_eq(const index& i1, const index& i2) {
      return ! less(i2, i1);
    }

    /// Index lexicographical greater-than comparison

    /// \param i1 The left index to be compared
    /// \param i2 The right index to be compared
    /// \return \c true when all elements of i1 are lexicographically less-than
    /// all elements of i2.
    static bool greater(const index& i1, const index& i2) {
      return less(i2, i1);
    }

    /// Index lexicographical greater-than-or-equal-to comparison

    /// \param i1 The left index to be compared
    /// \param i2 The right index to be compared
    /// \return \c true when all elements of i1 are lexicographically
    /// greater-than-or-equal-to all elements of i2.
    static bool greater_eq(const index& i1, const index& i2) {
      return ! less(i1, i2);
    }

    /// Least significant begin iterator factory selector

    /// Constness of the container will be observed when generating the
    /// iterator. That is, when the array container is const, the iterator
    /// type is const; and likewise for non-const arrays.
    /// \note If you need the const iterator type when you have a non-const
    /// array, use \c static_cast to cast the object to a const reference before
    /// passing it to this function.
    /// \tparam C The array container type
    /// \param c The array container
    /// \return An iterator to the least significant element of the array.
    template<typename C>
    static typename detail::CoordIterator<C, O>::iterator begin(C& c) {
      return detail::CoordIterator<C, O>::begin(c);
    }

    /// Least significant end iterator factory selector

    /// Constness of the container will be observed when generating the
    /// iterator. That is, when the array container is const, the iterator
    /// type is const; and likewise for non-const arrays.
    /// \note If you need the const iterator type when you have a non-const
    /// array, use \c static_cast to cast the object to a const reference before
    /// passing it to this function.
    /// \tparam C The array container type
    /// \param c The array container
    /// \return An iterator to the least significant element of the array.
    template<typename C>
    static typename detail::CoordIterator<C, O>::iterator end(C& c) {
      return detail::CoordIterator<C, O>::end(c);
    }

    /// Most significant begin iterator factory selector

    /// Constness of the container will be observed when generating the
    /// iterator. That is, when the array container is const, the iterator
    /// type is const; and likewise for non-const arrays.
    /// \note If you need the const iterator type when you have a non-const
    /// array, use \c static_cast to cast the object to a const reference before
    /// passing it to this function.
    /// \tparam C The array container type
    /// \param c The array container
    /// \return An iterator to the most significant element of the array.
    template<typename C>
    static typename detail::CoordIterator<C, O>::reverse_iterator rbegin(C& c) {
      return detail::CoordIterator<C, O>::rbegin(c);
    }

    /// Most significant end iterator factory selector

    /// Constness of the container will be observed when generating the
    /// iterator. That is, when the array container is const, the iterator
    /// type is const; and likewise for non-const arrays.
    /// \note If you need the const iterator type when you have a non-const
    /// array, use \c static_cast to cast the object to a const reference before
    /// passing it to this function.
    /// \tparam C The array container type
    /// \param c The array container
    /// \return An iterator to the most significant element of the array.
    template<typename C>
    static typename detail::CoordIterator<C, O>::reverse_iterator rend(C& c) {
      return detail::CoordIterator<C, O>::rend(c);
    }

  private:
    /// Calculate the weighted dimension values.

    /// \tparam InIter Input iterator type for the size
    /// \tparam OutIter Output iterator type for the weight
    /// \param[in] first The first iterator pointing to the least significant
    /// element of the size array
    /// \param[in] last The last iterator pointing to one past the most
    /// significant element of the size array
    /// \param[out] result The first iterator pointing to the least significant
    /// element of the weight array
    /// \throw nothing
    template<typename InIter, typename OutIter>
    static void calc_weight_(InIter first, InIter last, OutIter result) { // no throw
      typedef typename std::iterator_traits<OutIter>::value_type value_type;

      for(value_type weight = 1; first != last; ++first, ++result) {
        *result = weight;
        weight *= *first;
      }
    }

    /// Calculate the index of an ordinal.

    /// \tparam InIter Input iterator type for the weight
    /// \tparam OutIter Output iterator type for the coordinate index
    /// \param[in] o is the ordinal value
    /// \param[in] first The first iterator pointing to the most significant
    /// element of the weight array
    /// \param[in] last The last iterator pointing to one past the least
    /// significant element of the weight array
    /// \param[out] result An iterator to the first points to the most
    /// significant element of the coordinate index
    /// \throw nothing
    template<typename InIter, typename OutIter>
    static void calc_index_(ordinal_index o, InIter first, InIter last, OutIter result) {
      for(; first != last; ++first, ++result) {
        *result = o % *first;
        o -= *result * *first;
      }
    }

    /// Calculate the ordinal index of an array.

    /// \param index_first First coordinate index iterator
    /// \param index_last Last coordinate index iterator
    /// \param weight_first First weight iterator
    /// \return The ordinal index corresponding to the coordinate index
    /// \throw nothing
    template<typename IndexInIter, typename WeightInIter>
    static typename std::iterator_traits<IndexInIter>::value_type
    calc_ordinal_(IndexInIter index_first, IndexInIter index_last, WeightInIter weight_first) {
      return std::inner_product(index_first, index_last, weight_first,
          typename std::iterator_traits<IndexInIter>::value_type(0));
    }

    /// Calculate the ordinal index of an array.

    /// \param index_first First coordinate index iterator
    /// \param index_last Last coordinate index iterator
    /// \param weight_first First weight iterator
    /// \return The ordinal index corresponding to the coordinate index
    /// \throw nothing
    template<typename IndexInIter, typename WeightInIter, typename StartInIter>
    static typename std::iterator_traits<IndexInIter>::value_type
    calc_ordinal_(IndexInIter index_first, IndexInIter index_last, WeightInIter weight_first, StartInIter start_first) {
      typename std::iterator_traits<IndexInIter>::value_type o = 0;
      for(; index_first != index_last; ++index_first, ++weight_first, ++start_first)
        o += (*index_first - *start_first) * *weight_first;

      return o;
    }

    /// Increment a coordinate index

    /// \tparam ForIter The forward iterator type of the current index
    /// \tparam InIter The input iterator type for the start and finish indexes
    /// \param first_cur The first iterator pointing to the least significant
    /// element of the current index.
    /// \param last_cur The last iterator pointing to one past the most
    /// significant element of the current index.
    /// \param start The first iterator pointing to the least significant
    /// element of the start index
    /// \param start The first iterator pointing to the least significant
    /// element of the finish index
    /// \throw nothing
    template <typename ForIter, typename InIter>
    static void increment_coordinate_(ForIter first_cur, ForIter last_cur, InIter start, InIter finish) {
      for(; first_cur != last_cur; ++first_cur, ++start, ++finish) {
        // increment coordinate
        ++(*first_cur);

        // break if done
        if( *first_cur < *finish)
          return;

        // Reset current index to start value.
        *first_cur = *start;
      }
    }
  }; // class CoordinateSystem

  namespace detail {

    // The following code is designed to check for the sameness of the different
    // coordinate system properties at compile time.

    /// This class is used for compile-time coordinate system checking.

    /// This class is inherited from \c boost::true_type when the dimensions are
    /// the same, otherwise it is inherited from \c boost::false_type.
    /// \tparam T1 A CoordinateSystem<> type or a type with \c typename
    /// \c T1::coordinate_system, where \c coordinate_system is a
    /// \c CoordinateSystem<>.
    template <typename T1, typename T2>
    struct same_cs_dim : public boost::mpl::equal_to<
        boost::integral_constant<unsigned int, T1::coordinate_system::dim>,
        boost::integral_constant<unsigned int, T2::coordinate_system::dim> >::type
    { };

    template <unsigned int D1, unsigned int L1, DimensionOrderType O1, typename I1,
              unsigned int D2, unsigned int L2, DimensionOrderType O2, typename I2>
    struct same_cs_dim<CoordinateSystem<D1, L1, O1, I1>, CoordinateSystem<D2, L2, O2, I2> > :
        public boost::mpl::equal_to<boost::integral_constant<unsigned int, D1>,
        boost::integral_constant<unsigned int, D2> >::type
    { };

    template <unsigned int D1, unsigned int L1, DimensionOrderType O1, typename I1, typename T2>
    struct same_cs_dim<CoordinateSystem<D1, L1, O1, I1>, T2> :
        public boost::mpl::equal_to<boost::integral_constant<unsigned int, D1>,
        boost::integral_constant<unsigned int, T2::coordinate_system::dim> >::type
    { };

    template <typename T1, unsigned int D2, unsigned int L2, DimensionOrderType O2, typename I2>
    struct same_cs_dim<T1, CoordinateSystem<D2, L2, O2, I2> > :
        public boost::mpl::equal_to<boost::integral_constant<unsigned int, T1::coordinate_system::dim>,
        boost::integral_constant<unsigned int, D2> >::type
    { };

    /// This class is used for compile-time coordinate system checking.

    /// This class is inherited from \c boost::true_type when the levels are the
    /// same, otherwise it is inherited from \c boost::false_type.
    /// \tparam T1 A CoordinateSystem<> type or a type with \c typename
    /// \c T1::coordinate_system, where \c coordinate_system is a
    /// \c CoordinateSystem<>.
    /// \tparam T2 Same as T1.
    template <typename T1, typename T2>
    struct same_cs_level : public boost::mpl::equal_to<
        boost::integral_constant<unsigned int, T1::coordinate_system::level>,
        boost::integral_constant<unsigned int, T2::coordinate_system::level> >::type
    { };

    template <unsigned int D1, unsigned int L1, DimensionOrderType O1, typename I1,
              unsigned int D2, unsigned int L2, DimensionOrderType O2, typename I2>
    struct same_cs_level<CoordinateSystem<D1, L1, O1, I1>, CoordinateSystem<D2, L2, O2, I2> > :
        public boost::mpl::equal_to<boost::integral_constant<unsigned int, L1>,
        boost::integral_constant<unsigned int, L2> >::type
    { };

    template <unsigned int D1, unsigned int L1, DimensionOrderType O1, typename I1, typename T2>
    struct same_cs_level<CoordinateSystem<D1, L1, O1, I1>, T2> :
        public boost::mpl::equal_to<boost::integral_constant<unsigned int, L1>,
        boost::integral_constant<unsigned int, T2::coordinate_system::dim> >::type
    { };

    template <typename T1, unsigned int D2, unsigned int L2, DimensionOrderType O2, typename I2>
    struct same_cs_level<T1, CoordinateSystem<D2, L2, O2, I2> > :
        public boost::mpl::equal_to<
        boost::integral_constant<unsigned int, T1::coordinate_system::dim>,
        boost::integral_constant<unsigned int, L2> >::type
    { };

    /// This class is used for compile-time coordinate system checking.

    /// This class is inherited from \c boost::true_type when the orders are the
    /// same, otherwise it is inherited from \c boost::false_type.
    /// \tparam T1 A CoordinateSystem<> type or a type with \c typename
    /// \c T1::coordinate_system, where \c coordinate_system is a
    /// \c CoordinateSystem<>.
    /// \tparam T2 Same as T1.
    template <typename T1, typename T2>
    struct same_cs_order : public boost::mpl::equal_to<
        boost::integral_constant<DimensionOrderType, T1::coordinate_system::order>,
        boost::integral_constant<DimensionOrderType, T2::coordinate_system::order> >::type
    { };

    template <unsigned int D1, unsigned int L1, DimensionOrderType O1, typename I1,
              unsigned int D2, unsigned int L2, DimensionOrderType O2, typename I2>
    struct same_cs_order<CoordinateSystem<D1, L1, O1, I1>, CoordinateSystem<D2, L2, O2, I2> > :
        public boost::mpl::equal_to<
        boost::integral_constant<DimensionOrderType, O1>,
        boost::integral_constant<DimensionOrderType, O2> >::type
    { };

    template <unsigned int D1, unsigned int L1, DimensionOrderType O1, typename I1, typename T2>
    struct same_cs_order<CoordinateSystem<D1, L1, O1, I1>, T2> :
        public boost::mpl::equal_to<boost::integral_constant<DimensionOrderType, O1>,
        boost::integral_constant<DimensionOrderType, T2::coordinate_system::order> >::type
    { };

    template <typename T1, unsigned int D2, unsigned int L2, DimensionOrderType O2, typename I2>
    struct same_cs_order<T1, CoordinateSystem<D2, L2, O2, I2> > :
        public boost::mpl::equal_to<boost::integral_constant<DimensionOrderType, T1::coordinate_system::order>,
        boost::integral_constant<DimensionOrderType, O2> >::type
    { };

    /// This class is used for compile-time coordinate system checking.

    /// This class is inherited from \c boost::true_type when the ordinal_index
    /// types are the same, otherwise it is inherited from \c boost::false_type.
    /// \tparam T1 A CoordinateSystem<> type or a type with \c typename
    /// \c T1::coordinate_system, where \c coordinate_system is a
    /// \c CoordinateSystem<>.
    /// \tparam T2 Same as T1.
    template <typename T1, typename T2>
    struct same_cs_index : public boost::is_same<typename T1::coordinate_system::ordinal_index,
        typename T2::coordinate_system::ordinal_index> { };

    template <unsigned int D1, unsigned int L1, DimensionOrderType O1, typename I1,
              unsigned int D2, unsigned int L2, DimensionOrderType O2, typename I2>
    struct same_cs_index<CoordinateSystem<D1, L1, O1, I1>, CoordinateSystem<D2, L2, O2, I2> > :
        public boost::is_same<I1, I2> { };

    template <unsigned int D1, unsigned int L1, DimensionOrderType O1, typename I1, typename T2>
    struct same_cs_index<CoordinateSystem<D1, L1, O1, I1>, T2> :
        public boost::is_same<I1, typename T2::coordinate_system::ordinal_index> { };

    template <typename T1, unsigned int D2, unsigned int L2, DimensionOrderType O2, typename I2>
    struct same_cs_index<T1, CoordinateSystem<D2, L2, O2, I2> > :
        public boost::is_same<typename T1::coordinate_system::ordinal_index, I2> { };

    /// This class is used for compile-time coordinate system checking.

    /// This template will check that the two coordinate systems have the same
    /// level, ordering, and ordinal index type. If all three are the same, then
    /// \c compatible_coordinate_system will be inherited from
    /// \c boost::true_type, otherwise, it will be inherited form
    /// \c boost::false_type. See the Boost documentation for details.
    /// \tparam CS1 A CoordinateSystem<> type or a type with \c typename
    /// \c CS1::coordinate_system, where \c coordinate_system is a
    /// \c CoordinateSystem<>.
    /// \tparam CS2 Same as CS1.
    template <typename CS1, typename CS2>
    struct compatible_coordinate_system :
        public boost::integral_constant<bool, (same_cs_level<CS1, CS2>::value
        && same_cs_order<CS1, CS2>::value && same_cs_index<CS1, CS2>::value) >
    { };

  }  // namespace detail

} // namespace TiledArray

#endif // TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED
