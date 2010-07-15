#ifndef TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED
#define TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED

#include <TiledArray/type_traits.h>
#include <TiledArray/config.h>
#include <TiledArray/types.h>
#include <boost/array.hpp>
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

    template <unsigned int DIM>
    class DimensionOrder {
      typedef std::pair<unsigned int, unsigned int> DimOrderPair;
      typedef boost::array<unsigned int,DIM> IntArray;
    public:
      typedef typename IntArray::const_iterator const_iterator;
      typedef typename IntArray::const_reverse_iterator const_reverse_iterator;
      DimensionOrder(const DimensionOrderType& order) {
        // map dimensions to their importance order
        switch (order) {
          case increasing_dimension_order:
            for(unsigned int d = 0; d < DIM; ++d)
              ord_[d] = d;
          break;

          case decreasing_dimension_order:
            for(unsigned int d = 0; d < DIM; ++d)
              ord_[d] = DIM-d-1;
          break;

          default:
            throw std::runtime_error("Unknown dimension ordering. decreasing_dimension_order and increasing_dimension_order are allowed.");
          break;
        }

        init_();
      }

      unsigned int dim2order(unsigned int d) const { return ord_[d]; }
      unsigned int order2dim(unsigned int o) const { return dim_[o]; }
      /// use this to start iteration over dimensions in increasing order of their significance
      const_iterator begin() const { return dim_.begin(); }
      /// use this to end iteration over dimensions in increasing order of their significance
      const_iterator end() const { return dim_.end(); }
      /// use this to start iteration over dimensions in decreasing order of their significance
      const_reverse_iterator rbegin() const { return dim_.rbegin(); }
      /// use this to end iteration over dimensions in decreasing order of their significance
      const_reverse_iterator rend() const { return dim_.rend(); }
    private:

      static bool less_order_(const DimOrderPair& a, const DimOrderPair& b) {
        return a.second < b.second;
      }

      /// compute dim_ by inverting ord_ map
      void init_() {
        boost::array<DimOrderPair,DIM> sorted_by_order;
        for(unsigned int d=0; d<DIM; ++d)
          sorted_by_order[d] = DimOrderPair(d,ord_[d]);
        std::sort(sorted_by_order.begin(),sorted_by_order.end(), less_order_);
        // construct the order->dimension map
        for(unsigned int d=0; d<DIM; ++d)
          dim_[d] = sorted_by_order[d].first;
      }

      template <typename RandIter>
      bool valid_(RandIter first, RandIter last) {
        BOOST_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
        if((last - first) == DIM)
          return false;
        boost::array<int,DIM> count;
        count.assign(0);
        for(; first < DIM; ++first) {
          if( ((*first) >= DIM) || (count[ *first ] > 0) )
            return false;
          ++count[ *first ];
        }
        return true;
      }

      /// maps dimension to its order
      boost::array<unsigned int,DIM> ord_;
      /// maps order to dimension -- inverse of ord_
      boost::array<unsigned int,DIM> dim_;
    }; // class DimensionOrder

    template<typename C, detail::DimensionOrderType O>
    struct CoordIterator {
      typedef typename C::iterator iterator;
      typedef typename C::reverse_iterator reverse_iterator;

      static iterator begin(C& c) { return c.begin(); }
      static iterator end(C& c) { return c.end(); }
      static reverse_iterator rbegin(C& c) { return c.rbegin(); }
      static reverse_iterator rend(C& c) { return c.rend(); }
    };

    template<typename C, detail::DimensionOrderType O>
    struct CoordIterator<const C, O> {
      typedef typename C::const_iterator iterator;
      typedef typename C::const_reverse_iterator reverse_iterator;

      static iterator begin(const C& c) { return c.begin(); }
      static iterator end(const C& c) { return c.end(); }
      static reverse_iterator rbegin(const C& c) { return c.rbegin(); }
      static reverse_iterator rend(const C& c) { return c.rend(); }
    };

    template<typename C>
    struct CoordIterator<C, decreasing_dimension_order> {
      typedef typename C::reverse_iterator iterator;
      typedef typename C::iterator reverse_iterator;

      static iterator begin(C& c) { return c.rbegin(); }
      static iterator end(C& c) { return c.rend(); }
      static reverse_iterator rbegin(C& c) { return c.begin(); }
      static reverse_iterator rend(C& c) { return c.end(); }
    };

    template<typename C>
    struct CoordIterator<const C, decreasing_dimension_order> {
      typedef typename C::const_reverse_iterator iterator;
      typedef typename C::const_iterator reverse_iterator;

      static iterator begin(const C& c) { return c.rbegin(); }
      static iterator end(const C& c) { return c.rend(); }
      static reverse_iterator rbegin(const C& c) { return c.begin(); }
      static reverse_iterator rend(const C& c) { return c.end(); }
    };

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
    typedef typename detail::DimensionOrder<DIM>::const_iterator const_iterator;
    typedef typename detail::DimensionOrder<DIM>::const_reverse_iterator const_reverse_iterator;

    typedef detail::LevelTag<Level> level_tag;

    typedef I volume_type;
    typedef I ordinal_index;
    typedef ArrayCoordinate<I, DIM, level_tag > index;
    typedef boost::array<I, DIM> size_array;

    static const unsigned int dim = DIM;
    static const unsigned int level = Level;
    static const detail::DimensionOrderType order = O;

    static const_iterator begin() { return ordering_.begin(); }
    static const_reverse_iterator rbegin() { return ordering_.rbegin(); }
    static const_iterator end() { return ordering_.end(); }
    static const_reverse_iterator rend() { return ordering_.rend(); }

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
      calc_index_(i, begin(weight), end(weight), begin(result));
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

    /// Returns an iterator to the beginning of the least significant element.
    template<typename C>
    static typename detail::CoordIterator<C, O>::iterator begin(C& c) {
      return detail::CoordIterator<C, O>::begin(c);
    }

    /// Returns an iterator to the end of the least significant element.
    template<typename C>
    static typename detail::CoordIterator<C, O>::iterator end(C& c) {
      return detail::CoordIterator<C, O>::end(c);
    }

    /// Returns an iterator to the beginning of the most significant element.
    template<typename C>
    static typename detail::CoordIterator<C, O>::reverse_iterator rbegin(C& c) {
      return detail::CoordIterator<C, O>::rbegin(c);
    }

    /// Returns an iterator to the end of the most significant element.
    template<typename C>
    static typename detail::CoordIterator<C, O>::reverse_iterator rend(C& c) {
      return detail::CoordIterator<C, O>::rend(c);
    }

  private:
    /// Calculate the weighted dimension values.
    template<typename InIter, typename OutIter>
    static void calc_weight_(InIter first, InIter last, OutIter result) { // no throw
      for(typename std::iterator_traits<OutIter>::value_type weight = 1; first != last; ++first, ++result) {
        *result = weight;
        weight *= *first;
      }
    }

    /// Calculate the index of an ordinal.

    /// \arg \c o is the ordinal value.
    /// \arg \c [first, last) is the iterator range that contains the array
    /// weights from most significant to least significant.
    /// \arg \c result is an iterator to the index, which points to the most
    /// significant element.
    template<typename InIter, typename OutIter>
    static void calc_index_(ordinal_index o, InIter first, InIter last, OutIter result) {
      for(; first != last; ++first, ++result) {
        *result = o % *first;
        o -= *result * *first;
      }
    }

    /// Calculate the ordinal index of an array.

    /// \var \c [index_first, \c index_last) is a pair of iterators to the coordinate index.
    /// \var \c weight_first is an iterator to the array weights.
    template<typename IndexInIter, typename WeightInIter>
    static typename std::iterator_traits<IndexInIter>::value_type
    calc_ordinal_(IndexInIter index_first, IndexInIter index_last, WeightInIter weight_first) {
      return std::inner_product(index_first, index_last, weight_first,
          typename std::iterator_traits<IndexInIter>::value_type(0));
    }

    /// Calculate the ordinal index of an array.

    /// \var \c [index_first, \c index_last) is a pair of iterators to the coordinate index.
    /// \var \c weight_first is an iterator to the array weights.
    template<typename IndexInIter, typename WeightInIter, typename StartInIter>
    static typename std::iterator_traits<IndexInIter>::value_type
    calc_ordinal_(IndexInIter index_first, IndexInIter index_last, WeightInIter weight_first, StartInIter start_first) {
      typename std::iterator_traits<IndexInIter>::value_type o = 0;
      for(; index_first != index_last; ++index_first, ++weight_first, ++start_first)
        o += (*index_first - *start_first) * *weight_first;

      return o;
    }

    static detail::DimensionOrder<DIM> ordering_;
  };

  template <unsigned int DIM, unsigned int Level, detail::DimensionOrderType O, typename I>
  detail::DimensionOrder<DIM> CoordinateSystem<DIM, Level, O, I>::ordering_(O);

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
