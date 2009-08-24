#ifndef TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED
#define TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED

#include <boost/array.hpp>

namespace TiledArray {

  namespace detail {

    typedef enum {
      decreasing_dimension_order, // c-style
      increasing_dimension_order  // fortran
    } DimensionOrderType;

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

        init_(ord_.begin(), ord_.end());
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
      template <typename RandIter>
      void init_(RandIter first, RandIter last) {
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

      static iterator begin(C& c) { return c.begin(); }
      static iterator end(C& c) { return c.end(); }
    };

    template<typename C, detail::DimensionOrderType O>
    struct CoordIterator<const C, O> {
      typedef typename C::const_iterator iterator;

      static iterator begin(const C& c) { return c.begin(); }
      static iterator end(const C& c) { return c.end(); }
    };

    template<typename C>
    struct CoordIterator<C, decreasing_dimension_order> {
      typedef typename C::reverse_iterator iterator;

      static iterator begin(C& c) { return c.rbegin(); }
      static iterator end(C& c) { return c.rend(); }
    };

    template<typename C>
    struct CoordIterator<const C, decreasing_dimension_order> {
      typedef typename C::const_reverse_iterator iterator;

      static iterator begin(const C& c) { return c.rbegin(); }
      static iterator end(const C& c) { return c.rend(); }
    };

  }; // namespace detail

  /// CoordinateSystem is a policy class that specifies e.g. the order of significance of dimension.
  /// This allows to, for example, to define order of iteration to be compatible with C or Fortran arrays.
  /// Specifies the details of a D-dimensional coordinate system.
  /// The default is for the last dimension to be least significant.
  template <unsigned int DIM, detail::DimensionOrderType Order = detail::decreasing_dimension_order>
  class CoordinateSystem {
  public:
    typedef typename detail::DimensionOrder<DIM>::const_iterator const_iterator;
    typedef typename detail::DimensionOrder<DIM>::const_reverse_iterator const_reverse_iterator;

    static unsigned int dim() { return DIM; }
    static const detail::DimensionOrder<DIM>& ordering() { return ordering_; }
    static const_iterator begin() { return ordering_.begin(); }
    static const_reverse_iterator rbegin() { return ordering_.rbegin(); }
    static const_iterator end() { return ordering_.end(); }
    static const_reverse_iterator rend() { return ordering_.rend(); }

    template<typename C>
    static typename detail::CoordIterator<C, Order>::iterator begin(C& c) {
      return detail::CoordIterator<C, Order>::begin(c);
    }

    template<typename C>
    static typename detail::CoordIterator<C, Order>::iterator end(C& c) {
      return detail::CoordIterator<C, Order>::end(c);
    }

    static const detail::DimensionOrderType dimension_order = Order;
  private:
    static detail::DimensionOrder<DIM> ordering_;
  };

  template <unsigned int DIM, detail::DimensionOrderType Order>
  detail::DimensionOrder<DIM> CoordinateSystem<DIM,Order>::ordering_(Order);

} // namespace TiledArray

#endif // TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED
