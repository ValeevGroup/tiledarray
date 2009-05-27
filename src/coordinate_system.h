#ifndef COORDINATE_SYSTEM_H__INCLUDED
#define COORDINATE_SYSTEM_H__INCLUDED

#include <boost/array.hpp>

namespace TiledArray {

  namespace {
    // sort dimensions by their order
    struct DimOrderPair {
      unsigned int dim, order;
      DimOrderPair(unsigned int d = 0,unsigned int o = 0) : dim(d), order(o) {}
      /// compare by order
      bool operator<(const DimOrderPair& other) const {
        return order < other.order;
      }
    };
  };

  namespace detail {

    typedef enum {
      decreasing_dimension_order,
      increasing_dimension_order,
      general_dimension_order
    } DimensionOrderType;

    template <unsigned int DIM>
    class DimensionOrder {
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

          case general_dimension_order:
            throw std::runtime_error("general dimension ordering is not supported");
          break;
        }

        //
        // compute dim_ by inverting ord_ map
        //
        boost::array<DimOrderPair,DIM> sorted_by_order;
        for(unsigned int d=0; d<DIM; ++d)
          sorted_by_order[d] = DimOrderPair(d,ord_[d]);
        std::sort(sorted_by_order.begin(),sorted_by_order.end());
        // construct the order->dimension map
        for(unsigned int d=0; d<DIM; ++d)
          dim_[d] = sorted_by_order[d].dim;
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
      /// maps dimension to its order
      boost::array<unsigned int,DIM> ord_;
      /// maps order to dimension -- inverse of ord_
      boost::array<unsigned int,DIM> dim_;
    }; // class DimensionOrder

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

    static const detail::DimensionOrderType dimension_order = Order;
  private:
    static detail::DimensionOrder<DIM> ordering_;
  };

  template <unsigned int DIM, detail::DimensionOrderType Order>
  detail::DimensionOrder<DIM> CoordinateSystem<DIM,Order>::ordering_(Order);


} // namespace TiledArray

#endif // COORDINATE_SYSTEM_H__INCLUDED
