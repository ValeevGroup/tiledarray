#ifndef TILEDARRAY_ANNOTATION_H__INCLUDED
#define TILEDARRAY_ANNOTATION_H__INCLUDED

#include <variable_list.h>
#include <vector>

namespace TiledArray {

  template<typename I, unsigned int DIM, typename Tag, typename CS>
  class ArrayCoordinate;

  namespace detail {
    template<typename InIter, typename OutIter>
    void calc_weight(InIter first, InIter last, OutIter result);
  } // namespace detail

  namespace expressions {

    class Annotation;
    void swap(Annotation&, Annotation&);

    /// Annotation for n-dimensional arrays.

    /// Contains the dimension and variable information for an n-dimension array.
    class Annotation {
    public:
      typedef std::size_t ordinal_type;
      typedef std::size_t volume_type;
      typedef std::vector<std::size_t> size_array;

      /// Default constructor

      /// Creates an empty Annotation
      Annotation() : size_(), weight_(), n_(0), var_(),
          order_(detail::decreasing_dimension_order)
      { }

      /// Create an Annotation.
      /// \var \c [size_first, \c size_last) is the size of each dimension.
      /// \var \c var is the variable annotation.
      /// \var \c o is the dimension order (optional).
      template<typename InIter>
      Annotation(InIter size_first, InIter size_last, const VariableList& var,
          detail::DimensionOrderType o = detail::decreasing_dimension_order) :
          size_(size_first, size_last), weight_((o == detail::decreasing_dimension_order ?
          calc_weight_<detail::decreasing_dimension_order>(size_) :
          calc_weight_<detail::increasing_dimension_order>(size_))),
          n_(std::accumulate(size_first, size_last, std::size_t(1), std::multiplies<std::size_t>())),
          var_(var), order_(o)
      { }

      /// Create an Annotation.
      /// \var \c [size_first, \c size_last) is the size of each dimension.
      /// \var \c [weight_first, \c weight_last) is the weight of each dimension.
      /// \var \c n is the number of elements in the array
      /// \var \c var is the variable annotation.
      /// \var \c o is the dimension order (optional).
      template<typename SizeInIter, typename WeightInIter>
      Annotation(SizeInIter size_first, SizeInIter size_last, WeightInIter weight_first,
          WeightInIter weight_last, std::size_t n, const VariableList& var,
          detail::DimensionOrderType o = detail::decreasing_dimension_order) :
          size_(size_first, size_last), weight_(weight_first, weight_last),
          n_(n), var_(var), order_(o)
      { }

      /// Copy constructor
      Annotation(const Annotation& other) :
          size_(other.size_), weight_(other.weight_), n_(other.n_),
          var_(other.var_), order_(other.order_)
      { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move constructor
      Annotation(Annotation&& other) :
          size_(std::move(other.size_)), weight_(std::move(other.weight_)),
          n_(other.n_), var_(std::move(other.var_)), order_(other.order_)
      { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      ~Annotation() { }

      /// Annotation assignment operator.
      Annotation& operator =(const Annotation& other) {
        TA_ASSERT(var_ == other.var_, std::runtime_error,
            "The variable lists do not match.");

        size_ = other.size_;
        weight_ = other.weight_;
        n_ = other.n_;
        order_ = other.order_;

        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Annotation move assignment operator.
      Annotation& operator =(Annotation&& other) {
        if(this != &other)
          swap(other);

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Returns a constant reference to a vector with the dimension sizes.
      const size_array& size() const { return size_; }
      /// Returns a constant reference to a vector with the dimension weights.
      const size_array& weight() const { return weight_; }
      /// Returns the number of elements contained by the array.
      volume_type volume() const { return n_; }
      /// Returns a constant reference to variable list (the annotation).
      const VariableList& vars() const { return var_; }
      /// Returns the number of dimensions of the array.
      unsigned int dim() const { return var_.dim(); }
      /// Return the array storage order
      detail::DimensionOrderType order() const { return order_; }

      /// Returns true if the index \c i is included by the array.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      bool includes(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
        TA_ASSERT(dim() == DIM, std::runtime_error,
            "Coordinate dimension is not equal to array dimension.");
        TA_ASSERT(order() == O, std::runtime_error,
            "Coordinate order does not match array dimension order.");
        for(unsigned int d = 0; d < dim(); ++d)
          if(size_[d] <= i[d])
            return false;

        return true;
      }

      /// Returns true if the ordinal index is included by this array.
      bool includes(const ordinal_type& i) const {
        return (i < n_);
      }

      template<unsigned int DIM>
      Annotation& operator ^=(const Permutation<DIM>& p) {
        Annotation temp = p ^ *this;
        swap(temp);

        return *this;
      }

    protected:

      /// Returns the ordinal index for the given index.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      ordinal_type ord_(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
        const typename ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >::index init = 0;
        return std::inner_product(i.begin(), i.end(), weight_.begin(), init);
      }

      /// Returns the given ordinal index.
      ordinal_type ord_(const ordinal_type i) const { return i; }

      /// Class wrapper function for detail::calc_weight() function.
      template<TiledArray::detail::DimensionOrderType O>
      static size_array calc_weight_(const size_array& size) { // no throw
        size_array result(size.size(), 0);
        TiledArray::detail::calc_weight(
            TiledArray::detail::CoordIterator<const size_array, O>::begin(size),
            TiledArray::detail::CoordIterator<const size_array, O>::end(size),
            TiledArray::detail::CoordIterator<size_array, O>::begin(result));
        return result;
      }

      void swap(Annotation& other){
        std::swap(size_, other.size_);
        std::swap(weight_, other.weight_);
        std::swap(n_, other.n_);
        TiledArray::expressions::swap(var_, other.var_);
        std::swap(order_, other.order_);
      }

      friend void swap(Annotation&, Annotation&);

      size_array size_;         ///< tile size
      size_array weight_;       ///< dimension weights
      ordinal_type n_;          ///< tile volume
      VariableList var_;        ///< variable list
      TiledArray::detail::DimensionOrderType order_; ///< Array order

    }; // class Annotation

    /// Exchange the values of a0 and a1.
    inline void swap(Annotation& a0, Annotation& a1) {
      a0.swap(a1);
    }

    template <unsigned int DIM>
    Annotation operator ^(const Permutation<DIM>& p, const Annotation& t) {
      TA_ASSERT((t.dim() == DIM), std::runtime_error,
          "The permutation dimension is not equal to the tile dimensions.");
      Annotation::size_array new_size = p ^ t.size();
      return Annotation(new_size.begin(), new_size.end(), p ^ t.vars());
    }

  } // namespace expressions
} //namespace TiledArray


#endif // TILEDARRAY_ANNOTATION_H__INCLUDED
