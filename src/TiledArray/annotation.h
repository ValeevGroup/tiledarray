#ifndef TILEDARRAY_ANNOTATION_H__INCLUDED
#define TILEDARRAY_ANNOTATION_H__INCLUDED

#include <TiledArray/variable_list.h>
#include <TiledArray/array_ref.h>
#include <TiledArray/type_traits.h>
#include <boost/type_traits.hpp>
#include <vector>

namespace TiledArray {

  template<typename I, unsigned int DIM, typename Tag, typename CS>
  class ArrayCoordinate;

  namespace detail {
    template<typename InIter, typename OutIter>
    void calc_weight(InIter first, InIter last, OutIter result);
  } // namespace detail

  namespace expressions {

    template<typename I>
    class Annotation;
    template<typename I>
    void swap(Annotation<I>&, Annotation<I>&);

    /// Annotation for n-dimensional arrays.

    /// Contains the dimension and variable information for an n-dimension array.
    template<typename I>
    class Annotation {
    public:
      typedef Annotation<I> Annotation_;
      typedef typename boost::remove_const<I>::type ordinal_type;
      typedef typename boost::remove_const<I>::type volume_type;
      typedef detail::ArrayRef<I> size_array;

    protected:
      /// Default constructor

      /// Note: Annotation will not be in a usable state with this constructor.
      /// The calling derived class must call init_from_size_() to correctly
      /// initialize this object.
      Annotation() : size_(), weight_(), n_(0), var_(),
          order_(detail::decreasing_dimension_order)
      { }

      /// Initialize the annotation with a size array.

      /// This function will use the pointer data to store the size and weight
      /// information for the annotation, it's must be at least 2 * DIM so it
      /// can hold all the data. The size and weight data does not need to be
      /// calculated by the calling derived class, that will be done by this
      /// function. The derived class which calles this function is responsible
      /// for freeing data if it is dynamically allocated memory. All the data
      /// members of Annotation will be Initialized by the size, var, and o
      /// arguments.
      /// \var \c size is an array with the sizes of each dimension.
      /// \var \c var is the variable annotation.
      /// \var \c o is the dimension order.
      /// \var \c data is the array that Annotation will use to store the size and weight information.
      template<typename SizeArray>
      void init_from_size_(const SizeArray& size, const VariableList& var,
          detail::DimensionOrderType o, typename boost::remove_const<I>::type* data)
      {

        const unsigned int dim = var.dim();
        Annotation_::size_ = std::make_pair(data, data + dim);
        Annotation_::weight_ = std::make_pair(data + dim, data + dim + dim);
        std::copy(size.begin(), size.end(), size_.begin());
        detail::calc_weight(size_.begin(), size_.end(), weight_.begin());
        n_ = detail::volume(size_.begin(), size_.end());
        var_ = var;
        order_ = o;
      }

    public:

      /// Create an Annotation.

      /// \var \c [size_first, \c size_last) is the size of each dimension.
      /// \var \c [weight_first, \c weight_last) is the weight of each dimension.
      /// \var \c n is the number of elements in the array
      /// \var \c var is the variable annotation.
      /// \var \c o is the dimension order (optional).
      Annotation(I* size_first, I* size_last, I* weight_first, I* weight_last,
          volume_type n, const VariableList& var,
          detail::DimensionOrderType o = detail::decreasing_dimension_order) :
          size_(size_first, size_last), weight_(weight_first, weight_last),
          n_(n), var_(var), order_(o)
      {
        TA_ASSERT(var.dim() == (size_last - size_first), std::runtime_error,
            "Variable list dimensions do not match size initialization list dimensions.");
        TA_ASSERT(var.dim() == (weight_last - weight_first), std::runtime_error,
            "Variable list dimensions do not match weight initialization list dimensions.");
        TA_ASSERT(n == std::accumulate(size_first, size_last, ordinal_type(1), std::multiplies<ordinal_type>()),
            std::runtime_error, "Variable list dimensions do not match specified dimensions.");
      }

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
        size_ = std::move(other.size_);
        weight_ = std::move(other.weight_);
        n_ = other.n_;
        order_ = other.order_;

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
      template<typename T, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      bool includes(const ArrayCoordinate<T,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
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
        size_ ^= p;
        detail::calc_weight(size_.begin(), size_.end(), weight_.begin());
        var_ ^= p;
        return *this;
      }

    protected:

      /// Returns the ordinal index for the given index.
      template<typename T, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      ordinal_type ord_(const ArrayCoordinate<T,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
        const typename ArrayCoordinate<T,DIM,Tag, CoordinateSystem<DIM,O> >::index init = 0;
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

      void swap(Annotation& other) {
        detail::swap(size_, other.size_);
        detail::swap(weight_, other.weight_);
        std::swap(n_, other.n_);
        TiledArray::expressions::swap(var_, other.var_);
        std::swap(order_, other.order_);
      }

      friend void TiledArray::expressions::swap<>(Annotation_&, Annotation_&);

      size_array size_;         ///< tile size
      size_array weight_;       ///< dimension weights
      volume_type n_;           ///< tile volume
      VariableList var_;        ///< variable list
      TiledArray::detail::DimensionOrderType order_; ///< Array order

    }; // class Annotation

    /// Exchange the values of a0 and a1.
    template<typename I>
    void swap(Annotation<I>& a0, Annotation<I>& a1) {
      a0.swap(a1);
    }

  } // namespace expressions
} //namespace TiledArray


#endif // TILEDARRAY_ANNOTATION_H__INCLUDED
