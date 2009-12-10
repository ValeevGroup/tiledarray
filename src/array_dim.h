#ifndef TILEDARRAY_ARRAY_DIM_H__INCLUDED
#define TILEDARRAY_ARRAY_DIM_H__INCLUDED

#include <error.h>
#include <coordinate_system.h>
#include <coordinates.h>
#include <array_util.h>

namespace TiledArray {

  template <typename I, unsigned int DIM, typename Tag, typename CS>
  class ArrayCoordinate;

  namespace detail {

    template <typename I, std::size_t DIM>
    bool less(const boost::array<I,DIM>&, const boost::array<I,DIM>&);

    template <typename I, unsigned int DIM, typename Tag, typename CS>
    class ArrayDim;
    template <typename I, unsigned int DIM, typename Tag, typename CS>
    void swap(ArrayDim<I, DIM, Tag, CS>&, ArrayDim<I, DIM, Tag, CS>&);

    /// ArrayStorage is the base class for other storage classes.

    /// ArrayStorage stores array dimensions and is used to calculate ordinal
    /// values. It contains no actual array data; that is for the derived
    /// classes to implement. The array origin is always zero for all dimensions.
    template <typename I, unsigned int DIM, typename Tag, typename CS = CoordinateSystem<DIM> >
    class ArrayDim {
    public:
      typedef ArrayDim<I, DIM, Tag, CS> ArrayDim_;
      typedef I ordinal_type;
      typedef I volume_type;
      typedef CS coordinate_system;
      typedef ArrayCoordinate<ordinal_type, DIM, Tag, coordinate_system> index_type;
      typedef boost::array<ordinal_type,DIM> size_array;

      static unsigned int dim() { return DIM; }
      static detail::DimensionOrderType  order() { return coordinate_system::dimension_order; }

      /// Default constructor. Constructs a 0 dimension array.
      ArrayDim() : size_(), weight_(), n_(0) { // no throw
        size_.assign(0);
        weight_.assign(0);
      }

      /// Constructs an array with dimensions of size.
      ArrayDim(const size_array& size) : // no throw
          size_(size), weight_(calc_weight_(size)), n_(detail::volume(size))
      { }

      /// Copy constructor
      ArrayDim(const ArrayDim& other) : // no throw
          size_(other.size_), weight_(other.weight_), n_(other.n_)
      { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move constructor
      ArrayDim(ArrayDim&& other) : // no throw
          size_(std::move(other.size_)), weight_(std::move(other.weight_)), n_(other.n_)
      { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Destructor
      ~ArrayDim() { } // no throw

      /// Assignment operator
      ArrayDim_& operator =(const ArrayDim_& other) {
        size_ = other.size_;
        weight_ = other.weight_;
        n_ = other.n_;

        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Assignment operator
      ArrayDim_& operator =(ArrayDim_&& other) {
        size_ = std::move(other.size_);
        weight_ = std::move(other.weight_);
        n_ = other.n_;

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Returns the size of the array.
      const size_array& size() const { return size_; } // no throw

      /// Returns the number of elements in the array.
      ordinal_type volume() const { return n_; } // no throw

      /// Returns the dimension weights for the array.
      const size_array& weight() const { return weight_; } // no throw


      /// Returns true if i is less than the number of elements in the array.
      bool includes(const ordinal_type i) const { // no throw
        return i < n_;
      }

      /// Returns true if i is less than the number of elements in the array.
      bool includes(const index_type& i) const { // no throw
        return less<ordinal_type, DIM>(i.data(), size_);
      }

      /// computes an ordinal index for a given an index_type
      ordinal_type ordinal(const index_type& i) const {
        TA_ASSERT(includes(i), std::out_of_range,
            "Index is not included in the array range.");
        return ord(i);
      }

      /// Sets the size of object to the given size.
      void resize(const size_array& s) {
        size_ = s;
        weight_ = calc_weight_(s);
        n_ = detail::volume(s);
      }

      /// Helper functions that converts index_type to ordinal_type indexes.

      /// This function is overloaded so it can be called by template functions.
      /// No range checking is done. This function will not throw.
      ordinal_type ord(const index_type& i) const { // no throw
        return std::inner_product(i.begin(), i.end(), weight_.begin(), typename index_type::index(0));
      }

      ordinal_type ord(const ordinal_type i) const { return i; } // no throw


      /// Class wrapper function for detail::calc_weight() function.
      static size_array calc_weight_(const size_array& size) { // no throw
        size_array result;
        calc_weight(coordinate_system::begin(size), coordinate_system::end(size),
            coordinate_system::begin(result));
        return result;
      }

      friend void swap<>(ArrayDim_& first, ArrayDim_& second);

      size_array size_;
      size_array weight_;
      ordinal_type n_;
    }; // class ArrayDim

    template <typename I, unsigned int DIM, typename Tag, typename CS>
    void swap(ArrayDim<I, DIM, Tag, CS>& first, ArrayDim<I, DIM, Tag, CS>& second) {
      boost::swap(first.size_, second.size_);
      boost::swap(first.weight_, second.weight_);
      std::swap(first.n_, second.n_);
    }

  } // namespace detail

} // TiledArray

#endif // TILEDARRAY_ARRAY_DIM_H__INCLUDED
