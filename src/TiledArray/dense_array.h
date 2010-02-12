#ifndef TILEDARRAY_ARRAY_STORAGE_H__INCLUDED
#define TILEDARRAY_ARRAY_STORAGE_H__INCLUDED

//#include <TiledArray/error.h>
#include <TiledArray/range.h>
//#include <TiledArray/type_traits.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/array_dim.h>
#include <TiledArray/permutation.h>
#include <Eigen/Core>
//#include <boost/array.hpp>
//#include <boost/iterator/filter_iterator.hpp>
#include <boost/scoped_array.hpp>
//#include <boost/shared_ptr.hpp>
//#include <boost/make_shared.hpp>
//#include <boost/utility.hpp>
//#include <cstddef>
//#include <algorithm>
//#include <memory>
//#include <numeric>
//#include <iterator>
//#include <stdexcept>

namespace TiledArray {

  // Forward declarations
  template <unsigned int Level>
  class LevelTag;
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  class DenseArray;
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  void swap(DenseArray<T, DIM, Tag, CS>&, DenseArray<T, DIM, Tag, CS>&);
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  DenseArray<T,DIM,Tag,CS> operator ^(const Permutation<DIM>&, const DenseArray<T,DIM,Tag,CS>&);

  namespace detail {

  } // namespace detail


  /// DenseArrayStorage stores data for a dense N-dimensional Array. Data is
  /// stored in order in the order specified by the coordinate system template
  /// parameter. The default allocator used by array storage is std::allocator.
  /// All data is allocated and stored locally. Type T must be default-
  /// Constructible and copy-constructible. You may work around the default
  /// constructor requirement by specifying default values in
  template <typename T, unsigned int DIM, typename Tag = LevelTag<0>, typename CS = CoordinateSystem<DIM> >
  class DenseArray {
  private:
    typedef Eigen::aligned_allocator<T> alloc_type;
  public:
    typedef DenseArray<T,DIM,Tag,CS> DenseArrayStorage_;
    typedef detail::ArrayDim<std::size_t, DIM, Tag, CS> array_dim_type;
    typedef typename array_dim_type::index_type index_type;
    typedef typename array_dim_type::ordinal_type ordinal_type;
    typedef typename array_dim_type::volume_type volume_type;
    typedef typename array_dim_type::size_array size_array;
    typedef T value_type;
    typedef CS coordinate_system;
    typedef Tag tag_type;
    typedef T * iterator;
    typedef const T * const_iterator;
    typedef T & reference_type;
    typedef const T & const_reference_type;

    static unsigned int dim() { return DIM; }
    static detail::DimensionOrderType  order() { return coordinate_system::dimension_order; }

    /// Default constructor.

    /// Constructs an empty array. You must call
    /// DenseArrayStorage::resize(const size_array&) before the array can be
    /// used.
    DenseArray() : dim_(), data_(NULL), alloc_() { }

    /// Constructs an array with dimensions of size and fill it with val.
    DenseArray(const size_array& size, const value_type& val = value_type()) :
        dim_(size), data_(NULL), alloc_()
    {
      create_(val);
    }

    /// Construct the array with the given data.

    /// Constructs an array of size and fills it with the data indicated by
    /// the first and last input iterators. The range of data [first, last)
    /// must point to a range at least as large as the array being constructed.
    /// If the iterator range is smaller than the array, the constructor will
    /// throw an assertion error.
    template <typename InIter>
    DenseArray(const size_array& size, InIter first, InIter last) :
        dim_(size), data_(NULL), alloc_()
    {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      create_(first, last);
    }

    /// Copy constructor

    /// The copy constructor performs a deep copy of the data.
    DenseArray(const DenseArrayStorage_& other) :
        dim_(other.dim_), data_(NULL), alloc_()
    {
      create_(other.begin(), other.end());
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move constructor
    DenseArray(DenseArrayStorage_&& other) : dim_(std::move(other.dim_)),
        data_(other.data_), alloc_()
    {
      other.data_ = NULL;
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Destructor
    ~DenseArray() {
      destroy_();
    }

    DenseArrayStorage_& operator =(const DenseArrayStorage_& other) {
      DenseArrayStorage_ temp(other);
      swap(*this, temp);

      return *this;
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    DenseArrayStorage_& operator =(DenseArrayStorage_&& other) {
      if(this != &other) {
        destroy_();
        dim_ = std::move(other.dim_);
        data_ = other.data_;
        other.data_ = NULL;
      }
      return *this;
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// In place permutation operator.

    /// This function permutes its elements only.
    /// No assumptions are made about the data contained by this array.
    /// Therefore, if the data in each element of the array also needs to be
    /// permuted, it's up to the array owner to permute the data.
    DenseArrayStorage_& operator ^=(const Permutation<DIM>& p) {
      if(data_ != NULL) {
        DenseArrayStorage_ temp = p ^ (*this);
        swap(*this, temp);
      }
      return *this;
    }

    /// Resize the array. The current data common to both arrays is maintained.
    /// Any new elements added have be assigned a value of val. If val is not
    /// specified, the default constructor will be used for new elements.
    DenseArrayStorage_& resize(const size_array& size, value_type val = value_type()) {
      DenseArrayStorage_ temp(size, val);
      if(data_ != NULL) {
        // replace Range with ArrayDim?
        typedef Range<ordinal_type, DIM, Tag, coordinate_system > range_type;
        range_type range_temp(size);
        range_type range_curr(dim_.size_);
        range_type range_common = range_temp & range_curr;

        for(typename range_type::const_iterator it = range_common.begin(); it != range_common.end(); ++it)
          temp[ *it ] = operator[]( *it ); // copy common data.
      }
      swap(*this, temp);
      return *this;
    }

    /// Returns a raw pointer to the array elements. Elements are ordered from
    /// least significant to most significant dimension.
    value_type * data() { return data_; }

    /// Returns a constant raw pointer to the array elements. Elements are
    /// ordered from least significant to most significant dimension.
    const value_type * data() const { return data_; }

    // Iterator factory functions.
    iterator begin() { // no throw
      return data_;
    }

    iterator end() { // no throw
      return data_ + dim_.n_;
    }

    const_iterator begin() const { // no throw
      return data_;
    }

    const_iterator end() const { // no throw
      return data_ + dim_.n_;
    }

    /// Returns a reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    template <typename Index>
    reference_type at(const Index& i) {
      if(! dim_.includes(i))
        throw std::out_of_range("DenseArrayStorage<...>::at(...): Element is not in range.");

      return * (data_ + dim_.ord(i));
    }

    /// Returns a constant reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    template <typename Index>
    const_reference_type at(const Index& i) const {
      if(! dim_.includes(i))
        throw std::out_of_range("DenseArrayStorage<...>::at(...) const: Element is not in range.");

      return * (data_ + dim_.ord(i));
    }

    /// Returns a reference to the element at i.

    /// This No error checking is performed.
    template <typename Index>
    reference_type operator[](const Index& i) { // no throw for non-debug
#ifdef NDEBUG
      return * (data_ + dim_.ord(i));
#else
      return at(i);
#endif
    }

    /// Returns a constant reference to element i. No error checking is performed.
    template <typename Index>
    const_reference_type operator[](const Index& i) const { // no throw for non-debug
#ifdef NDEBUG
      return * (data_ + dim_.ord(i));
#else
      return at(i);
#endif
    }

    /// Return the sizes of each dimension.
    const size_array& size() const { return dim_.size(); }

    /// Returns the dimension weights.

    /// The dimension weights are used to calculate ordinal values and is useful
    /// for determining array boundaries.
    const size_array& weight() const { return dim_.weight(); }

    /// Returns the number of elements in the array.
    volume_type volume() const { return dim_.volume(); }

    /// Returns true if the given index is included in the array.
    bool includes(const index_type& i) const { return dim_.includes(i); }

    /// Returns true if the given index is included in the array.
    bool includes(const ordinal_type& i) const { return dim_.includes(i); }

    /// Returns the ordinal (linearized) index for the given index.

    /// If the given index is not included in the
    ordinal_type ordinal(const index_type& i) const { return dim_.ordinal(i); }

  private:
    /// Allocate and initialize the array.

    /// All elements will contain the given value.
    void create_(const value_type val) {
      TA_ASSERT(data_ == NULL, std::runtime_error,
          "Cannot allocate data to a non-NULL pointer.");
      data_ = alloc_.allocate(dim_.n_);
      for(ordinal_type i = 0; i < dim_.n_; ++i)
        alloc_.construct(data_ + i, val);
    }

    /// Allocate and initialize the array.

    /// All elements will be initialized to the values given by the iterators.
    /// If the iterator range does not contain enough elements to fill the array,
    /// the remaining elements will be initialized with the default constructor.
    template <typename InIter>
    void create_(InIter first, InIter last) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      TA_ASSERT(data_ == NULL, std::runtime_error,
          "Cannot allocate data to a non-NULL pointer.");
      data_ = alloc_.allocate(dim_.n_);
      ordinal_type i = 0;
      for(;first != last; ++first, ++i)
        alloc_.construct(data_ + i, *first);
      for(; i < dim_.n_; ++i)
        alloc_.construct(data_ + i, value_type());
    }

    /// Destroy the array
    void destroy_() {
      if(data_ != NULL) {
        value_type* d = data_;
        const value_type* const e = data_ + dim_.n_;
        for(; d != e; ++d)
          alloc_.destroy(d);

        alloc_.deallocate(data_, dim_.n_);
        data_ = NULL;
      }
    }

    friend void swap<>(DenseArrayStorage_& first, DenseArrayStorage_& second);

    array_dim_type dim_;
    value_type* data_;
    alloc_type alloc_;
  }; // class DenseArrayStorage


  /// Swap the data of the two arrays.
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  void swap(DenseArray<T, DIM, Tag, CS>& first, DenseArray<T, DIM, Tag, CS>& second) { // no throw
    detail::swap(first.dim_, second.dim_);
    std::swap(first.data_, second.data_);
  }

  /// Permutes the content of the n-dimensional array.
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  DenseArray<T,DIM,Tag,CS> operator ^(const Permutation<DIM>& p, const DenseArray<T,DIM,Tag,CS>& s) {
    DenseArray<T,DIM,Tag,CS> result(p ^ s.size());
    detail::Permute<DenseArray<T,DIM,Tag,CS> > f_perm(s);
    f_perm(p, result.begin(), result.end());

    return result;
  }

} // namespace TiledArray


namespace madness {
  namespace archive {

    template <class Archive, typename T, unsigned int DIM, typename Tag, typename CS>
    struct ArchiveLoadImpl<Archive, TiledArray::DenseArray<T,DIM,Tag,CS> > {
      typedef TiledArray::DenseArray<T,DIM,Tag,CS> dense_array_type;
      typedef typename dense_array_type::value_type value_type;

      static inline void load(const Archive& ar, dense_array_type& s) {
        typename dense_array_type::size_array size;
        ar & size;
        std::size_t n = TiledArray::detail::volume(size);
        boost::scoped_array<value_type> data(new value_type[n]);
        ar & wrap(data.get(),n);
        dense_array_type temp(size, data.get(), data.get() + n);

        TiledArray::swap(s, temp);
      }
    };

    template <class Archive, typename T, unsigned int DIM, typename Tag, typename CS>
    struct ArchiveStoreImpl<Archive, TiledArray::DenseArray<T,DIM,Tag,CS> > {
      typedef TiledArray::DenseArray<T,DIM,Tag,CS> DAS;
      typedef typename DAS::value_type value_type;

      static inline void store(const Archive& ar, const DAS& s) {
        ar & s.size();
        ar & wrap(s.begin(), s.volume());
      }
    };

  } // namespace archive
} // namespace madness

#endif // TILEDARRAY_ARRAY_STORAGE_H__INCLUDED
