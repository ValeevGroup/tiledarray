#ifndef ARRAY_STORAGE_H__INCLUDED
#define ARRAY_STORAGE_H__INCLUDED

#include <debug.h>
#include <block.h>
#include <boost/array.hpp>
#include <boost/scoped_array.hpp>
#include <madness_runtime.h>
#include <cstdlib>
#include <cstddef>
#include <algorithm>
#include <memory>
#include <cassert>
#include <stdexcept>

namespace TiledArray {

  // Forward declarations
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  class ArrayCoordinate;

  namespace detail {

    template <typename T, std::size_t DIM, typename CS>
    boost::array<T,DIM> calc_weight(const boost::array<T,DIM>&);

    template <typename T, std::size_t DIM>
    T volume(const boost::array<T,DIM>& a);
  } // namespace detail


  /// Array Storage is a base class for other array storage classes. ArrayStorage
  /// stores array dimensions and is used to calculate ordinal values. It contains
  /// no actual array information; that is for the derived classes to implement.
  template <unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class ArrayStorage {
  public:
    typedef CS coordinate_system;
    typedef boost::array<std::size_t,DIM> size_array;

    static const unsigned int dim() { return DIM; }

    /// Default constructor. Constructs a 0 dimension array.
    ArrayStorage() : size_(), weight_(), n_(0) {
      for(unsigned int d = 0; d < DIM; ++d) {
        size_[d] = 0;
        weight_[d] = 0;
      }
    }

    /// Constructs an array with dimensions of size.
    ArrayStorage(const size_array& size) :
        size_(size), weight_(calc_weight(size)), n_(detail::volume(size))
    { }

    /// Copy constructor
    ArrayStorage(const ArrayStorage& other) :
        size_(other.size_), weight_(other.weight_), n_(other.n_)
    {

    }

    /// Destructor
    ~ArrayStorage() { }

    ArrayStorage& operator =(const ArrayStorage& other) {
      ArrayStorage temp(other.size_);
      swap(temp);

      return *this;
    }

    ArrayStorage& operator ^=(const Permutation<DIM>& p) {
      size_array size = p ^ size_;
      ArrayStorage temp(size);

      swap(temp);
      return *this;
    }

    /// Resize the array. The current data common to both block is maintained.
    /// Any new elements added have uninitialized data.
    ArrayStorage& resize(const size_array& size) {
      ArrayStorage temp(size);
      swap(temp);
      return *this;
    }

    std::size_t volume() const { return n_; }

    /// Returns the dimension weights for the array.
    const size_array& weight() const { return weight_; }

  protected:

    /// Exchange the content of a DenseArrayStorage with this.
    void swap(ArrayStorage& other) { // no throw
      boost::swap(size_, other.size_);
      boost::swap(weight_, other.weight_);
      std::swap(n_, other.n_);
    }

    /// computes an ordinal index for a given an index_type
    template <typename I>
    std::size_t ordinal(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
      TA_ASSERT(includes(i));
      I result = dot_product(i.data(), weight_);
      return static_cast<std::size_t>(result);
    }

    template <typename I>
    bool includes(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
      for(unsigned int d = 0; d < DIM; ++d)
        if( (i[d] < 0) || (i[d] >= size_[d]) )
          return false;

      return true;
    }

    static size_array calc_weight(const size_array& sizes) { // no throw
      return detail::calc_weight<std::size_t, static_cast<std::size_t>(DIM), coordinate_system>(sizes);
    }

    size_array size_;
    size_array weight_;
    std::size_t n_;
  }; // class ArrayStorage

  /// ArrayStorage stores data for a dense N-dimensional Array.
  template <typename T, unsigned int DIM, typename Tag, typename CS = CoordinateSystem<DIM>, typename Allocator = std::allocator<T> >
  class DenseArrayStorage {
  public:
    typedef T value_type;
    typedef CS coordinate_system;
    typedef boost::array<std::size_t,DIM> size_array;
    typedef T * iterator;
    typedef const T * const_iterator;
    typedef T & reference_type;
    typedef const T & const_reference_type;

    static const unsigned int dim() { return DIM; }

    /// Default constructor. Constructs an empty array.
    DenseArrayStorage() : size_(), weight_(), n_(0), data_(NULL), alloc_() {
      for(unsigned int d = 0; d < DIM; ++d) {
        size_[d] = 0;
        weight_[d] = 0;
      }
    }

    /// Constructs an array with dimensions of size and fill it with val.
    DenseArrayStorage(const size_array& size, const value_type& val = value_type()) :
        size_(size), weight_(calc_weight(size)), n_(detail::volume(size)), data_(NULL), alloc_()
    {
      data_ = alloc_.allocate(n_);
      for(std::size_t i = 0; i < n_; ++i)
        alloc_.construct(data_ + i, val);
    }

    /// Constructs an array of size and fills it with the data indicated by
    /// the first and last input iterators. The range of data [first, last)
    /// must point to a range at least as large as the array being constructed.
    /// If the range is smaller than the array, then the constructor will
    /// throw an assertion error.
    template <typename InIter>
    DenseArrayStorage(const size_array& size, InIter first, InIter last) :
        size_(size), weight_(calc_weight(size)), n_(detail::volume(size)), data_(NULL), alloc_()
    {
      data_ = alloc_.allocate(n_);
      for(std::size_t i = 0; i < n_; ++i, ++first) {
        TA_ASSERT(first != last);
        alloc_.construct(data_ + i, *first);
      }
    }

    /// Copy constructor
    DenseArrayStorage(const DenseArrayStorage& other) :
        size_(other.size_), weight_(other.weight_), n_(other.n_), data_(NULL), alloc_(other.alloc_)
    {
      data_ = alloc_.allocate(n_);
      const_iterator it = other.begin();
      for(std::size_t i = 0; i < n_; ++i, ++it) {
        alloc_.construct(data_ + i, *it);
      }
    }

    /// Destructor
    ~DenseArrayStorage() {
      value_type* d = data_;
      for(std::size_t i = 0; i < n_; ++i, ++d)
        alloc_.destroy(d);

      alloc_.deallocate(data_, n_);
    }

    DenseArrayStorage& operator =(const DenseArrayStorage& other) {
      DenseArrayStorage temp(other.size_, other.begin(), other.end());
      swap(temp);

      return *this;
    }

    DenseArrayStorage& operator ^=(const Permutation<DIM>& p) {
      typedef Block<std::size_t, DIM, Tag, coordinate_system > block_type;
      block_type b(size_);
      DenseArrayStorage temp(*this);

      typename block_type::index_type ip;
      for(typename block_type::const_iterator it = b.begin(); it != b.end(); ++it) {
        ip = p ^ *it;
        data_[ordinal(ip)] = temp[ *it ];
      }

      swap(temp);
      return *this;
    }

    /// Resize the array. The current data common to both block is maintained.
    /// Any new elements added have uninitialized data.
    DenseArrayStorage& resize(const size_array& size) {
      typedef Block<std::size_t, DIM, Tag, coordinate_system > block_type;
      block_type block_temp(size);
      block_type block_curr(size_);
      block_type block_common = block_temp & block_curr;
      DenseArrayStorage temp(size);

      for(typename block_type::const_iterator it = block_common.begin(); it != block_common.end(); ++it)
        temp[ *it ] = (*this)[ *it ];

      swap(temp);
      return *this;
    }

    /// Iterator factory functions.
    iterator begin() { return data_; } // no throw
    iterator end() { return data_ + n_; } // no throw
    const_iterator begin() const { return data_; } // no throw
    const_iterator end() const { return data_ + n_; } // no throw

    /// Element access using the ordinal index with error checking
    reference_type at(const std::size_t i) {
      if( (i >= n_) || (i < 0) )
        throw std::out_of_range("DenseArrayStorage::at(const std::size_t&)");

      return data_[i];
    }

    /// Element access using the ordinal index with error checking
    const_reference_type at(const std::size_t i) const {
      if( (i >= n_) || (i < 0) )
        throw std::out_of_range("DenseArrayStorage::at(const std::size_t&)");

      return data_[i];
    }

    /// Element access using the element index with error checking
    template <typename I>
    reference_type at(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) {
      return at(ordinal(i));
    }

    /// Element access using the element index with error checking
    template <typename I>
    const_reference_type at(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const {
      return at(ordinal(i));
    }

    /// Element access using the ordinal index without error checking
    reference_type operator[](const std::size_t& i) { // no throw for non-debug
#ifdef TILED_ARRAY_DEBUG
      return at(i);
#else
      return data_[i];
#endif
    }

    /// Element access using the ordinal index without error checking
    const_reference_type operator[](const std::size_t& i) const { // no throw for non-debug
#ifdef TILED_ARRAY_DEBUG
      return at(i);
#else
      return data_[i];
#endif
    }

    /// Element access using the element index without error checking
    template <typename I>
    reference_type operator[](const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) { // no throw for non-debug
#ifdef TILED_ARRAY_DEBUG
      return at(i);
#else
      return data_[ordinal(i)];
#endif
    }

    /// Element access using the element index ArrayCoordinate<I, DIM, Tag, CS> error checking
    template <typename I>
    const_reference_type operator[](const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
#ifdef TILED_ARRAY_DEBUG
      return at(i);
#else
      retrun data_[ordinal(i)];
#endif
    }

    std::size_t volume() const { return n_; }

    /// Exchange the content of a DenseArrayStorage with this.
    void swap(DenseArrayStorage& other) { // no throw
      boost::swap(size_, other.size_);
      boost::swap(weight_, other.weight_);
      std::swap(data_, other.data_);
      std::swap(alloc_, other.alloc_);
    }

    /// Returns the dimension weights for the array.
    const size_array& weight() const { return weight_; }

  private:

    /// computes an ordinal index for a given an index_type
    template <typename I>
    std::size_t ordinal(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
      TA_ASSERT(includes(i));
      I result = dot_product(i.data(), weight_);
      return static_cast<std::size_t>(result);
    }

    template <typename I>
    bool includes(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
      for(unsigned int d = 0; d < DIM; ++d)
        if( (i[d] < 0) || (i[d] >= size_[d]) )
          return false;

      return true;
    }

    static size_array calc_weight(const size_array& sizes) { // no throw
      return detail::calc_weight<std::size_t, static_cast<std::size_t>(DIM), coordinate_system>(sizes);
    }

    size_array size_;
    size_array weight_;
    std::size_t n_;
    value_type * data_;
    Allocator alloc_;
  }; // class DenseArrayStorage

  template <typename T, unsigned int DIM, typename Tag, typename CS>
  DenseArrayStorage<T,DIM,Tag,CS>& operator ^(const Permutation<DIM>& p, const DenseArrayStorage<T,DIM,Tag,CS>& s) {
    typedef Block<std::size_t,DIM,Tag,CS> block_type;
    block_type b(s.size());
    DenseArrayStorage<T,DIM,Tag,CS> result(s.size());

    typename block_type::index_type ip;
    for(typename block_type::const_iterator it = b.begin(); it != b.end(); ++it) {
      ip = p ^ *it;
      result[ip] = s[ *it ];
    }

    return result;
  }

  namespace detail {
    template <typename T, std::size_t DIM, typename CS>
    boost::array<T,DIM> calc_weight(const boost::array<T,DIM>& sizes) { // no throw when T is a standard type
      typedef  detail::DimensionOrder<DIM> DimOrder;
      const DimOrder order = CS::ordering();
      boost::array<T,DIM> result;
      T weight = 1;

      for(typename DimOrder::const_iterator d = order.begin(); d != order.end(); ++d) {
        // calc ordinal weights.
        result[*d] = weight;
        weight *= sizes[*d];
      }

      return result;
    }

    /// Calculate the volume of an N-dimensional orthogonal.
    template <typename T, std::size_t DIM>
    T volume(const boost::array<T,DIM>& a) { // no throw when T is a standard type
      T result = 1;
      for(std::size_t d = 0; d < DIM; ++d)
        result *= ( a[d] < 0 ? -a[d] : a[d] );
      return result;
    }

  } // namespace detail

} // namespace TiledArray


namespace madness {
  namespace archive {

    template <class Archive, typename T, unsigned int DIM, typename Tag, typename CS>
    struct ArchiveLoadImpl<Archive, TiledArray::DenseArrayStorage<T,DIM,Tag,CS> > {
      typedef TiledArray::DenseArrayStorage<T,DIM,Tag,CS> DAS;
      typedef typename DAS::value_type value_type;

      static inline void load(const Archive& ar, DAS& s) {
        typename DAS::size_array size;
        ar & size;
        std::size_t n = TiledArray::detail::volume(size);
        boost::scoped_array<value_type> data = new value_type[n];
        ar & wrap(data,n);
        DAS temp(size, data, data + n);

        s.swap(temp);
      }
    };

    template <class Archive, typename T, unsigned int DIM, typename Tag, typename CS>
    struct ArchiveStoreImpl<Archive, TiledArray::DenseArrayStorage<T,DIM,Tag,CS> > {
      typedef TiledArray::DenseArrayStorage<T,DIM,Tag,CS> DAS;
      typedef typename DAS::value_type value_type;

      static inline void store(const Archive& ar, const DAS& s) {
        ar & s.size();
        ar & wrap(s.begin(), s.volume());
      }
    };

  }
}
#endif // ARRAY_STORAGE_H__INCLUDED
