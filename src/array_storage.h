#ifndef ARRAY_STORAGE_H__INCLUDED
#define ARRAY_STORAGE_H__INCLUDED

#include <cassert>
#include <block.h>
#include <coordinate_system.h>
#include <madness_runtime.h>
#include <boost/array.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/scoped_array.hpp>
#include <cstdlib>
#include <cstddef>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace TiledArray {

  // Forward declarations
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  class ArrayCoordinate;
  template <unsigned int Level>
  class LevelTag;

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

    /// Returns the number of elements in the array.
    std::size_t volume() const { return n_; }

    /// Returns the dimension weights for the array.
    const size_array& weight() const { return weight_; }


    /// Returns true if i is less than the number of elements in the array.
    bool includes(const std::size_t i) const {
      return i < n_;
    }

  protected:

    /// Returns true if the coordinate is in the array.
    template <typename I, typename Tag>
    bool includes(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
      for(unsigned int d = 0; d < DIM; ++d)
        if( (i[d] >= size_[d]) || (i[d] < 0) )
          return false;

      return true;
    }

    /// computes an ordinal index for a given an index_type
    template <typename I, typename Tag>
    std::size_t ordinal(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
      assert(includes(i));
      I result = dot_product(i.data(), weight_);
      return static_cast<std::size_t>(result);
    }

	/// Permute the size of the array.
    void permute(const Permutation<DIM>& p) {
      size_array size = p ^ size_;
      ArrayStorage temp(size);

      swap(temp);
    }

    /// Exchange the content of a DenseArrayStorage with this.
    void swap(ArrayStorage& other) { // no throw
      boost::swap(size_, other.size_);
      boost::swap(weight_, other.weight_);
      std::swap(n_, other.n_);
    }

    /// Class wrapper function for detail::calc_weight() function.
    static size_array calc_weight(const size_array& sizes) { // no throw
      return detail::calc_weight<std::size_t, static_cast<std::size_t>(DIM), coordinate_system>(sizes);
    }

    size_array size_;
    size_array weight_;
    std::size_t n_;
  }; // class ArrayStorage


  /// DenseArrayStorage stores data for a dense N-dimensional Array. Data is
  /// stored in order in the order specified by the coordinate system template
  /// parameter. The default allocator used by array storage is std::allocator.
  /// All data is allocated and stored locally.
  template <typename T, unsigned int DIM, typename Tag = LevelTag<0>, typename CS = CoordinateSystem<DIM>, typename Allocator = std::allocator<T> >
  class DenseArrayStorage : public ArrayStorage<DIM, CS> {
  public:
    typedef ArrayStorage<DIM, CS> ArrayStorage_;
    typedef T value_type;
    typedef CS coordinate_system;
    typedef typename ArrayStorage_::size_array size_array;
    typedef T * iterator;
    typedef const T * const_iterator;
    typedef T & reference_type;
    typedef const T & const_reference_type;

    static const unsigned int dim() { return DIM; }

    /// Default constructor. Constructs an empty array.
    DenseArrayStorage() : ArrayStorage_(), data_(NULL), alloc_() { }

    /// Constructs an array with dimensions of size and fill it with val.
    DenseArrayStorage(const size_array& size, const value_type& val = value_type()) :
        ArrayStorage_(size), data_(NULL), alloc_()
    {
      data_ = alloc_.allocate(this->n_);
      for(std::size_t i = 0; i < this->n_; ++i)
        alloc_.construct(data_ + i, val);
    }

    /// Constructs an array of size and fills it with the data indicated by
    /// the first and last input iterators. The range of data [first, last)
    /// must point to a range at least as large as the array being constructed.
    /// If the iterator range is smaller than the array, the constructor will
    /// throw an assertion error.
    template <typename InIter>
    DenseArrayStorage(const size_array& size, InIter first, InIter last) :
        ArrayStorage_(size), data_(NULL), alloc_()
    {
      data_ = alloc_.allocate(this->n_);
      for(std::size_t i = 0; i < this->n_; ++i, ++first) {
        assert(first != last);
        alloc_.construct(data_ + i, *first);
      }
    }

    /// Copy constructor
    DenseArrayStorage(const DenseArrayStorage& other) :
        ArrayStorage_(other), data_(NULL), alloc_(other.alloc_)
    {
      data_ = alloc_.allocate(this->n_);
      const_iterator it = other.begin();
      for(std::size_t i = 0; i < this->n_; ++i, ++it) {
        alloc_.construct(data_ + i, *it);
      }
    }

    /// Destructor
    ~DenseArrayStorage() {
      value_type* d = data_;
      for(std::size_t i = 0; i < this->n_; ++i, ++d)
        alloc_.destroy(d);

      alloc_.deallocate(data_, this->n_);
    }

    DenseArrayStorage& operator =(const DenseArrayStorage& other) {
      DenseArrayStorage temp(other.size_, other.begin(), other.end());
      swap(temp);

      return *this;
    }

    DenseArrayStorage& operator ^=(const Permutation<DIM>& p) {
      typedef Block<std::size_t, DIM, Tag, coordinate_system > block_type;
      block_type b(this->size_);
      DenseArrayStorage temp(*this);
      temp.permute(p);

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
      block_type block_curr(this->size_);
      block_type block_common = block_temp & block_curr;
      DenseArrayStorage temp(size);

      for(typename block_type::const_iterator it = block_common.begin(); it != block_common.end(); ++it)
        temp[ *it ] = (*this)[ *it ];

      swap(temp);
      return *this;
    }

    /// Iterator factory functions.
    iterator begin() { return data_; } // no throw
    iterator end() { return data_ + this->n_; } // no throw
    const_iterator begin() const { return data_; } // no throw
    const_iterator end() const { return data_ + this->n_; } // no throw

    /// Element access using the ordinal index with error checking
    reference_type at(const std::size_t i) {
      if( (i >= this->n_) || (i < 0) )
        throw std::out_of_range("DenseArrayStorage::at(const std::size_t&)");

      return data_[i];
    }

    /// Returns an iterator to the element indicated by i.
    template <typename I>
    iterator find(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) {
      return data_ + ordinal(i);
    }

    /// Returns a constant iterator to the element indicated by i.
    template <typename I>
    const_iterator find(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const {
      return data_ + ordinal(i);
    }

    /// Element access using the ordinal index with error checking
    const_reference_type at(const std::size_t i) const {
      if( (i >= this->n_) || (i < 0) )
        throw std::out_of_range("DenseArrayStorage::at(const std::size_t&)");

      return data_[i];
    }

    /// Element access using the element index with error checking
    template <typename I>
    reference_type at(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) {
      return at( ordinal(i) );
    }

    /// Element access using the element index with error checking
    template <typename I>
    const_reference_type at(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const {
      return at( ordinal(i) );
    }

    /// Element access using the ordinal index without error checking
    reference_type operator[](const std::size_t& i) { // no throw for non-debug
#ifdef NDEBUG
      return data_[i];
#else
      return at(i);
#endif
    }

    /// Element access using the ordinal index without error checking
    const_reference_type operator[](const std::size_t& i) const { // no throw for non-debug
#ifdef NDEBUG
      return data_[i];
#else
      return at(i);
#endif
    }

    /// Element access using the element index without error checking
    template <typename I>
    reference_type operator[](const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) { // no throw for non-debug
#ifdef NDEBUG
      return data_[ordinal(i)];
#else
      return at(i);
#endif
    }

    /// Element access using the element index ArrayCoordinate<I, DIM, Tag, CS> error checking
    template <typename I>
    const_reference_type operator[](const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
#ifdef NDEBUG
      return at(i);
#else
      return data_[ordinal(i)];
#endif
    }

    std::size_t volume() const { return ArrayStorage_::volume(); }

    /// Exchange the content of a DenseArrayStorage with this.
    void swap(DenseArrayStorage& other) { // no throw
      ArrayStorage_::swap(other);
      std::swap(data_, other.data_);
      std::swap(alloc_, other.alloc_);
    }

    /// Returns true if the coordinate is in the array.
    template <typename I>
    bool includes(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
      return ArrayStorage_::includes(i); // Use this wrapper function to preserve coordinate type safety.
    }

    /// computes an ordinal index for a given an index_type
    template <typename I>
    std::size_t ordinal(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
      return ArrayStorage_::ordinal(i); // Use this wrapper function to preserve coordinate type safety.
    }

  private:

    value_type * data_;
    Allocator alloc_;
  }; // class DenseArrayStorage

  /// DistributedArrayStorage stores array elements on one or more nodes of a
  /// cluster. Some of the data may exist on the local node. This class assumes
  /// that the T represents a type with a large amount of data and therefore
  /// will store and retrieve them individually. All communication and data transfer
  /// is handled by the madness library. Iterators will only iterate over local
  /// data. If we were to allow iteration over all data, all data would be sent
  /// to the local node.
  template <typename T, unsigned int DIM, typename Tag = LevelTag<1>, typename CS = CoordinateSystem<DIM> >
  class DistributedArrayStorage : public ArrayStorage<DIM, CS> {
  public:
    typedef ArrayStorage<DIM, CS> ArrayStorage_;
    typedef CS coordinate_system;
    typedef std::size_t ordinal_index;
    typedef typename ArrayStorage_::size_array size_array;
    typedef madness::Future<T> value_type;
    typedef ordinal_index key;
    typedef madness::WorldContainer<key,T> data_container;


    typedef typename data_container::iterator iterator;
    typedef typename data_container::const_iterator const_iterator;
    typedef madness::Future<T> reference_type;
    typedef madness::Future<T> const_reference_type;

    static const unsigned int dim() { return DIM; }

    /// Construct an array with a definite size, all data elements are uninitialized.
    DistributedArrayStorage(madness::World& world, const size_array& size) :
        ArrayStorage_(size), data_(madness::World& world)
    { }

    /// Create a shallow copy of
    DistributedArrayStorage& operator =(const DistributedArrayStorage& other) {

      return *this;
    }

    DistributedArrayStorage& operator ^=(const Permutation<DIM>& p) {

      return *this;
    }

    /// Resize the array. The current data common to both block is maintained.
    /// Any new elements added have uninitialized data.
    DistributedArrayStorage& resize(const size_array& size) {

      return *this;
    }

    /// Returns an iterator to the beginning local data.
    iterator begin() { return data_.begin(); }
    /// Returns an iterator to the end of the local data.
    iterator end() { return data_.end(); }
    /// Returns a constant iterator to the beginning of the local data.
    const_iterator begin() const { return data_.begin(); }
    /// Returns a constant iterator to the end of the local data.
    const_iterator end() const { return data_.end(); }

    template <typename I>
    madness::Future<iterator> find(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) {
      return data_.find(ordinal(i));
    }

    template <typename I>
    madness::Future<const_iterator> find(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) {
      return data_.find(ordinal(i));
    }

    /// Element access using the ordinal index with error checking
    reference_type at(const std::size_t i) {

    }

    /// Element access using the ordinal index with error checking
    const_reference_type at(const std::size_t i) const {

    }

    /// Element access using the element index with error checking
    template <typename I>
    reference_type at(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) {

    }

    /// Element access using the element index with error checking
    template <typename I>
    const_reference_type at(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const {

    }

    /// Element access using the ordinal index without error checking
    reference_type operator[](const std::size_t& i) { // no throw for non-debug
#ifdef TILED_ARRAY_DEBUG
      return at(i);
#else

#endif
    }

    /// Element access using the ordinal index without error checking
    const_reference_type operator[](const std::size_t& i) const { // no throw for non-debug
#ifdef TILED_ARRAY_DEBUG
      return at(i);
#else

#endif
    }

    /// Element access using the element index without error checking
    template <typename I>
    reference_type operator[](const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) { // no throw for non-debug
#ifdef TILED_ARRAY_DEBUG
      return at(i);
#else

#endif
    }

    /// Element access using the element index ArrayCoordinate<I, DIM, Tag, CS> error checking
    template <typename I>
    const_reference_type operator[](const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
#ifdef TILED_ARRAY_DEBUG
      return at(i);
#else

#endif
    }

    /// Returns the number of elements in the array.
    std::size_t volume() const { return ArrayStorage_::volume(); }

    /// Exchange the content of a DistributedArrayStorage with this.
    void swap(DistributedArrayStorage& other) { // no throw
      ArrayStorage_::swap(other);

    }

    /// Returns true if the coordinate is in the array.
    template <typename I>
    bool includes(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
      return ArrayStorage_::includes(i); // Use this wrapper function to preserve coordinate type safety.
    }

    /// computes an ordinal index for a given an index_type
    template <typename I>
    std::size_t ordinal(const ArrayCoordinate<I, DIM, Tag, coordinate_system>& i) const { // no throw for non-debug
      return ArrayStorage_::ordinal(i); // Use this wrapper function to preserve coordinate type safety.
    }

  private:
    /// No default construction. We need to initialize the data container with
    /// a world object to have a valid object.
    DistributedArrayStorage();

    data_container data_;
  }; // class DistributedArrayStorage

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
