#ifndef ARRAY_STORAGE_H__INCLUDED
#define ARRAY_STORAGE_H__INCLUDED

#include <block.h>
#include <madness_runtime.h>
#include <boost/array.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace TiledArray {

  // Forward declarations
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  class ArrayCoordinate;
  template <unsigned int Level>
  class LevelTag;
  template <unsigned int DIM>
  class Permutation;

  namespace detail {
    template <typename I, std::size_t DIM, typename CS>
    boost::array<I,DIM> calc_weight(const boost::array<I,DIM>&);
    template <typename I, std::size_t DIM>
    I volume(const boost::array<I,DIM>& a);
    template <typename I, unsigned int DIM, typename CS>
    bool less(const boost::array<I,DIM>&, const boost::array<I,DIM>&);
  } // namespace detail


  /// ArrayStorage is the base class for other storage classes.

  /// ArrayStorage stores array dimensions and is used to calculate ordinal
  /// values. It contains no actual array information; that is for the derived
  /// classes to implement.
  template <unsigned int DIM, typename Tag, typename CS = CoordinateSystem<DIM> >
  class ArrayStorage {
  public:
    typedef std::size_t ordinal_type;
    typedef CS coordinate_system;
    typedef ArrayCoordinate<ordinal_type, DIM, Tag, coordinate_system> index_type;
    typedef boost::array<ordinal_type,DIM> size_array;

    static const unsigned int dim() { return DIM; }

    /// Default constructor. Constructs a 0 dimension array.
    ArrayStorage() : size_(), weight_(), n_(0) { // no throw
      for(unsigned int d = 0; d < DIM; ++d) {
        size_[d] = 0;
        weight_[d] = 0;
      }
    }

    /// Constructs an array with dimensions of size.
    ArrayStorage(const size_array& size) : // no throw
        size_(size), weight_(calc_weight_(size)), n_(detail::volume(size))
    { }

    /// Copy constructor
    ArrayStorage(const ArrayStorage& other) : // no throw
        size_(other.size_), weight_(other.weight_), n_(other.n_)
    { }

    /// Destructor
    ~ArrayStorage() { } // no throw

    /// Returns the size of the array.
    size_array size() const { return size_; }

    /// Returns the number of elements in the array.
    ordinal_type volume() const { return n_; } // no throw

    /// Returns the dimension weights for the array.
    const size_array& weight() const { return weight_; } // no throw


    /// Returns true if i is less than the number of elements in the array.
    bool includes(const ordinal_type i) const { // no throw
      return i < n_;
    }

    bool includes(const index_type& i) const { // no throw
      return detail::less<ordinal_type, DIM, coordinate_system>(i.data(), size_);
    }

    /// computes an ordinal index for a given an index_type
    ordinal_type ordinal(const index_type& i) const { // no throw for non-debug
      assert(includes(i));
      ordinal_type result = dot_product(i.data(), weight_);
      return result;
    }

  protected:

    /// Helper functions that allow member template function to always work with
    /// ordinal_types
    ordinal_type ord_(const index_type& i) const { return ordinal(i); } // no throw
    ordinal_type ord_(const ordinal_type i) const { return i; } // no throw


    /// Exchange the content of a DenseArrayStorage with this.
    void swap_(ArrayStorage& other) { // no throw
      boost::swap(size_, other.size_);
      boost::swap(weight_, other.weight_);
      std::swap(n_, other.n_);
    }

    /// Class wrapper function for detail::calc_weight() function.
    static size_array calc_weight_(const size_array& sizes) { // no throw
      return detail::calc_weight<ordinal_type, static_cast<std::size_t>(DIM), coordinate_system>(sizes);
    }

    size_array size_;
    size_array weight_;
    ordinal_type n_;
  }; // class ArrayStorage


  /// DenseArrayStorage stores data for a dense N-dimensional Array. Data is
  /// stored in order in the order specified by the coordinate system template
  /// parameter. The default allocator used by array storage is std::allocator.
  /// All data is allocated and stored locally.
  template <typename T, unsigned int DIM, typename Tag = LevelTag<0>, typename CS = CoordinateSystem<DIM>, typename Allocator = std::allocator<T> >
  class DenseArrayStorage : public ArrayStorage<DIM, Tag, CS> {
  public:
    typedef ArrayStorage<DIM, Tag, CS> ArrayStorage_;
    typedef typename ArrayStorage_::index_type index_type;
    typedef typename ArrayStorage_::ordinal_type ordinal_type;
    typedef typename ArrayStorage_::size_array size_array;
    typedef T value_type;
    typedef CS coordinate_system;
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
      for(ordinal_type i = 0; i < this->n_; ++i)
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
      for(ordinal_type i = 0; i < this->n_; ++i, ++first) {
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
      for(ordinal_type i = 0; i < this->n_; ++i, ++it) {
        alloc_.construct(data_ + i, *it);
      }
    }

    /// Destructor
    ~DenseArrayStorage() {
      value_type* d = data_;
      for(ordinal_type i = 0; i < this->n_; ++i, ++d)
        alloc_.destroy(d);

      alloc_.deallocate(data_, this->n_);
    }

    DenseArrayStorage& operator =(const DenseArrayStorage& other) {
      DenseArrayStorage temp(other);
      swap(temp);

      return *this;
    }

    /// In place permutation operator.

    /// This function permutes its elements only.
    /// No assumptions are made about the data contained by this array.
    /// Therefore, if the data in each element of the array also needs to be
    /// permuted, it's up to the array owner to permute the data.
    DenseArrayStorage& operator ^=(const Permutation<DIM>& p) {
      assert(data_);
      DenseArrayStorage temp = p ^ (*this);
      swap(temp);
      return *this;
    }

    /// Resize the array. The current data common to both block is maintained.
    /// Any new elements added have uninitialized data.
    DenseArrayStorage& resize(const size_array& size) {
      typedef Block<ordinal_type, DIM, Tag, coordinate_system > block_type;
      block_type block_temp(size);
      block_type block_curr(this->size_);
      block_type block_common = block_temp & block_curr;
      DenseArrayStorage temp(size);

      for(typename block_type::const_iterator it = block_common.begin(); it != block_common.end(); ++it)
        temp[ *it ] = (*this)[ *it ];

      swap(temp);
      return *this;
    }

    // Iterator factory functions.
    iterator begin() { // no throw for non-debug
      assert(data_);
      return data_;
    }

    iterator end() { // no throw for non-debug
      assert(data_);
      return data_ + this->n_;
    }

    const_iterator begin() const { // no throw for non-debug
      assert(data_);
      return data_;
    }
    const_iterator end() const { // no throw for non-debug
      assert(data_);
      return data_ + this->n_;
    }

    /// Returns a constant iterator to the element indicated by i.

    /// If i is not included in the range of elements, the iterator will point
    /// to the end of the array. Valid types for Index are ordinal_type and
    /// index_type.
    template <typename Index>
    iterator find(const Index& i) {
      assert(data_);
      const ordinal_type i_ord = ord_(i);
      if(! includes(i_ord))
        i_ord = this->n_;

      return data_ + i_ord;
    }

    /// Returns a constant iterator to the element indicated by i.

    /// If i is not included in the range of elements, the iterator will point
    /// to the end of the array. Valid types for Index are ordinal_type and
    /// index_type.
    template <typename Index>
    const_iterator find(const Index& i) const {
      assert(data_);
      ordinal_type i_ord = ord_(i);
      if(! includes(i_ord))
        i_ord = this->n_;

      return data_ + i_ord;
    }

    /// Returns a reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    template <typename Index>
    reference_type at(const Index& i) {
      assert(data_);
      const ordinal_type i_ord = ord_(i);
      if(! includes(i_ord))
        throw std::out_of_range("template <typename Index> DenseArrayStorage::at(const Index&): Element is not in range.");

      return * (data_ + i_ord);
    }

    /// Returns a constant reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    template <typename Index>
    const_reference_type at(const Index& i) const {
      assert(data_);
      const ordinal_type i_ord = ord_(i);
      if(! includes(i_ord))
        throw std::out_of_range("template <typename Index> DenseArrayStorage::at(const Index&) const: Element is not in range.");

      return * (data_ + i_ord);
    }

    /// Returns a reference to the element at i.

    /// This No error checking is performed.
    template <typename Index>
    reference_type operator[](const Index& i) { // no throw for non-debug
#ifdef NDEBUG
      const ordinal_type i_ord = ord_(i);
      return * (data_ + i_ord);
#else
      return at(i);
#endif
    }

    /// Returns a constant reference to element i. No error checking is performed.
    template <typename Index>
    const_reference_type operator[](const Index& i) const { // no throw for non-debug
#ifdef NDEBUG
      const ordinal_type i_ord = ord_(i);
      return * (data_ + i_ord);
#else
      return at(i);
#endif
    }

    /// Exchange the content of a DenseArrayStorage with this.
    void swap(DenseArrayStorage& other) { // no throw
      ArrayStorage_::swap_(other);
      std::swap(data_, other.data_);
      std::swap(alloc_, other.alloc_);
    }

  private:

    value_type * data_;
    Allocator alloc_;
  }; // class DenseArrayStorage

  /// Stores an n-dimensional array across many nodes.

  /// DistributedArrayStorage stores array elements on one or more nodes of a
  /// cluster. Some of the data may exist on the local node. This class assumes
  /// that the T represents a type with a large amount of data and therefore
  /// will store and retrieve them individually. All communication and data transfer
  /// is handled by the madness library. Iterators will only iterate over local
  /// data. If we were to allow iteration over all data, all data would be sent
  /// to the local node.
  template <typename T, unsigned int DIM, typename Tag = LevelTag<1>, typename CS = CoordinateSystem<DIM> >
  class DistributedArrayStorage : public ArrayStorage<DIM, Tag, CS> {
  public:
    typedef ArrayStorage<DIM, Tag, CS> ArrayStorage_;
    typedef typename ArrayStorage_::index_type index_type;
    typedef typename ArrayStorage_::ordinal_type ordinal_type;
    typedef typename ArrayStorage_::size_array size_array;
    typedef CS coordinate_system;
    typedef T value_type;
    typedef ordinal_type key_type;
    typedef madness::WorldContainer<key_type,value_type> data_container;


    typedef typename data_container::iterator iterator;
    typedef typename data_container::const_iterator const_iterator;
    typedef T & reference_type;
    typedef const T & const_reference_type;

    static const unsigned int dim() { return DIM; }

    /// Construct an array with a definite size. All data elements are
    /// uninitialized. No communication occurs.
    DistributedArrayStorage(madness::World& world, const size_array& size) :
        ArrayStorage_(size), data_(world)
    { }

    /// Construct an array with a definite size and initializes the data.

    /// This constructor creates an array of size and fills in data with the
    /// list provided by the input iterators [first, last). It may be used to
    /// create a deep copy of another Distributed array. If store_local is
    /// true, only local data is stored. If store_local is false, all values
    /// will be stored. Non-local values will be sent to the owning process with
    /// non-blocking communication. store_local is true by default. You will
    /// want to set store_local to false when you are storing data where you do
    /// not know where the data will be stored.  InIter type must be an input
    /// iterator or compatible type and dereference to a
    /// std::pair<ordinal_type,value_type> or std::pair<index_type,value_type>
    /// type.
    ///
    /// Caution: If you set store_local to false, make sure you do not assign
    /// duplicated values in different processes. If you do not excess
    /// communication my occur, which will negatively affect performance. In
    /// addition, if the dupicate values are different, there is no way to
    /// predict which one will be the final value.
    template <typename InIter>
    DistributedArrayStorage(madness::World& world, const size_array& size, InIter first, InIter last) :
        ArrayStorage_(size), data_(world)
    {
      ordinal_type i_ord = 0;
      for(; first != last; ++first) {
        i_ord = ord(first->first);
        if( data_.is_local(i_ord) )
          data_.replace(i_ord, first->second);
      }

      data_.get_world().gop.fence(); // Make sure write is complete before proceeding.
    }

    /// Copy constructor. This is a shallow copy of the data with no communication.
    DistributedArrayStorage(const DistributedArrayStorage& other) :
        ArrayStorage_(other), data_(other.data_)
    { }

    ~DistributedArrayStorage() { }

    /// Create a shallow copy of the element data. No communication.
    DistributedArrayStorage& operator =(const DistributedArrayStorage& other) {
      // TODO: make this a strongly exception safe function.
      this->size_ = other.size_;
      this->weight_ = other.weight_;
      this->n_ = other.n_;
      data_ = other.data_; // shallow copy
      data_.get_world().gop.fence(); // Make sure write is complete before proceeding.
      return *this;
    }

    /// In place permutation operator.

    /// This function permutes its elements only.
    /// No assumptions are made about the data contained by this array.
    /// Therefore, if the data in each element of the array also needs to be
    /// permuted, it's up to the array owner to permute the data.
    DistributedArrayStorage& operator ^=(const Permutation<DIM>& p) {
      typedef Block<ordinal_type, DIM, Tag, coordinate_system> block_type;
      typedef typename block_type::const_iterator index_iterator;

      /// Construct temporary container.
      block_type b(this->size_);
      DistributedArrayStorage temp(data_.get_world(), p ^ (this->size_));

      // Iterate over all indices in the array. For each element data_.find() is
      // used to request data at the current index. If the data is  local, the
      // element is written into the temp array, otherwise it is skipped. When
      // the data is written, non-blocking communication may occur (when the new
      // location is not local).
      for(typename block_type::const_iterator it = b.begin(); it != b.end(); ++it) {
        typename data_container::const_accessor a;
        if( data_.find(a, ordinal( *it )))
          temp.data_.replace(temp.ordinal( p ^ *it ), a->second);
      }

      data_.get_world().gop.fence(); // it is now safe to write.
      data_ = temp.data_; // write all data, i.e do a shallow copy.

      // TODO: We may be able to eliminate this since the above assignment does
      // not communicate and should take no time
      data_.get_world().gop.fence(); // Make sure write is complete before proceeding.
      return *this;
    }

    /// Resize the array.

    /// This resize will maintain the data common to both block. Some
    /// non-blocking communication will likely occur. Any new elements added
    /// have uninitialized data.
    DistributedArrayStorage& resize(const size_array& size) {
      typedef Block<ordinal_type, DIM, Tag, coordinate_system> block_type;
      typedef typename block_type::const_iterator index_iterator;

      /// Construct temporary container.
      block_type original_blk(this->size_);
      block_type new_blk(size);
      block_type common_blk( new_blk & original_blk);
      DistributedArrayStorage temp(data_.get_world(), size);

      // Iterate over all indices in the array. For each element data_.find() is
      // used to request data at the current index. If the data is  local, the
      // element is written into the temp array, otherwise it is skipped. When
      // the data is written, non-blocking communication may occur (when the new
      // location is not local).
      for(typename block_type::const_iterator it = common_blk.begin(); it != common_blk.end(); ++it) {
        typename data_container::const_accessor a;
        if( data_.find(a, ordinal( *it )))
          temp.data_.replace(temp.ordinal( *it ), a->second);
      }

      data_.get_world().gop.fence(); // it is now safe to write.
      data_ = temp.data_; // write all data, i.e do a shallow copy.
      data_.get_world().gop.fence(); // Make sure write is complete before proceeding.
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

    /// Returns a Future iterator to an element at index i.

    /// This function will return an iterator to the element specified by index
    /// i. If the element is not local the it will use non-blocking communication
    /// to retrieve the data. The future will be immediately available if the data
    /// is local. Valid types for Index are ordinal_type or index_type.
    template <typename Index>
    madness::Future<iterator> find(const Index& i) {
      const ordinal_type i_ord = ord(i);
      return data_.find(i_ord);
    }

    /// Returns a Future const_iterator to an element at index i.

    /// This function will return a const_iterator to the element specified by
    /// index i. If the element is not local the it will use non-blocking
    /// communication to retrieve the data. The future will be immediately
    /// available if the data is local. Valid types for Index are ordinal_type
    /// or index_type.
    template <typename Index>
    madness::Future<const_iterator> find(const Index& i) const {
      const ordinal_type i_ord = ord(i);
      return data_.find(i_ord);
    }

    /// Returns a reference to local data element.

    /// This function will return a reference to local data only. It will throw
    /// std::out_of_range if i is not included in the array, and std::range_error
    /// if i is not a local element. Valid types for Index are ordinal_type or
    /// index_type.
    template <typename Index>
    reference_type at(const Index& i) {
      const ordinal_type i_ord = ord_(i);
      if(! includes(i_ord))
        throw std::out_of_range("template <typename Index> DistributedArrayStorage::at(const Index&): Element is not in range.");
      if(! data_.is_local(i_ord))
        throw std::range_error("template <typename Index> DistributedArrayStorage::at(const Index&): Element is not stored locally.");
      typename data_container::accessor t;
      data_.find(t, i_ord);
      return t->second;
    }

    /// Returns a reference to local data element.

    /// This function will return a reference to local data only. It will throw
    /// std::out_of_range if i is not included in the array, and std::range_error
    /// if i is not a local element. Valid types for Index are ordinal_type or
    /// index_type.
    template <typename Index>
    const_reference_type at(const Index i) const {
      const ordinal_type i_ord = ord(i);
      if(! includes(i_ord))
        throw std::out_of_range("template <typename Index> DistributedArrayStorage::at(const Index&) const: Element is not in range.");
      if(! data_.is_local(i_ord))
        throw std::range_error("template <typename Index> DistributedArrayStorage::at(const Index&) const: Element is not stored locally.");
      typename data_container::const_accessor t;
      bool local = data_.find(t, i_ord);
      return t->second;
    }

    /// Element access using the ordinal index without error checking
    template <typename Index>
    reference_type operator[](const Index& i) { // no throw for non-debug
#ifdef NDEBUG
      typename data_container::accessor t;
      ordinal_type i_ord = ord(i);
      data_.find(t, i_ord);
      return t->second;
#else
      return at(i);
#endif
    }

    /// Element access using the ordinal index without error checking
    template <typename Index>
    const_reference_type operator[](const Index& i) const { // no throw for non-debug
#ifdef NDEBUG
      typename data_container::accessor t;
      ordinal_type i_ord = ord(i);
      data_.find(t, i_ord);
      return t->second;
#else
      return at(i);
#endif
    }

    /// create a deep copy of distributed array and return a boost::shared_ptr
    /// to the new object.
    boost::shared_ptr<DistributedArrayStorage> clone() {
      // make a new, empty array with the same dimensions as the or
      boost::shared_ptr<DistributedArrayStorage> result =
          boost::make_shared<DistributedArrayStorage>(data_.get_world(), this->size_, begin(), end());

      return result;
    }

    template <typename Index>
    bool is_local(const Index& i) const {
      ordinal_type i_ord = ord_(i);
      return data_.is_local(i_ord);
    }

    void swap(DistributedArrayStorage& other) {
      ArrayStorage_::swap(other);
      data_container temp(data_);
      data_ = other.data_;
      other.data_ = temp;
    }

  private:

    /// No default construction. We need to initialize the data container with
    /// a world object to have a valid object.
    DistributedArrayStorage();

    data_container data_;
  }; // class DistributedArrayStorage

  template <typename T, unsigned int DIM, typename Tag, typename CS>
  DenseArrayStorage<T,DIM,Tag,CS> operator ^(const Permutation<DIM>& p, const DenseArrayStorage<T,DIM,Tag,CS>& s) {
    typedef Block<typename DenseArrayStorage<T,DIM,Tag,CS>::ordinal_type,DIM,Tag,CS> block_type;
    block_type b(s.size());
    DenseArrayStorage<T,DIM,Tag,CS> result(s.size());

    for(typename block_type::const_iterator it = b.begin(); it != b.end(); ++it) {
      result[p ^ *it] = s[ *it ];
    }

    return result;
  }

  /// DistributedArrayStorage permutation operator.

  /// This function permutes its elements only.
  /// No assumptions are made about the data contained by this array.
  /// Therefore, if the data in each element of the array also needs to be
  /// permuted, it's up to the array owner to permute the data.
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  DistributedArrayStorage<T,DIM,Tag,CS>& operator ^(const Permutation<DIM>& p, const DistributedArrayStorage<T,DIM,Tag,CS>& s) {
    typedef DistributedArrayStorage<T,DIM,Tag,CS> Store;
    typedef Block<typename Store::ordinal_type, DIM, Tag, CS> block_type;
    typedef typename block_type::const_iterator index_iterator;

    /// Construct temporary container.
    block_type b(s.size_);
    Store result(s.data_.get_world(), p ^ (s.size_));

    // Iterate over all indices in the array. For each element data_.find() is
    // used to request data at the current index. If the data is  local, the
    // element is written into the temp array, otherwise it is skipped. When
    // the data is written, non-blocking communication may occur (when the new
    // location is not local).
    for(typename block_type::const_iterator it = b.begin(); it != b.end(); ++it) {
      typename Store::data_container::const_accessor a;
      if( s.data_.find(a, s.ordinal( *it )))
        result.data_.replace(result.ordinal( p ^ *it ), a->second);
    }

    // not communicate and should take no time
    result.data_.get_world().gop.fence(); // Make sure write is complete before proceeding.
    return result;
  }

  namespace detail {
    template <typename T, std::size_t DIM, typename CS>
    boost::array<T,DIM> calc_weight(const boost::array<T,DIM>& sizes) { // no throw when T is a standard type
      boost::array<T,DIM> result;
      T weight = 1;

      for(typename CS::const_iterator d = CS::begin(); d != CS::end(); ++d) {
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
