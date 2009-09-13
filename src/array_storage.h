#ifndef TILEDARRAY_ARRAY_STORAGE_H__INCLUDED
#define TILEDARRAY_ARRAY_STORAGE_H__INCLUDED

#include <range.h>
#include <madness_runtime.h>
#include <array_util.h>
#include <Eigen/core>
//#include <boost/array.hpp>
//#include <boost/iterator/filter_iterator.hpp>
#include <boost/scoped_array.hpp>
//#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
//#include <cstddef>
//#include <algorithm>
//#include <memory>
#include <numeric>
//#include <iterator>
//#include <stdexcept>

namespace TiledArray {

  // Forward declarations
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  class ArrayCoordinate;
  template <unsigned int Level>
  class LevelTag;
  template <unsigned int DIM>
  class Permutation;
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  class DenseArrayStorage;
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  class DistributedArrayStorage;
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  DenseArrayStorage<T,DIM,Tag,CS> operator ^(const Permutation<DIM>&, const DenseArrayStorage<T,DIM,Tag,CS>&);
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  DistributedArrayStorage<T,DIM,Tag,CS>& operator ^(const Permutation<DIM>&, const DistributedArrayStorage<T,DIM,Tag,CS>&);

  namespace detail {
    template<typename I, unsigned int DIM, typename CS>
    bool less(const boost::array<I,DIM>&, const boost::array<I,DIM>&);
  } // namespace detail

  namespace detail {

    /// ArrayStorage is the base class for other storage classes.

    /// ArrayStorage stores array dimensions and is used to calculate ordinal
    /// values. It contains no actual array data; that is for the derived
    /// classes to implement. The array origin is always zero for all dimensions.
    template <typename I, unsigned int DIM, typename Tag, typename CS = CoordinateSystem<DIM> >
    class ArrayDim {
    public:
      typedef ArrayDim<I, DIM, Tag, CS> ArrayDim_;
      typedef I ordinal_type;
      typedef CS coordinate_system;
      typedef ArrayCoordinate<ordinal_type, DIM, Tag, coordinate_system> index_type;
      typedef boost::array<ordinal_type,DIM> size_array;

      static unsigned int dim() { return DIM; }

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
        return detail::less<ordinal_type, DIM>(i.data(), size_);
      }

      /// computes an ordinal index for a given an index_type
      ordinal_type ordinal(const index_type& i) const {
        TA_ASSERT(includes(i),
            std::out_of_range("ArrayDim<...>::ordinal(...): Index is not included in the array range."));
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


      /// Exchange the content of a DenseArrayStorage with this.

      /// Swap will exchange the data of the calling object with the function
      /// argument. This function does not throw.
      void swap(ArrayDim& other) { // no throw
        boost::swap(size_, other.size_);
        boost::swap(weight_, other.weight_);
        std::swap(n_, other.n_);
      }

      /// Class wrapper function for detail::calc_weight() function.
      static size_array calc_weight_(const size_array& size) { // no throw
        size_array result;
        calc_weight(coordinate_system::begin(size), coordinate_system::end(size),
            coordinate_system::begin(result));
        return result;
      }

      size_array size_;
      size_array weight_;
      ordinal_type n_;
    }; // class ArrayDim

  } // namespace detail

  /// DenseArrayStorage stores data for a dense N-dimensional Array. Data is
  /// stored in order in the order specified by the coordinate system template
  /// parameter. The default allocator used by array storage is std::allocator.
  /// All data is allocated and stored locally. Type T must be default-
  /// Constructible and copy-constructible. You may work around the default
  /// constructor requirement by specifying default values in
  template <typename T, unsigned int DIM, typename Tag = LevelTag<0>, typename CS = CoordinateSystem<DIM> >
  class DenseArrayStorage {
  private:
    typedef Eigen::aligned_allocator<T> alloc_type;
  public:
    typedef DenseArrayStorage<T,DIM,Tag,CS> DenseArrayStorage_;
    typedef detail::ArrayDim<std::size_t, DIM, Tag, CS> array_dim_type;
    typedef typename array_dim_type::index_type index_type;
    typedef typename array_dim_type::ordinal_type ordinal_type;
    typedef typename array_dim_type::size_array size_array;
    typedef T value_type;
    typedef CS coordinate_system;
    typedef T * iterator;
    typedef const T * const_iterator;
    typedef T & reference_type;
    typedef const T & const_reference_type;

    static unsigned int dim() { return DIM; }

    /// Default constructor.

    /// Constructs an empty array. You must call
    /// DenseArrayStorage::resize(const size_array&) before the array can be
    /// used.
    DenseArrayStorage() : dim_(), d_(NULL), alloc_() { }

    /// Constructs an array with dimensions of size and fill it with val.
    DenseArrayStorage(const size_array& size, const value_type& val = value_type()) :
        dim_(size), d_(NULL), alloc_()
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
    DenseArrayStorage(const size_array& size, InIter first, InIter last) :
        dim_(size), d_(NULL), alloc_()
    {
      create_(first, last);
    }

    /// Copy constructor

    /// The copy constructor performs a deep copy of the data.
    DenseArrayStorage(const DenseArrayStorage_& other) :
        dim_(other.dim_), d_(NULL), alloc_()
    {
      create_(other.begin(), other.end());
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move constructor
    DenseArrayStorage(DenseArrayStorage_&& other) : dim_(std::move(other.dim_)),
        d_(other.d_), alloc_()
    {
      other.d_ = NULL;
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Destructor
    ~DenseArrayStorage() {
      destroy_();
    }

    DenseArrayStorage_& operator =(const DenseArrayStorage_& other) {
      DenseArrayStorage_ temp(other);
      swap(temp);

      return *this;
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    DenseArrayStorage_& operator =(DenseArrayStorage_&& other) {
      if(this != &other) {
        destroy_();
        dim_ = std::move(other.dim_);
        d_ = other.d_;
        other.d_ = NULL;
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
      if(d_ != NULL) {
        DenseArrayStorage_ temp = p ^ (*this);
        swap(temp);
      }
      return *this;
    }

    /// Resize the array. The current data common to both arrays is maintained.
    /// Any new elements added have be assigned a value of val. If val is not
    /// specified, the default constructor will be used for new elements.
    DenseArrayStorage_& resize(const size_array& size, value_type val = value_type()) {
      DenseArrayStorage_ temp(size, val);
      if(d_ != NULL) {
        typedef Range<ordinal_type, DIM, Tag, coordinate_system > range_type;
        range_type range_temp(size);
        range_type range_curr(dim_.size_);
        range_type range_common = range_temp & range_curr;

        for(typename range_type::const_iterator it = range_common.begin(); it != range_common.end(); ++it)
          temp[ *it ] = operator[]( *it ); // copy common data.
      }
      swap(temp);
      return *this;
    }

    /// Returns a raw pointer to the array elements. Elements are ordered from
    /// least significant to most significant dimension.
    value_type * data() { return d_; }

    /// Returns a constant raw pointer to the array elements. Elements are
    /// ordered from least significant to most significant dimension.
    const value_type * data() const { return d_; }

    // Iterator factory functions.
    iterator begin() { // no throw
      return d_;
    }

    iterator end() { // no throw
      return d_ + dim_.n_;
    }

    const_iterator begin() const { // no throw
      return d_;
    }

    const_iterator end() const { // no throw
      return d_ + dim_.n_;
    }

    /// Returns a reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    template <typename Index>
    reference_type at(const Index& i) {
      if(! dim_.includes(i))
        throw std::out_of_range("DenseArrayStorage<...>::at(...): Element is not in range.");

      return * (d_ + dim_.ord(i));
    }

    /// Returns a constant reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    template <typename Index>
    const_reference_type at(const Index& i) const {
      if(! dim_.includes(i))
        throw std::out_of_range("DenseArrayStorage<...>::at(...) const: Element is not in range.");

      return * (d_ + dim_.ord(i));
    }

    /// Returns a reference to the element at i.

    /// This No error checking is performed.
    template <typename Index>
    reference_type operator[](const Index& i) { // no throw for non-debug
#ifdef NDEBUG
      return * (d_ + dim_.ord(i));
#else
      return at(i);
#endif
    }

    /// Returns a constant reference to element i. No error checking is performed.
    template <typename Index>
    const_reference_type operator[](const Index& i) const { // no throw for non-debug
#ifdef NDEBUG
      return * (d_ + dim_.ord(i));
#else
      return at(i);
#endif
    }

    /// Exchange the content of a DenseArrayStorage with this.
    void swap(DenseArrayStorage_& other) { // no throw
      dim_.swap(other.dim_);
      std::swap(d_, other.d_);
    }

    /// Return the sizes of each dimension.
    const size_array& size() const { return dim_.size(); }

    /// Returns the dimension weights.

    /// The dimension weights are used to calculate ordinal values and is useful
    /// for determining array boundaries.
    const size_array& weight() const { return dim_.weight(); }

    /// Returns the number of elements in the array.
    ordinal_type volume() const { return dim_.volume(); }

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
      TA_ASSERT(d_ == NULL,
          std::runtime_error("DenseArrayStorage<...>::create_(...): Cannot allocate data to a non-NULL pointer."));
      d_ = alloc_.allocate(dim_.n_);
      for(ordinal_type i = 0; i < dim_.n_; ++i)
        alloc_.construct(d_ + i, val);
    }

    /// Allocate and initialize the array.

    /// All elements will be initialized to the values given by the iterators.
    /// If the iterator range does not contain enough elements to fill the array,
    /// the remaining elements will be initialized with the default constructor.
    template <typename InIter>
    void create_(InIter first, InIter last) {
      TA_ASSERT(d_ == NULL,
          std::runtime_error("DenseArrayStorage<...>::create_(...): Cannot allocate data to a non-NULL pointer."));
      d_ = alloc_.allocate(dim_.n_);
      ordinal_type i = 0;
      for(;first != last; ++first, ++i)
        alloc_.construct(d_ + i, *first);
      for(; i < dim_.n_; ++i)
        alloc_.construct(d_ + i, value_type());
    }

    /// Destroy the array
    void destroy_() {
      if(d_ != NULL) {
        value_type* d = d_;
        const value_type* const e = d_ + dim_.n_;
        for(; d != e; ++d)
          alloc_.destroy(d);

        alloc_.deallocate(d_, dim_.n_);
        d_ = NULL;
      }
    }

    array_dim_type dim_;
    value_type* d_;
    alloc_type alloc_;
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
  class DistributedArrayStorage : public detail::ArrayDim<std::size_t, DIM, Tag, CS> {
  public:
    typedef detail::ArrayDim<std::size_t, DIM, Tag, CS> ArrayDim_;
    typedef typename ArrayDim_::index_type index_type;
    typedef typename ArrayDim_::ordinal_type ordinal_type;
    typedef typename ArrayDim_::size_array size_array;
    typedef CS coordinate_system;
    typedef T value_type;
    typedef ordinal_type key_type;
    typedef madness::WorldContainer<key_type,value_type> data_container;


    typedef typename data_container::iterator iterator;
    typedef typename data_container::const_iterator const_iterator;
    typedef T & reference_type;
    typedef const T & const_reference_type;

    static unsigned int dim() { return DIM; }

    /// Construct an array with a definite size. All data elements are
    /// uninitialized. No communication occurs.
    DistributedArrayStorage(madness::World& world, const size_array& size) :
        ArrayDim_(size), data_(world)
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
        ArrayDim_(size), data_(world)
    {
      ordinal_type i_ord = 0;
      for(; first != last; ++first) {
        i_ord = ord(first->first);
        if( data_.is_local(i_ord) )
          data_.replace(i_ord, first->second); // no communication
      }

      data_.get_world().gop.barrier(); // Make sure everyone is done writing
                                       // before proceeding.
    }

    /// Copy constructor. This is a shallow copy of the data with no communication.
    DistributedArrayStorage(const DistributedArrayStorage& other) :
        ArrayDim_(other), data_(other.data_)
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
      typedef Range<ordinal_type, DIM, Tag, coordinate_system> range_type;
      typedef typename range_type::const_iterator index_iterator;

      /// Construct temporary container.
      range_type b(this->size_);
      DistributedArrayStorage temp(data_.get_world(), p ^ (this->size_));

      // Iterate over all indices in the array. For each element d_.find() is
      // used to request data at the current index. If the data is  local, the
      // element is written into the temp array, otherwise it is skipped. When
      // the data is written, non-blocking communication may occur (when the new
      // location is not local).
      for(typename range_type::const_iterator it = b.begin(); it != b.end(); ++it) {
        typename data_container::const_accessor a;
        if( data_.find(a, ordinal( *it ))) {
          temp.data_.replace(temp.ordinal( p ^ *it ), a->second);
                                // Will communicate if destination is not local.
          data_.erase( ordinal(*it)); // The data is in the communication queue
                                      // and we will not need it again. This
                                      // should help reduce total memory
                                      // requirements during this operation.
        }
      }

      data_.get_world().gop.fence(); // Make sure everyone is done moving data.
      data_ = temp.data_; // write all data, i.e do a shallow copy.

      data_.get_world().gop.barrier(); // Make sure write is complete before proceeding.
      return *this;
    }

    /// Resize the array.

    /// This resize will maintain the data common to both arrays. Some
    /// non-blocking communication will likely occur. Any new elements added
    /// have uninitialized data.
    DistributedArrayStorage& resize(const size_array& size) {
      typedef Range<ordinal_type, DIM, Tag, coordinate_system> range_type;
      typedef typename range_type::const_iterator index_iterator;

      /// Construct temporary container.
      range_type original_blk(this->size_);
      range_type new_blk(size);
      range_type common_blk( new_blk & original_blk);
      DistributedArrayStorage temp(data_.get_world(), size);

      // Iterate over all indices in the array. For each element d_.find() is
      // used to request data at the current index. If the data is  local, the
      // element is written into the temp array, otherwise it is skipped. When
      // the data is written, non-blocking communication may occur (when the new
      // location is not local).
      for(typename range_type::const_iterator it = common_blk.begin(); it != common_blk.end(); ++it) {
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
      const ordinal_type i_ord = ord(i);
      if(! includes(i_ord))
        throw std::out_of_range("template <typename Index> DistributedArrayStorage::at(const Index&): Element is not in range.");
      if(! data_.is_local(i_ord))
        throw std::range_error("template <typename Index> DistributedArrayStorage::at(const Index&): Element is not stored locally.");
      typename data_container::accessor t;
      if(data_.find(t, i_ord)) {
        data_.replace(i_ord,value_type());
        if(data_.find(t, i_ord))
          throw std::runtime_error("template <typename Index> DistributedArrayStorage::at(const Index&): Unable to create element.");
      }

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
      d_.find(t, i_ord);
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
      d_.find(t, i_ord);
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
      ordinal_type i_ord = ord(i);
      return data_.is_local(i_ord);
    }

    void swap(DistributedArrayStorage& other) {
      ArrayDim_::swap(other);
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
    typedef Range<typename DenseArrayStorage<T,DIM,Tag,CS>::ordinal_type,DIM,Tag,CS> range_type;
    range_type b(s.size());
    DenseArrayStorage<T,DIM,Tag,CS> result(p ^ s.size());

    for(typename range_type::const_iterator it = b.begin(); it != b.end(); ++it) {
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
    typedef Range<typename Store::ordinal_type, DIM, Tag, CS> range_type;
    typedef typename range_type::const_iterator index_iterator;

    /// Construct temporary container.
    range_type b(s.size_);
    Store result(s.d_.get_world(), p ^ (s.size_));

    // Iterate over all indices in the array. For each element d_.find() is
    // used to request data at the current index. If the data is  local, the
    // element is written into the temp array, otherwise it is skipped. When
    // the data is written, non-blocking communication may occur (when the new
    // location is not local).
    for(typename range_type::const_iterator it = b.begin(); it != b.end(); ++it) {
      typename Store::data_container::const_accessor a;
      if( s.d_.find(a, s.ordinal( *it )))
        result.data_.replace(result.ordinal( p ^ *it ), a->second);
    }

    // not communicate and should take no time
    result.data_.get_world().gop.fence(); // Make sure write is complete before proceeding.
    return result;
  }

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
        boost::scoped_array<value_type> data(new value_type[n]);
        ar & wrap(data.get(),n);
        DAS temp(size, data.get(), data.get() + n);

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
#endif // TILEDARRAY_ARRAY_STORAGE_H__INCLUDED
