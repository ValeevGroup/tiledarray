#ifndef TILEDARRAY_DISTRIBUTED_ARRAY_H__INCLUDED
#define TILEDARRAY_DISTRIBUTED_ARRAY_H__INCLUDED

#include <TiledArray/config.h>
#include <TiledArray/array_dim.h>
#include <TiledArray/key.h>
#include <TiledArray/range.h>
#include <TiledArray/madness_runtime.h>


namespace TiledArray {

  template <unsigned int Level>
  class LevelTag;
  template <unsigned int DIM>
  class Permutation;
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  class DistributedArray;
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  void swap(DistributedArray<T, DIM, Tag, CS>&, DistributedArray<T, DIM, Tag, CS>&);

  /// Stores an n-dimensional array across many nodes.

  /// DistributedArrayStorage stores array elements on one or more nodes of a
  /// cluster. Some of the data may exist on the local node. This class assumes
  /// that the T represents a type with a large amount of data and therefore
  /// will store and retrieve them individually. All communication and data transfer
  /// is handled by the madness library. Iterators will only iterate over local
  /// data. If we were to allow iteration over all data, all data would be sent
  /// to the local node.
  template <typename T, unsigned int DIM, typename Tag = LevelTag<1>, typename CS = CoordinateSystem<DIM> >
  class DistributedArray {
    BOOST_STATIC_ASSERT(DIM < TA_MAX_DIM);

  public:
    typedef DistributedArray<T, DIM, Tag, CS> DistributedArray_;
    typedef detail::ArrayDim<std::size_t, DIM, Tag, CS> array_dim_type;
    typedef typename array_dim_type::index_type index_type;
    typedef typename array_dim_type::ordinal_type ordinal_type;
    typedef typename array_dim_type::volume_type volume_type;
    typedef typename array_dim_type::size_array size_array;
    typedef CS coordinate_system;

    // Note: Since key_type is actually two keys, all elements inserted into
    // the data_container must include both key_types so the array can function
    // correctly when given an index_type, ordinal_type, or key_type.
    typedef detail::Key<ordinal_type, index_type> key_type;

  private:

    ///
    struct ArrayHash : public std::unary_function<key_type, madness::hashT> {
      ArrayHash(const array_dim_type& d) : dim_(&d) {}
      void set(const array_dim_type& d) { dim_ = &d; }
      madness::hashT operator()(const key_type& k) const {
        const typename array_dim_type::ordinal_type o =
            (k.keys() == 2 ? dim_->ord(k.key2()) : k.key1() );
        return madness::hash(o);
      }
    private:
      ArrayHash();
      const array_dim_type* dim_;
    }; // struct ArrayHash

//    typedef detail::ArrayHash<key_type, array_dim_type> hasher_type;
    typedef madness::WorldContainer<key_type, T, ArrayHash > data_container;

  public:
    typedef typename data_container::pairT value_type;

    typedef typename data_container::iterator iterator;
    typedef typename data_container::const_iterator const_iterator;
    typedef T & reference_type;
    typedef const T & const_reference_type;
    typedef typename data_container::accessor accessor;
    typedef typename data_container::const_accessor const_accessor;

    static unsigned int dim() { return DIM; }

  private:
    // Operations not permitted.
    DistributedArray();
    DistributedArray(const DistributedArray_& other);
    DistributedArray_& operator =(const DistributedArray_& other);

  public:
    /// Constructs a zero size array.
    /// Construct an array with a definite size. All data elements are
    /// uninitialized. No communication occurs.
    DistributedArray(madness::World& world) :
        dim_(), data_(world, true, ArrayHash(dim_))
    { }

    /// Construct an array with a definite size. All data elements are
    /// uninitialized. No communication occurs.
    DistributedArray(madness::World& world, const size_array& size) :
        dim_(size), data_(world, true, ArrayHash(dim_))
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
    /// std::pair<index_type,value_type> type.
    ///
    /// Caution: If you set store_local to false, make sure you do not assign
    /// duplicated values in different processes. If you do not excess
    /// communication my occur, which will negatively affect performance. In
    /// addition, if the dupicate values are different, there is no way to
    /// predict which one will be the final value.
    template <typename InIter>
    DistributedArray(madness::World& world, const size_array& size, InIter first, InIter last) :
        dim_(size), data_(world, false, ArrayHash(dim_))
    {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      for(;first != last; ++first)
        if(is_local(first->first))
          insert(first->first, first->second);

      data_.process_pending();
      data_.get_world().gop.barrier(); // Make sure everyone is done writing
                                       // before proceeding.
    }

    ~DistributedArray() { }

    /// Copy the content of this array into the other array.

    /// Performs a deep copy of this array into the other array. The content of
    /// the other array will be deleted. This function is blocking and may cause
    /// some communication.
    void clone(const DistributedArray_& other) {
      DistributedArray_ temp(data_.get_world(), other.dim_.size());
      temp.insert(other.begin(), other.end());
      data_.clear();
      swap(*this, temp);
      data_.get_world().gop.fence();
    }

    /// Inserts an element into the array

    /// Inserts a local element into the array. This will initiate non-blocking
    /// communication that will replace or insert in a remote location. Local
    /// element insertions with insert_remote() are equivilant to insert(). If
    /// the element is already present, the previous element will be destroyed.
    /// If the element is not in the range of for the array, a std::out_of_range
    /// exception will be thrown.
    template<typename Key>
    void insert(const Key& i, const_reference_type v) {
      TA_ASSERT(includes(i), std::out_of_range, "The index is not in range.");
      data_.replace(make_key_(i), v);
    }

    /// Inserts an element into the array

    /// Inserts a local element into the array. This will initiate non-blocking
    /// communication that will replace or insert in a remote location. Local
    /// element insertions with insert_remote() are equivilant to insert(). If
    /// the element is already present, the previous element will be destroyed.
    /// If the element is not in the range of for the array, a std::out_of_range
    /// exception will be thrown.
    template<typename Key>
    void insert(const std::pair<Key, T>& e) {
      insert(e.first, e.second);
    }

    template<typename InIter>
    void insert(InIter first, InIter last) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      for(;first != last; ++first)
        insert(first->first, first->second);
    }

    /// Erases the element specified by the index

    /// This function removes the element specified by the index, and performs a
    /// non-blocking communication for non-local elements.
    template<typename Key>
    void erase(const Key& i) {
      data_.erase(key_(i));
    }

    /// Erase the range of iterators

    /// The iterator range must point to a list of pairs where std::pair::first
    /// is the index to be deleted. It is intended to be used with a range of
    /// element iterators.
    template<typename InIter>
    void erase(InIter first, InIter last) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      for(; first != last; ++first)
        erase(first->first);
    }

    /// Erase all elements of the array.
    void clear() {
      data_.clear();
    }

    /// In place permutation operator.

    /// This function will permute the elements of the array. This function is a global sync point.
    DistributedArray_& operator ^=(const Permutation<DIM>& p) {
      typedef Range<ordinal_type, DIM, Tag, CS> range_type;

      /// Construct temporary container.
      range_type r(dim_.size());
      DistributedArray_ temp(data_.get_world(), p ^ (dim_.size()));

      // Iterate over all indices in the array. For each element d_.find() is
      // used to request data at the current index. If the data is  local, the
      // element is written into the temp array, otherwise it is skipped. When
      // the data is written, non-blocking communication may occur (when the new
      // location is not local).
      const_accessor a;
      key_type k;
      for(typename range_type::const_iterator it = r.begin(); it != r.end(); ++it) {
        k = dim_.ord(*it);
        if( data_.find(a, k)) {
          temp.insert(p ^ *it, a->second);
          a.release();
          data_.erase(k);
        }
      }

      // not communicate and should take no time
      swap(*this, temp);
      data_.get_world().gop.barrier();
      return *this;
    }

    /// Resize the array.

    /// This resize will maintain the data common to both arrays. Some
    /// non-blocking communication will likely occur. Any new elements added
    /// have uninitialized data. This function is a global sync point.
    DistributedArray_& resize(const size_array& size, bool keep_data = true) {
      typedef Range<ordinal_type, DIM, Tag, coordinate_system> range_type;

      /// Construct temporary container.
      DistributedArray temp(data_.get_world(), size);

      if(keep_data) {
        range_type common_rng(range_type(dim_.size_) & range_type(size));

        // Iterate over all indices in the array. For each element d_.find() is
        // used to request data at the current index. If the data is  local, the
        // element is written into the temp array, otherwise it is skipped. When
        // the data is written, non-blocking communication may occur (when the new
        // location is not local).
        typename data_container::const_accessor a;
        key_type k;
        for(typename range_type::const_iterator it = common_rng.begin(); it != common_rng.end(); ++it) {
          k = dim_.ord(*it);
          if( data_.find(a, k))
            temp.data_.replace(k, a->second);
          a.release();
        }

      }

      swap(*this, temp);

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

    /// Return the sizes of each dimension.
    const size_array& size() const { return dim_.size(); }

    /// Returns the dimension weights.

    /// The dimension weights are used to calculate ordinal values and is useful
    /// for determining array boundaries.
    const size_array& weight() const { return dim_.weight(); }

    /// Returns the number of elements in the array.
    volume_type volume(bool local = false) const { return ( local ? data_.size() : dim_.volume() ); }

    /// Returns true if the given index is included in the array.
    template<typename Key>
    bool includes(const Key& i) const { return dim_.includes(i); }

    /// Returns true if the given key is included in the array.
    bool includes(const key_type& k) const {
      TA_ASSERT((k.keys() & 3) != 0, std::runtime_error, "Key is not initialized.");

      if(k.keys() & 1)
        return dim_.includes(k.key1());

      return dim_.includes(k.key2());
    }

    /// Returns a Future iterator to an element at index i.

    /// This function will return an iterator to the element specified by index
    /// i. If the element is not local the it will use non-blocking communication
    /// to retrieve the data. The future will be immediately available if the data
    /// is local. Valid types for Index are ordinal_type or index_type.
    template<typename Key>
    madness::Future<iterator> find(const Key& i) {
      return data_.find(key_(i));
    }

    /// Returns a Future const_iterator to an element at index i.

    /// This function will return a const_iterator to the element specified by
    /// index i. If the element is not local the it will use non-blocking
    /// communication to retrieve the data. The future will be immediately
    /// available if the data is local. Valid types for Index are ordinal_type
    /// or index_type.
    template<typename Key>
    madness::Future<const_iterator> find(const Key& i) const {
      return data_.find(key_(i));
    }

    /// Sets an accessor to point to a local data element.

    /// This function will set an accessor to point to a local data element only.
    /// It will return false if the data element is remote or not found.
    template<typename Key>
    bool find(accessor& acc, const Key& i) {
      return data_.find(acc, key_(i));
    }

    /// Sets a const_accessor to point to a local data element.

    /// This function will set a const_accessor to point to a local data element
    /// only. It will return false if the data element is remote or not found.
    template<typename Key>
    bool find(const_accessor& acc, const Key& i) const {
      return data_.find(acc, key_(i));
    }

    /// Returns true if index i is stored locally.

    /// Note: This function does not check for the presence of an element at
    /// the given location. If the index is not included in the range, then the
    /// result will be erroneous.
    template<typename Key>
    bool is_local(const Key& i) const {
      return data_.is_local(key_(i));
    }

    template<typename Key>
    ProcessID owner(const Key& i) const {
      return data_.owner(key_(i));
    }

    madness::World& get_world() const {
      return data_.get_world();
    }

  private:

    /// Converts an ordinal into an index
    index_type get_index_(const ordinal_type i) const {
      index_type result;
      detail::calc_index(i, coordinate_system::rbegin(dim_.weight()),
          coordinate_system::rend(dim_.weight()),
          coordinate_system::rbegin(result));
      return result;
    }

    /// Returns the ordinal given a key
    ordinal_type ord_(const key_type& k) const {
      return k.key1();
    }

    /// Returns the ordinal given an index
    ordinal_type ord_(const index_type& i) const {
      return dim_.ord(i);
    }

    /// Returns the given ordinal
    ordinal_type ord_(const ordinal_type& i) const {
      return i;
    }

    /// Returns a key (key1_type)
    key_type key_(const ordinal_type& i) const {
      return key_type(i);
    }

    /// Returns a key (key1_type)
    key_type key_(const index_type& i) const {
      return key_type(dim_.ord(i));
    }

    /// Returns the given key
    key_type key_(const key_type& k) const {
      return k;
    }

    /// Returns a key that contains both key types, base on an index
    key_type make_key_(const index_type& i) const {
      return key_type(dim_.ord(i), i);
    }

    /// returns a key that contains both key types, pase on an ordinal
    key_type make_key_(const ordinal_type& i) const {
      return key_type(i, get_index_(i));
    }

    /// Returns the give key if b
    key_type make_key_(const key_type& k) const {
      TA_ASSERT( k.keys() & 3u , std::runtime_error, "No valid keys are assigned.");
      if(k.keys() == 3u)
        return k;

      key_type result((k.keys() & 1u ? k.key1() : ord_(k.key2())),
                      (k.keys() & 2u ? k.key2() : get_index_(k.key1())));
      return result;
    }

    friend void swap<>(DistributedArray_&, DistributedArray_&);

    array_dim_type dim_;
    data_container data_;
  }; // class DistributedArrayStorage

  /// Swap the data of the two distributed arrays.
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  void swap(DistributedArray<T, DIM, Tag, CS>& first, DistributedArray<T, DIM, Tag, CS>& second) {
    detail::swap(first.dim_, second.dim_);
    madness::swap(first.data_, second.data_);
    first.data_.get_hash().set(first.dim_);
    second.data_.get_hash().set(second.dim_);
  }

} // namespace TiledArray

#endif // TILEDARRAY_DISTRIBUTED_ARRAY_H__INCLUDED
