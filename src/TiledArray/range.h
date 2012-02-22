#ifndef TILEDARRAY_RANGE_H__INCLUDED
#define TILEDARRAY_RANGE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/range_iterator.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/transform_iterator.h>
#include <algorithm>
#include <vector>
#include <iterator>
#include <functional>

namespace TiledArray {

  // Forward declaration of TiledArray components.
  class Permutation;
  template <typename>
  class Range;
  template <typename>
  class StaticRange;
  class DynamicRange;


  namespace detail {

    template <typename SizeArray>
    inline std::size_t calc_volume(const SizeArray& size) {
      return size.size() ? std::accumulate(size.begin(), size.end(), typename SizeArray::value_type(1),
          std::multiplies<typename SizeArray::value_type>()) : 0ul;
    }

    template<typename InIter, typename OutIter>
    inline void calc_weight_helper(InIter first, InIter last, OutIter result) { // no throw
      typedef typename std::iterator_traits<OutIter>::value_type value_type;

      for(value_type weight = 1; first != last; ++first, ++result) {
        *result = (*first != 0ul ? weight : 0);
        weight *= *first;
      }
    }

    template <typename WArray, typename SArray>
    inline void calc_weight(WArray& weight, const SArray& size, detail::DimensionOrderType order) {
      TA_ASSERT(weight.size() == size.size());
      if(order == detail::increasing_dimension_order)
        calc_weight_helper(size.begin(), size.end(), weight.begin());
      else
        calc_weight_helper(size.rbegin(), size.rend(), weight.rbegin());
    }

    template <typename Index, typename WeightArray, typename StartArray>
    inline typename Index::value_type calc_ordinal(const Index& index, const WeightArray& weight, const StartArray& start) {
      typename Index::value_type o = 0;
      const typename Index::value_type dim = index.size();
      for(std::size_t i = 0ul; i < dim; ++i)
        o += (index[i] - start[i]) * weight[i];

      return o;
    }


    template <typename Index, typename Ordinal, typename RangeType>
    inline void calc_index(Index& index, Ordinal o, const RangeType& range) {
      const std::size_t dim = index.size();

      if(range.order() == detail::increasing_dimension_order) {
        for(std::size_t i = 0; i < dim; ++i) {
          const std::size_t ii = dim - i;
          const typename Index::value_type x = (o / range.weight()[ii]);
          o -= x * range.weight()[ii];
          index[ii] = x + range.start()[ii];
        }
      } else {
        for(std::size_t i = 0ul; i < dim; ++i) {
          const typename Index::value_type x = (o / range.weight()[i]);
          o -= x * range.weight()[i];
          index[i] = x + range.start()[i];
        }
      }
    }

    template <typename ForIter, typename InIterStart, typename InIterFinish>
    inline void increment_coordinate_helper(ForIter first_cur, ForIter last_cur, InIterStart start, InIterFinish finish) {
      for(; first_cur != last_cur; ++first_cur, ++start, ++finish) {
        // increment coordinate
        ++(*first_cur);

        // break if done
        if( *first_cur < *finish)
          return;

        // Reset current index to start value.
        *first_cur = *start;
      }
    }

    template <typename Index, typename RangeType>
    inline void increment_coordinate(Index& index, const RangeType& range) {
      if(range.order() == detail::increasing_dimension_order)
        increment_coordinate_helper(index.begin(), index.end(), range.start().begin(), range.finish().begin());
      else
        increment_coordinate_helper(index.rbegin(), index.rend(), range.start().rbegin(), range.finish().rbegin());


      // if the current location was set to start then it was at the end and
      // needs to be reset to equal finish.
      if(std::equal(index.begin(), index.end(), range.start().begin()))
        std::copy(range.finish().begin(), range.finish().end(), index.begin());
    }


    template <typename>
    struct RangeTraits;


    template <typename CS>
    struct RangeTraits<StaticRange<CS> > {
      typedef CS coordinate_system;
      typedef typename CS::index index;
      typedef typename CS::size_array size_array;
    };

    template <>
    struct RangeTraits<DynamicRange> {
      typedef void coordinate_system;
      typedef std::vector<std::size_t> index;
      typedef std::vector<std::size_t> size_array;
    };

  }  // namespace detail


  /// Range data of an N-dimensional tensor.

  /// \tparam Derived The type of the class that uses this class as a base class
  /// (i.e. curiously recurring template pattern).
  /// Range is an interface class for range objects. It handles operations that
  /// are common to both fixed size and non-fixed dimension ranges.
  /// \note This object should not be used directly. Instead use \c StaticRange
  /// for fixed dimension or \c DynamicRange for non-fixed dimension tiled
  /// ranges.
  template <typename Derived>
  class Range {
  public:
    typedef Range<Derived> Range_;

    typedef std::size_t size_type;
    typedef typename TiledArray::detail::RangeTraits<Derived>::coordinate_system coordinate_system;
    typedef typename detail::RangeTraits<Derived>::index index;
    typedef typename detail::RangeTraits<Derived>::size_array size_array;

    typedef detail::RangeIterator<index, Range_> const_iterator;
    friend class detail::RangeIterator<index, Range_>;

  private:

    // used to access the derived class's data
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

  public:

    // Compiler generated destructor is OK

    Range_& operator=(const Range_& other) {
      derived() = other.derived();
      return *this;
    }

    template <typename D>
    Range_& operator=(const Range<D>& other) {
      derived() = other.derived();
      return *this;
    }

    /// Range dimension accessor

    /// \return The number of dimensions in the range.
    unsigned int dim() const { return derived().dim(); }

    /// Data ordering accessor

    /// \return The ordering enum for the tensor data
    detail::DimensionOrderType order() const { return derived().order(); }

    /// Start coordinate accessor

    /// \return The lower bound of the range
    const index& start() const { return derived().start(); } // no throw

    /// Finish coordinate accessor

    /// \return The upper bound of the range
    const index& finish() const { return derived().finish(); } // no throw

    /// Size array accessor

    /// \return An array with the size of each dimension.
    const size_array& size() const { return derived().size(); } // no throw

    /// Weight array accessor

    /// \return An array with the step size of each dimension
    const size_array& weight() const { return derived().weight(); } // no throw

    /// Range volume accessor

    /// \return The total number of elements in the range.
    size_type volume() const { return detail::calc_volume(size()); }

    /// Index iterator factory

    /// The iterator dereferences to an index. The order of iteration matches
    /// the data layout of a dense tensor.
    /// \return An iterator that holds the start element index of a tensor.
    const_iterator begin() const { return const_iterator(start(), this); }

    /// Index iterator factory

    /// The iterator dereferences to an index. The order of iteration matches
    /// the data layout of a dense tensor.
    /// \return An iterator that holds the finish element index of a tensor.
    const_iterator end() const { return const_iterator(finish(), this); }

    /// Check the coordinate to make sure it is within the range.

    /// \tparam Index The coordinate index array type
    /// \param i The coordinate index to check for inclusion in the range
    /// \return \c true when \c i \c >= \c start and \c i \c < \c f, otherwise
    /// \c false
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, bool>::type
    includes(const Index& index) const {
      TA_ASSERT(index.size() == dim());
      for(std::size_t i = 0ul; i < dim(); ++i)
        if((index[i] < start()[i]) || (index[i] >= finish()[i]))
          return false;

      return true;
    }

    /// Check the ordinal index to make sure it is within the range.

    /// \param i The ordinal index to check for inclusion in the range
    /// \return \c true when \c i \c >= \c 0 and \c i \c < \c volume
    template <typename Ordinal>
    typename madness::enable_if<std::is_integral<Ordinal>, bool>::type
    includes(Ordinal i) const {
      return include_ordinal_(i);
    }

    /// Permute the tile given a permutation.
    Range_& operator ^=(const Permutation& p) {
      TA_ASSERT(p.dim() == dim());
      Derived temp(p ^ start(), p ^ finish(), order());
      derived().swap(temp);

      return *this;
    }

    /// Change the dimensions of the range.
    template <typename Index>
    Range_& resize(const Index& start, const Index& finish) {
      Derived temp(start, finish, order());
      derived().swap(temp);

      return *this;
    }

    /// calculate the ordinal index of /c i

    /// This function is just a pass-through so the user can call \c ord() on
    /// a template parameter that can be an index or a size_type.
    /// \param i Ordinal index
    /// \return \c i (unchanged)
    template <typename Ordinal>
    typename madness::enable_if<std::is_integral<Ordinal>, Ordinal>::type
    ord(Ordinal i) const {
      TA_ASSERT(includes(i));
      return i;
    }

    /// calculate the ordinal index of /c i

    /// Convert an index to an ordinal index.
    /// \param i The index to be converted to an ordinal index
    /// \return The ordinal index of the index \c i
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, size_type>::type
    ord(const Index& index) const {
      TA_ASSERT(index.size() == dim());
      TA_ASSERT(includes(index));
      size_type o = 0;
      for(std::size_t i = 0ul; i < dim(); ++i)
        o += (index[i] - start()[i]) * weight()[i];

      return o;
    }

    /// calculate the index of /c i

    /// Convert an ordinal index to an index.
    /// \param i Ordinal index
    /// \return The index of the ordinal index
    template <typename Ordinal>
    typename madness::enable_if<std::is_integral<Ordinal>, index>::type
    idx(Ordinal o) const {
      TA_ASSERT(includes(o));
      index i;
      detail::calc_index(size_index_(i), o, *this);
      return i;
    }

    /// calculate the index of /c i

    /// This function is just a pass-through so the user can call \c idx() on
    /// a template parameter that can be an index or a size_type.
    /// \param i The index
    /// \return \c i (unchanged)
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, const Index&>::type
    idx(const Index& i) const {
      TA_ASSERT(includes(i));
      return i;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
      derived().serialize(ar);
    }

  private:

    template <typename T>
    std::vector<T>& size_index_(std::vector<T>& i) const {
      if(i.size() != dim())
        i.resize(dim());

      return i;
    }

    template <typename T>
    T& size_index_(T& i) const {
      TA_ASSERT(i.size() == dim());
      return i;
    }

    template <typename Ordinal>
    typename boost::enable_if<std::is_signed<Ordinal>, bool>::type
    include_ordinal_(Ordinal i) const {
      return (i >= 0ul) && (i < volume());
    }

    template <typename Ordinal>
    typename boost::disable_if<std::is_signed<Ordinal>, bool>::type
    include_ordinal_(Ordinal i) const {
      return i < volume();
    }

    void increment(index& i) const {
      detail::increment_coordinate(i, *this);
    }

    void advance(index& i, std::ptrdiff_t n) const {
      const size_type o = ord(i) + n;

      if(n >= volume())
        std::copy(finish().begin(), finish().end(), i.begin());
      else
        i = idx(o);
    }

    std::ptrdiff_t distance_to(const index& first, const index& last) const {
      return ord(last) - ord(first);
    }
  }; // class Range



  /// Range stores dimension information for a block of tiles or elements.

  /// Range is used to obtain and/or store start, finish, size, and volume
  /// information. It also provides index iteration over its range.
  template <typename CS>
  class StaticRange : public Range<StaticRange<CS> > {
  public:
    typedef StaticRange<CS> StaticRange_;
    typedef Range<StaticRange_> base;
    typedef typename base::size_type size_type;
    typedef typename base::coordinate_system coordinate_system;
    typedef typename base::index index;
    typedef typename base::size_array size_array;

    /// Default constructor. The range has 0 size and the origin is set at 0.
    StaticRange() :
        start_(), finish_(), size_(), weight_()
    {}

    /// Constructor defined by an upper and lower bound. All elements of
    /// finish must be greater than or equal to those of start.
    template <typename Index>
    StaticRange(const Index& start, const Index& finish, detail::DimensionOrderType o = coordinate_system::order) :
        start_(), finish_(), size_(), weight_()
    {
      TA_ASSERT(order() == o);
      TA_ASSERT(start.size() == finish.size());
      TA_ASSERT( (std::equal(start.begin(), start.end(), finish.begin(),
          std::less_equal<typename coordinate_system::ordinal_index>())) );

      for(std::size_t i = 0; i < dim(); ++i) {
        start_[i] = start[i];
        finish_[i] = finish[i];
        size_[i] = finish[i] - start[i];
      }
      detail::calc_weight(weight_, size_, order());
    }

    /// Constructor defined by an upper and lower bound. All elements of
    /// finish must be greater than or equal to those of start.
    template <typename SizeArray>
    StaticRange(const SizeArray& size, detail::DimensionOrderType o) :
        start_(), finish_(), size_(), weight_()
    {
      TA_ASSERT(order() == o);

      for(std::size_t i = 0; i < dim(); ++i) {
        start_[i] = 0;
        finish_[i] = size[i];
        size_[i] = size[i];
      }
      detail::calc_weight(weight_, size_, order());
    }

    /// Copy Constructor
    StaticRange(const StaticRange_& other) : // no throw
        start_(other.start_), finish_(other.finish_), size_(other.size_),
        weight_(other.weight_)
    { }

    template <typename Derived>
    StaticRange(const Range<Derived>& other) : // no throw
      start_(), finish_(), size_(), weight_()
    {
      TA_ASSERT(order() == other.order());
      TA_ASSERT(dim() == other.dim());

      std::copy(other.start().begin(), other.start().end(), start_.begin());
      std::copy(other.finish().begin(), other.finish().end(), finish_.begin());
      std::copy(other.size().begin(), other.size().end(), size_.begin());
      std::copy(other.weight().begin(), other.weight().end(), weight_.begin());
    }

    ~StaticRange() { }

    StaticRange_& operator=(const StaticRange_& other) {
      start_ = other.start_;
      finish_ = other.finish_;
      size_ = other.size_;
      weight_ = other.weight_;

      return *this;
    }

    template <typename D>
    StaticRange_& operator=(const Range<D>& other) {
      TA_ASSERT(dim() == other.dim());
      std::copy(other.start().begin(), other.start().end(), start_.begin());
      std::copy(other.finish().begin(), other.finish().end(), finish_.begin());
      std::copy(other.size().begin(), other.size().end(), size_.begin());
      std::copy(other.weight().begin(), other.weight().end(), weight_.begin());

      return *this;
    }

    static unsigned int dim() { return coordinate_system::dim; }

    static detail::DimensionOrderType order() { return coordinate_system::order; }

    /// Returns the lower bound of the range
    const index& start() const { return start_; } // no throw

    /// Returns the upper bound of the range
    const index& finish() const { return finish_; } // no throw

    /// Returns an array with the size of each dimension.
    const size_array& size() const { return size_.data(); } // no throw

    const size_array& weight() const { return weight_.data(); } // no throw

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & start_ & finish_ & size_ & weight_;
    }

    void swap(StaticRange_& other) {
      TiledArray::swap(start_, other.start_);
      TiledArray::swap(finish_, other.finish_);
      TiledArray::swap(size_, other.size_);
      TiledArray::swap(weight_, other.weight_);
    }

  private:

    index start_;    ///< Tile origin
    index finish_;   ///< Tile upper bound
    index size_;     ///< Dimension sizes
    index weight_;   ///< Dimension weights
  }; // class Range

  class DynamicRange : public Range<DynamicRange> {
  public:
    typedef DynamicRange DynamicRange_;
    typedef Range<DynamicRange_> base;
    typedef base::size_type size_type;
    typedef base::coordinate_system coordinate_system;
    typedef base::index index;
    typedef base::size_array size_array;

    /// Default constructor. The range has 0 size and the origin is set at 0.
    DynamicRange() :
        start_(), finish_(), size_(), weight_(), order_(detail::decreasing_dimension_order)
    {}

    /// Constructor defined by an upper and lower bound. All elements of
    /// finish must be greater than or equal to those of start.
    template <typename Index>
    DynamicRange(const Index& start, const Index& finish, detail::DimensionOrderType order) :
        start_(start.begin(), start.end()),
        finish_(finish.begin(), finish.end()),
        size_(detail::make_tran_it(finish.begin(), start.begin(), std::minus<std::size_t>()),
            detail::make_tran_it(finish.end(), start.end(), std::minus<std::size_t>())),
        weight_(start.size()),
        order_(order)
    {
      TA_ASSERT(start.size() == finish.size());
      TA_ASSERT( (std::equal(start_.begin(), start_.end(), finish_.begin(),
          std::less_equal<std::size_t>())) );

      detail::calc_weight(weight_, size_, order_);
    }

    template <typename SizeArray>
    DynamicRange(const SizeArray& size, detail::DimensionOrderType order) :
        start_(size.size(), 0ul),
        finish_(size.begin(), size.end()),
        size_(size.begin(), size.end()),
        weight_(size.size()),
        order_(order)
    {
      detail::calc_weight(weight_, size_, order_);
    }

    /// Copy Constructor
    DynamicRange(const DynamicRange_& other) : // no throw
        start_(other.start_), finish_(other.finish_), size_(other.size_),
        weight_(other.weight_), order_(other.order_)
    {}

    template <typename Derived>
    DynamicRange(const Range<Derived>& other) : // no throw
        start_(other.start().begin(), other.start().end()),
        finish_(other.finish().begin(), other.finish().end()),
        size_(other.size().begin(), other.size().end()),
        weight_(other.weight().begin(), other.weight().end()),
        order_(other.order())
    {}

    ~DynamicRange() {}

    DynamicRange_& operator=(const DynamicRange_& other) {
      start_ = other.start_;
      finish_ = other.finish_;
      size_ = other.size_;
      weight_ = other.weight_;
      order_ = other.order_;

      return *this;
    }

    template <typename D>
    DynamicRange_& operator=(const Range<D>& other) {
      if(dim() != other.dim()) {
        start_.resize(other.dim());
        finish_.resize(other.dim());
        size_.resize(other.dim());
        weight_.resize(other.dim());
      }
      std::copy(other.start().begin(), other.start().end(), start_.begin());
      std::copy(other.finish().begin(), other.finish().end(), finish_.begin());
      std::copy(other.size().begin(), other.size().end(), size_.begin());
      std::copy(other.weight().begin(), other.weight().end(), weight_.begin());
      order_ = other.order();

      return *this;
    }

    unsigned int dim() const { return size_.size(); }

    detail::DimensionOrderType order() const { return order_; }

    /// Returns the lower bound of the range
    const index& start() const { return start_; } // no throw

    /// Returns the upper bound of the range
    const index& finish() const { return finish_; } // no throw

    /// Returns an array with the size of each dimension.
    const size_array& size() const { return size_; } // no throw

    const size_array& weight() const { return weight_; } // no throw

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & start_ & finish_ & size_ & weight_ & order_;
    }

    void swap(DynamicRange_& other) {
      std::swap(start_, other.start_);
      std::swap(finish_, other.finish_);
      std::swap(size_, other.size_);
      std::swap(weight_, other.weight_);
      std::swap(order_, other.order_);
    }

  private:

    index start_;    ///< Tile origin
    index finish_;   ///< Tile upper bound
    index size_;     ///< Dimension sizes
    index weight_;   ///< Dimension weights
    detail::DimensionOrderType order_;
  }; // class Range



  /// Exchange the values of the give two ranges.
  template <typename CS>
  void swap(Range<CS>& r0, Range<CS>& r1) { // no throw
    r0.swap(r1);
  }

  /// Return the union of two range (i.e. the overlap). If the ranges do not
  /// overlap, then a 0 size range will be returned.
//  template <typename Derived1, typename Derived2>
//  Range<CS> operator &(const Range<CS>& b1, const Range<CS>& b2) {
//    Range<CS> result;
//    typename Range<CS>::index start, finish;
//    typename Range<CS>::index::value_type s1, s2, f1, f2;
//    for(unsigned int d = 0; d < CS::dim; ++d) {
//      s1 = b1.start()[d];
//      f1 = b1.finish()[d];
//      s2 = b2.start()[d];
//      f2 = b2.finish()[d];
//      // check for overlap
//      if( (s2 < f1 && s2 >= s1) || (f2 < f1 && f2 >= s1) ||
//          (s1 < f2 && s1 >= s2) || (f1 < f2 && f1 >= s2) )
//      {
//        start[d] = std::max(s1, s2);
//        finish[d] = std::min(f1, f2);
//      } else {
//        return result; // no overlap for this index
//      }
//    }
//    result.resize(start, finish);
//    return result;
//  }

  /// Returns a permuted range.
  template <typename Derived>
  Derived operator ^(const Permutation& perm, const Range<Derived>& r) {
    TA_ASSERT(perm.dim() == r.dim());
    return Derived(perm ^ r.start(), perm ^ r.finish(), r.order());
  }

  template <typename Derived>
  const Range<Derived>& operator ^(const detail::NoPermutation& perm, const Range<Derived>& r) {
    return r;
  }

  /// Returns true if the start and finish are equal.
  template <typename Derived1, typename Derived2>
  bool operator ==(const Range<Derived1>& r1, const Range<Derived2>& r2) {
#ifdef NDEBUG
    return (r1.dim() == r2.dim()) &&  (r1.order() == r2.order()) &&
        ( std::equal(r1.start().begin(), r1.start().end(), r2.start().begin()) ) &&
        ( std::equal(r1.finish().begin(), r1.finish().end(), r2.finish().begin()) );
#else
    return (r1.dim() == r2.dim()) && (r1.order() == r2.order()) &&
        ( std::equal(r1.start().begin(), r1.start().end(), r2.start().begin()) ) &&
        ( std::equal(r1.finish().begin(), r1.finish().end(), r2.finish().begin()) ) &&
        ( std::equal(r1.size().begin(), r1.size().end(), r2.size().begin()) ) &&
        ( std::equal(r1.weight().begin(), r1.weight().end(), r2.weight().begin()) ); // do an extra size check to catch bugs.
#endif
  }

  /// Returns true if the start and finish are not equal.
  template <typename Derived1, typename Derived2>
  bool operator !=(const Range<Derived1>& r1, const Range<Derived2>& r2) {
    return ! operator ==(r1, r2);
  }

  /// range output operator.
  template<typename CS>
  std::ostream& operator<<(std::ostream& out, const Range<CS>& r) {
    out << "[ ";
    detail::print_array(out, r.start());
    out << ", ";
    detail::print_array(out, r.finish());
    out << " )";
    return out;
  }

} // namespace TiledArray
#endif // TILEDARRAY_RANGE_H__INCLUDED
