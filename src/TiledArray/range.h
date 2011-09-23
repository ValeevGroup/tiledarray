#ifndef TILEDARRAY_RANGE_H__INCLUDED
#define TILEDARRAY_RANGE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/coordinates.h>
#include <TiledArray/range_iterator.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/transform_iterator.h>
#include <boost/utility/enable_if.hpp>
#include <algorithm>

namespace TiledArray {

  // Forward declaration of TiledArray components.
  template <unsigned int>
  class Permutation;
  template <typename>
  class Range;
  template <typename>
  class StaticRange;
  class DynamicRange;
//  template <typename I, unsigned int DIM, typename Tag>
//  void swap(ArrayCoordinate<I,DIM,Tag>& c1, ArrayCoordinate<I,DIM,Tag>& c2);
//  template <typename CS>
//  void swap(Range<CS>&, Range<CS>&);
//  template <typename CS>
//  Range<CS> operator &(const Range<CS>&, const Range<CS>&);
//  template <unsigned int DIM, typename CS>
//  Range<CS> operator ^(const Permutation<DIM>&, const Range<CS>&);
//  template <typename CS>
//  bool operator ==(const Range<CS>&, const Range<CS>&);
//  template <typename CS>
//  bool operator !=(const Range<CS>&, const Range<CS>&);
//  template <typename CS>
//  Range<CS> operator ^(const Permutation<1>&, const Range<CS>&);
//  template <typename CS>
//  std::ostream& operator<<(std::ostream&, const Range<CS>&);




  namespace detail {

    template <typename SizeArray>
    inline std::size_t calc_volume(const SizeArray& size) {
      return std::accumulate(size.begin(), size.end(), typename SizeArray::value_type(1),
          std::multiplies<typename SizeArray::value_type>());
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


    template <typename Index, typename Ordinal, typename SizeArray>
    inline void calc_index(Index& index, Ordinal o, const SizeArray& weight, const SizeArray& start, detail::DimensionOrderType order) {
      const std::size_t dim = index.size();

      if(order == detail::increasing_dimension_order) {
        for(std::size_t i = 0; i < dim; ++i) {
          const std::size_t ii = dim - i;
          const typename Index::value_type x = (o / weight[ii]);
          o -= x * weight[ii];
          index[ii] = x + start[ii];
        }
      } else {
        for(std::size_t i = 0ul; i < dim; ++i) {
          const typename Index::value_type x = (o / weight[i]);
          o -= x * weight[i];
          index[i] = x + start[i];
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
      typedef typename CS::volume_type volume_type;
      typedef typename CS::index index;
      typedef typename CS::ordinal_index ordinal_index;
      typedef typename CS::size_array size_array;
    };

    template <>
    struct RangeTraits<DynamicRange> {
      typedef void coordinate_system;
      typedef std::size_t volume_type;
      typedef std::vector<std::size_t> index;
      typedef std::size_t ordinal_index;
      typedef std::vector<std::size_t> size_array;
    };

  }  // namespace detail



  template <typename Derived>
  class Range {
  public:
    typedef Range<Derived> Range_;

    typedef typename TiledArray::detail::RangeTraits<Derived>::coordinate_system coordinate_system;
    typedef typename detail::RangeTraits<Derived>::volume_type volume_type;
    typedef typename detail::RangeTraits<Derived>::index index;
    typedef typename detail::RangeTraits<Derived>::ordinal_index ordinal_index;
    typedef typename detail::RangeTraits<Derived>::size_array size_array;

    typedef detail::RangeIterator<index, Range_> const_iterator;
    friend class detail::RangeIterator<index, Range_>;

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    // Compiler generated constructor and destructor are OK

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
    detail::DimensionOrderType order() const { return derived().order(); }

    /// Returns the lower bound of the range
    const index& start() const { return derived().start(); } // no throw

    /// Returns the upper bound of the range
    const index& finish() const { return derived().finish(); } // no throw

    /// Returns an array with the size of each dimension.
    const size_array& size() const { return derived().size(); } // no throw

    const size_array& weight() const { return derived().weight(); } // no throw

    // iterator factory functions
    const_iterator begin() const { return const_iterator(start(), this); }
    const_iterator end() const { return const_iterator(finish(), this); }


    /// Returns the number of elements in the range.
    volume_type volume() const {
      return detail::calc_volume(size());
    }

    /// Check the coordinate to make sure it is within the range.

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
    template <unsigned int DIM>
    Range_& operator ^=(const Permutation<DIM>& p) {
      TA_ASSERT(DIM == dim());
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
    /// a template parameter that can be an index or an ordinal_index.
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
    typename madness::disable_if<std::is_integral<Index>, ordinal_index>::type
    ord(const Index& index) const {
      TA_ASSERT(index.size() == dim());
      TA_ASSERT(includes(index));
      ordinal_index o = 0;
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
      detail::calc_index(size_index_(i), o, weight(), start());
      return i;
    }

    /// calculate the index of /c i

    /// This function is just a pass-through so the user can call \c idx() on
    /// a template parameter that can be an index or an ordinal_index.
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

    void swap(Range_& other) {
      derived().swap(other.derived());
    }

  private:

    template <typename T>
    std::vector<T>& size_index_(std::vector<T>& i) {
      if(i.size() != dim())
        i.resize(dim());

      return i;
    }

    template <typename T, std::size_t N>
    std::array<T,N>& size_index_(std::array<T,N>& i) {
      TA_ASSERT(dim() == N);
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
      const ordinal_index o = ord(i) + n;

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
    typedef typename base::coordinate_system coordinate_system;
    typedef typename base::volume_type volume_type;
    typedef typename base::index index;
    typedef typename base::ordinal_index ordinal_index;
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

  /// Range stores dimension information for a block of tiles or elements.

  /// Range is used to obtain and/or store start, finish, size, and volume
  /// information. It also provides index iteration over its range.
  template <unsigned int L, detail::DimensionOrderType O, typename I>
  class Range<CoordinateSystem<1u, L, O, I> > {
  public:
    typedef Range<CoordinateSystem<1u, L, O, I> > Range_;
    typedef CoordinateSystem<1u, L, O, I> coordinate_system;

    typedef typename coordinate_system::volume_type volume_type;
    typedef typename coordinate_system::ordinal_index index;
    typedef typename coordinate_system::ordinal_index ordinal_index;
    typedef typename coordinate_system::ordinal_index size_array;

    typedef detail::RangeIterator<index, Range_> const_iterator;
    friend class detail::RangeIterator< index , Range_ >;

    /// Default constructor. The range has 0 size and the origin is set at 0.
    Range() :
        start_(), finish_(), size_()
    {}

    /// Constructor defined by an upper and lower bound. All elements of
    /// finish must be greater than or equal to those of start.
    Range(const index start, const index finish) :
        start_(start), finish_(finish), size_(finish - start)
    {
      TA_ASSERT( start <= finish );
    }

    /// Copy Constructor
    Range(const Range_& other) : // no throw
        start_(other.start_), finish_(other.finish_), size_(other.size_)
    {}

    ~Range() {}

    // iterator factory functions
    const_iterator begin() const { return const_iterator(start_, this); }
    const_iterator end() const { return const_iterator(finish_, this); }

    /// Returns the lower bound of the range
    index start() const { return start_; } // no throw

    /// Returns the upper bound of the range
    index finish() const { return finish_; } // no throw

    /// Returns an array with the size of each dimension.
    size_array size() const { return size_; } // no throw

    size_array weight() const { return 1; } // no throw

    /// Returns the number of elements in the range.
    volume_type volume() const {
      return size_;
    }

    /// Check the coordinate to make sure it is within the range.
    bool includes(const index& i) const {
      return (start_ <= i) && (i < finish_);
    }

    /// Assignment Operator.
    Range_& operator =(const Range_& other) {
      start_ = other.start_;
      finish_ = other.finish_;
      size_ = other.size_;
      return *this;
    }

    /// Permute the tile given a permutation.
    Range_& operator ^=(const Permutation<1>& p) { return *this; }

    /// Change the dimensions of the range.
    Range_& resize(const index& start, const index& finish) {
      Range_ temp(start, finish);
      swap(temp);

      return *this;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & start_ & finish_ & size_;
    }

    void swap(Range_& other) {
      std::swap(start_, other.start_);
      std::swap(finish_, other.finish_);
      std::swap(size_, other.size_);
    }

  private:

    void increment(index& i) const {
      ++i;
    }

    index start_;    ///< Tile origin
    index finish_;   ///< Tile upper bound
    index size_;     ///< Dimension sizes
  }; // class Range


  class DynamicRange : public Range<DynamicRange> {
  public:
    typedef DynamicRange DynamicRange_;
    typedef Range<DynamicRange_> base;
    typedef base::coordinate_system coordinate_system;
    typedef base::volume_type volume_type;
    typedef base::index index;
    typedef base::ordinal_index ordinal_index;
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
  template <unsigned int DIM, typename Derived>
  Derived operator ^(const Permutation<DIM>& perm, const Range<Derived>& r) {
    TA_ASSERT(DIM == r.dim());
    return Derived(perm ^ r.start(), perm ^ r.finish(), r.order());
  }

  /// Returns true if the start and finish are equal.
  template <typename Derived1, typename Derived2>
  bool operator ==(const Range<Derived1>& r1, const Range<Derived2>& r2) {
#ifdef NDEBUG
    return ( r1.start() == r2.start() ) && ( r1.finish() == r2.finish() );
#else
    return ( r1.start() == r2.start() ) && ( r1.finish() == r2.finish() ) &&
        (r1.size() == r2.size()) && (r1.weight() == r2.weight()); // do an extra size check to catch bugs.
#endif
  }

  /// Returns true if the start and finish are not equal.
  template <typename Derived1, typename Derived2>
  bool operator !=(const Range<Derived1>& r1, const Range<Derived2>& r2) {
    return ! operator ==(r1, r2);
  }

  /// Returns a permuted range.
  template <typename Derived>
  Range<Derived> operator ^(const Permutation<1>& perm, const Range<Derived>& r) {
    TA_ASSERT(r.dim() == 1);
    return r;
  }

  /// ostream output orperator.
  template<typename CS>
  std::ostream& operator<<(std::ostream& out, const Range<CS>& blk) {
    out << "[ " << blk.start() << ", " << blk.finish() << " )";
    return out;
  }

} // namespace TiledArray
#endif // TILEDARRAY_RANGE_H__INCLUDED
