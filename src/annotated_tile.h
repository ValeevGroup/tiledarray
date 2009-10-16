#ifndef TILEDARRAY_ANNOTATED_TILE_H__INCLUDED
#define TILEDARRAY_ANNOTATED_TILE_H__INCLUDED

#include <error.h>
#include <variable_list.h>
#include <range.h>
#include <type_traits.h>
#include <boost/type_traits.hpp>
#include <Eigen/Core>
#include <vector>
#include <numeric>
#include <cstddef>

namespace TiledArray {
  // forward declaration
  template<typename T, unsigned int DIM, typename CS>
  class Tile;
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  class ArrayCoordinate;
  template <unsigned int DIM>
  class Permutation;
  template <unsigned int DIM>
  class LevelTag;
  namespace expressions {
    template<typename T>
    class AnnotatedTile;
    template <unsigned int DIM, typename T>
    AnnotatedTile<T> operator ^(const Permutation<DIM>&, const AnnotatedTile<T>&);
  }

  namespace detail {
    template<typename InIter, typename OutIter>
    void calc_weight(InIter first, InIter last, OutIter result);
  } // namespace detail


  namespace expressions {

    template<typename Exp0, typename Exp1, typename Op>
    struct BinaryTileExp;
    template<typename Exp, typename Op>
    struct UnaryTileExp;

    /// Annotated tile.
    template<typename T>
    class AnnotatedTile {
    public:
      typedef AnnotatedTile<T> AnnotatedTile_;
      typedef std::size_t ordinal_type;
      typedef std::size_t volume_type;
      typedef typename boost::remove_const<T>::type value_type;
      typedef typename detail::add_const<boost::is_const<T>::value, value_type>::type& reference_type;
      typedef const value_type & const_reference_type;
      typedef typename detail::add_const<boost::is_const<T>::value, value_type>::type* ptr_type;
      typedef const value_type * const_ptr_type;
      typedef std::vector<std::size_t> size_array;
      typedef ptr_type iterator;
      typedef const_ptr_type const_iterator;
    private:
      typedef Eigen::aligned_allocator<value_type> alloc_type;
    public:

      /// Default constructor

      /// Creates an annotated tile with no size or dimensions.
      AnnotatedTile() : data_(NULL), size_(), weight_(), n_(0), var_(),
          order_(detail::decreasing_dimension_order), owner_(false), alloc_()
      { }

      /// Create an annotated tile from a tile.

      /// Construct an annotated tile from tile. The data in the annotated tile
      /// is stored in the original tile. The annotated tile does not own the
      /// data and will not free any memory.
      /// \var \c t is the tile to be annotated.
      /// \var \c var is the variable annotation.
      template<unsigned int DIM, detail::DimensionOrderType O>
      AnnotatedTile(const Tile<value_type, DIM, CoordinateSystem<DIM, O> >& t, const VariableList& var) :
          data_(const_cast<T*>(t.data())), size_(t.size().begin(), t.size().end()),
          weight_(t.weight().begin(), t.weight().end()), n_(t.volume()),
          var_(var), order_(O), owner_(false), alloc_()
      {
        TA_ASSERT( t.dim() == var_.dim() ,
            std::runtime_error("AnnotatedTile<...>::AnnotatedTile(...): The number of variables in the variable list does not match the tile dimensions."));
      }

      /// Create an annotated tile with a constant initial value.

      /// The annotated tile will be set to \c size and data will be initialized
      /// with a value of \c val. The annotated tile is the owner of the data
      /// and will free the memory used by the tile.
      /// \var \c size is the size of each dimension.
      /// \var \c var is the variable annotation.
      /// \var \c val is the initial value of elements in the tile (optional).
      AnnotatedTile(const size_array& size, const VariableList& var,
          value_type val = value_type(), detail::DimensionOrderType o =
          detail::decreasing_dimension_order) :
          data_(NULL), size_(size), weight_((o == detail::decreasing_dimension_order ?
          calc_weight_<detail::decreasing_dimension_order>(size) :
          calc_weight_<detail::increasing_dimension_order>(size))),
          n_(std::accumulate(size.begin(), size.end(), std::size_t(1), std::multiplies<std::size_t>())),
          var_(var), order_(o), owner_(false), alloc_()
      {
        create_(val);
      }

      ///  Create an annotated tile with a data initialization list.

      /// The annotated tile will be set to \c size and data will be initialized
      /// with the list \c first to \c last. The data will be copied to the
      /// annotated tile and the tile will be the owner of the data. The tile
      /// will free the data.
      /// \var \c size is the size of each dimension.
      /// \var \c var is the variable annotation.
      /// \var \c first, \c last is the tile data initialization list.
      template<typename InIter>
      AnnotatedTile(const size_array& size, const VariableList& var, InIter first, InIter last,
          TiledArray::detail::DimensionOrderType o = TiledArray::detail::decreasing_dimension_order) :
          data_(NULL), size_(size), weight_((o == TiledArray::detail::decreasing_dimension_order ?
              calc_weight_<TiledArray::detail::decreasing_dimension_order>(size) :
              calc_weight_<TiledArray::detail::increasing_dimension_order>(size))),
          n_(std::accumulate(size.begin(), size.end(), std::size_t(1), std::multiplies<std::size_t>())),
          var_(var), order_(o), owner_(false), alloc_()
      {
        create_(first, last);
      }

      /// Copy constructor

      /// A shallow copy of the tile is created from the other annotated tile.
      /// \var \c other is the tile to be copied.
      AnnotatedTile(const AnnotatedTile_& other) :
          data_(other.data_), size_(other.size_), weight_(other.weight_),
          n_(other.n_), var_(other.var_), order_(other.order_), owner_(false),
          alloc_(other.alloc_)
      { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move constructor

      /// Move the data from the other annotated tile to this tile.
      /// \var \c other is the tile to be moved.
      AnnotatedTile(AnnotatedTile_&& other) :
          data_(other.data_), size_(std::move(other.size_)),
          weight_(std::move(other.weight_)), n_(other.n_), var_(std::move(other.var_)),
          order_(other.order_), owner_(other.owner_), alloc_(std::move(other.alloc_))
      {
        other.data_ = NULL;
        other.owner_ = false;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      ~AnnotatedTile() { destroy_(); }

      /// Annotated tile assignment operator.
      AnnotatedTile_& operator =(const AnnotatedTile_& other) {
        if(this != &other) {
          TA_ASSERT(var_ == other.var_,
              std::runtime_error("AnnotatedTile<...>::operator=(const AnnotatedTile_& other): The variable lists do not match."));
          if(owner_) {
            destroy_();
            data_ = other.data_;
            size_ = other.size_;
            weight_ = other.weight_;
            n_ = other.n_;
            order_ = other.order_;
          } else {
            TA_ASSERT(size_ == other.size_,
                std::runtime_error("AnnotatedTile<...>::operator=(const AnnotatedTile&): Right-hand tile dimensions do not match the dimensions of the referenced tile."));
            TA_ASSERT(order_ == other.order_,
                std::runtime_error("AnnotatedTile<...>::operator=(const AnnotatedTile&): Tile orders do not match."));
            std::copy(other.begin(), other.end(), data_);
          }

          alloc_ = other.alloc_;
        }

        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Annotated tile move assignment operator.
      AnnotatedTile_& operator =(AnnotatedTile_&& other) {
        if(this != &other) {
          TA_ASSERT(var_ == other.var_,
              std::runtime_error("AnnotatedTile<...>::operator=(AnnotatedTile_&& other): The variable lists do not match."));
          if(owner_) {
            swap(other);
          } else {
            TA_ASSERT(size_ == other.size_,
                std::runtime_error("AnnotatedTile<...>::operator=(AnnotatedTile&&): Right-hand tile dimensions do not match the dimensions of the referenced tile."));
            TA_ASSERT(order_ == other.order_,
                std::runtime_error("AnnotatedTile<...>::operator=(AnnotatedTile&&): Tile orders do not match."));
            std::copy(other.begin(), other.end(), data_);

            alloc_ = std::move(other.alloc_);
          }
        }

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// AnnotatedTile assignment operator
      template<typename Exp0, typename Exp1, typename Op>
      AnnotatedTile_& operator =(const BinaryTileExp<Exp0, Exp1, Op>& e) {
        *this = e.eval();

        return *this;
      }

      /// AnnotatedTile assignment operator
      template<typename Exp, typename Op>
      AnnotatedTile_& operator =(const UnaryTileExp<Exp, Op>& e) {
        *this = e.eval();

        return *this;
      }

      iterator begin() { return data_; }
      const_iterator begin() const { return data_; }
      iterator end() { return data_ + n_; }
      const_iterator end() const { return data_ + n_; }
      /// Returns a pointer to the tile data.

      /// The pointer returned by this function may be constant if the annotated
      /// tile type is constant (e.g. AnnotatedTile<const double,
      /// decreasing_dimension_order>).
      ptr_type data() { return data_; }
      /// Returns a constant pointer to the tile data.
      const_ptr_type data() const { return data_; }
      /// Returns a constant reference to a vector with the dimension sizes.
      const size_array& size() const { return size_; }
      /// Returns a constant reference to a vector with the dimension weights.
      const size_array& weight() const { return weight_; }
      /// Returns the number of elements contained by the tile.
      volume_type volume() const { return n_; }
      /// Returns a constant reference to variable list (the annotation).
      const VariableList& vars() const { return var_; }
      /// Returns the number of dimensions of the tile.
      unsigned int dim() const { return var_.dim(); }
      /// Return the array storage order
      detail::DimensionOrderType order() const { return order_; }

      /// Returns a reference to element i (range checking is performed).

      /// This function provides element access to the element located at index i.
      /// If i is not included in the range of elements, std::out_of_range will be
      /// thrown. Valid types for Index are ordinal_type and index_type.
      template <typename Index>
      reference_type at(const Index& i) {
        if(! includes(i))
          throw std::out_of_range("DenseArrayStorage<...>::at(...): Element is not in range.");

        return * (data_ + ord_(i));
      }

      /// Returns a constant reference to element i (range checking is performed).

      /// This function provides element access to the element located at index i.
      /// If i is not included in the range of elements, std::out_of_range will be
      /// thrown. Valid types for Index are ordinal_type and index_type.
      template <typename Index>
      const_reference_type at(const Index& i) const {
        if(! includes(i))
          throw std::out_of_range("DenseArrayStorage<...>::at(...) const: Element is not in range.");

        return * (data_ + ord_(i));
      }

      /// Returns a reference to the element at i.

      /// This No error checking is performed.
      template <typename Index>
      reference_type operator[](const Index& i) { // no throw for non-debug
#ifdef NDEBUG
        return * (data_ + ord_(i));
#else
        return at(i);
#endif
      }

      /// Returns a constant reference to element i. No error checking is performed.
      template <typename Index>
      const_reference_type operator[](const Index& i) const { // no throw for non-debug
#ifdef NDEBUG
        return * (data_ + ord_(i));
#else
        return at(i);
#endif
      }

      /// Returns true if the index \c i is included by the tile.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      bool includes(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
        TA_ASSERT(dim() == DIM,
            std::runtime_error("AnnotatedTile<...>::includes(...): Coordinate dimension is not equal to tile dimension."));
        TA_ASSERT(order() == O,
            std::runtime_error("AnnotatedTile<...>::includes(...): Coordinate order does not match tile order."));
        for(unsigned int d = 0; d < dim(); ++d)
          if(size_[d] <= i[d])
            return false;

        return true;
      }

      /// Returns true if the ordinal index is included by the tile.
      bool includes(const ordinal_type& i) const {
        return (i < n_);
      }

      template<unsigned int DIM>
      AnnotatedTile_& operator ^=(const Permutation<DIM>& p) {
        TA_ASSERT(owner_,
            std::runtime_error("AnnotatedTile<...>::operator^=(...): This annotated tile cannot be permuted in place because it references another tile."));
        AnnotatedTile_ temp = p ^ *this;
        swap(temp);

        return *this;
      }

      void swap(AnnotatedTile_& other) {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(weight_, other.weight_);
        std::swap(n_, other.n_);
        var_.swap(other.var_);
        std::swap(order_, other.order_);
        std::swap(owner_, other.owner_);
        std::swap(alloc_, other.alloc_);
      }

    private:

      /// Returns the ordinal index for the given index.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      ordinal_type ord_(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
        const typename ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >::index init = 0;
        return std::inner_product(i.begin(), i.end(), weight_.begin(), init);
      }

      /// Returns the given ordinal index.
      ordinal_type ord_(const ordinal_type i) const { return i; }

      /// Allocate and initialize the array w/ a constant value.

      /// All elements will contain the given value.
      void create_(const value_type val) {
        owner_ = true;
        data_ = alloc_.allocate(n_);
        for(std::size_t i = 0; i < n_; ++i)
          alloc_.construct(data_ + i, val);
      }

      /// Allocate and initialize the array.

      /// All elements will be initialized to the values given by the iterators.
      /// If the iterator range does not contain enough elements to fill the array,
      /// the remaining elements will be initialized with the default constructor.
      template <typename InIter>
      void create_(InIter first, InIter last) {
        owner_ = true;
        data_ = alloc_.allocate(n_);
        std::size_t i = 0;
        for(; first != last; ++first, ++i)
          alloc_.construct(data_ + i, *first);
        for(; i < n_; ++i)
          alloc_.construct(data_ + i, value_type());
      }

      /// Destroy the array
      void destroy_() {
        if(!owner_)
          return;
        value_type* p = const_cast<value_type*>(data_);
        const_ptr_type const end = data_ + n_;
        for(; p != end; ++p)
          alloc_.destroy(p);

        alloc_.deallocate(const_cast<value_type*>(data_), n_);
        data_ = NULL;
        owner_ = false;
      }

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

      ptr_type data_;           ///< tile data
      size_array size_;         ///< tile size
      size_array weight_;       ///< dimension weights
      std::size_t n_;           ///< tile volume
      VariableList var_;        ///< variable list
      TiledArray::detail::DimensionOrderType order_; ///< Array order
      bool owner_;              ///< true when tile data is owned by this object
      alloc_type alloc_;        ///< allocator

    }; // class AnnotatedTile


    template <unsigned int DIM, typename T>
    AnnotatedTile<T> operator ^(const Permutation<DIM>& p, const AnnotatedTile<T>& t) {
      TA_ASSERT((t.dim() == DIM),
          std::runtime_error("operator^(const Permutation<DIM>&, const AnnotatedTile<T>&): The permutation dimension is not equal to the tile dimensions."));

      AnnotatedTile<T> result(p ^ t.size(), p ^ t.vars());
      detail::Permute<AnnotatedTile<T> > f_perm(t);
      f_perm(p, result.begin(), result.end());

      return result;
    }

  } // namespace expressions
} // namespace TiledArray
#endif // TILEDARRAY_ANNOTATED_TILE_H__INCLUDED
