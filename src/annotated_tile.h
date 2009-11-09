#ifndef TILEDARRAY_ANNOTATED_TILE_H__INCLUDED
#define TILEDARRAY_ANNOTATED_TILE_H__INCLUDED

#include <error.h>
#include <annotation.h>
#include <type_traits.h>
#include <boost/type_traits.hpp>
#include <Eigen/Core>
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
    template<typename Exp0, typename Exp1, typename Op>
    struct BinaryTileExp;
    template<typename Exp, typename Op>
    struct UnaryTileExp;
    template<typename T>
    class AnnotatedTile;
    template<typename T>
    void swap(AnnotatedTile<T>&, AnnotatedTile<T>&);
    template <unsigned int DIM, typename T>
    AnnotatedTile<T> operator ^(const Permutation<DIM>&, const AnnotatedTile<T>&);
  }

  namespace expressions {

    /// Annotated tile.
    template<typename T>
    class AnnotatedTile : public Annotation {
    public:
      typedef AnnotatedTile<T> AnnotatedTile_;
      typedef Annotation::ordinal_type ordinal_type;
      typedef Annotation::volume_type volume_type;
      typedef typename boost::remove_const<T>::type value_type;
      typedef typename detail::add_const<boost::is_const<T>::value, value_type>::type& reference_type;
      typedef const value_type & const_reference_type;
      typedef typename detail::add_const<boost::is_const<T>::value, value_type>::type* ptr_type;
      typedef const value_type * const_ptr_type;
      typedef Annotation::size_array size_array;
      typedef ptr_type iterator;
      typedef const_ptr_type const_iterator;
    private:
      typedef Eigen::aligned_allocator<value_type> alloc_type;
    public:

      /// Default constructor

      /// Creates an annotated tile with no size or dimensions.
      AnnotatedTile() :
          Annotation(), data_(NULL), owner_(false), alloc_()
      { }

      /// Create an annotated tile from a tile.

      /// Construct an annotated tile from tile. The data in the annotated tile
      /// is stored in the original tile. The annotated tile does not own the
      /// data and will not free any memory.
      /// \var \c t is the tile to be annotated.
      /// \var \c var is the variable annotation.
      template<unsigned int DIM, detail::DimensionOrderType O>
      AnnotatedTile(const Tile<value_type, DIM, CoordinateSystem<DIM, O> >& t, const VariableList& var) :
          Annotation(t.size().begin(), t.size().end(), t.weight().begin(),
          t.weight().end(), t.volume(), var, O), data_(const_cast<T*>(t.data())),
          owner_(false), alloc_()
      {
        TA_ASSERT( t.dim() == var_.dim() , std::runtime_error,
            "The number of variables in the variable list does not match the tile dimensions.");
      }

      /// Create an annotated tile with a constant initial value.

      /// The annotated tile will be set to \c size and data will be initialized
      /// with a value of \c val. The annotated tile is the owner of the data
      /// and will free the memory used by the tile.
      /// \var \c size is the size of each dimension.
      /// \var \c var is the variable annotation.
      /// \var \c val is the initial value of elements in the tile (optional).
      template<typename SizeArray>
      AnnotatedTile(const SizeArray& size, const VariableList& var,
          value_type val = value_type(), detail::DimensionOrderType o =
          detail::decreasing_dimension_order) :
          Annotation(size.begin(), size.end(), var, o), data_(NULL),
          owner_(false), alloc_()
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
      template<typename SizeArray, typename InIter>
      AnnotatedTile(const SizeArray& size, const VariableList& var, InIter first, InIter last,
          TiledArray::detail::DimensionOrderType o = TiledArray::detail::decreasing_dimension_order) :
          Annotation(size.begin(), size.end(), var, o), data_(NULL),
          owner_(false), alloc_()
      {
        create_(first, last);
      }

      /// Copy constructor

      /// A shallow copy of the tile is created from the other annotated tile.
      /// \var \c other is the tile to be copied.
      AnnotatedTile(const AnnotatedTile_& other) :
          Annotation(other), data_(other.data_), owner_(false), alloc_(other.alloc_)
      { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move constructor

      /// Move the data from the other annotated tile to this tile.
      /// \var \c other is the tile to be moved.
      AnnotatedTile(AnnotatedTile_&& other) :
          Annotation(std::move(other)), data_(other.data_), owner_(other.owner_),
          alloc_(std::move(other.alloc_))
      {
        other.data_ = NULL;
        other.owner_ = false;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      ~AnnotatedTile() { destroy_(); }

      /// Annotated tile assignment operator.
      AnnotatedTile_& operator =(const AnnotatedTile_& other) {
        if(this != &other) {
          TA_ASSERT(this->var_ == other.var_, std::runtime_error, "The variable lists do not match.");
          if(owner_) {
            destroy_();
            data_ = other.data_;
            Annotation::operator=(other);
          } else {
            TA_ASSERT(size_ == other.size_, std::runtime_error,
                "Right-hand tile dimensions do not match the dimensions of the referenced tile.");
            TA_ASSERT(order_ == other.order_, std::runtime_error,
                "Tile orders do not match.");
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
          TA_ASSERT(var_ == other.var_, std::runtime_error,
              "The variable lists do not match.");
          if(owner_) {
            swap(*this, other);
          } else {
            TA_ASSERT(size_ == other.size_, std::runtime_error,
                "Right-hand tile dimensions do not match the dimensions of the referenced tile.");
            TA_ASSERT(order_ == other.order_, std::runtime_error,
                "Tile orders do not match.");
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
      iterator end() { return data_ + this->n_; }
      const_iterator end() const { return data_ + this->n_; }

      /// Returns a pointer to the tile data.

      /// The pointer returned by this function may be constant if the annotated
      /// tile type is constant (e.g. AnnotatedTile<const double,
      /// decreasing_dimension_order>).
      ptr_type data() { return data_; }
      /// Returns a constant pointer to the tile data.
      const_ptr_type data() const { return data_; }

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

      template<unsigned int DIM>
      AnnotatedTile_& operator ^=(const Permutation<DIM>& p) {
        TA_ASSERT(owner_, std::runtime_error,
            "This annotated tile cannot be permuted in place because it references another tile.");
        AnnotatedTile_ temp = p ^ *this;
        swap(*this, temp);

        return *this;
      }

    private:

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

      friend void TiledArray::expressions::swap<>(AnnotatedTile_&, AnnotatedTile_&);

      ptr_type data_;           ///< tile data
      bool owner_;              ///< true when tile data is owned by this object
      alloc_type alloc_;        ///< allocator

    }; // class AnnotatedTile

    /// Exchange the content of the two annotated tiles.
    template<typename T>
    void swap(AnnotatedTile<T>& t0, AnnotatedTile<T>& t1) {
      t0.swap(t1);
      std::swap(t0.data_, t1.data_);
      std::swap(t0.owner_, t1.owner_);
      std::swap(t0.alloc_, t1.alloc_);
    }

    template <unsigned int DIM, typename T>
    AnnotatedTile<T> operator ^(const Permutation<DIM>& p, const AnnotatedTile<T>& t) {
      TA_ASSERT((t.dim() == DIM), std::runtime_error,
          "The permutation dimension is not equal to the tile dimensions.");

      AnnotatedTile<T> result(p ^ t.size(), p ^ t.vars());
      detail::Permute<AnnotatedTile<T> > f_perm(t);
      f_perm(p, result.begin(), result.end());

      return result;
    }

  } // namespace expressions
} // namespace TiledArray
#endif // TILEDARRAY_ANNOTATED_TILE_H__INCLUDED
