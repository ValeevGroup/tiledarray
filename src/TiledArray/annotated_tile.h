#ifndef TILEDARRAY_ANNOTATED_TILE_H__INCLUDED
#define TILEDARRAY_ANNOTATED_TILE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/array_ref.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/variable_list.h>
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
  template <typename T, unsigned int DIM, typename CS, typename C>
  class Array;

  namespace expressions {
    namespace tile {
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

      /// Annotated tile.
      template<typename T>
      class AnnotatedTile {
      public:
        typedef AnnotatedTile<T> AnnotatedTile_;
        typedef typename boost::remove_const<T>::type value_type;
        typedef std::size_t index;
        typedef index ordinal_type;
        typedef ordinal_type index_type;
        typedef index volume_type;
        typedef typename detail::add_const<boost::is_const<T>::value, value_type>::type& reference_type;
        typedef const value_type & const_reference_type;
        typedef typename detail::add_const<boost::is_const<T>::value, value_type>::type* pointer;
        typedef const value_type * const_pointer;
        typedef detail::ArrayRef<index> size_array;
        typedef pointer iterator;
        typedef const_pointer const_iterator;
      private:
        typedef Eigen::aligned_allocator<value_type> alloc_type;

      public:
        /// Default constructor
        AnnotatedTile() : order_(TiledArray::detail::increasing_dimension_order),
            size_(NULL, NULL), weight_(NULL, NULL), n_(0), data_(NULL), var_(),
            owner_(false)
        { }

        /// Create an annotated tile from a tile.

        /// Construct an annotated tile from tile. The data in the annotated tile
        /// is stored in the original tile. The annotated tile does not own the
        /// data and will not free any memory.
        /// \var \c t is the tile to be annotated.
        /// \var \c var is the variable annotation.
        template<unsigned int DIM, detail::DimensionOrderType O>
        AnnotatedTile(const Tile<value_type, DIM, CoordinateSystem<DIM, O> >& t, const VariableList& v) :
            // Note: Even though constness is being removed from these pointers
            // the content cannot be modified when T is const.
            order_(O), size_(const_cast<index*>(t.size().begin()), const_cast<index*>(t.size().end())),
            weight_(const_cast<index*>(t.weight().begin()), const_cast<index*>(t.weight().end())),
            n_(t.volume()), data_(const_cast<value_type*>(t.data())), var_(v),
            owner_(false)
        { }

        /// Create an annotated tile with a constant initial value.

        /// The annotated tile will be set to \c size and data will be initialized
        /// with a value of \c val. The annotated tile is the owner of the data
        /// and will free the memory used by the tile.
        /// \var \c size is the size of each dimension.
        /// \var \c var is the variable annotation.
        /// \var \c val is the initial value of elements in the tile (optional).
        template<typename SizeArray>
        AnnotatedTile(const SizeArray& size, const VariableList& var,
            value_type val = value_type(),
            detail::DimensionOrderType o = detail::decreasing_dimension_order) :
            order_(o), size_(create_size_(size.begin(), size.end())),
            weight_(create_weight_()), n_(calc_volume_()),
            data_(create_data_(val)), var_(var), owner_(true)
        { }

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
            order_(o), size_(create_size_(size.begin(), size.end())),
            weight_(create_weight_()), n_(calc_volume_()),
            data_(create_data_(first, last)), var_(var), owner_(true)
        {
          // Note: The transoform iterators used in tile math do not have the
          // correct iterator_catagory so the static assertion is failing when it
          // should not. Or, the problem may be from zip iterator. Or, it may be
          // from a combination of the two. Or, it could be from the use of
          // pointers as the base iterators. Or, the fact that boost tuples always
          // use 10 parameters. More information is needed.
          // Todo: Fix the iterators in tile math so they assert correctly.
//          BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
        }

        /// Copy constructor

        /// A shallow copy of the tile is created from the other annotated tile.
        /// \var \c other is the tile to be copied.
        AnnotatedTile(const AnnotatedTile_& other) : order_(other.order_),
            size_(( other.owner_ ? create_size_(other.size_.begin(), other.size_.end()) : other.size())),
            weight_(( other.owner_ ? weight_ = create_weight_(other.weight_.begin(), other.weight_.end()) : other.weight())),
            n_(other.n_), data_((other.owner_ ? create_data_(other.begin(), other.end()) : other.data_)),
            var_(other.var_), owner_(false)
        { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
        /// Move constructor

        /// Move the data from the other annotated tile to this tile.
        /// \var \c other is the tile to be moved.
        AnnotatedTile(AnnotatedTile_&& other) :
            order_(other.order_), size_(other.size_.begin(), other.size_.end()),
            weight_(other.weight_.begin(), other.weight_.end()), n_(other.n_),
            data_(other.data_), var_(std::move(other.var_)), owner_(other.owner_)
        {
          other.size_ = size_array(NULL, NULL);
          other.weight_ = size_array(NULL, NULL);
          other.data_ = NULL;
          other.owner_ = false;
        }
#endif // __GXX_EXPERIMENTAL_CXX0X__

        ~AnnotatedTile() {
          destroy_();
        }

        /// Annotated tile assignment operator.

        /// If this AnnotatedTile references a Tile object, the tile data of the
        /// other tile is copied to this tile. In this case, the dimensions of
        /// both tiles must match. Otherwise, a deep copy of the other tile is
        /// performed. If the other tile is a reference itself, this tile will
        /// reference that same tile.
        AnnotatedTile_& operator =(const AnnotatedTile_& other) {
          if(this == &other)
            return *this;     // Do not copy yourself.

          if(owner_) {
            // This is not a reference to a Tile so we will copy other.
            order_ = other.order_;
            var_ = other.var_;
            if(! other.owner_) {
              // The other tile is a reference and we can safely copy its pointers.
              size_ = other.size_;
              weight_ = other.weight_;
              n_ = other.n_;
              data_ = other.data_;
            } else {
              // The other tile is not a reference so we need to do a deep copy.
              if(dim() != other.dim()) {
                // the dimensions are different so we need to reallocate.
                destroy_array_(size_);
                destroy_array_(weight_);
                size_ = create_size_(other.size_.begin(), other.size_.end());
                weight_ = create_weight_(other.weight_.begin(), other.weight_.end());
              } else {
                // The dimensions are the same so we only need to copy.
                std::copy(other.size_.begin(), other.size_.end(), size_.begin());
                std::copy(other.weight_.begin(), other.weight_.end(), weight_.end());
              }
              if(n_ != other.n_) {
                // The volumes are different so we need to reallocate data.
                destroy_data_();
                n_ = other.n_;
                data_ = create_data_(other.begin(), other.end());
              } else {
                // The volumes are the same so we only need to copy.
                std::copy(other.begin(), other.end(), data_);
              }
            }
          } else {
            // This is a reference to a Tile, so we need to verify that the
            // dimensions of the other tile match this tile and copy the data.
            TA_ASSERT(size_ == other.size_, std::runtime_error,
                "Right-hand tile dimensions do not match the left-hand tile dimensions.");
            TA_ASSERT(order_ == other.order_, std::runtime_error,
                "Tile orders do not match.");
            TA_ASSERT(this->var_ == other.var_, std::runtime_error,
                "The variable lists do not match.");
            std::copy(other.begin(), other.end(), data_);
          }

          return *this;
        }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
        /// Annotated tile move assignment operator.
        AnnotatedTile_& operator =(AnnotatedTile_&& other) {
          if(owner_) {
            swap_(other);
          } else {
            // This is a reference to a Tile, so we need to verify that the
            // dimensions of the other tile match this tile and copy the data.
            TA_ASSERT(size_ == other.size_, std::runtime_error,
                "Right-hand tile dimensions do not match the left-hand tile dimensions.");
            TA_ASSERT(order_ == other.order_, std::runtime_error,
                "Tile orders do not match.");
            TA_ASSERT(this->var_ == other.var_, std::runtime_error,
                "The variable lists do not match.");
            std::copy(other.begin(), other.end(), begin());
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

        /// Returns an iterator to the first element of the tile data.
        iterator begin() { return data_; }
        /// Returns a const iterator to the first element of the tile data.
        const_iterator begin() const { return data_; }
        /// Returns an iterator to the end of the data array.
        iterator end() { return data_ + this->n_; }
        /// Returns a const iterator to the end of the data array.
        const_iterator end() const { return data_ + this->n_; }

        /// Returns a pointer to the tile data.
        pointer data() { return data_; }
        /// Returns a const pointer to the tile data.
        const_pointer data() const { return data_; }
        /// Returns a constant reference to a vector with the dimension sizes.
        const size_array& size() const { return size_; }
        /// Returns a constant reference to a vector with the dimension weights.
        const size_array& weight() const { return weight_; }
        /// Returns the number of elements contained by the array.
        volume_type volume() const { return n_; }
        /// Returns a constant reference to variable list (the annotation).
        const VariableList& vars() const { return var_; }
        /// Returns the number of dimensions of the array.
        unsigned int dim() const { return var_.dim(); }
        /// Return the array storage order
        detail::DimensionOrderType order() const { return order_; }

        /// Returns true if the index \c i is included by the array.
        template<typename II, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        bool includes(const ArrayCoordinate<II,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
          TA_ASSERT(dim() == DIM, std::runtime_error,
              "Coordinate dimension is not equal to array dimension.");
          TA_ASSERT(order() == O, std::runtime_error,
              "Coordinate order does not match array dimension order.");
          for(unsigned int d = 0; d < dim(); ++d)
            if(size_[d] <= i[d])
              return false;

          return true;
        }

        /// Returns true if the ordinal index is included by this array.
        bool includes(const ordinal_type& i) const {
          return (i < n_);
        }

        /// Returns a reference to element i (range checking is performed).

        /// This function provides element access to the element located at index i.
        /// If i is not included in the range of elements, std::out_of_range will be
        /// thrown. Valid types for Index are ordinal_type and index_type.
        template <typename Index>
        reference_type at(const Index& i) {
          if(! includes(i))
            TA_EXCEPTION( std::out_of_range , "TiledArray range check failure:",
                "Element is not in range.");

          return * (data_ + ord_(i));
        }

        /// Returns a constant reference to element i (range checking is performed).

        /// This function provides element access to the element located at index i.
        /// If i is not included in the range of elements, std::out_of_range will be
        /// thrown. Valid types for Index are ordinal_type and index_type.
        template <typename Index>
        const_reference_type at(const Index& i) const {
          if(! includes(i))
            TA_EXCEPTION(std::out_of_range, "TiledArray range check failure:",
                "Element is not in range.");

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
          swap_(temp);

          return *this;
        }

      private:

        /// Allocate and initialize the array w/ a constant value.

        /// All elements will contain the given value.
        value_type* create_data_(const value_type val) {
          owner_ = true;
          value_type* first = alloc_.allocate(n_);
          value_type* end = first + n_;
          for(value_type* it = first; it != end; ++it)
            alloc_.construct(it, val);

          return first;
        }

        /// Allocate and initialize the array.

        /// All elements will be initialized to the values given by the iterators.
        /// If the iterator range does not contain enough elements to fill the array,
        /// the remaining elements will be initialized with the default constructor.
        template <typename InIter>
        value_type* create_data_(InIter first, InIter last) {
          value_type* data = alloc_.allocate(n_);
          value_type* it = data;
          ;
          for(; first != last; ++first, ++it)
            alloc_.construct(it, *first);
          const value_type val = value_type();
          for(const value_type* const end = data + n_; it != end; ++it)
            alloc_.construct(it, val);

          return data;
        }

        /// Destroy the array
        void destroy_data_() {
          const value_type* const end = data_ + n_;
          for(value_type* first = data_; first != end; ++first)
            alloc_.destroy(first);

          alloc_.deallocate(data_, n_);
          data_ = NULL;
        }

        static size_array create_array_(const unsigned int dim) {
          index* a = new index[dim];
          return size_array(a, a + dim);
        }

        static void destroy_array_(size_array& a) {
          delete [] a.c_array();
          a = size_array(NULL, NULL);
        }

        template<typename InIter>
        static size_array create_size_(InIter first, InIter last) {
          size_array result = create_array_(std::distance(first, last));
          std::copy(first, last, result.begin());

          return result;
        }

        size_array create_weight_() {
          size_array result = create_array_(std::distance(size_.begin(), size_.end()));
          if(order_ == detail::increasing_dimension_order)
            calc_weight_<detail::increasing_dimension_order>(result);
          else
            calc_weight_<detail::decreasing_dimension_order>(result);

          return result;
        }

        template<typename InIter>
        static size_array create_weight_(InIter first, InIter last) {
          const std::size_t dim = std::distance(first, last);
          index* weight = new index[dim];
          size_array result(weight, weight + dim);
          std::copy(first, last, result.begin());

          return result;
        }



        void destroy_() {
          if(!owner_)
            return;
          destroy_data_();
          destroy_array_(weight_);
          destroy_array_(size_);
          owner_ = false;
        }

        /// Returns the ordinal index for the given index.
        template<typename II, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        ordinal_type ord_(const ArrayCoordinate<II,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
          return std::inner_product(i.begin(), i.end(), weight_.begin(), II(0));
        }

        /// Returns the given ordinal index.
        ordinal_type ord_(const ordinal_type i) const { return i; }

        /// Class wrapper function for detail::calc_weight() function.
        template<detail::DimensionOrderType O>
        void calc_weight_(size_array& weight) { // no throw
          typedef detail::CoordIterator<size_array, O> CI;
          TiledArray::detail::calc_weight(CI::begin(size_), CI::end(size_),  CI::begin(weight));
        }

        volume_type calc_volume_() const {
          return detail::volume(size_.begin(), size_.end());
        }

        void swap_(AnnotatedTile_& other) {
          std::swap(order_, other.order_);
          detail::swap(size_, other.size_);
          detail::swap(weight_, other.weight_);
          std::swap(n_, other.n_);
          std::swap(data_, other.data_);
          std::swap(var_, other.var_);
          std::swap(owner_, other.owner_);
          std::swap(alloc_, other.alloc_);
        }

        friend void swap<>(AnnotatedTile_&, AnnotatedTile_&);
        template <class, typename>
        friend struct madness::archive::ArchiveStoreImpl;
        template <class, typename>
        friend struct madness::archive::ArchiveLoadImpl;
        template <typename, unsigned int, typename, typename>
        friend class Array;


        TiledArray::detail::DimensionOrderType order_; ///< Array order
        size_array size_;         ///< tile size
        size_array weight_;       ///< dimension weights
        volume_type n_;           ///< tile volume
        value_type* data_;        ///< tile data
        VariableList var_;        ///< variable list
        bool owner_;              ///< true when tile data is owned by this object
        alloc_type alloc_;        ///< allocator

      }; // class AnnotatedTile

      /// Exchange the content of the two annotated tiles.
      template<typename T>
      void swap(AnnotatedTile<T>& t0, AnnotatedTile<T>& t1) {
        t0.swap_(t1);
      }

      template <unsigned int DIM, typename T>
      AnnotatedTile<T> operator ^(const Permutation<DIM>& p, const AnnotatedTile<T>& t) {
        TA_ASSERT((t.dim() == DIM), std::runtime_error,
            "The permutation dimension is not equal to the tile dimensions.");

        AnnotatedTile<T> result(p ^ t.size(), p ^ t.vars(), T());
        detail::Permute<AnnotatedTile<T> > f_perm(t);
        f_perm(p, result.begin(), result.end());

        return result;
      }

    } // namespace tile
  } // namespace expressions
} // namespace TiledArray

namespace madness {
  namespace archive {

    // Forward declarations
    template <class Archive, typename T>
    struct ArchiveLoadImpl;
    template <class Archive, typename T>
    struct ArchiveStoreImpl;

    /// Provides input archiving capability for AnnotatedTiles.
    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, TiledArray::expressions::tile::AnnotatedTile<T> > {
      typedef TiledArray::expressions::tile::AnnotatedTile<T> atile_type;

      /// Loads an AnnotatedTile from an archive
      static void load(const Archive& ar, atile_type& t = 0) {
        t.destroy_();
        unsigned int dim = 0;
        ar & t.order_ & dim;
        t.size_ = atile_type::create_array_(dim);
        t.weight_ = atile_type::create_array_(dim);
        ar & wrap(t.size_.c_array(), dim) & wrap(t.weight_.c_array(), dim) & t.n_;
        t.data_ = t.create_data_(T());
        ar & wrap(t.data_, t.n_) & t.var_;
        t.owner_ = true;
      }
    }; // struct ArchiveLoadImpl<Archive, TiledArray::expression::tile::AnnotatedTile<T> >

    /// Provides output archiving capability for AnnotatedTiles.
    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, TiledArray::expressions::tile::AnnotatedTile<T> > {
      typedef TiledArray::expressions::tile::AnnotatedTile<T> atile_type;

      /// Stores an AnnotatedTile to an archive
      static void store(const Archive& ar, const atile_type& t) {
        const unsigned int dim = t.dim();
        ar & t.order_ & dim & wrap(t.size_.data(), dim) & wrap(t.weight_.data(), dim)
            & t.n_ & wrap(t.data_, t.n_) & t.var_;
      }
    }; // struct ArchiveStoreImpl<Archive, TiledArray::expression::tile::AnnotatedTile<T> >

  } // namespace archive
} // namespace madness

#endif // TILEDARRAY_ANNOTATED_TILE_H__INCLUDED
