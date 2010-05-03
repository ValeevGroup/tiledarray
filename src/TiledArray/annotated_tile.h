#ifndef TILEDARRAY_ANNOTATED_TILE_H__INCLUDED
#define TILEDARRAY_ANNOTATED_TILE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/array_ref.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/variable_list.h>
#include <TiledArray/tile.h>
#include <boost/type_traits.hpp>
#include <boost/shared_array.hpp>
#include <boost/make_shared.hpp>
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

      template<typename T, typename I>
      class AnnotatedTileImplBase {
      public:
        typedef AnnotatedTileImplBase<T, I> TileHolderBase_;
        typedef T value_type;
        typedef I index_type;
        typedef I ordinal_type;
        typedef I volume_type;
        typedef T& reference;
        typedef const T& const_reference;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef detail::ArrayRef<const index_type> size_array;

        typedef pointer iterator;
        typedef const_pointer const_iterator;

        /// virtual destructor.
        virtual ~AnnotatedTileImplBase() { }

        // iterator interface
        virtual iterator begin() = 0;
        virtual const_iterator begin() const = 0;
        virtual iterator end() = 0;
        virtual const_iterator end() const = 0;

        // data access interface
        virtual pointer data() = 0;
        virtual const_pointer data() const = 0;
        virtual bool initialized() const = 0;

        // tile information access
        virtual size_array size() const = 0;
        virtual size_array weight() const = 0;
        virtual volume_type volume() const = 0;
        virtual unsigned int dim() const = 0;
        virtual detail::DimensionOrderType order() const = 0;
        virtual bool includes(const index_type) const = 0;

        // Element access
        virtual reference at(const index_type) = 0;
        virtual const_reference at(const index_type) const = 0;
        virtual reference operator[](const index_type) = 0;
        virtual const_reference operator[](const index_type) const = 0;

      }; // class AnnotatedTileImplBase

      template<typename T>
      class AnnotatedTileRefImpl : public AnnotatedTileImplBase<typename T::value_type, typename T::ordinal_type> {
      public:
        typedef AnnotatedTileImplBase<typename T::value_type, typename T::ordinal_type> AnnotatedTileImplBase_;
        typedef typename AnnotatedTileImplBase_::value_type value_type;
        typedef typename AnnotatedTileImplBase_::index_type index_type;
        typedef typename AnnotatedTileImplBase_::ordinal_type ordinal_type;
        typedef typename AnnotatedTileImplBase_::volume_type volume_type;
        typedef typename AnnotatedTileImplBase_::reference reference;
        typedef typename AnnotatedTileImplBase_::const_reference const_reference;
        typedef typename AnnotatedTileImplBase_::pointer pointer;
        typedef typename AnnotatedTileImplBase_::const_pointer const_pointer;
        typedef typename AnnotatedTileImplBase_::size_array size_array;

        typedef typename AnnotatedTileImplBase_::iterator iterator;
        typedef typename AnnotatedTileImplBase_::const_iterator const_iterator;

        typedef T tile_type;

        // Constructors
        AnnotatedTileRefImpl() : tile_(NULL) { }
        AnnotatedTileRefImpl(tile_type& t) : tile_(&t) { }
        AnnotatedTileRefImpl(const tile_type& t) : tile_(const_cast<tile_type*>(&t)) { }
        AnnotatedTileRefImpl(const AnnotatedTileRefImpl<T>& other) : tile_(other.tile_) { }

        // virtual destructor
        virtual ~AnnotatedTileRefImpl() { }

        // iterator interface
        virtual iterator begin() { return tile_->begin(); }
        virtual const_iterator begin() const { return tile_->begin(); }
        virtual iterator end() { return tile_->end(); }
        virtual const_iterator end() const { return tile_->end(); }

        // data access interface
        virtual pointer data() { return tile_->data(); }
        virtual const_pointer data() const { return tile_->data(); }
        virtual bool initialized() const { return tile_->initialized(); }

        // tile information access
        virtual size_array size() const { return size_array(tile_->size()); }
        virtual size_array weight() const { return size_array(tile_->weight()); }
        virtual volume_type volume() const { return tile_->volume(); }
        virtual unsigned int dim() const { return T::dim; }
        virtual detail::DimensionOrderType order() const { return T::order; }
        virtual bool includes(const index_type i) const { return tile_->includes(i); }

        // Element access
        virtual reference at(const index_type i) { return tile_->at(i); }
        virtual const_reference at(const index_type i) const { return tile_->at(i); }
        virtual reference operator[](const index_type i) { return (*tile_)[i]; }
        virtual const_reference operator[](const index_type i) const { return (*tile_)[i]; }

      private:
        tile_type* tile_;
      }; // class AnnotatedTileImplBase

      template<typename T, typename I>
      class AnnotatedTileDataImpl : public AnnotatedTileImplBase<T, I> {
      private:
        typedef AnnotatedTileDataImpl<T, I> AnnotatedTileDataImpl_;
        typedef AnnotatedTileImplBase<T, I> AnnotatedTileImplBase_;
        typedef Eigen::aligned_allocator<T> alloc_type;

      public:
        typedef T value_type;
        typedef I index_type;
        typedef I ordinal_type;
        typedef I volume_type;
        typedef T& reference;
        typedef const T& const_reference;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef detail::ArrayRef<const index_type> size_array;

        typedef pointer iterator;
        typedef const_pointer const_iterator;

        AnnotatedTileDataImpl() :
            dim_(0),
            volume_(0),
            data_(),
            size_weight_(),
            order_(detail::decreasing_dimension_order)
        { }

        template<typename SizeInIter, typename DataInIter>
        AnnotatedTileDataImpl(SizeInIter s_first, SizeInIter s_last, DataInIter d_first, DataInIter d_last, detail::DimensionOrderType o) :
            dim_(std::distance(s_first, s_last)),
            volume_(detail::volume(s_first, s_last)),
            data_(init_data_(d_first, d_last)),
            size_weight_(init_size_weight_(s_first, s_last)),
            order_(o)
        { }

        template<typename InIter>
        AnnotatedTileDataImpl(InIter first, InIter last, value_type v, detail::DimensionOrderType o) :
            dim_(std::distance(first, last)),
            volume_(detail::volume(first, last)),
            data_(init_data_(v)),
            size_weight_(init_size_weight_(first, last)),
            order_(o)
        { }

        /// virtual destructor.
        virtual ~AnnotatedTileDataImpl() { }

        // iterator interface
        virtual iterator begin() { return data_.get(); }
        virtual const_iterator begin() const { return data_.get(); }
        virtual iterator end() { return data_.get() + volume_; }
        virtual const_iterator end() const { return data_.get() + volume_; }

        // data access interface
        virtual pointer data() { return data_.get(); }
        virtual const_pointer data() const { return data_.get(); }
        virtual bool initialized() const { return data_; }

        // tile information access
        virtual size_array size() const {
          return size_array(size_weight_.get(), size_weight_.get() + dim_);
        }
        virtual size_array weight() const {
          return size_array(size_weight_.get() + dim_, size_weight_.get() + 2 * dim_);
        }
        virtual volume_type volume() const { return volume_; }
        virtual unsigned int dim() const { return dim_; }
        virtual detail::DimensionOrderType order() const { return order_; }
        virtual bool includes(const index_type i) const { return i < volume_; }

        // Element access
        virtual reference at(const index_type i) {
          if(! includes(i))
            TA_EXCEPTION(std::out_of_range, "Index is out of range.");
          return data_[i];
        }

        virtual const_reference at(const index_type i) const {
          if(! includes(i))
            TA_EXCEPTION(std::out_of_range, "Index is out of range.");
          return data_[i];
        }
        virtual reference operator[](const index_type i) {
#ifdef NDEBUG
          return data_[i];
#else
          return at(i);
#endif
        }

        virtual const_reference operator[](const index_type i) const {
#ifdef NDEBUG
          return data_[i];
#else
          return at(i);
#endif
        }

      private:

        boost::shared_array<value_type> init_data_(value_type val) const {
          // Todo: Implement this with an aligned allocator.
          boost::shared_array<value_type> a(new value_type[volume_]);
          std::fill(a.get(), a.get() + volume_, val);
          return a;
        }

        template<typename InIter>
        boost::shared_array<value_type> init_data_(InIter first, InIter last) const {
          boost::shared_array<value_type> a(new value_type[volume_]);
          std::fill(std::copy(first, last, a.get()), a.get() + volume_, value_type());

          return a;
        }

        template<typename InIter>
        boost::shared_array<ordinal_type> init_size_weight_(InIter first, InIter last) const {
          boost::shared_array<ordinal_type> a(new ordinal_type[2 * dim_]);
          std::copy(first, last, a.get());
          detail::calc_weight(first, last, a.get() + dim_);

          return a;
        }

        unsigned int dim_;
        volume_type volume_;
        boost::shared_array<value_type> data_;
        boost::shared_array<ordinal_type> size_weight_;
        detail::DimensionOrderType order_;
      }; // class AnnotatedTileDataImpl

      /// Annotated tile.
      template<typename T>
      class AnnotatedTile {
      private:
        typedef AnnotatedTile<T> AnnotatedTile_;
        typedef AnnotatedTileImplBase<T, std::size_t> AnnotatedTileImplBase_;

      public:
        typedef typename AnnotatedTileImplBase_::value_type value_type;
        typedef typename AnnotatedTileImplBase_::index_type index_type;
        typedef typename AnnotatedTileImplBase_::ordinal_type ordinal_type;
        typedef typename AnnotatedTileImplBase_::volume_type volume_type;
        typedef typename AnnotatedTileImplBase_::reference reference;
        typedef typename AnnotatedTileImplBase_::const_reference const_reference;
        typedef typename AnnotatedTileImplBase_::pointer pointer;
        typedef typename AnnotatedTileImplBase_::const_pointer const_pointer;
        typedef typename AnnotatedTileImplBase_::size_array size_array;

        typedef typename AnnotatedTileImplBase_::iterator iterator;
        typedef typename AnnotatedTileImplBase_::const_iterator const_iterator;

      public:
        /// Default constructor
        AnnotatedTile() : tile_(), var_() { }

        /// Create an annotated tile from a tile.

        /// Construct an annotated tile from tile. The data in the annotated tile
        /// is stored in the original tile. The annotated tile does not own the
        /// data and will not free any memory.
        /// \var \c t is the tile to be annotated.
        /// \var \c var is the variable annotation.
        template<unsigned int DIM, detail::DimensionOrderType O>
        AnnotatedTile(const Tile<T, DIM, CoordinateSystem<DIM, O> >& t, const VariableList& v) :
            tile_(create_tile_ptr_(t)), var_(v)
        { }

        /// Create an annotated tile from a tile.

        /// Construct an annotated tile from tile. The data in the annotated tile
        /// is stored in the original tile. The annotated tile does not own the
        /// data and will not free any memory.
        /// \var \c t is the tile to be annotated.
        /// \var \c var is the variable annotation.
        template<unsigned int DIM, detail::DimensionOrderType O>
        AnnotatedTile(Tile<T, DIM, CoordinateSystem<DIM, O> >& t, const VariableList& v) :
            tile_(create_tile_ptr_(t)), var_(v)
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
            tile_(create_tile_ptr_(size, o, val)), var_(var)
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
            tile_(create_tile_ptr_(size, o, first, last)), var_(var)
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
        AnnotatedTile(const AnnotatedTile_& other) :
            tile_(other.tile_), var_(other.var_)
        { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
        /// Move constructor

        /// Move the data from the other annotated tile to this tile.
        /// \var \c other is the tile to be moved.
        AnnotatedTile(AnnotatedTile_&& other) :
            tile_(std::move(other.tile_)), var_(std::move(other.var_))
        { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

        ~AnnotatedTile() { }

        /// Annotated tile assignment operator.

        /// The data of the other tile will be copied into this tile. The tile
        /// dimensions, data ordering, and variable list must match, otherwise
        /// a runtime exception is thrown.
        AnnotatedTile_& operator =(const AnnotatedTile_& other) {
          if(tile_.get() == NULL) {
            tile_ = other.tile_;
          } else {
            if(tile_->data() == other.tile_->data())
              return *this;     // Do not copy yourself.

            // This is a reference to a Tile, so we need to verify that the
            // dimensions of the other tile match this tile and copy the data.
            TA_ASSERT(tile_->size() == other.tile_->size(), std::runtime_error,
                "Right-hand tile dimensions do not match the left-hand tile dimensions.");
            TA_ASSERT(tile_->order() == other.tile_->order(), std::runtime_error,
                "Tile orders do not match.");
            TA_ASSERT(this->var_ == other.var_, std::runtime_error,
                "The variable lists do not match.");
            std::copy(other.begin(), other.end(), tile_->begin());
          }

          return *this;
        }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
        /// Annotated tile move assignment operator.
        AnnotatedTile_& operator =(AnnotatedTile_&& other) {
          swap_(other);

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

        /// Make a shallow copy of \c other tile.
        AnnotatedTile_& copy(const AnnotatedTile_& other) {
          tile_ = other.tile_;
          var_ = other.var_;
          return *this;
        }

        /// Returns an iterator to the first element of the tile data.
        iterator begin() { return tile_->begin(); }
        /// Returns a const iterator to the first element of the tile data.
        const_iterator begin() const { return tile_->begin(); }
        /// Returns an iterator to the end of the data array.
        iterator end() { return tile_->end(); }
        /// Returns a const iterator to the end of the data array.
        const_iterator end() const { return tile_->end(); }

        /// Returns a pointer to the tile data.
        pointer data() { return tile_->data(); }
        /// Returns a const pointer to the tile data.
        const_pointer data() const { return tile_->data(); }
        /// Returns a constant reference to a vector with the dimension sizes.
        const size_array size() const { return tile_->size(); }
        /// Returns a constant reference to a vector with the dimension weights.
        const size_array weight() const { return tile_->weight(); }
        /// Returns the number of elements contained by the array.
        volume_type volume() const { return tile_->volume(); }
        /// Returns a constant reference to variable list (the annotation).
        const VariableList& vars() const { return var_; }
        /// Returns the number of dimensions of the array.
        unsigned int dim() const { return tile_->dim(); }
        /// Return the array storage order
        detail::DimensionOrderType order() const { return tile_->order(); }

        /// Returns true if the index \c i is included by the array.
        template<typename Index>
        bool includes(const Index& i) const { return tile_->includes(ord_(i)); }


        /// Returns a reference to element i (range checking is performed).

        /// This function provides element access to the element located at index i.
        /// If i is not included in the range of elements, std::out_of_range will be
        /// thrown. Valid types for Index are ordinal_type and index_type.
        template <typename Index>
        reference at(const Index& i) { return tile_->at(ord_(i)); }

        /// Returns a constant reference to element i (range checking is performed).

        /// This function provides element access to the element located at index i.
        /// If i is not included in the range of elements, std::out_of_range will be
        /// thrown. Valid types for Index are ordinal_type and index_type.
        template <typename Index>
        const_reference at(const Index& i) const { return tile_->at(ord_(i)); }

        /// Returns a reference to the element at i.

        /// This No error checking is performed.
        template <typename Index>
        reference operator[](const Index& i) { return (*tile_)[ord_(i)]; }

        /// Returns a constant reference to element i. No error checking is performed.
        template <typename Index>
        const_reference operator[](const Index& i) const { return (*tile_)[ord_(i)]; }

        template<unsigned int DIM>
        AnnotatedTile_& operator ^=(const Permutation<DIM>& p) {
          TA_ASSERT((dim() == DIM), std::runtime_error,
              "The permutation dimension is not equal to the tile dimensions.");

          AnnotatedTile<T> result(p ^ size(), p ^ vars(), T());
          detail::Permute<AnnotatedTile<T> > f_perm(*this);
          f_perm(p, result.begin(), result.end());

          swap_(result);

          return *this;
        }

        /// returns true if the tile data is initialized.
        bool initialized() const { return tile_->initialized(); }

      private:

        template<typename tileT>
        static boost::shared_ptr<AnnotatedTileImplBase_> create_tile_ptr_(tileT& t) {
          return boost::dynamic_pointer_cast<AnnotatedTileImplBase_>(boost::make_shared<AnnotatedTileRefImpl<tileT> >(t));
        }

        template<typename SizeArray>
        static boost::shared_ptr<AnnotatedTileImplBase_> create_tile_ptr_(const SizeArray& s, const detail::DimensionOrderType o, value_type v) {
          return boost::dynamic_pointer_cast<AnnotatedTileImplBase_>(
              boost::make_shared<AnnotatedTileDataImpl<value_type, index_type> >(s.begin(), s.end(), v, o));
        }

        template<typename SizeArray, typename InIter>
        static boost::shared_ptr<AnnotatedTileImplBase_> create_tile_ptr_(const SizeArray& s, const detail::DimensionOrderType o, InIter first, InIter last) {
          return boost::dynamic_pointer_cast<AnnotatedTileImplBase_>(
              boost::make_shared<AnnotatedTileDataImpl<value_type, index_type> >(s.begin(), s.end(), first, last, o));
        }

        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        ordinal_type ord_(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
          return std::inner_product(i.begin(), i.end(), tile_->weight().begin(), ordinal_type(0));
        }

        ordinal_type ord_(ordinal_type i) const {
          return i;
        }


        void swap_(AnnotatedTile_& other) {
          boost::swap(tile_, other.tile_);
          expressions::swap(var_, other.var_);
        }

        friend void swap<>(AnnotatedTile_&, AnnotatedTile_&);
        template <class, typename>
        friend struct madness::archive::ArchiveStoreImpl;
        template <class, typename>
        friend struct madness::archive::ArchiveLoadImpl;
        template <typename, unsigned int, typename, typename>
        friend class Array;

        boost::shared_ptr<AnnotatedTileImplBase_> tile_; ///< Base pointer to tile holder.
        VariableList var_;                        ///< variable list

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
      static void load(const Archive& ar, atile_type& t) {
        unsigned int dim = 0;
        typename atile_type::volume_type vol = 0;
        ar & dim & vol;
        std::vector<typename atile_type::index_type> size(dim);
        std::vector<typename atile_type::value_type> data(vol);
        TiledArray::detail::DimensionOrderType order;
        TiledArray::expressions::VariableList var;
        ar & order & wrap(size.data(), dim) & wrap(data.data(), vol) & var;

        t.tile_.reset();
        t.tile_ = atile_type::create_tile_ptr_(size, order, data.begin(), data.end());
        t.var_ = var;
      }
    }; // struct ArchiveLoadImpl<Archive, TiledArray::expression::tile::AnnotatedTile<T> >

    /// Provides output archiving capability for AnnotatedTiles.
    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, TiledArray::expressions::tile::AnnotatedTile<T> > {
      typedef TiledArray::expressions::tile::AnnotatedTile<T> atile_type;

      /// Stores an AnnotatedTile to an archive
      static void store(const Archive& ar, const atile_type& t) {
        ar & t.dim() & t.volume() & t.order() & wrap(t.size().begin(), t.dim())
            & wrap(t.begin(), t.volume()) & t.vars();
      }
    }; // struct ArchiveStoreImpl<Archive, TiledArray::expression::tile::AnnotatedTile<T> >

  } // namespace archive
} // namespace madness

#endif // TILEDARRAY_ANNOTATED_TILE_H__INCLUDED
