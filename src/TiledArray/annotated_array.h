#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <TiledArray/annotation.h>
#include <TiledArray/array_ref.h>
#include <TiledArray/transform_iterator.h>
#include <TiledArray/madness_runtime.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/type_traits.hpp>

namespace TiledArray {

  template<typename T, typename I>
  class BaseArray;
  template<typename T, unsigned int DIM, typename CS, typename C>
  class Array;

  namespace detail {
    template<typename T>
    class array_tile;
  } // namespace detail

  namespace expressions {
    namespace array {

      template<typename T>
      class AnnotatedTile;

      template<typename Exp0, typename Exp1, typename Op>
      class BinaryArrayExp;
      template<typename Exp, typename Op>
      class UnaryArrayExp;
      template<typename T>
      class AnnotatedArray;
      template<typename T>
      class AnnotatedTile;
      template<typename T>
      void swap(AnnotatedArray<T>&, AnnotatedArray<T>&);



      template<typename T, typename I >
      class ArrayHolderBase {
      public:
        typedef ArrayHolderBase<T, I> ArrayHolderBase_;
        typedef I index_type;
        typedef I ordinal_type;
        typedef I volume_type;
        typedef tile::AnnotatedTile<T> tile_type;
        typedef std::pair<const index_type, madness::Future<tile_type> > value_type;
        typedef detail::ArrayRef<const index_type> size_array;
        typedef detail::RangeData<I> range_type;

      public:
        // Iterator typedefs
        typedef detail::PolyTransformIterator<value_type> iterator;
        typedef detail::PolyTransformIterator<const value_type> const_iterator;

        // Clone the array
        virtual boost::shared_ptr<ArrayHolderBase_> clone(madness::World&, bool copy_data = false) const = 0;
        virtual boost::shared_ptr<ArrayHolderBase_> clone(bool copy_data = false) const = 0;

        // Iterators which return futures to annotated tiles.
        virtual iterator begin(const VariableList&) = 0;
        virtual const_iterator begin(const VariableList&) const = 0;
        virtual iterator end(const VariableList&) = 0;
        virtual const_iterator end(const VariableList&) const = 0;

        // Basic array modification interface.
        virtual void insert(const index_type, const madness::Future<tile_type>&, const VariableList&) = 0;
        virtual void erase(const index_type) = 0;
        virtual void clear() = 0;

        // Returns information on the array tiles.
        virtual bool is_local(const index_type) const = 0;
        virtual bool includes(const index_type) const = 0;
        virtual size_array size() const = 0;
        virtual size_array weight() const = 0;
        virtual volume_type volume(bool local = false) const = 0;
        virtual unsigned int dim() const = 0;
        virtual detail::DimensionOrderType order() const = 0;
        virtual range_type range() const = 0;

        // Remote communication
        virtual iterator find(const ordinal_type, const expressions::VariableList&) = 0;
        virtual const_iterator find(const ordinal_type, const expressions::VariableList&) const = 0;

        // public access functions.
        virtual madness::World& get_world() const = 0;


      }; // class ArrayHolderBase

      /// Holds a pointer to an Array object.

      /// This class implements the interface required by an AnnotatedArray.
      template<typename A>
      class ArrayHolder : public ArrayHolderBase<typename detail::array_tile<typename A::tile_type>::value_type, typename A::ordinal_type> {
      public:
        typedef A array_type;
        typedef ArrayHolder<A> ArrayHolder_;

      private:
        typedef typename detail::array_tile<typename A::tile_type>::value_type tile_value_type;

      public:
        typedef ArrayHolderBase<tile_value_type, typename A::ordinal_type> ArrayHolderBase_;

        typedef typename ArrayHolderBase_::value_type value_type;
        typedef typename ArrayHolderBase_::index_type index_type;
        typedef typename ArrayHolderBase_::ordinal_type ordinal_type;
        typedef typename ArrayHolderBase_::volume_type volume_type;
        typedef typename ArrayHolderBase_::size_array size_array;
        typedef typename ArrayHolderBase_::range_type range_type;
        typedef typename ArrayHolderBase_::tile_type tile_type;

      public:
        // Iterator typedefs
        typedef typename ArrayHolderBase_::iterator iterator;
        typedef typename ArrayHolderBase_::const_iterator const_iterator;

      private:

        /// This class is used to convert tiles from the base array to future annotated tiles.
        template<typename Arg, typename Res>
        class MakeFutATile {
        private:
          MakeFutATile();
        public:
          typedef typename boost::call_traits<Res>::value_type result_type;
          typedef typename boost::call_traits<Arg>::param_type argument_type;

          /// Constructor
          MakeFutATile(const expressions::VariableList& var) : var_(var) { }

          /// Converts a from type argument_type to result type.
          result_type operator()(argument_type a) const {
            return result_type(a.first.key1(), make_fut_annotation(a.second));
          }

        private:
          template<typename U>
          madness::Future<tile_type> make_fut_annotation(const madness::Future<U>& f) const {
            return make_fut_annotation(f.get());
          }

          template<typename U>
          madness::Future<tile_type> make_fut_annotation(madness::Future<U>& f) const {
            return make_fut_annotation(f.get());
          }

          template<unsigned int DIM, typename CS>
          madness::Future<tile_type> make_fut_annotation(const Tile<tile_value_type, DIM, CS>& t) const {
            return madness::Future<tile_type>(t(var_));
          }

          template<unsigned int DIM, typename CS>
          madness::Future<tile_type> make_fut_annotation(Tile<tile_value_type, DIM, CS>& t) const {
            return madness::Future<tile_type>(t(var_));
          }

          madness::Future<tile_type> make_fut_annotation(const tile::AnnotatedTile<tile_value_type>& t) const {
            return madness::Future<tile_type>(t);
          }

          madness::Future<tile_type> make_fut_annotation(tile::AnnotatedTile<tile_value_type>& t) const {
            return madness::Future<tile_type>(t);
          }

          const expressions::VariableList& var_;
        }; // struct MakeFutATile

      public:

        /// Constructor
        ArrayHolder(boost::shared_ptr<array_type> a) : array_(a) { }

        /// virtual destructor
        virtual ~ArrayHolder() { }

        /// Clone the array

        /// \var \c w is a madness world reference.
        /// \var \c copy_data (optional), if true, the data of the original array will be
        /// copied to the clone. The default value is false.
        virtual boost::shared_ptr<ArrayHolderBase_> clone(madness::World& w, bool copy_data = false) const {
          boost::shared_ptr<array_type> array = boost::make_shared<array_type>(w, array_->range());

          if(copy_data) {
            for(typename array_type::const_iterator it = array_->begin(); it != array_->end(); ++it)
              array->insert(*it);
          }

          return boost::dynamic_pointer_cast<ArrayHolderBase_>(boost::make_shared<ArrayHolder_>(array));
        }

        /// Clone the array

        /// \var \c copy_data (optional), if true, the data of the original array will be
        virtual boost::shared_ptr<ArrayHolderBase_> clone(bool copy_data = false) const {
          return clone(array_->get_world(), copy_data);
        }

        /// Returns an iterator

        /// The iterators points to the first local tile and dereference to a
        /// pair where the first is the ordinal index of the tile and second is
        /// a future of an annotated tiles.
        /// \var \c v is the variable list given to the annotated tiles.
        virtual iterator begin(const VariableList& v) {
          return iterator(array_->begin(),
              MakeFutATile<typename array_type::iterator::reference,
              typename iterator::value_type>(v));
        }

        /// Returns a const_iterator

        /// The iterators points to the first local tile and dereference to a
        /// pair where the first is the ordinal index of the tile and second is
        /// a future of an annotated tiles.
        /// \var \c v is the variable list given to the annotated tiles.
        virtual const_iterator begin(const VariableList& v) const {
          return const_iterator(array_->begin(),
              MakeFutATile<typename array_type::const_iterator::reference,
              const typename const_iterator::value_type>(v));
        }

        /// Returns an iterator

        /// The iterators points to the end of the local tiles.
        /// \var \c v is the variable list given to the annotated tiles.
        virtual iterator end(const VariableList& v) {
          return iterator(array_->end(),
              MakeFutATile<typename array_type::iterator::reference,
              typename iterator::value_type>(v));
        }

        /// Returns an iterator

        /// The iterators points to the end of the local tiles.
        /// \var \c v is the variable list given to the annotated tiles.
        virtual const_iterator end(const VariableList& v) const {
          return const_iterator(array_->end(),
              MakeFutATile<typename array_type::const_iterator::reference,
              const typename const_iterator::value_type>(v));
        }

        /// Insert a tile at index i.

        /// Inserts a tile a the given index. The tile will contain the data given
        /// by the array reference, d.
        /// \var \c i is the ordinal index where the tile will be inserted.
        /// \var \c t will be copied into the destination.
        virtual void insert(const index_type i, const madness::Future<tile_type>& t, const VariableList& v) {
          ArrayInserter<typename array_type::tile_type>::insert(array_, i, t, v);
        }

        /// Erase the tile at the given index, i.
        virtual void erase(const index_type i) { array_->erase(i); }

        /// Erase all local tiles.
        virtual void clear() { array_->clear(); }

        /// Returns true if the given index is stored locally.
        virtual bool is_local(const index_type i) const { return array_->is_local(i); }

        /// Returns true if the tile is included in the array range.
        virtual bool includes(const index_type i) const { return array_->includes(i); }

        /// Returns the size of each dimension of the array.
        virtual size_array size() const { return array_->size(); }

        /// Returns the weights of the array dimensions.
        virtual size_array weight() const { return array_->weight(); }

        /// Returns the number of tiles in the array.

        /// volume will return the number of tiles in the array if local is false
        /// and will return the number of tiles that are stored locally. Note:
        /// This does not include tiles that may be stored locally but are not
        /// present.
        virtual volume_type volume(bool local = false) const { return array_->volume(local); }

        /// Returns the number of dimensions of the array.
        virtual unsigned int dim() const { return array_type::dim; }

        /// Returns the dimension ordering of the array.
        virtual detail::DimensionOrderType order() const { return array_type::order; }

        /// Return the range data for the array.
        virtual range_type range() const { return array_->range().range_data(); }

        /// Returns an iterator to the element at i.

        /// This function will return an iterator to the tile at the given index.
        /// If the tile is stored remotely
        virtual iterator find(const ordinal_type i, const expressions::VariableList& v) {
          madness::Future<typename array_type::iterator> fut_it = array_->find(i);
          // Todo: We need to remove the call to get() and find a way around
          // the fact that find() returns a future, probably should involve tasks.
          return iterator(fut_it.get(),
              MakeFutATile<typename array_type::iterator::reference,
              typename iterator::value_type>(v));
        }


        virtual const_iterator find(const ordinal_type i, const expressions::VariableList& v) const {
          madness::Future<typename array_type::iterator> fut_it = array_->find(i);
          // Todo: We need to remove the call to get() and find a way around
          // the fact that find() returns a future, probably should involve a
          // call back.
          return const_iterator(fut_it.get(),
              MakeFutATile<typename array_type::const_iterator::reference,
              const typename const_iterator::value_type>(v));
        }


        // public access functions.
        virtual madness::World& get_world() const { return array_->world; }

        /// Clean-up function for an array reference.

        /// This function is the clean-up operation for an array reference. Its
        /// purpose is to prevent boost shared pointer from calling delete on an
        /// array pointer that is not dynamically allocated.
        static void no_delete(array_type*) { /* do nothing */ }

      private:

        template<typename U>
        struct ArrayInserter;

        template<unsigned int DIM, typename CS>
        struct ArrayInserter<Tile<tile_value_type, DIM, CS> > {
          static void insert(boost::shared_ptr<array_type>& a, const index_type i,
              const madness::Future<tile_type>& t, const VariableList&)
          {
            a->insert(i, Tile<tile_value_type, DIM, CS>(t.get()));
          }
        }; // struct ArrayInserter

        template<typename U>
        struct ArrayInserter<tile::AnnotatedTile<U> > {
          static void insert(boost::shared_ptr<array_type>& a, const index_type i,
              const madness::Future<tile_type>& t, const VariableList& v)
          {
            a->insert(i, t.get()(v));
          }
        }; // struct ArrayInserter

        template<unsigned int DIM, typename CS>
        struct ArrayInserter<madness::Future<Tile<tile_value_type, DIM, CS> > > {
          static void insert(boost::shared_ptr<array_type>& a, const index_type i,
              const madness::Future<tile_type>& t, const VariableList& v)
          {
            a->insert(i, madness::Future<Tile<tile_value_type, DIM, CS> >(Tile<tile_value_type, DIM, CS>(t.get())));
          }
        }; // struct ArrayInserter<madness::Future<Tile<tile_value_type, DIM, CS> > >

        template<typename U>
        struct ArrayInserter<madness::Future<tile::AnnotatedTile<U> > > {
          static void insert(boost::shared_ptr<array_type>& a, const index_type i,
              const madness::Future<tile_type>& t, const VariableList&)
          {
            a->insert(i, t);
          }
        }; // struct ArrayInserter<madness::Future<Tile<tile_value_type, DIM, CS> > >

        boost::shared_ptr<array_type> array_; ///< Shared pointer to the array.

      }; // class ArrayHolder



      /// Annotated Array implementation class
      template<typename T>
      class AnnotatedArray {
      public:
        typedef AnnotatedArray<T> AnnotatedArray_;
        typedef tile::AnnotatedTile<T> tile_type;
        typedef std::size_t ordinal_type;
        typedef ordinal_type index_type;

      private:
        typedef ArrayHolderBase<T, index_type> ArrayHolderBase_;

      public:
        typedef typename ArrayHolderBase_::volume_type volume_type;
        typedef typename ArrayHolderBase_::value_type value_type;
        typedef typename ArrayHolderBase_::size_array size_array;
        typedef typename ArrayHolderBase_::range_type range_type;
        typedef typename ArrayHolderBase_::iterator iterator;
        typedef typename ArrayHolderBase_::const_iterator const_iterator;

      private:
        /// Default constructor
        AnnotatedArray();

      public:

        /// creates an array living in world and described by shape. Optional
        /// val specifies the default value of every element
        template<unsigned int DIM, typename CS, typename C>
        AnnotatedArray(Array<T, DIM, CS, C>& a, const VariableList& v) :
            array_(create_array_ptr_(a)), var_(v)
        { }

        /// creates an array living in world and described by shape. Optional
        /// val specifies the default value of every element
        template<unsigned int DIM, typename CS, typename C>
        AnnotatedArray(const Array<T, DIM, CS, C>& a, const VariableList& v) :
            array_(create_array_ptr_(a)), var_(v)
        { }


        template<typename R>
        AnnotatedArray(madness::World& w, const R& r, const VariableList& v,
            detail::DimensionOrderType o = detail::decreasing_dimension_order) :
            array_(create_array_ptr_(w, r, o)), var_(v)
        { }

        /// Copy constructor
        AnnotatedArray(const AnnotatedArray_& other) :
            array_(other.array_), var_(other.var_)
        { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
        /// Move constructor
        AnnotatedArray(AnnotatedArray_&& other) :
            array_(std::move(other.array_)), var_(std::move(other.var_))
        { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

        /// Destructor
        ~AnnotatedArray() { }

        /// Copy assignment operator
        AnnotatedArray_ operator=(const AnnotatedArray_& other) {
          array_ = other.array_;
          var_ = other.var_;

          return *this;
        }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
        /// Move assignment operator
        AnnotatedArray_ operator=(AnnotatedArray_&& other) {
          // Move other data to this object
          array_ = std::move(other.array_);
          var_ = std::move(other.var_);

          return *this;
        }
#endif // __GXX_EXPERIMENTAL_CXX0X__

        /// AnnotatedTile assignment operator
        template<typename Exp0, typename Exp1, typename Op>
        AnnotatedArray_& operator =(const BinaryArrayExp<Exp0, Exp1, Op>& e) {
          *this = e.eval();
          return *this;
        }

        /// AnnotatedTile assignment operator
        template<typename Exp, typename Op>
        AnnotatedArray_& operator =(const UnaryArrayExp<Exp, Op>& e) {
          *this = e.eval();
          return *this;
        }

        /// Clone this object
        AnnotatedArray_ clone(madness::World& w, bool copy_data = true) const {
          AnnotatedArray_ result;
          result.array_ = array_->clone(w, copy_data);
          result.var_ = var_;

          return result;
        }

        /// Clone this object
        AnnotatedArray_ clone(bool copy_data = true) const {
          AnnotatedArray_ result;
          result.array_ = array_->clone(copy_data);
          result.var_ = var_;

          return result;
        }

        /// Returns an iterator to an AnnotatedTile object.

        /// This function will return an AnnotatedTile object, which points to
        /// the tiles contained by the AnnotatedArray.
        iterator begin() { return array_->begin(var_); }
        const_iterator begin() const { return const_cast<const ArrayHolderBase_*>(array_.get())->begin(var_); }
        iterator end() { return array_->end(var_); }
        const_iterator end() const { return const_cast<const ArrayHolderBase_*>(array_.get())->end(var_); }

        /// Return a size array with the size for annotated array.
        size_array size() const { return array_->size(); }
        /// Return a size array with the weight for annotated array.
        size_array weight() const { return array_->weight(); }
        /// Return the number of tiles in the annotated array.
        volume_type volume(bool local = false) const { return array_->volume(local); }
        /// Returns a constant reference to variable list (the annotation).
        const VariableList& vars() const { return var_; }
        /// Returns the number of dimensions of the array.
        unsigned int dim() const { return array_->dim(); }
        /// Return the array storage order
        detail::DimensionOrderType order() const { return array_->order(); }
        /// Return the tiled range data for the array.
        range_type range() const { return array_->range(); }
        /// Return a future to an AnnotatedTile Object.
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        iterator find(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) {
          return array_->find(ord_(i), var_);
        }
        /// Returns an iterator to the element at index, i.
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        const_iterator find(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
          return array_->find(ord_(i), var_);
        }
        /// Returns an iterator to the tile at the given index, i.
        iterator find(const ordinal_type i) {
          return array_->find(i, var_);
        }
        /// Returns an iterator to the tile at the given index, i.
        const_iterator find(const ordinal_type i) const {
          return const_cast<const ArrayHolderBase_*>(array_.get())->find(i, var_);
        }

        /// Inserts a tile into the array.

        /// Copies the given tile into the array. Non-local insertions will initiate
        /// non-blocking communication.
        template<typename Index, typename Tile>
        void insert(const Index i, const Tile& t) {
          array_->insert(ord_(i), madness::Future<Tile>(t), var_);
        }

        /// Inserts a tile into the array.

        /// Copies the given tile into the array. Non-local insertions will initiate
        /// non-blocking communication.
        template<typename Index, typename Tile>
        void insert(const Index i, const madness::Future<Tile>& t) {
          array_->insert(ord_(i), t, var_);
        }

        /// Inserts a tile into the array.

        /// Copies the given value_type into the array. Non-local insertions will
        /// initiate non-blocking communication.
        template<typename Index, typename Tile>
        void insert(const std::pair<const Index, Tile>& v) {
          insert(v.first, v.second);
        }

        /// Erases a tile from the array.

        /// This will remove the tile at the given index. It will initiate
        /// non-blocking for non-local tiles.
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        void erase(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) {
          TA_ASSERT(this->dim() == DIM, std::runtime_error,
              "The index dimensions is not equal to the array dimensions.");
          array_.erase(this->ord_(i));
        }

        /// Erase a range of tiles from the array.

        /// This will remove the range of tiles from the array. The iterator must
        /// dereference to value_type (std::pair<index_type, tile_type>). It will
        /// initiate non-blocking communication for non-local tiles.
        template<typename InIter>
        void erase(InIter first, InIter last) {
          BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
          for(; first != last; ++first);
            array_.erase(this->ord_(first->first));
        }

        /// Removes all tiles from the array.
        void clear() {
          array_->clear();
        }

        /// Permutes the array. This will initiate blocking communication.
        template<unsigned int DIM>
        AnnotatedArray_& operator ^=(const Permutation<DIM>& p) {
          TA_ASSERT(this->dim() == DIM, std::runtime_error,
              "The permutation dimensions is not equal to the array dimensions.");
          var_ ^= p;
          array_ ^= p; // move the tiles to the correct location. Blocking communication here.

          return *this;
        }

        /// Returns true if the tile specified by index is stored locally.
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        bool is_local(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
          TA_ASSERT(this->dim() == DIM, std::runtime_error,
              "The index dimensions is not equal to the array dimensions.");
          return array_.is_local(this->ord_(i));
        }

        /// Return a reference to the world object of the array.
        madness::World& get_world() const { return array_->get_world(); }

      private:

        template<typename arrayT>
        static boost::shared_ptr<ArrayHolderBase_> create_array_ptr_(const boost::shared_ptr<arrayT>& a) {
          return boost::dynamic_pointer_cast<ArrayHolderBase_>(boost::make_shared<ArrayHolder<arrayT> >(a));
        }

        template<unsigned int DIM, detail::DimensionOrderType O, typename C>
        static boost::shared_ptr<ArrayHolderBase_> create_array_ptr_(Array<T, DIM, CoordinateSystem<DIM, O>, C>& a) {
          typedef Array<T, DIM, CoordinateSystem<DIM, O>, C> arrayT;
          boost::shared_ptr<arrayT> array(&a, &ArrayHolder<arrayT>::no_delete);
          return create_array_ptr_(array);
        }

        template<unsigned int DIM, detail::DimensionOrderType O, typename C>
        static boost::shared_ptr<ArrayHolderBase_> create_array_ptr_(const Array<T, DIM, CoordinateSystem<DIM, O>, C>& a) {
          typedef const Array<T, DIM, CoordinateSystem<DIM, O>, C> arrayT;
          boost::shared_ptr<arrayT> array(&a, &ArrayHolder<arrayT>::no_delete);
          return create_array_ptr_(array);
        }

        template<typename TRange>
        static boost::shared_ptr<ArrayHolderBase_> create_array_ptr_(madness::World& w, const TRange& r, const detail::DimensionOrderType o) {
          if(o == detail::increasing_dimension_order)
            return MakeArray<1, detail::increasing_dimension_order,
                madness::Future<tile::AnnotatedTile<T> > >::make(w, r, r.dim);
          else
            return MakeArray<1, detail::decreasing_dimension_order,
            madness::Future<tile::AnnotatedTile<T> > >::make(w, r, r.dim);
        }

        template<unsigned int DIM, detail::DimensionOrderType O, typename C>
        struct MakeArray {
          typedef Array<T, DIM, CoordinateSystem<DIM, O>, C> arrayT;

          static const unsigned int dim = DIM;
          static const detail::DimensionOrderType order = O;

          template<typename R>
          static boost::shared_ptr<ArrayHolderBase_> make(madness::World& w, const R& r, const unsigned int d) {
            if(d != DIM)
              return MakeArray<DIM + 1, detail::decreasing_dimension_order, C>::make(w, r, d);

            // create a tile.
            typename arrayT::tiled_range_type range(r);
            boost::shared_ptr<arrayT> array = boost::make_shared<arrayT>(w, range);

            return create_array_ptr_(array);
          }

        };

        template<detail::DimensionOrderType O, typename C>
        struct MakeArray<TA_MAX_DIM, O, C> {
          typedef Array<T, TA_MAX_DIM, CoordinateSystem<TA_MAX_DIM, O>, C> arrayT;

          static const unsigned int dim = TA_MAX_DIM;
          static const detail::DimensionOrderType order = O;

          template<typename TRange>
          static boost::shared_ptr<ArrayHolderBase_> make(madness::World&,const TRange&, const unsigned int) {
            TA_EXCEPTION(std::runtime_error,
                "The maximum number of dimensions was exceeded. Rerun configure and specify a larger number of dimensions.");

            return boost::shared_ptr<ArrayHolderBase_>();
          }

          template<typename TRange>
          static boost::shared_ptr<ArrayHolderBase_> make(const TRange&, const unsigned int) {
            TA_EXCEPTION(std::runtime_error,
                "The maximum number of dimensions was exceeded. Rerun configure and specify a larger number of dimensions.");

            return boost::shared_ptr<ArrayHolderBase_>();
          }
        };

        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        ordinal_type ord_(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) {
          return std::inner_product(i.begin(), i.end(), array_->weight().begin(), ordinal_type(0));
        }

        ordinal_type ord_(const index_type i) const { return i; }

        /// Private swap function.
        void swap_(AnnotatedArray_& other) { // no throw
          boost::swap(array_, other.array_);
          expressions::swap(var_, other.var_);
        }

        friend void swap<>(AnnotatedArray_&, AnnotatedArray_&);

        boost::shared_ptr<ArrayHolderBase_> array_; ///< shared pointer to the array referenced by the annotation.
        VariableList var_; ///< The variable list of the annotation.
      }; // class AnnotatedArray

      /// Swap two Annotated arrays.
      template<typename T>
      void swap(AnnotatedArray<T>& a0, AnnotatedArray<T>& a1) { // no throw
        a0.swap_(a1);
      }



    } // namespace array
  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
