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

      private:
        // Annotated tile typedefs
        typedef std::pair<const ordinal_type, madness::Future<tile::AnnotatedTile<value_type> > > data_type;


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
        virtual void insert(const index_type, const value_type*, const value_type*) = 0;
        virtual void insert(const index_type, const detail::ArrayRef<const value_type>&) = 0;
        virtual void insert(const index_type, const tile_type&) = 0;
        virtual void erase(const index_type) = 0;
        virtual void clear() = 0;

        // Returns information on the array tiles.
        virtual bool is_local(const index_type) const = 0;
        virtual bool includes(const index_type) const = 0;
        virtual size_array size() const = 0;
        virtual size_array weight() const = 0;
        virtual volume_type volume(bool local = false) const = 0;

        // Remote communication
        virtual iterator find(const ordinal_type, const expressions::VariableList&) = 0;
        virtual const_iterator find(const ordinal_type, const expressions::VariableList&) const = 0;

        // public access functions.
        virtual madness::World& get_world() const = 0;


      }; // class ArrayHolderBase

      /// Holds a pointer to an Array object.

      /// This class implements the interface required by an AnnotatedArray.
      template<typename A, typename T, typename I>
      class ArrayHolder : public ArrayHolderBase<T, I> {
      public:
        typedef A array_type;
        typedef ArrayHolder<A, T, I> ArrayHolder_;
        typedef ArrayHolderBase<T, I> ArrayHolderBase_;

        typedef typename ArrayHolderBase_::value_type value_type;
        typedef typename ArrayHolderBase_::index_type index_type;
        typedef typename ArrayHolderBase_::ordinal_type ordinal_type;
        typedef typename ArrayHolderBase_::volume_type volume_type;
        typedef typename ArrayHolderBase_::size_array size_array;
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
            return result_type(a.first.key1(), typename result_type::second_type(a.second(var_)));
          }

        private:
          const expressions::VariableList& var_;
        }; // struct MakeFutATile

      public:

        /// Constructor
        ArrayHolder(array_type* a) : array_(a) { }

        /// Clone the array

        /// \var \c w is a madness world reference.
        /// \var \c copy_data (optional), if true, the data of the original array will be
        /// copied to the clone. The default value is false.
        virtual boost::shared_ptr<ArrayHolderBase_> clone(madness::World& w, bool copy_data = false) const {
          boost::shared_ptr<ArrayHolderBase_> result(dynamic_cast<ArrayHolderBase_*>(new array_type(w, array_->range())));
          if(copy_data) {
            array_type* r = dynamic_cast<array_type*>(result.get());
            for(typename array_type::const_iterator it = array_->begin(); it != array_->end(); ++it)
              r->insert(*it);
          }

          return result;
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
        /// by the pointers [first, last).
        /// \var \c i is the ordinal index where the tile will be inserted.
        /// \var \c [first, \c last) is the data that will be placed in the tile.
        virtual void insert(const index_type i, const value_type* first, const value_type* last) {
          array_->insert(i, first, last);
        }

        /// Insert a tile at index i.

        /// Inserts a tile a the given index. The tile will contain the data given
        /// by the array reference, d.
        /// \var \c i is the ordinal index where the tile will be inserted.
        /// \var \c d is the data that will be placed in the tile.
        virtual void insert(const index_type i, const detail::ArrayRef<const value_type>& d) {
          array_->insert(i, d.begin(), d.end());
        }

        /// Insert a tile at index i.

        /// Inserts a tile a the given index. The tile will contain the data given
        /// by the array reference, d.
        /// \var \c i is the ordinal index where the tile will be inserted.
        /// \var \c t will be copied into the destination.
        virtual void insert(const index_type i, const tile_type& t) {
          array_->insert(i, t.begin(), t.end());
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
          // the fact that find() returns a future, probably should involve tasks.
          return const_iterator(fut_it.get(),
              MakeFutATile<typename array_type::const_iterator::reference,
              const typename const_iterator::value_type>(v));
        }


        // public access functions.
        virtual madness::World& get_world() const { return array_->world; }

      private:
        array_type* array_;

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
        typedef typename ArrayHolderBase_::iterator iterator;
        typedef typename ArrayHolderBase_::const_iterator const_iterator;

        /// Default constructor
        AnnotatedArray() : array_(), var_() { }

        /// creates an array living in world and described by shape. Optional
        /// val specifies the default value of every element
        template<typename A>
        AnnotatedArray(A* a, const VariableList& v) :
            array_(dynamic_cast<ArrayHolderBase_*>(new ArrayHolder<A, T, index_type>(a))), var_(v)
        { }

        template<typename Range>
        AnnotatedArray(madness::World& w, const Range& r, const VariableList& v,
            detail::DimensionOrderType o = detail::decreasing_dimension_order) :
            array_(create_array_(w, r, o)), var_()
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
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O, typename Tile>
        void insert(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i, const Tile& t) {
          TA_ASSERT(this->dim() == DIM, std::runtime_error,
              "The index dimensions is not equal to the array dimensions.");
          array_->insert(this->ord_(i), t);
        }

        /// Inserts a tile into the array.

        /// Copies the given value_type into the array. Non-local insertions will
        /// initiate non-blocking communication.
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O, typename Tile>
        void insert(const std::pair<const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >, Tile>& v) {
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

        template<typename Range>
        static boost::shared_ptr<ArrayHolderBase_> create_array_(madness::World& w, const Range& r, const detail::DimensionOrderType o) {
          if(o == detail::increasing_dimension_order)
            return create_array_<0, detail::increasing_dimension_order>(w, r, r.dim());
          else
            return create_array_<0, detail::decreasing_dimension_order>(w, r, r.dim());
        }

        template<typename Range, unsigned int DIM, detail::DimensionOrderType O>
        static boost::shared_ptr<ArrayHolderBase_> create_array_(madness::World& w, const Range& r, const unsigned int d) {
          typedef Array<value_type, DIM, LevelTag<1>, CoordinateSystem<DIM, O> > array_type;
          if(d == DIM)
            return boost::shared_ptr<ArrayHolderBase_>(new array_type(w, r));
          else
            return create_array_<DIM + 1, detail::decreasing_dimension_order>(w, r, d);
        }

        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        ordinal_type ord_(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) {
          return std::inner_product(i.begin(), i.end(), array_->weight().begin(), ordinal_type(0));
        }

        /// Private swap function.
        void swap_(AnnotatedArray_& other) { // no throw
          boost::swap(array_, other.array_);
          expressions::swap(var_, other.var_);
        }

        /// Clean-up function for an array reference.

        /// This function is the clean-up operation for an array reference. Its
        /// purpose is to prevent boost shared pointer from calling delete on an
        /// array pointer that is not dynamically allocated.
        static void no_delete(ArrayHolderBase_*) { /* do nothing */ }

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
