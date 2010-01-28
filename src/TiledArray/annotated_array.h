#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <TiledArray/annotation.h>
#include <TiledArray/madness_runtime.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/type_traits.hpp>

namespace TiledArray {

  template<typename T>
  class BaseArray;
  template<typename T, unsigned int DIM, typename CS>
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

      /// Annotated Array implementation class
      template<typename T>
      class AnnotatedArrayImpl : public madness::WorldObject< AnnotatedArrayImpl<T> >,
          Annotation<typename detail::add_const<boost::is_const<T>::value, std::size_t >::type> {
      public:
        typedef Annotation<std::size_t> Annotation_;
        typedef AnnotatedArrayImpl<T> AnnotatedArrayImpl_;
        typedef typename detail::add_const<boost::is_const<T>::value, std::size_t >::type index;
        typedef typename boost::remove_const<T>::type value_type;
        typedef typename Annotation_::ordinal_type ordinal_type;
        typedef typename Annotation_::volume_type volume_type;
        typedef typename Annotation_::size_array size_array;
        typedef typename detail::add_const<boost::is_const<T>::value, BaseArray<value_type>* >::type array_ptr;

      private:
        // Prohibited operations
        AnnotatedArrayImpl();
        AnnotatedArrayImpl(const AnnotatedArrayImpl_&);
        AnnotatedArrayImpl_ operator=(const AnnotatedArrayImpl_&);

      public:
        /// creates an array living in world and described by shape. Optional
        /// val specifies the default value of every element
        AnnotatedArrayImpl(array_ptr a, const VariableList& v, detail::DimensionOrderType o) :
            madness::WorldObject<AnnotatedArrayImpl_>(a->get_world()),
            Annotation_(const_cast<index*>(a->size_pair().first), const_cast<index*>(a->size_pair().second),
            const_cast<index*>(a->weight_pair().first), const_cast<index*>(a->weight_pair().second),
            a->volume(), v, o), array_(a)
        {
          this->process_pending();
          this->get_world().gop.fence();
        }

        /// AnnotatedTile assignment operator
        template<typename Exp0, typename Exp1, typename Op>
        AnnotatedArrayImpl_& operator =(const BinaryArrayExp<Exp0, Exp1, Op>& /*e*/) {
          TA_ASSERT(false, std::runtime_error, "Not yet implemented.");
          return *this;
        }

        /// AnnotatedTile assignment operator
        template<typename Exp, typename Op>
        AnnotatedArrayImpl_& operator =(const UnaryArrayExp<Exp, Op>& /*e*/) {
          TA_ASSERT(false, std::runtime_error, "Not yet implemented.");
          return *this;
        }

        /// Returns an iterator to an AnnotatedTile object.

        /// This function will return an AnnotatedTile object, which points to
        /// the tiles contained by the AnnotatedArray.


        /// Inserts a tile into the array.

        /// Copies the given tile into the array. Non-local insertions will initiate
        /// non-blocking communication.
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O, typename Tile>
        void insert(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i, const Tile& t) {
          TA_ASSERT(this->dim() == DIM, std::runtime_error,
              "The index dimensions is not equal to the array dimensions.");
          array_.insert(this->ord_(i), t);
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
          array_.clear();
        }

        /// Permutes the array. This will initiate blocking communication.
        template<unsigned int DIM>
        AnnotatedArrayImpl_& operator ^=(const Permutation<DIM>& p) {
          TA_ASSERT(this->dim() == DIM, std::runtime_error,
              "The permutation dimensions is not equal to the array dimensions.");
          Annotation_::operator^=(p);
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

      private:

        array_ptr array_;
      }; // class AnnotatedArrayImpl

      /// Annotated array

      template<typename T>
      class AnnotatedArray {
      public:
        typedef AnnotatedArray<T> AnnotatedArray_;
        typedef typename AnnotatedArrayImpl<T>::value_type value_type;
        typedef typename AnnotatedArrayImpl<T>::ordinal_type ordinal_type;
        typedef typename AnnotatedArrayImpl<T>::volume_type volume_type;
        typedef typename AnnotatedArrayImpl<T>::size_array size_array;
        typedef typename AnnotatedArrayImpl<T>::array_ptr array_ptr;

      private:
        // Prohibited operations
        AnnotatedArray();

      public:
        /// creates an array living in world and described by shape. Optional
        /// val specifies the default value of every element
        template<unsigned int DIM, detail::DimensionOrderType O>
        AnnotatedArray(Array<value_type, DIM, CoordinateSystem<DIM, O> >* a, const VariableList& v) :
            pimpl_(boost::make_shared<AnnotatedArrayImpl<T> >(dynamic_cast<array_ptr>(a), v, O))
        { }

        /// Copy constructor
        AnnotatedArray(const AnnotatedArray_& other) : pimpl_(other.pimpl_)
        { }

        /// Assignment operator
        AnnotatedArray_ operator=(const AnnotatedArray_& other) {
          pimpl_ = other.pimpl_;
        }

        /// AnnotatedTile assignment operator
         template<typename Exp0, typename Exp1, typename Op>
         AnnotatedArray_& operator =(const BinaryArrayExp<Exp0, Exp1, Op>& e) {
           pimpl_->operator=(e);
           return *this;
         }

         /// AnnotatedTile assignment operator
         template<typename Exp, typename Op>
         AnnotatedArray_& operator =(const UnaryArrayExp<Exp, Op>& e) {
           pimpl_->operator=(e);
           return *this;
         }

        /// Inserts a tile into the array.

        /// Copies the given tile into the array. Non-local insertions will initiate
        /// non-blocking communication.
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O, typename Tile>
        void insert(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i, const Tile& t) {
          pimpl_->insert(i, t);
        }

        /// Inserts a tile into the array.

        /// Copies the given value_type into the array. Non-local insertions will
        /// initiate non-blocking communication.
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O, typename Tile>
        void insert(const std::pair<const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >, Tile>& v) {
          pimpl_->insert(v);
        }

        /// Erases a tile from the array.

        /// This will remove the tile at the given index. It will initiate
        /// non-blocking for non-local tiles.
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        void erase(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) {
          pimpl_->erase(i);
        }

        /// Erase a range of tiles from the array.

        /// This will remove the range of tiles from the array. The iterator must
        /// dereference to value_type (std::pair<index_type, tile_type>). It will
        /// initiate non-blocking communication for non-local tiles.
        template<typename InIter>
        void erase(InIter first, InIter last) {
          BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
          pimpl_->erase(first, last);
        }

        /// Removes all tiles from the array.
        void clear() {
          pimpl_->clear();
        }

        /// Permutes the array. This will initiate blocking communication.
        template<unsigned int DIM>
        AnnotatedArray_& operator ^=(const Permutation<DIM>& p) {
          pimpl_->operator ^=(p);
          return *this;
        }

        /// Returns true if the tile specified by index is stored locally.
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        bool is_local(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {

          return pimpl_->is_local(i);;
        }

      private:

        friend void TiledArray::expressions::array::swap<>(AnnotatedArray_&, AnnotatedArray_&);

        boost::shared_ptr<AnnotatedArrayImpl<T> > pimpl_;
      }; // class AnnotatedArray

      template<typename T>
      void swap(AnnotatedArray<T>& a0, AnnotatedArray<T>& a1) {
        boost::swap(a0.pimp_, a1.pimp_);
      }



    } // namespace array
  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
