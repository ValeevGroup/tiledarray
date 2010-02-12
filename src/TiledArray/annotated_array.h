#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <TiledArray/annotation.h>
#include <TiledArray/madness_runtime.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/type_traits.hpp>

namespace TiledArray {

  template<typename T, typename I>
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
      class AnnotatedArray {
      private:
        typedef BaseArray<T, std::size_t> BaseArray_;
      public:
        typedef AnnotatedArray<T> AnnotatedArray_;
        typedef typename BaseArray_::ordinal_type ordinal_type;
        typedef ordinal_type index_type;
        typedef typename BaseArray_::volume_type volume_type;
        typedef typename BaseArray_::value_type value_type;
        typedef typename BaseArray_::data_type data_type;
        typedef typename BaseArray_::const_data_type const_data_type;
        typedef typename BaseArray_::size_array size_array;
        typedef typename detail::add_const<boost::is_const<T>::value, BaseArray_* >::type array_ptr;
        typedef typename BaseArray_::iterator_atile iterator;
        typedef typename BaseArray_::const_iterator_atile const_iterator;

      public:
        /// Default constructor
        AnnotatedArray() : array_(NULL), var_(), owner_(false) { }

        /// creates an array living in world and described by shape. Optional
        /// val specifies the default value of every element
        AnnotatedArray(array_ptr a, const VariableList& v) :
            array_(a), var_(v), owner_(false)
        { }


        AnnotatedArray(const AnnotatedArray_& other) :
            array_((other.owner_ ? other.array_->clone(true) : other.array_)),
            var_(other.var_), owner_(other.owner_)
        { }

        ~AnnotatedArray() {
          if(owner_)
            delete array_;
        }

        AnnotatedArray_ operator=(const AnnotatedArray_& other) {
          if(other.owner_)
            array_ = other.array_->clone(true);
          else
            other.array_;
          var_ = other.var_;
          owner_ = other.owner_;

          return *this;
        }

        /// AnnotatedTile assignment operator
        template<typename Exp0, typename Exp1, typename Op>
        AnnotatedArray_& operator =(const BinaryArrayExp<Exp0, Exp1, Op>& /*e*/) {
          TA_ASSERT(false, std::runtime_error, "Not yet implemented.");
          return *this;
        }

        /// AnnotatedTile assignment operator
        template<typename Exp, typename Op>
        AnnotatedArray_& operator =(const UnaryArrayExp<Exp, Op>& /*e*/) {
          TA_ASSERT(false, std::runtime_error, "Not yet implemented.");
          return *this;
        }

        /// Returns an iterator to an AnnotatedTile object.

        /// This function will return an AnnotatedTile object, which points to
        /// the tiles contained by the AnnotatedArray.
        iterator begin() { return array_->begin_atile(); }
        const_iterator begin() const { return array_->begin_const_atile(); }
        iterator end() { return array_->end_atile(); }
        const_iterator end() const { return array_->end_const_atile(); }

        /// Return a future to an AnnotatedTile Object.
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        madness::Future<data_type> find(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) {
          return array_->find(ord_(i), var_);
        }
        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        madness::Future<const_data_type> find(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
          return array_->find(ord_(i), var_);
        }
        madness::Future<data_type> find(const ordinal_type i) {
          return array_->find(i, var_);
        }
        madness::Future<const_data_type> find(const ordinal_type i) const {
          return array_->find(i, var_);
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
          array_.clear();
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

      private:

        template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
        ordinal_type ord_(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) {
          return std::inner_product(i.begin(), i.end(), array_->weight_ref().begin(), ordinal_type(0));
        }

        friend void swap<>(AnnotatedArray_&, AnnotatedArray_&);

        array_ptr array_;
        VariableList var_;
        bool owner_;
      }; // class AnnotatedArray


      template<typename T>
      void swap(AnnotatedArray<T>& a0, AnnotatedArray<T>& a1) {
        boost::swap(a0.pimp_, a1.pimp_);
      }



    } // namespace array
  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
