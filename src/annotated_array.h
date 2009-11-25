#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <annotation.h>
#include <madness_runtime.h>

namespace TiledArray {

  template<typename T>
  class BaseArray;
  template<typename T, unsigned int DIM, typename CS>
  class Array;

  namespace expressions {

    template<typename T>
    class AnnotatedTile;

    template<typename Exp0, typename Exp1, typename Op>
    class BinaryArrayExp;
    template<typename Exp, typename Op>
    class UnaryArrayExp;
    template<typename T>
    class AnnotatedArray;
    template<typename T>
    void swap(AnnotatedArray<T>&, AnnotatedArray<T>&);

    /// Annotated Array

    template<typename T>
    class AnnotatedArray : public madness::WorldObject< AnnotatedArray<T> >, Annotation {
    public:
      typedef AnnotatedArray<T> AnnotatedArray_;
      typedef typename Annotation::ordinal_type ordinal_type;
      typedef typename Annotation::volume_type volume_type;
      typedef typename Annotation::size_array size_array;

    private:
      typedef BaseArray<T>* array_ptr;

      // Prohibited operations
      AnnotatedArray();
      AnnotatedArray(const AnnotatedArray_&);
      AnnotatedArray_ operator=(const AnnotatedArray_&);

    public:
      /// creates an array living in world and described by shape. Optional
      /// val specifies the default value of every element
      template<unsigned int DIM, detail::DimensionOrderType O>
      AnnotatedArray(const Array<T, DIM, CoordinateSystem<DIM, O> >& a, VariableList v) :
          madness::WorldObject<AnnotatedArray_>(a.get_world()),
          Annotation(a.size().begin(), a.size().begin(), a.weight().begin(),
          a.weight().end(), a.volume(), v, O), array_(&a)
      {
        this->process_pending();
        this->get_world().gop.fence();
      }

      /// AnnotatedTile assignment operator
       template<typename Exp0, typename Exp1, typename Op>
       AnnotatedArray_& operator =(const BinaryArrayExp<Exp0, Exp1, Op>& e) {

         return *this;
       }

       /// AnnotatedTile assignment operator
       template<typename Exp, typename Op>
       AnnotatedArray_& operator =(const UnaryArrayExp<Exp, Op>& e) {

         return *this;
       }

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
        Annotation::operator^=(p);
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

      friend void TiledArray::expressions::swap<>(AnnotatedArray_&, AnnotatedArray_&);

      array_ptr array_;
    }; // class AnnotatedArray

    template<typename T>
    void swap(AnnotatedArray<T>& a0, AnnotatedArray<T>&) {
      TA_ASSERT(false, std::runtime_error, "Not yet implemented.");
    }


  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
