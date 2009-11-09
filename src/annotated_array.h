#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <annotation.h>
#include <madness_runtime.h>

namespace TiledArray {

  template<typename T, unsigned int DIM, typename CS>
  class Array;

  namespace expressions {

    template<typename T>
    class AnnotatedTile;
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
      typedef AnnotatedTile<T> tile_type;
    private:
      typedef madness::WorldContainer<ordinal_type, tile_type > data_container;

    public:

      typedef typename data_container::value_type value_type;
      typedef tile_type & reference_type;
      typedef const tile_type & const_reference_type;
      typedef typename data_container::accessor accessor;
      typedef typename data_container::const_accessor const_accessor;
      typedef typename data_container::iterator iterator;
      typedef typename data_container::const_iterator const_iterator;

    private:



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
          a.weight().end(), a.volume(), v, O), tiles_(a.get_world(), true)
      {
        for(typename Array<T, DIM, CoordinateSystem<DIM, O> >::const_iterator it = a.begin(); it != a.end(); ++it)
          insert(this->ord_(it->first), tile_type(it->second, v));
        this->process_pending();
        this->get_world().gop.fence();
      }

      /// Inserts a tile into the array.

      /// Copies the given tile into the array. Non-local insertions will initiate
      /// non-blocking communication.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O, typename Tile>
      void insert(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i, const Tile& t) {
        TA_ASSERT(this->dim() == DIM, std::runtime_error,
            "The index dimensions is not equal to the array dimensions.");
        tiles_.insert(this->ord_(i), t);
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
        tiles_.erase(this->ord_(i));
      }

      /// Erase a range of tiles from the array.

      /// This will remove the range of tiles from the array. The iterator must
      /// dereference to value_type (std::pair<index_type, tile_type>). It will
      /// initiate non-blocking communication for non-local tiles.
      template<typename InIter>
      void erase(InIter first, InIter last) {
        for(; first != last; ++first);
          tiles_.erase(this->ord_(first->first));
      }

      /// Removes all tiles from the array.
      void clear() {
        tiles_.clear();
      }

      /// Returns an iterator to the first local tile.
      iterator begin() { return tiles_.begin(); }
      /// returns a const_iterator to the first local tile.
      const_iterator begin() const { return tiles_.begin(); }
      /// Returns an iterator to the end of the local tile list.
      iterator end() { return tiles_.end(); }
      /// Returns a const_iterator to the end of the local tile list.
      const_iterator end() const { return tiles_.end(); }

      /// Permutes the array. This will initiate blocking communication.
      template<unsigned int DIM>
      AnnotatedArray_& operator ^=(const Permutation<DIM>& p) {
        TA_ASSERT(this->dim() == DIM, std::runtime_error,
            "The permutation dimensions is not equal to the array dimensions.");
        for(iterator it = begin(); it != end(); ++it)
          it->second ^= p; // permute the individual tile
        Annotation::operator^=(p);
        tiles_ ^= p; // move the tiles to the correct location. Blocking communication here.

        return *this;
      }

      /// Returns true if the tile specified by index is stored locally.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      bool is_local(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
        TA_ASSERT(this->dim() == DIM, std::runtime_error,
            "The index dimensions is not equal to the array dimensions.");
        return tiles_.is_local(this->ord_(i));
      }

      /// Returns a Future iterator to an element at index i.

      /// This function will return an iterator to the element specified by index
      /// i. If the element is not local the it will use non-blocking communication
      /// to retrieve the data. The future will be immediately available if the data
      /// is local.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      madness::Future<iterator> find(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) {
        TA_ASSERT(this->dim() == DIM, std::runtime_error,
            "The index dimensions is not equal to the array dimensions.");
        return tiles_.find(this->ord_(i));
      }

      /// Returns a Future const_iterator to an element at index i.

      /// This function will return a const_iterator to the element specified by
      /// index i. If the element is not local the it will use non-blocking
      /// communication to retrieve the data. The future will be immediately
      /// available if the data is local.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      madness::Future<const_iterator> find(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
        TA_ASSERT(this->dim() == DIM, std::runtime_error,
            "The index dimensions is not equal to the array dimensions.");
        return tiles_.find(this->ord_(i));
      }

      /// Sets an accessor to point to a local data element.

      /// This function will set an accessor to point to a local data element only.
      /// It will return false if the data element is remote or not found.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      bool find(accessor& acc, const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) {
        TA_ASSERT(this->dim() == DIM, std::runtime_error,
            "The index dimensions is not equal to the array dimensions.");
        return tiles_.find(acc, this->ord_(i));
      }

      /// Sets a const_accessor to point to a local data element.

      /// This function will set a const_accessor to point to a local data element
      /// only. It will return false if the data element is remote or not found.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      bool find(const_accessor& acc, const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
        TA_ASSERT(this->dim() == DIM, std::runtime_error,
            "The index dimensions is not equal to the array dimensions.");
        return tiles_.find(acc, this->ord_(i));
      }

    private:

      friend void TiledArray::expressions::swap<>(AnnotatedArray_&, AnnotatedArray_&);

      data_container tiles_;
    }; // class AnnotatedArray

    template<typename T>
    void swap(AnnotatedArray<T>& a0, AnnotatedArray<T>&) {
      TA_ASSERT(false, std::runtime_error, "Not yet implemented.");
    }


  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
