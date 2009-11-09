#ifndef TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
#define TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED

#include <variable_list.h>
#include <vector>

namespace TiledArray {
  namespace expressions {

    /// Annotated Array
    template<typename T>
    class AnnotatedArray : public Annotation {
    public:
      typedef AnnotatedArray<T> AnnotatedArray_;
    private:
      typedef DistributedArrayStorage<tile_type, DIM, LevelTag<1>, coordinate_system> data_container;

    public:
      typedef typename data_container::index_type index_type;
      typedef typename tile_type::index_type tile_index_type;
      typedef typename Annotation::ordinal_type ordinal_type;
      typedef typename Annotation::volume_type volume_type;
      typedef typename Annotation::size_array size_array;
      typedef typename data_container::value_type value_type;
      typedef TiledRange<ordinal_type, DIM, CS> tiled_range_type;
      typedef tile_type & reference_type;
      typedef const tile_type & const_reference_type;
      typedef typename data_container::accessor accessor;
      typedef typename data_container::const_accessor const_accessor;
      typedef typename data_container::iterator iterator;
      typedef typename data_container::const_iterator const_iterator;

    private:

      friend void swap<>(AnnotatedArray_& a0, AnnotatedArray_&);

      // Prohibited operations
      AnnotatedArray();
      AnnotatedArray(const AnnotatedArray_&);
      AnnotatedArray_ operator=(const AnnotatedArray_&);

    public:
      /// creates an array living in world and described by shape. Optional
      /// val specifies the default value of every element
      AnnotatedArray(madness::World& world, const tiled_range_type& rng) :
          madness::WorldObject<AnnotatedArray_>(world), range_(rng), tiles_(world, rng.tiles().size())
      {
        this->process_pending();
      }

      /// Inserts a tile into the array.

      /// Copies the given tile into the array. Non-local insertions will initiate
      /// non-blocking communication.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O, typename Tile>
      void insert(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i, const Tile& t) {
        TA_ASSERT(this->dim() == DIM, std::runtime_error,
            "The index dimensions is not equal to the array dimensions.");
        tiles_.insert(i, t);
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
        tiles_.erase(i);
      }

      /// Erase a range of tiles from the array.

      /// This will remove the range of tiles from the array. The iterator must
      /// dereference to value_type (std::pair<index_type, tile_type>). It will
      /// initiate non-blocking communication for non-local tiles.
      template<typename InIter>
      void erase(InIter first, InIter last) {
        tiles_.erase(first, last);
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
        range_ ^= p;
        tiles_ ^= p; // move the tiles to the correct location. Blocking communication here.

        return *this;
      }

      /// Returns true if the tile specified by index is stored locally.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      bool is_local(const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
        TA_ASSERT(this->dim() == DIM, std::runtime_error,
            "The index dimensions is not equal to the array dimensions.");
        return tiles_.is_local(i);
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
        return tiles_.find(i);
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
        return tiles_.find(i);
      }

      /// Sets an accessor to point to a local data element.

      /// This function will set an accessor to point to a local data element only.
      /// It will return false if the data element is remote or not found.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      bool find(accessor& acc, const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) {
        TA_ASSERT(this->dim() == DIM, std::runtime_error,
            "The index dimensions is not equal to the array dimensions.");
        return tiles_.find(acc, i);
      }

      /// Sets a const_accessor to point to a local data element.

      /// This function will set a const_accessor to point to a local data element
      /// only. It will return false if the data element is remote or not found.
      template<typename I, unsigned int DIM, typename Tag, TiledArray::detail::DimensionOrderType O>
      bool find(const_accessor& acc, const ArrayCoordinate<I,DIM,Tag, CoordinateSystem<DIM,O> >& i) const {
        TA_ASSERT(this->dim() == DIM, std::runtime_error,
            "The index dimensions is not equal to the array dimensions.");
        return tiles_.find(acc, i);
      }

    private:

      tiled_range_type range_;
      data_container tiles_;
    }; // class AnnotatedArray

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_ANNOTATED_ARRAY_H__INCLUDED
