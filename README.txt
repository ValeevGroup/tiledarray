
Old htalib Classes:

Tuple<DIM>
DIM-dimensional vector if integers

Triplet
represents a range of indices, e.g. Triplet(l,h,s,m) represents a range
of indices from l to h-1 with stride s
        
Shape<DIM>
describes a shape of a DIM-dimensional array as a sequence of tile indices.
In htalib Shape represented a rectangular dense shape (with complex striding patterns).
We'd like to have a more general Shape. It must provide means to iterate over elements.

MemMapping<DIM>
represents mapping of a set of tiles to memory locations

AbstractArray<T,DIM>
is an abstract interface to an DIM-dimensional array. It has a shape (Shape<DIM>),
a memory map (MemMapping<DIM>) and allows random access to tiles and scalars.

Array<T,DIM,TRAIT>
is a concrete implementation of AbstractArray described by TRAIT.
Examples of TRAIT: SerialDense, SerialSparse, MPISerialDense, MPISerialSparse, etc.

HTA<T,DIM,TRAIT>
is a wrapper around Array<T,DIM,TRAIT>, with math operators and functions added.

-----------

New Classes:

template <typename T, unsigned int DIM> AbstractTiledArray {
}

template <typename T, unsigned int DIM, class ArrayTrait> TiledArray {
  public:
  typedef ArrayTrait::Shape Shape;
  
  private:
  shared_ptr<Shape> shape_;
  shared_ptr<TileMap> tiles_;   // class that maps tiles of a shape to (potentially, distributed) memory
}

/// This class maps linearized tile indices to the nodes and memory locations
/// distribution of tiles to nodes is done by hashing
/// memory location can be determined for local tiles ONLY
class TileMap {
  public:
  // need some data here?
  TileMap() {}

  virtual MemoryHandle find(size_t linear_tile_idx) =0;
  virtual void register_tile(size_t linear_tile_idx, ptr_t tile) =0;
}

class LocalTileMap {
  public:
  LocalTileMap() : TileMap() {}
  
  private:
  /// maps linear_tile_idx to offset
  std::map<size_t, size_t> offsets_;
}

class DistributedTileMap;

class MemoryAllocator {
  public:
  /// allocate a chunk of s bytes
  virtual size_t allocate(size_t s) =0;
}

class LocalMemoryAllocator {
  public:
  size_t allocate(size_t s) {
    return static_cast<size_t,char*>(new char[s]);
  }
}

template <unsigned int DIM> class AbstractShape {
  virtual size_t linear_idx(const Tuple<DIM>& tile_idx) const =0;
  virtual bool includes(const Tuple<DIM>& tile_idx) const =0;
}

template <unsigned int DIM> class AbstractShapeIterator {
  const AbstractShapeIterator& operator++() {
    ++current_idx_;
    while(! shape_->includes(current_idx_) ) {
      ++current_idx_;
    }
    return *this;
  }
  
  private:
    AbstractShape<DIM>* shape_;
    Tuple<DIM> current_idx_;
}


template <unsigned int DIM, class Predicate> class Shape : public AbstractShape<DIM> {
  bool includes(const Tuple<DIM>& tile_idx) {
    return pred_->includes(tile_idx);
  }
  
  private:
    shared_ptr<Predicate> pred_;
}
