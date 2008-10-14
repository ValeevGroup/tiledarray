
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

template <typename T, unsigned int DIM, typename ArrayPolicy = DefaultArrayPolicy > Array : public DistributedObject {
public:
  typedef ArrayPolicy Policy;
  typedef Tuple<DIM> TileIndex;
  typedef Tuple<DIM> ElementIndex;
  typedef typename ArrayPolicy::Hash<TileIndex>::Result TileKey;
  typedef typename Policy::Tile Tile;

  /// array is defined by its shape
  Array(const shared_ptr<AbstractShape>&);
  /// assign each element to a
  void assign(T a);
  
  /// where is tile k
  unsigned int proc(const TileIndex& k);
  /// access tile k
  Future<Tile> tile(const TileIndex& k);

  /// make new Array by applying permutation P
  Array transpose(const Tuple<DIM>& P);

  /// bind a string to this array to make operations look normal
  /// e.g. R("ijcd") += T2("ijab") . V("abcd") or T2new("iajb") = T2("ijab")
  /// runtime then can figure out how to implement operations
  ArrayExpression operator()(const char*) const;
  
private:
  shared_ptr<AbstractShape> shape_;
  shared_ptr< DistributedContainer< TupleKey<DIM>, tile_t > > data_;
}

DistributedObject is an abstract distributed data structure and DistributedContainer
is a distributed map that manages key->value pairs. DenseTile<T,DIM> is a linearized
DIM-dimensional array.

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
