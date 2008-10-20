Design philosophy:

There are 2 fundamental reasons for tiling: physics and performance. Both of these
may require multiple levels of tiling hierarchy. Unfortunately, it may not be trivial to decouple or overlap
the two hierarchies. Thus we eschew the arbitrarily-deep tiling hierarchy in favor of a single level of tiling.
This is driven by the following considerations:
1) end user should not worry about tiling for performance, the interface should be focused on the needs of
domain application (quantum physics).
2) tiling in quantum physics corresponds to spatial/energetic locality (outside local domains interactions
are classical and do not require operator matrices).
3) the problem of tiling for performance is challenging and has been only solved for simple cases and regular memory
hierarchies (e.g. BLAS). We must reuse on external libraries and should optimize only at a high level and without
impact on interface (e.g. tile fusion, etc. to construct data representation most suitable for performance).
4) item 2 also suggests that nonuniform(irregular) tiling must be allowed to provide the necessary flexibility for physics.

With single level of tiling Array interface must explicitly deal with elements AND tiles (compare to HTA, where
tiles are the elements at a given level).

/// coordinate in a DIM-dimensional space
class ArrayCoordinate<T,DIM,Tag>

typedef ArrayCoordinate<uint,DIM,ElementTag> ElementIndex<DIM>;
typedef ArrayCoordinate<uint,DIM,TileTag> TileIndex<DIM>;

/// 1-d nonuniformly-tiled range
class Range {
}

/// rectangular box in a DIM-dimensional space is defined by DIM Ranges
class Orthotope<DIM> {
  /// 
}

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

  /// Low-level interface will only allow permutations and efficient direct contractions
  /// it should be sufficient to use with an optimizing array expression compiler

  /// make new Array by applying permutation P
  Array transpose(const Tuple<DIM>& P);

  /// Higher-level interface will be be easier to use but necessarily less efficient since it will allow more complex operations
  /// implemented in terms of permutations and contractions by a runtime
  
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
