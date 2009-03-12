Design philosophy:

There are 2 fundamental reasons for tiling: physics and performance. Both of these
may require multiple levels of tiling hierarchy. Unfortunately, it may not be trivial to decouple or overlap
the two hierarchies. Thus we eschew the arbitrarily-deep tiling hierarchy in favor of a single level of tiling.
This is driven by the following considerations:
1) end user should not worry about tiling for performance, the interface should be focused on the needs of
domain application (quantum physics).
2) tiling in quantum physics corresponds to spatial/energetic locality (outside local domains interactions
are classical and do not require operator matrices) or symmetry (not always worth it, point group symmetry
conflicts with the "local" tiling).
3) the problem of tiling for performance is challenging and has been only solved for simple cases and regular memory
hierarchies (e.g. BLAS). We must reuse on external libraries and should optimize only at a high level and without
impact on interface (e.g. tile fusion, etc. to construct data representation most suitable for performance).
4) item 2 also suggests that nonuniform(irregular) tiling must be allowed to provide the necessary flexibility for physics.

With single level of tiling Array interface must explicitly deal with elements AND tiles (compare to HTA, where
tiles ARE the elements at a given level).

/// coordinate in a DIM-dimensional space, whose properties are given by CoordinateSystem. CoordinateSystem has a static default,
/// i.e. C-arrays will not be interoperable with Fortran-arrays
class ArrayCoordinate<T,DIM,Tag,CoordinateSystem>

typedef ArrayCoordinate<uint,DIM,ElementTag> ElementIndex<DIM>;
typedef ArrayCoordinate<uint,DIM,TileTag> TileIndex<DIM>;

/// 1-d nonuniformly-tiled range
class Range1 {
  class Tile;
}
/// rectangular box in a DIM-dimensional space is defined by DIM Ranges
class Range<DIM> {
  class Tile;
}

/// The main player: Array
/// Serves as base to various implementations (local, replicatd, distributed)
///
template <typename T, unsigned int DIM> Array {
public:
  typedef Range<DIM> Range;
  typedef typename Range::TileIterator RangeIterator;
  typedef Shape<Range> Shape;
  typedef ShapeIterator<RangeIterator> ShapeIterator;
  // ElementIndex, TileIndex, etc.

  /// Tile is implemented in terms of boost::multi_array
  /// it provides reshaping, iterators, etc., and supports direct access to the raw pointer.
  /// array layout must match that given by CoordinateSystem (i.e. both C, or both Fortran)
  class Tile : public boost::multi_array<T,DIM> {
  };

  class Iterator : public boost::iterator_facade {
  public:
    Iterator(const shared_ptr<ShapeIterator>& i) : current_index_(i) {}
    const Tile& operator*() const;
  private:
    shared_ptr<ShapeIterator> current_index_;
  };
  Iterator begin() const {
    shared_ptr<ShapeIterator> si_begin( shape_->begin() );
    return Iterator(si_begin);
  }
  Iterator end() const {
    shared_ptr<ShapeIterator> si_end( shape_->end() );
    return Iterator(si_end);
  }
  
  /// array is defined by its shape
  Array(const shared_ptr<Shape>&);
  
  /// assign each element to a
  virtual void assign(T a) =0;
  
  /// where is tile k
  virtual unsigned int proc(const TileIndex& k) =0;
  /// access tile k
  virtual Future<Tile> tile(const TileIndex& k) =0;

  /// Low-level interface will only allow permutations and efficient direct contractions
  /// it should be sufficient to use with an optimizing array expression compiler

  /// make new Array by applying permutation P
  virtual Array transpose(const Permutation<DIM>& P) =0;

  /// Higher-level interface will be be easier to use but necessarily less efficient since it will allow more complex operations
  /// implemented in terms of permutations and contractions by a runtime
  
  /// bind a string to this array to make operations look normal
  /// e.g. R("ijcd") += T2("ijab") . V("abcd") or T2new("iajb") = T2("ijab")
  /// runtime then can figure out how to implement operations
  // ArrayExpression operator()(const char*) const;
  
private:
  shared_ptr<Shape> shape_;
}

/// DistributedObject is an abstract distributed data structure and DistributedContainer
/// is a distributed map that manages key->value pairs. DenseTile<T,DIM> is a linearized
/// DIM-dimensional array.
template <typename T, unsigned int DIM, typename ArrayPolicy = DefaultArrayPolicy > Array : public DistributedObject {
public:
  typedef ArrayPolicy Policy;
  typedef TileIndex<DIM> TileIndex;
  typedef ElementIndex<DIM> ElementIndex;
  typedef typename ArrayPolicy::Hash<TileIndex>::Result TileKey;
  
  /// Tile is implemented in terms of boost::multi_array
  /// it provides reshaping, iterators, etc., and supports direct access to the raw pointer.
  /// array layout must match that given by CoordinateSystem (i.e. both C, or both Fortran)
  class Tile : public boost::multi_array {
  };

  /// array is defined by its shape
  Array(const shared_ptr<Shape>&);
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
  shared_ptr<Shape> shape_;
  shared_ptr< DistributedContainer< TupleKey<DIM>, tile_t > > data_;
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

template <typename Range> class Shape {
  public:
    typedef class IndexIterator<Range::tile_iterator, Shape> Iterator;
    
    virtual bool includes(const Tuple<DIM>& tile_idx) const =0;

    virtual void increment(ShapeIterator&) =0;
    virtual Iterator begin() const =0;
    virtual Iterator end() const =0;
    
}

template <typename Range, class Predicate> class PredShape : public Shape<Range> {
  public:
    PredShape(const Predicate& pred) : pred_(pred) {}
    bool includes(const Tuple<DIM>& tile_idx) {
      return pred_->includes(tile_idx);
    }
    
    shared_ptr<ShapeIterator> begin() const {
      
    }

    void increment(ShapeIterator& i) {
      ++i.current_idx_;
      while(! pred_.includes(i.current_idx_) ) {
        ++i.current_idx_;
      }
    }
    
  private:
    Predicate pred_;
}
