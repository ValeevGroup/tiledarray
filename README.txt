
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
is a wrapper around Array<T,DIM,TRAIT>, with perhaps some extra functionality?

-----------

New Classes:

template <typename T, unsigned int DIM> TA {
  private:
  shared_ptr<AbstractShape> shape_;
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
