#ifndef PREDICATE_H__INCLUDED
#define PREDICATE_H__INCLUDED

namespace TiledArray {

  /** Predicate that maps an DIM-tuple to a boolean.
      The output is computed as f(P(T)), where T is the input tuple,
      P is a permutation, and f is a predicate.
   */
  template<unsigned int DIM> class TupleFilter {
    public:
      
      virtual bool operator ()(const Tuple<DIM>& T) =0;
      virtual TupleFilter<DIM>& permute(const Tuple<DIM>& perm) =0;
      
  };
  
  template<unsigned int DIM> class OffTupleFilter : public TupleFilter<DIM> {
    public:
      
      virtual bool operator ()(const Tuple<DIM>& tup) {
        return true;
      }
      
      TupleFilter<DIM>& permute(const Tuple<DIM>& perm) {
        return *this;
      }
  };

}
; // end of namespace TiledArray

#endif // PREDICATE_H__INCLUDED
