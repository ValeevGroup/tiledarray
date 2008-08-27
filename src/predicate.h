#ifndef PREDICATE_H__INCLUDED
#define PREDICATE_H__INCLUDED

#include <tuple.h>

namespace TiledArray {
  
  /**
   * Predicate that maps an DIM-tuple to a boolean.
   * The output is computed as f(P(T)), where T is the input tuple,
   * P is a permutation, and f is a predicate.
   * 
   * All classes inherited from TupleFilter must define a default constructor,
   * and the virtual operator() function. The operator() function should call
   * translate(const Tuple<DIM>&) on the tuple passed to it if
   * m_apply_permutation is true, to translate it to the current permutation of
   * the shape.
   */
  template<unsigned int DIM> class TupleFilter {
    public:
      
      /// Copy constructor
      TupleFilter(const TupleFilter& tf) :
        m_permutation(tf.m_permutation),
            m_apply_permutation(tf.m_apply_permutation) {
      }
      
      /// pure virtual predicate function
      virtual bool operator ()(const Tuple<DIM>& T) const =0;

      /// Apply permutation
      Tuple<DIM>& permute(const Tuple<DIM>& trans) {
        m_apply_permutation = true;
        return m_permutation.permute(trans);
      }
      
      /// Reset the permutation to the default value of no permutation.
      void reset() {
        m_apply_permutation = false;
        for (unsigned int index = 0; index < DIM; ++index)
          m_permutation[index] = index;
      }
      
      virtual void print(std::ostream& o) const =0;

    protected:
      /// Current permutation of the TupleFilter
      Tuple<DIM> m_permutation;
      /// Used to determine if a transpose has been applied.
      bool m_apply_permutation;

      /// Default constructor
      TupleFilter() :
        m_apply_permutation(false) {
        for (unsigned int index = 0; index < DIM; ++index)
          m_permutation[index] = index;
      }
      
      /**
       * Translates index to the from the current permutation of the array
       * to the original shape of the array. m_apply_permutation should be
       * checked before calling this function.
       */
      Tuple<DIM> translate(const Tuple<DIM>& index) const {
        Tuple<DIM> tmp(index);
        return tmp.permute(tmp);
      }
      
  };
  
  template<unsigned int DIM> std::ostream& operator<<(std::ostream& o,
                                                      const TupleFilter<DIM>& f) {
    f.print(o);
    return o;
  }
  
  template<unsigned int DIM> class OffTupleFilter : public TupleFilter<DIM> {
    public:
      
      /// Default constructor
      OffTupleFilter() :
        TupleFilter<DIM>() {
      }
      
      /// Copy constructor
      OffTupleFilter(const OffTupleFilter& otf) :
        TupleFilter<DIM>(otf) {
      }
      
      /// Assignment operator
      OffTupleFilter& operator =(const OffTupleFilter& otf) {
        this->m_permutation = otf.m_permutation;
        this->m_apply_permutation = otf.m_apply_m_permutation;
        
        return *this;
      }
      
      bool operator ()(const Tuple<DIM>& tup) const {
        return true;
      }
      
      void print(std::ostream& o) const {
        o << "OffTupleFilter";
        if (this->m_apply_permutation)
          o << "( perm=" << this->m_permutation << " )";
      }
  };

}
; // end of namespace TiledArray

#endif // PREDICATE_H__INCLUDED
