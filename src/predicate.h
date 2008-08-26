#ifndef PREDICATE_H__INCLUDED
#define PREDICATE_H__INCLUDED

#include <tuple.h>

namespace TiledArray {

  /**
   * Predicate that maps an DIM-tuple to a boolean.
   * The output is computed as f(P(T)), where T is the input tuple,
   * P is a permutation, and f is a predicate.
   */
  template<unsigned int DIM> class TupleFilter {
    /// Current permutation of the TupleFilter
    Tuple<DIM> m_permutation;
    /// Used to determine if a transpose has been applied.
    bool m_apply_permutation;
	  
  public:
      
    TupleFilter() :
      m_apply_permutation(false)
    {
      for(unsigned int index = 0; index < DIM; ++index)
        m_permutation[index] = index;
    }
    	
    	
    virtual bool operator ()(const Tuple<DIM>& T) =0;
 
    // Forward permutation of set by one.
    inline Tuple<DIM>& permute() {
      m_apply_permutation = true;

      return m_permutation.permute();
    }

    /// Reverse permutation of set by one.
    inline Tuple<DIM>& reverse_permute() {
      m_apply_permutation = true;

   	  return m_permutation.reverse_permute();
    }
      
    inline Tuple<DIM>& permute(const Tuple<DIM>& trans) {
    	m_apply_permutation = true;
    	return m_permutation.permute(trans);
    }

    /// Reset the permutation to the default value of no permutation.
    void reset_permutation() {
  	  m_apply_permutation = false;
      for(unsigned int index = 0; index < DIM; ++index)
        m_permutation[index] = index;
    }

  protected:
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
