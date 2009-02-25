#ifndef PREDICATE_H__INCLUDED
#define PREDICATE_H__INCLUDED

#include <coordinates.h>
#include <permutation.h>
#include <range.h>

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
  template<unsigned int DIM> class AbstractPred {
    public:

      /// Default constructor
      AbstractPred() :
        perm_(Permutation<DIM>::unit()),
        apply_perm_(false)
      { }

      /// Copy constructor
      AbstractPred(const AbstractPred& pred) :
        perm_(pred.m_permutation),
        apply_perm_(pred.m_apply_permutation)
      { }

      /// pure virtual predicate function
      template <typename T, typename Tag, typename CS>
      virtual bool includes(const ArrayCoordinate<T,DIM,Tag,CS>& index) const =0;

      /// predicate function
      template <typename T, typename Tag, typename CS>
      bool operator ()(const ArrayCoordinate<T,DIM,Tag,CS>& index) const {
        return this->includes(index);
      }

      /// Apply permutation
      void permute(const Permutation<DIM>& perm) {
        apply_perm_ = true;
        perm_ = perm;
      }

      /// Reset the permutation to the default value of no permutation.
      void reset() {
        apply_perm_ = false;
        perm_ = Permutation<DIM>::unit();
      }

//      virtual void print(std::ostream& o) const =0;

    protected:
      /// Current permutation of the TupleFilter
      Permutation<DIM> perm_;
      /// Used to determine if a transpose has been applied.
      bool apply_perm_;

      /**
       * Translates index to the from the current permutation of the array
       * to the original shape of the array.
       */
      template <typename T, typename Tag, typename CS>
      ArrayCoordinate<T,DIM,Tag,CS> translate(const ArrayCoordinate<T,DIM,Tag,CS>& index) const {
    	if(apply_perm_)
          return perm_ ^ index;
    	else
          return index;
      }

  };

/*
  template<unsigned int DIM> std::ostream& operator<<(std::ostream& o,
                                                      const AbstractPred<DIM>& f) {
    f.print(o);
    return o;
  }
*/

  template<unsigned int DIM> class DensePred : public AbstractPred<DIM> {
    public:

      /// Default constructor
      DensePred() :
          AbstractPred<DIM>() {
      }

      /// Copy constructor
      DensePred(const DensePred& pred) :
    	  AbstractPred<DIM>(pred)
      { }

      /// Assignment operator
      AbstractPred& operator =(const AbstractPred& pred) {
        perm_ = pred.perm_;
        apply_perm_ = pred.apply_perm_;

        return *this;
      }

      /// pure virtual predicate function
      template <typename T, typename Tag, typename CS>
      virtual bool operator ()(const ArrayCoordinate<T,DIM,Tag,CS>& index) const {
        return true;
      }
/*
      void print(std::ostream& o) const {
        o << "OffTupleFilter";
        if (this->apply_perm_)
          o << "( perm=" << this->perm_ << " )";
      }
*/
  };
}
; // end of namespace TiledArray

#endif // PREDICATE_H__INCLUDED
