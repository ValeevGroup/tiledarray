#ifndef PREDICATE_H__INCLUDED
#define PREDICATE_H__INCLUDED

#include <coordinates.h>
#include <permutation.h>
#include <range.h>

namespace TiledArray {

  /*
   * The Predicates must be Assignable, Copy Constructible, and
   * the expression p(x) must be valid where p is an object of type
   * Predicate, x is an object of type iterator_traits<Iterator>::value_type,
   * and where the type of p(x) must be convertible to bool.
   */

  template<unsigned int DIM>
  class DensePred {
    public:

      /// Default constructor
      DensePred()
      { }

      /// Copy constructor
      DensePred(const DensePred& pred)
      { }

      DensePred<DIM>& operator =(const DensePred<DIM>& pred) {

        return *this;
      }

      /// pure virtual predicate function
      template <typename T, typename Tag, typename CS>
      bool includes(const ArrayCoordinate<T,DIM,Tag,CS>& index) const {
    	  return true;
      }

      /// predicate function
      template <typename T, typename Tag, typename CS>
      bool operator ()(const ArrayCoordinate<T,DIM,Tag,CS>& index) const {
        return includes(index);
      }

  }; // class DensePred


  template <unsigned int DIM>
  class LowerTrianglePred {
  public:
    // Default constructor
    LowerTrianglePred() :
      perm_(),
      apply_perm_(false)
    { }

    /// Copy constructor
	LowerTrianglePred(const LowerTrianglePred<DIM>& pred) :
        perm_(pred.perm_),
        apply_perm_(pred.apply_perm_)
    { }

    /// Assignment operator
    LowerTrianglePred<DIM>& operator =(const LowerTrianglePred<DIM>& pred) {
      apply_perm_ = pred.apply_perm_;
      perm_ = pred.perm_;

      return *this;
	}

    /// Returns true if index is included in the shape.
    template <typename T, typename Tag, typename CS>
    bool includes(const ArrayCoordinate<T,DIM,Tag,CS>& index) const {
      const ArrayCoordinate<T,DIM,Tag,CS> perm_index = translate(index);

      for(unsigned int d = 1; d <= DIM; ++d)
        if(perm_index[d - 1] > perm_index[d])
          return false;

  	  return true;
    }

    /// predicate function
    template <typename T, typename Tag, typename CS>
    bool operator ()(const ArrayCoordinate<T,DIM,Tag,CS>& index) const {
      return includes(index);
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

  }; // class LowerTrianglePred

} // namespace TiledArray

#endif // PREDICATE_H__INCLUDED
