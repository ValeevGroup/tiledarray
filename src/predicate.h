#ifndef PREDICATE_H__INCLUDED
#define PREDICATE_H__INCLUDED

#include <coordinates.h>
#include <permutation.h>

namespace TiledArray {

  /**
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

      /// predicate function
      template <typename T, typename Tag, typename CS>
      bool includes(const ArrayCoordinate<T,DIM,Tag,CS>& index) const {
    	  return true;
      }

      /// predicate operator
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

      for(unsigned int d = 1; d < DIM; ++d)
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
    void operator ^=(const Permutation<DIM>& perm) {
      apply_perm_ = true;
      perm_ = perm;

      return *this;
    }

    /// Reset the permutation to the default value of no permutation.
    void reset_perm() {
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

  /// This predicate combines two predicates to act as a single predicate
  /// ComboPred::includes(i) returns true when both Pred1::includes(i) and
  /// Pred2::includes(i) returns true.
  template<unsigned int DIM, typename Pred1, typename Pred2>
  class ComboPred {
  public:

    /// Default constructor
	ComboPred() : p1_(), p2_() { }

	/// Primary constructor which accepts two initialized predicates.
    ComboPred(const Pred1& p1, const Pred2& p2) : p1_(p1), p2_(p2) { }

    /// Copy constructor
    ComboPred(const ComboPred<DIM,Pred1,Pred2>& other) :
        p1_(other.p1_), p2_(other.p2) { }

    /// predicate function
    template <typename T, typename Tag, typename CS>
    bool includes(const ArrayCoordinate<T,DIM,Tag,CS>& index) const {
      return p1_.includes(index) && p2_.includes(index);
    }

    /// predicate operator
    template <typename T, typename Tag, typename CS>
    bool operator ()(const ArrayCoordinate<T,DIM,Tag,CS>& index) const {
      return includes(index);
    }

  private:
    Pred1 p1_;
    Pred2 p2_;
  }; // class DensePred



} // namespace TiledArray

#endif // PREDICATE_H__INCLUDED
