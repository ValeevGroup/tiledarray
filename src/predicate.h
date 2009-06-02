#ifndef PREDICATE_H__INCLUDED
#define PREDICATE_H__INCLUDED

#include <permutation.h>
#include <boost/shared_ptr.hpp>

namespace TiledArray {

  /**
   * The Predicates must be Assignable, Copy Constructible, and
   * the expression p(x) must be valid where p is an object of type
   * Predicate, x is an object of type iterator_traits<Iterator>::value_type,
   * and where the type of p(x) must be convertible to bool.
   */

  template<unsigned int DIM, typename Index>
  class DensePred {
    public:

      /// Default constructor
      DensePred() /* throw() */
      { }

      /// Copy constructor
      DensePred(const DensePred& pred) /* throw() */
      { }

      DensePred& operator =(const DensePred& pred) { /* throw() */

        return *this;
      }

      /// predicate function
      bool includes(const Index& i) const { /* throw() */
    	  return true;
      }

      /// predicate operator
      bool operator ()(const Index& i) const { /* throw() */
        return includes(i);
      }

      /// Permute the predicate
      DensePred& operator ^=(const Permutation<DIM>& perd) /* throw() */
      { return *this; }

      /// Reset the predicate to its default state.
      void reset() {}

  }; // class DensePred


  template <unsigned int DIM, typename Index>
  class LowerTrianglePred {
  public:
    // Default constructor
    LowerTrianglePred() :
      perm_(),
      apply_perm_(false)
    { }

    /// Copy constructor
	LowerTrianglePred(const LowerTrianglePred<DIM,Index>& pred) :
        perm_(pred.perm_),
        apply_perm_(pred.apply_perm_)
    { }

    /// Assignment operator
    LowerTrianglePred& operator =(const LowerTrianglePred& pred) {
      apply_perm_ = pred.apply_perm_;
      perm_ = pred.perm_;

      return *this;
	}

    /// Returns true if index is included in the shape.
    bool includes(const Index& i) const {
      const Index perm_index = (apply_perm_ ? perm_ ^ i : i );

      for(unsigned int d = 1; d < DIM; ++d)
        if(perm_index[d - 1] > perm_index[d])
          return false;

  	  return true;
    }

    /// predicate function
    bool operator ()(const Index& i) const {
      return includes(i);
    }

    /// Apply permutation
    void operator ^=(const Permutation<DIM>& perm) {
      apply_perm_ = true;
      perm_ = perm;

      return *this;
    }

    /// Reset to the default default state. This well reset the permutation.
    void reset() {
      apply_perm_ = false;
      perm_ = Permutation<DIM>::unit();
    }

  protected:
    /// Current permutation of the TupleFilter
    Permutation<DIM> perm_;
    /// Used to determine if a transpose has been applied.
    bool apply_perm_;

  }; // class LowerTrianglePred


  /// Predicate which tests to see if an element of the container at an Index
  /// is stored locally. The container must have a function with the following
  /// Signature Container::is_local(Index) const.
  template <typename Container, typename Index>
  class LocalPred {
  public:
    LocalPred(const boost::weak_ptr<Container> c) : cont_(c) {}
    ~LocalPred() {}

    /// predicate function
    bool includes(const Index& i) const {
      return cont_->is_local(i);
    }

    /// predicate operator
    bool operator ()(const Index& i) const {
      return includes(i);
    }

    /// Permute the predicate
    template <unsigned int DIM>
    LocalPred& operator ^=(const Permutation<DIM>& pred) {
      return *this;
    }

  private:
    const boost::weak_ptr<Container> cont_;

  }; // class LocalPred

  /// This predicate combines two predicates to act as a single predicate
  /// ComboPred::includes(i) returns true when both Pred1::includes(i) and
  /// Pred2::includes(i) returns true.
  template <typename Pred1, typename Pred2, typename Index>
  class ComboPred {
  public:

    /// Default constructor
	ComboPred() : p1_(), p2_() { }

	/// Primary constructor which accepts two initialized predicates.
    ComboPred(const Pred1& p1, const Pred2& p2) : p1_(p1), p2_(p2) { }

    /// Copy constructor
    ComboPred(const ComboPred& other) :
        p1_(other.p1_), p2_(other.p2_) { }

    /// predicate function
    bool includes(const Index& i) const {
      return p1_.includes(i) && p2_.includes(i);
    }

    /// predicate operator
    bool operator ()(const Index& i) const {
      return includes(i);
    }

    /// Permute the predicate
    template <unsigned int DIM>
    ComboPred& operator ^=(const Permutation<DIM>& pred) {
      p1_ ^= pred;
      p2_ ^= pred;
      return *this;
    }

  private:
    Pred1 p1_;
    Pred2 p2_;
  }; // class DensePred



} // namespace TiledArray

#endif // PREDICATE_H__INCLUDED
