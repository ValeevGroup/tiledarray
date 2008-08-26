/**
 * IBM SOFTWARE DISCLAIMER 
 *
 * This file is part of htalib, a library for hierarchically tiled arrays.
 * Copyright (c) 2005 IBM Corporation.
 *
 * Permission to use, copy, modify and distribute this software for
 * any noncommercial purpose and without fee is hereby granted,
 * provided that this copyright and permission notice appear on all
 * copies of the software. The name of the IBM Corporation may not be
 * used in any advertising or publicity pertaining to the use of the
 * software. IBM makes no warranty or representations about the
 * suitability of the software for any purpose.  It is provided "AS
 * IS" without any express or implied warranty, including the implied
 * warranties of merchantability, fitness for a particular purpose and
 * non-infringement.  IBM shall not be liable for any direct,
 * indirect, special or consequential damages resulting from the loss
 * of use, data or projects, whether in an action of contract or tort,
 * arising out of or in connection with the use or performance of this
 * software.  
 */

/*
 * Version: $Id: Shape.h,v 1.159 2007/02/14 16:09:58 bikshand Exp $
 * Authors: Ganesh Bikshandi, Christoph von Praun
 */

#ifndef SHAPE_H__INCLUDED
#define SHAPE_H__INCLUDED

#include <cstddef>
#include <tuple.h>
#include <orthotope.h>
#include <algorithm>
#include <boost/smart_ptr.hpp>
#include <boost/iterator/filter_iterator.hpp>

namespace TiledArray {
  
  template <unsigned int DIM> class AbstractShape {
    public:
      
      virtual size_t ord(const Tuple<DIM>& element_index) const = 0;

      virtual Tuple<DIM> coord(size_t linear_index) const = 0;

      virtual bool included(const Tuple<DIM>& element_idx) const = 0;

      virtual const Orthotope<DIM>* orthotope() const = 0;
  };
  
  // Shape class defines a multi-dimensional, rectilinear
  // coordinate system and its mapping to an underlying dense,
  // linearized representation. The mapping to ordinals assumes
  // that DIM-1 is the least signifcant dimension. Shape provides
  // an input iterator, which iterates an ordinal value and
  // tuple index simutaniously.
  //
  // Instances of Shape are immutable.
  //

  template <unsigned int DIM, class PREDICATE>
  class Shape : public AbstractShape<DIM> {
    private:
      // Iterator spec for ShapeIterator class.
      class ShapeIteratorSpec {
        public:
          typedef int iterator_type;
          typedef Shape<DIM, PREDICATE> collection_type;
          typedef std::input_iterator_tag iterator_category;
          typedef Tuple<DIM> value;
          typedef value* pointer;
          typedef const value* const_pointer;
          typedef value& reference;
          typedef const value& const_reference;
      };

      // ShapeIterator is an input iterator that iterates over
      // Shape. The iterator assumes row major access and DIM-1 is the least
      // significant dimention.
      class ShapeIterator : public Iterator<ShapeIteratorSpec> {
        public:
          // Iterator typedef's
          typedef typename Iterator<ShapeIteratorSpec>::iterator_type        iterator_type;
          typedef typename Iterator<ShapeIteratorSpec>::collection_type      collection_type;
          typedef typename Iterator<ShapeIteratorSpec>::iterator_category    iterator_catagory;
          typedef typename Iterator<ShapeIteratorSpec>::reference reference;
          typedef typename Iterator<ShapeIteratorSpec>::const_reference      const_reference;
          typedef typename Iterator<ShapeIteratorSpec>::pointer pointer;
          typedef typename Iterator<ShapeIteratorSpec>::const_pointer        const_pointer;
          typedef typename Iterator<ShapeIteratorSpec>::value value;
          typedef typename Iterator<ShapeIteratorSpec>::difference_type      difference_type;

        private:
          
          const collection_type& m_coll; // Reference to the collection that will be iterated over
          value m_value; // current value of the iterator

        public:
          
          // Default construction not allowed (required by forward iterators)
          ShapeIterator() {
            assert(false);
          }
          
          // Main constructor function
          ShapeIterator(const collection_type& coll, const iterator_type& cur) :
            Iterator<ShapeIteratorSpec>(cur), m_coll(coll),
                m_value(coll.coord(cur)) {
          }
          
          // Copy constructor (required by all iterators)
          ShapeIterator(const ShapeIterator& it) :
            Iterator<ShapeIteratorSpec>(it.m_current), m_coll(it.m_coll),
                m_value(it.m_value) {
          }
          
          // Prefix increment (required by all iterators)
          ShapeIterator& operator ++() {
            assert(this->m_current != -1);
            this->advance();
            return *this;
          }
          
          // Postfix increment (required by all iterators)
          ShapeIterator operator ++(int) {
            assert(this->m_current != -1);
            ShapeIterator tmp(*this);
            this->advance();
            return tmp;
          }
          
          // Equality operator (required by input iterators)
          inline bool operator ==(const ShapeIterator& it) const {
            return (this->base() == it.base());
          }
          
          // Inequality operator (required by input iterators)
          inline bool operator !=(const ShapeIterator& it) const {
            return ! (this->operator ==(it));
          }
          
          // Dereference operator (required by input iterators)
          inline const value& operator *() const {
            assert(this->m_current != -1);
            assert(this->m_current < this->m_coll->orthotope()->count());
            return this->m_value;
          }
          
          // Dereference operator (required by input iterators)
          inline const value& operator ->() const {
            assert(this->m_current != -1);
            assert(this->m_current < this->m_coll->orthotope()->count());
            return this->m_value;
          }
          
          int ord() const {
            assert(this->m_current != -1);
            assert(this->m_current < this->m_coll->orthotope()->count());
            return this->m_current;
          }
          
          /* This is for debugging only. Not doen in an overload of operator<<
           * because it seems that gcc 3.4 does not manage inner class declarations of 
           * template classes correctly
           */
          char print(std::ostream& ost) const {
            ost << "Shape<" << DIM << ">::iterator(" << "current="
                << this->m_current << " currentTuple=" << this->m_value << ")";
            return '\0';
          }
          
        private:
          
          void advance(int n = 1) {
            assert(this->m_current != -1);
            // Don't increment if at the end the end.
            assert(this->m_current < this->m_coll.orthotope()->count());
            // Don't increment past the end.

            // Precalculate increment
            this->m_current += n;
            this->m_value[DIM - 1] += n;
            
            if (this->m_value[DIM - 1] >= this->m_coll.orthotope()->high()[DIM - 1]) {
              // The end ofleast signifcant coordinate was exceeded,
              // so recalculate value (the current tuple).
              if (this->m_current < this->m_coll.orthotope()->count())
                this->m_value = this->m_coll.coord(this->m_current);
              else
                this->m_current = -1; // The end was reached.
            }
            
            HTA_DEBUG(3, "Shape::Iterator::advance this=" << this
                << ", current=" << this->m_current << ", current tuple="
                << this->m_value);
          }
      };

    public:
      // Shape typedef's
      typedef PREDICATE predicate;
      typedef boost::filter_iterator<predicate, ShapeIterator> iterator;

    protected:
      /// Pointer to the orthotope described by shape.
      Orthotope<DIM>* m_orthotope;
      /// Shared pointer to predicate object; it defines which elements are present. TODO does it need to be a pointer?
      boost::shared_ptr<predicate> m_pred;
      /// Linear step is used to calculate linear indices
      Tuple<DIM> m_linear_step;

    private:
      void
      init_linear_step_()
      {
    	Tuple<DIM> h = m_orthotope.high();
    	Tuple<DIM> l = m_orthotope.low();
        this->m_linear_step[DIM - 1] = 1;
        for (int dim = DIM - 1; dim > 0; --dim)
          this->m_linear_step[dim - 1] = (h[dim] - l[dim]) * this->m_linear_step[dim];
      }

    public:
      
      /// Default constructor not allowed.
      Shape() {
        assert(false);
      }
      
      /// Constructor
      Shape(const Orthotope<DIM>* ortho, const boost::shared_ptr<predicate> pred) :
        m_orthotope(ortho), m_pred(pred) {
    	init_linear_step_();
      }
      
      /// Copy constructor
      Shape(const Shape<DIM, PREDICATE>& s) :
        m_orthotope(s.m_orthotope), m_pred(s.m_pred) {
    	init_linear_step_();
      }
      
      virtual ~Shape() {
      }
      
      // Assignment operator
      inline Shape<DIM, PREDICATE>& operator =(const Shape<DIM, PREDICATE>& s) {
        this->m_orthotope = s.m_orthotope;
        this->m_pred = s.m_pred;
        
        return *this;
      }
      
      // Returns a pointer to the orthotope described by this Shape.
      virtual inline const Orthotope<DIM>* orthotope() const {
        return this->m_orthotope;
      }
      
      virtual inline bool included(const Tuple<DIM>& element_index) const {
        return this->m_orthotope->contains(element_index)
            && (*m_pred)(element_index);
      }

      /**
       * Ordinal value of Tuple. Ordinal value does not include offset
       * and does not consider step. If the shape starts at (5, 3), then (5, 4)
       * has ord 1 on row-major. 
       * The ordinal value of orthotope()->low() is always 0.
       * The ordinal value of orthotope()->high() is always <= linearCard().
       */
      virtual inline size_t ord(const Tuple<DIM>& coord) const {
        assert(this->m_orthotope->contains(coord));
        return static_cast<size_t>(VectorOps<Tuple<DIM>, DIM>::dotProduct((coord
            - this->m_orthotope->low()), this->m_linear_step));
      }
      
      // Returns the element index of the element referred to by linear_index
      virtual Tuple<DIM> coord(size_t linear_index) const {
        assert(linear_index >= 0 && linear_index < this->m_orthotope->count());
        
        Tuple<DIM> element_index;
        
        // start with most significant dimension 0.
        for (unsigned int dim = 0; linear_index > 0 && dim < DIM; ++dim) {
          element_index[dim] = linear_index / this->m_linear_step[dim];
          linear_index -= element_index[dim] * this->m_linear_step[dim];
        }
        
        assert(linear_index == 0);
        // Sanity check

        // Offset the value so it is inside shape.
        element_index += this->m_orthotope->low();
        
        return element_index;
      }
      
      /**
       * Iterator factory
       */
      inline iterator begin() const {
        return iterator(*(this->m_pred), ShapeIterator(*this),
                        ShapeIterator(*this, -1));
      }
      
      /**
       * Iterator factory
       */
      inline iterator end() const {
        return iterator(*(this->m_pred), ShapeIterator(*this, -1),
                        ShapeIterator(*this, -1));
      }
  };
  
  template<int DIM, class PREDICATE> std::ostream& operator <<(
                                                               std::ostream& out,
                                                               const Shape<DIM, PREDICATE>& s) {
    out << "Shape<" << DIM << ">(" << " @=" << &s << " low=" << s.low()
        << " high=" << s.high() << " size=" << s.size() << " linearStep="
        << s.linear_step() << ")";
    return out;
  }

}
; // end of namespace TiledArray

#endif // SHAPE_H__INCLUDED
