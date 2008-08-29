/*
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
#include <predicate.h>
#include <algorithm>
#include <boost/smart_ptr.hpp>
#include <boost/iterator/filter_iterator.hpp>

namespace TiledArray {

  /**
   * AbstractShape class defines a subspace of an Orthotope. AbstractShapeIterator can be used to iterate
   * over AbstractShape.
   */
  template <unsigned int DIM>
  class AbstractShape {
  public:
    
    /// Abstract interface to Shape::iterator
    class Iterator {
    public:
      virtual ~Iterator();
      virtual Iterator& operator++() =0;
    };
    
    /// Maps element to a linearized index ("ordinal"). Computation of the ordinal assumes that DIM-1 is the least significant dimension.
    virtual size_t ord(const Tuple<DIM>& element_index) const = 0;
    virtual Tuple<DIM> coord(size_t linear_index) const = 0;
    virtual bool includes_tile(const Tuple<DIM>& tile_index) const = 0;
    virtual bool includes(const Tuple<DIM>& element_index) const = 0;
    virtual const Orthotope<DIM>* orthotope() const = 0;
    virtual const TupleFilter<DIM>* const tile_pred() const = 0;
    virtual const TupleFilter<DIM>* const element_pred() const = 0;
    
    /// Abstract iterators must be returned via pointers (compare to Shape::begin())
    /// TODO cannot be implemented yet because ShapeIterator doesn't compile
    //virtual boost::shared_ptr<Iterator> abegin() const =0;
  };
  
  /**
   * Shape is a templated implementation of AbstractShape. Shape
   * provides an input iterator, which iterates an ordinal value and tuple index
   * simultaneously.
   */
  template <unsigned int DIM, class TILE_PREDICATE = OffTupleFilter<DIM>, class ELEMENT_PREDICATE = OffTupleFilter<DIM> >
  class Shape : public AbstractShape<DIM> {
    private:
      /// Iterator spec for ShapeIterator class.
      class ShapeIteratorSpec {
        public:
          typedef int                                             iterator_type;
          typedef Shape<DIM, TILE_PREDICATE, ELEMENT_PREDICATE>   collection_type;
          typedef std::input_iterator_tag                         iterator_category;
          typedef Tuple<DIM>                                      value;
          typedef value*                                          pointer;
          typedef const value*                                    const_pointer;
          typedef value&                                          reference;
          typedef const value&                                    const_reference;
      };

      /** 
       * ShapeIterator is an input iterator that iterates over Shape. The
       * iterator assumes row major access and DIM-1 is the least
       * significant dimension.
       */
      class ShapeIterator : public Iterator<ShapeIteratorSpec> {
        public:
          // Iterator typedef
          typedef typename Iterator<ShapeIteratorSpec>::iterator_type     iterator_type;
          typedef typename Iterator<ShapeIteratorSpec>::collection_type   collection_type;
          typedef typename Iterator<ShapeIteratorSpec>::iterator_category iterator_catagory;
          typedef typename Iterator<ShapeIteratorSpec>::reference         reference;
          typedef typename Iterator<ShapeIteratorSpec>::const_reference   const_reference;
          typedef typename Iterator<ShapeIteratorSpec>::pointer           pointer;
          typedef typename Iterator<ShapeIteratorSpec>::const_pointer     const_pointer;
          typedef typename Iterator<ShapeIteratorSpec>::value             value;
          typedef typename Iterator<ShapeIteratorSpec>::difference_type   difference_type;

        private:
          /// Lower limit of iterator
          Tuple<DIM> m_low;
          /// Upper limit of iterator
          Tuple<DIM> m_high;
          /// Number of elements in each dimention
          Tuple<DIM> m_size;
          /// Cached values for calculating linear offset
          Tuple<DIM> m_linear_step;
          /// number of elements contained by the iterator
          iterator_type m_nelements;
          /// current value of the iterator
          value m_value;
          /// Flag when m_value needs to be calculated
          bool m_calc_value;

          /// Default construction not allowed (required by forward iterators)
          ShapeIterator();
          
          ///  Initialize linear step data.
          void init_linear_step_() {
            m_linear_step[DIM - 1] = 1;
            for (int dim = DIM - 1; dim > 0; --dim)
              m_linear_step[dim - 1] = (m_high[dim] - m_low[dim]) * m_linear_step[dim];
          }         

          /// Returns the element index of the element referred to by linear_index
          void calc_coord_() const {
            iterator_type linear_index = this->m_current;
            
            // start with most significant dimension 0.
            for (unsigned int dim = 0; linear_index > 0 && dim < DIM; ++dim) {
              m_value[dim] = linear_index / m_linear_step[dim];
              linear_index -= m_value[dim] * m_linear_step[dim];
            }
            
            // Sanity check
            assert(linear_index == 0);
            
            // Offset the value so it is inside shape.
            m_value += m_low;
            
            // flag value as calculated.
            m_calc_value = false;
          }

          /// Advance the iterator n increments
          void advance_(int n = 1) {
            // Don't increment if at the end.
            assert(this->m_current != -1);
            // Don't increment past the end.
            assert(this->m_current < m_nelements);

            // Precalculate increment
            this->m_current += n;
            m_value[DIM - 1] += n;
            
            if (m_value[DIM - 1] >= m_high[DIM - 1]) {
              // The end ofleast signifcant coordinate was exceeded,
              // so flag m_value to be recalculated.
              m_calc_value = true;
              if (this->m_current < m_nelements)
                this->m_current = -1; // The end was reached.
            }
          }

        public:
          
          /// Main constructor function
          ShapeIterator(const Tuple<DIM>& low, const Tuple<DIM>& high, const iterator_type& cur) :
            Iterator<ShapeIteratorSpec>(cur),
            m_low(low),
            m_high(high),
            m_size(high - low),
            m_linear_step(0),
            m_nelements(VectorOps<Tuple<DIM>, DIM>::selfProduct(high - low)),
            m_value(0),
            m_calc_value(false)
          {
        	init_linear_step_();
        	calc_coord_(); 
          }
          
          /// Copy constructor (required by all iterators)
          ShapeIterator(const ShapeIterator& it) :
            Iterator<ShapeIteratorSpec>(it.m_current),
            m_low(it.m_low),
            m_high(it.m_high),
            m_size(it.m_size),
            m_linear_step(it.m_linear_step),
            m_nelements(it.m_nelements),
            m_value(it.m_value),
            m_calc_value(it.m_calc_value)
          {}
          
          /// Prefix increment (required by all iterators)
          ShapeIterator& operator ++() {
            assert(this->m_current != -1);
            advance_();
            return *this;
          }
          
          /// Postfix increment (required by all iterators)
          ShapeIterator operator ++(int) {
            assert(this->m_current != -1);
            ShapeIterator tmp(*this);
            advance_();
            return tmp;
          }
          
          /// Equality operator (required by input iterators)
          bool operator ==(const ShapeIterator& it) const {
            return (this->base() == it.base());
          }
          
          /// Inequality operator (required by input iterators)
          bool operator !=(const ShapeIterator& it) const {
            return ! (this->operator ==(it));
          }
          
          /// Dereference operator (required by input iterators)
          const value& operator *() const {
            assert(this->m_current != -1);
            assert(this->m_current < m_nelements);
               
            // Check to se if m_value needs to be calculated
            if(m_calc_value)
              calc_coord_();
 
            return m_value;
          }
          
          /// Dereference operator (required by input iterators)
          const value& operator ->() const {
            return (operator *());
          }
          
          typename Orthotope<DIM>::index_t ord() const {
            assert(this->m_current != -1);
            assert(this->m_current < m_nelements);
            return static_cast<typename Orthotope<DIM>::index_t>(this->m_current);
          }
          
          /**
           * This is for debugging only. Not done in an overload of operator<<
           * because it seems that gcc 3.4 does not manage inner class
           * declarations of template classes correctly.
           */
          char print(std::ostream& ost) const {
            ost << "Shape<" << DIM << ">::iterator(" << "current="
                << this->m_current << " currentTuple=" << m_value << ")";
            return '\0';
          }
      }; // end of ShapeIterator

    public:
      // Shape typedef's
      typedef TILE_PREDICATE tile_predicate;
      typedef ELEMENT_PREDICATE element_predicate;
      typedef boost::filter_iterator<tile_predicate, ShapeIterator> tile_iterator;
      typedef boost::filter_iterator<element_predicate, ShapeIterator> element_iterator;

    protected:
      /// Pointer to the orthotope described by shape.
      Orthotope<DIM>* m_orthotope;
      /// Predicate object; it defines which tiles are present.
      tile_predicate m_tpred;
      /// Predicate object; it defines which elements are present.
      element_predicate m_epred;
      /// Linear step is a set of cached values used to calculate linear offsets.
      Tuple<DIM> m_linear_step;

    private:
      
      /// Wraps iterator for delivery via AbstractShape::Iterator
      class AbstractIterator : public AbstractShape<DIM>::Iterator {
        public:
          typedef typename AbstractShape<DIM>::Iterator parent_type;
          typedef typename Shape::tile_iterator wrapped_type;
          
          AbstractIterator(const wrapped_type& iter) : iter_(iter) {}
          
          /// Implementation of AbstractShape<DIM>::Iterator::operator++
          parent_type& operator++() {
            ++iter_;
            return *this;
          }
          
        private:
          wrapped_type iter_;
      };

      ///  Initialize linear step data.
      void init_linear_step_() {
    	Tuple<DIM> h = m_orthotope->high();
    	Tuple<DIM> l = m_orthotope->low();
        m_linear_step[DIM - 1] = 1;
        for (int dim = DIM - 1; dim > 0; --dim)
          m_linear_step[dim - 1] = (h[dim] - l[dim]) * m_linear_step[dim];
      }
      
      /// Default constructor not allowed
      Shape();

    public:
      
      /// Constructor
      Shape(Orthotope<DIM>* ortho, const tile_predicate& tpred = tile_predicate(), const element_predicate& epred = element_predicate()) :
        m_orthotope(ortho),
        m_tpred(tpred),
        m_epred(epred),
        m_linear_step(0)
      {
    	  init_linear_step_();
      }
      
      /// Copy constructor
      Shape(const Shape<DIM, TILE_PREDICATE, ELEMENT_PREDICATE>& s) :
        m_orthotope(s.m_orthotope),
        m_tpred(s.m_tpred),
        m_epred(s.m_epred),
        m_linear_step(s.m_linear_step)
      {}
      
      ~Shape() {
      }
      
      /// Assignment operator
      Shape<DIM, TILE_PREDICATE, ELEMENT_PREDICATE>& operator =(const Shape<DIM, TILE_PREDICATE, ELEMENT_PREDICATE>& s) {
        m_orthotope = s.m_orthotope;
        m_tpred = s.m_tpred;
        m_epred = s.m_epred;
        m_linear_step = s.m_linear_step;

        return *this;
      }
      
      /// Returns a pointer to the orthotope that supports this Shape.
      virtual const Orthotope<DIM>* orthotope() const {
        return m_orthotope;
      }

      /// Returns a pointer to the tile predicate
      virtual const TupleFilter<DIM>* const tile_pred() const {
        return &m_tpred;
      }

      /// Returns a reference to the element predicate
      virtual const TupleFilter<DIM>* const element_pred() const {
    	  return &m_epred;
      }

      /**
       * Returns true if the value at tile_index is non-zero and is contained
       * by the Shape.
       */
      virtual bool includes_tile(const Tuple<DIM>& tile_index) const {
        return (tile_index < m_orthotope->tile_size()) && m_tpred(tile_index);
      }      

      /**
       * Returns true if the value at element_index is non-zero and is contained
       * by the Shape.
       */
      virtual bool includes(const Tuple<DIM>& element_index) const {
    	if(!includes_tile(m_orthotope->tile(element_index)))
    		return false;

        return m_orthotope->includes(element_index) && m_epred(element_index);
      }
      
      /**
       * Ordinal value of Tuple. Ordinal value does not include offset
       * and does not consider step. If the shape starts at (5, 3), then (5, 4)
       * has ord 1 on row-major. 
       * The ordinal value of orthotope()->low() is always 0.
       * The ordinal value of orthotope()->high() is always <= linearCard().
       */
      virtual size_t ord(const Tuple<DIM>& coord) const {
        assert(m_orthotope->includes(coord));
        return static_cast<size_t>(VectorOps<Tuple<DIM>, DIM>::dotProduct((coord
            - m_orthotope->low()), m_linear_step));
      }
      
      /// Returns the element index of the element referred to by linear_index
      virtual Tuple<DIM> coord(size_t linear_index) const {
        assert(linear_index >= 0 && linear_index < m_orthotope->nelements());
        
        Tuple<DIM> element_index;
        
        // start with most significant dimension 0.
        for (unsigned int dim = 0; linear_index > 0 && dim < DIM; ++dim) {
          element_index[dim] = linear_index / m_linear_step[dim];
          linear_index -= element_index[dim] * m_linear_step[dim];
        }
        
        assert(linear_index == 0);
        // Sanity check

        // Offset the value so it is inside shape.
        element_index += this->m_orthotope->low();
        
        return element_index;
      }

      ///Permute shape
      void permute(const Tuple<DIM>& perm) {
    	m_tpred.permute(perm);
    	m_epred.permute(perm);
    	m_orthotope->permute(perm);
    	init_linear_step_();
      }

      /// Tile iterator factory
      tile_iterator tile_begin() const {
        return tile_iterator(m_tpred, ShapeIterator(Tuple<DIM>(0), m_orthotope->tile_size()),
        		ShapeIterator(Tuple<DIM>(0), m_orthotope->tile_size(), -1));
      }
      
      /// Tile iterator factory
      tile_iterator tile_end() const {
        return tile_iterator(m_tpred, ShapeIterator(Tuple<DIM>(0), m_orthotope->tile_size(), -1),
        		ShapeIterator(Tuple<DIM>(0), m_orthotope->tile_size(), -1));
      }

      /// Element Iterator factory
      element_iterator begin() const {
        return element_iterator(m_tpred, ShapeIterator(m_orthotope->low(), m_orthotope->high()),
        		ShapeIterator(m_orthotope->low(), m_orthotope->high(), -1));
      }
      
      /// Iterator factory
      element_iterator end() const {
        return element_iterator(m_tpred, ShapeIterator(m_orthotope->low(), m_orthotope->high(), -1),
        		ShapeIterator(m_orthotope->low(), m_orthotope->high(), -1));
      }

      /// Element Iterator factory for a single tile
      element_iterator begin(const Tuple<DIM>& tile_index) const {
        return element_iterator(m_tpred, ShapeIterator(m_orthotope->low(tile_index), m_orthotope->high(tile_index)),
        		ShapeIterator(m_orthotope->low(tile_index), m_orthotope->high(tile_index), -1));
      }
      
      /// Element iterator factory for a single tile
      element_iterator end(const Tuple<DIM>& tile_index) const {
        return element_iterator(m_tpred, ShapeIterator(m_orthotope->low(tile_index), m_orthotope->high(tile_index), -1),
        		ShapeIterator(m_orthotope->low(tile_index), m_orthotope->high(tile_index), -1));
      }
     
     // currently doesn't compile because iterator is broken 
#if 0
      /// Implements AbstractShape::abegin()
      boost::shared_ptr<typename AbstractShape<DIM>::Iterator> abegin() const {
        return boost::shared_ptr<typename AbstractShape<DIM>::Iterator>(
            new AbstractIterator(this->tile_begin())
            );
      }
#endif
      
  };
  
  template<unsigned int DIM, class PREDICATE>
  std::ostream& operator <<(std::ostream& out,const Shape<DIM, PREDICATE>& s) {
    out << "Shape<" << DIM << ">(" << " @=" << &s << " orth="
        << *(s.orthotope()) << " )";
    return out;
  }

}
; // end of namespace TiledArray

#endif // SHAPE_H__INCLUDED
