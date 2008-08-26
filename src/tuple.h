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
 * Version: $Id: Tuple.h,v 1.65 2007/01/11 01:11:32 bikshand Exp $
 * Authors: Ganesh Bikshandi, Christoph von Praun
 */

/*
 * Updated: 7/16/2008
 * Author: Justus Calvin
 * 
 * Changes:
 * - Data is now stored in an STL vector.
 * - Modified the default class to handle n dimentions.
 * - Addes STL style iterator functionallity.
 * - Removed seq.
 * - Added Tiled Array namespace
 */

#ifndef TUPLE_H__INCLUDED
#define TUPLE_H__INCLUDED

#include <assert.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <VectorOps.h>

namespace TiledArray {
  
  template <unsigned int DIM> class Tuple {
  public:
	  // Typedefs
	  typedef int value_t;
      typedef std::vector<value_t>::iterator iterator;
      typedef std::vector<value_t>::const_iterator const_iterator;
      
  private:

	  /// vector of DIM-dimentional point
	  std::vector<value_t> m_values;

  public:
      


      inline explicit Tuple() :
        m_values(DIM, 0) {
      }
      
      inline explicit Tuple(const value_t value) :
        m_values(DIM, value) {
      }
      
      inline explicit Tuple(const value_t* values) :
        m_values(values, values + DIM) {
      }
      
      inline explicit Tuple(const std::vector<value_t>& values) :
        m_values(DIM, values) {
      }
      
      /** 
       * Copy constructor 
       */
      inline Tuple(const Tuple<DIM>& tup) :
        m_values(tup.m_values) {
      }
      
      // STL style coordinate iterator functions

      // Returns an interator to the first coordinate
      iterator begin() {
        return m_values.begin();
      }
      
      // Returns a constant iterator to the first coordinate. 
      const_iterator begin() const {
        return m_values.begin();
      }
      
      // Returns an iterator to one element past the last coordinate.
      iterator end() {
        return m_values.end();
      }
      
      // Returns a constant iterator to one element past the last coordinate.
      const_iterator end() const {
        return m_values.end();
      }
      
      // Arithmatic operators
      inline Tuple<DIM> operator +(const Tuple<DIM>& tup) const {
        Tuple<DIM> ret;
        VectorOps<std::vector<value_t>, DIM>::add(ret.m_values, m_values,
                                        tup.m_values);
        return ret;
      }
      
      inline Tuple<DIM>& operator +=(const Tuple<DIM>& tup) {
        VectorOps<std::vector<value_t>, DIM>::addIn(m_values, tup.m_values);
        
        return (*this);
      }
      
      inline Tuple<DIM>& operator -=(const Tuple<DIM>& tup) {
        VectorOps<std::vector<value_t>, DIM>::subIn(m_values, tup.m_values);
        
        return (*this);
      }
      
      inline Tuple<DIM> operator +(const value_t val) const {
        Tuple<DIM> ret(val);
        VectorOps<std::vector<value_t>, DIM>::addIn(m_values, ret);
        return ret;
      }
      
      inline Tuple<DIM> operator -(const Tuple<DIM>& other) const {
        Tuple<DIM> ret;
        VectorOps<std::vector<value_t>, DIM>::sub(ret.m_values, m_values,
                                        other.m_values);
        return ret;
      }
      
      inline Tuple<DIM> operator -(value_t val) const {
        Tuple<DIM> ret(val);
        VectorOps<std::vector<value_t>, DIM>::subIn(m_values, ret);
        return ret;
      }
      
      inline Tuple<DIM> operator -() const {
        Tuple<DIM> ret;
        VectorOps<std::vector<value_t>, DIM>::uminus(ret.m_values, m_values);
        return ret;
      }
      
      inline Tuple<DIM> operator *(const Tuple<DIM>& other) const {
        Tuple<DIM> ret;
        VectorOps<std::vector<value_t>, DIM>::mult(ret.m_values, m_values,
                                         other.m_values);
        return ret;
      }
      
      inline Tuple<DIM> operator /(const Tuple<DIM>& other) const {
        Tuple<DIM> ret;
        VectorOps<std::vector<value_t>, DIM>::div(ret.m_values, m_values,
                                        other.m_values);
        return ret;
      }
      
      inline Tuple<DIM> operator %(const Tuple<DIM>& tup) const {
        Tuple<DIM> ret;
        VectorOps<std::vector<value_t>, DIM>::mod(ret.m_values, m_values,
                                        tup.m_values);
        return ret;
      }
      
      // Comparison Operators
      inline bool operator ==(const Tuple<DIM>& tup) const {
        return VectorOps<std::vector<value_t>, DIM>::equal(m_values, tup.m_values);
      }
      
      inline bool operator !=(const Tuple<DIM>& tup) const {
        return !(operator ==(tup));
      }
      
      inline bool operator <(const Tuple<DIM>& tup) const {
        return VectorOps<std::vector<value_t>, DIM>::less(m_values, tup.m_values);
      }
      
      inline bool operator <=(const Tuple<DIM>& tup) const {
        return (VectorOps<std::vector<value_t>, DIM>::less(m_values, tup.m_values)
            || VectorOps<std::vector<value_t>, DIM>::equal(m_values, tup.m_values));
      }
      
      inline bool operator>(const Tuple<DIM>& tup) const {
        return !(operator <=(tup));
      }
      
      inline bool operator >=(const Tuple<DIM>& tup) const {
        return !(operator <(tup));
      }
      
      inline Tuple<DIM>& operator =(const Tuple<DIM> & tup) {
        std::copy(tup.m_values.begin(), tup.m_values.end(),
        		m_values.begin());
        
        return (*this);
      }
      
      inline const value_t& operator[](unsigned int dim) const {
#ifdef NDEBUG
    	return m_values.at(dim);
#else
        return m_values[dim];
#endif
      }
      
      inline value_t& operator[](unsigned int dim) {
#ifdef NDEBUG
    	return m_values.at(dim);
#else
        return m_values[dim];
#endif
      }
      
      /**
       * mask a value in a tuple with the given value (default 0).
       *
       * @param dim    the dimension to mask
       * @param value  the value to put into the dimension
       */
      Tuple<DIM> mask(unsigned int dim, value_t value = 0) const {
        Tuple<DIM> ret = *this;
        ret[dim] = value;
        return ret;
      }
      
      // Forward permutation of set by one.
      Tuple<DIM>& permute() {
        value_t temp = m_values[0];
        
        for (unsigned int dim = 0; dim < m_values.size() - 1; ++dim)
          m_values[dim] = m_values[dim + 1];
        
        m_values[m_values.size() - 1] = temp;
        
        return (*this);
      }
      
      // Reverse permutation of set by one.
      Tuple<DIM>& reverse_permute() {
        unsigned int dim = m_values.size() - 1;
        
        // Store the value of the last element
        value_t temp = m_values[dim];
        
        // shift all elements to the left
        for (; dim > 0; --dim)
          m_values[dim] = m_values[dim - 1];
        
        // place the value of the last element in the first.
        m_values[0] = temp;
        
        return (*this);
      }
      
      /**
       * User defined permutation of a tuple.
       *
       * @param perm - Tuple must include each index number (0, 1, ..., DIM-1) once and only once.
       */
      Tuple<DIM>& permute(const Tuple<DIM>& perm) {
#if (TA_DLEVEL >= 0)
        // Ensure each index is present and listed only once.
        unsigned int dim_count = 0;
        for (unsigned int dim = 0; dim < m_values.size(); ++dim)
          dim_count += std::count(perm.begin(), perm.end(), dim);
        
        // Incorrect permutation, do nothing.
        assert(dim_count == DIM);
#endif
        
        Tuple<DIM> temp(*this);
        for (unsigned int dim = 0; dim < DIM; ++dim)
          m_values[dim] = temp.m_values[perm.m_values[dim]];
        
        return (*this);
      }
      
    private:
      
      /** forbid heap allocation */
      void* operator new(size_t size) throw () {
        assert(false);
        return NULL;
      }
      
      /**  forbid heap allocation */
      void operator delete(void* to_delete) {
        assert(false);
      }
      
  };
  
  template <unsigned int DIM> std::ostream& operator<<(std::ostream& output,
                                                       const Tuple<DIM>& tup) {
    output << "(";
    for (unsigned int dim = 0; dim < DIM - 1; ++dim)
      output << tup[dim] << ", ";
    output << tup[DIM-1] << ")";
    return output;
  }
  ;

}
; // end of namespace TiledArray

#endif /*TUPLE_H_*/
