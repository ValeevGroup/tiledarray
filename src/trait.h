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
 * Version: $Id: Traits.h,v 1.16 2006/10/23 11:41:32 fraguela Exp $
 * Authors: Ganesh Bikshandi, Christoph von Praun
 */

#ifndef  __TRAITS_H__INCLUDED
#define  __TRAITS_H__INCLUDED

namespace TiledArray {
  
  /* serial HTA implementation */
  template <typename T, unsigned int DIM> class LocalDenseTrait {
      typedef T ValueType;
      typedef OffTupleFilter<DIM> PredicateType;
      typedef Shape<DIM, PredicateType> ShapeType;

      typedef LocalAllocator<T> AllocatorType;
      typedef Math<T> MathKernalType;

      typedef LocalDenseTrait<T,DIM> TraitType;
  };

} // namespace TiledArray

#endif // __SERIAL_TRAIT_H__INCLUDED
