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
 * Version: $Id: Array.h,v 1.241 2007/08/13 16:14:42 fraguela Exp $
 * Authors: Christoph von Praun
 */

#ifndef  __ARRAY_H__
#define  __ARRAY_H__

#include <cassert>
#include <vector>
#include <algorithm>

#include "AbstractArray.h"
#include "ConstDistribution.h"
#include "CustomDistribution.h"
#include "GenericDistribution.h"
#include "PermutedDistribution.h"
#include "Distribution.h"
#include "MemMapping.h"
#include "Operators.h"
#include "Order.h"
#include "Shape.h"
#include "Traits.h"
#include "Tuple.h"
#include "debug/Timers.h"
#include "debug/Tracing.h"
#include "util/RefCounted.h"
#include "util/TypeHelper.h"
#include "util/VectorOps.h"
#include "util/Misc.h"
#include "MemMapping.h"
#include "LocalMemMapping.h"
#include "IrregularMemMapping.h"


namespace TILED_ARRAY_NAMESPACE
{

/** 
 * Abstract superclass of HTAs. This is the superclass of all arrays and basically serves as 
 * a container for common implementation of shared memory distributed arrays.
 */

template <typename T, int DIM, typename TRAIT> class HTA;

template<typename T, int DIM, typename TRAIT>
class Array : public AbstractArray<T, DIM>, public RefCounted {

protected:
  
  /**
   * @param dist  is optional -- it might not be known at the time whan this 
   *              array is allocated (could be a shell array, i.e an array where the 
   *              tiles are not initialized)
   */
  Array (size_t levels, 
	 const Shape<DIM>& s, 
	 const typename TRAIT::TypeAllocator& alloc,
	 const MemMapping<DIM>* mmap, 
	 int home) :
    RefCounted(),
    memMapping__(mmap), 
    level_(levels),
    home_(home),
    shape_(s), 
    alloc_(alloc), 
    dist__(NULL), 
    parent__(NULL),
    numTiles_(0),
    localTiles_(NULL)
  {
    if (mmap != NULL) {
      mmap->refCount_(1);
      // assert (mmap->tiling()[LEVEL] == s.size());
      // might be violated if this array is a view on a physical array that is different in size
    }
    
    // initialize the distribution
    if (this->level() > 0 && home != Traits::Default::nullPlace()) 
      dist__ = &ConstDistribution<DIM>::get(home);
    
    if(level_ == 0) is_leaf_ = true;
    else is_leaf_ = false;

     if (!this->isDistributed() && this->level() > 0)
        {
           numTiles_ = s.card();
           localTiles_ = new int [numTiles_];
           for (int i = 0; i < numTiles_; i++)
             localTiles_[i] = i;
         }

    // for arrays that are distributed, the distribution, numTiles_ and localTiles_ are initialized in initDistribution
  }

private:

  /** 
   * Copy constructor - declare explicitly to avoid that the compiler 
   * generates one - forbidden 
   */
  Array (const Array<T, DIM, TRAIT>& rhs)
  {
    assert (false);
  }
  
public:

  /**
   * Virtual destructor to make sure that 
   * subclass destructors are called.
   */
  virtual ~Array() 
  {
    // de-alloc distribution!
    if (memMapping__ != NULL && memMapping__->refCount_(-1) == 0)
      delete memMapping__;

    if (dist__ != NULL && dist__->refCount_(-1) == 0)
      {
	delete dist__;
      }
    if (localTiles_ != NULL)
      { 
         delete [] localTiles_;
         localTiles_ = NULL;
      }
  }

  /** accessor method MATLAB style */
  typename TRAIT::TypeImpl* tileAt (int n, const Shape<DIM>* sv) const  
  {
    assert (this->level() > 0);
    return actual_().tileAt(n, sv);
  }

  typename TRAIT::TypeImpl* tileAt (const Shape<DIM>& s) const
  {
    return actual_().tileAt(s);
  }

  /** leaf convolution operator */
  void lconv (const T* weights, 
	      Array<T, DIM, TRAIT>& dst) const 
  {
    if(this->isLeaf())
    {
      this->lconv_leaf_(weights,dst);
    }
    else{
      assert (this->level() > 0);
      // this is a limitation of the current implementation
      assert(this->shape() == dst.shape());
    
      int ntiles = this->shape().card();
      for (int i=0; i < ntiles; i++) {
	typename TRAIT::TypeTileImpl* dst_tile = dst.tileAt(i);
	if (dst_tile->Array<T, DIM, TRAIT>::isLocal()) {
	  const typename TRAIT::TypeTileImpl* tile = this->tileAt(i);
	  tile->lconv(weights, *dst_tile);
	}
      }
    }
  }
  

 public:
  /** convolution operator */
  void lconv_leaf_ (const T* weights, 
		    Array<T, DIM, TRAIT>& dst) const 
  {
    // this is a limitation of the current implementation
    assert (weights != NULL);
    assert (this->shape().size() == dst.shape().size());
    assert (this->shape().isContiguous());    
    assert (this->memMap().isContiguous());    
    assert (dst.shape().isContiguous());
    assert (dst.memMap().isContiguous());

    switch (DIM) {
    case 1:
      lconv1D_(weights[0], weights[1], this->actual_(), dst.actual_()); 
      break;
    case 2:
      lconv2D_(weights[0], weights[1], weights[2], this->actual_(), dst.actual_()); 
      break;
    case 3:
      lconv3D_(weights[0], weights[1], weights[2], weights[3], this->actual_(), dst.actual_()); 
      break;
    default:
      assert (false);
    } 
    return;
  }


private:

  // specialization of convolution  for DIM==1 
  static void lconv1D_ (const T w0, const T w1, 
			const typename TRAIT::TypeImpl& arg,
			typename TRAIT::TypeImpl& dst) 
  {
    assert (DIM == 0);

    int size = dst.shape().size()[0];
    int step_dst = dst.memMap().linearStep()[0] * dst.shape().step()[0];
    int step_arg = arg.memMap().linearStep()[0] * arg.shape().step()[0];
    
    for (int i = 1; i < size-1; ++ i) {
      T u0 = arg.scalarAt_(i * step_arg);
      T u1 = arg.scalarAt_(i-1 * step_arg) + arg.scalarAt_(i+1 * step_arg);
      dst.scalarAt_(i * step_dst) = u0 * w0 +  u1 * w1;
    }
  }

  static void lconv2D_ (const T w0, const T w1, const T w2,
			const typename TRAIT::TypeImpl& arg,
			typename TRAIT::TypeImpl& dst) 
  {
    assert (DIM == 2);
    assert (arg.shape().step() == Tuple<DIM>::one);
    assert (dst.shape().step() == Tuple<DIM>::one);

    int size_0 = arg.shape().size()[0]; // most significant
    int size_1 = arg.shape().size()[1];
    T* a = arg.raw();
    T* d = dst.raw();

    for (int i = 1; i < size_0 - 1; ++ i) {
      for (int j = 1; j < size_1 - 1; ++ j) {
	T u0 = 
	  a[i     * size_1 + j];
	T u1 = 
	  a[(i-1) * size_1 + j] +
	  a[i     * size_1 + (j-1)] +
	  a[(i+1) * size_1 + j] +
	  a[i     * size_1 + (j+1)];
	T u2 = 
	  a[(i-1) * size_1 + (j-1)] + 
	  a[(i-1) * size_1 + (j+1)] +
	  a[(i+1) * size_1 + (j-1)] + 
	  a[(i+1) * size_1 + (j+1)];
	d[i * size_1 + j] = w0 * u0 + w1 * u1 + w2 * u2;
      }
    }
  }

  static void lconv3D_ (const T w0, const T w1, const T w2, const T w3,
			const typename TRAIT::TypeImpl& arg,
			typename TRAIT::TypeImpl& dst) 
  {
    assert (DIM == 3);
    assert (arg.shape().step() == Tuple<DIM>::one);
    assert (dst.shape().step() == Tuple<DIM>::one);
    
    const Tuple<DIM> step_arg = arg.memMapping__->linearStep() * arg.shape().step();
    int step_arg_0 = step_arg[0];
    int step_arg_1 = step_arg[1];
    int step_arg_2 = step_arg[2];

    const Tuple<DIM> step_dst = dst.memMapping__->linearStep() * dst.shape().step();
    int step_dst_0 = step_dst[0];
    int step_dst_1 = step_dst[1];

    int size_0 = arg.shape().size()[0]; // most significant
    int size_1 = arg.shape().size()[1];
    int size_2 = arg.shape().size()[2];
    
    double* u1 = new double [size_2];
    double* u2 = new double [size_2];
    
    const T* a = arg.raw();
    T* d = dst.raw();
    
    for (int i = 1; i < size_0 - 1; i++) {
      for (int j = 1; j < size_1 - 1; j++) {
	
        for (int k = 0; k < size_2; k++) {
	  // can the compiler hoist + k from the index computation here ?
	  u1[k] = 
	    a[i * step_arg_0    + (j-1) * step_arg_1 + k] + 
	    a[i * step_arg_0   + (j+1) * step_arg_1  + k] + 
	    a[(i-1) * step_arg_0 + j * step_arg_1    + k] + 
	    a[(i+1) * step_arg_0 + j * step_arg_1    + k];
	  u2[k] = 
	    a[(i-1) * step_arg_0 + (j-1) * step_arg_1 + k] +
	    a[(i-1) * step_arg_0 + (j+1) * step_arg_1 + k] + 
	    a[(i+1) * step_arg_0 + (j-1) * step_arg_1 + k] + 
	    a[(i+1) * step_arg_0 + (j+1) * step_arg_1 + k];
	}
	
        for (int k = 1; k < size_2 - 1; k++) {
	  d[i * step_dst_0  + j * step_dst_1 + k] = 
	    w0 * a[i * step_arg_0 + j * step_arg_1 + k]  +
	    w1 * (u1[k] + a[i * step_arg_0 + j * step_arg_1 + k-1] + a[i * step_arg_0 + j * step_arg_1 + k+1]) +
	    w2 * (u2[k] + u1[k-1] + u1[k+1]) +
	    w3 * (u2[k-1] + u2[k+1]);
	}
      }
    }

    delete [] u1;
    delete [] u2;
  }

 public:
 
  template<typename OP>
  void hmap (OP& op, size_t level, bool* mask = NULL);
   
  template<typename OP>
  void hmap (OP& op, size_t level, Array<T, DIM, TRAIT>* arg1, bool* mask = NULL);
  
  template<typename OP>
  void hmap (OP& op, size_t level, Array<T, DIM, TRAIT>* arg1, Array<T, DIM, TRAIT>* arg2, bool* mask = NULL);

  template<typename OP>
  void hmap (OP& op,
	     size_t level,
	     Array<T, DIM, TRAIT>* arg1 ,
	     Array<T, DIM, TRAIT>* arg2 ,
	     Array<T, DIM, TRAIT>* arg3,
	     bool* mask = NULL);
 
  template<typename OP>
  void hmap (OP& op,
	     size_t level,
	     Array<T, DIM, TRAIT>* arg1 ,
	     Array<T, DIM, TRAIT>* arg2 ,
	     Array<T, DIM, TRAIT>* arg3 ,
	     Array<T, DIM, TRAIT>* arg4,
	     bool* mask = NULL);
 

 template<typename OP>
  void hmap (OP& op,
	     size_t level,
	     Array<T, DIM, TRAIT>* arg1 ,
	     Array<T, DIM, TRAIT>* arg2 ,
	     Array<T, DIM, TRAIT>* arg3 ,
	     Array<T, DIM, TRAIT>* arg4 ,
	     Array<T, DIM, TRAIT>* arg5,
	     bool* mask = NULL);
  
template<typename OP>
  void hmap (OP& op,
	     size_t level,
	     Array<T, DIM, TRAIT>* arg1,
	     Array<T, DIM, TRAIT>* arg2,
	     Array<T, DIM, TRAIT>* arg3,
	     Array<T, DIM, TRAIT>* arg4,
	     Array<T, DIM, TRAIT>* arg5,
	     Array<T, DIM, TRAIT>* arg6,
	     Array<T, DIM, TRAIT>* arg7,
	     bool* mask = NULL);

  template<typename OP>
  void hmap (int n,
	     const Shape<DIM>* const* s1,
	     const Shape<DIM>* const* s2,	     
	     OP& op,
	     size_t level,
	     Array<T, DIM, TRAIT>* arg1,
	     bool* mask = NULL);
  

 template<typename OP>
  void hmap (int n,
	     const Shape<DIM>* const* s1,
	     const Shape<DIM>* const* s2,
	     const Shape<DIM>* const* s3,	     
	     OP& op,
	     size_t level,
	     Array<T, DIM, TRAIT>* arg1,
	     Array<T, DIM, TRAIT>* arg2,
	     bool* mask = NULL);

 template<typename OP>
  void hmap (int n,
	     const Shape<DIM>* const* s1,
	     const Shape<DIM>* const* s2,
	     const Shape<DIM>* const* s3,	     
	     const Shape<DIM>* const* s4,	     
	     OP& op,
	     size_t level,
	     Array<T, DIM, TRAIT>* arg1,
	     Array<T, DIM, TRAIT>* arg2,
	     Array<T, DIM, TRAIT>* arg3,
	     bool* mask = NULL);
 

template<typename OP>
  void hmap (int n,
	     const Shape<DIM>* const* s1,
	     const Shape<DIM>* const* s2,
	     const Shape<DIM>* const* s3,	     
	     const Shape<DIM>* const* s4,	     
	     const Shape<DIM>* const* s5,	     
	     OP& op,
	     size_t level,
	     Array<T, DIM, TRAIT>* arg1,
	     Array<T, DIM, TRAIT>* arg2,
	     Array<T, DIM, TRAIT>* arg3,
	     Array<T, DIM, TRAIT>* arg4,
	     bool* mask = NULL);

 template<typename OP>
  void hmap (int n,
	     const Shape<DIM>* const* s1,
	     const Shape<DIM>* const* s2,
	     const Shape<DIM>* const* s3,	     
	     const Shape<DIM>* const* s4,	     
	     const Shape<DIM>* const* s5,	     
	     const Shape<DIM>* const* s6,	     
	     OP& op,
	     size_t level,
	     Array<T, DIM, TRAIT>* arg1,
	     Array<T, DIM, TRAIT>* arg2,
	     Array<T, DIM, TRAIT>* arg3,
	     Array<T, DIM, TRAIT>* arg4,
	     Array<T, DIM, TRAIT>* arg5,
	     bool* mask = NULL);


 template<typename OP>
  void hmap (int n,
	     const Shape<DIM>* const* s1,
	     const Shape<DIM>* const* s2,
	     const Shape<DIM>* const* s3,	     
	     const Shape<DIM>* const* s4,	     
	     const Shape<DIM>* const* s5,	     
	     const Shape<DIM>* const* s6,
	     const Shape<DIM>* const* s7,
	     const Shape<DIM>* const* s8,	     
	     OP& op,
	     size_t level,
	     Array<T, DIM, TRAIT>* arg1,
	     Array<T, DIM, TRAIT>* arg2,
	     Array<T, DIM, TRAIT>* arg3,
	     Array<T, DIM, TRAIT>* arg4,
	     Array<T, DIM, TRAIT>* arg5,
	     Array<T, DIM, TRAIT>* arg6,
	     Array<T, DIM, TRAIT>* arg7,
	     bool* mask = NULL);

  template<typename OP>
  void hmap3D_ (int n,
		const Shape<DIM>* const* s1,
		const Shape<DIM>* const* s2,
		OP& op,
		size_t level,
		Array<T, DIM, TRAIT>* arg1,
		bool* mask = NULL,
		int off =0);
 
  template<typename OP>
  void hmap3D_ (int n,
		const Shape<DIM>* const* s1,
		const Shape<DIM>* const* s2,
		const Shape<DIM>* const* s3,
		OP& op,
		size_t level,
		Array<T, DIM, TRAIT>* arg1,
		Array<T, DIM, TRAIT>* arg2,
		bool* mask = NULL,
		int off =0);


  template<typename OP>
  void hmap3D_ (int n,
		const Shape<DIM>* const* s1,
		const Shape<DIM>* const* s2,
		const Shape<DIM>* const* s3,
		const Shape<DIM>* const* s4,		
		OP& op,
		size_t level,
		Array<T, DIM, TRAIT>* arg1,
		Array<T, DIM, TRAIT>* arg2,
		Array<T, DIM, TRAIT>* arg3,
		bool* mask = NULL,
		int off =0);

  template<typename OP>
  void hmap3D_ (int n,
		const Shape<DIM>* const* s1,
		const Shape<DIM>* const* s2,
		const Shape<DIM>* const* s3,
		const Shape<DIM>* const* s4,		
		const Shape<DIM>* const* s5,		
		OP& op,
		size_t level,
		Array<T, DIM, TRAIT>* arg1,
		Array<T, DIM, TRAIT>* arg2,
		Array<T, DIM, TRAIT>* arg3,
		Array<T, DIM, TRAIT>* arg4,
		bool* mask = NULL,
		int off =0);
 
  template<typename OP>
  void hmap3D_ (int n,
		const Shape<DIM>* const* s1,
		const Shape<DIM>* const* s2,
		const Shape<DIM>* const* s3,
		const Shape<DIM>* const* s4,		
		const Shape<DIM>* const* s5,		
		const Shape<DIM>* const* s6,		
		OP& op,
		size_t level,
		Array<T, DIM, TRAIT>* arg1,
		Array<T, DIM, TRAIT>* arg2,
		Array<T, DIM, TRAIT>* arg3,
		Array<T, DIM, TRAIT>* arg4,
		Array<T, DIM, TRAIT>* arg5,
		bool* mask = NULL,
		int off =0);
 
  template<typename OP>
  void hmap3D_ (int n,
		const Shape<DIM>* const* s1,
		const Shape<DIM>* const* s2,
		const Shape<DIM>* const* s3,
		const Shape<DIM>* const* s4,		
		const Shape<DIM>* const* s5,		
		const Shape<DIM>* const* s6,		
		const Shape<DIM>* const* s7,		
		const Shape<DIM>* const* s8,		
		OP& op,
		size_t level,
		Array<T, DIM, TRAIT>* arg1,
		Array<T, DIM, TRAIT>* arg2,
		Array<T, DIM, TRAIT>* arg3,
		Array<T, DIM, TRAIT>* arg4,
		Array<T, DIM, TRAIT>* arg5,
		Array<T, DIM, TRAIT>* arg6,
		Array<T, DIM, TRAIT>* arg7,
		bool* mask = NULL,
		int off =0);
  
  /** reduction over all dimensions */
  template <typename OP> T reduce (const OP& binary_op) const
  {
    
    if(this->isLeaf())
    {
      return static_cast<const typename TRAIT::TypeTileImpl*>(this)->reduce_leaf_(binary_op);
    }
    else
    {
      T overall_res = TypeHelper<T>::identityVal(binary_op);
      T tile_res;
      int ntiles = this->shape().card();
      for (int i = 0; i < ntiles; i++) {
	typename TRAIT::TypeTileImpl* tile = tileAt(i);
	if (tile->Array<T, DIM, TRAIT>::isLocal()) {
	  tile_res = tile->reduce(binary_op);
	  overall_res = binary_op(overall_res, tile_res);	
	}
      }
      HTA_DEBUG(5, "Array::reduce arr =" << *this << " res=" << overall_res);
      return overall_res;
    }
  }


  template <typename OP> 
  void reduce (const OP& binary_op, 
		int dim,
		typename TRAIT::TypeImpl& dst) const
    
  {
    actual_().reduce_(binary_op, dim, dst);
  }


  template <typename OP> 
  void reduce (int oplevel,
	       const OP& binary_op, 
	       int dim,
	       typename TRAIT::TypeImpl& dst) const    
  {
    actual_().reduce_ (oplevel, binary_op, dim, dst);
  }
  
  // nullary map -----------------------------------------------------------

  template <typename OP>
  static void map (const OP& nullary_op, 
		   Array<T, DIM, TRAIT>& dst)
  {
    if(dst.isLeaf() )
    {
      TRAIT::TypeTileImpl::map_leaf_(nullary_op, dst);
    }
    else
    {
      int end = dst.shape().card();
      for (int i = 0 ; i < end; i++) {
	typename TRAIT::TypeTileImpl* dst_tile = dst.tileAt (i);
	if (dst_tile && dst_tile->Array<T, DIM, TRAIT>::isLocal()) 
	  TRAIT::TypeTileImpl::map (nullary_op, *dst_tile);
      }
    }
  }

  // unary map -----------------------------------------------------------

  //with accumulation
  template <typename OP, typename OP2>
  static void mapAccum (const OP& unary_op,
			const OP2& acc_op,
			const Array<T, DIM, TRAIT>& arg, 
			Array<T, DIM, TRAIT>& dst)
  {
    //    cout << "here 0 " << endl;
    if(arg.isLeaf() )
    {
      TRAIT::TypeTileImpl::map_leaf_accum_(unary_op, acc_op, arg, dst);
    }
    else
    {
      // this is a limitation of the current implementation
      assert(arg.shape() == dst.shape());
      int ntiles = arg.shape().card();
      for (int i = 0; i < ntiles; i++) {
	typename TRAIT::TypeTileImpl* dst_tile = dst.tileAt(i);
	if (dst_tile->Array<T, DIM, TRAIT>::isLocal()) {
	  const typename TRAIT::TypeTileImpl* tile = arg.tileAt(i);
	  TRAIT::TypeTileImpl::mapAccum (unary_op, acc_op, *tile, *dst_tile);
	}
      }
    }
  }
    
  template <typename OP>
  static void map (const OP& unary_op, 
		   const Array<T, DIM, TRAIT>& arg, 
		   Array<T, DIM, TRAIT>& dst)
  {
    if(arg.isLeaf() )
    {
      TRAIT::TypeTileImpl::map_leaf_(unary_op, arg, dst);
    }
    else
    {
      // this is a limitation of the current implementation
      assert(arg.shape() == dst.shape());
      int ntiles = arg.shape().card();
      for (int i = 0; i < ntiles; i++) {
	typename TRAIT::TypeTileImpl* dst_tile = dst.tileAt(i);
	if (dst_tile->Array<T, DIM, TRAIT>::isLocal()) {
	  const typename TRAIT::TypeTileImpl* tile = arg.tileAt(i);
	  TRAIT::TypeTileImpl::map(unary_op, *tile, *dst_tile);
	}
      }
    }
  }
  
  template <typename OP, typename TRAIT2>
  static void map (const OP& unary_op, 
		   const Array<T, DIM, TRAIT>& arg, 
		   Array<int, DIM, TRAIT2>& dst)
  {
    assert(arg.shape() == dst.shape());
    if(arg.isLeaf() )
    {
      TRAIT::TypeTileImpl::map_leaf_(unary_op, arg, dst);
    }
    else
    {
      int ntiles = arg.shape().card();
      for (int i = 0; i < ntiles; i++) {
	typename TRAIT2::TypeTileImpl* dst_tile = dst.tileAt(i);
	if (dst_tile->isLocal()) {
	  const typename TRAIT::TypeTileImpl* tile = arg.tileAt(i);
	  TRAIT::TypeTileImpl::map(unary_op, *tile, *dst_tile);
	}
      }  
    }
  }

  // binary map -----------------------------------------------------------

  template <typename OP>
  static void map (const OP& binary_op, 
		   const Array<T, DIM, TRAIT>& larg, 
		   const Array<T, DIM, TRAIT>& rarg, 
		   Array<T, DIM, TRAIT>& dst)
  {
    if(larg.isLeaf() )
    {
      TRAIT::TypeTileImpl::map_leaf_(binary_op, larg, rarg, dst);
    }
    else
    {
      //non leaf
      // this is a limitation of the current implementation
      
      if(larg.shape() != dst.shape()){
	      TRAIT::TypeTileImpl::map_non_homogeneous_l(binary_op, larg, rarg, dst);
      } else if(rarg.shape() != dst.shape()){
	      TRAIT::TypeTileImpl::map_non_homogeneous_r(binary_op, larg, rarg, dst);
      } else {
	assert(larg.shape() == dst.shape());
	assert(rarg.shape() == dst.shape());
	
	for (int i = 0 ; i < larg.numTiles_; i++) {
          int j = larg.localTiles_[i];
	  typename TRAIT::TypeTileImpl* dst_tile = dst.tileAt(j);
	  if (dst_tile && dst_tile->Array<T, DIM, TRAIT>::isLocal()) {
	    const typename TRAIT::TypeTileImpl* larg_sub = larg.tileAt(j);
	    const typename TRAIT::TypeTileImpl* rarg_sub = rarg.tileAt(j);
	    TRAIT::TypeTileImpl::map(binary_op, *larg_sub, *rarg_sub, *dst_tile);
	  }
	} 
      }
    }
  }

  template <typename OP, typename T2, typename TRAIT2>
  static void map (const OP& binary_op, 
		   const Array<T, DIM, TRAIT>&   larg, 
		   const Array<T2, DIM, TRAIT2>& rarg, 
		   Array<T, DIM, TRAIT>& dst)
  {
    if(larg.isLeaf() )
    {
      TRAIT::TypeTileImpl::map_leaf_(binary_op, larg, rarg, dst);
    }
    else
    {
      //non leaf
      // this is a limitation of the current implementation
      assert(larg.shape() == dst.shape());
      assert(rarg.shape() == dst.shape());

      // TODO: this might be done in parallel 
      int ntiles = larg.numTiles_;
      for (int i=0; i < ntiles; i++) {
        int j = larg.localTiles_[i];
	typename TRAIT::TypeTileImpl* dst_tile = dst.tileAt(j);
	if (dst_tile->Array<T, DIM, TRAIT>::isLocal()) {
	  const typename TRAIT::TypeTileImpl* larg_sub = larg.tileAt(j);
	  const typename TRAIT2::TypeTileImpl* rarg_sub = rarg.tileAt(j);
	  TRAIT::TypeTileImpl::map(binary_op, *larg_sub, *rarg_sub, *dst_tile);
	}
      }
    }
  }
  
  template <typename OP>
  static void map_non_homogeneous_l (const OP& binary_op, 
		       const Array<T, DIM, TRAIT>& larg, 
		       const Array<T, DIM, TRAIT>& rarg, 
		       Array<T, DIM, TRAIT>& dst)

  {
    assert (larg.level() < rarg.level());

    int ntiles = rarg.shape().card();
    for (int i =0; i < ntiles; i++){
      if (dst.Array<T, DIM, TRAIT>::isLocal()) {
	TRAIT::TypeTileImpl::map(binary_op, larg, *rarg.tileAt(i), dst);
      }
    }
  }
  
  template <typename OP>
  static void map_non_homogeneous_r (const OP& binary_op, 
		       const Array<T, DIM, TRAIT>& larg, 
		       const Array<T, DIM, TRAIT>& rarg,
		       Array<T, DIM, TRAIT>& dst) 
  {
    if(dst.isLeaf())
    {
      TRAIT::TypeTileImpl::map_leaf_(binary_op, larg, rarg, dst);
    }
    else
    {
      assert (larg.level() > rarg.level());
      int ntiles = larg.shape().card();
      for (int i =0; i != ntiles; i++) {
	typename TRAIT::TypeTileImpl* dst_tile = dst.tileAt(i);
	if (dst_tile->Array<T, DIM, TRAIT>::isLocal()) {
	  TRAIT::TypeTileImpl::map(binary_op, *larg.tileAt(i), rarg, *dst_tile);	
	}
      }
    }
  }

  // tileAt -----------------------------------------------------------

  inline typename TRAIT::TypeTileImpl* tileAt (const Tuple<DIM>& t) const
  {
    return actual_().tileAt(t);
  }
  
  inline typename TRAIT::TypeTileImpl* tileAt (int tile_idx) const
  {
    return actual_().tileAt(tile_idx);
  }

  //gti add inline
  const Shape<DIM>& shape () const // __attribute__((always_inline))
  { 
    return shape_; 
  }
  
  inline const AbstractArray<T, DIM>& parent () const 
  { 
    assert (parent__ != NULL);
    return *parent__;
  }

  inline bool isLocal () const 
  { 
    return home_ == Traits::Default::myPlace();
  }
  
  inline bool isDistributed () const 
  { 
    return home_ == Traits::Default::nullPlace();
  }

  AbstractArray<T, DIM>& accessTile (const Tuple<DIM>& t) const 
  {
    assert (this->level() > 0);
    return *tileAt(t);
  }

  AbstractArray<T, DIM>& accessTile (int idx) const
  {
    assert (this->level() > 0);
    return *tileAt(idx);
  }

  T& accessScalar (const Tuple<DIM>& t) const
  {
    return scalarAt(t);
  }

  /**
   * @return  a pointer to the underlying memory. If array is local, then the pointer 
   *          will refer to the first variable in the array. Otherwise NULL is returned.
   *          method must not be called on an HTA that is distributed.
   */
  inline typename TRAIT::TypeInternal raw() const
  {
    assert (!isDistributed());

    typename TRAIT::TypeInternal ret;
    if (this->Array<T, DIM, TRAIT>::isLocal()) {
      ret = this->alloc_.get(this->memMap().offset());
      this->alloc_.recycle();
      HTA_DEBUG(3, "Array::raw - offset=" << this->memMap().offset() << "ret=" << ret);
      assert (ret != NULL);
    } else 
      ret = NULL;     
    HTA_DEBUG(1, "Array::raw - ret=" << ret);
    return ret;
  }
  
  inline int home() const
  {
    return home_;
  }
  
  /** 
   * this method is just used internally for assertion checking 
   *
   * @return   true if the variables in this tile are local or cached locally 
   *           such that the operations can be computed without communication.
   */
  virtual bool isAvailable_() const = 0;
  
  inline const MemMapping<DIM>& memMap () const // __attribute__((always_inline))
  {
    assert (memMapping__);
    // assert (home() == Traits::Default::myPlace());
    // CVP_TODO, it would be nice if this assertion would eventually hold
    // after all one processor should not have to know about the mapping 
    // details of other processors. The only place where a mapping of a remote processor is 
    // used is in MPI communicate exchange - we need to determine the size of the 
    // receive buffer and the linearStep at the receiveBuffer

    //gti; see how we can reenable thsi assert; assert (memMapping__->level() == (int)this->level());
    return *memMapping__;
  }  

  inline const Tuple<DIM>& leafSize() const 
  {
    assert (memMapping__);
    return this->memMapping__->leafSize();
  }

  /**
   * @return the distribution of the data if it is known.
   */
  inline const Distribution<DIM>& dist () const // __attribute__((always_inline))
  {
    if (this->level() == 0)
      return ConstDistribution<DIM>::get(home_);
    else {
      assert (dist__ != NULL);
      return *dist__;
    }
  }
  
  //  ------- allocation

  /** 
   * top-down creation and allocation of HTAs 
   */
  static typename TRAIT::TypeImpl* alloc (const size_t levels,
					  const typename Tuple<DIM>::Seq& tiling, 
					  const Order order,
					  const Distribution<DIM>* dist = NULL)
  {
    // distribution
    if (dist == NULL)
      dist = TRAIT::TypeHelper::getDistribution_();
    assert (dist != NULL);
    
    // mem-mapping
    MemMapping<DIM>* mmap = TRAIT::TypeHelper::getMemMapping_(order, tiling, levels, levels, dist); 

    // allocator
    int total_size = mmap->linearCard();
    int block_size = (levels == 0 || Traits::Default::nPlaces() == 1) ? total_size : (*mmap)[0].linearCard();
    typename TRAIT::TypeAllocator alloc = TRAIT::TypeHelper::getAllocator_(total_size, block_size, NULL);

    //cout << "GBDEBUG SIZE " << block_size << endl;

    // allocate array
    //BBF: If levels == 0 home must be a specific place
    int home = levels ? Traits::Default::rootPlace() : (*dist)[0];
    typename TRAIT::TypeImpl* ret = recursiveCreate_<TRAIT>(levels, tiling, alloc, *mmap, *dist, home);

    if (ret->isDistributed())
     {
       ret->initDistribution(dist);
       ret->initLocalTiles();
     }
    return ret;
  }

  /**
   * @param tiling  tiling info. tiling[i] is tiling at level i.
   * @param data    pointer to array pif scalars that is used as a backing storage 
   *                for the HTA. The created HTA does not take responsibility to 
   *                de-allocate this memory.
   * @param order   memory layout
   * @return pointer to HTA
   */
  static typename TRAIT::TypeImpl* allocLocal (size_t levels, 
					       const typename Tuple<DIM>::Seq& tiling, 
					       T* data, 
					       const Order order) 
  {
    const MemMapping<DIM>* mmap = TRAIT::TypeHelper::getMemMapping_(order, tiling, levels, 0);
    int size = mmap->linearCard();
    typename TRAIT::TypeAllocator alloc = TRAIT::TypeHelper::getAllocator_(size, size, data); 
    const Distribution<DIM>& dist = ConstDistribution<DIM>::get(Traits::Default::myPlace());
    typename TRAIT::TypeImpl* ret = recursiveCreate_<TRAIT>(levels, tiling, alloc, *mmap, dist, Traits::Default::myPlace());
    return ret;
  }
  
  /*
   * similar to allocLocal above, but allocates on a given processor
   */
  static typename TRAIT::TypeImpl* allocLocal (size_t levels, 
					       const typename Tuple<DIM>::Seq& tiling, 
					       T* data, 
					       const Order order,
					       int processor) 
  {
    const MemMapping<DIM>* mmap = TRAIT::TypeHelper::getMemMapping_(order, tiling, levels, 0);
    int size = mmap->linearCard();
    typename TRAIT::TypeAllocator alloc = TRAIT::TypeHelper::getAllocator_(size, size, data); 
    const Distribution<DIM>& dist = ConstDistribution<DIM>::get(processor);
    typename TRAIT::TypeImpl* ret = recursiveCreate_<TRAIT>(levels, tiling, alloc, *mmap, dist, processor);
    return ret;
  }
  
  /**
   * similar to allocLocal, but allocates on processor -1
   * So no-one ones this tile
   * 'alloc' is empty.
   */
  //  static typename TRAIT::TypeImpl* allocDummy (size_t levels, 
  //				       const typename Tuple<DIM>::Seq& tiling, 
  //		       const Order order) 
  //{
  // typename TRAIT::TypeAllocator alloc;
  // typename TRAIT::TypeImpl* ret = recursiveCreate_<TRAIT>(levels, tiling, alloc);
  // return ret;
  //}
  
  static typename TRAIT::TypeImpl* allocDummy (size_t levels,
                                               const typename Tuple<DIM>::Seq& tiling,
                                               const Order order)
  {
    const MemMapping<DIM>* mmap = TRAIT::TypeHelper::getMemMapping_(order, tiling, levels, 0);
    int size = mmap->linearCard();
    typename TRAIT::TypeAllocator alloc;
    typename TRAIT::TypeImpl* ret = recursiveCreate_<TRAIT>(levels, tiling, alloc, *mmap);
    return ret;
  }

  static typename TRAIT::TypeImpl* allocShell (const size_t levels, 
					       const Tuple<DIM> tiling, 
					       const Distribution<DIM>* dist = NULL)
  {
    // allocator
    typename TRAIT::TypeAllocator alloc;
    
    // distribution
    if (dist == NULL)
      dist = TRAIT::TypeHelper::getDistribution_();
    
    // mem-mampping will be computed later after all tiles are assigned 
    // (see initTile, initMemMapping)

    Shape<DIM> s(tiling);
    int len = s.card();
    // the memory referenced by htav will be managed and released the newly created object, 
    // the tiles are not yet initialized ...
    typename TRAIT::TypeTileImpl** htav = new typename TRAIT::TypeTileImpl* [len];
    memset(htav, 0, len * sizeof(unsigned));
    typename TRAIT::TypeImpl* ret = new typename TRAIT::TypeImpl(levels, htav, s, alloc, NULL, Traits::Default::myPlace());

    if (ret->isDistributed())
     {
        ret->initDistribution(dist);
        ret->initLocalTiles();
     }
    return ret;
  }

  //-------------shell allocation

  /** 
   *  This method is used only for partial HTA creation.
   *  For full creation use alloc.
   */
  static typename TRAIT::TypeImpl* allocShell (size_t levels, 
					       const typename Tuple<DIM>::Seq tiling, 
					       const Distribution<DIM>* dist = NULL)
  {
    assert (tiling.length() <= levels);
    
    // distribution
    if (dist == NULL)
      dist = TRAIT::TypeHelper::getDistribution_();
    typename TRAIT::TypeImpl* ret = recursiveCreateShell_<TRAIT>(levels, tiling, *dist, Traits::Default::myPlace());
    return ret;
  }
  
  
  /** 
   * This method is called on the top-level HTA after a bottom up creation (allocShell) 
   * or non-homogenous tileAt operation
   */
  void initMemMapping () 
  {
    //should be called only once!
    assert (this->memMapping__ == NULL);

    if (this->isHomogeneous_()) {
      typename Tuple<DIM>::Seq tiling;
      const AbstractArray<T, DIM>* tmp = this;
      for (int i = this->level(); i >= 0; --i) {
	tiling[i] = tmp->shape().size();
	if (i > 0) 
	  tmp = &(tmp->accessTile (tmp->shape().low())); 
      }

      this->memMapping__ = TRAIT::TypeHelper::getMemMapping_(TILE, tiling, this->level(), 0);
      this->memMapping__->refCount_(1);
    } else {      
      initMemMapping (0, Tuple<DIM>::zero);            
    }
  }
  
  int ref () const
  {
    return refCount_();
  }

  //for serialHTAs this is a no-op
  void updateShape()
  {
    
  }
  
  const MemMapping<DIM>* initMemMapping (int offset, Tuple<DIM> position) 
  {      
    //assert (this->memMapping__ == NULL);

    int my_offset = offset;
    // calculate size of one tile at leaf level
    Shape<DIM> s = this->shape();
    int tile_size = s.card();
    
    int j = 0;
    
    const MemMapping<DIM>** children = new const MemMapping<DIM>* [tile_size];

    //initialize children - they follow the same order as the parent
    for (typename Shape<DIM>::iterator i = s.begin(); j < tile_size; i++, j++) {
      Tuple<DIM> pos_cur_level = (*i);
      
      typename TRAIT::TypeTileImpl* tile = this->tileAt(j);
      
      if (level_ > 1)
	children[j] = (MemMapping<DIM>*) tile->initMemMapping (offset, pos_cur_level);
      else
	{
	  /* Leaf should always have memMapping__ != NULL */
	  assert (tile->memMapping__);
	  children[j] = tile->memMapping__;
	}                  
      
      children[j] ->refCount_(1);            
      
      offset += children[j]->linearCard();
    }

    // compute leaf_size 
    Tuple<DIM> size = s.size();
    Tuple<DIM> curLeafSize = Tuple<DIM>::zero;
    for (int i = 0; i < DIM; i++)
      {
	for (int j = 0; j < size[i]; j++) {	  
	  Tuple<DIM> t =  Tuple<DIM>::zero;
	  t[i] = j;
	  curLeafSize[i] += this->tileAt(t)->memMapping__->leafSize()[i];	  
	}
      }
 
    this->memMapping__ = new IrregularMemMapping<DIM> (children, tile_size, level_, my_offset, level_, 
						       curLeafSize, position * curLeafSize, NULL);
    this->memMapping__->refCount_(1);

    delete [] children;
    return memMapping__;
  }
 
   
  inline void initDistribution (const Distribution<DIM>* dist) 
  {
    assert (this->dist__ == NULL);
    if (dist == NULL) {
      // distribution not know in close form, create a custom distribution
      dist = new CustomDistribution<T, DIM>(*this);
    }
    dist->refCount_(1);
    this->dist__ = dist;

   // initLocalTiles();
  }

  //TODO :  I don't think this is used anymore
  inline void initDistribution (const Shape<DIM> shape) 
  {

    if (this->dist__ != NULL)
      delete this->dist__;

    // distribution not know in close form, create a custom distribution
    
    
    Distribution<DIM>* dist = new GenericDistribution<DIM>(shape);        
    dist->refCount_(1);
    this->dist__ = dist;
    initLocalTiles();
  }

  //TODO: This is redundant
  inline void initDistribution (const Distribution<DIM>& dist) 
  {

    if (this->dist__ != NULL)
      delete this->dist__;

    // distribution not know in close form, create a custom distribution
            
    this->dist__ = dist.clone();
    this->dist__->refCount_(1);
    initLocalTiles();  
  }

  inline void initLocalTiles ()
   {

      int place = Traits::Default::myPlace();
      for (int i = 0; i < this->shape_.card(); i++)
           if ((*dist__)[i] == place)
               numTiles_++;

      localTiles_ = new int [numTiles_];
      for (int i = 0, j =0; i < this->shape_.card(); i++)
           if ((*dist__)[i] == place)
               localTiles_[j++] = i;
     //cout << numTiles_ << endl;
   }

  //-------------assignment

  /**
   * Assignment operator 
   * pointwise copy of values of all tiles.
   */
  Array<T, DIM, TRAIT>& operator= (const Array<T, DIM, TRAIT>& rhs)
  {
    assert (Shape<DIM>::conformable(shape(), rhs.shape()));
    //assert (this->level() >= 1);
    if (this->level() > 0) { //BBF I think this is actually if (!this->isLeaf())
      if (this != &rhs) {
        int end = this->shape().card();
        for (int i = 0; i < end; ++i) {
          *this->tileAt(i) = *rhs.tileAt(i);
        }
      }
    } else {
      // resort to specific implementation for this type
      actual_() = rhs.actual_();
    }
    return *this;
  }

  /*
  Array<T, DIM, TRAIT>& operator= (const Array<T, DIM, TRAIT>& rhs)
  {
    // resort to specific implementation for this type
    actual_() = rhs.actual_();
    return *this;
    }
 */

  /**
   * Assignment operator (does not accommodate case LEVEL == LEVEL2)
   */
  template <typename TRAIT2>
  typename TRAIT::TypeImpl& operator= (const Array<T, DIM, TRAIT2>& rhs) 
  {
    assert (this->level() != rhs.level());
    assert (0); // not implemented
    return actual_();
  }

  /**
   * clones the structure f the HTA, but does not copy the values. 
   * the result HTA is initialized with 0 
   */
  inline typename TRAIT::TypeImpl* clone (bool copy_values = false) const 
  {
    typename TRAIT::TypeImpl* ret;
    
    // mapping to memory must be known
    assert (memMapping__ != NULL);
    
    //GBDEGUG: works only for one-to-one mapping
    int rank = Traits::Default::myPlace();
    int total_size = memMapping__->linearCard();
    // cout << "FFFF " << total_size << " " << (*memMapping__)[rank].linearCard() << endl;
    int block_size = (this->level() == 0 || Traits::Default::nPlaces() == 1) ? total_size : (*memMapping__)[rank].linearCard();
     typename TRAIT::TypeAllocator alloc = TRAIT::TypeHelper::getAllocator_(total_size, block_size, NULL);

    /* clone returns an identical memMapping__ with new linearStep */
    const MemMapping<DIM>* memMap = memMapping__->clone();
    //memMap->refCount_(1);

    // then recursively re-create the HTA-tree using this allocator
    // and the memory mapping
    ret = actual_().clone_ (NULL, alloc, *memMap, copy_values);

    return ret;
  }

  /**
   * returns a new HTA with permuted structure and values
   */
  /*
   * TODO: elimate clone/copy for 1D and 1 X N or N X 1 HTAs (e.g CG)
   *      Just permute the shape
   */
  inline typename TRAIT::TypeImpl* permute (const Tuple<DIM>& perm, 
					    typename TRAIT::TypeImpl* dst = NULL,
					    unsigned permute_mask = 0xFF) const  
  {
    assert (dst == NULL); // limitation of the current implementation
    
    // clone the allocator (only for the root HTA)
    // --> permute should not be recursive! (clone_ is recursive)
    //use clone_ internally in the library. permute is only for the public.

    typename TRAIT::TypeImpl* ret;
    typename TRAIT::TypeAllocator alloc = alloc_.clone();


    typename Tuple<DIM>::Seq tiling_permuted = 
      Tuple<DIM>::permute(tiling(), perm, permute_mask);

    // permute the memory mapping
    const MemMapping<DIM>* mmap = this->memMap().create(tiling_permuted, this->dist());
    // then recursively re-create the HTA-tree using this allocator
    ret = actual_().clone_(&perm, alloc, *mmap, true, permute_mask);
    return ret;
  }
  
  /**
   * Permute 2 successive levels 
   * eg; h(i, j, k)(m, n, p) = h(p, j, m)(k, n, i), for perm = [0, 2]

   * @arg1: the permute dimensions (i, j) (Tuple<2>)
   * output: a new HTA after permute

   * Limitations: works only for 2 level HTA 
   * works only for square decomposition. 

   * Limitations: Only permutes two dimensions at a  time
   * check if this will work for permuting  more than one dimensions at a time
   */
  
  inline typename TRAIT::TypeImpl* dpermute (const Tuple<DIM> perm,
					     typename TRAIT::TypeImpl* result) 
  {

    //(1) create a clone of *this*
    typename Tuple<DIM>::Seq tiling = this->tiling();
    const MemMapping<DIM>* mmap = this->memMap().create(tiling, this->dist());
    typename TRAIT::TypeImpl* ret = actual_().clone_(NULL, alloc_.clone(), *mmap, false);

    //(2) perform the exchange
    typename Shape<DIM>::iterator i = this->shape().begin();
    typename Shape<DIM>::iterator end = this->shape().end();

    for (; i != end; ++i) {

      typename Shape<DIM>::iterator j = this->tileAt(0)->shape().begin();
      typename Shape<DIM>::iterator end_j = this->tileAt(0)->shape().end();
      
      for (; j != end_j; ++j){
		
	Tuple<DIM> ii = (*i);
	Tuple<DIM> jj = (*j);
	
	// ((i, j, k), (m, n, p), [0, 2]) =  ((p, j, m), (k, n, i))
	crossSwap(ii, jj, perm);
	
	*ret->tileAt(*i)->tileAt(*j) = *this->tileAt(ii)->tileAt(jj);
      }    
    }

    //(4) copy back to this + local transpose of the leaves
    Tuple<DIM> permVector;
    for (int i = 0; i < DIM; ++i)
      permVector[i] = i;    
    permVector = permVector.swap(perm[0], perm[1]); 
    
    //(4) in place transposition
    ret->copy_ (&permVector, result, 1);
    
    delete ret;
    
    return result;
  }

  // ------- stubs that forward calls to the method implemented for the actual type

  inline T& scalarAt (const Tuple<DIM>& t) const
  {
    return actual_().scalarAt(t);
  }


  inline T& scalarAt (const int& i) const
  {
    return actual_().scalarAt_ (i);
  }

  inline typename TRAIT::TypeImpl* scalarAt (const Shape<DIM>& s) const 
  {
    assert (this->level() == 0);
    return actual_().scalarAt(s);
  }

  const typename Tuple<DIM>::Seq& tiling() const
  {
    return this->memMap().tiling();
  }
     
  /**
   * internal method (not part of the public interface)
   */
  inline T& scalarAt_ (int idx) const  
  {
    assert (this->level() == 0);
    return actual_().scalarAt_(idx);
  }

  /**
   * internal method (not part of the public interface)
   */
  inline T& scalarAt_ (const typename Shape<DIM>::iterator& it) const 
  {
    assert (this->level() == 0);
    return actual_().scalarAt_(it);
  }

  /** 
   * htlib internal method, not opart of the public api
   *
   * @return  a references to the actual type of object - 
   *          this is used for the Barton/Nackman trick 
   */
  inline typename TRAIT::TypeImpl& actual_() // __attribute__((always_inline))
  {
    return static_cast<typename TRAIT::TypeImpl&>(*this);
  }

  inline const typename TRAIT::TypeImpl& actual_() const // __attribute__((always_inline))
  {
    return static_cast<const typename TRAIT::TypeImpl&>(*this);
  }


  //----------SCAN-----------------------------------------------------

  /**
   * First implementation of operator framework
   */

  /**
   * prefix sum computation on an HTA
   * TODO: works only for 1D or 1 X N 2D HTAs
   */
  void scan (int oplevel, Array<T, DIM, TRAIT>& output) const
  {
    assert (DIM == 1);
    assert (output.shape() == this->shape()); // current limitation

    if (this->isLeaf()) 
      {
	scan_leaf_ (output);
      }
    else {
      for (int i = 0; i < this->shape().card(); i++)
	this->tileAt(i)->scan(oplevel, *(output.tileAt(i)));
      
      if (oplevel >= this->level()) 
	{
	  for (int i = 1; i < output.shape().card(); i++)
	    {
	      Tuple <DIM> end = output.tileAt(i-1)->memMap().leafSize() - Tuple<DIM>::one;
	      int& last = output.tileAt(i-1)->scalarAt(end);
	      Array<T, DIM, TRAIT>* tmp = output.tileAt(i); 
	      map(binder2nd<plus<T> >(plus<T>(), last), *tmp, *tmp);
	    }
	}
    }
  }
  
  void scan_leaf_ (Array<T, DIM, TRAIT>& output) const
  {
    output.scalarAt_(0) = this->scalarAt_(0);
    for (int i = 1; i < this->shape().card(); i++)
      {
	output.scalarAt_(i) = this->scalarAt_(i) + output.scalarAt_(i-1);
      }
  }
  
  /**
   * prefix sum computation on an HTA
   * TODO: works only for 1D or 1 X N 2D HTAs
   */
  void hta_scan (int oplevel, Array<T, DIM, TRAIT>& output) const
  {
    assert (output.shape() == this->shape()); // current limitation
    assert (oplevel <= this->level());
    assert (this->level() == 0);
    if (oplevel != this->level()) {
      for (int i = 0; i < this->shape().card(); i++)
	this->tileAt(i)->scan(oplevel, output->tileAt(i));
    } else {

      output.tileAt(0) = this->tileAt(0);

      for (int i = 1; i < this->shape().card(); i++) {
	output.tileAt(i) = this->tileAt(i) + output.tileAt(i-1);
      }
      
    }       
    
  }

  //============TILE-BY-TILE operations ========================== should be done by map() future ===============

  /**
   * apply Fourier Transform on 
   * on the entire *raw* array
   * stored in each of the tiles.
   * @arg1 = dimension of transform (0..DIM-1).
   * @arg2 = direction (forward = 1, backward = -1).
   * @arg3 = dest. array (NULL by default) => *this* is overwritten
   */
  virtual void ft (int dim, int dir, typename TRAIT::TypeImpl* dst = NULL) const 
  {
    assert (this->memMap().order() == ROW || this->memMap().order() == DIST);           

    assert (dim >=0 && dim < DIM);
  
    assert (dst == NULL); // current limitation

    // fft must be done along a dimension that is not partitioned into tiles at this level!
    assert (this->shape().size()[dim] == 1);

    // stride between two consecutive elements along 'dim' 
    int stride[2] = {0, this->tileAt(0)->memMap().linearStep()[dim]}; 
    
    // compute the transform for each tile -- write the output over the input
    int ntiles = this->shape().card();
    
    Tuple<DIM> tile_size = this->memMap().leafSize() / this->shape().size();
    
    int size = tile_size[dim];  // size of the transform 
    
    tile_size[dim] = 1;

    Shape<DIM> s(tile_size);     //shape with the dim_th dimension = 1
    
    typename TRAIT::TypeMathKernel mkl(size, stride);
    
    for (int i = 0 ; i < ntiles; ++i) {
      
      typename TRAIT::TypeTileImpl* tile = this->tileAt (i);
      
      if (!tile->Array<T, DIM, TRAIT>::isLocal()) continue; //if not local, skip;
      
      for (typename Shape<DIM>::iterator j = s.begin(); j != s.end(); ++j) {
	
	if (dir == 1) { //forward FOURIER transfrom (time --> freqency domain)

	  mkl.dftForward(&tile->scalarAt(*j));
	  
	} else // inverse FOURIER transform (frequency --> time domain)
	  
	  mkl.dftBackward(&tile->scalarAt(*j));
      }					    
    }
  }

  void accumArray (typename TRAIT::TypeImpl* indices,
		   int rshift = 0,
		   int lshift = 0,
		   size_t level = 0)
  {    
    if (this->level() > level) {
      typename Shape<DIM>::iterator i = this->shape().begin();
      typename Shape<DIM>::iterator end = this->shape().end();
      for ( ; i != end; i++) {
	Tuple<DIM> t = *i;
	typename TRAIT::TypeTileImpl* tile = this->tileAt(t);
	if (tile && tile->Array<T, DIM, TRAIT>::isLocal()) tile->accumArray(indices->tileAt(t), rshift, lshift, level);
      }
    } else {
      if (rshift != 0) {
	int size = indices->shape().card();
	for (int i = 0; i < size; i++) {
	  int idx = indices->scalarAt_(i) >> rshift;
	  this->scalarAt_(idx)++;
	} 
      } else { /* for both rshift = 0, and default path*/	
	//GBDEBUG: since we don't have flatten for non-homognenous non-leaf HTAs....
	//I do this shorcut
	if (indices->level() == 0) {
	  int size = indices->shape().card();
	  for (int i = 0; i < size; i++) {
	    int idx = indices->scalarAt_(i) << lshift;
	    this->scalarAt_(idx)++;
	  } 
	} else if (indices->level() == 1) { /* in IS, indices->level() == this->level() + 1 */
	  typename Shape<DIM>::iterator i = indices->shape().begin();
	  typename Shape<DIM>::iterator end = indices->shape().end();
	  for ( ; i != end; ++i) {
	    Tuple<DIM> t = *i;
	    typename TRAIT::TypeTileImpl* tile = indices->tileAt(t);
	    int size = tile->shape().card();
	    for (int j =0; j < size; ++j) {
	      int idx = tile->scalarAt_(j) << lshift;
	      this->scalarAt_(idx)++;
	    }
	  }
	}
      }	
    }
  }


  /**
   * local sort for each tile
   * The reason for name HTAsort is to avoid
   * confusion with *sort* of stl.
   */  
  void HTAsort (size_t level = 0) 
  {
    assert (DIM == 1); // current limitation
    
    if (this->level() > level) {
      typename Shape<DIM>::iterator i = this->shape().begin();
      typename Shape<DIM>::iterator end = this->shape().end();
      for ( ; i != end; i++) {
	Tuple<DIM> t = *i;
	typename TRAIT::TypeTileImpl* tile = this->tileAt(t);
	if (tile->Array<T, DIM, TRAIT>::isLocal()) tile->HTAsort (level);
      }
    } else {
      int N = this->memMap().leafSize()[0];
      
      //cout << N << endl;
	
	T* a = new T[N];
		
	/* copy the values of the tiles to a */
	{
	  int k = 0;
	  typename Shape<DIM>::iterator i = this->shape().begin();
	  typename Shape<DIM>::iterator end = this->shape().end();
	  for ( ; i != end; ++i) {
	    Tuple<DIM> t = *i;
	    typename TRAIT::TypeTileImpl* tile = this->tileAt(t);
	    typename Shape<DIM>::iterator j = tile->shape().begin();
	    typename Shape<DIM>::iterator jEnd = tile->shape().end();
	    for (; j != jEnd; ++j) {
	      assert (k < N);
	      a[k++] = tile->scalarAt(*j);
	    }
	  }
	}

	/* sort using stl sort routine */
	sort (a, a + N);
	
	/* copy back the sorted values to the tiles */
	{
	  int k = 0;
	  typename Shape<DIM>::iterator  i = this->shape().begin();
	  typename Shape<DIM>::iterator end = this->shape().end();
	  for ( ; i != end; ++i) {
	    Tuple<DIM> t = *i;
	    typename TRAIT::TypeTileImpl* tile = this->tileAt(t);
	    typename Shape<DIM>::iterator j = tile->shape().begin();
	    typename Shape<DIM>::iterator jEnd = tile->shape().end();
	    for (; j != jEnd; ++j) {
	      tile->scalarAt(*j) = a[k++];
	    }
	  }
	}

	delete [] a;
      }
    }  
  
private:
  
  template<class T2, int DIM2, typename TRAIT2>
  friend class HTA;
  template<class T2, int DIM2, typename TRAIT2>
  friend class HTAImpl;
  
  // all other template instance of this class should be friends (required when accessing print_)
  template<class T2, int DIM2, typename TRAIT2>
  friend class Array;

  template<typename T2, int DIM2, typename TRAIT2>
  friend class Router;

  inline Order order_() const
  {
    return memMap().order();
  }

protected:
    
  // all tiles must be regular HTAs
  bool isHomogeneous_() const
  {
   bool ret = true;
   if (this->level() > 0) {
     int card = this->shape().card();
     assert (card > 0);

     Tuple<DIM> tile_size =  this->tileAt(0)->shape().size();
     for (int i = 1; ret && i < card; ++i) {
       if (this->tileAt(i)->shape().size() != tile_size) 
	 ret = false;
     }
     if (ret) {
       for (int i = 0; ret && i < card; ++i) {
	 if (!this->tileAt(i)->isHomogeneous_())
	   ret = false;
       }
     }
   }
   return ret;
  }
  
public:

  // ------- communication and invalidation

  inline void communicate_ (const Distribution<DIM>& target_dist, 
			    Array<T, DIM, TRAIT>* target_arr = NULL) const
  {
    // to be overridden by subclasses
    assert (false);
  }
  
  template <typename T2, typename TRAIT2>
  inline void communicate_ (Array<T2, DIM, TRAIT2>& target) const
  {
    // to be overridden by subclasses
    assert (false);
  } 
  
  inline void invalidate_ () const
  {
    // to be overridden by subclasses
    assert (false);
  }

private:
  
  /**
   * Factory method for HTA implementation - top down creation.
   * 
   * @param tiling       array of tiles (index 0: leaf tiling)
   * @param pos          position in the coordinates of current level
   * @param linear_step  linear step at level 0
   * @param alloc        allocator that returns references to array variables
   * @param dist         distribution of the data
   * @return             pointer to implementation at the current level
   */ 
  template<typename TRAIT1>
  static typename TRAIT1::TypeImpl* recursiveCreate_ (size_t level,
						      const typename Tuple<DIM>::Seq& tiling,  
						      const typename TRAIT1::TypeAllocator& alloc, 
						      const MemMapping<TRAIT1::dim>& mmap,
						      const Distribution<DIM>& dist, 
						      int home)
  {
    typename TRAIT1::TypeImpl* ret = NULL;
    Shape<TRAIT1::dim> s(tiling[level]);
    
    HTA_DEBUG (1, "Array" << "<LEVEL=" << level << ">::recursiveCreate_ mmap=" << mmap);

    if (level > 0) {
      int len = s.card();
      
      //cout << "GBDEBUG home " << endl;
      // the memory referenced by htav will be managed and released the newly created object
      typename TRAIT1::TypeTileImpl** htav = new typename TRAIT1::TypeTileImpl* [len];
      
      for (int j = 0; j < len; ++j) {
	int home_tile = dist[j];
	const Distribution<DIM>& dist_tile = ConstDistribution<DIM>::get(home_tile);
	//cout << "GBDEBUG home " << home_tile << " " << len << endl;
	//gti
	htav[j] = recursiveCreate_<typename TRAIT1::TraitSubref>(level-1, tiling, alloc, mmap[j], dist_tile, home_tile); 
      }
      ret = new typename TRAIT1::TypeImpl(level, htav, s, alloc, &mmap, home);      
    } else {  // leaf case
      switch (mmap.order()) {
      case TILE:
	ret = new typename TRAIT1::TypeImpl(level, NULL, s, alloc, &mmap, home);
	break;
      case ROW: 
	ret = new typename TRAIT1::TypeImpl(level, NULL, s, alloc, &mmap, home);
	break;
      default:
	assert (false);
	break;
      }
    }
    return ret;
  }
  


 template<typename TRAIT1>
  static typename TRAIT1::TypeImpl* recursiveCreate_ (size_t level,
						      const typename Tuple<DIM>::Seq& tiling,  
						      const typename TRAIT1::TypeAllocator& alloc, 
						      const MemMapping<TRAIT1::dim>& mmap)
  {
    typename TRAIT1::TypeImpl* ret;
    Shape<TRAIT1::dim> s(tiling[level]);
    
    HTA_DEBUG (1, "Array" << "<LEVEL=" << level << ">::recursiveCreate_ mmap=" << mmap);
    
    if (level > 0) {
      int len = s.card();
      
      // the memory referenced by htav will be managed and released the newly created object
      typename TRAIT1::TypeTileImpl** htav = new typename TRAIT1::TypeTileImpl* [len];
      
      for (int j = 0; j < len; ++ j) {
	htav[j] = recursiveCreate_<typename TRAIT1::TraitSubref>(level-1, tiling, alloc, mmap[j]); 
      }
      ret = new typename TRAIT1::TypeImpl(level, htav, s, alloc, &mmap, -1);
      
    } else {  // leaf case
      switch (mmap.order()) {
      case TILE:
	ret = new typename TRAIT1::TypeImpl(level, NULL, s, alloc, &mmap, -1);
	break;
      case ROW: 
	ret = new typename TRAIT1::TypeImpl(level, NULL, s, alloc, &mmap, -1);
	break;
      default:
	assert (false);
	break;
      }
    }
    return ret;
  }
  /*
 template<typename TRAIT1>
  static typename TRAIT1::TypeImpl* recursiveCreate_ (size_t level,
						      const typename Tuple<DIM>::Seq& tiling,  
						      const typename TRAIT1::TypeAllocator& alloc)						      
  {
    typename TRAIT1::TypeImpl* ret;
    Shape<TRAIT1::dim> s(tiling[level]);
    
    HTA_DEBUG (1, "Array" << "<LEVEL=" << level << ">::recursiveCreate_ mmap=");
    
    if (level > 0) {
      int len = s.card();
      
      // the memory referenced by htav will be managed and released the newly created object
      typename TRAIT1::TypeTileImpl** htav = new typename TRAIT1::TypeTileImpl* [len];
      
      for (int j = 0; j < len; ++ j) {
	htav[j] = recursiveCreate_<typename TRAIT1::TraitSubref>(level-1, tiling, alloc); 
      }
      ret = new typename TRAIT1::TypeImpl(level, htav, s, alloc, NULL, -1);
      
    } else {  // leaf case
	ret = new typename TRAIT1::TypeImpl(level, NULL, s, alloc, NULL, -1);
    }
    return ret;
    }*/
 
  template <typename TRAIT1>
    static typename TRAIT1::TypeImpl* recursiveCreateShell_(size_t level, const typename Tuple<DIM>::Seq tiling, const Distribution<DIM>& dist, int home)
  {
    typename TRAIT1::TypeAllocator alloc;
    Shape<DIM> s(tiling[0]);
    int len = s.card();
    // the memory referenced by htav will be managed and released the newly created object, 
    // the tiles are not yet initialized ...
    typename TRAIT1::TypeTileImpl** htav = new typename TRAIT1::TypeTileImpl* [len];
    memset(htav, 0, len * sizeof(unsigned));
    // we do not yet specify the distribution .... only after all tiles are assigned.
    
    typename TRAIT1::TypeImpl* tmp  =  new typename TRAIT1::TypeImpl(level, htav, s, alloc, NULL, home);
    
    typename Shape<DIM>::iterator i = s.begin();
    typename Shape<DIM>::iterator end = s.end();
    
    int j = 0;
    if (tiling.length() > 1)
      for (; i != end; ++i, ++j) {
        int home_tile = dist[j];
        const Distribution<DIM>& dist_tile = ConstDistribution<DIM>::get(home_tile);
        tmp->initTile(*i, recursiveCreateShell_<typename TRAIT1::TraitSubref>(level - 1, tiling.trailing(1), dist_tile, home_tile));
      }
    
    return tmp;
  }

  template<class T1, int DIM1, typename TRAIT1>
  friend ostream& operator<< (ostream& ost, const Array<T1, DIM1,TRAIT1>& a);

  virtual void print_(ostream& ost, int indent = 0) const = 0;

public:

  inline bool isLeaf () const
  {
    //    cout << "here " << MPI::myThread() << endl;
    //cout << is_leaf_ << endl;
    //cout << "after " << MPI::myThread() << endl;
    return is_leaf_;
  }

  size_t level() const 
  {
    return level_;
  }

public:

  //GDBEBUG:  I made them public because isLeaf(), memMapping() etc.
  //are not getting inlined for some reason.

  bool is_leaf_;

  /** 
   * the distribution of this array that allows to clone / permute it 
   * this distribution can be null for arrays can be null if the HTA is 
   * not yet completely initialized (bottom up creation, shell HTA)
   */
  const MemMapping<DIM>* memMapping__;

protected:

  size_t level_;
  
  int    home_;

  /** shape of this array */
  Shape<DIM> shape_;

  /** allocator that manages the memory underlying this array / tile */
  const typename TRAIT::TypeAllocator alloc_;

  /* 
   * tile to processor mapping. This is only set for top-level HTAs
   */
  const Distribution<DIM>* dist__;

  const AbstractArray<T, DIM>* parent__;

public:

  int numTiles_;

  int* localTiles_; 

};//end Array class

template<class T, int DIM, typename TRAIT>
ostream& operator << (ostream& ost, const Array<T, DIM, TRAIT>& a) 
{
  a.print_(ost);
  return ost;
}

} // TILED_ARRAY_NAMESPACE

#endif /* __ARRAY_H__ */
