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
* Version: $Id: MathKernel.h,v 1.7 2006/05/25 15:26:06 vonpraun Exp $
* Authors: Christoph von Praun
*/

#ifndef  __MATH_KERNEL_H__
#define  __MATH_KERNEL_H__
#include <cassert>
/** 
* Provides efficient implementations of 
* math-kernels. This is an abstract shell -- 
* the implementations corresponding to a specific 
* numeric library are defined in subclasses.
*/

namespace TILED_ARRAY_NAMESPACE
{

template<typename T>
class Math
{

public:
	Math() {}

	Math(int size, const int* stride) { }

	virtual void dftForward (T* arr) 
	{
		assert (false);
	}

	virtual void dftBackward (T* arr) 
	{
		assert (false);
	} 

	static void dcscmm (char *transa, 
		int *m, int *n, int *k, 
		T *alpha, 
		char *matdescra, T *val, int *indx,  int *pntrb, int *pntre, 
		T *b, int *ldb, 
		T *beta, 
		T *c, int *ldc) 
	{
		assert (false);
	}

	static void dcsrmm (char *transa, 
		int *m, int *n, int *k, 
		T *alpha, 
		char *matdescra, T *val, int *indx,  int *pntrb, int *pntre, 
		T *b, int *ldb, 
		T *beta, 
		T *c, int *ldc) 
	{
		assert (false);
	}

};

}

#endif /* __MATH KERNEL_H__ */
