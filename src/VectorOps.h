/*
 * Copyright (c) 2005 IBM Corporation
 *
 * This file is part of htalib, a library for hierarchically tiled arrays.
 */

/*
 * Version: $Id: VectorOps.h,v 1.8 2007/02/14 16:09:58 bikshand Exp $
 * Authors: Ganesh Bikshandi, Christoph von Praun, Basilio B Fraguela
 */

#ifndef  VECTOR_OPS_H__
#define  VECTOR_OPS_H__

#include "Tuple.h"

namespace TILED_ARRAY_NAMESPACE
{

template <typename T, int DIM>
class VectorOps
{

public:  
	static inline int
	dotProduct(const T &x, const T &y)
	{
		return x[DIM-1] * y[DIM-1] + VectorOps<T, DIM-1>::dotProduct(x, y);
	}

	static inline int
	selfProduct(const T &x)
	{
		return x[DIM-1] * VectorOps<T, DIM-1>::selfProduct(x);
	}  

	static inline int selfSum(const T &x) {
		return x[DIM-1] + VectorOps<T, DIM-1>::selfSum(x);
	}

#define DECL_VOP(NAME, OP)  \
	static inline void  \
	NAME(T &x, const T &y)  \
	{  \
		x[DIM-1] OP y[DIM-1];  \
		VectorOps<T, DIM-1>::NAME (x, y);  \
	}

	DECL_VOP(addIn, +=)
	DECL_VOP(subIn, -=)
	DECL_VOP(multIn, *=)
	DECL_VOP(divIn, /=)
	DECL_VOP(modIn, %=)
	DECL_VOP(copy, =)

#define DECL_VOPR(NAME, OP)  \
	static inline void  \
	NAME(T &r, const T &x, const T &y)  \
	{  \
		r[DIM-1] = x[DIM-1] OP y[DIM-1];  \
		VectorOps<T, DIM-1>::NAME (r, x, y);  \
	}

	DECL_VOPR(add, +)
	DECL_VOPR(sub, -)
	DECL_VOPR(mult, *)
	DECL_VOPR(div, /)
	DECL_VOPR(mod, %)

#define DECL_VUNOPR(NAME, OP)  \
	static inline void  \
	NAME(T &r, const T &x)  \
	{  \
		r[DIM-1] = OP(x[DIM-1]);  \
		VectorOps<T, DIM-1>::NAME (r, x);  \
	}

	DECL_VUNOPR(uminus, -)

#define DECL_VCOMPOPR(NAME, OP)  \
	static inline bool  \
	NAME(const T &x, const T &y) \
	{  \
		return (x[DIM-1] OP y[DIM-1]) && VectorOps<T, DIM-1>::NAME(x, y); \
	}

	DECL_VCOMPOPR(equal, ==)
	DECL_VCOMPOPR(nequal, !=)
	DECL_VCOMPOPR(less, <)
	DECL_VCOMPOPR(greater, >)
	DECL_VCOMPOPR(lesseq, <=)
	DECL_VCOMPOPR(greaereq, >=)

#define DECL_VCOMPPRIOPR(NAME, OP)  \
	static inline bool  \
	NAME(const T &x, const T &y)  \
	{  \
		return (VectorOps<T, DIM-1>::NAME(x, y) ? true : x[DIM-1] OP y[DIM-1]);  \
	}

	DECL_VCOMPPRIOPR(lessPriority, <)
	DECL_VCOMPPRIOPR(greaterPriority, >)
	DECL_VCOMPPRIOPR(lesseqPriority, <=)
	DECL_VCOMPPRIOPR(greatereqPriority, >=)

	static inline bool
	different(const T &x, const T &y)
	{
		return (x[DIM-1] != y[DIM-1]) || VectorOps<T, DIM-1>::different(x, y);
	}

	static inline bool
	increment(const T& max, T &x)
	{
		assert(max[DIM - 1] > x[DIM - 1]);
		assert(x[DIM - 1] >= 0);
		bool atEnd = false;
		if(++x[DIM - 1] >= max[DIM - 1])
		{
			x[DIM - 1] = 0;
			atEnd = VectorOps<T, DIM-1>::increment(max, x);
		}

		return ((max[DIM - 1] - 1) == x[DIM - 1]) && atEnd;
	}

};


template <typename T>
class VectorOps<T, 1>
{
  
public:  
	static inline int dotProduct(const T &x, const T &y)
	{
		return x[0] * y[0];
	}

	static inline int selfProduct(const T &x) 
	{
		return x[0];
	}

	static inline int selfSum(const T &x) 
	{
		return x[0];
	}

#define DECL_VOP1(NAME,OP)			      \
	static inline void NAME (T &x, const T &y) {        \
		x[0] OP y[0];                                     \
	}

	DECL_VOP1(addIn, +=)
	DECL_VOP1(subIn, -=)
	DECL_VOP1(multIn, *=)
	DECL_VOP1(divIn, /=)
	DECL_VOP1(modIn, %=)
	DECL_VOP1(copy, =)

#define DECL_VOPR1(NAME, OP)		                	\
	static inline void NAME (T &r, const T &x, const T &y) {	\
		r[0] = x[0] OP y[0];                                        \
	}

	DECL_VOPR1(add, +)
	DECL_VOPR1(sub, -)
	DECL_VOPR1(mult, *)
	DECL_VOPR1(div, /)
	DECL_VOPR1(mod, %)

#define DECL_VUNOPR1(NAME, OP)               	\
	static inline void NAME (T &r, const T &x) {	\
		r[0] = OP(x[0]);		        	\
	}

	DECL_VUNOPR1(uminus, -)

#define DECL_VCOMPOPR1(NAME, OP)  \
	static inline bool  \
	NAME(const T &x, const T &y) \
	{  \
	return (x[0] == y[0]); \
	}

	DECL_VCOMPOPR1(equal, ==)
	DECL_VCOMPOPR1(nequal, !=)
	DECL_VCOMPOPR1(less, <)
	DECL_VCOMPOPR1(greater, >)
	DECL_VCOMPOPR1(lesseq, <=)
	DECL_VCOMPOPR1(greaereq, >=)
	
#define DECL_VCOMPPRIOPR1(NAME, OP)  \
	static inline bool  \
	NAME(const T &x, const T &y)  \
	{  \
	return (x[0] OP y[0]);  \
	}

	DECL_VCOMPPRIOPR1(lessPriority, <)
	DECL_VCOMPPRIOPR1(greaterPriority, >)
	DECL_VCOMPPRIOPR1(lesseqPriority, <=)
	DECL_VCOMPPRIOPR1(greatereqPriority, >=)


	static inline bool different(const T &x, const T &y) {
		return (x[0] != y[0]);
	}

	static inline bool
	increment(const T& max, T &x)
	{
		assert(max[0] > x[0]);
		assert(x[0] >= 0);
		if(++x[0] >= max[0])
			x[0] = 0;

		return ((max[0] - 1) == x[0]);
	}

};

}

#endif /* VECTOR_OPS_H_ */
