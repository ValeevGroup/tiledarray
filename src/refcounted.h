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
 * Version: $Id: RefCounted.h,v 1.4 2006/07/08 20:56:45 vonpraun Exp $
 * Authors: Ganesh Bikshandi, Christoph von Praun
 */

#ifndef REF_COUNTED_H__
#define REF_COUNTED_H__


namespace TILED_ARRAY_NAMESPACE
{

class RefCounted 
{
	bool m_disabled;
	int m_refCount;
	RefCounted *m_ref;

public:

  /**
   * reference count mechanism that allow 'client' of this object, 
   * (other distributions, rep. HTA instances) to decide when to 
   * de-allocate
   *
   * @param  the delta of the refCount 
   * @return value of refCount, after val is added
   *         or a positive value if the ref-counting is disabled.
   */
	inline int
	RefCount(int val = 0) const
	{
		int ret;
		if(!m_disabled)
		{
			RefCounted* this_non_const = const_cast<RefCounted*>(this);
			if (val != 0) 
			{
				this_non_const->m_refCount += val;
				ret = this_non_const->m_refCount;
				assert (ret >= 0);
			}
		}
		else
		{
			ret = 1;
		}

		return ret;
	}

  /* disable ref-countinge.g. for instances that are 
   * allocated on the stack of statically */
	inline void
	DisableRefCounting() 
	{
		m_disabled = true;
	}

protected:

	RefCounted () :
		 m_disabled(false), m_refCount(0)
	{}
  

};

}

#endif // REF_COUNTED_H__
