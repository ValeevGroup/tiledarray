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
 * Version: $Id: Tracing.h,v 1.3 2006/05/16 14:20:02 vonpraun Exp $
 * Authors: Ganesh Bikshandi, Christoph von Praun
 */


#ifndef TRACING_H__INCLUDED
#define TRACING_H__INCLUDED

#include "process.h"

#ifndef TA_DLEVEL
#define TA_DLEVEL -1
#endif

#ifndef TA_WLEVEL
#define TA_WLEVEL -1
#endif

#ifdef WIN32
#define __attribute__(...)
#endif

static int g_dlevel = TA_DLEVEL;
static int g_wlevel = TA_WLEVEL;

static void
dlevel(int l) __attribute__((unused));
static void
wlevel(int l) __attribute__((unused));

static void
dlevel(int l)
{
	g_dlevel = l;
}

static void
wlevel(int l)
{
	g_wlevel = l;
}


/* debugging */
#if (TA_DLEVEL >= 0)
#define TA_DEBUG(L,X)    { if (L <= g_dlevel) { ::std::cout << "DEBG[" << L << ", " << Process::myPlace() << "]  " << X << ::std::endl << ::std::flush; } }
#else
#define TA_DEBUG(L,X)    { ; }
#endif /* HTA_DLEVEL */


/* warnings */
#if (TA_WLEVEL >= 0)
#define TA_WARN(L,X)    { if (L <=  g_wlevel) { ::std::cout << "WARN[" << L << ", " << Process::myPlace() << "]  " << X << ::std::endl << ::std::flush; } }
#else
#define TA_WARN(L,X)    { ; }
#endif /* HTA_WLEVEL */


/* deprecation */
#ifdef TA_REPORT_DEPRECATION
#define HTA_DEPRECATED(X)    { ::std::cout << "DEPRECATED: " << X << ::std::endl << ::std::flush; }
#else
#define HTA_DEPRECATED(X)    { ; }
#endif /* HTA_REPORT_DEPRECATION */

#endif // TRACING_H__INCLUDED
