#ifndef ERROR_H__INCLUDED
#define ERROR_H__INCLUDED

#ifdef TA_EXCEPTION_ERROR
#ifdef TA_ASSERT_ERROR
#undef TA_ASSERT_ERROR
#warning "TA_EXCPETION_ERROR supersedes TA_ASSERT_ERROR"
#endif
#include <stdexcept>
#define TA_ASSERT( a , e )  if(! ( a ) ) { throw e ; }
#endif

#ifdef TA_ASSERT_ERROR
#include <cassert>
#define TA_ASSERT( a , e ) assert( a )
#endif

#ifndef TA_ASSERT
#define TA_ASSERT( a , e ) { ; }
#endif


#endif // ERROR_H__INCLUDED
