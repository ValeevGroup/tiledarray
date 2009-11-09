#ifndef ERROR_H__INCLUDED
#define ERROR_H__INCLUDED

#ifdef TA_EXCEPTION_ERROR
#ifdef TA_ASSERT_ERROR
#undef TA_ASSERT_ERROR
#warning "TA_EXCPETION_ERROR supersedes TA_ASSERT_ERROR"
#endif
#include <stdexcept>
#define TA_STRINGIZE( s ) #s
#define TA_EXCEPTION_MESSAGE( file , line , func , mess ) \
  "TiledArray Assertion failure in " TA_STRINGIZE( func ) ":" \
  TA_STRINGIZE( file ) "(" TA_STRINGIZE( line ) "): " mess
#define TA_ASSERT( a , e , m )  \
  if(! ( a ) ) \
    { throw e ( TA_EXCEPTION_MESSAGE( __FILE__ , __LINE__ , __FUNCTION__ , m ) ) ; }
#endif

#ifdef TA_ASSERT_ERROR
#include <cassert>
#define TA_ASSERT( a , e , m ) assert( a )
#endif

#ifndef TA_ASSERT
#define TA_ASSERT( a , e , m ) { ; }
#endif


#endif // ERROR_H__INCLUDED
