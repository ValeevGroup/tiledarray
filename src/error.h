#ifndef ERROR_H__INCLUDED
#define ERROR_H__INCLUDED

#ifdef TA_EXCEPTION_ERROR
#ifdef TA_ASSERT_ERROR
#undef TA_ASSERT_ERROR
#warning "TA_EXCEPTION_ERROR supersedes TA_ASSERT_ERROR"
#endif
#include <stdexcept>
#define TA_STRINGIZE( s ) #s
#define TA_EXCEPTION_MESSAGE( file , line , func , type ,  mess ) \
  type TA_STRINGIZE( func ) ":" \
  TA_STRINGIZE( file ) "(" TA_STRINGIZE( line ) "): " mess
#define TA_ASSERT( a , e , m )  \
  if(! ( a ) ) \
    { throw e ( TA_EXCEPTION_MESSAGE( __FILE__ , __LINE__ , __FUNCTION__ , "TiledArray Assertion failure in " , m ) ) ; }

#define TA_EXCEPTION( e , t ,  m ) \
    throw e ( TA_EXCEPTION_MESSAGE( __FILE__ , __LINE__ , __FUNCTION__ , t , m ) )

#else

#ifdef TA_ASSERT_ERROR
#include <cassert>
#define TA_ASSERT( a , e , m ) assert( a )
#define TA_EXCEPTION( e , m ) exit()

#else
#define TA_ASSERT( a , e , m ) { ; }
#define TA_EXCEPTION( e , m ) { ; }
#endif // TA_ASSERT_ERROR

#endif //TA_EXCEPTION_ERROR

#endif // ERROR_H__INCLUDED
