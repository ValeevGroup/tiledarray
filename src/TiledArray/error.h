#ifndef TILEDARRAY_ERROR_H__INCLUDED
#define TILEDARRAY_ERROR_H__INCLUDED

#include <TiledArray/config.h>

// Check for default error checking method, which is determined by TA_DEFAULT
// error. It is defined in TiledArray/config.h.
#ifdef TA_DEFAULT_ERROR
#if !defined(TA_EXCEPTION_ERROR) && !defined(TA_EXCEPTION_ERROR) && !defined(TA_EXCEPTION_ERROR)
#if TA_DEFAULT_ERROR == 0
#define TA_NO_ERROR
#elif TA_DEFAULT_ERROR == 1
#define TA_EXCEPTION_ERROR
#elif TA_DEFAULT_ERROR == 2
#define TA_ASSERT_ERROR
#endif // TA_DEFAULT_ERROR == ?
#endif // !defined(TA_EXCEPTION_ERROR) && !defined(TA_EXCEPTION_ERROR) && !defined(TA_EXCEPTION_ERROR)
#endif // TA_DEFAULT_ERROR

#ifdef TA_EXCEPTION_ERROR
// This section defines the behavior for TiledArray assertion error checking
// which will throw exceptions.
#ifdef TA_ASSERT_ERROR
#undef TA_ASSERT_ERROR
#warning "TA_EXCEPTION_ERROR supersedes TA_ASSERT_ERROR. TA_ASSERT_ERROR will be undefined."
#endif
#include <stdexcept>
namespace TiledArray {
  namespace detail {
     inline void exception_break() { }
  }
}


#define TA_STRINGIZE( s ) #s
#define TA_EXCEPTION_MESSAGE( file , line , func , type ,  mess ) \
  type TA_STRINGIZE( func ) ":" \
  TA_STRINGIZE( file ) "(" TA_STRINGIZE( line ) "): " mess
#define TA_ASSERT( a , e , m )  \
  if(! ( a ) ) \
    { \
      TiledArray::detail::exception_break(); \
      throw e ( TA_EXCEPTION_MESSAGE( __FILE__ , __LINE__ , __FUNC__ , "TiledArray Assertion failure in " , m ) ) ; \
    }

#define TA_EXCEPTION( e , t ,  m ) \
    throw e ( TA_EXCEPTION_MESSAGE( __FILE__ , __LINE__ , __FUNC__ , t , m ) )

#elif defined(TA_ASSERT_ERROR)
// This sections defines behavior for TiledArray assertion error checking which
// uses assertions.
#include <cassert>
#define TA_ASSERT( a , e , m ) assert( a )
#define TA_EXCEPTION( e , t , m ) exit(1)

#else
// This section defines behavior for TiledArray assertion error checking which
// does no error checking.
#warning "TiledArray will perform no error checking."
#define TA_ASSERT( a , e , m ) { ; }
#define TA_EXCEPTION( e , t , m ) exit(1)

#endif //TA_EXCEPTION_ERROR

#endif // TILEDARRAY_ERROR_H__INCLUDED
