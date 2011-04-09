#ifndef TILEDARRAY_ERROR_H__INCLUDED
#define TILEDARRAY_ERROR_H__INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAVE_STATIC_ASSERT

#define TA_STATIC_ASSERT( a ) static_assert( a )

#else

#include <boost/static_assert.hpp>
#define TA_STATIC_ASSERT( a ) BOOST_STATIC_ASSERT( a )
#endif // TILEDARRAY_HAVE_STATIC_ASSERT

// Check for default error checking method, which is determined by TA_DEFAULT
// error. It is defined in TiledArray/config.h.
#ifdef TA_DEFAULT_ERROR
# if !defined(TA_EXCEPTION_ERROR) && !defined(TA_ASSERT_ERROR) && !defined(TA_NO_ERROR)
#  if TA_DEFAULT_ERROR == 0
#   define TA_NO_ERROR
#  elif TA_DEFAULT_ERROR == 1
#   define TA_EXCEPTION_ERROR
#  elif TA_DEFAULT_ERROR == 2
#   define TA_ASSERT_ERROR
#  endif // TA_DEFAULT_ERROR == ?
# endif // !defined(TA_EXCEPTION_ERROR) && !defined(TA_EXCEPTION_ERROR) && !defined(TA_EXCEPTION_ERROR)
#endif // TA_DEFAULT_ERROR

#ifdef TA_EXCEPTION_ERROR
// This section defines the behavior for TiledArray assertion error checking
// which will throw exceptions.
#ifdef TA_ASSERT_ERROR
#undef TA_ASSERT_ERROR
// WARNING: TA_EXCEPTION_ERROR supersedes TA_ASSERT_ERROR.
#endif
#include <stdexcept>
namespace TiledArray {
  /// Place a break point on this function to stop before TiledArray exceptions are thrown.
  inline void exception_break() { }
}


#define TA_STRINGIZE( s ) #s

#define TA_EXCEPTION_MESSAGE( file , line , mess ) \
  "TiledArray: exception at " TA_STRINGIZE( file ) "(" TA_STRINGIZE( line ) "): " mess

#define TA_EXCEPTION( e , m ) \
    throw e ( TA_EXCEPTION_MESSAGE( __FILE__ , __LINE__ , m ) )

#define TA_ASSERT( a , e , m )  \
  if(! ( a ) ) \
    { \
      TiledArray::exception_break(); \
      TA_EXCEPTION( e , m ) ; \
    }

#elif defined(TA_ASSERT_ERROR)
// This sections defines behavior for TiledArray assertion error checking which
// uses assertions.
#include <cassert>
#define TA_ASSERT( a , e , m ) assert( a )
#define TA_EXCEPTION( e , m ) exit(1)
#else
// This section defines behavior for TiledArray assertion error checking which
// does no error checking.
// WARNING: TiledArray will perform no error checking.
#define TA_ASSERT( a , e , m ) { ; }
#define TA_EXCEPTION( e , m ) exit(1)

#endif //TA_EXCEPTION_ERROR

#define TA_CHECK( a , e , m )  \
  if(! ( a ) ) \
    { \
      TA_EXCEPTION( e , m ) ; \
    }

#endif // TILEDARRAY_ERROR_H__INCLUDED
