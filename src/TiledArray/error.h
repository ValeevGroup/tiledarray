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
#include <exception>
namespace TiledArray {

  class Exception : public std::exception {
  public:
    Exception(const char* m) : message_(m) { }

    virtual const char* what() const throw() { return message_; }

  private:
    const char* message_;
  }; // class Exception

  /// Place a break point on this function to stop before TiledArray exceptions are thrown.
  inline void exception_break() { }
} // namespace TiledArray


#define TA_STRINGIZE( s ) #s

#define TA_EXCEPTION_MESSAGE( file , line , mess ) \
  "TiledArray: exception at " file "(" TA_STRINGIZE( line ) "): " mess

#define TA_EXCEPTION( m ) \
    throw TiledArray::Exception ( TA_EXCEPTION_MESSAGE( __FILE__ , __LINE__ , m ) )

#define TA_ASSERT( a )  \
  if(! ( a ) ) \
    { \
      TiledArray::exception_break(); \
      TA_EXCEPTION( "assertion failure" ) ; \
    }

#define TA_TEST( a )  TA_ASSERT( a )

#elif defined(TA_ASSERT_ERROR)
// This sections defines behavior for TiledArray assertion error checking which
// uses assertions.
#include <cassert>
#define TA_ASSERT( a ) assert( a )
#define TA_EXCEPTION( m ) exit(1)
#define TA_TEST( a )  TA_ASSERT( a )
#else
// This section defines behavior for TiledArray assertion error checking which
// does no error checking.
// WARNING: TiledArray will perform no error checking.
#define TA_ASSERT( a ) { ; }
#define TA_EXCEPTION( m ) exit(1)
#define TA_TEST( a )  a

#endif //TA_EXCEPTION_ERROR

#define TA_CHECK( a )  \
  if(! ( a ) ) \
    { \
      TA_EXCEPTION( "check failure" ) ; \
    }

#endif // TILEDARRAY_ERROR_H__INCLUDED
