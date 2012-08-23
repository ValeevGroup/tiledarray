#ifndef TILEDARRAY_EPIK_H__INCLUDED
#define TILEDARRAY_EPIK_H__INCLUDED

#ifdef EPIK
#include <epik_user.h>
#else
#define EPIK_TRACER(n)
#define EPIK_USER_REG(n,str)
#define EPIK_USER_START(n)
#define EPIK_USER_END(n)
#define EPIK_FUNC_START()
#define EPIK_FUNC_END()
#define EPIK_PAUSE_START()
#define EPIK_PAUSE_END()
#define EPIK_FLUSH_TRACE()
#endif // EPIK


#endif // TILEDARRAY_EPIK_H__INCLUDED
