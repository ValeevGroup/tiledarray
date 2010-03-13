#ifndef TILEDARRAY_CONFIGURE_H__INCULDED
#define TILEDARRAY_CONFIGURE_H__INCULDED

/* Defines the default error checking behavior. none = 0, throw = 1, assert = 2 */
#define TA_DEFAULT_ERROR 1

/* TiledArray will use blas for some math operations. */
/* #undef TA_USE_CBLAS */
/* Maximum number of allowed dimensions. */
#define TA_MAX_DIM 10

#endif // TILEDARRAY_CONFIGURE_H__INCULDED
