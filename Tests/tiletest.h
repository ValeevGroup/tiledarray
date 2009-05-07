#ifndef TILETEST_H__INCLUDED
#define TILETEST_H__INCLUDED

#ifdef TEST_TILE
#define RUN_TILE_TEST	{ TileTest(); }
#else
#define RUN_TILE_TEST	{ ; }
#endif

void TileTest();

#endif // TILETEST_H__INCLUDED
