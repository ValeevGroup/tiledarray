#ifndef TILEMAPTEST_H__INCLUDED
#define TILEMAPTEST_H__INCLUDED

#ifdef TEST_TILEMAP
#define RUN_TILEMAP_TEST	{ TileMapTest(); }
#else
#define RUN_TILEMAP_TEST	{ ; }
#endif

void TileMapTest();

#endif // TILEMAPTEST_H__INCLUDED
