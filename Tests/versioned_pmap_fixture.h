#ifndef TILEDARRAY_VERSIONED_PMAP_FIXTURE_H__INCLUDED
#define TILEDARRAY_VERSIONED_PMAP_FIXTURE_H__INCLUDED

#include "global_fixture.h"
#include "TiledArray/versioned_pmap.h"

struct VersionedPmapFixture {
  typedef TiledArray::detail::VersionedPmap<GlobalFixture::coordinate_system::key_type> pmap_type;

  VersionedPmapFixture() : m(GlobalFixture::world->size()) { }

  pmap_type m;
};

#endif // TILEDARRAY_VERSIONED_PMAP_FIXTURE_H__INCLUDED
