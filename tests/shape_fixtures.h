#ifndef TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED
#define TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED

#include "TiledArray/range.h"
#include "range_fixture.h"
#include "versioned_pmap_fixture.h"
#include "TiledArray/sparse_shape.h"
#include "TiledArray/dense_shape.h"
#include <world/worlddc.h>

class FakeArray {
public:
  typedef TiledArray::StaticRange<GlobalFixture::coordinate_system> range_type;
  typedef range_type::index index;
  typedef range_type::ordinal_index ordinal_index;
  typedef madness::WorldDCDefaultPmap<std::size_t> pmap_type;
  typedef madness::WorldDCPmapInterface<std::size_t> pmap_interface;

  FakeArray() :
    range_(index(0), index(5)), pmap_(new pmap_type(& GlobalFixture::world))
  { }

  template <typename Index>
  bool includes(const Index& i) const { return range_.includes(i); }



  const range_type& range() const { return range_; }

  const std::shared_ptr<pmap_interface>& get_pmap() const {
    return std::static_pointer_cast<pmap_interface>(pmap_);
  }

private:
  range_type range_;
  std::shared_ptr<pmap_type> pmap_;
};

struct BaseShapeFixture {
  typedef expressions::Tile<int, typename GlobalFixture::coordinate_system> TileN;
  typedef TileN::index index;
  typedef TileN::ordinal_index ordinal_index;
  typedef TiledArray::Range<GlobalFixture::coordinate_system> RangeN;
  typedef TiledArray::Shape<GlobalFixture::coordinate_system> ShapeT;
  typedef madness::WorldDCDefaultPmap<std::size_t> PmapT;

  static const RangeN r;
  static const PmapT m;
};

struct DenseShapeFixture : public virtual BaseShapeFixture {
  typedef TiledArray::DenseShape<GlobalFixture::coordinate_system> DenseShapeT;

  DenseShapeFixture() :
    ds(r, m)
  { }

  DenseShapeT ds;
};


struct SparseShapeFixture : public virtual BaseShapeFixture {
  typedef TiledArray::SparseShape<GlobalFixture::coordinate_system> SparseShapeT;

  SparseShapeFixture() :
      list(SparseShapeFixture::make_list(r.volume())),
      ss(* GlobalFixture::world, r, m, list.begin(), list.end())
  { }

  static std::vector<std::size_t> make_list(const std::size_t size) {
    std::vector<std::size_t> result;
    result.reserve((size / 3) + 1);
    for(std::size_t i = 0; i < size; ++i) {
      if(i % 3 == 0)
        result.push_back(i);
    }

    return result;
  }

  std::vector<std::size_t> list;
  SparseShapeT ss;
};

struct ShapeFixture : public DenseShapeFixture, public SparseShapeFixture {
  typedef ShapeT* ShapePtr;

  ShapeFixture() { }

};

#endif // TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED
