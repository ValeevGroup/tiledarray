#ifndef TILEDARRAY_RANGE_FIXTURE_H__INCLUDED
#define TILEDARRAY_RANGE_FIXTURE_H__INCLUDED

struct RangeFixture {
  typedef TiledArray::Range<std::size_t, 3> Range3;
  typedef TiledArray::Range<std::size_t, 3, TiledArray::LevelTag<1>,
      TiledArray::CoordinateSystem<3, TiledArray::detail::increasing_dimension_order> > FRange3;
  typedef Range3::size_array size_array;
  typedef Range3::index_type index_type;
  typedef Range3::volume_type volume_type;

  RangeFixture();

  ~RangeFixture() { }

  Range3 r;
  size_array size;
  size_array weight;
  index_type start;
  index_type finish;
  volume_type volume;
  const index_type p000;
  const index_type p111;
  const index_type p222;
  const index_type p333;
  const index_type p444;
  const index_type p555;
  const index_type p666;
};

#endif // TILEDARRAY_RANGE_FIXTURE_H__INCLUDED
