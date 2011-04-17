#ifndef TILEDARRAY_TEST_MATH_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_MATH_FIXTURE_H__INCLUDED

#include <cstddef>
#include "unit_test_config.h"
#include "TiledArray/range.h"
#include "TiledArray/tile.h"
#include "TiledArray/annotated_array.h"

// Emulate an N-dimensional array class
template <typename T, typename CS>
class FakeArray {
public:
  typedef CS                                                coordinate_system;
  typedef typename CS::index                                index;
  typedef typename CS::ordinal_index                        ordinal_index;
  typedef typename CS::volume_type                          volume_type;
  typedef typename CS::size_array                           size_array;
  typedef T                                                 value_type;
  typedef value_type&                                       reference;
  typedef const value_type&                                 const_reference;
  typedef typename std::vector<value_type>::iterator        iterator;
  typedef typename std::vector<value_type>::const_iterator  const_iterator;
  typedef TiledArray::Range<CS>                             range_type;

  FakeArray() :
      r_(), d_()
  { }

  FakeArray(const range_type& r, value_type v = value_type()) :
      r_(r), d_(r.volume(), v)
  { }

  template <typename InIter>
  FakeArray(const range_type& r, InIter first, InIter last) :
      r_(r), d_(first, last)
  { assert(typename range_type::volume_type(std::distance(first, last)) == r.volume()); }

  FakeArray(const FakeArray& other) :
      r_(other.r_), d_(other.d_)
  { }

  FakeArray& operator=(const FakeArray& other) {
    r_ = other.r_;
    std::copy(other.d_.begin(), other.d_.end(), d_.begin());
    return *this;
  }

  iterator begin() { return d_.begin(); }
  const_iterator begin() const { return d_.begin(); }
  iterator end() { return d_.end(); }
  const_iterator end() const { return d_.end(); }

  reference at(ordinal_index i) { return d_.at(i); }
  const_reference at(ordinal_index i) const { return d_.at(i); }
  reference operator[](ordinal_index i) { return d_[i]; }
  const_reference operator[](ordinal_index i) const { return d_[i]; }

  reference at(const index& i) { return d_.at(ord(i)); }
  const_reference at(const index& i) const { return d_.at(ord(i)); }
  reference operator[](const index& i) { return d_[ord(i)]; }
  const_reference operator[](const index& i) const { return d_[ord(i)]; }

  const range_type& range() const { return r_; }

  FakeArray& resize(const range_type& r) {
    r_ = r;
    d_.resize(r.volume());
  }

private:

  ordinal_index ord(const index& i) const {
    return coordinate_system::calc_ordinal(i, r_.weight());
  }

  range_type r_;
  std::vector<value_type> d_;
};

struct MathFixture {
  typedef TiledArray::Tile<int, GlobalFixture::coordinate_system> array_type;
  typedef TiledArray::CoordinateSystem<GlobalFixture::coordinate_system::dim + 1,
        GlobalFixture::coordinate_system::level,
        GlobalFixture::coordinate_system::order,
        GlobalFixture::coordinate_system::ordinal_index> coordinate_system1;
  typedef FakeArray<int, coordinate_system1> array1_type;
  typedef array_type::range_type range_type;
  typedef TiledArray::expressions::AnnotatedArray<array_type > array_annotation;
  typedef array_annotation::index index;
  typedef array_annotation::ordinal_index ordinal_index;

  MathFixture()
  { }

  static std::string make_var_list(std::size_t first = 0,
      std::size_t last = GlobalFixture::element_coordinate_system::dim);

  static const TiledArray::expressions::VariableList vars;
  static const range_type r;
  static const array_type f1;
  static const array_type f2;
  static const array_type f3;

  static const array_annotation a1;
  static const array_annotation a2;
  static const array_annotation a3;
};

#endif // TILEDARRAY_TEST_MATH_FIXTURE_H__INCLUDED
