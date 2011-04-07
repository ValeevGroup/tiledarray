#ifndef TILEDARRAY_ARRAY_FIXTURES_H__INCLUDED
#define TILEDARRAY_ARRAY_FIXTURES_H__INCLUDED

#include "range_fixture.h"
#include "TiledArray/variable_list.h"
#include "TiledArray/annotated_array.h"

using namespace TiledArray;
using TiledArray::expressions::AnnotatedArray;
using TiledArray::expressions::VariableList;

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
  typedef Range<CS>                                         range_type;

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

  reference at(std::size_t n) { return d_.at(n); }
  const_reference at(std::size_t n) const { return d_.at(n); }
  reference operator[](std::size_t n) { return d_[n]; }
  const_reference operator[](std::size_t n) const { return d_[n]; }

  const range_type& range() const { return r_; }

  FakeArray& resize(const range_type& r) {
    r_ = r;
    d_.resize(r.volume());
  }

private:
  range_type r_;
  std::vector<value_type> d_;
};

struct AnnotatedArrayFixture {
  typedef FakeArray<int, GlobalFixture::coordinate_system> array_type;
  typedef array_type::range_type range_type;
  typedef TiledArray::expressions::AnnotatedArray<array_type > fake_annotation;
  typedef fake_annotation::index index;

  static const VariableList vars;
  static const range_type r;
  static const array_type t;

  AnnotatedArrayFixture();

  static std::string make_var_list();

  fake_annotation at;
}; // struct AnnotatedArrayFixture

#endif // TILEDARRAY_ARRAY_FIXTURES_H__INCLUDED
