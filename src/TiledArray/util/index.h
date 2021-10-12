#ifndef TILEDARRAY_UTIL_INDEX_H__INCLUDED
#define TILEDARRAY_UTIL_INDEX_H__INCLUDED

// Samuel R. Powell, 2021
#include "TiledArray/expressions/fwd.h"

#include <TiledArray/error.h>
#include <TiledArray/permutation.h>
#include <TiledArray/util/vector.h>

#include <iosfwd>
#include <string>

namespace TiledArray::index {

template <typename T>
using small_vector = container::svector<T>;

small_vector<std::string> tokenize(const std::string &s);

small_vector<std::string> validate(const small_vector<std::string> &v);

std::string join(const small_vector<std::string> &v);

template <typename T, typename U>
using enable_if_string = std::enable_if_t<std::is_same_v<T, std::string>, U>;

/// an n-index, with n a runtime parameter
template <typename T>
class Index {
 public:
  using container_type = small_vector<T>;
  using value_type = typename container_type::value_type;

  Index() = default;
  Index(container_type &&s) : data_(std::move(s)) {}

  template <typename S, typename U = void>
  Index(const S &s) : data_(s.begin(), s.end()) {}

  template <typename U = void>
  Index(const std::string &s) : Index(index::tokenize(s)) {}

  template <typename U = void>
  Index(const char *s) : Index(std::string(s)) {}

  template <typename U = void>
  operator std::string() const {
    return index::join(data_);
  }

  explicit operator bool() const { return !data_.empty(); }

  bool operator==(const Index &other) const {
    return (this->data_ == other.data_);
  }

  bool operator!=(const Index &other) const { return !(*this == other); }

  size_t size() const { return data_.size(); }

  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }

  auto find(const T &v) const {
    return std::find(this->begin(), this->end(), v);
  }

  const auto &operator[](size_t idx) const { return data_.at(idx); }

  size_t indexof(const T &v) const {
    for (size_t i = 0; i < this->size(); ++i) {
      if (this[i] == v) return i;
    }
    return -1;
  }

  /// Returns true if argument exists in the Index object, else returns false
  bool contains(const T &v) const { return (this->find(v) != this->end()); }

 private:
  container_type data_;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Index<T> &idx) {
  os << std::string(idx);
  return os;
}

/// (stable) intersect of 2 Index objects
/// @param[in] a an Index object
/// @param[in] b an Index object
/// @pre a and b do not have duplicates
template <typename T>
Index<T> operator&(const Index<T> &a, const Index<T> &b) {
  typename Index<T>::container_type r;
  for (const auto &s : a) {
    if (!b.contains(s)) continue;
    r.push_back(s);
  }
  return Index<T>(r);
}
/// union of 2 Index objects
/// @param[in] a an Index object
/// @param[in] b an Index object
/// @pre a and b do not have duplicates
template <typename T>
Index<T> operator|(const Index<T> &a, const Index<T> &b) {
  typename Index<T>::container_type r;
  r.assign(a.begin(), a.end());
  for (const auto &s : b) {
    if (a.contains(s)) continue;
    r.push_back(s);
  }
  return Index<T>(r);
}

/// concatenation of 2 Index objects
/// @param[in] a an Index object
/// @param[in] b an Index object
/// @note unline operator| @p a and @p b can have have duplicates
template <typename T>
Index<T> operator+(const Index<T> &a, const Index<T> &b) {
  typename Index<T>::container_type r;
  r.assign(a.begin(), a.end());
  r.insert(r.end(), b.begin(), b.end());
  return Index<T>(r);
}

/// "difference" of  2 Index objects, i.e. elements of a that are not in b
/// @param[in] a an Index object
/// @param[in] b an Index object
/// @note unline operator& @p a and @p b can have have duplicates
template <typename T>
Index<T> operator-(const Index<T> &a, const Index<T> &b) {
  typename Index<T>::container_type r;
  for (const auto &s : a) {
    if (b.contains(s)) continue;
    r.push_back(s);
  }
  return Index<T>(r);
}

/// elements that are exclusively in @p a or @p b
/// @param[in] a an Index object
/// @param[in] b an Index object
/// @pre a and b do not have duplicates
template <typename T>
inline Index<T> operator^(const Index<T> &a, const Index<T> &b) {
  return (a | b) - (a & b);
}

template <typename T>
size_t rank(const Index<T> &idx) {
  return idx.size();
}

template <typename T>
Index<T> sorted(const Index<T> &a) {
  typename Index<T>::container_type r(a.begin(), a.end());
  std::sort(r.begin(), r.end());
  return Index<T>(r);
}

template <typename T>
Permutation permutation(const Index<T> &s, const Index<T> &p) {
  assert(sorted(s) == sorted(p));
  small_vector<size_t> m;
  m.reserve(p.size());
  for (size_t i = 0; i != p.size(); ++i) {
    m.push_back(s.indexof(p[i]));
  }
  return Permutation(m);
}

template <typename T, bool Inverse>
auto permute(const Permutation &p, const Index<T> &s,
             std::bool_constant<Inverse>) {
  if (!p) return s;
  using R = typename Index<T>::container_type;
  R r(p.size());
  detail::permute_n(p.size(), p.begin(), s.begin(), r.begin(),
                    std::bool_constant<Inverse>{});
  return Index<T>{r};
}

/// @brief Index-annotated collection of objects
/// @tparam Value
/// This is a map using Index::element_type as key
template <typename K, typename V>
struct IndexMap {
  using key_type = K;
  using value_type = V;

  IndexMap(const Index<K> &keys, std::initializer_list<V> s)
      : IndexMap(keys, s.begin(), s.end()) {}

  template <typename S>
  IndexMap(const Index<K> &keys, S &&s) : IndexMap(keys, s.begin(), s.end()) {}

  template <typename It>
  IndexMap(const Index<K> &keys, It begin, It end) {
    auto it = begin;
    data_.reserve(keys.size());
    for (auto &&key : keys) {
      assert(it != end);
      data_.emplace_back(std::pair<K, V>{key, *it});
      ++it;
    }
    assert(it == end);
  }

  IndexMap(const small_vector<std::pair<K, V> > &data) : data_(data) {}

  /// @return const iterator pointing to the element associated with @p key
  auto find(const key_type &key) const {
    return std::find_if(data_.begin(), data_.end(),
                        [&key](const auto &v) { return key == v.first; });
  }

  /// @return reference to the element associated with @p key
  /// @throw TA::Exception if @p key is not in this map
  const auto &operator[](const key_type &key) const {
    auto it = find(key);
    if (it != data_.end()) return it->second;
    throw TiledArray::Exception("IndexMap::at(key): key not found");
  }

  /// @param[in] idx an Index object
  /// @return directly-addressable sequence of elements corresponding to the
  /// keys in @p idx
  auto operator[](const Index<K> &idx) const {
    small_vector<value_type> result;
    result.reserve(idx.size());
    for (auto &&key : idx) {
      result.emplace_back(this->operator[](key));
    }
    return result;
  }

  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }

 private:
  small_vector<std::pair<key_type, value_type> > data_;
};

template <typename K, typename V>
bool operator==(const IndexMap<K, V> &lhs, const IndexMap<K, V> &rhs) {
  for (const auto &[k, v] : lhs) {
    if (rhs.find(k) == rhs.end() || v != rhs[k]) return false;
  }
  for (const auto &[k, v] : rhs) {
    if (lhs.find(k) == lhs.end()) return false;
  }
  return true;
}

/// TODO to be filled by Sam
template <typename K, typename V>
IndexMap<K, V> operator|(const IndexMap<K, V> &a, const IndexMap<K, V> &b) {
  small_vector<std::pair<K, V> > d(a.begin(), a.end());
  for (const auto [k, v] : b) {
    if (a.find(k) != a.end()) {
      TA_ASSERT(a[k] == b[k]);
      continue;
    }
    d.push_back(std::pair(k, v));
  }
  return IndexMap(d);
}

}  // namespace TiledArray::index

namespace TiledArray {

using Index = TiledArray::index::Index<std::string>;
using TiledArray::index::IndexMap;

/// converts the annotation of an expression to an Index
template <typename Array>
auto idx(const std::string &s) {
  if constexpr (detail::is_tensor_of_tensor_v<typename Array::value_type>) {
    auto semi = std::find(s.begin(), s.end(), ';');
    assert(semi != s.end());
    auto first = std::string(s.begin(), semi);
    auto second = std::string(semi + 1, s.end());
    return std::tuple<Index, Index>{first, second};
  } else {
    return std::tuple<Index>{s};
  }
}

/// converts the annotation of an expression to an Index
template <typename A, bool Alias>
auto idx(const expressions::TsrExpr<A, Alias> &e) {
  return idx<A>(e.annotation());
}

}  // namespace TiledArray
#endif
