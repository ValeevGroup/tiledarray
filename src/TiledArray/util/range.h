#include <TiledArray/util/vector.h>

#include <vector>
#include <boost/iterator/counting_iterator.hpp>

namespace TiledArray::range {

template<typename T>
using small_vector = container::svector<T>;

struct Range {
  using value_type = int64_t;
  using iterator = boost::counting_iterator<value_type>;
  template<class Pair>
  explicit Range(Pair &&pair) : Range(pair.first, pair.second) {}
  Range(value_type begin, value_type end) : begin_(begin), end_(end) {}
  auto begin() const { return iterator(begin_); }
  auto end() const { return iterator(end_); }
  auto size() const { return end_ - begin_; }
protected:
  const value_type begin_, end_;

};

template<typename R, typename T = small_vector<typename R::value_type> >
struct RangeProduct {

  using ranges_type = std::vector<R>;

public:

  RangeProduct() = default;
  RangeProduct(std::initializer_list<R> ranges) : ranges_(ranges) {}

  RangeProduct& operator *= (R a) {
    this->ranges_.push_back(a);
    return *this;
  }

  const auto& ranges() const { return ranges_; }

  struct iterator {
    using iterator1 = decltype(std::begin(ranges_type{}[0]));
    auto operator*() const {
      T r;
      for (auto &it : its_) { r.push_back(*it); }
      return r;
    }
    bool operator!=(const iterator &other) const {
      return !(this->p_ == other.p_ && this->its_ == other.its_);
    }
    iterator& operator++() {
      size_t i = its_.size();
      auto &ranges = p_->ranges();
      while (i > 0) {
        --i;
        ++its_[i];
        if (i == 0) break;
        if (its_[i] != std::end(ranges[i])) break;
        its_[i] = std::begin(ranges[i]);
      }
      return *this;
    }
  private:
    friend class RangeProduct;
    explicit iterator(const RangeProduct *p, bool End = false) {
      this->p_ = p;
      using std::begin;
      using std::end;
      for (const auto& r : p->ranges()) {
        auto it = (End ? end(r) : begin(r));
        its_.push_back(it);
        End = false;
      }
    }
  private:
    const RangeProduct *p_;
    small_vector<iterator1> its_;
  };

  auto begin() const {
    return iterator(this);
  }

  auto end() const {
    return iterator(this, true);
  }

protected:
  ranges_type ranges_;

};

inline RangeProduct<Range> operator*(Range a, Range b){
  return RangeProduct<Range>({a, b});
};

template<typename R, typename T>
RangeProduct<R,T> operator*(const RangeProduct<R,T>& a, Range b) {
  return RangeProduct<R,T>(a) *= b;
};

template<typename R, typename F>
void cartesian_foreach(const std::vector<R>& rs, F f) {
  using It = decltype(std::begin(rs[0]));
  using T = typename R::value_type;
  small_vector<It> its, ends;
  for (const auto& r : rs) {
    its.push_back(std::begin(r));
    ends.push_back(std::end(r));
  }
  while (its.front() != ends.front()) {
    small_vector<T> s;
    s.reserve(its.size());
    for (auto& it : its) {
      s.push_back(*it);
    }
    f(s);
    size_t i = its.size();
    while (i > 0) {
      --i;
      ++its[i];
      if (i == 0) break;
      if (its[i] != ends[i]) break;
      its[i] = std::begin(rs[i]);
    }
  }
}

}  // namespace TiledArray::expressions
