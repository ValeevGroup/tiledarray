#ifndef TILEDARRAY_EINSUM_STRING_H
#define TILEDARRAY_EINSUM_STRING_H

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/join.hpp>
#include <string>
#include <vector>

namespace TiledArray::Einsum::string {
namespace {

  // Split delimiter must match completely
  template<typename T = std::string, typename U = T>
  std::pair<T,U> split2(const std::string& s, const std::string &d) {
    auto pos = s.find(d);
    if (pos == s.npos) return { T(s), U("") };
    return { T(s.substr(0,pos)), U(s.substr(pos+d.size())) };
  }

  // Split delimiter must match completely
  std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> res;
    return boost::split(res, s, [&d](char c) { return c == d; } /*boost::is_any_of(d)*/);
  }

  std::string trim(const std::string& s) {
    return boost::trim_copy(s);
  }

  template <typename T>
  std::string str(const T& obj) {
    std::stringstream ss;
    ss << obj;
    return ss.str();
  }

  template<typename T, typename U = std::string>
  std::string join(const T &s, const U& j = U("")) {
    std::vector<std::string> strings;
    for (auto e : s) {
      strings.push_back(str(e));
    }
    return boost::join(strings, j);
  }

}
}

#endif //TILEDARRAY_EINSUM_STRING_H
